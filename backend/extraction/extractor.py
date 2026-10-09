"""
The extraction pipeline, end to end.

    photo
      -> surfaces          wall/floor/ceiling, for structural rejection
      -> Grounding DINO    one pass, whatever the prompt list holds
      -> filtering         size, structure, duplicates  (before SAM, so every
                           rejection is a SAM call not made)
      -> SAM 2             one encode, boxes decoded in batches
      -> selection         best candidate per box, or reject it
      -> cleanup           deterministic OpenCV tidy-up
      -> overlap           reflections suppressed, contested pixels settled

Inference runs at a bounded working resolution; every mask is then mapped back
to the photo's exact original dimensions, and every RGB pixel that reaches a PNG
is read from the original photo. The working copy exists so that a 12-megapixel
phone upload does not cost a gigabyte of intermediate masks — it never reaches
the output.

Three inference calls for a whole room: one SegFormer, one Grounding DINO, one
SAM 2 encode (plus a cheap decode per batch of boxes). Nothing is re-run per
object.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from extraction import cleanup, dino, filtering, mirrors, openings, overlap, props, sam2, selection
from extraction.config import ExtractionConfig
import matting
import surfaces


# Guided-filter window for edge/boundary refinement, as a fraction of the
# photo's longest edge, clamped in `to_original`. Wide enough to move a
# boundary that interpolation left a few pixels off, narrow enough not to
# bleed across a chair leg or a leaf stem.
EDGE_REFINE_FRACTION = 0.0035


class ExtractionError(RuntimeError):
    """Extraction could not be completed. Carries a reason fit to show a user."""


@dataclass
class ExtractedObject:
    """One accepted object, at the working resolution."""

    identifier: str
    label: str
    confidence: float
    box: tuple[int, int, int, int]
    mask: np.ndarray  # (H, W) bool at working resolution
    is_mirror: bool
    metrics: dict = field(default_factory=dict)
    cleanup_steps: dict = field(default_factory=dict)

    @property
    def quality(self) -> float:
        """
        How much this object's mask is trusted when it contests pixels.

        Detection confidence and mask rank both matter: a confident detection
        with a poor mask should not win, and neither should a crisp mask of
        something the detector was unsure about.
        """
        return 0.5 * self.confidence + 0.5 * float(self.metrics.get("rank", 0.0))


@dataclass
class ExtractionResult:
    """Everything one photo produced."""

    objects: list[ExtractedObject]
    surface_masks: dict[str, np.ndarray]
    working_shape: tuple[int, int]
    original_shape: tuple[int, int]
    rejected: list[dict]
    timings: dict
    detections_raw: int

    # Only populated when config.debug is set: the raw detections, the ones that
    # survived filtering, and each object's mask as SAM returned it before
    # cleanup — the three things needed to tell *where* a bad cut-out went wrong.
    trace: dict = field(default_factory=dict)


def _fit_within(rgb: np.ndarray, longest_edge: int) -> np.ndarray:
    """Downscale so the longest edge is at most `longest_edge`. Never upscales."""
    height, width = rgb.shape[:2]

    scale = longest_edge / max(height, width)

    if scale >= 1.0:
        return rgb

    return cv2.resize(
        rgb,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def _bounds(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Tight bounding box of a mask, in its own coordinates."""
    ys, xs = np.nonzero(mask)

    if ys.size == 0:
        return (0, 0, 0, 0)

    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def to_original(
    mask: np.ndarray, shape: tuple[int, int], guide: np.ndarray | None = None
) -> np.ndarray:
    """
    Map a working-resolution mask back to the photo's exact dimensions.

    Resized as a float and thresholded at 0.5 rather than resized as a binary
    image: nearest-neighbour on a boolean mask reproduces the working grid's
    staircase at full size, and the matting pass then faithfully feathers that
    staircase instead of the object's real edge.

    When `guide` — the original photograph — is supplied, the interpolated
    boundary is additionally snapped onto the photo's own edges before
    thresholding. This is the edge/boundary refinement stage: interpolation
    alone carries a boundary decided on SAM's 256x256 logit grid, and no
    downstream pass can recover detail that grid never held.

    Omitting `guide` gives exactly the previous behaviour, so every caller that
    does not want refinement is unaffected.
    """
    height, width = shape

    if mask.shape != shape:
        coarse = cv2.resize(
            mask.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR
        )
    elif guide is None:
        return mask
    else:
        coarse = mask.astype(np.float32)

    if guide is None:
        return coarse >= 0.5

    radius = int(np.clip(round(EDGE_REFINE_FRACTION * max(height, width)), 3, 12))

    band = max(2, radius)

    # Refine inside the object's own neighbourhood only. The guided filter
    # allocates several float images the size of what it is given, and running
    # it over the whole canvas for every object would cost hundreds of
    # megabytes on a large upload for no benefit — the boundary being corrected
    # is right here.
    binary = coarse >= 0.5

    if not binary.any():
        return binary

    ys, xs = np.nonzero(binary)

    pad = 2 * band + 2

    y0 = max(0, int(ys.min()) - pad)
    y1 = min(height, int(ys.max()) + 1 + pad)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(width, int(xs.max()) + 1 + pad)

    refined = np.zeros(shape, dtype=bool)

    refined[y0:y1, x0:x1] = matting.snap_to_edges(
        np.ascontiguousarray(coarse[y0:y1, x0:x1]),
        np.ascontiguousarray(guide[y0:y1, x0:x1]),
        radius,
        band,
    )

    return refined


def extract(rgb: np.ndarray, config: ExtractionConfig) -> ExtractionResult:
    """
    Run the whole pipeline over one photo.

    `rgb` is the original, full-resolution, EXIF-corrected photo. Raises
    ExtractionError when a required model is missing; an empty object list is a
    valid result, not an error — some photos genuinely contain nothing to lift.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ExtractionError("Expected an RGB image.")

    original_shape = rgb.shape[:2]

    work = _fit_within(rgb, config.max_inference_edge)

    working_shape = work.shape[:2]

    timings: dict = {}

    # ---- surfaces -------------------------------------------------------
    clock = time.perf_counter()

    try:
        # One forward pass gives both the structural surfaces and the semantic
        # mirror class that decides object-vs-mirror further down.
        (
            surface_masks,
            mirror_map,
            background_map,
            glazing_map,
            class_map,
            class_labels,
        ) = surfaces.analyse(work)
    except surfaces.SurfacesUnavailable as error:
        raise ExtractionError(str(error)) from error
    except Exception as error:
        raise ExtractionError(f"Surface segmentation failed: {error}") from error

    # Two maps, used for two different jobs — see filtering.structural_map.
    structural = filtering.structural_map(
        surface_masks, working_shape, config.segmentation.surface_guard_px
    )

    trim_surface = filtering.structural_map(
        surface_masks,
        working_shape,
        config.segmentation.surface_guard_px,
        only=config.segmentation.trim_surfaces,
    )

    # What an enclosed gap in a mask has to be made of before `cleanup.clean`
    # is allowed to seal it. The structural map alone is not enough: glazing is
    # not wall, floor or ceiling, and a window seen between a plant's leaves is
    # a gap that must stay open or the cut-out pastes a piece of window back
    # over the retiled room. Adding the background map here keeps that gap open
    # while still letting a vase standing on a table be sealed into it.
    hole_guard = structural | background_map

    timings["surfaces_s"] = round(time.perf_counter() - clock, 2)

    # ---- detection ------------------------------------------------------
    clock = time.perf_counter()

    try:
        detections = dino.detect(work, config)
    except dino.DinoUnavailable as error:
        raise ExtractionError(str(error)) from error
    except Exception as error:
        raise ExtractionError(f"Grounding DINO failed: {error}") from error

    timings["detection_s"] = round(time.perf_counter() - clock, 2)

    raw_count = len(detections)

    trace: dict = {"raw_detections": list(detections)} if config.debug else {}

    detections, rejected = filtering.apply(detections, structural, working_shape, config)

    if config.debug:
        trace["kept_detections"] = list(detections)
        trace["sam_masks"] = []

    if not detections:
        return ExtractionResult(
            objects=[],
            surface_masks=surface_masks,
            working_shape=working_shape,
            original_shape=original_shape,
            rejected=rejected,
            timings=timings,
            detections_raw=raw_count,
            trace=trace,
        )

    # ---- segmentation ---------------------------------------------------
    clock = time.perf_counter()

    try:
        embedding = sam2.encode(work)

        logits, scores, _ = sam2.masks_for_boxes(embedding, [d.box for d in detections])
    except sam2.Sam2Unavailable as error:
        raise ExtractionError(str(error)) from error
    except Exception as error:
        raise ExtractionError(f"SAM 2 segmentation failed: {error}") from error

    timings["segmentation_s"] = round(time.perf_counter() - clock, 2)

    # ---- selection, validation, cleanup ---------------------------------
    clock = time.perf_counter()

    objects: list[ExtractedObject] = []

    # Mirrors are collected here and emitted after the loop, one per surface.
    mirror_candidates: list[tuple[np.ndarray, str, float, dict]] = []

    for index, detection in enumerate(detections):
        chosen, report = selection.choose(
            logits[index],
            scores[index],
            embedding,
            detection.box,
            structural,
            trim_surface,
            # A window detected *as an object* keeps its own glazing; glazing
            # inside anything else is the window behind it.
            None if config.is_glazing(detection.label) else background_map,
            config,
        )

        if chosen is None:
            rejected.append(
                {
                    "label": detection.label,
                    "confidence": round(detection.confidence, 3),
                    "box": list(detection.box),
                    "reason": report.get("reason", "no acceptable mask"),
                    "candidates": report.get("candidates", []),
                }
            )
            continue

        if config.debug:
            trace["sam_masks"].append((detection.label, chosen.mask.copy()))

        # Object, mirror or reflection — decided on pixel evidence rather than
        # on what the detector called it. Done before cleanup so a confirmed
        # mirror is completed to its full surface and then tidied as one piece.
        verdict, evidence = mirrors.classify(
            chosen.mask, detection.names, mirror_map, config, glazing_map
        )

        if verdict == "reflection":
            rejected.append(
                {
                    "label": detection.label,
                    "confidence": round(detection.confidence, 3),
                    "box": list(detection.box),
                    "reason": evidence.get("verdict", "a reflection in a mirror"),
                }
            )
            continue

        if verdict == "mirror":
            # Held back rather than emitted: several boxes can land on one
            # mirror, and they are collapsed into a single entry per surface
            # once every detection has been classified.
            mirror_candidates.append(
                (chosen.mask, detection.label, detection.confidence, evidence)
            )
            continue

        mask, steps = cleanup.clean(chosen.mask, config, hole_guard)

        if not mask.any():
            rejected.append(
                {
                    "label": detection.label,
                    "confidence": round(detection.confidence, 3),
                    "box": list(detection.box),
                    "reason": "mask was empty after cleanup",
                }
            )
            continue

        metrics = {**chosen.metrics, "chosen_candidate": chosen.candidate, **evidence}

        if detection.aliases:
            metrics["also_detected_as"] = list(detection.aliases)

        label = detection.label

        if evidence.get("detector_called_it_mirror"):
            # The pixels disproved the name. Publishing "mirror" on something
            # this pipeline has just determined is not a mirror would make the
            # ALL_OBJECTS membership list read like a bug, so the disproven
            # label is replaced and kept in the metadata instead.
            metrics["detector_label"] = label

            label = "object"

        objects.append(
            ExtractedObject(
                identifier=f"object_{len(objects) + 1:03d}",
                label=label,
                confidence=detection.confidence,
                box=detection.box,
                mask=mask,
                is_mirror=False,
                metrics=metrics,
                cleanup_steps=steps,
            )
        )

    # ---- mirrors: one entry per surface ---------------------------------
    for entry in mirrors.group(mirror_candidates, mirror_map):
        cleaned, steps = cleanup.clean(entry["mask"], config, hole_guard)

        if not cleaned.any():
            continue

        objects.append(
            ExtractedObject(
                identifier=f"object_{len(objects) + 1:03d}",
                label=entry["label"],
                confidence=entry["confidence"],
                box=_bounds(cleaned),
                mask=cleaned,
                is_mirror=True,
                metrics=entry["evidence"],
                cleanup_steps=steps,
            )
        )

    # A mirror the detector never boxed would otherwise disappear: every
    # detection lying on it is dropped as a reflection, so nothing would carry
    # it. Adding the bare region keeps the guarantee that a mirror is never lost
    # and never lands among the ordinary objects.
    for region in mirrors.unclaimed(
        mirror_map, [item.mask for item in objects if item.is_mirror]
    ):
        cleaned, steps = cleanup.clean(region, config, hole_guard)

        if not cleaned.any():
            continue

        objects.append(
            ExtractedObject(
                identifier=f"object_{len(objects) + 1:03d}",
                label="mirror",
                confidence=0.0,
                box=_bounds(cleaned),
                mask=cleaned,
                is_mirror=True,
                metrics={
                    "source": "semantic mirror class",
                    "mirror_pixel_share": 1.0,
                    "detector_called_it_mirror": False,
                    "verdict": "mirror surface no detection box covered",
                },
                cleanup_steps=steps,
            )
        )

    # ---- curtains, blinds and windows nothing boxed (openings.py) --------
    # Additive: only regions no accepted mask claims, so every object above
    # is left exactly as it is.
    claimed = np.zeros(working_shape, dtype=bool)

    for item in objects:
        claimed |= item.mask

    found_openings, refused_openings = openings.sweep(
        embedding,
        class_map,
        class_labels,
        claimed,
        structural,
        surface_masks.get("floor", np.zeros(working_shape, dtype=bool)),
        config,
        work,
    )

    rejected.extend(refused_openings)

    for mask, label, metrics in found_openings:
        cleaned, steps = cleanup.clean(mask, config, hole_guard)

        if not cleaned.any():
            continue

        objects.append(
            ExtractedObject(
                identifier=f"object_{len(objects) + 1:03d}",
                label=label,
                confidence=0.0,
                box=_bounds(cleaned),
                mask=cleaned,
                is_mirror=False,
                metrics=metrics,
                cleanup_steps=steps,
            )
        )

    # ---- every prop nothing named (props.py) ---------------------------
    # Additive: a prop is whatever the class map calls NOT structure that no
    # accepted mask claims. Confident wall/floor is trimmed off its outer edge
    # exactly as for a detection, so no strip of old floor rides along.
    for item in objects:
        claimed |= item.mask

    found_props, refused_props = props.sweep(
        embedding, class_map, class_labels, claimed, structural, config, work
    )

    rejected.extend(refused_props)

    for mask, label, metrics in found_props:
        trimmed = selection.trim_surfaces(mask, trim_surface, background_map, config)

        cleaned, steps = cleanup.clean(trimmed, config, hole_guard)

        if not cleaned.any():
            continue

        objects.append(
            ExtractedObject(
                identifier=f"object_{len(objects) + 1:03d}",
                label=label,
                confidence=0.0,
                box=_bounds(cleaned),
                mask=cleaned,
                is_mirror=False,
                metrics=metrics,
                cleanup_steps=steps,
            )
        )

    timings["selection_s"] = round(time.perf_counter() - clock, 2)

    # ---- overlap and reflections ----------------------------------------
    objects, reflections = overlap.suppress_reflections(objects, config)

    rejected.extend(reflections)

    adjustments = overlap.resolve(objects, config)

    # Reassignment can empty a mask outright; such an object has nothing left to
    # write and is dropped rather than producing a blank PNG.
    surviving = []

    for item in objects:
        if item.mask.any():
            surviving.append(item)
            continue

        rejected.append(
            {
                "id": item.identifier,
                "label": item.label,
                "confidence": round(item.confidence, 3),
                "reason": "every pixel was claimed by a stronger overlapping object",
            }
        )

    # Renumber so the ids that reach the client have no gaps in them.
    for position, item in enumerate(surviving, start=1):
        item.identifier = f"object_{position:03d}"

    if adjustments:
        timings["overlap_adjustments"] = len(adjustments)

    return ExtractionResult(
        objects=surviving,
        surface_masks=surface_masks,
        working_shape=working_shape,
        original_shape=original_shape,
        rejected=rejected,
        timings=timings,
        detections_raw=raw_count,
        trace=trace,
    )
