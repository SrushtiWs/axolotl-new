"""
Turning accepted masks into files.

Exactly two images come out of a room photo, always both, whatever the photo
holds:

    ALL_OBJECTS.png   every accepted non-mirror object, full canvas
    MIRRORS_ONLY.png  every accepted mirror, full canvas

Both are RGBA at the photo's exact original size, and the two are disjoint: an
object appears in one or the other, never in both. A room with no mirror still
gets a MIRRORS_ONLY.png — it is simply empty, which is the honest answer and
keeps the output shape identical for every input.

`objects.json` is written alongside them. It is metadata, not an image: what was
detected, what was rejected and why, and the measurements behind each decision.

Two rules govern everything here.

**Original pixels only.** RGB comes from the uploaded photograph and alpha comes
from the mask. Nothing is generated, repainted or upscaled into existence, so a
cut-out keeps the object's real texture, colour and detail.

**Original coordinates only.** Masks are mapped back to the photo's exact
dimensions before anything is written, and each layer is the size of the photo
with every object sitting where it sits in the room. Compositing a layer back
over a retiled floor is then a plain alpha blend at the origin — no placement
maths, no resizing, nothing that can drift.

and it is not needed here: overlapping pixels have already been settled by
The two layer canvases are the union output of the accepted object masks. Object
records below are metadata only; no per-object image is written. Per-object
matting is used only to produce the final boundary alpha before each refined
mask is unioned into its layer. Overlapping pixels have already been settled by
`overlap.resolve`, and the final layer pass resolves any remaining cross-layer
contests so the two masks are disjoint.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import layers as layer_tools
import matting
from extraction import filtering, inpaint, selection
from extraction.config import ExtractionConfig
from extraction.extractor import ExtractionResult, to_original

# Padding around an object's bounding box so its edge is not clipped.
PAD = 6

# Where the refined matte is cut into a hard edge.
#
# Alpha in these files is binary: a pixel is the object or it is not. The matte
# is still computed, because it is what puts the boundary on the photo's own
# edge rather than on SAM's 256x256 grid — but it is then thresholded instead of
# being written as a soft ramp, so nothing partially transparent reaches the
# PNG. That is what removes feathering, glow and the soft halo that a ramp
# produces over a new background.
#
# The cut is above 0.5 on purpose. A pixel at the boundary of a photograph is
# already a mix of object and background, and the matte reports roughly how much
# of it is object; taking everything above half keeps pixels that are still a
# substantial blend. Biasing the cut inward drops the most contaminated of them
# while leaving the shape where it is to within a pixel.
ALPHA_CUTOFF = 0.6

# How deep the decontamination rim reaches, in pixels.
#
# Thresholding alone cannot clean the edge: the outermost surviving pixels are
# fully opaque and still carry the blend they had in the photo, which is exactly
# the pale rim a cut-out shows against a dark background. Those pixels keep
# their alpha and take their *colour* from just inside the object, so the shape
# is untouched and no background-contaminated pixel is written.
#
# One pixel, deliberately. The rim is filled from the eroded core, so a feature
# only a few pixels wide has little core left to fill from; at two the thin
# parts this pipeline works to preserve — chair legs, leaf stems, handles —
# start taking their colour from elsewhere in the object.
RIM_PX = 1

# How identical two refined masks must be before the second is treated as a
# duplicate of the first. Near-identity on purpose: anything lower would merge
# objects that genuinely overlap.
DUPLICATE_MASK_IOU = 0.92

# A detached component smaller than this is debris from a mask that was cut,
# not a part of an object. Matches the 16-pixel floor `cleanup` already uses for
# the same judgement, so the two stages agree on what "too small to be real" is.
MIN_FRAGMENT_PX = 16

ALL_OBJECTS_FILENAME = layer_tools.OUTPUT_FILENAMES["all_objects"]
MIRRORS_FILENAME = layer_tools.OUTPUT_FILENAMES["mirrors"]

# The room with those two layers removed and the surfaces behind them rebuilt.
# Opaque RGB, not RGBA: it is an ordinary photograph of an emptied room, and a
# transparent pixel in it would defeat the point.
CLEAN_ROOM_FILENAME = "CLEAN_ROOM.png"


def _crop_box(mask: np.ndarray, pad: int) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)

    if ys.size == 0:
        return None

    height, width = mask.shape

    return (
        max(0, int(xs.min()) - pad),
        max(0, int(ys.min()) - pad),
        min(width, int(xs.max()) + 1 + pad),
        min(height, int(ys.max()) + 1 + pad),
    )


def _duplicate_of(mask: np.ndarray, written: list[np.ndarray]) -> float | None:
    """
    Whether this mask is one already written, and by how much.

    `overlap.resolve` settles *contested* pixels between two objects, but it
    deliberately leaves both masks alone when their quality is close — which is
    correct for a lamp standing on a table and wrong for two detections that
    refined into the same region. Only at this point, with every refinement
    applied, can two masks be compared as they will actually be written.

    The bar is intentionally near-identity. Anything lower would start merging
    objects that genuinely overlap, which is the outcome this whole pipeline
    works to avoid.
    """
    area = int(mask.sum())

    if area == 0:
        return None

    for other in written:
        union = int((mask | other).sum())

        if union and int((mask & other).sum()) / union >= DUPLICATE_MASK_IOU:
            return int((mask & other).sum()) / union

    return None


def _cut(rgb: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple] | None:
    """
    One object, cropped and cut to a hard edge.

    Two stages, and the split between them is the point.

    `matting.refine` lifts the binary mask into a soft alpha that follows the
    photo's own gradients. That matte is used for *where* the boundary is — it
    is what puts the edge on the object rather than on SAM's coarse grid — and
    is then thresholded at `ALPHA_CUTOFF`. Nothing partially transparent is
    written, so the file has no feather, no glow and no soft halo.

    A hard threshold on its own leaves the pale rim that gives a cut-out away,
    because the surviving boundary pixels are opaque but are still a blend of
    the object and the room behind it. `matting.decontaminate` fixes that, but
    only acts on pixels it is told are partial — so it is handed a synthetic
    coverage map that marks the outer rim as partial while the real alpha stays
    binary. The rim keeps full opacity and takes its colour from inside the
    object.

    Shape, position and the object's interior are untouched: only the colour of
    a one-pixel rim changes, and only where that colour was contaminated by the
    background in the first place.
    """
    box = _crop_box(mask, PAD)

    if box is None:
        return None

    x0, y0, x1, y1 = box

    crop_rgb = np.ascontiguousarray(rgb[y0:y1, x0:x1])
    crop_mask = np.ascontiguousarray(mask[y0:y1, x0:x1])

    matte = matting.refine(crop_mask, crop_rgb)

    alpha = (matte >= ALPHA_CUTOFF).astype(np.float32)

    if not alpha.any():
        return None

    solid = alpha > 0.0

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * RIM_PX + 1, 2 * RIM_PX + 1)
    )

    core = cv2.erode(solid.astype(np.uint8), kernel).astype(bool)

    # Full coverage inside, "partial" on the rim, empty outside. `decontaminate`
    # keeps the colour of anything at or above CORE_ALPHA and refills the rest
    # from the nearest core pixel, which is precisely the rim.
    coverage = np.where(core, 1.0, np.where(solid, 0.5, 0.0)).astype(np.float32)

    return matting.decontaminate(crop_rgb, coverage), alpha, box


def _rgba(colour: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """
    Pack colour and alpha, clearing the colour of anything fully transparent.

    What lands in the file is the 8-bit alpha, so that is what decides which
    pixels are visible. An alpha of 0.001 rounds to zero in the PNG; keeping its
    colour would leave a scrap of room behind a pixel nothing can see, which is
    exactly what "no background in the cut-out" rules out.
    """
    opacity = np.clip(np.rint(alpha * 255.0), 0, 255).astype(np.uint8)

    visible = opacity > 0

    rgba = np.zeros((*opacity.shape, 4), dtype=np.uint8)

    rgba[..., :3] = np.where(visible[..., None], colour, 0)
    rgba[..., 3] = opacity

    return rgba


def _drop_fragments(layer: np.ndarray) -> dict:
    """
    Clear detached specks from a finished layer, in place.

    Returns what was removed, for the metadata. Anything at or above
    `MIN_FRAGMENT_PX` is left exactly as it is, so this can only ever delete
    debris — never a thin part of a real object, whose component area is
    hundreds of pixels even when it is two pixels wide.
    """
    solid = layer[..., 3] > 0

    if not solid.any():
        return {"removed": 0, "components": 0}

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        solid.astype(np.uint8), connectivity=8
    )

    doomed = [
        index
        for index in range(1, count)
        if int(stats[index, cv2.CC_STAT_AREA]) < MIN_FRAGMENT_PX
    ]

    if not doomed:
        return {"removed": 0, "components": max(0, count - 1)}

    speck = np.isin(labels, doomed)

    removed = int(speck.sum())

    layer[speck] = 0

    return {"removed": removed, "components": len(doomed)}


def _clean_room(
    rgb: np.ndarray,
    all_objects: np.ndarray,
    mirrors: np.ndarray,
    result: ExtractionResult,
    destination: Path,
) -> dict:
    """
    Write `CLEAN_ROOM.png`: the photo with its objects removed and the floor and
    wall behind them rebuilt.

    Returns the metadata describing the fill, including the verification
    measurements. This never raises: the two object layers are the contract of
    this endpoint and they are already on disk by the time this runs, so a
    missing inpainting model or a failed pass degrades to "no clean room" and
    says why, rather than failing the extraction.
    """
    height, width = rgb.shape[:2]

    hole = inpaint.hole_from_layers(all_objects, mirrors)

    try:
        clean, info = inpaint.clean_room(rgb, hole)
    except inpaint.InpaintUnavailable as error:
        return {"available": False, "reason": str(error)}
    except Exception as error:  # onnxruntime raises its own failure types
        return {"available": False, "reason": f"inpainting failed: {error}"}

    # What was actually filled, which is the hole grown by the rim — the same
    # region `clean_room` wrote, so the verification measures the real thing.
    radius = info["grow_px"]

    if radius > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
        )
        filled = cv2.dilate(hole.astype(np.uint8), kernel).astype(bool)
    else:
        filled = hole

    # The floor/wall split at the photo's own size, so the check can ask whether
    # the floor was rebuilt out of floor and the wall out of wall.
    surface_masks = {
        label: to_original(mask, (height, width))
        for label, mask in result.surface_masks.items()
        if mask.any()
    }

    info["available"] = True
    info["filename"] = CLEAN_ROOM_FILENAME
    info["checks"] = inpaint.verify(rgb, clean, filled, surface_masks)

    Image.fromarray(clean, mode="RGB").save(destination / CLEAN_ROOM_FILENAME)

    return info


def write(
    rgb: np.ndarray,
    result: ExtractionResult,
    config: ExtractionConfig,
    destination: Path,
) -> dict:
    """
    Write the two union layers for one photo and return detection metadata.

    `rgb` must be the original, full-resolution photo: it is the only source of
    colour for anything written here.
    """
    destination.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()

    height, width = rgb.shape[:2]

    all_objects = np.zeros((height, width, 4), dtype=np.uint8)
    mirrors = np.zeros((height, width, 4), dtype=np.uint8)

    records: list[dict] = []

    # The structural map at the photo's own resolution, built once rather than
    # per object, so the final validation can test a refined mask against the
    # same wall/floor/ceiling evidence the earlier stages used.
    structural = to_original(
        filtering.structural_map(
            result.surface_masks,
            result.working_shape,
            config.segmentation.surface_guard_px,
        ),
        (height, width),
    )

    # Full masks already accepted into a union layer, for the duplicate check.
    written: list[np.ndarray] = []

    for item in result.objects:
        # The original photo doubles as the guide for edge refinement: the
        # boundary is snapped onto its real edges before it becomes alpha.
        full_mask = to_original(item.mask, (height, width), rgb)

        # ---- final mask validation --------------------------------------
        ok, checks = selection.validate_final(full_mask, structural, config)

        item.metrics.update(checks)

        if not ok:
            result.rejected.append(
                {
                    "id": item.identifier,
                    "label": item.label,
                    "confidence": round(float(item.confidence), 3),
                    "reason": checks.get("reason", "failed final validation"),
                }
            )
            continue

        duplicate = _duplicate_of(full_mask, written)

        if duplicate is not None:
            result.rejected.append(
                {
                    "id": item.identifier,
                    "label": item.label,
                    "confidence": round(float(item.confidence), 3),
                    "reason": f"final mask is {duplicate:.0%} identical to an object "
                    "already written — a duplicate of it",
                }
            )
            continue

        cut = _cut(rgb, full_mask)

        if cut is None:
            continue

        written.append(full_mask)

        colour, alpha, box = cut

        x0, y0, x1, y1 = box

        rgba = _rgba(colour, alpha)

        # Composite into the layer this object belongs to. Mirrors go to their
        # own layer and are kept out of ALL_OBJECTS entirely, so the two layers
        # are disjoint and an object is never written twice.
        canvas = mirrors if item.is_mirror else all_objects

        window = canvas[y0:y1, x0:x1]

        stronger = rgba[..., 3] > window[..., 3]

        window[stronger] = rgba[stronger]

        ys, xs = np.nonzero(full_mask)

        tight = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]

        records.append(
            {
                "id": item.identifier,
                "label": item.label,
                "confidence": round(float(item.confidence), 4),
                "is_mirror": bool(item.is_mirror),
                "layer": "mirrors" if item.is_mirror else "all_objects",
                # The object's own extent in the original photo. No per-object
                # image is written — the object lives in its layer, at these
                # coordinates, which is all a caller needs to find it.
                "bbox": tight,
                # The detection box that prompted SAM, in original coordinates.
                "detection_bbox": _scaled_box(item.box, result.working_shape, (height, width)),
                "mask_area": int(full_mask.sum()),
                "alpha_area": int(round(float(alpha.sum()))),
                "source_width": int(width),
                "source_height": int(height),
                "metrics": item.metrics,
                "cleanup": item.cleanup_steps,
            }
        )

    # The two layers must be disjoint: a mirror never appears in ALL_OBJECTS.png
    # and an ordinary object never appears in MIRRORS_ONLY.png.
    #
    # Routing each object to one canvas is not quite enough to guarantee that.
    # `overlap.resolve` leaves two masks alone when their quality is close, and
    # `_cut` then feathers each one a couple of pixels past its own boundary, so
    # an object standing against a mirror can leave a thin fringe in both files
    # — measured at 157 px on one room before this, and growing with the number
    # of objects near the mirror. Awarding each contested pixel to whichever
    # layer holds it more strongly settles it at the only point where both
    # layers exist, and is a no-op when they do not overlap.
    contested = (all_objects[..., 3] > 0) & (mirrors[..., 3] > 0)

    if contested.any():
        to_mirror = contested & (mirrors[..., 3] >= all_objects[..., 3])

        all_objects[to_mirror] = 0
        mirrors[contested & ~to_mirror] = 0

    # Detached specks left by the two steps above.
    #
    # `overlap.resolve` subtracting a contested region, and the disjointness
    # award just before this, both cut across masks that were already written.
    # A cut can strand a handful of pixels from the body it belonged to, and
    # those reach the PNG as debris floating beside the object — measured at
    # one- and two-pixel components on real rooms, eight of them in one photo.
    #
    # They are removed here, at the only point where each layer is final, and
    # only when they are detached from everything and smaller than
    # `MIN_FRAGMENT_PX`. That floor is deliberately tiny: a cable, a chair leg
    # or a leaf tip is thin but *long*, so its component area is far above it,
    # and nothing with real extent is touched.
    fragments = {
        "all_objects": _drop_fragments(all_objects),
        "mirrors": _drop_fragments(mirrors),
    }

    Image.fromarray(all_objects, mode="RGBA").save(destination / ALL_OBJECTS_FILENAME)
    Image.fromarray(mirrors, mode="RGBA").save(destination / MIRRORS_FILENAME)

    result.timings["png_s"] = round(time.perf_counter() - started, 2)

    started = time.perf_counter()

    # ---- the room behind the objects ------------------------------------
    #
    # Both layers stay exactly as they were written above — they are what puts
    # the objects back at the end, and nothing here touches them. This only
    # reads their alpha to know which pixels have no room behind them yet.
    clean_room = _clean_room(rgb, all_objects, mirrors, result, destination)

    result.timings["inpaint_s"] = round(time.perf_counter() - started, 2)

    metadata = {
        "source": {"width": int(width), "height": int(height)},
        "working_resolution": [int(result.working_shape[1]), int(result.working_shape[0])],
        "counts": {
            "raw_detections": result.detections_raw,
            "accepted": len(records),
            "mirrors": sum(1 for r in records if r["is_mirror"]),
            "objects": sum(1 for r in records if not r["is_mirror"]),
            "rejected": len(result.rejected),
        },
        "timings": result.timings,
        "objects": records,
        "rejected": result.rejected,
        "models": {
            "detection": "Grounding DINO tiny (ONNX, CPU)",
            "segmentation": "SAM 2.1 Hiera-large (ONNX, CPU)",
            "surfaces": "SegFormer-B4 ADE20K (ONNX, CPU)",
            "matting": "colour guided filter + foreground decontamination",
            "inpainting": clean_room.get("model", "unavailable"),
        },
        "thresholds": {
            "box_threshold": config.detection.box_threshold,
            "text_threshold": config.detection.text_threshold,
            "duplicate_iou": config.detection.duplicate_iou,
        },
        "clean_room": clean_room,
        "fragments_removed": fragments,
    }

    (destination / "objects.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return metadata


def _scaled_box(
    box: tuple[int, int, int, int],
    working: tuple[int, int],
    original: tuple[int, int],
) -> list[int]:
    """A working-resolution detection box expressed in original pixels."""
    if working == original:
        return list(box)

    scale_x = original[1] / working[1]
    scale_y = original[0] / working[0]

    return [
        int(round(box[0] * scale_x)),
        int(round(box[1] * scale_y)),
        int(round(box[2] * scale_x)),
        int(round(box[3] * scale_y)),
    ]
