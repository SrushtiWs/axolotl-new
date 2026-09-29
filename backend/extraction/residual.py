"""
The objects Grounding DINO never proposed.

`dino -> filtering -> sam2 -> selection -> cleanup -> overlap` can only ever
segment things the detector put a box around. Whatever it did not name is not in
`ALL_OBJECTS.png`, so `inpaint` is given no hole there and the object survives
into `CLEAN_ROOM.png` — a floor lamp's arm as a dark line across the wall, a
plant's outer leaves as green fragments floating over the new tiles.

This module closes that gap using evidence the pipeline already paid for.
`surfaces` runs SegFormer over every photo and reads five of its 150 ADE20K
classes; the other 145 are exactly the ones that name a missed object. A pixel
the class map calls neither a room surface, nor architecture, nor glazing, and
which no accepted mask claims, is by construction something standing in the room
that nothing lifted out.

    accepted masks + class map
        -> residual        not surface, not architecture, not glazing, not claimed
        -> regions         closed, component-filtered, size-gated
        -> SAM 2           each region box-prompted against the cached embedding
        -> validate        refuse anything that is mostly room surface
        -> ExtractedObject joins the layer, and so gets a hole in the clean room

The encoder has already run for this photo, so the whole sweep is decoder calls.
Measured on a 1408x768 living room: 25 regions, 1.8 s, against the 66 s the
detector itself costs.

Two things this deliberately does not do. It never proposes a region the class
map calls glazing or sky — a window is the view out of the room, not a prop, and
the pipeline keeps it out of the object layer unless a detection *is* the window.
And it never returns a mask that is mostly confident wall, floor or ceiling,
which is the guard that separates finding a lamp's pole from annexing the lit
patch of tile behind it.
"""

from __future__ import annotations

import cv2
import numpy as np

from extraction import sam2
from extraction.config import ExtractionConfig


def _bounds(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)

    if ys.size == 0:
        return (0, 0, 0, 0)

    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def residual_map(
    classes: np.ndarray,
    labels: dict[int, str],
    claimed: np.ndarray,
    excluded: frozenset[str],
) -> np.ndarray:
    """
    Pixels that name an object and belong to no accepted mask.

    `excluded` is every class name that is *not* evidence of a prop: the room
    surfaces, the architecture that cannot be carried out of the room, and the
    glazing and sky that are the view through it.
    """
    thing = np.zeros(classes.shape, dtype=bool)

    for class_id in np.unique(classes):
        if labels.get(int(class_id), "") in excluded:
            continue

        thing |= classes == class_id

    return thing & ~claimed


def regions(
    residual: np.ndarray, config: ExtractionConfig
) -> list[tuple[np.ndarray, tuple[int, int, int, int], int]]:
    """
    The residual split into regions worth prompting, largest first.

    Closed before components are found, so a leaf the class map broke into three
    specks is one region rather than three rejected ones.
    """
    rules = config.residual

    if not residual.any():
        return []

    radius = max(0, int(rules.close_radius))

    solid = residual.astype(np.uint8)

    if radius:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
        )
        solid = cv2.morphologyEx(solid, cv2.MORPH_CLOSE, kernel)

    count, labelled, stats, _ = cv2.connectedComponentsWithStats(solid, connectivity=8)

    frame = float(residual.size)

    found = []

    for index in range(1, count):
        x, y, w, h, area = stats[index]

        if area < rules.min_region_px:
            continue

        if area / frame > rules.max_region_frame_fraction:
            continue

        found.append((labelled == index, (int(x), int(y), int(x + w), int(y + h)), int(area)))

    found.sort(key=lambda item: item[2], reverse=True)

    return found[: max(0, int(rules.max_regions))]


def _best_candidate(
    logits: np.ndarray,
    scores: np.ndarray,
    embedding: sam2.Embedding,
    region: np.ndarray,
    structural: np.ndarray,
    config: ExtractionConfig,
) -> tuple[np.ndarray | None, dict]:
    """
    Which of SAM's three candidates best explains this residual region.

    Scored on three things, and the middle one is what keeps the sweep honest:
    how much of the region the mask actually covers, how little of the mask is
    confident room surface, and SAM's own predicted IoU.
    """
    rules = config.residual

    region_px = max(1, int(region.sum()))

    best: np.ndarray | None = None
    best_value = -np.inf
    best_report: dict = {}

    for index in range(logits.shape[0]):
        mask = sam2.to_full_resolution(logits[index], embedding, 0.0)

        area = int(mask.sum())

        if area == 0:
            continue

        agreement = float((mask & region).sum()) / region_px
        surface_share = float((mask & structural).sum()) / area

        value = (
            0.5 * agreement + 0.3 * (1.0 - surface_share) + 0.2 * float(scores[index])
        )

        if value > best_value:
            best_value = value
            best = mask
            best_report = {
                "agreement": round(agreement, 3),
                "surface_share": round(surface_share, 3),
                "sam_predicted_iou": round(float(scores[index]), 3),
                "candidate": index,
            }

    if best is None:
        return None, {"reason": "SAM returned no non-empty candidate"}

    # Mostly surface means the mask grew into the wall or floor behind the
    # object rather than onto the object. Refuse it outright: a hole punched in
    # a surface is worse than an object left standing.
    if best_report["surface_share"] > rules.max_surface_share:
        return None, {
            **best_report,
            "reason": f"{best_report['surface_share']:.0%} of the mask is confident "
            f"wall/floor/ceiling (max {rules.max_surface_share:.0%})",
        }

    if best.sum() / float(region.size) > rules.max_region_frame_fraction:
        return None, {
            **best_report,
            "reason": "SAM's mask covers more of the frame than a missed prop can",
        }

    # SAM answering a different question is not a failure worth discarding the
    # evidence over — the class map's own region is still a real object. Fall
    # back to it rather than losing the pixels.
    if best_report["agreement"] < rules.min_region_agreement:
        return region, {**best_report, "source": "class map region (SAM disagreed)"}

    return best, {**best_report, "source": "SAM mask from class map region"}


def sweep(
    embedding: sam2.Embedding,
    classes: np.ndarray,
    labels: dict[int, str],
    claimed: np.ndarray,
    structural: np.ndarray,
    excluded: frozenset[str],
    config: ExtractionConfig,
) -> tuple[list[tuple[np.ndarray, str, dict]], list[dict]]:
    """
    Find and segment the objects no detection covered.

    Returns `(found, rejected)`, where each `found` entry is
    `(mask, label, metrics)` ready to become an `ExtractedObject`, and every
    region that was refused says why.

    `claimed` is the union of every accepted mask, so nothing here can duplicate
    an object that was already lifted out; each accepted region is added to it
    as it is taken, so two adjacent regions cannot both claim the same pixels.
    """
    rules = config.residual

    if not rules.enabled:
        return [], []

    residual = residual_map(classes, labels, claimed, excluded)

    candidates = regions(residual, config)

    if not candidates:
        return [], []

    boxes = [box for _, box, _ in candidates]

    logits, scores, _ = sam2.masks_for_boxes(embedding, boxes)

    taken = claimed.copy()

    found: list[tuple[np.ndarray, str, dict]] = []
    rejected: list[dict] = []

    for index, (region, box, area) in enumerate(candidates):
        # What the class map calls this region, by majority. Reported rather
        # than guessed at: it is the only name anything has for an object the
        # detector never saw.
        ids, counts = np.unique(classes[region], return_counts=True)
        label = labels.get(int(ids[counts.argmax()]), "object") or "object"

        mask, report = _best_candidate(
            logits[index], scores[index], embedding, region, structural, config
        )

        if mask is None:
            rejected.append(
                {
                    "label": label,
                    "confidence": 0.0,
                    "box": list(box),
                    "source": "residual sweep",
                    "reason": report.get("reason", "no acceptable mask"),
                }
            )
            continue

        # Never re-claim pixels an accepted object already holds.
        mask = mask & ~taken

        if int(mask.sum()) < rules.min_region_px:
            rejected.append(
                {
                    "label": label,
                    "confidence": 0.0,
                    "box": list(box),
                    "source": "residual sweep",
                    "reason": "nothing left after removing pixels already claimed",
                }
            )
            continue

        taken |= mask

        found.append(
            (
                mask,
                label,
                {
                    **report,
                    "source_stage": "residual sweep",
                    "detector_saw_it": False,
                    "region_px": area,
                    "final_mask_area": int(mask.sum()),
                    "verdict": (
                        f"no detection covered this region; the class map calls it "
                        f"{label!r}"
                    ),
                },
            )
        )

    return found, rejected
