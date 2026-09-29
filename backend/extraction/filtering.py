"""
Deciding which Grounding DINO detections are worth segmenting.

Three jobs, in order:

  * size sanity — a box that is the whole frame, or a speck, is not an object;
  * duplicate handling — a permissive prompt list is *designed* to fire several
    phrases at one physical object ("sofa", "couch", "large sofa"), and exactly
    one of them must survive;
  * structural rejection — floor, wall and ceiling are the surfaces being
    retiled, and a detection that is really one of them must never become a
    cut-out.

None of this runs a model. It is arithmetic over boxes, plus a lookup against
the structural map, so it is cheap enough to run before SAM rather than after —
which is the point. Every detection removed here is a SAM call not made.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from extraction.config import ExtractionConfig
from extraction.dino import Detection


def _intersection(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])

    return max(0, x1 - x0) * max(0, y1 - y0)


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    overlap = _intersection(a, b)

    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])

    union = area_a + area_b - overlap

    return overlap / union if union > 0 else 0.0


def containment(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Intersection over the smaller box — catches one box sitting inside another."""
    overlap = _intersection(a, b)

    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])

    smaller = min(area_a, area_b)

    return overlap / smaller if smaller > 0 else 0.0


def _union_box(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> tuple[int, int, int, int]:
    """The smallest box containing both. Both are already inside the frame."""
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def by_size(
    detections: list[Detection], shape: tuple[int, int], config: ExtractionConfig
) -> tuple[list[Detection], list[dict]]:
    """Drop boxes that are too large or too small to be a piece of furniture."""
    height, width = shape

    frame = float(height * width)

    kept: list[Detection] = []
    rejected: list[dict] = []

    for detection in detections:
        fraction = detection.area / frame

        if fraction > config.detection.max_box_frame_fraction:
            rejected.append(
                {
                    "label": detection.label,
                    "confidence": round(detection.confidence, 3),
                    "box": list(detection.box),
                    "reason": f"box covers {fraction:.0%} of the frame — this is the room",
                }
            )
            continue

        if fraction < config.detection.min_box_frame_fraction:
            rejected.append(
                {
                    "label": detection.label,
                    "confidence": round(detection.confidence, 3),
                    "box": list(detection.box),
                    "reason": f"box covers {fraction:.2%} of the frame — too small to be an object",
                }
            )
            continue

        kept.append(detection)

    return kept, rejected


def _is_duplicate(
    candidate: Detection, winner: Detection, config: ExtractionConfig
) -> bool:
    """
    Whether two detections are the same physical object.

    Overlap alone cannot decide this, and treating it as though it could is a
    mistake with teeth: measured on a real bedroom photo, containment-merging
    across labels absorbed the *bed* into a neighbouring detection and the
    largest object in the room vanished from the output.

    What separates the two cases is the label. Several phrases firing on one
    object — "sofa"/"couch", or the six mirror prompts — produce boxes that
    agree closely, so a high IoU is the honest signal for that, and it is the
    only thing allowed to merge across labels.

    Containment is a different relationship: a small box inside a large one is
    usually a real object standing in front of another — a lamp on a table, a
    bedside table against a bed. So containment merges only when both
    detections carry the *same* label, where it means one phrase fired twice on
    one object at different extents.

    The same label is still not enough on its own, and the missing half is
    size. "Different extents" means boxes of comparable size; a box many times
    the area of the one it contains is a second object, whatever it is called.
    Measured on a bedroom photo, the bed came back as "chair" at 135,960 px and
    an armchair in front of it as "chair" at 5,886 px. The armchair scored
    higher, so the bed — the largest object in the room — was merged into it and
    never reached SAM. Requiring the two boxes to be within
    `duplicate_area_ratio` of each other keeps the case this rule was written
    for and stops that one.
    """
    if iou(candidate.box, winner.box) >= config.detection.duplicate_iou:
        return True

    if candidate.label != winner.label:
        return False

    if containment(candidate.box, winner.box) < config.detection.duplicate_containment:
        return False

    larger = max(candidate.area, winner.area)

    if larger <= 0:
        return False

    return min(candidate.area, winner.area) / larger >= config.detection.duplicate_area_ratio


def deduplicate(
    detections: list[Detection], config: ExtractionConfig
) -> tuple[list[Detection], list[dict]]:
    """
    Collapse detections that are the same physical object.

    The whole reason "sofa", "couch" and "large sofa" are all in the prompt list
    is to catch the object under whichever word the model prefers for it;
    keeping all three would produce three PNGs of one sofa. Boxes are considered
    strongest-first, so the surviving detection also carries the most confident
    label.

    Mirrors are the case this matters most for: six prompts point at them, and
    this is what turns those six back into one mirror.

    **The winner keeps the strongest label, but the widest extent.** Merging
    used to discard every box except the winner's, and the winner is chosen on
    confidence — which says how sure the model is of the *word*, not how much of
    the object the box covers. Measured on a room with a plant standing beside a
    sofa, the plant won as "artwork" with a box ending at y=295, its foliage
    only; two merged detections of the same plant put it at y=349 and y=396,
    down through the leaves to the rim of its pot. The winner's truncated box
    went to SAM, so the plant was segmented without its lower half and
    `ALL_OBJECTS.png` held a bush floating above its own pot.

    So a merged box is folded into the winner's extent rather than thrown away.
    Every box in that union already passed `_is_duplicate` against the winner,
    which is what makes it evidence about one object's size rather than a second
    object being annexed: an IoU merge bounds the union at about 1.5x the
    smaller box, and a same-label containment merge at about 3x, both of them
    limited by a box Grounding DINO actually produced.

    Comparisons run against the winner's *original* box, not its growing one, so
    a chain of merges cannot walk a box across the room one neighbour at a time.
    """
    ordered = sorted(detections, key=lambda item: item.confidence, reverse=True)

    kept: list[Detection] = []
    aliases: list[list[str]] = []
    extents: list[tuple[int, int, int, int]] = []
    merged: list[dict] = []

    for detection in ordered:
        duplicate_of = None

        for position, winner in enumerate(kept):
            if _is_duplicate(detection, winner, config):
                duplicate_of = position
                break

        if duplicate_of is None:
            kept.append(detection)
            aliases.append([])
            extents.append(detection.box)
            continue

        extents[duplicate_of] = _union_box(extents[duplicate_of], detection.box)

        # The merged phrase is not thrown away: it is recorded on the winner, so
        # a "mirror door" absorbed by a stronger "door" can still route the
        # object to the mirror layer.
        if detection.label not in aliases[duplicate_of]:
            aliases[duplicate_of].append(detection.label)

        merged.append(
            {
                "label": detection.label,
                "confidence": round(detection.confidence, 3),
                "box": list(detection.box),
                "merged_into": kept[duplicate_of].label,
                "extent": list(extents[duplicate_of]),
                "reason": "same physical object as a stronger detection",
            }
        )

    resolved = [
        replace(detection, aliases=tuple(names), box=extent)
        for detection, names, extent in zip(kept, aliases, extents)
    ]

    return resolved[: config.detection.max_detections], merged


def structural_map(
    surfaces: dict[str, np.ndarray],
    shape: tuple[int, int],
    guard: int,
    only: tuple[str, ...] | None = None,
) -> np.ndarray:
    """
    Surfaces, eroded back from their own uncertain boundary.

    `only` restricts which surfaces are included. Two maps are built from this,
    and the difference between them matters:

      * all three surfaces, for *rejecting* a detection whose box is really the
        room — that judgement is made on the box, before SAM runs, and being
        wrong about it costs one spurious object;
      * the floor alone, for *trimming* pixels out of an accepted mask — that
        edits SAM's boundary, and being wrong about it damages a real object.

    The asymmetry is deliberate. Measured on a real bedroom, SegFormer labelled
    large patches of a beige headboard as "wall"; trimming against that map
    punched holes straight through the bed. The floor is both the surface the
    tile renderer actually repaints — which is the entire reason to trim — and
    the one the model reads reliably, so trimming is restricted to it.

    Eroding first is what makes either map safe to use: only pixels the model is
    confident about count, so a cabinet standing against a wall keeps its edge.
    """
    import cv2

    combined = np.zeros(shape, dtype=bool)

    for label, mask in surfaces.items():
        if only is not None and label not in only:
            continue

        if mask is not None and mask.shape == shape:
            combined |= mask

    if not combined.any() or guard <= 0:
        return combined

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * guard + 1, 2 * guard + 1))

    return cv2.erode(combined.astype(np.uint8), kernel).astype(bool)


def by_structure(
    detections: list[Detection], surface: np.ndarray, config: ExtractionConfig
) -> tuple[list[Detection], list[dict]]:
    """
    Reject detections that are really the floor, the wall or the ceiling.

    Semantic filtering alone is not enough here. Grounding DINO does not have
    "floor" in its prompt list, so a floor never comes back under that name —
    but a polished floor genuinely does come back as "table", and on a real
    photo it did: a box spanning the full width of the room. What gives it away
    is not its label but that almost every pixel under it is confident floor.

    The test is on the *box*, before SAM runs, and its threshold
    (`max_box_surface_overlap`) is set so that only a box which is essentially
    nothing but surface fails it. That is much stricter than the equivalent test
    on a finished mask, and the asymmetry is the whole point: a box is weak
    evidence.

    A box drawn tightly around a real object standing against a wall is mostly
    wall — it holds the object plus every surface pixel visible around it.
    Measured on a bare kitchen, judging boxes at the mask threshold rejected 18
    of 19 detections, the sink, countertop and window among them, because every
    fixture in that room is flush against tiled walls. Whether a detection is
    genuinely a surface is decided later, on its mask, where the evidence is.
    """
    if not surface.any():
        return detections, []

    kept: list[Detection] = []
    rejected: list[dict] = []

    for detection in detections:
        if config.is_structural(detection.label):
            rejected.append(
                {
                    "label": detection.label,
                    "confidence": round(detection.confidence, 3),
                    "box": list(detection.box),
                    "reason": "structural label",
                }
            )
            continue

        x0, y0, x1, y1 = detection.box

        window = surface[y0:y1, x0:x1]

        if window.size == 0:
            continue

        ratio = float(window.sum()) / float(window.size)

        if ratio >= config.segmentation.max_box_surface_overlap:
            rejected.append(
                {
                    "label": detection.label,
                    "confidence": round(detection.confidence, 3),
                    "box": list(detection.box),
                    "reason": f"{ratio:.1%} of the box is confident wall/floor/ceiling — "
                    "nothing but surface",
                }
            )
            continue

        kept.append(detection)

    return kept, rejected


def apply(
    detections: list[Detection],
    surface: np.ndarray,
    shape: tuple[int, int],
    config: ExtractionConfig,
) -> tuple[list[Detection], list[dict]]:
    """Run the whole filter chain, returning survivors and an audit trail."""
    detections, too_big = by_size(detections, shape, config)

    detections, structural = by_structure(detections, surface, config)

    detections, duplicates = deduplicate(detections, config)

    return detections, too_big + structural + duplicates
