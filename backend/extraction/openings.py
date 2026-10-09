"""
Curtains, blinds and windows Grounding DINO never boxed.

A missed curtain is not in `ALL_OBJECTS.png`, so `inpaint` leaves it in
`CLEAN_ROOM.png`, the floor/wall stage then calls it wall, and tiles are drawn
over the curtain. SegFormer has already labelled every pixel of the original
photo (`surfaces.analyse`), and ADE20K names these coverings and openings
directly. A region it names that no accepted mask claims is lifted out here as
an ordinary object: inpainted in the clean room, never tiled, restored over the
tiles exactly where it was.

    class map (curtain / blind / windowpane) - accepted masks
        -> regions          closed, size-gated (fractions of the frame)
        -> floor-ring guard a "window" surrounded by floor is a reflection
        -> SAM 2            box-prompted on the cached embedding
        -> ExtractedObject  joins the object layer

Mirrors are not handled here: `mirrors.unclaimed` already does this for them.
Nothing here changes an accepted object; it only adds regions nothing claimed.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from extraction import sam2
from extraction.config import ExtractionConfig


def _disc(radius: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def confidence(image: np.ndarray) -> np.ndarray:
    """SegFormer's class probabilities, (classes, H, W), at the image's size."""
    import surfaces

    native = surfaces.confidence_maps(image)["probabilities"]
    h, w = image.shape[:2]

    return np.stack([cv2.resize(plane, (w, h), interpolation=cv2.INTER_LINEAR) for plane in native])


def sweep(
    embedding: sam2.Embedding,
    classes: np.ndarray,
    labels: dict[int, str],
    claimed: np.ndarray,
    structural: np.ndarray,
    floor: np.ndarray,
    config: ExtractionConfig,
    image: np.ndarray,
) -> tuple[list[tuple[np.ndarray, str, dict]], list[dict]]:
    """
    Find and segment the curtains, blinds and windows no detection covered.

    Returns `(found, rejected)`; each `found` entry is `(mask, label, metrics)`
    at working resolution, ready to become an `ExtractedObject`, and every
    region that was refused says why.
    """
    rules = config.openings

    if not rules.enabled:
        return [], []

    wanted = [k for k, v in labels.items() if v in rules.classes]

    if not wanted:
        return [], []

    h, w = classes.shape
    diagonal = math.hypot(h, w)
    frame = float(h * w)

    region_map = np.isin(classes, wanted) & ~claimed

    if not region_map.any():
        return [], []

    radius = max(1, int(round(rules.close_radius_fraction * diagonal)))
    solid = cv2.morphologyEx(region_map.astype(np.uint8), cv2.MORPH_CLOSE, _disc(radius))

    count, labelled, stats, _ = cv2.connectedComponentsWithStats(solid, connectivity=8)

    ring_px = max(1, int(round(rules.ring_fraction * diagonal)))

    # SegFormer's per-class probabilities, at working resolution — computed
    # only if a region needs them (see min_object_ring_share).
    probabilities = None
    inverse = {v: k for k, v in labels.items()}

    found: list[tuple[np.ndarray, str, dict]] = []
    rejected: list[dict] = []
    candidates: list[tuple[np.ndarray, tuple[int, int, int, int], str]] = []

    for index in range(1, count):
        x, y, bw, bh, area = stats[index]
        box = (int(x), int(y), int(x + bw), int(y + bh))
        region = (labelled == index) & ~claimed

        ids, counts = np.unique(classes[region & np.isin(classes, wanted)], return_counts=True)
        label = labels.get(int(ids[counts.argmax()]), "opening") if ids.size else "opening"

        if area / frame < rules.min_region_frame_fraction:
            continue

        if label == "curtain" and bh / max(1, bw) < rules.min_curtain_height_to_width:
            rejected.append({
                "label": label, "confidence": 0.0, "box": list(box), "source": "openings sweep",
                "reason": f"wider than tall ({bh / max(1, bw):.2f}) — a curtain hangs; wall texture, not a curtain",
            })
            continue

        ring = cv2.dilate(region.astype(np.uint8), _disc(ring_px)).astype(bool) & ~region
        floor_share = float((ring & floor).sum()) / max(1, int(ring.sum()))

        if floor_share > rules.max_floor_ring_share:
            rejected.append({
                "label": label, "confidence": 0.0, "box": list(box), "source": "openings sweep",
                "reason": f"{floor_share:.0%} of its surroundings is floor — a reflection, not an opening",
            })
            continue

        object_share = float((ring & claimed).sum()) / max(1, int(ring.sum()))

        if object_share < rules.min_object_ring_share:
            if probabilities is None:
                probabilities = confidence(image)
            sure = float((probabilities[inverse[label]][region] > 0.5).mean()) if label in inverse else 0.0

            if sure < rules.confident_share:
                rejected.append({
                    "label": label, "confidence": 0.0, "box": list(box), "source": "openings sweep",
                    "reason": f"an island in the wall touching no object ({object_share:.0%} of its "
                    f"surroundings) and only {sure:.0%} sure — wall texture, not an opening",
                })
                continue

        candidates.append((region, box, label))

    if not candidates:
        return [], rejected

    logits, scores, _ = sam2.masks_for_boxes(embedding, [box for _, box, _ in candidates])

    taken = claimed.copy()

    for index, (region, box, label) in enumerate(candidates):
        region_px = max(1, int(region.sum()))
        best, best_value, report = None, -np.inf, {}

        for k in range(logits[index].shape[0]):
            mask = sam2.to_full_resolution(logits[index][k], embedding, 0.0)
            area = int(mask.sum())
            if area == 0:
                continue
            agreement = float((mask & region).sum()) / region_px
            surface_share = float((mask & structural).sum()) / area
            value = 0.5 * agreement + 0.3 * (1.0 - surface_share) + 0.2 * float(scores[index][k])
            if value > best_value:
                best, best_value = mask, value
                report = {"agreement": round(agreement, 3), "surface_share": round(surface_share, 3),
                          "sam_predicted_iou": round(float(scores[index][k]), 3), "candidate": k}

        if best is None or report["surface_share"] > rules.max_surface_share \
                or report["agreement"] < rules.min_region_agreement:
            # SAM grew onto the wall or answered a different question: the
            # class map's own region is still the opening.
            mask, source = region, "class map region"
        else:
            # SAM's edge, but never beyond the region's own neighbourhood.
            near = cv2.dilate(region.astype(np.uint8), _disc(ring_px)).astype(bool)
            mask, source = best & near, "SAM mask from class map region"

        mask = mask & ~taken

        if mask.sum() / frame < rules.min_region_frame_fraction:
            rejected.append({
                "label": label, "confidence": 0.0, "box": list(box), "source": "openings sweep",
                "reason": "nothing left after removing pixels already claimed",
            })
            continue

        taken |= mask

        found.append((mask, label, {
            **report,
            "source": source,
            "source_stage": "openings sweep",
            "detector_saw_it": False,
            "region_px": region_px,
            "verdict": f"no detection covered this {label}; the class map names it",
        }))

    return found, rejected
