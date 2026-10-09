"""
Every prop Grounding DINO did not name.

The detector only finds what its prompt list names, and no list can name every
prop a room may hold. The room's structure, though, is short and fixed (see
`PropsConfig.structure_classes`). So a prop is defined the other way round:
a region SegFormer's class map labels as NOT structure, that no accepted mask
claims, is something standing in the room that nothing lifted out.

    class map - structure classes - accepted masks
        -> regions          residual.regions (closed, size-gated)
        -> evidence guard   touches an accepted object, or SegFormer is sure
        -> SAM 2            residual._best_candidate on the cached embedding
        -> ExtractedObject  joins the object layer

The guard is what keeps tile design and wall texture out: an island in a wall
that touches no object is accepted only when SegFormer is sure of its class.
Nothing here changes an accepted object; it only adds regions nothing claimed.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from extraction import residual, sam2
from extraction.config import ExtractionConfig
from extraction.openings import confidence


def _disc(radius: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def sweep(
    embedding: sam2.Embedding,
    classes: np.ndarray,
    labels: dict[int, str],
    claimed: np.ndarray,
    structural: np.ndarray,
    config: ExtractionConfig,
    image: np.ndarray,
) -> tuple[list[tuple[np.ndarray, str, dict]], list[dict]]:
    """
    Find and segment the props no detection covered.

    Returns `(found, rejected)`; each `found` entry is `(mask, label, metrics)`
    at working resolution, ready to become an `ExtractedObject`.
    """
    rules = config.props

    if not rules.enabled:
        return [], []

    leftover = residual.residual_map(classes, labels, claimed, rules.structure_classes)

    regions = residual.regions(leftover, config)

    if not regions:
        return [], []

    h, w = classes.shape
    ring_px = max(1, int(round(rules.ring_fraction * math.hypot(h, w))))
    inverse = {v: k for k, v in labels.items()}
    probabilities = None

    rejected: list[dict] = []
    candidates: list[tuple[np.ndarray, tuple[int, int, int, int], str, dict]] = []

    for region, box, area in regions:
        ids, counts = np.unique(classes[region], return_counts=True)
        label = labels.get(int(ids[counts.argmax()]), "object") or "object"

        ring = cv2.dilate(region.astype(np.uint8), _disc(ring_px)).astype(bool) & ~region
        object_share = float((ring & claimed).sum()) / max(1, int(ring.sum()))
        evidence = {"object_ring_share": round(object_share, 3)}

        if object_share < rules.min_object_ring_share:
            if probabilities is None:
                probabilities = confidence(image)
            sure = float((probabilities[inverse[label]][region] > 0.5).mean()) if label in inverse else 0.0
            evidence["class_sure_share"] = round(sure, 3)

            if sure < rules.confident_share:
                rejected.append({
                    "label": label, "confidence": 0.0, "box": list(box), "source": "props sweep",
                    "reason": f"touches no object ({object_share:.0%} of its surroundings) and "
                    f"only {sure:.0%} sure — wall texture or tile design, not a prop",
                })
                continue

        candidates.append((region, box, label, evidence))

    if not candidates:
        return [], rejected

    logits, scores, _ = sam2.masks_for_boxes(embedding, [c[1] for c in candidates])

    taken = claimed.copy()
    found: list[tuple[np.ndarray, str, dict]] = []

    for index, (region, box, label, evidence) in enumerate(candidates):
        mask, report = residual._best_candidate(
            logits[index], scores[index], embedding, region, structural, config
        )

        if mask is None:
            rejected.append({
                "label": label, "confidence": 0.0, "box": list(box), "source": "props sweep",
                "reason": report.get("reason", "no acceptable mask"),
            })
            continue

        mask = mask & ~taken

        if int(mask.sum()) < config.residual.min_region_px:
            continue

        taken |= mask

        found.append((mask, label, {
            **report,
            **evidence,
            "source_stage": "props sweep",
            "detector_saw_it": False,
            "region_px": int(region.sum()),
            "verdict": f"no detection covered this region; the class map calls it {label!r}",
        }))

    return found, rejected
