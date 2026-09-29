"""
Settling masks that claim the same pixels.

Two different situations look identical to a naive union and must not be
treated the same way:

  * a lamp standing on a table, or a chair in front of a cabinet. Both objects
    are real, both masks are right, and the overlap is simply one object being
    in front of the other. Nothing should be deleted.
  * two detections that grew into overlapping versions of the same region, or a
    mask that leaked into its neighbour. Here the contested pixels belong to one
    object and riding along in both layers duplicates them.

The rule used is that contested pixels move only when one mask is *clearly*
better than the other. Within the margin the masks are left exactly as they
are, because a wrong reassignment takes a bite out of a real boundary — and a
bitten boundary is far more visible in a cut-out than a few shared pixels.

Being clearly better is not on its own enough to start cutting, because how
much of the weaker mask is at stake decides what cutting would do to its edge:

  * most of it   -> the two masks are the same object found twice. Subtracting
                    leaves a crumb where an object was, and silently discards
                    whatever the loser had and the winner did not. They are
                    unioned instead, and the weaker entry retires.
  * a good part  -> two different objects that genuinely share a region. Both
                    end up in the same layer, so sharing the pixels is
                    invisible and cutting them is a notch in a real edge. Both
                    masks are kept whole.
  * a sliver     -> a mask that has leaked a little into its neighbour. This is
                    the case reassignment was written for, and the only one it
                    still runs on.

The mirror case is separate and stricter. A mirror shows the room again, so the
furniture reflected in it gets detected a second time: a real chair and a chair
that exists only as a reflection. The reflection is not an object in the room
and must not become a cut-out, or compositing it back will paste a second chair
over the retiled floor.
"""

from __future__ import annotations

import numpy as np

from extraction.config import ExtractionConfig


def _overlap(left: np.ndarray, right: np.ndarray) -> float:
    """Intersection over the smaller mask — catches containment, not just overlap."""
    smaller = min(int(left.sum()), int(right.sum()))

    if smaller == 0:
        return 0.0

    return float((left & right).sum()) / float(smaller)


def suppress_reflections(objects: list, config: ExtractionConfig) -> tuple[list, list[dict]]:
    """
    Drop non-mirror objects that lie inside a mirror.

    Containment is measured against the union of every mirror, so a room with
    two mirrors suppresses reflections in both. A mirror is never suppressed by
    another mirror — two mirrors facing each other are still two real mirrors.
    """
    mirrors = [item for item in objects if item.is_mirror]

    if not mirrors:
        return objects, []

    union = np.zeros_like(mirrors[0].mask)

    for mirror in mirrors:
        union |= mirror.mask

    kept = []
    dropped = []

    for item in objects:
        if item.is_mirror:
            kept.append(item)
            continue

        area = int(item.mask.sum())

        if area == 0:
            continue

        inside = float((item.mask & union).sum()) / float(area)

        if inside >= config.overlap.reflection_containment:
            dropped.append(
                {
                    "id": item.identifier,
                    "label": item.label,
                    "confidence": round(item.confidence, 3),
                    "reason": f"{inside:.0%} inside a mirror — a reflection, not an object",
                }
            )
            continue

        kept.append(item)

    return kept, dropped


def resolve(objects: list, config: ExtractionConfig) -> list[dict]:
    """
    Reassign contested pixels between overlapping objects, in place.

    Objects are compared strongest-first, so a confident, well-formed mask keeps
    its pixels and the weaker one gives them up. Returns a record of every
    reassignment for the metadata.
    """
    rules = config.overlap

    ordered = sorted(objects, key=lambda item: item.quality, reverse=True)

    adjustments: list[dict] = []

    for index, stronger in enumerate(ordered):
        for weaker in ordered[index + 1 :]:
            if not stronger.mask.any() or not weaker.mask.any():
                continue

            shared = _overlap(stronger.mask, weaker.mask)

            if shared < rules.contest_threshold:
                continue

            if stronger.quality - weaker.quality < rules.decisive_margin:
                # Too close to call. A lamp on a table lands here, and both
                # masks survive intact — which is the wanted outcome.
                adjustments.append(
                    {
                        "kept_both": [stronger.identifier, weaker.identifier],
                        "overlap": round(shared, 3),
                        "reason": "quality too close to reassign — treated as one object in front of another",
                    }
                )
                continue

            contested = stronger.mask & weaker.mask

            taken = int(contested.sum())

            weaker_area = int(weaker.mask.sum())

            # How much of the weaker mask is at stake. This is the number that
            # decides what carving would actually do to its boundary.
            weaker_share = taken / float(weaker_area) if weaker_area else 0.0

            if weaker_share >= rules.merge_containment:
                # The same object, found twice. Subtracting one from the other
                # leaves a crumb where an object was and throws away whatever
                # the loser had that the winner lacked — a lamp's pole, a
                # chair's leg. Union them instead: every pixel either version
                # claimed is kept, once, on the stronger object.
                stronger.mask = stronger.mask | weaker.mask
                weaker.mask = np.zeros_like(weaker.mask)

                adjustments.append(
                    {
                        "winner": stronger.identifier,
                        "merged": weaker.identifier,
                        "overlap": round(shared, 3),
                        "containment": round(weaker_share, 3),
                        "reason": f"{weaker_share:.0%} of this mask is inside the other — "
                        "the same object detected twice, so the two were merged",
                    }
                )
                continue

            if weaker_share > rules.max_bite:
                # Two different objects with a substantial shared region. Both
                # composite into the same layer, so sharing those pixels is
                # invisible; cutting them out is a notch in a real edge.
                adjustments.append(
                    {
                        "kept_both": [stronger.identifier, weaker.identifier],
                        "overlap": round(shared, 3),
                        "containment": round(weaker_share, 3),
                        "reason": f"carving would take {weaker_share:.0%} out of the weaker "
                        "mask — too much to cut from a real boundary",
                    }
                )
                continue

            weaker.mask = weaker.mask & ~contested

            adjustments.append(
                {
                    "winner": stronger.identifier,
                    "loser": weaker.identifier,
                    "overlap": round(shared, 3),
                    "containment": round(weaker_share, 3),
                    "pixels_reassigned": taken,
                }
            )

    return adjustments
