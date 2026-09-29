"""
Choosing and validating one mask per detection.

SAM 2 answers every box with three masks. They are its interpretations of "the
thing here, the part of it, the whole it belongs to" — all three are legitimate
segmentations of *something*, and only the prompting box says which one was
asked for. Taking candidate 0, or the one with the highest predicted IoU, picks
the right mask often enough to look fine and wrong often enough to matter: on a
sofa, the highest-IoU candidate is frequently a single cushion.

So candidates are ranked against the box that prompted them, and the winner
still has to pass a set of measurable checks before it is accepted. Every metric
computed here is kept and returned, because when a cut-out comes out wrong the
first question is always which of these numbers was unreasonable.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from extraction import sam2
from extraction.config import ExtractionConfig


@dataclass(frozen=True)
class Selected:
    """The chosen mask for one detection, with the numbers that chose it."""

    mask: np.ndarray  # (H, W) bool, at the working resolution
    candidate: int
    metrics: dict


def _components(mask: np.ndarray) -> int:
    count, _ = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)

    return max(0, count - 1)


def _measure(
    mask: np.ndarray,
    box: tuple[int, int, int, int],
    surface: np.ndarray,
    sam_iou: float,
) -> dict:
    """Every number used to judge a mask, computed once."""
    x0, y0, x1, y1 = box

    area = int(mask.sum())

    bbox_area = max(1, (x1 - x0) * (y1 - y0))

    frame = int(mask.size)

    inside = int(mask[y0:y1, x0:x1].sum())

    overlap = int((mask & surface).sum()) if surface.any() else 0

    return {
        "sam_predicted_iou": round(float(sam_iou), 4),
        "mask_area": area,
        "bbox_area": int(bbox_area),
        "mask_to_bbox_ratio": round(area / bbox_area, 4),
        "inside_box_ratio": round(inside / area, 4) if area else 0.0,
        "frame_fraction": round(area / frame, 5) if frame else 0.0,
        "surface_overlap_ratio": round(overlap / area, 4) if area else 0.0,
        "connected_component_count": _components(mask),
    }


def _rank(metrics: dict, config: ExtractionConfig) -> float:
    """
    How well a candidate answers the box it was prompted with.

    Two terms: how much of the box the mask fills, and how much SAM believes in
    it. Filling the box is weighted higher because it is the part that knows
    what was asked for — SAM's own score is confident about cushions too.
    """
    weight = config.segmentation.box_agreement_weight

    fill = min(1.0, metrics["mask_to_bbox_ratio"])

    base = weight * fill * metrics["inside_box_ratio"] + (1.0 - weight) * metrics[
        "sam_predicted_iou"
    ]

    # Prefer the candidate that claims the least wall/floor/ceiling.
    #
    # Filling more of the box scores higher, and for one detection on a bedroom
    # wall that rewarded the wrong answer: SAM offered a tight 22,589-pixel mask
    # of a ledge and a 198,745-pixel one that swallowed a band of the wall
    # behind it, and the larger filled more of the box. Penalising surface
    # overlap settles it the right way round.
    #
    # It is a ranking term, not a rejection. Where every candidate sits against
    # a surface — a kitchen fixture flush to a tiled wall — all three are
    # penalised alike, their order is unchanged, and the best still wins.
    penalty = config.segmentation.surface_rank_penalty * metrics["surface_overlap_ratio"]

    return base * max(0.0, 1.0 - penalty)


def _failures(metrics: dict, config: ExtractionConfig) -> list[str]:
    """Which acceptance checks a mask fails, by name. Empty means accepted."""
    rules = config.segmentation

    failures = []

    if metrics["mask_area"] < rules.min_mask_pixels:
        failures.append(f"mask_area {metrics['mask_area']} below {rules.min_mask_pixels}")

    if metrics["frame_fraction"] > rules.max_frame_fraction:
        failures.append(
            f"mask covers {metrics['frame_fraction']:.0%} of the frame — SAM escaped the object"
        )

    if metrics["inside_box_ratio"] < rules.min_inside_box:
        failures.append(
            f"only {metrics['inside_box_ratio']:.0%} of the mask is inside its detection box"
        )

    if metrics["mask_to_bbox_ratio"] < rules.min_mask_to_bbox:
        failures.append(
            f"mask fills only {metrics['mask_to_bbox_ratio']:.0%} of its box — a fragment"
        )

    # Surface overlap only condemns a mask that is also large enough to *be* a
    # surface. A small mask reading as mostly wall is an object standing against
    # one — the surface model labels the two identically — and rejecting it
    # loses real fixtures. See `surface_veto_min_frame`.
    if metrics["surface_overlap_ratio"] > rules.max_surface_overlap and (
        metrics["frame_fraction"] >= rules.surface_veto_min_frame
        or metrics["surface_overlap_ratio"] >= rules.surface_veto_always
    ):
        failures.append(
            f"{metrics['surface_overlap_ratio']:.0%} of the mask is wall/floor/ceiling "
            f"and it covers {metrics['frame_fraction']:.0%} of the frame — a surface, "
            "not an object on one"
        )

    return failures


def validate_final(
    mask: np.ndarray, surface: np.ndarray | None, config: ExtractionConfig
) -> tuple[bool, dict]:
    """
    Last check on a mask, after every refinement stage and before it becomes
    alpha in a PNG.

    `_failures` above judges a mask the moment SAM returns it. Four things then
    happen to it — OpenCV cleanup, mirror completion, overlap reassignment and
    edge refinement — any of which can shrink it, grow it or move its boundary,
    and none of which re-checks the result. A mask that passed at selection and
    was subsequently reduced to a handful of pixels, or grown out over the
    floor, currently reaches the output unexamined.

    The thresholds are deliberately the same ones selection already uses, and
    expressed as fractions so they mean the same thing at working resolution and
    at the photo's own. Nothing new or stricter is introduced here: the purpose
    is to catch a mask that *became* invalid, not to raise the bar on what was
    valid in the first place. A legitimate object that passed selection and was
    not damaged afterwards passes this unchanged.
    """
    rules = config.segmentation

    area = int(mask.sum())

    frame = int(mask.size)

    metrics: dict = {"final_mask_area": area}

    if area == 0:
        return False, {**metrics, "reason": "mask is empty after refinement"}

    fraction = area / max(1, frame)

    metrics["final_frame_fraction"] = round(fraction, 5)

    # Scaled from the working-resolution pixel floor so the same physical size
    # is required whatever resolution the photo happens to be.
    minimum = max(1, int(rules.min_mask_pixels * frame / (config.max_inference_edge ** 2)))

    if area < minimum:
        return False, {
            **metrics,
            "reason": f"only {area} px survived refinement (needs {minimum})",
        }

    if fraction > rules.max_frame_fraction:
        return False, {
            **metrics,
            "reason": f"mask covers {fraction:.0%} of the photo after refinement",
        }

    if surface is not None and surface.any():
        overlap = float((mask & surface).sum()) / area

        metrics["final_surface_overlap"] = round(overlap, 4)

        # Same size guard as `_failures`: only a mask big enough to be a surface
        # can be condemned for looking like one.
        if overlap > rules.max_surface_overlap and (
            fraction >= rules.surface_veto_min_frame
            or overlap >= rules.surface_veto_always
        ):
            return False, {
                **metrics,
                "reason": f"{overlap:.0%} of the refined mask is wall/floor/ceiling "
                f"and it covers {fraction:.0%} of the frame",
            }

    return True, metrics


def trim_surfaces(
    mask: np.ndarray,
    surface: np.ndarray,
    background: np.ndarray | None,
    config: ExtractionConfig,
) -> np.ndarray:
    """
    Take confident wall/floor/ceiling pixels back out of a mask — but only from
    its outside edge.

    The trim exists because these cut-outs are composited over a *retiled*
    floor, so a skirt of old floor riding along under an object is not cosmetic:
    it paints the original floor back over the new tiles.

    Trimming naively is worse than not trimming at all, though, and measurably
    so. SegFormer is the weaker model here — that is exactly why SAM 2 was given
    the boundary — and on a real bedroom it labelled large patches of a beige
    headboard as "wall". Subtracting that map wholesale punched holes through
    the middle of the bed, which showed up in ALL_OBJECTS.png as the room
    seeing through the furniture.

    Two guards keep it honest, both learned from real failures.

    **Only near the boundary.** Contamination is an edge phenomenon: a flap of
    wall hanging off a sofa arm, or a strip of floor under a chair, touches the
    mask's rim by construction. A patch of headboard the surface model misread
    as wall sits deep inside. So only surface pixels within a band of the
    boundary may go, which removes the first and cannot reach the second.

    SAM's own confidence was tried here instead and measured worse on every
    object: it left 35 surface pixels in the chair against 0, 747 in the plant
    against 0, and 638 in the drawer against 31, because a good deal of the
    floor a mask bleeds onto is floor SAM is perfectly confident about.

    **Only from the outside edge.** A removal that does not reach the mask's
    outer boundary is enclosed by object on every side, and is restored — a
    second guard for a mask thin enough that its interior falls inside the
    band.

    `background` — glazing and sky — is trimmed *without* the band, anywhere in
    the mask. Those classes have no history of being misread on furniture, and
    the case that motivated them needs the reach: a window seen between a
    plant's leaves is 2,419 pixels deep inside the mask, enclosed by leaf on
    every side, and every cautious guard above protects it.

    Thin parts survive all of this: the maps are eroded before they arrive here,
    so a chair leg standing on the floor sits in the gap that erosion opens
    around it rather than inside the floor region.
    """
    has_background = background is not None and background.any()

    if not surface.any() and not has_background:
        return mask

    area = int(mask.sum())

    if area == 0:
        return mask

    rules = config.segmentation

    band = max(
        rules.trim_edge_band_min_px,
        int(round(max(mask.shape) * rules.trim_edge_band_fraction)),
    )

    # Distance from every mask pixel to the nearest pixel outside it. Small
    # means "on the rim", which is exactly where bleed lives.
    depth = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)

    trimmed = mask & ~(surface & (depth <= band))

    if has_background:
        trimmed &= ~background

    removed = mask & ~trimmed

    if not removed.any():
        return mask

    # Anything the trim would take that is enclosed by surviving mask goes back.
    count, components = cv2.connectedComponents(removed.astype(np.uint8), connectivity=8)

    if count > 1:
        # A removal touches the outside if it is adjacent to a pixel that was
        # never in the mask. Dilating the mask's complement by one gives exactly
        # that neighbourhood.
        outside = cv2.dilate(
            (~mask).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        ).astype(bool)

        for index in range(1, count):
            region = components == index

            if not (region & outside).any():
                trimmed |= region

    if int(trimmed.sum()) < area * config.segmentation.min_surviving_after_trim:
        # Even edge-only trimming can disagree with the object wholesale. That
        # is a disagreement about what the object *is*, not where its edge runs,
        # and deleting a real object is the worse error.
        return mask

    return trimmed


def choose(
    logits: np.ndarray,
    scores: np.ndarray,
    embedding: sam2.Embedding,
    box: tuple[int, int, int, int],
    surface: np.ndarray,
    trim_surface: np.ndarray,
    background: np.ndarray | None,
    config: ExtractionConfig,
) -> tuple[Selected | None, dict]:
    """
    Pick the best of SAM's candidates for one box, or reject all three.

    Returns (selection, report). The report is returned either way and always
    names what happened, so a rejected object is inspectable rather than simply
    absent.
    """
    best: Selected | None = None
    best_rank = -1.0

    considered = []

    for index in range(logits.shape[0]):
        # The logit map, not just its sign: refinement needs to know which
        # pixels SAM is confident about and which it is merely guessing at.
        confidence = sam2.logits_at_full_resolution(logits[index], embedding)

        mask = confidence > config.segmentation.mask_logit_threshold

        if not mask.any():
            considered.append({"candidate": index, "rejected": ["empty mask"]})
            continue

        mask = trim_surfaces(mask, trim_surface, background, config)

        metrics = _measure(mask, box, surface, float(scores[index]))

        failures = _failures(metrics, config)

        considered.append(
            {"candidate": index, "rank": round(_rank(metrics, config), 4), **metrics,
             "rejected": failures}
        )

        if failures:
            continue

        rank = _rank(metrics, config)

        if rank > best_rank:
            best_rank = rank
            best = Selected(mask=mask, candidate=index, metrics={**metrics, "rank": round(rank, 4)})

    report = {"candidates": considered}

    if best is None:
        report["outcome"] = "rejected"
        report["reason"] = "no SAM candidate passed the acceptance checks"
    else:
        report["outcome"] = "accepted"
        report["chosen_candidate"] = best.candidate

    return best, report
