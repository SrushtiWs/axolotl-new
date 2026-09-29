"""
Deterministic OpenCV cleanup of a SAM mask.

SAM 2 produces a good boundary and a slightly untidy interior: a few detached
specks where the box clipped something else, pinholes inside textured surfaces,
the occasional single-pixel spur. This module removes those and nothing else.

The governing rule is that cleanup must not shrink the object. Every operation
here is chosen against that:

  * closing runs at a radius of 2 and only ever *adds* pixels, so it cannot eat
    a chair leg or a leaf tip;
  * opening runs at a radius of 1 — smaller than the closing — because opening
    removes pixels and anything larger starts taking thin real features;
  * hole filling is limited to enclosed gaps, so the space between chair legs
    (which reaches the image border) stays open;
  * component removal is relative to the mask's own area, not the frame's, so a
    small object is not judged by the standards of a large one.

There is no unconditional erosion anywhere in this file, deliberately.
"""

from __future__ import annotations

import cv2
import numpy as np

from extraction.config import ExtractionConfig


def _disk(radius: int) -> np.ndarray:
    radius = max(1, int(radius))

    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def drop_small_components(mask: np.ndarray, minimum: int) -> np.ndarray:
    """Keep components at or above `minimum` pixels. Always keeps the largest."""
    count, components, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )

    if count <= 1:
        return mask

    areas = stats[1:, cv2.CC_STAT_AREA]

    # The largest component is the object; it survives whatever the threshold
    # says, so a small object can never be cleaned away entirely.
    largest = int(np.argmax(areas)) + 1

    kept = np.zeros(mask.shape, dtype=bool)

    for index in range(1, count):
        if index == largest or int(stats[index, cv2.CC_STAT_AREA]) >= minimum:
            kept |= components == index

    return kept


def fill_small_holes(
    mask: np.ndarray,
    maximum: int,
    surface: np.ndarray | None = None,
    max_surface_share: float = 0.5,
) -> np.ndarray:
    """
    Close pinholes inside the object, leaving real gaps alone.

    A hole is a background component that does not touch the image border.
    Filling only the small ones keeps the space between a chair's legs open
    while sealing the speckle a segmentation leaves inside a textured door.

    Size is not enough to tell those apart, though, and this step was measured
    doing real damage because of it: on a monstera, the gaps *between the
    leaves* are enclosed by leaf on every side and comfortably under the size
    cap, so they were filled — putting 1,518 pixels of wall back into a mask the
    trim had just cleaned, and leaving a dark slab of wall showing through the
    plant in ALL_OBJECTS.png.

    What separates them is what the hole is made of. A pinhole inside a cabinet
    door is cabinet-coloured; a gap between leaves is wall. So a hole that is
    mostly wall, floor or ceiling is not a hole in the object at all — it is the
    room, seen through it — and stays open however small it is.
    """
    count, components, stats, _ = cv2.connectedComponentsWithStats(
        (~mask).astype(np.uint8), connectivity=4
    )

    if count <= 1:
        return mask

    touching = set(
        np.unique(
            np.concatenate(
                [components[0, :], components[-1, :], components[:, 0], components[:, -1]]
            )
        ).tolist()
    )

    filled = mask.copy()

    for index in range(1, count):
        if index in touching:
            continue

        area = int(stats[index, cv2.CC_STAT_AREA])

        if area > maximum:
            continue

        if surface is not None and surface.any():
            hole = components == index

            if float((hole & surface).sum()) / float(area) > max_surface_share:
                continue

        filled |= components == index

    return filled


def smooth_contours(mask: np.ndarray, config: ExtractionConfig) -> np.ndarray:
    """
    Redraw the mask from its own outer contours.

    This is what removes the one-pixel ragged fringe a thresholded logit map
    leaves behind, without moving the boundary: the contour *is* where the mask
    said its edge was, and filling it back in simply discards detached fringe
    pixels and degenerate stubs too short to be an outline.

    Holes are preserved by re-drawing them, so an object with a genuine opening
    does not become solid.
    """
    contours, hierarchy = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours or hierarchy is None:
        return mask

    redrawn = np.zeros(mask.shape, dtype=np.uint8)

    outer = [
        contour
        for contour, meta in zip(contours, hierarchy[0])
        if meta[3] < 0 and len(contour) >= config.cleanup.min_contour_points
    ]

    inner = [
        contour
        for contour, meta in zip(contours, hierarchy[0])
        if meta[3] >= 0 and len(contour) >= config.cleanup.min_contour_points
    ]

    if not outer:
        return mask

    cv2.drawContours(redrawn, outer, -1, 1, thickness=cv2.FILLED)
    cv2.drawContours(redrawn, inner, -1, 0, thickness=cv2.FILLED)

    return redrawn.astype(bool)


def clean(
    mask: np.ndarray, config: ExtractionConfig, surface: np.ndarray | None = None
) -> tuple[np.ndarray, dict]:
    """
    Run the cleanup chain over one object mask.

    Returns the cleaned mask and a record of what each step cost, in pixels, so
    an over-aggressive setting is visible in the metadata rather than only in
    the output image.
    """
    rules = config.cleanup

    if not mask.any():
        return mask, {"skipped": "empty mask"}

    start = int(mask.sum())

    steps: dict = {"area_in": start}

    # Seal hairline breaks first: a break that survives into component removal
    # turns one object into two, and the smaller half then looks like debris.
    sealed = cv2.morphologyEx(
        mask.astype(np.uint8), cv2.MORPH_CLOSE, _disk(rules.close_radius)
    ).astype(bool)

    steps["after_close"] = int(sealed.sum())

    opened = cv2.morphologyEx(
        sealed.astype(np.uint8), cv2.MORPH_OPEN, _disk(rules.open_radius)
    ).astype(bool)

    # Opening is the one step here that can remove a real feature, so it is
    # kept only while it stays a cleanup rather than a haircut.
    if int(opened.sum()) >= int(sealed.sum()) * 0.97:
        sealed = opened

    steps["after_open"] = int(sealed.sum())

    area = max(1, int(sealed.sum()))

    deduped = drop_small_components(sealed, max(16, int(area * rules.min_component_fraction)))

    steps["after_components"] = int(deduped.sum())

    filled = fill_small_holes(
        deduped, max(16, int(area * rules.max_hole_fraction)), surface
    )

    steps["after_holes"] = int(filled.sum())

    final = smooth_contours(filled, config)

    steps["after_contours"] = int(final.sum())

    steps["area_out"] = int(final.sum())

    steps["net_change_fraction"] = round((steps["area_out"] - start) / max(1, start), 4)

    # A cleanup that removed most of the object is a misconfiguration, not a
    # cleanup. Returning the input is the safe failure.
    if steps["area_out"] < start * 0.5:
        steps["reverted"] = "cleanup removed more than half the mask"

        return mask, steps

    return final, steps
