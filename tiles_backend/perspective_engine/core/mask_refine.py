"""
Making a mask follow the edges that are actually in the photograph.

Two problems, both about BOUNDARIES rather than about which pixels are which.

1. A semantic segmenter draws a confident but approximate outline. It is right
   about what a region is and a few pixels vague about where it stops, so a
   mask used directly tiles a little way onto the skirting, the window frame,
   the leg of a bed.

2. Combining two masks pixel-by-pixel destroys whatever crisp boundary either
   of them had. The depth-derived mask is necessarily smooth -- it comes from
   a blurred normal field -- so `semantic AND geometric` erodes the semantic
   outline by the blur radius and leaves it ragged. That is visible as a wall
   whose tiling stops short of the corner and as a halo around every object.

So: combine at REGION level, then snap the result to the image's own edges.
Neither step moves a boundary far; both stop it being invented.
"""

import numpy as np
import cv2


def _box(img, r):
    return cv2.boxFilter(img, -1, (2 * r + 1, 2 * r + 1), normalize=True,
                         borderType=cv2.BORDER_REFLECT)


def guided_filter(guide_gray, src, radius, eps):
    """
    He et al. guided filter, single-channel guide.

    Implemented here rather than taken from cv2.ximgproc because that lives in
    opencv-contrib, which this environment does not have. It is six box filters
    and some arithmetic, and it is the standard way to make one image's edges
    govern another's -- here, the room photograph's edges governing the mask's.
    """
    I = guide_gray.astype(np.float32)
    p = src.astype(np.float32)

    mean_I = _box(I, radius)
    mean_p = _box(p, radius)
    corr_I = _box(I * I, radius)
    corr_Ip = _box(I * p, radius)

    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    return _box(a, radius) * I + _box(b, radius)


def snap_to_edges(mask_bool, room_bgr, radius_fraction=0.012, eps=1e-4,
                  threshold=0.5):
    """
    Pull a mask's boundary onto the nearest real edge in the photograph.

    The mask goes in as a 0/1 field, is smoothed by the guided filter with the
    room image as the guide -- so it diffuses freely across flat paint and
    barely at all across a skirting board or a window frame -- and is
    thresholded back. A boundary sitting in the middle of flat wall gets pulled
    to whichever edge is nearest; a boundary already on an edge does not move.
    """
    if not np.any(mask_bool):
        return mask_bool

    h, w = mask_bool.shape[:2]
    guide = cv2.cvtColor(room_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    radius = max(2, int(round(radius_fraction * np.hypot(h, w))))

    refined = guided_filter(guide, mask_bool.astype(np.float32), radius, eps)
    return refined >= threshold


def veto_by_region(mask_bool, veto_bool, min_area_fraction=0.004):
    """
    Remove whole OBJECTS a second signal disagrees about, not stray pixels.

    `mask_bool AND NOT veto_bool` is the obvious combination and the wrong one:
    the veto comes from a smoothed field, so intersecting pixel-by-pixel eats
    the mask's boundary everywhere at once and speckles the interior. What the
    veto is actually good for is spotting a whole thing that does not belong --
    a cupboard face in the middle of a wall, a rug in the middle of a floor.

    So the disagreement is grouped into connected regions and only regions
    that are BOTH big enough to be an object AND solid enough to survive an
    erosion are cut out. Everything thinner is boundary noise and is left
    alone, which is what keeps the outline crisp.
    """
    if not np.any(mask_bool) or veto_bool is None or not np.any(veto_bool):
        return mask_bool, {"removed_regions": 0, "removed_pixels": 0}

    h, w = mask_bool.shape[:2]
    disagreement = (mask_bool & veto_bool).astype(np.uint8)
    min_area = max(64, int(min_area_fraction * h * w))

    num, labels, stats, _ = cv2.connectedComponentsWithStats(disagreement, 8)
    remove = np.zeros_like(mask_bool)
    removed_regions = 0

    # An object is SOLID; a boundary smear is THIN. Erosion is the cleanest
    # test of that: a strip a pixel or two wide disappears under it, a cupboard
    # face does not. Area alone cannot tell them apart -- a one-pixel smear
    # running the full width of a wall has a large area and is still a smear,
    # and removing it erodes exactly the outline this function exists to keep.
    thin_radius = max(2, int(round(0.004 * np.hypot(h, w))))
    erode_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * thin_radius + 1, 2 * thin_radius + 1))

    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        region = labels == i
        if not np.any(cv2.erode(region.astype(np.uint8), erode_kernel)):
            continue                      # thin: a boundary smear, not an object
        remove |= region
        removed_regions += 1

    out = mask_bool & ~remove
    return out, {
        "removed_regions": removed_regions,
        "removed_pixels": int(np.count_nonzero(remove)),
    }


def largest_regions(mask_bool, min_area_fraction=0.002):
    """Drop specks. Keeps every region big enough to be a real surface."""
    if not np.any(mask_bool):
        return mask_bool
    h, w = mask_bool.shape[:2]
    min_area = max(64, int(min_area_fraction * h * w))
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_bool.astype(np.uint8), 8)
    keep = np.zeros(num, dtype=bool)
    for i in range(1, num):
        keep[i] = stats[i, cv2.CC_STAT_AREA] >= min_area
    return keep[labels]


def fill_interior_holes(mask_bool, max_hole_fraction=0.0015, protect=None):
    """
    Fill holes that are too small to be an object.

    A wall mask peppered with pinholes from segmentation noise renders as
    tiling full of dots. Only SMALL holes are filled -- a window or a cupboard
    is a large hole and must stay open, which is the whole reason there is a
    size test rather than a plain flood fill.

    `protect` is a second guard for the case size alone gets wrong: a light
    switch or a small picture is a real object and a small hole, so it would
    pass the size test and be tiled over. Any hole overlapping `protect` is
    left open regardless of how small it is.
    """
    if not np.any(mask_bool):
        return mask_bool

    h, w = mask_bool.shape[:2]
    max_hole = max(16, int(max_hole_fraction * h * w))

    holes = (~mask_bool).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(holes, 8)

    out = mask_bool.copy()
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] > max_hole:
            continue
        # A hole touching the frame edge is not enclosed; it is outside.
        if (stats[i, cv2.CC_STAT_LEFT] == 0 or stats[i, cv2.CC_STAT_TOP] == 0 or
                stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] >= w or
                stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] >= h):
            continue
        hole = labels == i
        if protect is not None and np.any(hole & protect):
            continue                      # a real object, however small
        out |= hole
    return out
