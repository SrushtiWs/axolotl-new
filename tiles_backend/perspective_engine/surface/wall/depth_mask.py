"""
Wall detection from the DEPTH MAP, as geometry rather than semantics.

Why this exists alongside the segmenter
---------------------------------------
Mask2Former answers "what class is this pixel?". That is the right question
for a floor -- a chair is a chair, so it is simply not floor -- but it is only
half the question for a wall. A wall is defined as much by its SHAPE as by its
label: a plain painted surface and a radiator bolted to it can both be
labelled "wall" by a semantic model, and no amount of class filtering
separates them, because the difference is that one is flat and in the wall
plane and the other sticks out of it.

Depth answers the other half. Reconstruct the surface normal at every pixel
and a wall becomes obvious geometrically:

    floor / ceiling ->  normal points UP or DOWN   (|n_y| near 1)
    wall            ->  normal points SIDEWAYS     (|n_y| near 0)
    clutter         ->  normal points everywhere   (no consistency)

So this module classifies by verticality of the surface, and the two signals
are combined: the segmenter says where a wall roughly is, the depth map says
which of those pixels are actually the flat wall face.

The catch, stated plainly
-------------------------
The depth map reaching us is a Turbo-coloured PNG whose decode is monotonic
but warped by ~13%, and MiDaS output is relative to begin with. Absolute
angles from it are not trustworthy. What IS trustworthy is the CONTRAST
between "normal points up" and "normal points sideways" -- that survives a
monotonic warp comfortably, because it is a 90-degree distinction, not a
2-degree one. Everything here is built on that distinction and nothing finer.
"""

import numpy as np
import cv2

from ...core.mask_refine import veto_by_region

#: A surface counts as a wall when its normal is within this of horizontal.
#: |n_y| = 0 is perfectly vertical wall, |n_y| = 1 is perfectly flat floor.
#:
#: Measured on real rooms: the floor's median verticality is 0.98-0.99 and a
#: wall's is 0.20-0.38, so the two populations are far apart and the threshold
#: only has to land between them. 0.60 does, with margin either side. An
#: earlier 0.45 was inside the wall population's own spread and punched large
#: holes in perfectly good wall.
MAX_VERTICAL_COMPONENT = 0.60

#: The verticality field is smoothed before thresholding, over this fraction
#: of the image diagonal. Thresholding the raw field turns its streaky noise
#: into ragged holes; smoothing a scalar field first is both cheaper and less
#: destructive than trying to repair the binary mask afterwards with
#: morphology.
VERTICALITY_SMOOTH_FRACTION = 0.02

#: Depth is smoothed before normals are taken. Un-smoothed MiDaS output gives
#: normals that swing wildly pixel to pixel and classify nothing. Expressed as
#: a fraction of the image diagonal so it behaves the same at any resolution.
SMOOTH_FRACTION = 0.012

#: Normals are computed over a window this many pixels wide. Larger than a
#: 3x3 Sobel on purpose: a wall's depth gradient is gentle, and a tight kernel
#: measures noise rather than slope.
NORMAL_WINDOW_FRACTION = 0.02

#: Connected components below this share of the image are dropped.
MIN_COMPONENT_FRACTION = 0.002


def _odd(n: int) -> int:
    n = max(3, int(n))
    return n if n % 2 == 1 else n + 1


def back_project(depth_val, cx, cy, f, depth_contrast=1.0):
    """
    Dense 3D point cloud in camera coordinates, one point per pixel.

    Same inversion the renderer uses, so a wall found here is a wall the
    renderer will agree with.
    """
    h, w = depth_val.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    Z = 1000.0 / (depth_val * depth_contrast + 0.05)
    X = ((xx - cx) * Z) / f
    Y = ((yy - cy) * Z) / f
    return X, Y, Z


def surface_normals(depth_val, cx, cy, f, depth_contrast=1.0,
                    smooth_fraction=SMOOTH_FRACTION,
                    window_fraction=NORMAL_WINDOW_FRACTION):
    """
    Per-pixel surface normal, from the depth map alone.

    The normal is the cross product of the point cloud's two image-space
    derivatives -- the standard construction. Both the pre-smoothing and the
    wide derivative window matter more than the formula: raw MiDaS output
    produces normals that flip between neighbouring pixels, and no threshold
    can rescue a classification made on those.

    Returns (nx, ny, nz), each HxW, unit length where defined.
    """
    h, w = depth_val.shape[:2]
    diag = float(np.hypot(h, w))

    # Edge-preserving smooth, so the corner between two walls stays a corner
    # instead of being rounded into a curve that reads as neither.
    d = depth_val.astype(np.float32)
    sigma_px = max(1.0, smooth_fraction * diag)
    d = cv2.bilateralFilter(d, d=0, sigmaColor=0.08, sigmaSpace=sigma_px)

    X, Y, Z = back_project(d, cx, cy, f, depth_contrast)

    # Normalise the cloud's scale before differentiating.
    #
    # Z runs to ~20000 and X, Y scale with it, and a wide derivative kernel
    # multiplies that by large coefficients -- the first version used
    # cv2.Sobel at ksize 31, whose weights are astronomical, and the cross
    # product overflowed float32 to infinity. Every normal came back as 0/0
    # and verticality was 0.000 everywhere, which classified the entire image
    # as wall including the floor.
    #
    # A normal is invariant under uniform scaling of the point cloud -- scale
    # by s and the cross product scales by s^2, which normalisation divides
    # straight back out -- so dividing through by the median depth costs
    # nothing and keeps every quantity near 1.
    scale = float(np.median(Z))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    X, Y, Z = X / scale, Y / scale, Z / scale

    # Average over the window with a box blur, then take plain central
    # differences. Same effect as a wide derivative kernel, without the
    # coefficients that caused the overflow.
    k = _odd(window_fraction * diag)
    X = cv2.blur(X, (k, k))
    Y = cv2.blur(Y, (k, k))
    Z = cv2.blur(Z, (k, k))

    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    Z = Z.astype(np.float64)

    dXdy, dXdx = np.gradient(X)
    dYdy, dYdx = np.gradient(Y)
    dZdy, dZdx = np.gradient(Z)

    nx = dYdx * dZdy - dZdx * dYdy
    ny = dZdx * dXdy - dXdx * dZdy
    nz = dXdx * dYdy - dYdx * dXdy

    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    norm = np.maximum(norm, 1e-12)
    return nx / norm, ny / norm, nz / norm


def verticality(depth_val, cx, cy, f, depth_contrast=1.0):
    """
    |n_y| per pixel: 0 = a perfectly vertical surface, 1 = perfectly flat.

    This single number is the whole classifier. It is a 90-degree distinction,
    which is why it survives a depth map that is only monotonically correct.
    """
    _nx, ny, _nz = surface_normals(depth_val, cx, cy, f, depth_contrast)
    return np.abs(ny)


def wall_mask_from_depth(depth_val, cx, cy, f, depth_contrast=1.0,
                         max_vertical=MAX_VERTICAL_COMPONENT,
                         min_component_fraction=MIN_COMPONENT_FRACTION):
    """
    Wall mask from depth alone. Returns (mask_bool, info).

    No semantic input at all -- this is the geometry's own opinion, useful both
    as a mask in its own right and as a cross-check on the segmenter.
    """
    h, w = depth_val.shape[:2]
    vert = verticality(depth_val, cx, cy, f, depth_contrast)

    # Smooth the SCALAR field, not the binary mask that comes out of it.
    k = _odd(VERTICALITY_SMOOTH_FRACTION * np.hypot(h, w))
    vert = cv2.blur(vert.astype(np.float32), (k, k))

    mask = vert <= max_vertical

    # Close first, then a light open. Closing fills the pinholes depth noise
    # leaves in a wall face; opening afterwards with a smaller radius removes
    # stragglers without eating back into the surface.
    radius = max(1, int(round(0.006 * np.hypot(h, w))))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, small)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    min_area = max(64, int(min_component_fraction * h * w))
    keep = np.zeros(num, dtype=bool)
    for i in range(1, num):
        keep[i] = stats[i, cv2.CC_STAT_AREA] >= min_area
    mask = keep[labels]

    info = {
        "source": "depth",
        "max_vertical_component": float(max_vertical),
        "coverage": float(mask.mean()),
        "median_verticality_inside": float(np.median(vert[mask])) if np.any(mask) else None,
        "median_verticality_outside": float(np.median(vert[~mask])) if np.any(~mask) else None,
    }
    return mask, info


def classify_room_surfaces(depth_val, cx, cy, f, depth_contrast=1.0,
                           max_vertical=MAX_VERTICAL_COMPONENT):
    """
    Split the WHOLE ROOM into floor, wall and ceiling, from one depth map.

    Two questions, both answered by the reconstructed geometry:

      1. Is this surface flat or upright?   |n_y| near 1 vs near 0
      2. If flat, is it below the camera or above it?   sign of Y

    The camera convention is +y DOWN, so a point below the lens has Y > 0.
    Floor and ceiling are both "flat" and are told apart by nothing more than
    that sign -- which is why this needs the full-room depth map and not a
    per-surface one: a map cropped to the wall has no floor left in it to
    measure, and the horizon that separates the two is exactly what has been
    thrown away.

    Returns (masks, info) where masks has keys "floor", "wall", "ceiling".
    """
    h, w = depth_val.shape[:2]

    vert = verticality(depth_val, cx, cy, f, depth_contrast)
    k = _odd(VERTICALITY_SMOOTH_FRACTION * np.hypot(h, w))
    vert = cv2.blur(vert.astype(np.float32), (k, k))

    _X, Y, _Z = back_project(depth_val, cx, cy, f, depth_contrast)

    flat = vert > max_vertical
    below = Y > 0.0

    raw = {
        "floor": flat & below,
        "ceiling": flat & ~below,
        "wall": ~flat,
    }

    radius = max(1, int(round(0.006 * np.hypot(h, w))))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    min_area = max(64, int(MIN_COMPONENT_FRACTION * h * w))

    masks = {}
    for name, m in raw.items():
        m8 = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        m8 = cv2.morphologyEx(m8, cv2.MORPH_OPEN, small)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(m8, connectivity=8)
        keep = np.zeros(num, dtype=bool)
        for i in range(1, num):
            keep[i] = stats[i, cv2.CC_STAT_AREA] >= min_area
        masks[name] = keep[labels]

    info = {
        "source": "depth-geometry",
        "coverage": {n: float(m.mean()) for n, m in masks.items()},
        "median_verticality": {
            n: (float(np.median(vert[m])) if np.any(m) else None)
            for n, m in masks.items()
        },
    }
    return masks, info


def refine_with_depth(semantic_mask, depth_val, cx, cy, f, depth_contrast=1.0,
                      mode="combined", max_vertical=MAX_VERTICAL_COMPONENT):
    """
    Combine the segmenter's answer with the depth map's. Returns (mask, info).

    mode:
      "semantic"  the segmenter alone -- the previous behaviour, unchanged
      "depth"     the depth map alone
      "combined"  semantic AND geometrically wall-like  (default)

    "combined" is an INTERSECTION, and deliberately so. The two signals fail in
    opposite directions: the segmenter over-includes (it labels a radiator or a
    protruding sill as wall because that is what surrounds it), while depth
    over-includes differently (a tall cupboard face is a vertical plane and
    looks exactly like a wall to geometry). Taking only what both agree on
    keeps each one's mistakes out of the result, at the cost of trimming a
    little genuine wall at the edges -- which is the right trade when the
    alternative is tiling over a radiator.
    """
    if mode == "semantic":
        return semantic_mask, {"source": "semantic", "mode": mode}

    depth_wall, dinfo = wall_mask_from_depth(
        depth_val, cx, cy, f, depth_contrast, max_vertical=max_vertical
    )

    if mode == "depth":
        return depth_wall, {**dinfo, "mode": mode}

    # Region-level veto, NOT a pixel-wise intersection.
    #
    # `semantic AND depth` was the first attempt and it damaged exactly what
    # the caller cares about. The depth mask comes from a blurred normal
    # field, so intersecting pixel-by-pixel erodes the segmenter's crisp
    # outline by the blur radius all the way round, speckles the interior, and
    # leaves a halo at every object edge -- a wall whose tiling stops short of
    # its own corner. What depth is genuinely good for is spotting a whole
    # THING that does not belong: a cupboard face standing in the wall plane,
    # a radiator, a rug on a floor. So only regions big enough to be an object
    # AND consistently disagreed with are cut out; smaller disagreements are
    # boundary noise and are left alone.
    sem_px = int(np.count_nonzero(semantic_mask))
    combined, veto_info = veto_by_region(semantic_mask, ~depth_wall)
    both_px = int(np.count_nonzero(combined))

    info = {
        "source": "semantic+depth",
        "mode": mode,
        "semantic_pixels": sem_px,
        "depth_pixels": int(np.count_nonzero(depth_wall)),
        "combined_pixels": both_px,
        "removed_by_depth": sem_px - both_px,
        "removed_by_depth_share": (sem_px - both_px) / sem_px if sem_px else 0.0,
        "veto": veto_info,
        "depth_info": dinfo,
    }

    # Refusing to act is better than acting on nonsense: if the veto throws
    # away most of the wall, depth disagreed with the segmenter so completely
    # that one of them is wrong and there is no way to tell which.
    if sem_px and both_px < 0.25 * sem_px:
        info["fallback"] = (
            f"depth removed {100 * (1 - both_px / sem_px):.0f}% of the semantic "
            "wall; using the semantic mask alone"
        )
        return semantic_mask, info

    return combined, info
