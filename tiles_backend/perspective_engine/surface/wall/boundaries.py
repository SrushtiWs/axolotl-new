"""
Where a wall really ends: its junctions with the floor, the ceiling and the
next wall -- found without letting the furniture in front of it have a vote.

The problem this exists to solve
--------------------------------
Every earlier attempt derived a wall's edges from the SILHOUETTE of its mask:
the convex hull of it, or the outermost pixel in each row and column. That is
wrong in a specific and very visible way. A sofa standing against a wall hides
the bottom of it, so the mask's lowest pixel in those columns sits on the top
of the SOFA -- and any fit through those points traces the sofa's back, not the
skirting board. Measured on a real living room, the fitted "floor-wall
junction" ran along the top of the sofa cushions for the whole left half of the
image, roughly 250 px above the real junction.

The convex hull does not fix it either. The hull spans a bite taken out of the
middle of an edge, but a sofa that reaches the bottom-left CORNER of the wall
does not leave a bite -- it removes the corner, and the hull happily follows.

What this module does instead
-----------------------------
A junction is sampled COLUMN BY COLUMN (or row by row, for the vertical
corners) and each sample is accepted only if the surface immediately beyond the
wall's last pixel there is the surface the junction is supposed to be with:

    bottom   the pixel below the wall's lowest pixel must be FLOOR
    top      the pixel above the wall's highest pixel must be CEILING
    sides    the pixel beyond the wall's outermost pixel must be another wall,
             or the frame edge

If it is an object instead -- sofa, curtain, cabinet, plant -- that column is
OCCLUDED and contributes nothing. The wall continues behind the object, and
where it continues is a question for the columns that can still see it.

The surviving samples are fitted with RANSAC rather than least squares or
Huber. That matters more than it sounds: the outliers here are not scattered
noise, they are a coherent block -- fifty consecutive columns all sitting on
the top of the same sofa. Huber down-weights points by how far they are from
the current fit, so a large coherent block simply captures the fit and the
strays become the sofa's few opposers. RANSAC asks instead which single line
the largest number of samples agree on, and a real skirting board running the
width of the room always wins that vote.
"""

import math

import numpy as np
import cv2

#: How far beyond the wall's last pixel to look for the surface it meets, as a
#: fraction of the image diagonal. Big enough to cross a skirting board or a
#: cornice, small enough that "beyond" still means "adjacent".
JUNCTION_REACH_FRACTION = 0.018

#: A junction needs this many accepted samples before a line is fitted to it.
MIN_JUNCTION_SAMPLES = 10

#: RANSAC inlier band, as a fraction of the image diagonal. A real architectural
#: junction is straight to well within this; a line that only holds its
#: agreement by opening the band this wide is not one line.
RANSAC_BAND_FRACTION = 0.008

#: A fitted junction is trusted only if this share of its samples agree with it.
MIN_INLIER_RATIO = 0.55

#: ...and only if its inliers stretch across this share of the wall's own extent
#: along that edge. A line agreed on by a tight cluster in one corner says
#: nothing about where the edge runs at the other end of the wall.
MIN_INLIER_SPAN_FRACTION = 0.30

#: Even a well-supported fit is refused if it points this far from the hull
#: edge it would replace. Not a substitute for the tests above -- a backstop
#: against the pathological case where a wall region wraps a corner and its
#: "junction" is genuinely two lines meeting at an angle.
MAX_HULL_DEVIATION_DEG = 45.0

#: Deterministic sampling. Two runs on the same photograph must produce the
#: same walls; a wall whose boundary moves between renders is unusable.
RANSAC_SEED = 20250825
RANSAC_ITERATIONS = 300


def _reach(shape):
    h, w = shape[:2]
    return max(3, int(round(JUNCTION_REACH_FRACTION * math.hypot(h, w))))


def contact_samples(inst_mask, other_mask, object_mask, side, shape, max_samples=400):
    """
    Points along the wall's junction with `other_mask`, occluded columns dropped.

    `side` is "bottom" (junction with the floor) or "top" (with the ceiling).
    Returns (points Nx2, info).
    """
    h, w = shape[:2]
    reach = _reach(shape)
    cols = np.where(inst_mask.any(axis=0))[0]
    if len(cols) == 0:
        return np.empty((0, 2)), {"samples": 0, "occluded": 0, "no_contact": 0}

    step = max(1, len(cols) // max_samples)
    pts = []
    occluded = 0
    no_contact = 0

    for x in cols[::step]:
        rows = np.where(inst_mask[:, x])[0]
        if len(rows) == 0:
            continue
        y = int(rows.max()) if side == "bottom" else int(rows.min())

        if side == "bottom":
            lo, hi = y + 1, min(h, y + 1 + reach)
        else:
            lo, hi = max(0, y - reach), y
        if lo >= hi:
            continue                       # the wall runs out of frame here

        strip = slice(lo, hi)
        meets_other = bool(other_mask[strip, x].any()) if other_mask is not None else False
        meets_object = bool(object_mask[strip, x].any()) if object_mask is not None else False

        if meets_other:
            pts.append((float(x), float(y)))
        elif meets_object:
            occluded += 1                  # something stands here; the wall
        else:                              # continues behind it
            no_contact += 1

    return (np.asarray(pts, dtype=np.float64).reshape(-1, 2),
            {"samples": len(pts), "occluded": occluded, "no_contact": no_contact})


def side_samples(inst_mask, object_mask, side, shape, max_samples=400):
    """
    Points along the wall's left or right vertical edge, occluded rows dropped.

    The edge of a wall is where it meets the NEXT wall, or where it leaves the
    frame. A row whose outermost wall pixel merely abuts a curtain or a wardrobe
    is not showing the edge of the wall, it is showing the edge of the curtain,
    and it is excluded.
    """
    h, w = shape[:2]
    reach = _reach(shape)
    rows = np.where(inst_mask.any(axis=1))[0]
    if len(rows) == 0:
        return np.empty((0, 2)), {"samples": 0, "occluded": 0}

    step = max(1, len(rows) // max_samples)
    pts = []
    occluded = 0

    for y in rows[::step]:
        cols = np.where(inst_mask[y])[0]
        if len(cols) == 0:
            continue
        x = int(cols.min()) if side == "left" else int(cols.max())

        # Touching the frame edge IS the boundary -- nothing is hiding it.
        at_frame = (x <= 1) if side == "left" else (x >= w - 2)
        if at_frame:
            pts.append((float(x), float(y)))
            continue

        if side == "left":
            lo, hi = max(0, x - reach), x
        else:
            lo, hi = x + 1, min(w, x + 1 + reach)
        if lo >= hi:
            pts.append((float(x), float(y)))
            continue

        if object_mask is not None and bool(object_mask[y, lo:hi].any()):
            occluded += 1
            continue
        pts.append((float(x), float(y)))

    return (np.asarray(pts, dtype=np.float64).reshape(-1, 2),
            {"samples": len(pts), "occluded": occluded})


def _line_from_two(p, q):
    """Normalised (a, b, c) with a*x + b*y + c = 0, or None if degenerate."""
    dx, dy = q[0] - p[0], q[1] - p[1]
    n = math.hypot(dx, dy)
    if n < 1e-6:
        return None
    a, b = -dy / n, dx / n
    return a, b, -(a * p[0] + b * p[1])


def fit_line_ransac(points, shape, iterations=RANSAC_ITERATIONS, seed=RANSAC_SEED):
    """
    The line the largest number of these samples agree on.

    RANSAC, not least squares and not Huber, because the outliers along a
    junction are a coherent block -- a whole sofa back -- and any method that
    weights by residual lets a large enough block become the model. Consensus
    counting does not care how coherent the outliers are, only how many they
    are.

    Returns ((p0, p1), info) or (None, info).
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    n = len(pts)
    info = {"samples": n, "inliers": 0, "ratio": 0.0}
    if n < MIN_JUNCTION_SAMPLES:
        return None, info

    band = max(2.0, RANSAC_BAND_FRACTION * math.hypot(*shape[:2]))
    rng = np.random.default_rng(seed)

    best_inliers = None
    best_count = -1
    for _ in range(iterations):
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
        line = _line_from_two(pts[i], pts[j])
        if line is None:
            continue
        a, b, c = line
        dist = np.abs(a * pts[:, 0] + b * pts[:, 1] + c)
        inliers = dist <= band
        count = int(np.count_nonzero(inliers))
        if count > best_count:
            best_count = count
            best_inliers = inliers

    if best_inliers is None or best_count < MIN_JUNCTION_SAMPLES:
        return None, info

    # Refit through the consensus set: RANSAC picks WHICH points belong to the
    # line, least squares over those points places it accurately.
    keep = pts[best_inliers]
    vx, vy, x0, y0 = cv2.fitLine(
        keep.astype(np.float32).reshape(-1, 1, 2), cv2.DIST_L2, 0, 0.01, 0.01).ravel()

    info.update({
        "inliers": best_count,
        "ratio": best_count / float(n),
        "span_x": float(keep[:, 0].max() - keep[:, 0].min()),
        "span_y": float(keep[:, 1].max() - keep[:, 1].min()),
    })
    return ((float(x0), float(y0)), (float(x0 + vx), float(y0 + vy))), info


#: A backstop only. An edge is judged by WHICH SIDE OF IT THE WALL LIES ON,
#: not by its angle -- see `edge_bounds_region`. These limits exist purely to
#: reject a line so degenerate that the side test is meaningless.
#:
#: The angle test that used to do this job was wrong, and wrong in a way that
#: mattered. A floor-wall junction is horizontal only for a wall square to the
#: camera; a wall running away down one side of the room has a junction that
#: rakes steeply, and on a real photograph one measured 44 degrees off
#: horizontal. Rejecting it as "too diagonal" and substituting a horizontal
#: through the region's lowest row, then intersecting that with the leaning
#: side edge, threw the corner 170 px beyond the wall and drew the quad across
#: the floor. Obliqueness is the normal condition of a wall in a photograph,
#: not an error.
MAX_JUNCTION_LEAN_DEG = 75.0
MAX_CORNER_LEAN_DEG = 75.0

#: An edge must have this share of the wall on the correct side of it.
EDGE_SIDE_AGREEMENT = 0.90

#: ...with this much slack, as a fraction of the diagonal, so pixels sitting a
#: few pixels the wrong side of their own boundary do not count against it.
EDGE_SIDE_TOLERANCE_FRACTION = 0.012


def edge_bounds_region(line, region_mask, side, shape, max_samples=4000):
    """
    Does the wall actually lie on the correct side of this line?

    The question an edge has to answer is not "what angle are you" but "is the
    wall above you, below you, to your left or to your right". That works for
    a wall square to the camera and equally for one seen nearly edge-on, where
    the junctions rake steeply and every fixed angle threshold is wrong.
    """
    ys, xs = np.where(region_mask)
    if len(xs) == 0:
        return True, 1.0
    step = max(1, len(xs) // max_samples)
    xs = xs[::step].astype(np.float64)
    ys = ys[::step].astype(np.float64)

    (x0, y0), (x1, y1) = line
    dx, dy = x1 - x0, y1 - y0
    tol = EDGE_SIDE_TOLERANCE_FRACTION * math.hypot(*shape[:2])

    if side in ("bottom", "top"):
        if abs(dx) < 1e-6:
            return False, 0.0            # a vertical cannot be a floor line
        at = y0 + (xs - x0) * (dy / dx)
        good = (ys <= at + tol) if side == "bottom" else (ys >= at - tol)
    else:
        if abs(dy) < 1e-6:
            return False, 0.0            # a horizontal cannot be a wall corner
        at = x0 + (ys - y0) * (dx / dy)
        good = (xs >= at - tol) if side == "left" else (xs <= at + tol)

    share = float(np.count_nonzero(good)) / len(xs)
    return share >= EDGE_SIDE_AGREEMENT, share


def _lean_from(line, horizontal):
    """Degrees between this line and horizontal (or vertical)."""
    (x0, y0), (x1, y1) = line
    ang = math.degrees(math.atan2(y1 - y0, x1 - x0)) % 180.0
    ref = 0.0 if horizontal else 90.0
    d = abs(ang - ref) % 180.0
    return d if d <= 90.0 else 180.0 - d


def fit_junction(points, shape, extent, horizontal, side=None, region_mask=None):
    """
    A junction line, or None if the samples do not support one.

    `extent` is how far the wall itself runs along this edge (its column span
    for a horizontal junction, its row span for a vertical one). A fit is only
    accepted if its inliers stretch across a real share of that -- a line
    agreed on by a huddle of samples in one corner is not evidence about where
    the edge runs at the other end.

    `side` and `region_mask` add the test that actually matters: the wall has
    to be ON THE RIGHT SIDE of the line. A bottom edge with half the wall below
    it is not a bottom edge, whatever its angle.
    """
    line, info = fit_line_ransac(points, shape)
    if line is None:
        return None, {**info, "reason": "no-consensus"}

    if info["ratio"] < MIN_INLIER_RATIO:
        return None, {**info, "reason": "low-agreement"}

    span = info["span_x"] if horizontal else info["span_y"]
    if extent > 0 and span < MIN_INLIER_SPAN_FRACTION * extent:
        return None, {**info, "reason": "short-span", "extent": float(extent)}

    lean = _lean_from(line, horizontal)
    limit = MAX_JUNCTION_LEAN_DEG if horizontal else MAX_CORNER_LEAN_DEG
    if lean > limit:
        return None, {**info, "reason": f"leans({lean:.0f})"}

    if side is not None and region_mask is not None:
        ok, share = edge_bounds_region(line, region_mask, side, shape)
        info["side_share"] = round(share, 3)
        if not ok:
            return None, {**info, "reason": f"wrong-side({share:.2f})"}

    return line, {**info, "reason": "ok", "lean_deg": float(lean)}


def occluders_of(inst_mask, object_mask, shape, min_enclosure=0.45):
    """
    The objects standing in FRONT of this wall, as opposed to beside it.

    Needed because the wall's geometry must be fitted to where the wall would
    be if the room were empty, and the only evidence for that is which objects
    the wall surrounds. An object whose border is mostly against this wall --
    a picture, a wardrobe, a sofa back -- is in front of it; one that merely
    touches it along one side is next to it, and filling it in would extend the
    wall into a place it does not go.

    Returns a boolean mask of the occluding objects.
    """
    if object_mask is None or not np.any(object_mask):
        return np.zeros(shape[:2], dtype=bool)

    h, w = shape[:2]
    reach = max(2, _reach(shape) // 2)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * reach + 1, 2 * reach + 1))

    num, comp, stats, _ = cv2.connectedComponentsWithStats(
        object_mask.astype(np.uint8), 8)
    out = np.zeros((h, w), dtype=bool)

    for i in range(1, num):
        piece = comp == i
        if stats[i, cv2.CC_STAT_AREA] < 64:
            continue
        ring = cv2.dilate(piece.astype(np.uint8), kernel).astype(bool) & ~piece
        ring_px = int(np.count_nonzero(ring))
        if ring_px == 0:
            continue
        against = int(np.count_nonzero(ring & inst_mask))
        if against / ring_px >= min_enclosure:
            out |= piece
    return out


def amodal_region(inst_mask, object_mask, shape):
    """
    The wall as it would be with the furniture taken away.

    Used ONLY to fit geometry to. What gets rendered is still the visible wall
    with the objects cut out of it -- an object is an occlusion, not a hole in
    the plane, and the distinction is the whole reason a wall behind a sofa
    still gets tiled down to the skirting.
    """
    return inst_mask | occluders_of(inst_mask, object_mask, shape)


# ======================================================================
# Objects the segmenter did not name
# ======================================================================
#
# The object mask comes from ADE20K's 150 classes, and its failures are not
# random: they are whole things confidently called "wall". A bookshelf built
# into a wall, a curtain hanging flat against one, a fireplace recess, a
# panelled TV unit -- all read as wall to a classifier, and all get tiled over.
# Adding classes cannot fix this; the model has already decided.
#
# But a wall has a PLANE, fitted from the room's own depth, and these things do
# not lie on it. A bookshelf stands 300 mm proud of the wall behind it; a
# recess sits back from it; a curtain hangs in front of it. So once a wall's
# plane is known, every pixel inside it can be asked a question the segmenter
# cannot answer: are you actually on this surface?
#
# This is the same region-level discipline the rest of the pipeline uses.
# Whole regions are removed, never individual pixels: the depth map is smooth,
# so a pixel-wise test eats the boundary everywhere at once and leaves the
# outline ragged. And a region has to survive an erosion to count as an object,
# which is what stops a one-pixel depth smear along a corner being read as a
# wardrobe.

#: How far off its own plane a pixel may sit and still be that wall, as a
#: fraction of the region's median depth. The depth map's units are arbitrary
#: -- Z = 1000 / (d * contrast + 0.05) ran from 952 to 20000 on a real room --
#: so the tolerance has to be relative or it means nothing.
#:
#: Deliberately looser than the RANSAC inlier band the plane was fitted with
#: (0.06). This test REMOVES wall, so it has to be sure: at 0.06 it shaved the
#: far end of every obliquely-seen wall, where reconstruction error grows with
#: distance. Anything a person would call an object stands far further off the
#: wall than this.
OFF_PLANE_TOLERANCE = 0.14

#: A region must be at least this share of the wall to be an object rather
#: than depth noise.
MIN_OFF_PLANE_SHARE = 0.012


def off_plane_regions(inst_mask, plane, depth_val, cx, cy, focal, depth_contrast,
                      tolerance=OFF_PLANE_TOLERANCE,
                      min_share=MIN_OFF_PLANE_SHARE):
    """
    The parts of this wall region that are not on this wall's plane.

    Returns (mask, info). The mask is what to cut out: things standing in front
    of the wall or set back into it, whatever the segmenter called them.
    """
    info = {"regions": 0, "pixels": 0}
    if plane is None or not np.any(inst_mask):
        return np.zeros(inst_mask.shape, dtype=bool), {**info, "reason": "no-plane"}

    ys, xs = np.where(inst_mask)
    d = depth_val[ys, xs]
    Z = 1000.0 / (d * depth_contrast + 0.05)
    X = ((xs.astype(np.float32) - cx) * Z) / focal
    Y = ((ys.astype(np.float32) - cy) * Z) / focal

    a, b, c, dd = plane
    dist = np.abs(a * X + b * Y + c * Z + dd)
    scale = float(np.median(np.abs(Z))) if len(Z) else 1.0
    band = tolerance * max(scale, 1e-6)

    off = np.zeros(inst_mask.shape, dtype=bool)
    off[ys, xs] = dist > band
    info["band"] = float(band)
    info["depth_scale"] = scale
    info["off_pixels_raw"] = int(np.count_nonzero(off))

    total = int(np.count_nonzero(inst_mask))
    min_area = max(96, int(min_share * total))
    thin = max(2, int(round(0.004 * math.hypot(*inst_mask.shape[:2]))))
    erode_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * thin + 1, 2 * thin + 1))

    num, comp, stats, _ = cv2.connectedComponentsWithStats(off.astype(np.uint8), 8)
    out = np.zeros(inst_mask.shape, dtype=bool)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        piece = comp == i
        # An object is SOLID. A strip a couple of pixels wide along a corner is
        # depth blur, and removing it erodes exactly the outline the rest of
        # this pipeline exists to keep crisp.
        if not np.any(cv2.erode(piece.astype(np.uint8), erode_kernel)):
            continue
        out |= piece
        info["regions"] += 1

    info["pixels"] = int(np.count_nonzero(out))
    info["share"] = round(info["pixels"] / max(total, 1), 4)

    # A wall that comes back mostly "off its own plane" has a bad plane, not a
    # room full of furniture. Trust the plane less than the wall.
    if info["pixels"] > 0.5 * total:
        return np.zeros(inst_mask.shape, dtype=bool), {**info, "reason": "plane-unreliable"}

    return out, {**info, "reason": "ok"}


def _plane_distance(mask, plane, depth_val, cx, cy, focal, depth_contrast):
    """Median distance from this region's points to a plane, and the region's scale."""
    ys, xs = np.where(mask)
    if len(xs) == 0 or plane is None:
        return float("inf"), 1.0
    d = depth_val[ys, xs]
    Z = 1000.0 / (d * depth_contrast + 0.05)
    X = ((xs.astype(np.float32) - cx) * Z) / focal
    Y = ((ys.astype(np.float32) - cy) * Z) / focal
    a, b, c, dd = plane
    dist = np.abs(a * X + b * Y + c * Z + dd)
    return float(np.median(dist)), float(np.median(np.abs(Z)))


def separate_off_plane(regions, depth_val, cx, cy, focal, depth_contrast,
                       tolerance=OFF_PLANE_TOLERANCE):
    """
    Split what is off a wall's plane into OBJECTS and OTHER WALLS.

    Both look identical to a plane test on one wall at a time, and telling them
    apart matters in both directions. A built-in bookshelf and the near face of
    a pillar are equally "not on this wall"; cut the bookshelf and the wall is
    right, cut the pillar and a whole strip of wall goes untiled -- measured at
    35 596 px on a real room, the exact strip the reclaim step had just
    recovered.

    The question that separates them is simple: does some OTHER detected wall's
    plane explain this region? If one does, the region is that wall and is
    moved there. If none does, it is standing in front of the room's walls, and
    that is what an object is.

    `regions` is a list of {"mask", "plane"}. Returns (masks, object_mask, info).
    """
    masks = [np.array(r["mask"], dtype=bool, copy=True) for r in regions]
    planes = [r.get("plane") for r in regions]
    shape = masks[0].shape if masks else depth_val.shape
    objects = np.zeros(shape[:2], dtype=bool)
    info = []

    for i, mask in enumerate(masks):
        off, oinfo = off_plane_regions(mask, planes[i], depth_val, cx, cy, focal,
                                       depth_contrast, tolerance=tolerance)
        entry = {"wall": i, **oinfo, "moved": 0, "cut": 0}
        if not np.any(off):
            info.append(entry)
            continue

        num, comp, stats, _ = cv2.connectedComponentsWithStats(off.astype(np.uint8), 8)
        for k in range(1, num):
            piece = comp == k
            if not np.any(piece):
                continue

            best_j, best_d = None, None
            for j, other in enumerate(planes):
                if j == i or other is None:
                    continue
                d, scale = _plane_distance(piece, other, depth_val, cx, cy,
                                           focal, depth_contrast)
                if d <= tolerance * max(scale, 1e-6) and (best_d is None or d < best_d):
                    best_j, best_d = j, d

            masks[i] = masks[i] & ~piece
            if best_j is not None:
                masks[best_j] = masks[best_j] | piece
                entry["moved"] += int(np.count_nonzero(piece))
            else:
                objects |= piece
                entry["cut"] += int(np.count_nonzero(piece))
        info.append(entry)

    return masks, objects, info
