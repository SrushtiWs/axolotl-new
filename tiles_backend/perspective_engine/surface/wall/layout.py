"""
Room layout: wall corners first, wall surfaces from them.

The difference from everything else in this package
---------------------------------------------------
Elsewhere a wall is whatever set of pixels survives classification, so its
outline is organic -- it wanders around furniture, frays where depth is soft,
and stops wherever the segmenter lost confidence. A real wall is not like
that. A wall is a FLAT QUADRILATERAL bounded by four straight lines: the
floor-wall junction below, the ceiling-wall junction above, and a vertical
corner at each side where it meets the next wall or leaves the frame.

So this module works the other way round. Find those bounding lines, intersect
them to get corner points, and the wall is the quad between them. Two things
follow that pixel classification cannot give:

  * the edges are STRAIGHT, because they are lines by construction;
  * the wall is COMPLETE, because a sofa standing in front of it no longer
    puts a bite in its outline -- the quad spans corner to corner and the
    objects are removed afterwards as holes.

Order matters: the quad establishes the wall's full extent, then objects are
subtracted from it. Doing it the other way -- fitting a quad to a mask that
already has furniture bitten out of it -- fits the furniture, not the wall.
"""

import math

import numpy as np
import cv2

#: An edge snaps to a detected line only if their directions agree within this.
MAX_SNAP_ANGLE_DEG = 12.0

#: ...and the line lies within this fraction of the diagonal of the edge.
MAX_SNAP_DISTANCE_FRACTION = 0.03

#: A detected line must be at least this fraction of the diagonal to be
#: architecture rather than texture.
MIN_LINE_FRACTION = 0.10


def _order_quad(pts):
    """Corners as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    centre = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - centre[1], pts[:, 0] - centre[0])
    pts = pts[np.argsort(ang)]
    # Rotate so the first point is the one nearest the top-left.
    start = int(np.argmin(pts.sum(axis=1)))
    return np.roll(pts, -start, axis=0)


def quad_from_mask(mask_bool):
    """
    The four corners of the wall this mask belongs to.

    Taken from the CONVEX HULL, not the raw outline. The hull is what spans
    furniture: a wall with a wardrobe against it has a bite out of its mask,
    and fitting a polygon to that outline traces the wardrobe. The hull ignores
    the bite and keeps the wall's real extent, which is exactly the difference
    between segmenting what is visible and identifying the surface.

    Returns a (4, 2) float array, or None if the region cannot support a quad.
    """
    m = mask_bool.astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    biggest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(biggest) < 16:
        return None

    hull = cv2.convexHull(biggest)
    peri = cv2.arcLength(hull, True)

    # Loosen the tolerance until the hull reduces to four vertices. Starting
    # tight and growing finds the LEAST simplification that still gives a quad.
    for frac in np.linspace(0.005, 0.12, 40):
        approx = cv2.approxPolyDP(hull, frac * peri, True)
        if len(approx) == 4:
            return _order_quad(approx)
        if len(approx) < 4:
            break

    # Never reduced cleanly to four: fall back to the minimum-area rectangle,
    # which is always a quad and never worse than the raw outline.
    return _order_quad(cv2.boxPoints(cv2.minAreaRect(hull)))


def _line_angle(p, q):
    return math.degrees(math.atan2(q[1] - p[1], q[0] - p[0])) % 180.0


def _angle_gap(a, b):
    d = abs(a - b) % 180.0
    return d if d <= 90.0 else 180.0 - d


def _point_line_distance(pt, a, b):
    ax, ay = a
    bx, by = b
    px, py = pt
    denom = math.hypot(bx - ax, by - ay)
    if denom < 1e-9:
        return float("inf")
    return abs((bx - ax) * (ay - py) - (ax - px) * (by - ay)) / denom


def _intersect(p1, p2, p3, p4):
    """Intersection of the infinite lines p1p2 and p3p4, or None if parallel."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-9:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
    return (px, py)


def architectural_lines(room_bgr, min_fraction=MIN_LINE_FRACTION):
    """
    Long straight lines in the photograph -- skirtings, cornices, corners.

    The same LSD detector the vanishing-point code uses, filtered to segments
    long enough to be architecture. Short segments are texture, grout and
    picture frames, and snapping a wall's edge to one of those is worse than
    not snapping at all.
    """
    from ...camera.vp_from_mask import detect_lines_lsd

    gray = cv2.cvtColor(room_bgr, cv2.COLOR_BGR2GRAY)
    lines = detect_lines_lsd(gray)
    if lines is None or len(lines) == 0:
        return np.empty((0, 4), np.float32)

    h, w = room_bgr.shape[:2]
    min_len = min_fraction * float(np.hypot(h, w))
    keep = [ln for ln in lines
            if math.hypot(ln[2] - ln[0], ln[3] - ln[1]) >= min_len]
    return np.array(keep, dtype=np.float32).reshape(-1, 4)


def snap_quad_to_lines(quad, lines, image_shape):
    """
    Replace each edge of the quad with the real architectural line it sits on.

    An edge derived from a mask is straight but only approximately placed --
    it follows where the segmenter thought the wall stopped. The skirting
    board and the corner are actual lines in the photograph, so each edge is
    matched to the most collinear long line and adopts its direction; the
    corners are then re-derived by intersecting consecutive edges, which is
    what makes them meet exactly instead of nearly.
    """
    if quad is None or lines is None or len(lines) == 0:
        return quad, {"snapped_edges": 0}

    h, w = image_shape[:2]
    max_dist = MAX_SNAP_DISTANCE_FRACTION * float(np.hypot(h, w))

    edges = []
    snapped = 0
    for i in range(4):
        a = tuple(quad[i])
        b = tuple(quad[(i + 1) % 4])
        edge_angle = _line_angle(a, b)
        mid = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)

        best = None
        best_score = None
        for x1, y1, x2, y2 in lines:
            la = _line_angle((x1, y1), (x2, y2))
            gap = _angle_gap(edge_angle, la)
            if gap > MAX_SNAP_ANGLE_DEG:
                continue
            dist = _point_line_distance(mid, (x1, y1), (x2, y2))
            if dist > max_dist:
                continue
            score = gap / MAX_SNAP_ANGLE_DEG + dist / max_dist
            if best_score is None or score < best_score:
                best_score = score
                best = ((x1, y1), (x2, y2))

        if best is not None:
            edges.append(best)
            snapped += 1
        else:
            edges.append((a, b))

    # Corners are where consecutive edges meet.
    corners = []
    for i in range(4):
        p = _intersect(edges[i - 1][0], edges[i - 1][1], edges[i][0], edges[i][1])
        if p is None or not all(np.isfinite(p)):
            corners.append(tuple(quad[i]))
        else:
            # Refuse an intersection that has run off to infinity; a nearly
            # parallel pair produces a "corner" thousands of pixels outside
            # the frame, which would drag the whole wall with it.
            if abs(p[0]) > 4 * w or abs(p[1]) > 4 * h:
                corners.append(tuple(quad[i]))
            else:
                corners.append(p)

    return _order_quad(corners), {"snapped_edges": snapped}


def quad_mask(quad, shape):
    """Filled quadrilateral as a boolean mask."""
    h, w = shape[:2]
    out = np.zeros((h, w), dtype=np.uint8)
    if quad is None:
        return out.astype(bool)
    cv2.fillPoly(out, [np.round(np.asarray(quad)).astype(np.int32)], 1)
    return out.astype(bool)


def corner_points(quads, image_shape, dedupe_fraction=0.02):
    """
    Every distinct corner in the room layout.

    Adjacent walls share a corner, so the same point arrives twice from two
    quads; near-duplicates are merged so the result is the room's corner set
    rather than a per-wall list.
    """
    h, w = image_shape[:2]
    tol = dedupe_fraction * float(np.hypot(h, w))
    pts = []
    for q in quads:
        if q is None:
            continue
        for p in np.asarray(q):
            if all(math.hypot(p[0] - e[0], p[1] - e[1]) > tol for e in pts):
                pts.append((float(p[0]), float(p[1])))
    return pts


#: How far outside its own mask a layout line may still be drawn, as a
#: fraction of the diagonal -- enough to cover the boundary itself.
LAYOUT_CLIP_SLACK_FRACTION = 0.006


def draw_layout(room_bgr, quads, colour=(0, 0, 0), thickness=None,
                draw_corners=True, masks=None):
    """
    The room's wall layout drawn over the photograph.

    Black by default: on a room that is mostly pale wall and floor, black is
    the only colour that stays legible everywhere without competing with the
    tiles underneath it.

    `masks` clips each quad's edges to the surface that quad describes. This
    matters because the quad and the surface are not the same thing: the quad
    is four infinite lines intersected, and where two of them meet at a shallow
    angle the corner lands past the end of the wall. The MASK is already cut
    against the floor and the ceiling, so nothing wrong is ever tiled -- but an
    unclipped overlay draws a line running on across the floor, and that line
    is what anyone looking at the picture judges the detection by. Drawn
    clipped, each edge is still perfectly straight; it just stops where its
    wall stops.
    """
    out = room_bgr.copy()
    h, w = out.shape[:2]
    t = thickness or max(2, int(round(0.0022 * np.hypot(h, w))))

    allowed = None
    if masks is not None:
        union = np.zeros((h, w), dtype=bool)
        for m in masks:
            if m is not None:
                union |= m
        if np.any(union):
            slack = max(2, int(round(LAYOUT_CLIP_SLACK_FRACTION * np.hypot(h, w))))
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * slack + 1, 2 * slack + 1))
            allowed = cv2.dilate(union.astype(np.uint8), kernel).astype(bool)

    def draw_edge(a, b):
        if allowed is None:
            cv2.line(out, (int(round(a[0])), int(round(a[1]))),
                     (int(round(b[0])), int(round(b[1]))), colour, t, cv2.LINE_AA)
            return
        # Walk the edge and draw only the stretches that lie on a wall.
        n = max(2, int(math.hypot(b[0] - a[0], b[1] - a[1])))
        xs = np.linspace(a[0], b[0], n)
        ys = np.linspace(a[1], b[1], n)
        inside = np.zeros(n, dtype=bool)
        ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        inside[ok] = allowed[ys[ok].astype(np.int32), xs[ok].astype(np.int32)]
        i = 0
        while i < n:
            if not inside[i]:
                i += 1
                continue
            j = i
            while j + 1 < n and inside[j + 1]:
                j += 1
            cv2.line(out, (int(round(xs[i])), int(round(ys[i]))),
                     (int(round(xs[j])), int(round(ys[j]))), colour, t, cv2.LINE_AA)
            i = j + 1

    for q in quads:
        if q is None:
            continue
        pts = np.asarray(q, dtype=np.float64).reshape(-1, 2)
        for i in range(len(pts)):
            draw_edge(pts[i], pts[(i + 1) % len(pts)])

    if draw_corners:
        r = max(3, int(round(0.004 * np.hypot(h, w))))
        for p in corner_points(quads, out.shape):
            x, y = int(round(p[0])), int(round(p[1]))
            if allowed is not None:
                if not (0 <= x < w and 0 <= y < h and allowed[y, x]):
                    continue          # a corner past the end of its own wall
            cv2.circle(out, (x, y), r, colour, -1, cv2.LINE_AA)

    return out


def wall_surfaces(instances, room_bgr, exclude=None, snap=True):
    """
    Turn each detected wall region into a clean quadrilateral surface.

    Returns a list of dicts: quad, mask, and how the quad was arrived at.

    `exclude` is everything that is not wall -- floor, ceiling and objects. It
    is subtracted AFTER the quad is built, never before, so the quad describes
    the wall plane and the subtraction only punches holes in it.
    """
    lines = architectural_lines(room_bgr) if snap else np.empty((0, 4), np.float32)
    out = []

    for inst in instances:
        quad = quad_from_mask(inst.mask)
        if quad is None:
            continue

        snap_info = {"snapped_edges": 0}
        if snap:
            quad, snap_info = snap_quad_to_lines(quad, lines, room_bgr.shape)

        mask = quad_mask(quad, room_bgr.shape)
        if exclude is not None:
            mask = mask & ~exclude

        out.append({
            "index": inst.index,
            "quad": [[float(x), float(y)] for x, y in np.asarray(quad)],
            "mask": mask,
            "snap": snap_info,
            "quad_pixels": int(np.count_nonzero(mask)),
            "region_pixels": int(inst.pixel_count),
        })

    return out, {"architectural_lines": int(len(lines))}


# ======================================================================
# Quads built from the room's ACTUAL junctions
# ======================================================================
#
# quad_from_mask above takes the convex hull and simplifies it. That spans
# furniture correctly, but the edges it produces are only as good as the mask's
# silhouette: the bottom edge lands wherever the wall region happened to stop,
# which on a room with a skirting board or a bit of segmentation bleed is
# several pixels off the real floor line, and on a wall whose mask is clipped
# by an object it can be badly off.
#
# A wall's edges are not properties of its silhouette. They are junctions with
# other surfaces:
#
#     bottom  where this wall meets the FLOOR
#     top     where this wall meets the CEILING
#     sides   where this wall meets the NEXT WALL, or leaves the frame
#
# So each edge is fitted to the points along that junction, and the corners are
# the intersections of the fitted lines. That is what makes an edge sit exactly
# on the skirting instead of near it.

# Sampling and fitting those junctions lives in boundaries.py, which is where
# the reason they can be trusted lives too: a junction is sampled only in the
# columns where it is actually VISIBLE, so the sofa standing in front of the
# wall contributes nothing to where the skirting is.

def quad_from_junctions(inst_mask, floor_mask, ceiling_mask, shape,
                        lines=None, fallback=True, object_mask=None):
    """
    A wall quad whose edges ARE the room's junctions.

    Each of the four edges is fitted to the pixels along the junction it
    represents, optionally snapped to a long architectural line, and the
    corners are the intersections of consecutive edges. An edge that has no
    junction to fit -- a wall running out of frame at the top, say -- falls
    back to the hull quad's corresponding edge so the quad is always complete.

    `object_mask` is what makes the fit trustworthy. Without it, the junction
    is sampled from the wall's silhouette, and a sofa standing against the wall
    puts the sample on top of the sofa; the fitted "floor line" then runs along
    the cushions. With it, occluded columns are recognised and dropped, and the
    line is fitted only where the junction is actually visible.

    Returns (quad, info) or (None, info).
    """
    from .boundaries import (
        MAX_HULL_DEVIATION_DEG, amodal_region, contact_samples,
        edge_bounds_region, fit_junction, side_samples,
    )

    info = {"source": "junctions"}

    # Geometry is fitted to the wall AS IF EMPTY; only the render uses the
    # visible mask. A wall behind a sofa is still a wall down to the skirting.
    fit_mask = amodal_region(inst_mask, object_mask, shape)
    hull_quad = quad_from_mask(fit_mask) if fallback else None

    cols = np.where(fit_mask.any(axis=0))[0]
    rows = np.where(fit_mask.any(axis=1))[0]
    if len(cols) == 0 or len(rows) == 0:
        return None, {**info, "stage": "empty"}
    col_extent = float(cols.max() - cols.min())
    row_extent = float(rows.max() - rows.min())

    fitted = {}
    detail = {}

    def junction(side, other, horizontal, extent):
        pts, sinfo = contact_samples(fit_mask, other, object_mask, side, shape)
        line, finfo = fit_junction(pts, shape, extent, horizontal,
                                   side=side, region_mask=fit_mask)
        detail[side] = {**sinfo, **finfo}
        return line

    def edge(side, horizontal, extent):
        pts, sinfo = side_samples(fit_mask, object_mask, side, shape)
        line, finfo = fit_junction(pts, shape, extent, horizontal,
                                   side=side, region_mask=fit_mask)
        detail[side] = {**sinfo, **finfo}
        return line

    top_fit = junction("top", ceiling_mask, True, col_extent)
    bottom_fit = junction("bottom", floor_mask, True, col_extent)
    left_fit = edge("left", False, row_extent)
    right_fit = edge("right", False, row_extent)

    def hull_edge(i):
        if hull_quad is None:
            return None
        return (tuple(hull_quad[i]), tuple(hull_quad[(i + 1) % 4]))

    # Hull corners are ordered top-left, top-right, bottom-right, bottom-left,
    # so its edges are top, right, bottom, left in that order.
    #
    # The hull edge is the fallback, not the arbiter. A junction fit that has
    # survived the agreement and span tests in `fit_junction` is BETTER evidence
    # than the hull -- the hull is exactly the thing that traces the sofa. The
    # angle test that remains is only a backstop against a wall region that
    # wraps a corner, whose junction is genuinely two lines meeting at an angle
    # and cannot be one.
    def sane_hull_edge(hull_i, name):
        """
        The hull's edge, unless the wall is on the wrong side of it.

        Judged by side rather than by angle. A hull edge is a chord between two
        extreme points of the region, and on an L-shaped region that chord can
        cut through the wall itself; but a steep chord is not by itself wrong.
        A wall running away down one side of the room has a floor junction that
        rakes -- one measured 44 degrees off horizontal -- and calling that "too
        diagonal" and substituting a horizontal threw the quad across the floor.
        Obliqueness is normal. Having the wall on both sides of your bottom edge
        is not.
        """
        h_edge = hull_edge(hull_i)
        if h_edge is not None:
            ok, _share = edge_bounds_region(h_edge, fit_mask, name, shape)
            if ok:
                return h_edge

        detail.setdefault(name, {})["hull"] = "cuts-the-wall-used-axis"
        if name == "top":
            y = float(rows.min())
            return ((0.0, y), (1.0, y))
        if name == "bottom":
            y = float(rows.max())
            return ((0.0, y), (1.0, y))
        x = float(cols.min()) if name == "left" else float(cols.max())
        return ((x, 0.0), (x, 1.0))

    def prefer(fit, hull_i, name):
        h_edge = sane_hull_edge(hull_i, name)
        if fit is None:
            fitted[name] = False
            return h_edge
        if h_edge is None:
            fitted[name] = True
            return fit
        gap = _angle_gap(_line_angle(*fit), _line_angle(*h_edge))
        if gap > MAX_HULL_DEVIATION_DEG:
            fitted[name] = False
            detail.setdefault(name, {})["reason"] = "hull-deviation"
            return h_edge
        fitted[name] = True
        return fit

    top = prefer(top_fit, 0, "top")
    right_e = prefer(right_fit, 1, "right")
    bottom = prefer(bottom_fit, 2, "bottom")
    left_e = prefer(left_fit, 3, "left")

    info["fitted"] = fitted
    info["edges"] = detail

    if any(e is None for e in (top, right_e, bottom, left_e)):
        return (hull_quad, {**info, "stage": "fell-back-to-hull"})

    ordered = [top, right_e, bottom, left_e]
    if lines is not None and len(lines):
        snapped = []
        n_snapped = 0
        for a, b in ordered:
            best = _best_matching_line((a, b), lines, shape)
            if best is not None:
                snapped.append(best)
                n_snapped += 1
            else:
                snapped.append((a, b))
        ordered = snapped
        info["snapped_edges"] = n_snapped

    # Corners are where consecutive edges meet -- but an intersection of two
    # INFINITE lines is not bounded by the wall that produced them. Two edges
    # meeting at a shallow angle put their crossing point far outside it: on a
    # real room a left wall's leaning side edge crossed its bottom edge 170 px
    # past the wall, and the quad was drawn across the floor. A corner that has
    # wandered that far from the hull corner it stands in for is not a better
    # estimate of it, so the hull corner is kept.
    corners = []
    h, w = shape[:2]
    drift_limit = MAX_CORNER_DRIFT_FRACTION * math.hypot(h, w)
    drifted = 0
    for i in range(4):
        p = _intersect(ordered[i - 1][0], ordered[i - 1][1],
                       ordered[i][0], ordered[i][1])
        bad = p is None or not all(np.isfinite(p))
        if not bad and hull_quad is not None:
            hx, hy = hull_quad[i]
            if math.hypot(p[0] - hx, p[1] - hy) > drift_limit:
                bad = True
                drifted += 1
        if not bad and (abs(p[0]) > 4 * w or abs(p[1]) > 4 * h):
            bad = True
        if bad:
            if hull_quad is None:
                return None, {**info, "stage": "degenerate-corner"}
            corners.append(tuple(hull_quad[i]))
        else:
            corners.append(p)
    if drifted:
        info["drifted_corners"] = drifted

    quad = _order_quad(corners)

    # A quad built from four independently fitted lines can still come out
    # nonsense -- two near-parallel edges push their corner far away and the
    # shape shears into a sliver or a wedge many times the size of the wall.
    # Checked against the region it came from, which is the only thing that
    # knows how big this wall actually is.
    ok, why = _quad_is_plausible(quad, fit_mask, shape)
    if not ok:
        info["rejected"] = why
        if hull_quad is None:
            return None, {**info, "stage": "implausible"}
        return hull_quad, {**info, "stage": "implausible-used-hull"}

    info["stage"] = "ok"
    return quad, info


#: How far an intersected corner may sit from the hull corner it replaces, as a
#: fraction of the image diagonal. Generous -- spanning the furniture in front
#: of a wall legitimately moves a corner a long way -- but finite, which is the
#: difference between a corner on the far side of the wall and one on the far
#: side of the room.
MAX_CORNER_DRIFT_FRACTION = 0.18


#: A quad may not be smaller than this share of the region it was fitted to,
#: nor larger than this multiple of it. Both bounds are loose: a wall's quad
#: legitimately exceeds its visible mask (that is the point -- it spans the
#: furniture), and legitimately falls short of it where the mask has frayed.
#: What they catch is the sheared, collapsed or runaway shape.
MIN_QUAD_AREA_RATIO = 0.55
MAX_QUAD_AREA_RATIO = 4.0


def _quad_is_plausible(quad, region_mask, shape):
    """Is this quadrilateral a believable wall, or a shear artefact?"""
    if quad is None or len(quad) != 4:
        return False, "not-a-quad"

    pts = np.asarray(quad, dtype=np.float64)
    if not np.all(np.isfinite(pts)):
        return False, "non-finite"

    area = abs(cv2.contourArea(pts.astype(np.float32)))
    region_area = float(np.count_nonzero(region_mask))
    if region_area <= 0:
        return False, "empty-region"
    ratio = area / region_area
    if ratio < MIN_QUAD_AREA_RATIO:
        return False, f"too-small({ratio:.2f})"
    if ratio > MAX_QUAD_AREA_RATIO:
        return False, f"too-large({ratio:.2f})"

    if not cv2.isContourConvex(np.round(pts).astype(np.int32)):
        return False, "not-convex"

    # A corner may sit outside the frame -- a wall running off the edge of the
    # photograph has corners beyond it -- but not by more than the frame again.
    h, w = shape[:2]
    if (pts[:, 0].min() < -w or pts[:, 0].max() > 2 * w
            or pts[:, 1].min() < -h or pts[:, 1].max() > 2 * h):
        return False, "corner-far-outside-frame"

    return True, "ok"


def _best_matching_line(edge, lines, image_shape):
    """The long detected line most collinear with this edge, if any."""
    (a, b) = edge
    h, w = image_shape[:2]
    max_dist = MAX_SNAP_DISTANCE_FRACTION * float(np.hypot(h, w))
    edge_angle = _line_angle(a, b)
    mid = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)

    best, best_score = None, None
    for x1, y1, x2, y2 in lines:
        gap = _angle_gap(edge_angle, _line_angle((x1, y1), (x2, y2)))
        if gap > MAX_SNAP_ANGLE_DEG:
            continue
        dist = _point_line_distance(mid, (x1, y1), (x2, y2))
        if dist > max_dist:
            continue
        score = gap / MAX_SNAP_ANGLE_DEG + dist / max_dist
        if best_score is None or score < best_score:
            best_score, best = score, ((x1, y1), (x2, y2))
    return best


def wall_surfaces_from_junctions(instances, room_bgr, floor_mask, ceiling_mask,
                                 exclude=None, snap=True, object_mask=None,
                                 semantic_wall=None):
    """
    wall_surfaces, but with every edge fitted to a real junction.

    Same contract and same ordering rule: the quad establishes the wall's full
    extent first, objects are subtracted from it afterwards.

    Two walls' quads may overlap where their fitted corners disagree by a few
    pixels; the contested strip is given to whichever wall's own region is
    nearer, so no pixel is claimed twice and no pixel is dropped.
    """
    lines = architectural_lines(room_bgr) if snap else np.empty((0, 4), np.float32)
    out = []

    for inst in instances:
        quad, qinfo = quad_from_junctions(
            inst.mask, floor_mask, ceiling_mask, room_bgr.shape,
            lines=lines if snap else None, object_mask=object_mask,
        )
        if quad is None:
            continue

        mask = quad_mask(quad, room_bgr.shape)
        if exclude is not None:
            mask = mask & ~exclude

        # A quad that ends up covering far less than the region it came from is
        # degenerate -- two of its edges came out nearly parallel and the
        # corners collapsed. Better the region itself than a sliver.
        if int(np.count_nonzero(mask)) < 0.35 * max(inst.pixel_count, 1):
            fallback_mask = inst.mask.copy()
            if exclude is not None:
                fallback_mask = fallback_mask & ~exclude
            if np.count_nonzero(fallback_mask) > np.count_nonzero(mask):
                mask = fallback_mask
                qinfo = {**qinfo, "stage": "quad-degenerate-used-region"}

        out.append({
            "index": inst.index,
            "quad": [[float(x), float(y)] for x, y in np.asarray(quad)],
            "mask": mask,
            "region": inst.mask,
            "snap": {"snapped_edges": qinfo.get("snapped_edges", 0)},
            "junctions": qinfo,
            "quad_pixels": int(np.count_nonzero(mask)),
            "region_pixels": int(inst.pixel_count),
        })

    out = _resolve_overlaps(out, room_bgr.shape)
    out, reclaim_info = reclaim_unclaimed_wall(
        out, semantic_wall, exclude if exclude is not None
        else np.zeros(room_bgr.shape[:2], bool), room_bgr.shape)
    for surf in out:
        surf["mask"] = tidy_surface(surf["mask"], room_bgr.shape)
        surf.pop("region", None)
        surf["quad_pixels"] = int(np.count_nonzero(surf["mask"]))

    return out, {"architectural_lines": int(len(lines)), **reclaim_info}


def _resolve_overlaps(surfaces, shape):
    """
    Give every contested pixel to exactly one wall.

    Quads are fitted independently, so two adjacent walls' quads overlap in a
    strip along their shared corner. Left alone that strip gets tiled twice,
    once by each wall's grid, and the seam shows as a doubled band. The pixel
    goes to whichever wall's own detected REGION is nearer to it, which is the
    only evidence available about which side of the corner it is on.
    """
    if len(surfaces) < 2:
        return surfaces

    dists = []
    for surf in surfaces:
        region = surf.get("region")
        if region is None or not np.any(region):
            dists.append(np.full(shape[:2], np.inf, dtype=np.float32))
            continue
        dists.append(cv2.distanceTransform(
            (~region).astype(np.uint8), cv2.DIST_L2, 3).astype(np.float32))

    stack = np.stack(dists, axis=0)
    owner = np.argmin(stack, axis=0)

    overlap = np.zeros(shape[:2], dtype=np.int32)
    for surf in surfaces:
        overlap += surf["mask"].astype(np.int32)
    contested = overlap > 1
    if not np.any(contested):
        return surfaces

    for i, surf in enumerate(surfaces):
        surf["mask"] = surf["mask"] & ~(contested & (owner != i))
    return surfaces


#: A reclaimed strip must touch the wall it is given to -- within this fraction
#: of the diagonal. A detached blob of wall elsewhere in the room belongs to a
#: wall that was not detected, not to the nearest one that was.
RECLAIM_TOUCH_FRACTION = 0.01


def reclaim_unclaimed_wall(surfaces, sem_wall, blocked, shape):
    """
    Give back the wall the quads fell short of.

    Four straight edges cannot follow a wall exactly, and where they fall
    inside it the difference is not cosmetic: it is a strip of ORIGINAL wall
    left visible along the bottom of the render, under a hard straight line
    where the tiling stops. Measured on a real room, 2.4% of the frame -- a
    wedge along the whole floor-wall junction of the back wall, plus the full
    height of a pillar's return face.

    Reclaimed pixels are bounded by the segmenter's own wall outline, so this
    can only ever recover wall the photograph already showed as wall. It never
    expands past it. Where the quad and the visible architecture disagree and
    the quad is the smaller, the architecture wins.

    `blocked` is everything the wall may not take: floor, ceiling, objects.
    """
    if not surfaces or sem_wall is None or not np.any(sem_wall):
        return surfaces, {"reclaimed_pixels": 0, "reclaimed_regions": 0}

    claimed = np.zeros(shape[:2], dtype=bool)
    for surf in surfaces:
        claimed |= surf["mask"]

    spare = sem_wall & ~claimed & ~blocked
    if not np.any(spare):
        return surfaces, {"reclaimed_pixels": 0, "reclaimed_regions": 0}

    h, w = shape[:2]
    touch = max(2, int(round(RECLAIM_TOUCH_FRACTION * np.hypot(h, w))))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * touch + 1, 2 * touch + 1))

    # Nearest wall for every pixel, by distance to that wall's own mask.
    dists = [cv2.distanceTransform((~s["mask"]).astype(np.uint8), cv2.DIST_L2, 3)
             if np.any(s["mask"]) else np.full((h, w), np.inf, np.float32)
             for s in surfaces]
    owner = np.argmin(np.stack(dists, axis=0), axis=0)

    num, comp, stats, _ = cv2.connectedComponentsWithStats(
        spare.astype(np.uint8), 8)
    taken = 0
    regions = 0
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] < 64:
            continue
        piece = comp == i
        if not np.any(cv2.dilate(piece.astype(np.uint8), kernel).astype(bool)
                      & claimed):
            continue                      # detached: an undetected wall, not this one
        # A strip straddling a corner is split between the two walls it touches.
        for idx in np.unique(owner[piece]):
            part = piece & (owner == idx)
            if np.any(part):
                surfaces[int(idx)]["mask"] = surfaces[int(idx)]["mask"] | part
        taken += int(np.count_nonzero(piece))
        regions += 1

    return surfaces, {"reclaimed_pixels": taken, "reclaimed_regions": regions}


#: A fragment smaller than this share of the wall it belongs to is not part of
#: that wall; it is a shard left behind where an object was cut out.
MIN_FRAGMENT_SHARE = 0.02


def tidy_surface(mask_bool, shape, min_fragment_share=MIN_FRAGMENT_SHARE):
    """
    Remove the debris that subtracting objects from a quad leaves behind.

    Cutting a sofa, a curtain and a run of skirting out of a quad does not
    only make holes: it shaves slivers off the edges and strands little islands
    between objects. Each is its own connected component, each renders as a
    stray patch of tile floating in the room, and none of them is wall anyone
    would want tiled. Judged as a share of THIS wall rather than of the frame,
    so a small wall does not lose its whole self and a large one still sheds
    its shards.
    """
    from ...core.mask_refine import fill_interior_holes

    if not np.any(mask_bool):
        return mask_bool

    total = int(np.count_nonzero(mask_bool))
    num, comp, stats, _ = cv2.connectedComponentsWithStats(
        mask_bool.astype(np.uint8), 8)
    limit = max(64, int(min_fragment_share * total))
    keep = np.zeros(num, dtype=bool)
    for i in range(1, num):
        keep[i] = stats[i, cv2.CC_STAT_AREA] >= limit
    out = keep[comp]

    if not np.any(out):                    # every piece was small: keep the best
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) if num > 1 else 0
        out = comp == biggest

    # Pinholes along an object's cut edge are noise, not openings. Real
    # openings -- windows, doorways -- are far above the size limit inside
    # fill_interior_holes and stay open.
    return fill_interior_holes(out)
