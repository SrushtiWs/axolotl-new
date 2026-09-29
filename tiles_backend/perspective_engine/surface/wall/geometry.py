"""
Each wall's own direction, from its own lines.

The wall pipeline orients a wall by picking one of the ROOM's horizontal
vanishing points (`constraints.choose_plane`). When the room shows only one
converging direction -- a camera facing the back wall, where the side walls
converge and the back wall's lines run parallel across the frame -- every wall
is handed that one direction, the back wall included, and a wall that faces
the camera is projected as if it ran away from it. Measured on an atrium: the
back wall reconstructed so edge-on that its whole face held one quarter of one
tile.

A wall's own horizontal lines say which way that wall runs. Its skirting, its
cornice, the sills and heads of its windows, and above all the edge where it
meets the floor all lie along it, so they all converge on the
wall's own along-wall vanishing point -- and for a wall facing the camera, that
point is at infinity. So the fit here is done in homogeneous coordinates, where
a point at infinity is just a point with w = 0, instead of the finite (x, y)
that `plane_from_along_wall_vp` accepts.

    wall lines + the floor junction
        -> RANSAC over pairs of lines (seeded), inliers by angle
        -> weighted least squares over the inliers (SVD)
        -> along-wall direction D = K^-1 v
        -> plumb wall normal n = (Dz, 0, -Dx), facing the camera

A wall with too little line evidence gets None, and the pipeline keeps its own
room-VP choice for it.
"""

from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np

from ...core.plane_fit import WallConstraint
from .constraints import MIN_FACING_COMPONENT
from .vp import MIN_LINE_FRACTION, VP_SEED, VERTICAL_TOLERANCE_DEG, _wall_lines

#: A segment agrees with a vanishing point when it points at it within this.
INLIER_ANGLE_DEG = 2.0

#: The floor junction is the most reliable horizontal line a wall has -- a
#: plain painted wall often has no other -- so it counts this many times.
#:
#: Only the floor junction: the floor mask is exact, so the edge where a wall
#: meets it is the wall's real base. A ceiling junction was tried and dropped:
#: with no ceiling mask, "neither wall nor floor" also takes in stairs, a
#: balcony, or the frame edge, and on an atrium it supplied false edges.
JUNCTION_WEIGHT = 3.0

#: At least this many segments must agree before a wall's own direction is
#: trusted over the room-level choice.
MIN_INLIERS = 3

#: The fitted direction must be carried by at least this share of the wall's
#: horizontal evidence, by length, or it is not the wall's direction.
MIN_SUPPORT = 0.4

#: And by at least this much of it, as a fraction of the wall's width.
MIN_SUPPORT_OF_WIDTH = 0.5

#: A junction edge is the wall's boundary pixels with floor (or ceiling) within
#: this many pixels below (or above) them.
JUNCTION_REACH_PX = 4

RANSAC_ITERATIONS = 400


def _segments_as_lines(segs: np.ndarray):
    """Homogeneous lines, lengths, midpoints and unit directions of segments."""
    p1 = np.column_stack([segs[:, 0], segs[:, 1], np.ones(len(segs))])
    p2 = np.column_stack([segs[:, 2], segs[:, 3], np.ones(len(segs))])
    lines = np.cross(p1, p2)
    lines /= np.maximum(np.hypot(lines[:, 0], lines[:, 1]), 1e-9)[:, None]
    delta = np.column_stack([segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]])
    lengths = np.hypot(delta[:, 0], delta[:, 1])
    dirs = delta / np.maximum(lengths, 1e-9)[:, None]
    mids = np.column_stack([(segs[:, 0] + segs[:, 2]) / 2, (segs[:, 1] + segs[:, 3]) / 2])
    return lines, lengths, mids, dirs


def _angles_to(v: np.ndarray, mids: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    """Each segment's angle, in degrees, from the direction towards `v`."""
    towards = np.column_stack([v[0] - mids[:, 0] * v[2], v[1] - mids[:, 1] * v[2]])
    towards /= np.maximum(np.hypot(towards[:, 0], towards[:, 1]), 1e-12)[:, None]
    cos = np.abs(np.sum(towards * dirs, axis=1))
    return np.degrees(np.arccos(np.clip(cos, 0.0, 1.0)))


def _fit(segs: np.ndarray, weights: np.ndarray):
    """The vanishing point most of the weighted segments agree on, homogeneous."""
    lines, lengths, mids, dirs = _segments_as_lines(segs)
    score_of = lengths * weights

    rng = np.random.default_rng(VP_SEED)
    best, best_score = None, -1.0

    pairs = [(i, j) for i in range(len(segs)) for j in range(i + 1, len(segs))]
    if len(pairs) > RANSAC_ITERATIONS:
        pairs = [pairs[k] for k in rng.choice(len(pairs), RANSAC_ITERATIONS, replace=False)]

    for i, j in pairs:
        v = np.cross(lines[i], lines[j])
        norm = np.linalg.norm(v)
        if norm < 1e-12:
            continue
        v = v / norm
        inliers = _angles_to(v, mids, dirs) <= INLIER_ANGLE_DEG
        score = float(score_of[inliers].sum())
        if score > best_score:
            best, best_score = inliers, score

    if best is None or best.sum() < 2:
        return None, None

    # Weighted least squares over the inliers: the point closest to all of
    # their lines. The smallest singular vector handles w -> 0 (infinity).
    w = np.sqrt(score_of[best])[:, None]
    _, _, vt = np.linalg.svd(lines[best] * w)
    v = vt[-1]
    v = v / np.linalg.norm(v)

    return v, best


def _junction_segment(mask: np.ndarray, other: np.ndarray, below: bool) -> Optional[np.ndarray]:
    """
    The straight edge where `mask` meets `other` just below (or above) it.

    Collected as the wall's boundary pixels with `other` within reach, then fitted
    with one robust line (cv2.fitLine, Huber) and cut to the points' extent.
    """
    h, w = mask.shape
    hit = np.zeros_like(mask)

    for k in range(1, JUNCTION_REACH_PX + 1):
        shifted = np.zeros_like(other)
        if below:
            shifted[:-k] = other[k:]
        else:
            shifted[k:] = other[:-k]
        hit |= mask & shifted

    ys, xs = np.nonzero(hit)
    if len(xs) < 20 or xs.max() - xs.min() < 0.05 * w:
        return None

    points = np.column_stack([xs, ys]).astype(np.float32)
    vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    t = (points - [x0, y0]) @ np.array([vx, vy])
    a, b = t.min(), t.max()
    return np.array([x0 + a * vx, y0 + a * vy, x0 + b * vx, y0 + b * vy], dtype=np.float32)


def wall_direction(room_bgr, wall_mask: np.ndarray, floor_mask: np.ndarray,
                   all_walls: np.ndarray, f: float, cx: float, cy: float) -> Optional[dict]:
    """
    One wall's along-wall vanishing point and plumb orientation, from its lines.

    `wall_mask` is this wall and `floor_mask` the floor. `all_walls` is kept in
    the signature for callers; the ceiling junction it served was dropped (see
    JUNCTION_WEIGHT). Returns None when the evidence is too thin to trust, and
    the caller keeps the pipeline's own choice.
    """
    h, w = wall_mask.shape
    min_len = MIN_LINE_FRACTION * float(np.hypot(h, w))

    on_wall, _total = _wall_lines(room_bgr, wall_mask, min_len)

    segs, weights, kinds = [], [], []

    for x1, y1, x2, y2 in on_wall:
        dx, dy = float(x2 - x1), float(y2 - y1)
        if math.degrees(math.atan2(abs(dx), abs(dy))) <= VERTICAL_TOLERANCE_DEG:
            continue  # plumb: says nothing about which way the wall runs
        segs.append((x1, y1, x2, y2)); weights.append(1.0); kinds.append("line")

    seg = _junction_segment(wall_mask, floor_mask, below=True)
    if seg is not None:
        segs.append(tuple(seg)); weights.append(JUNCTION_WEIGHT); kinds.append("floor-junction")

    if len(segs) < 2:
        return None

    segs = np.array(segs, dtype=np.float64)
    weights = np.array(weights, dtype=np.float64)

    v, inliers = _fit(segs, weights)
    if v is None:
        return None

    _, lengths, _, _ = _segments_as_lines(segs)
    support = float((lengths * weights)[inliers].sum())
    total = float((lengths * weights).sum())
    xs = np.nonzero(wall_mask)[1]
    width = float(xs.max() - xs.min() + 1) if len(xs) else 1.0

    if (inliers.sum() < MIN_INLIERS or support < MIN_SUPPORT * total
            or support < MIN_SUPPORT_OF_WIDTH * width):
        return None

    # Along-wall direction in the camera frame, K^-1 v; sign is irrelevant.
    vx, vy, vw = (float(x) for x in v)
    D = np.array([(vx - cx * vw) / f, (vy - cy * vw) / f, vw])
    D /= np.linalg.norm(D)

    a, c = D[2], -D[0]
    norm = math.hypot(a, c)
    if norm < 1e-9:
        return None
    a, c = a / norm, c / norm

    a, b, c, _ = WallConstraint().orient(a, 0.0, c, 1.0)
    if abs(c) < MIN_FACING_COMPONENT:
        return None

    at_infinity = abs(vw) < 1e-6 * max(abs(vx), abs(vy), 1.0)
    yaw = math.degrees(math.atan2(a, -c))

    return {
        "vanishing_point": {
            "homogeneous": [vx, vy, vw],
            "at_infinity": bool(at_infinity),
            "image": None if at_infinity else [vx / vw, vy / vw],
            # For a wall facing the camera, its lines' own image direction.
            "image_direction_deg": math.degrees(math.atan2(vy, vx)) if at_infinity else None,
        },
        "normal": [float(a), float(b), float(c)],
        "yaw_deg": float(yaw),
        "faces": "front" if abs(yaw) < 25 else ("left-side" if yaw > 0 else "right-side"),
        "evidence": {
            "segments": int(len(segs)),
            "inliers": int(inliers.sum()),
            "junctions": [k for k, keep in zip(kinds, inliers) if keep and k != "line"],
            "support_share": round(support / total, 3) if total else 0.0,
        },
    }


# ---------------------------------------------------------------------------
# A wall with no direction of its own: the room's, from the floor.
#
# A plain side wall often has no horizontal line on it at all, and its floor
# junction can be hidden behind a sofa. `wall_direction` then returns None, the
# room-level vanishing points may be missing too, and the pipeline falls back
# to a plane facing the camera -- a side wall running away from the camera
# gets a flat grid. Measured on a living room: the whole left wall was tiled
# fronto-parallel while the floor, the sofa and the ceiling edge all converge
# on one depth vanishing point.
#
# The floor already fixes the room's frame: gravity (its normal) and its depth
# direction (its accepted vanishing point). In a rectangular room every wall
# runs along one of the two horizontal axes that frame gives. So the choice is
# binary -- along the depth axis (a side wall) or across it (a back wall) -- and
# it is made by the wall's own edges: its top edge, its floor junction and any
# horizontal lines on it. A candidate is accepted only when those edges clearly
# prefer it AND its plane puts every pixel of the wall in front of the camera.

#: A wall edge agrees with a candidate direction within this angle.
MANHATTAN_AGREE_DEG = 5.0

#: The winner needs at least this share of the wall's edge length ...
MANHATTAN_MIN_SHARE = 0.5

#: ... and at least this much more than the other candidate.
MANHATTAN_MARGIN = 0.2

#: Share of the wall's pixels the chosen plane must reach in front of the camera.
MANHATTAN_MIN_REACH = 0.98


def _boundary_segments(wall_mask: np.ndarray, all_walls: np.ndarray, min_len: float):
    """
    The wall's top edge, as straight segments: its highest pixel per column
    where the pixel above belongs to no wall and is not the image's top row.
    """
    h, w = wall_mask.shape
    cols = np.where(wall_mask.any(axis=0))[0]
    if len(cols) < 2:
        return []
    top = np.argmax(wall_mask[:, cols], axis=0)
    keep = (top > 2) & ~all_walls[np.clip(top - 1, 0, h - 1), cols]
    pts = np.column_stack([cols[keep], top[keep]]).astype(np.float32)
    if len(pts) < 10:
        return []
    approx = cv2.approxPolyDP(pts.reshape(-1, 1, 2), max(2.0, 0.004 * math.hypot(h, w)), False)[:, 0, :]
    segs = []
    for a, b in zip(approx[:-1], approx[1:]):
        if abs(b[0] - a[0]) < 2:          # a jump between columns, not an edge
            continue
        # A steep run is where something cuts into the wall (a curtain, a
        # door frame), not the wall's top edge: it says nothing about which
        # way the wall runs.
        if math.degrees(math.atan2(abs(b[0] - a[0]), abs(b[1] - a[1]))) <= VERTICAL_TOLERANCE_DEG:
            continue
        if math.hypot(*(b - a)) >= min_len:
            segs.append((float(a[0]), float(a[1]), float(b[0]), float(b[1])))
    return segs


def _plumb_normal(direction: np.ndarray):
    """Plumb wall normal for a horizontal along-wall camera-frame direction, oriented as the pipeline orients it."""
    a, c = float(direction[2]), -float(direction[0])
    norm = math.hypot(a, c)
    if norm < 1e-9:
        return None
    a, b, c, _ = WallConstraint().orient(a / norm, 0.0, c / norm, 1.0)
    return np.array([a, b, c], dtype=np.float64)


def _reach(normal: np.ndarray, mask: np.ndarray, f: float, cx: float, cy: float) -> float:
    """Share of the mask's pixels on the camera side of the plane's vanishing line."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return 0.0
    den = normal[0] * (xs - cx) / f + normal[1] * (ys - cy) / f + normal[2]
    side = den < 0
    return float(max(side.mean(), 1.0 - side.mean()))


def manhattan_candidates(room_bgr, wall_mask: np.ndarray, floor_mask: np.ndarray,
                         all_walls: np.ndarray, floor_geometry: dict,
                         f: float, cx: float, cy: float):
    """
    The room frame's two wall directions for this wall, each with how much of
    the wall's edge length agrees with it: {"side": {...}, "back": {...}}, or
    None when the floor fixes no frame. Also returns the edge segments used.
    """
    if not floor_geometry or floor_geometry.get("status") != "detected":
        return None, []
    vp = (floor_geometry.get("vanishing_points") or {}).get("depth_vp")
    plane = floor_geometry.get("plane")
    if vp is None or plane is None:
        return None, []

    up = np.array(plane[:3], dtype=np.float64)
    up /= np.linalg.norm(up) or 1.0
    if up[1] > 0:                              # gravity points to the image top (-y)
        up = -up

    depth = np.array([(vp[0] - cx) / f, (vp[1] - cy) / f, 1.0])
    depth -= (depth @ up) * up
    depth /= np.linalg.norm(depth) or 1.0
    across = np.cross(up, depth)
    across /= np.linalg.norm(across) or 1.0

    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    h, w = wall_mask.shape
    min_len = MIN_LINE_FRACTION * float(np.hypot(h, w))

    segs = [(s, "top-edge") for s in _boundary_segments(wall_mask, all_walls, min_len)]
    junction = _junction_segment(wall_mask, floor_mask, below=True)
    if junction is not None:
        segs.append((tuple(float(v) for v in junction), "floor-junction"))
    on_wall, _total = _wall_lines(room_bgr, wall_mask, min_len)
    for x1, y1, x2, y2 in on_wall:
        if math.degrees(math.atan2(abs(x2 - x1), abs(y2 - y1))) > VERTICAL_TOLERANCE_DEG:
            segs.append(((float(x1), float(y1), float(x2), float(y2)), "line"))

    candidates = {}
    for name, along in (("side", depth), ("back", across)):
        normal = _plumb_normal(along)
        candidates[name] = {
            "share": 0.0,
            "normal": normal,
            "vp": K @ along,
            "reach": 0.0 if normal is None else _reach(normal, wall_mask, f, cx, cy),
        }
    if not segs:
        return candidates, segs

    arr = np.array([s for s, _ in segs], dtype=np.float64)
    # Every edge counts by its length alone here. The floor junction's extra
    # weight in `wall_direction` assumes an exact floor edge; where curtains or
    # furniture stood the floor mask's edge is irregular, and in a two-way
    # choice one bent junction must not outvote a straight ceiling edge.
    _, lengths, mids, dirs = _segments_as_lines(arr)
    total = float(lengths.sum()) or 1.0
    for c in candidates.values():
        agree = _angles_to(c["vp"], mids, dirs) < MANHATTAN_AGREE_DEG
        c["share"] = float(lengths[agree].sum()) / total
    return candidates, segs


def axis_of(normal, candidates) -> Optional[str]:
    """Which room axis ("side" / "back") a wall normal runs closest to."""
    if not candidates or normal is None:
        return None
    n = np.asarray(normal, float)
    n /= np.linalg.norm(n) or 1.0
    best, best_cos = None, -1.0
    for name, c in candidates.items():
        if c["normal"] is None:
            continue
        cos = abs(float(n @ c["normal"]))
        if cos > best_cos:
            best, best_cos = name, cos
    return best


def manhattan_direction(room_bgr, wall_mask: np.ndarray, floor_mask: np.ndarray,
                        all_walls: np.ndarray, floor_geometry: dict,
                        f: float, cx: float, cy: float) -> Optional[dict]:
    """
    A wall's direction from the floor's room frame, when the wall's own lines
    could not give one. Same return shape as `wall_direction`, or None when the
    floor was not detected or the wall's edges do not clearly choose.
    """
    candidates, segs = manhattan_candidates(
        room_bgr, wall_mask, floor_mask, all_walls, floor_geometry, f, cx, cy)
    if not candidates or not segs:
        return None

    best = max(candidates, key=lambda k: candidates[k]["share"])
    other = "back" if best == "side" else "side"
    win, lose = candidates[best], candidates[other]
    if (win["normal"] is None or win["share"] < MANHATTAN_MIN_SHARE
            or win["share"] - lose["share"] < MANHATTAN_MARGIN
            or win["reach"] < MANHATTAN_MIN_REACH):
        return None

    a, b, c = (float(x) for x in win["normal"])
    vx, vy, vw = (float(x) for x in win["vp"])
    at_infinity = abs(vw) < 1e-6 * max(abs(vx), abs(vy), 1.0)
    yaw = math.degrees(math.atan2(a, -c))
    return {
        "vanishing_point": {
            "homogeneous": [vx, vy, vw],
            "at_infinity": bool(at_infinity),
            "image": None if at_infinity else [vx / vw, vy / vw],
            "image_direction_deg": math.degrees(math.atan2(vy, vx)) if at_infinity else None,
        },
        "normal": [a, b, c],
        "yaw_deg": float(yaw),
        # Which side of the room: a side wall is left or right of the depth
        # vanishing point in the image (the normal's sign says nothing here).
        "faces": "front" if best == "back" else (
            "left-side" if float(np.nonzero(wall_mask)[1].mean()) < float(candidates["side"]["vp"][0] / candidates["side"]["vp"][2])
            else "right-side"),
        "source": "floor-manhattan",
        "evidence": {
            "axis": best,
            "share": {k: round(v["share"], 3) for k, v in candidates.items()},
            "mask_reach": round(win["reach"], 4),
            "segments": {kind: sum(1 for _, k in segs if k == kind) for kind in ("top-edge", "floor-junction", "line")},
        },
    }
