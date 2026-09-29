"""
The room's walls from where they stand on the floor.

Every wall meets the floor along a straight line, and where the room turns a
corner that line bends. So the floor/wall junction -- the wall's lowest pixel
in each column, kept only where floor lies directly beneath it -- is a
polyline whose straight runs are the room's walls and whose bends are its
corners:

    |\\                                   /|
    | \\______   back wall (flat run)  __/ |     flat run      -> faces the camera
    |  pillar \\______________________/    |     receding run  -> runs into the room
     left wall                          right wall

That gives, per wall and from the image alone:

  * its exact left and right boundary (the corner columns);
  * its direction: a run's image line meets the horizon at the wall's
    along-wall vanishing point (at infinity for a run parallel to it);
  * and, where two or more receding runs agree on one point, the room's depth
    vanishing point -- the horizon -- even on a bare floor that gives the
    floor-only detector nothing to work with.

Only pixels where WALL sits directly on FLOOR count, so a sofa's outline or a
table edge is never mistaken for a junction (the failure that kept the older
silhouette VP in camera/floor_boundary.py disconnected). The layout is used
only when the junction is visible along most of the wall (JUNCTION_MIN_SHARE);
otherwise the caller keeps its own split.
"""

from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np

#: Floor within this many pixels below a wall's lowest pixel makes it a junction.
JUNCTION_REACH_PX = 6

#: The layout is used only when the junction is seen under this share of the
#: wall mask's columns (furniture along a wall hides it).
JUNCTION_MIN_SHARE = 0.75

#: Polyline simplification, as a share of the image diagonal.
SIMPLIFY_FRACTION = 0.006

#: A run shorter than this share of the image width is a jog, not a wall.
MIN_RUN_FRACTION = 0.03

#: Consecutive runs within this angle are the same wall.
COLLINEAR_DEG = 6.0

#: A run within this angle of the horizon runs parallel to the image plane.
FLAT_DEG = 3.0

#: Receding runs support a depth vanishing point within this angle.
VP_AGREE_DEG = 3.0

#: Receding runs, by length, that must agree on the depth vanishing point.
VP_MIN_SHARE = 0.6

#: A receding wall within this angle of the room's depth direction runs along
#: it (a rectangular room), and takes that direction exactly: a short junction
#: run's few degrees of noise would otherwise tilt a whole side wall.
SNAP_DEG = 20.0


def junction_points(floor: np.ndarray, wall: np.ndarray, objects: Optional[np.ndarray] = None) -> np.ndarray:
    """
    (N, 2) float (x, y): per column, the wall's lowest pixel where floor lies
    just below it. Where an object was removed from the photo, that edge was
    painted in by the clean-room fill, not seen, so it does not count.
    """
    h, w = wall.shape
    cols = np.where(wall.any(axis=0))[0]
    if not len(cols):
        return np.empty((0, 2))
    bottom = (h - 1) - np.argmax(wall[::-1, cols], axis=0)
    below = np.zeros(len(cols), dtype=bool)
    for k in range(1, JUNCTION_REACH_PX + 1):
        below |= floor[np.clip(bottom + k, 0, h - 1), cols]
    keep = below & (bottom < h - 2)
    if objects is not None:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * JUNCTION_REACH_PX + 1,) * 2)
        grown = cv2.dilate(objects.astype(np.uint8), k).astype(bool)
        keep &= ~grown[np.clip(bottom, 0, h - 1), cols]
    return np.column_stack([cols[keep], bottom[keep]]).astype(np.float64)


def _angle_deg(seg) -> float:
    return math.degrees(math.atan2(seg[3] - seg[1], seg[2] - seg[0]))


def _line(seg):
    return np.cross([seg[0], seg[1], 1.0], [seg[2], seg[3], 1.0])


#: A break in the junction wider than this share of the image width ends a
#: run: nothing is known about the wall across it, so no run may bridge it.
GAP_FRACTION = 0.015


def runs(floor: np.ndarray, wall: np.ndarray, objects: Optional[np.ndarray] = None):
    """
    The junction's straight runs, left to right, as (x0, y0, x1, y1) with the
    share of wall columns they were measured on. None when too little of the
    junction is visible.
    """
    h, w = wall.shape
    pts = junction_points(floor, wall, objects)
    wall_cols = int(wall.any(axis=0).sum())
    share = len(pts) / max(wall_cols, 1)
    if share < JUNCTION_MIN_SHARE or len(pts) < 20:
        return None, share
    pts = pts[np.argsort(pts[:, 0])]
    eps = SIMPLIFY_FRACTION * float(np.hypot(h, w))

    # Simplify each unbroken stretch of junction on its own.
    breaks = np.where(np.diff(pts[:, 0]) > max(8.0, GAP_FRACTION * w))[0] + 1
    segs = []
    for chain in np.split(pts, breaks):
        if len(chain) < 2:
            continue
        poly = cv2.approxPolyDP(chain.astype(np.float32).reshape(-1, 1, 2), eps, False)[:, 0, :].astype(float)
        segs += [list(poly[i]) + list(poly[i + 1]) for i in range(len(poly) - 1)]
    segs = [s for s in segs if s[2] > s[0]]          # left to right only

    # Short jogs join their longer neighbour (they are mask noise at a corner).
    min_run = MIN_RUN_FRACTION * w
    changed = True
    while changed and len(segs) > 1:
        changed = False
        for i, s in enumerate(segs):
            if s[2] - s[0] >= min_run:
                continue
            if i == 0:
                j = 1
            elif i == len(segs) - 1:
                j = i - 1
            else:
                left, right = segs[i - 1], segs[i + 1]
                j = i - 1 if (left[2] - left[0]) >= (right[2] - right[0]) else i + 1
            a, b = (segs[j], s) if j < i else (s, segs[j])
            merged = [a[0], a[1], b[2], b[3]]
            segs[min(i, j)] = merged
            del segs[max(i, j)]
            changed = True
            break

    # Consecutive runs on one line are one wall.
    out = []
    for s in segs:
        if out and abs(_angle_deg(out[-1]) - _angle_deg(s)) < COLLINEAR_DEG:
            out[-1] = [out[-1][0], out[-1][1], s[2], s[3]]
        else:
            out.append(list(s))
    return out, share


def depth_vp(segs, w: int) -> Optional[dict]:
    """
    The room's depth vanishing point where the receding runs agree, or None.
    Receding = not parallel to the image's horizontal.
    """
    rec = [s for s in segs if abs(_angle_deg(s)) >= FLAT_DEG and (s[2] - s[0]) >= MIN_RUN_FRACTION * w]
    if len(rec) < 2:
        return None
    lengths = np.array([math.hypot(s[2] - s[0], s[3] - s[1]) for s in rec])
    total = float(lengths.sum())
    best, best_share = None, 0.0
    for i in range(len(rec)):
        for j in range(i + 1, len(rec)):
            v = np.cross(_line(rec[i]), _line(rec[j]))
            if abs(v[2]) < 1e-9:
                continue
            p = v[:2] / v[2]
            agree = 0.0
            for s, L in zip(rec, lengths):
                mid = np.array([(s[0] + s[2]) / 2, (s[1] + s[3]) / 2])
                d = np.array([s[2] - s[0], s[3] - s[1]])
                to = p - mid
                cos = abs(float(d @ to)) / (np.linalg.norm(d) * np.linalg.norm(to) + 1e-9)
                if math.degrees(math.acos(min(1.0, cos))) < VP_AGREE_DEG:
                    agree += L
            if agree / total > best_share:
                best, best_share = p, agree / total
    if best is None or best_share < VP_MIN_SHARE:
        return None
    return {"point": [float(best[0]), float(best[1])], "share": round(best_share, 3),
            "runs": len(rec)}


def _unit_dir(v, f, cx, cy):
    D = np.array([(v[0] - cx * v[2]) / f, (v[1] - cy * v[2]) / f, v[2]])
    return D / (np.linalg.norm(D) or 1.0)


def _direction(seg, horizon_y: float, f: float, cx: float, cy: float,
               depth_vp=None) -> Optional[dict]:
    """A run's along-wall vanishing point on the horizon, and the plumb wall it implies."""
    from ...core.plane_fit import WallConstraint

    horizon = np.array([0.0, 1.0, -horizon_y])            # y = horizon_y (level camera)
    v = np.cross(_line(seg), horizon)
    snapped = False
    if abs(_angle_deg(seg)) < FLAT_DEG or abs(v[2]) < 1e-9:
        v = np.array([seg[2] - seg[0], seg[3] - seg[1], 0.0])   # parallel: at infinity
    elif depth_vp is not None:
        dv = np.array([depth_vp[0], depth_vp[1], 1.0])
        a_run, a_depth = _unit_dir(v, f, cx, cy), _unit_dir(dv, f, cx, cy)
        # Only the horizontal (x, z) components decide which way a plumb wall runs.
        h_run, h_depth = a_run[[0, 2]], a_depth[[0, 2]]
        cos = abs(float(h_run @ h_depth)) / ((np.linalg.norm(h_run) * np.linalg.norm(h_depth)) or 1.0)
        if math.degrees(math.acos(min(1.0, cos))) < SNAP_DEG:
            v, snapped = dv, True
    D = np.array([(v[0] - cx * v[2]) / f, (v[1] - cy * v[2]) / f, v[2]])
    n = np.linalg.norm(D)
    if n < 1e-12:
        return None
    D /= n
    a, c = float(D[2]), -float(D[0])
    norm = math.hypot(a, c)
    if norm < 1e-9:
        return None
    a, b, c, _ = WallConstraint().orient(a / norm, 0.0, c / norm, 1.0)
    at_inf = abs(v[2]) < 1e-9
    yaw = math.degrees(math.atan2(a, -c))
    mid_x = (seg[0] + seg[2]) / 2.0
    flat = abs(_angle_deg(seg)) < FLAT_DEG
    return {
        "vanishing_point": {
            "homogeneous": [float(x) for x in v],
            "at_infinity": bool(at_inf),
            "image": None if at_inf else [float(v[0] / v[2]), float(v[1] / v[2])],
            "image_direction_deg": math.degrees(math.atan2(v[1], v[0])) if at_inf else None,
        },
        "normal": [float(a), float(b), float(c)],
        "yaw_deg": float(yaw),
        "faces": "front" if flat else ("left-side" if (v[0] / v[2] if not at_inf else cx) > mid_x else "right-side"),
        "source": "floor-junction",
        "evidence": {"junction_run": [round(float(x), 1) for x in seg], "horizon_y": round(float(horizon_y), 1),
                     "snapped_to_depth_vp": snapped},
    }


def layout(floor: np.ndarray, wall: np.ndarray, horizon_y: Optional[float],
           f: float, cx: float, cy: float, depth=None, segs=None, share=None, objects=None):
    """
    [(mask, direction)] -- one per wall, left to right -- or (None, reason).
    Every WALL_MASK pixel goes to exactly one wall: by column, between the
    corners the junction bends at.
    """
    h, w = wall.shape
    if segs is None:
        segs, share = runs(floor, wall, objects)
    if segs is None:
        return None, {"stage": "junction-hidden", "junction_share": round(share, 3)}
    if len(segs) < 1:
        return None, {"stage": "no-runs", "junction_share": round(share, 3)}
    if horizon_y is None:
        return None, {"stage": "no-horizon", "junction_share": round(share, 3)}

    # Corner columns: where one run ends and the next begins.
    corners = [(segs[i][2] + segs[i + 1][0]) / 2.0 for i in range(len(segs) - 1)]
    edges = [-1.0] + corners + [float(w)]
    cols = np.arange(w)[None, :]
    walls = []
    for i, s in enumerate(segs):
        in_span = (cols > edges[i]) & (cols <= edges[i + 1])
        mask = wall & np.broadcast_to(in_span, wall.shape)
        if not mask.any():
            continue
        d = _direction(s, horizon_y, f, cx, cy, depth)
        walls.append((mask, d))
    return walls, {"stage": "ok", "junction_share": round(share, 3), "runs": [
        [round(float(x), 1) for x in s] for s in segs], "corners": [round(c, 1) for c in corners]}
