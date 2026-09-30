"""
Vanishing points from the room's own mask boundaries, for rooms with little
texture (plain walls, bare floors): few image lines, but the surfaces still
meet along straight edges.

Line sources, all from the Clean Room masks, never from texture:

  * floor/wall junction and wall/ceiling line -- horizontal lines in the walls'
    planes; the receding ones meet at the room's DEPTH vanishing point;
  * wall/wall boundaries -- the vertical corner edges between wall pieces;
    they meet at VP_Y (vertical_boundary_segments, for camera/vertical_vp).

Only unbroken stretches (surface/wall/edge_layout._stretch_runs: no bridging
across hidden gaps) and only LONG runs count (MIN_LENGTH_FRACTION of the image
diagonal): a short run is mask noise at a skirting or a cornice, not structure.

Every VP comes with its evidence, so it can be judged rather than trusted:

  support       inlier run length / all candidate run length
  spread_deg    median angle between an inlier run and the direction from its
                midpoint to the VP
  inlier_ratio  inlier runs / candidate runs
  confidence    support x min(1, inliers / MIN_INLIERS_FULL) x exp(-spread / SPREAD_SCALE_DEG)

and `accepted` is confidence >= MIN_CONFIDENCE with at least MIN_INLIERS runs.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

#: Candidate runs must be at least this share of the image diagonal.
MIN_LENGTH_FRACTION = 0.05
#: A run supports a VP when it points at it within this angle.
INLIER_DEG = 2.0
#: Runs within this angle of the image horizontal face the camera: no depth.
FLAT_DEG = 3.0
#: Acceptance.
MIN_INLIERS = 2
MIN_INLIERS_FULL = 4
SPREAD_SCALE_DEG = 2.0
MIN_CONFIDENCE = 0.35


def _angle_to(seg, vp) -> float:
    """Angle (deg) between segment `seg` and the direction from its midpoint to homogeneous `vp`."""
    x1, y1, x2, y2 = seg
    mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    if abs(vp[2]) < 1e-9:
        tx, ty = vp[0], vp[1]
    else:
        tx, ty = vp[0] / vp[2] - mx, vp[1] / vp[2] - my
    a = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0
    b = math.degrees(math.atan2(ty, tx)) % 180.0
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def _line(seg):
    l = np.cross([seg[0], seg[1], 1.0], [seg[2], seg[3], 1.0])
    return l / (math.hypot(l[0], l[1]) or 1.0)


def fit_vp(segs: list, finite_only: bool = False) -> Optional[dict]:
    """RANSAC over run pairs; refined by length-weighted least squares on the inliers."""
    if len(segs) < MIN_INLIERS:
        return None
    lengths = np.array([math.hypot(s[2] - s[0], s[3] - s[1]) for s in segs])
    best = None
    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            v = np.cross(_line(segs[i]), _line(segs[j]))
            if np.linalg.norm(v) < 1e-12 or (finite_only and abs(v[2]) < 1e-9):
                continue
            res = np.array([_angle_to(s, v) for s in segs])
            inl = res <= INLIER_DEG
            score = lengths[inl].sum()
            if best is None or score > best[0]:
                best = (score, v, inl)
    if best is None or best[2].sum() < MIN_INLIERS:
        return None
    _, v, inl = best
    L = np.array([_line(s) * lengths[k] for k, s in enumerate(segs) if inl[k]])
    _, _, vt = np.linalg.svd(L)
    v = vt[-1]
    res = np.array([_angle_to(s, v) for s in segs])
    inl = res <= INLIER_DEG
    n_in = int(inl.sum())
    support = float(lengths[inl].sum() / lengths.sum())
    spread = float(np.median(res[inl])) if n_in else 90.0
    confidence = support * min(1.0, n_in / MIN_INLIERS_FULL) * math.exp(-spread / SPREAD_SCALE_DEG)
    at_inf = abs(v[2]) < 1e-9 * max(abs(v[0]), abs(v[1]), 1.0)
    return {
        "homogeneous": [float(x) for x in v],
        "image": None if at_inf else [float(v[0] / v[2]), float(v[1] / v[2])],
        "at_infinity": bool(at_inf),
        "runs": len(segs), "inliers": n_in,
        "support": round(support, 3), "spread_deg": round(spread, 3), "inlier_ratio": round(n_in / len(segs), 3),
        "confidence": round(float(confidence), 3),
        "accepted": bool(n_in >= MIN_INLIERS and confidence >= MIN_CONFIDENCE),
        "inlier_runs": [[round(float(x), 1) for x in s] for k, s in enumerate(segs) if inl[k]],
    }


def edge_runs(floor: np.ndarray, wall: np.ndarray, objects: Optional[np.ndarray] = None) -> list:
    """Long straight runs of the floor junction and the ceiling line, unbroken stretches only."""
    from ..surface.wall import edge_layout
    from ..surface.wall import junction_layout as jl

    h, w = wall.shape
    min_len = MIN_LENGTH_FRACTION * math.hypot(h, w)
    runs = []
    for pts, src in ((jl.junction_points(floor, wall, objects), "floor-junction"),
                     (edge_layout.ceiling_points(wall, objects), "ceiling-line")):
        for stretch in edge_layout._stretch_runs(pts, w, h):
            for s in stretch:
                if math.hypot(s[2] - s[0], s[3] - s[1]) >= min_len:
                    runs.append((s, src))
    return runs


def depth_vp(floor: np.ndarray, wall: np.ndarray, objects: Optional[np.ndarray] = None) -> Optional[dict]:
    """The room's depth VP from the receding junction / ceiling runs, with its evidence."""
    runs = [(s, src) for s, src in edge_runs(floor, wall, objects)
            if abs(math.degrees(math.atan2(s[3] - s[1], s[2] - s[0]))) >= FLAT_DEG]
    fitted = fit_vp([s for s, _ in runs], finite_only=True)
    if fitted is not None:
        fitted["sources"] = sorted({src for s, src in runs})
    return fitted


def vertical_boundary_segments(wall_masks: dict, min_len: float) -> list:
    """
    The vertical corner edges between wall pieces: per boundary column pair of
    two adjacent pieces, the rows where they touch, as one fitted segment.
    """
    segs = []
    items = sorted(wall_masks.items())
    for (_, a), (_, b) in zip(items, items[1:]):
        touch = a[:, :-1] & b[:, 1:] | b[:, :-1] & a[:, 1:]
        ys, xs = np.nonzero(touch)
        if len(ys) < 2 or ys.max() - ys.min() < min_len:
            continue
        A = np.vstack([ys, np.ones_like(ys)]).T.astype(float)
        k, c = np.linalg.lstsq(A, xs.astype(float), rcond=None)[0]     # x = k*y + c
        y0, y1 = float(ys.min()), float(ys.max())
        segs.append([k * y0 + c, y0, k * y1 + c, y1])
    return segs


# ---------------------------------------------------------------------------
# One pool of long structural lines, and one way to score a VP against it.

#: A VP is confident with at least this many inlier lines, this support and
#: at most this spread (the same rule for every VP the room reports).
ACCEPT_INLIERS = 4
ACCEPT_SUPPORT = 0.4
ACCEPT_SPREAD_DEG = 1.0

#: A pool line must lie on the floor or a wall (not a removed object) for at
#: least this share of its length; plumb lines within this of vertical are out.
ON_SURFACE_SHARE = 0.8
PLUMB_DEG = 20.0


def long_line_pool(room_bgr, floor: np.ndarray, wall: np.ndarray, objects: Optional[np.ndarray] = None) -> list:
    """
    Long, receding structural lines: LSD lines of at least MIN_LENGTH_FRACTION
    of the diagonal lying on floor or wall (not on a removed object), neither
    plumb nor image-horizontal -- plus the long floor-junction / ceiling-line
    runs. Short, random and texture lines never enter.
    """
    import cv2

    from .vp_from_mask import detect_lines_lsd

    h, w = floor.shape
    diag = math.hypot(h, w)
    surface = cv2.dilate((floor | wall).astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    if objects is not None:
        surface &= ~objects
    segs = []
    lines = detect_lines_lsd(cv2.cvtColor(room_bgr, cv2.COLOR_BGR2GRAY))
    for x1, y1, x2, y2 in np.asarray(lines if lines is not None else [], float).reshape(-1, 4):
        length = math.hypot(x2 - x1, y2 - y1)
        a = abs(math.degrees(math.atan2(y2 - y1, x2 - x1))) % 180.0
        if length < MIN_LENGTH_FRACTION * diag or min(a, 180.0 - a) < FLAT_DEG or abs(a - 90.0) < PLUMB_DEG:
            continue
        t = np.linspace(0.0, 1.0, 9)
        xs = np.clip((x1 + t * (x2 - x1)).astype(int), 0, w - 1)
        ys = np.clip((y1 + t * (y2 - y1)).astype(int), 0, h - 1)
        if surface[ys, xs].mean() >= ON_SURFACE_SHARE:
            segs.append([float(x1), float(y1), float(x2), float(y2)])
    segs += [list(s) for s, _ in edge_runs(floor, wall, objects)
             if abs(math.degrees(math.atan2(s[3] - s[1], s[2] - s[0]))) >= FLAT_DEG]
    return segs


def score_vp(segs: list, point) -> dict:
    """How well the pool's lines point at image point `point` (x, y)."""
    if not segs or point is None:
        return {"lines": len(segs), "inliers": 0, "support": 0.0, "spread_deg": None, "inlier_ratio": 0.0,
                "confident": False}
    v = np.array([point[0], point[1], 1.0])
    lengths = np.array([math.hypot(s[2] - s[0], s[3] - s[1]) for s in segs])
    res = np.array([_angle_to(s, v) for s in segs])
    inl = res <= INLIER_DEG
    n = int(inl.sum())
    support = float(lengths[inl].sum() / lengths.sum())
    spread = float(np.median(res[inl])) if n else None
    return {"lines": len(segs), "inliers": n, "support": round(support, 3),
            "spread_deg": None if spread is None else round(spread, 3), "inlier_ratio": round(n / len(segs), 3),
            "confident": bool(n >= ACCEPT_INLIERS and support >= ACCEPT_SUPPORT
                              and spread is not None and spread <= ACCEPT_SPREAD_DEG)}
