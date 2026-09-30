"""
A wall's direction from its own two edges, fitted robustly.

The floor junction and the ceiling line are both horizontal lines lying in the
wall's plane, so each meets the horizon at the wall's along-wall vanishing
point -- at infinity for a wall facing the camera. Each edge is fitted with a
seeded RANSAC line (mask boundaries are noisy and have jogs at skirtings and
cornices), each fit gives the wall's horizontal direction, and with both edges
seen the two must agree:

  * both edges fitted and agreeing  -> validated, direction = their weighted mean
  * one edge fitted with good support -> validated, from that edge
  * edges disagreeing, or too little support -> the best estimate, NOT validated,
    with the reason (never forced to look certain)

A receding wall within junction_layout.SNAP_DEG (20 deg) of the room's depth
direction takes that direction exactly (a rectangular room); a wall whose
edges are within FLAT_DEG of the image horizontal faces the camera.

Returns the same direction dict junction_layout._direction does, plus
"fit" (per edge: line, inliers, span, residual), "confidence" and "validated".
"""

from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np

from . import edge_layout
from . import junction_layout as jl

#: RANSAC: a boundary pixel within this of the line supports it.
INLIER_PX = 3.0
ITERATIONS = 300
SEED = 1337

#: An edge fit is usable with at least this many supporting columns, spanning
#: this share of the image width, and this share of its own points.
MIN_INLIERS = 30
MIN_SPAN_FRACTION = jl.MIN_RUN_FRACTION
MIN_INLIER_SHARE = 0.5

#: Junction and ceiling fits of one wall must agree within this (horizontal
#: direction, degrees) for the direction to count as validated.
AGREE_DEG = 8.0


def _ransac_line(pts: np.ndarray):
    """(a, b, c) normalised image line, inlier mask) or (None, None)."""
    if len(pts) < MIN_INLIERS:
        return None, None
    rng = np.random.default_rng(SEED)
    best = None
    n = len(pts)
    for _ in range(ITERATIONS):
        i, j = rng.choice(n, 2, replace=False)
        p, q = pts[i], pts[j]
        if abs(q[0] - p[0]) < 2.0:                 # edges run across the image
            continue
        l = np.cross([p[0], p[1], 1.0], [q[0], q[1], 1.0])
        l = l / (math.hypot(l[0], l[1]) or 1.0)
        inl = np.abs(pts @ l[:2] + l[2]) <= INLIER_PX
        if best is None or inl.sum() > best[1].sum():
            best = (l, inl)
    if best is None:
        return None, None
    inl = best[1]
    vx, vy, x0, y0 = cv2.fitLine(pts[inl].astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    l = np.cross([x0, y0, 1.0], [x0 + vx, y0 + vy, 1.0])
    l = l / (math.hypot(l[0], l[1]) or 1.0)
    inl = np.abs(pts @ l[:2] + l[2]) <= INLIER_PX
    return l, inl


def _azimuth(line, horizon_y: float, f: float, cx: float) -> float:
    """Horizontal direction (deg in [0, 180)) of a horizontal 3D line imaged as `line`."""
    a, b, c = line
    if abs(a) < 1e-12:                              # image-horizontal: parallel to the image plane
        return 90.0
    x = -(b * horizon_y + c) / a                    # where it meets the horizon
    return math.degrees(math.atan2(x - cx, f)) % 180.0


def _circ_diff(a: float, b: float) -> float:
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def fit(mask: np.ndarray, floor: np.ndarray, objects: Optional[np.ndarray], horizon_y: Optional[float],
        f: float, cx: float, cy: float, depth_vp=None) -> Optional[dict]:
    """The wall's direction from its own junction and ceiling line, or None when neither is usable."""
    if horizon_y is None or not mask.any():
        return None
    h, w = mask.shape
    edges = {"floor-junction": jl.junction_points(floor, mask, objects),
             "ceiling-line": edge_layout.ceiling_points(mask, objects)}
    fits = {}
    for name, pts in edges.items():
        line, inl = _ransac_line(pts)
        if line is None:
            continue
        span = float(pts[inl, 0].max() - pts[inl, 0].min()) if inl.any() else 0.0
        share = float(inl.mean())
        angle = math.degrees(math.atan2(-line[0], line[1]))            # image angle of the line
        angle = (angle + 90.0) % 180.0 - 90.0
        usable = inl.sum() >= MIN_INLIERS and span >= MIN_SPAN_FRACTION * w and share >= MIN_INLIER_SHARE
        fits[name] = {"line": [float(v) for v in line], "inliers": int(inl.sum()), "points": int(len(pts)),
                      "inlier_share": round(share, 3), "span_px": round(span, 1), "image_angle_deg": round(angle, 2),
                      "residual_px": round(float(np.median(np.abs(pts[inl] @ line[:2] + line[2]))), 2) if inl.any() else None,
                      "usable": bool(usable)}
    usable = {k: v for k, v in fits.items() if v["usable"]}
    if not usable:
        return None

    flat = all(abs(v["image_angle_deg"]) < jl.FLAT_DEG for v in usable.values())
    azs = {k: (90.0 if flat else _azimuth(np.array(v["line"]), horizon_y, f, cx)) for k, v in usable.items()}
    reasons = []
    if len(usable) == 2:
        disagree = _circ_diff(azs["floor-junction"], azs["ceiling-line"])
        wa, wb = usable["floor-junction"]["inliers"], usable["ceiling-line"]["inliers"]
        # Weighted circular mean on the doubled angle (directions are mod 180).
        ang = [math.radians(2 * azs[k]) for k in ("floor-junction", "ceiling-line")]
        az = math.degrees(math.atan2(wa * math.sin(ang[0]) + wb * math.sin(ang[1]),
                                     wa * math.cos(ang[0]) + wb * math.cos(ang[1]))) / 2.0 % 180.0
        if disagree > AGREE_DEG:
            reasons.append(f"junction and ceiling line disagree by {disagree:.1f} deg")
    else:
        disagree = None
        (only,) = usable
        az = azs[only]
        if usable[only]["inlier_share"] < 0.7:
            reasons.append(f"only the {only} was usable, {100 * usable[only]['inlier_share']:.0f}% of its points on one line")

    snapped = False
    if not flat and depth_vp is not None:
        az_depth = math.degrees(math.atan2(depth_vp[0] - cx, f)) % 180.0
        if _circ_diff(az, az_depth) < jl.SNAP_DEG:
            az, snapped = az_depth, True

    # Back to the image: the along-wall VP on the horizon (homogeneous; at
    # infinity for a wall facing the camera).
    dz, dx = math.cos(math.radians(az)), math.sin(math.radians(az))
    v = np.array([cx * dz + f * dx, horizon_y * dz, dz])
    at_inf = abs(dz) < 1e-9 or flat
    if at_inf:
        v = np.array([1.0, 0.0, 0.0]) if flat else np.array([f * dx, 0.0, 0.0])
    D = np.array([(v[0] - cx * v[2]) / f, (v[1] - horizon_y * v[2]) / f * 0.0, v[2]])   # horizontal direction
    a, c = float(D[2]), -float(D[0])
    norm = math.hypot(a, c)
    if norm < 1e-9:
        return None
    from ...core.plane_fit import WallConstraint
    a, b, c, _ = WallConstraint().orient(a / norm, 0.0, c / norm, 1.0)
    xs = np.nonzero(mask.any(axis=0))[0]
    mid_x = float(xs.mean())
    vx = v[0] / v[2] if not at_inf else None
    faces = "front" if at_inf and flat else ("left-side" if (vx if vx is not None else cx) > mid_x else "right-side")

    support = sum(v_["inliers"] for v_ in usable.values())
    confidence = min(1.0, support / 200.0) * (1.0 if disagree is None else max(0.0, 1.0 - disagree / (2 * AGREE_DEG)))
    confidence *= (1.0 if len(usable) == 2 else 0.8)
    return {
        "vanishing_point": {
            "homogeneous": [float(x) for x in v],
            "at_infinity": bool(at_inf),
            "image": None if at_inf else [float(v[0] / v[2]), float(v[1] / v[2])],
            "image_direction_deg": math.degrees(math.atan2(v[1], v[0])) if at_inf else None,
        },
        "normal": [float(a), float(b), float(c)],
        "yaw_deg": float(math.degrees(math.atan2(a, -c))),
        "faces": faces,
        "source": "+".join(sorted(usable)),
        "evidence": {"horizon_y": round(float(horizon_y), 1), "snapped_to_depth_vp": snapped,
                     "edges_disagree_deg": None if disagree is None else round(disagree, 2)},
        "fit": fits,
        "confidence": round(float(confidence), 3),
        "validated": not reasons,
        "validation": "; ".join(reasons) or None,
    }
