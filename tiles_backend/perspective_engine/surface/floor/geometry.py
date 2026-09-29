"""
The floor's perspective geometry, detected once and kept.

The floor pipeline detects its vanishing points and focal length inside every
render. This runs the SAME detection -- `vp.extract_boundary`, `vp.resolve`
(floor lines plus the mask's floor-to-wall junction, RANSAC, confidence gate)
and `resolve_focal_length` (EXIF, then two-VP calibration, then the prior) --
and records the result, so that one detection is what every tile render of the
room projects with, and so the geometry can be inspected.

On top of what the pipeline itself decides, it measures what that geometry
implies: the camera's pitch and viewing direction, the floor plane, how much of
FLOOR_MASK the plane actually reaches, and how big the visible floor is.

`status`:
  detected       the detector's vanishing point was accepted and its floor
                 plane reaches the floor mask
  out-of-view    it was accepted, but its horizon puts the plane where the mask
                 is not (a staircase's curved edges once scored 0.74 this way)
  rejected       below the confidence gate, or no vanishing point at all

Only a `detected` floor is used as-is. The other two are resolved at render
time, when the room's dimensions are known, from the backend's estimated
camera -- which is what makes the fallback room-size aware.
"""

from __future__ import annotations

import math

import numpy as np

from ...camera.focal_estimate import resolve_focal_length
from ...core import raycast
from ..base import GeometryEvidence
from . import vp as floor_vp

#: A detected floor plane must reach at least this share of the floor mask.
MIN_COVERAGE = 0.5


def _plane_normal(horizon_y: float, cy: float, f: float, gain: float = 1.0):
    """The floor normal `constraints.plane_from_horizon` derives, without its depth term."""
    vpy_cam = (horizon_y - cy) / f * gain
    ny = -1.0 / math.sqrt(1.0 + vpy_cam * vpy_cam)
    nz = -vpy_cam * ny
    return 0.0, ny, nz


def detect(room_bgr, floor_bool: np.ndarray, opts, camera_height_mm: float) -> dict:
    """The floor's geometry, as a JSON-ready dict. See the module docstring."""
    h, w = floor_bool.shape
    cx, cy = w / 2.0, h / 2.0

    boundary = floor_vp.extract_boundary(floor_bool, room_bgr.shape, enabled=opts.use_floor_boundary)
    ev = floor_vp.resolve(room_bgr, floor_bool, opts, cx, cy, boundary=boundary)

    f, focal_info = resolve_focal_length(
        image_width=w,
        exif_focal_px=opts.exif_focal_px,
        vp1=ev.vp1_raw,
        vp2=ev.vp2_raw,
        principal_point=(cx, cy),
        fallback_focal=opts.focal_length,
        auto=opts.auto_focal_length,
    )

    info = ev.info or {}
    geometry = {
        "canvas": [int(w), int(h)],
        "principal_point": [cx, cy],
        "focal_px": float(f),
        "focal_source": focal_info.get("source"),
        "hfov_deg": float(focal_info.get("hfov_deg", 0.0)),
        "status": "rejected",
        "vanishing_points": {
            "source": ev.source,
            "confidence": float(ev.confidence),
            "threshold": float(opts.vp_confidence_threshold),
            "depth_vp": None if ev.vp_x is None else [float(ev.vp_x), float(ev.vp_y)],
            "horizon_y": None if ev.vp_horizon_y is None else float(ev.vp_horizon_y),
            "vp1": None if ev.vp1_raw is None else [float(v) for v in ev.vp1_raw],
            "vp2": None if ev.vp2_raw is None else [float(v) for v in ev.vp2_raw],
            "stage": info.get("stage"),
            "floor_lines": info.get("floor_lines"),
            "boundary_lines": info.get("boundary_lines"),
            "total_lines": info.get("total_lines"),
            "confidence_terms": info.get("confidence_terms"),
        },
        "boundary": None if boundary is None else {
            "corners": [[float(x), float(y)] for x, y in boundary.get("corners", [])],
            "visible_fraction": float(boundary.get("visible_fraction", 0.0)),
            "floor_coverage": float(boundary.get("floor_coverage", 0.0)),
        },
    }

    if ev.source != "auto" or ev.vp_horizon_y is None:
        return geometry

    # ---- what the accepted vanishing point implies ----
    a, b, c = _plane_normal(ev.vp_horizon_y, cy, f, opts.depth_perspective_gain)
    plane = (a, b, c, float(camera_height_mm))  # |d| = camera height, so units are mm

    X, Y, Z, valid = raycast.intersect_rays_with_plane(h, w, cx, cy, f, plane)
    coverage = float((valid & floor_bool).sum()) / max(float(floor_bool.sum()), 1.0)

    e_u, e_v = raycast.build_plane_basis(a, b, c)
    baseline = raycast.vanishing_point_angle(ev.vp_x, ev.vp_y, cx, cy, f, e_u, e_v)

    geometry.update({
        "status": "detected" if coverage >= MIN_COVERAGE else "out-of-view",
        "coverage": round(coverage, 4),
        "camera": {
            "pitch_deg": math.degrees(math.atan2(cy - ev.vp_horizon_y, f)),
            "height_mm": float(camera_height_mm),
            # Where "parallel to the walls" points, relative to the camera's
            # own axes: the grid's baseline rotation.
            "floor_direction_deg": math.degrees(baseline),
        },
        "plane": [float(a), float(b), float(c), float(camera_height_mm)],
    })

    seen = valid & floor_bool
    if seen.any():
        u, v = raycast.project_to_plane_uv(X, Y, Z, e_u, e_v)
        cos_r, sin_r = math.cos(baseline), math.sin(baseline)
        ur, vr = u * cos_r - v * sin_r, u * sin_r + v * cos_r
        geometry["visible_floor_mm"] = [
            float(ur[seen].max() - ur[seen].min()),
            float(vr[seen].max() - vr[seen].min()),
        ]

    return geometry


def evidence_from(geometry: dict, scale: float = 1.0) -> GeometryEvidence:
    """
    The GeometryEvidence the floor pipeline would have detected, from stored
    geometry. `scale` maps the stored canvas onto the render's.
    """
    vps = geometry["vanishing_points"]
    s = float(scale)
    point = lambda p: None if p is None else (p[0] * s, p[1] * s)  # noqa: E731

    ev = GeometryEvidence(
        vp_x=None if vps["depth_vp"] is None else vps["depth_vp"][0] * s,
        vp_y=None if vps["depth_vp"] is None else vps["depth_vp"][1] * s,
        vp_horizon_y=None if vps["horizon_y"] is None else vps["horizon_y"] * s,
        vp1_raw=point(vps["vp1"]),
        vp2_raw=point(vps["vp2"]),
        confidence=float(vps["confidence"]),
        source=vps["source"],
    )
    ev.info = {"stage": vps.get("stage"), "from": "stored floor geometry"}
    return ev


# ---------------------------------------------------------------------------
# When the floor's own lines are not enough: the room's structure.
#
# A glossy or plain floor gives the floor-only detector little to work with,
# and its confidence gate rightly rejects it. The render then fell back to the
# backend's estimated camera, whose pitch is solved so that a box of the TYPED
# room length fits the photo -- so the tile count followed the typed number,
# not the room. Measured on a bedroom: with the default 10 ft the camera was
# pitched 23.6 deg down, the horizon put at y = -67, and a 1200 x 1800 mm tile
# covered a quarter of the floor; the walls' own lines put the horizon at
# y = 293, near the middle of the frame, where the photo shows it.
#
# Every wall's horizontal lines (skirting, ceiling edge, window heads, the
# floor/wall junction) converge on the room's horizontal vanishing points --
# `wall.vp.room_horizontal_vps` already finds them. The floor shares the same
# horizon, so the one of those points that is the room's depth direction
# fixes the floor exactly as an accepted floor vanishing point would. It is
# used only when it passes the checks a real horizon must pass.

#: Share of the floor mask that must lie below the structural horizon.
STRUCTURE_MIN_COVERAGE = 0.95

#: Plausible camera pitch (degrees, + = looking down) for a room photograph.
STRUCTURE_PITCH_RANGE = (-15.0, 45.0)

#: Floor lines agree with the candidate vanishing point within this angle.
STRUCTURE_AGREE_DEG = 3.0


def _floor_line_support(room_bgr, floor_bool, vp, min_len):
    """Length share of the floor's own non-vertical lines that point at `vp`."""
    import cv2
    from ...camera.vp_from_mask import detect_lines_lsd, filter_lines_by_floor_mask

    gray = cv2.cvtColor(room_bgr, cv2.COLOR_BGR2GRAY)
    lines = detect_lines_lsd(gray)
    if lines is None or not len(lines):
        return 0.0, 0
    lines = filter_lines_by_floor_mask(np.asarray(lines, np.float64), floor_bool, min_length=min_len)
    if not len(lines):
        return 0.0, 0
    d = lines[:, 2:4] - lines[:, 0:2]
    length = np.hypot(d[:, 0], d[:, 1])
    mid = (lines[:, 0:2] + lines[:, 2:4]) / 2.0
    to_vp = np.asarray(vp, float)[None, :] - mid
    cos = np.abs((d * to_vp).sum(axis=1)) / (length * np.linalg.norm(to_vp, axis=1) + 1e-9)
    agree = np.degrees(np.arccos(np.clip(cos, 0.0, 1.0))) < STRUCTURE_AGREE_DEG
    return float(length[agree].sum() / max(length.sum(), 1e-9)), int(len(lines))


def from_room_structure(room_bgr, floor_bool: np.ndarray, room_vps, rejected: dict,
                        camera_height_mm: float, depth_perspective_gain: float = 1.0,
                        junction_vp: dict | None = None) -> dict | None:
    """
    The floor's geometry from the room's structural horizontal vanishing
    points, or None when none of them passes. `rejected` is the floor-only
    detection this replaces; its canvas, focal length and boundary are kept and
    the rejected detection is recorded alongside.
    """
    points = [(tuple(vp), None) for vp in (room_vps or [])]
    if junction_vp:
        # The walls' own floor junctions agreeing on one point: their consensus
        # is the evidence, as floor lines are for the others (bare floors have none).
        points.append((tuple(junction_vp["point"]), float(junction_vp["share"])))
    if not points:
        return None
    h, w = floor_bool.shape
    cx, cy = w / 2.0, h / 2.0
    f = float(rejected["focal_px"])
    floor_px = max(float(floor_bool.sum()), 1.0)
    min_len = 0.03 * float(np.hypot(h, w))

    candidates = []
    for vp, given_support in points:
        x, y = float(vp[0]), float(vp[1])
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        pitch = math.degrees(math.atan2(cy - y, f))   # + = looking down, as in `detect`
        a, b, c = _plane_normal(y, cy, f, depth_perspective_gain)
        plane = (a, b, c, float(camera_height_mm))
        X, Y, Z, valid = raycast.intersect_rays_with_plane(h, w, cx, cy, f, plane)
        coverage = float((valid & floor_bool).sum()) / floor_px
        support, n_lines = _floor_line_support(room_bgr, floor_bool, (x, y), min_len)
        if given_support is not None:
            support = max(support, given_support)
        candidates.append({
            "vp": (x, y), "pitch_deg": pitch, "coverage": coverage, "support": support,
            "floor_lines": n_lines, "plane": plane, "valid": valid, "XYZ": (X, Y, Z),
            # The depth direction is the one seen inside (or nearest) the frame.
            "off_frame": max(0.0, abs(x - cx) - cx) / w,
        })

    ok = [c for c in candidates
          if c["coverage"] >= STRUCTURE_MIN_COVERAGE
          and STRUCTURE_PITCH_RANGE[0] <= c["pitch_deg"] <= STRUCTURE_PITCH_RANGE[1]]
    if not ok:
        return None
    best = min(ok, key=lambda c: (c["off_frame"], -c["support"]))
    # A depth direction the floor's own lines contradict is not taken: some
    # floor lines (tile joints, the floor/wall junction) must point at it.
    if best["support"] <= 0.0:
        return None

    x, y = best["vp"]
    a, b, c, _ = best["plane"]
    e_u, e_v = raycast.build_plane_basis(a, b, c)
    baseline = raycast.vanishing_point_angle(x, y, cx, cy, f, e_u, e_v)
    others = [c2["vp"] for c2 in candidates if c2 is not best]

    geometry = dict(rejected)
    geometry.update({
        "status": "detected",
        "coverage": round(best["coverage"], 4),
        "vanishing_points": {
            **rejected["vanishing_points"],
            "source": "auto",            # replayed exactly as an accepted detection
            "origin": "room-structure",  # ... but it came from the walls' lines
            "confidence": round(best["support"], 4),
            "depth_vp": [x, y],
            "horizon_y": y,
            "vp1": [x, y],
            "vp2": list(others[0]) if others else None,
            "floor_only_detection": {
                k: rejected["vanishing_points"].get(k)
                for k in ("confidence", "threshold", "stage", "confidence_terms")
            },
            "structure_check": {
                "coverage": round(best["coverage"], 4),
                "pitch_deg": round(best["pitch_deg"], 2),
                "floor_line_support": round(best["support"], 4),
                "floor_lines": best["floor_lines"],
            },
        },
        "camera": {
            "pitch_deg": math.degrees(math.atan2(cy - y, f)),
            "height_mm": float(camera_height_mm),
            "floor_direction_deg": math.degrees(baseline),
        },
        "plane": [float(a), float(b), float(c), float(camera_height_mm)],
    })

    X, Y, Z = best["XYZ"]
    seen = best["valid"] & floor_bool
    if seen.any():
        u, v = raycast.project_to_plane_uv(X, Y, Z, e_u, e_v)
        cos_r, sin_r = math.cos(baseline), math.sin(baseline)
        ur, vr = u * cos_r - v * sin_r, u * sin_r + v * cos_r
        geometry["visible_floor_mm"] = [
            float(ur[seen].max() - ur[seen].min()),
            float(vr[seen].max() - vr[seen].min()),
        ]
    return geometry
