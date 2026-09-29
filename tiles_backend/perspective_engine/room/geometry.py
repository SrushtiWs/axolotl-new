"""
RoomGeometry: one metric room, one camera, one scale.

Coordinate system (room frame, millimetres):

    X = room width   (left wall -> right wall, runs left-to-right in the image)
    Y = up           (floor Y = 0, ceiling Y = H)
    Z = room depth   (front wall Z = 0 -> back wall Z = L, away from the camera)

Two stages, so the image is analysed once and the typed room size is applied
per request:

  frame(...)   from the image only, in units of the CAMERA HEIGHT (h = 1):
               gravity (the floor plane), the X/Z axes (the floor's depth
               vanishing point), each wall's position from its own floor
               junction, the ceiling height from the walls' top edges, and the
               detected room corners. Nothing metric is assumed yet.

  solve(frame, dims_mm)
               ONE scale s = camera height in mm, fitted to the user's width
               and height where the image measures them; the 1500 mm camera
               prior only when there is no metric input. Builds the 3D box and
               camera, projects the corners back into the image, measures the
               reprojection error, and reports CONFLICT when the typed size
               and the photo disagree -- the typed values are never changed.

Everything reused: the floor plane/horizon (surface/floor/geometry), wall
directions (surface/wall/*), junction pixels (surface/wall/junction_layout),
focal length (camera/focal_estimate), VP_Y (camera/vertical_vp).
"""

from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np

MM_PER_FOOT = 304.8

#: The camera-height prior used only when nothing metric is known.
CAMERA_HEIGHT_PRIOR_MM = 1500.0

#: A wall normal within this of a room axis is that axis's wall.
AXIS_TOLERANCE_DEG = 20.0

#: Junction / top-edge columns a wall needs before it is measured.
MIN_COLUMNS = 15

#: Share of a wall's columns that must show its floor junction for the wall
#: to be placed from it.
MIN_JUNCTION_SHARE = 0.15

#: Scale candidates disagreeing with the fit by more than this -> CONFLICT.
CONFLICT_TOLERANCE = 0.12

#: Plausible camera heights for a room photograph (mm).
CAMERA_HEIGHT_RANGE_MM = (700.0, 2600.0)

SOURCES = ("USER_INPUT", "ESTIMATED", "DETECTED", "CONFLICT", "UNKNOWN")


def _unit(v):
    v = np.asarray(v, np.float64)
    return v / (np.linalg.norm(v) or 1.0)


def feet_to_mm(value_ft) -> Optional[float]:
    """The one feet -> millimetres conversion."""
    if value_ft is None:
        return None
    v = float(value_ft)
    return v * MM_PER_FOOT if v > 0 else None


def _rays(xs, ys, f, cx, cy):
    xs = np.asarray(xs, np.float64)
    ys = np.asarray(ys, np.float64)
    return np.stack([(xs - cx) / f, (ys - cy) / f, np.ones_like(xs)], axis=-1)


def _on_plane(rays, n, d):
    """Rays through the origin meeting n.P + d = 0; (points, valid)."""
    den = rays @ n
    t = -d / np.where(np.abs(den) < 1e-12, 1e-12, den)
    return rays * t[..., None], (np.abs(den) > 1e-12) & (t > 0)


def _top_pixels(mask, wall_mask):
    cols = np.where(mask.any(axis=0))[0]
    if not len(cols):
        return np.empty((0, 2))
    top = np.argmax(mask[:, cols], axis=0)
    keep = (top > 2) & ~wall_mask[np.clip(top - 1, 0, mask.shape[0] - 1), cols]
    return np.column_stack([cols[keep], top[keep]]).astype(np.float64)


def _fit_line(points):
    """Image line (a, b, c) through points, robust; None when too few."""
    if len(points) < MIN_COLUMNS:
        return None
    vx, vy, x0, y0 = cv2.fitLine(points.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    return np.cross([x0, y0, 1.0], [x0 + vx, y0 + vy, 1.0])


def _meet(l1, l2):
    if l1 is None or l2 is None:
        return None
    p = np.cross(l1, l2)
    if abs(p[2]) < 1e-9:
        return None
    return [float(p[0] / p[2]), float(p[1] / p[2])]


# --------------------------------------------------------------------------- frame
def frame(floor_geo: dict, walls: list, masks: dict, floor: np.ndarray, wall_mask: np.ndarray,
          f: float, cx: float, cy: float, vertical: dict,
          objects: Optional[np.ndarray] = None) -> dict:
    """
    The room as the image measures it, in camera-height units. JSON-ready.
    `walls` are the stored wall entries (with "index" and "direction"), `masks`
    their pixel masks by index.
    """
    from ..surface.wall.junction_layout import junction_points

    out = {"units": "camera heights (h = 1)", "status": "ok", "walls": {},
           "canvas": [int(floor.shape[1]), int(floor.shape[0])],
           "K": [[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], "vertical_vp": vertical}

    if (floor_geo or {}).get("status") != "detected" or not floor_geo.get("plane"):
        out["status"] = "no-floor-plane"
        return out
    vps = floor_geo.get("vanishing_points") or {}
    depth_vp = vps.get("depth_vp")
    if depth_vp is None:
        out["status"] = "no-depth-vp"
        return out

    up = _unit(floor_geo["plane"][:3])
    if up[1] > 0:
        up = -up
    Zf = np.array([(depth_vp[0] - cx) / f, (depth_vp[1] - cy) / f, 1.0])
    Zf = _unit(Zf - (Zf @ up) * up)
    if Zf[2] < 0:
        Zf = -Zf

    # The room's axes. The walls ARE the room's structure, so the largest wall
    # with a direction (by visible floor junction) fixes them; the floor's depth
    # vanishing point is the fallback, and the disagreement is reported.
    from ..surface.wall.junction_layout import junction_points as _jp
    # Only a wall that FACES the camera (by its own evidence, not by comparison
    # with the floor VP) defines the axes: its normal is -Z.
    best, best_cols = None, 0
    for w in walls:
        d = w.get("direction") or {}
        m = masks.get(int(w["index"]))
        if m is None or not d.get("normal") or d.get("faces") != "front":
            continue
        cols = len(_jp(floor, m, objects))
        if cols > best_cols:
            best, best_cols = d, cols
    Z, axes_source = Zf, "floor depth vanishing point"
    if best is not None and best_cols >= 4 * MIN_COLUMNS:
        nh = np.array(best["normal"], np.float64)
        nh = _unit(nh - (nh @ up) * up)
        Zw = -nh if -nh[2] >= 0 else nh
        Z, axes_source = _unit(Zw), f"back wall facing the camera ({best_cols} junction columns)"
    X = _unit(np.cross(Z, up))          # runs left-to-right in the image
    out["axes_camera"] = {"X": X.tolist(), "Y": up.tolist(), "Z": Z.tolist()}
    out["axes_source"] = axes_source
    out["floor_vp_vs_axes_deg"] = math.degrees(math.acos(min(1.0, abs(float(Zf @ Z)))))
    out["floor_vp_confidence"] = float(vps.get("confidence") or 0.0)
    out["floor_vp_origin"] = vps.get("origin", "floor-lines")

    cos_tol = math.cos(math.radians(AXIS_TOLERANCE_DEG))
    measured = {}
    for w in walls:
        i = int(w["index"])
        d = w.get("direction") or {}
        m = masks.get(i)
        entry = {"id": w["id"], "class": "unknown"}
        if m is None or not d.get("normal"):
            entry["class"] = "no-direction"
            out["walls"][w["id"]] = entry
            continue
        n = _unit(d["normal"])
        if abs(n @ X) >= cos_tol:
            entry["class"] = "side"
        elif abs(n @ Z) >= cos_tol:
            entry["class"] = "back"
        else:
            entry["class"] = "oblique"

        J = junction_points(floor, m, objects)
        span = int(m.any(axis=0).sum())
        entry["junction_share"] = round(len(J) / max(span, 1), 3)
        # A few junction pixels at one end cannot place a whole wall.
        if len(J) >= max(MIN_COLUMNS, MIN_JUNCTION_SHARE * span):
            P, ok = _on_plane(_rays(J[:, 0], J[:, 1], f, cx, cy), up, 1.0)
            P = P[ok]
            if len(P) >= MIN_COLUMNS:
                med = np.median(P, axis=0)
                d_units = -float(n @ med)
                if d_units < 0:
                    n, d_units = -n, -d_units
                entry.update({
                    "junction_columns": int(len(P)),
                    "plane_units": [float(n[0]), float(n[1]), float(n[2]), d_units],
                    "x": float(np.median(P @ X)), "z": float(np.median(P @ Z)),
                    "z_max": float(np.percentile(P @ Z, 95)),
                    "junction_line": [float(v) for v in _fit_line(J)] if _fit_line(J) is not None else None,
                })
                T = _top_pixels(m, wall_mask)
                if len(T) >= MIN_COLUMNS:
                    Q, okq = _on_plane(_rays(T[:, 0], T[:, 1], f, cx, cy), n, d_units)
                    heights = (Q[okq] @ up) + 1.0
                    heights = heights[heights > 0.2]
                    if len(heights) >= MIN_COLUMNS:
                        entry["top_height"] = float(np.median(heights))
                        entry["top_columns"] = int(len(heights))
                        tl = _fit_line(T)
                        entry["top_line"] = [float(v) for v in tl] if tl is not None else None
                measured[w["id"]] = entry
        out["walls"][w["id"]] = entry

    sides = [e for e in measured.values() if e["class"] == "side"]
    backs = [e for e in measured.values() if e["class"] == "back" and e["z"] > 0]
    # The back wall is the farthest wall facing the camera; the side walls of
    # the box are the ones that reach it (a pillar or a recess nearer the
    # camera is not the room's side).
    back = max(backs, key=lambda e: e["z"], default=None)
    reach = (lambda e: e["z_max"]) if back is None else (lambda e: -abs(e["z_max"] - back["z"]))
    left = max((e for e in sides if e["x"] < 0), key=reach, default=None)
    right = max((e for e in sides if e["x"] > 0), key=reach, default=None)
    tops = [(e["top_height"], e["top_columns"]) for e in measured.values() if e.get("top_height")]
    ceiling = None
    if tops:
        hs = np.array([t[0] for t in tops]); ws = np.array([t[1] for t in tops], float)
        order = np.argsort(hs); cum = np.cumsum(ws[order])
        ceiling = float(hs[order][np.searchsorted(cum, cum[-1] / 2)])

    out["box_units"] = {
        "left_x": left["x"] if left else None, "left_wall": left["id"] if left else None,
        "right_x": right["x"] if right else None, "right_wall": right["id"] if right else None,
        "back_z": back["z"] if back else None, "back_wall": back["id"] if back else None,
        "ceiling_y": ceiling, "ceiling_walls": len(tops),
    }

    # Detected room corners in the image: where the chosen walls' own
    # junction / top-edge lines meet.
    corners = {}
    for side, key in ((left, "left"), (right, "right")):
        if side is None or back is None:
            continue
        corners[f"floor_back_{key}"] = _meet(np.array(side["junction_line"]) if side.get("junction_line") else None,
                                             np.array(back["junction_line"]) if back.get("junction_line") else None)
        corners[f"ceiling_back_{key}"] = _meet(np.array(side["top_line"]) if side.get("top_line") else None,
                                               np.array(back["top_line"]) if back.get("top_line") else None)
    out["detected_corners_2d"] = {k: v for k, v in corners.items() if v is not None}
    return out


# --------------------------------------------------------------------------- solve
def _project(K, P):
    P = np.asarray(P, np.float64)
    if P[2] <= 1e-6:
        return None
    p = K @ P
    return [float(p[0] / p[2]), float(p[1] / p[2])]


def solve(fr: dict, dims_mm: Optional[dict] = None) -> dict:
    """
    RoomGeometry for one request. `dims_mm` = {"width", "length", "height"} in
    millimetres (already converted from feet), any of them None.
    """
    dims_mm = {k: (float(v) if v else None) for k, v in (dims_mm or {}).items()}
    room = {
        "coordinate_system": {"X": "room width", "Y": "up", "Z": "room depth", "units": "mm",
                              "origin": "floor, at the left wall and the front wall"},
        "geometry_version": 5,
        "status": "INSUFFICIENT",
        "input_mm": dims_mm,
    }
    if fr.get("status") != "ok" or "axes_camera" not in fr:
        room["reason"] = fr.get("status", "no room frame")
        room["camera_height_mm"] = CAMERA_HEIGHT_PRIOR_MM
        room["camera_height_source"] = "CAMERA_HEIGHT_PRIOR"
        return room

    K = np.array(fr["K"], np.float64)
    X = np.array(fr["axes_camera"]["X"]); Y = np.array(fr["axes_camera"]["Y"]); Z = np.array(fr["axes_camera"]["Z"])
    box = fr["box_units"]
    W_u = (box["right_x"] - box["left_x"]) if box["left_x"] is not None and box["right_x"] is not None else None
    H_u = box["ceiling_y"]
    B_u = box["back_z"]
    W, L, H = dims_mm.get("width"), dims_mm.get("length"), dims_mm.get("height")

    # ---- ONE scale: camera height in mm --------------------------------------
    cands = []
    if W and W_u and W_u > 0:
        cands.append(("width", W / W_u, 1.0))
    if H and H_u and H_u > 0:
        cands.append(("height", H / H_u, 0.8))
    if cands:
        logs = np.array([math.log(c[1]) for c in cands]); ws = np.array([c[2] for c in cands])
        s = float(math.exp((logs * ws).sum() / ws.sum()))
        s_source = "USER_INPUT(" + "+".join(c[0] for c in cands) + ")"
    else:
        s, s_source = CAMERA_HEIGHT_PRIOR_MM, "CAMERA_HEIGHT_PRIOR"
    residuals = {c[0]: round(c[1] / s - 1.0, 4) for c in cands}

    length_conflict = None
    if L and B_u:
        # The camera stands inside the room: the room is at least as deep as
        # the back wall is from the camera.
        needed = B_u * s
        if L < 0.95 * needed:
            length_conflict = round(L / needed - 1.0, 4)
    conflict = any(abs(r) > CONFLICT_TOLERANCE for r in residuals.values()) or length_conflict is not None

    def dim(user, measured_units, lower_bound=False):
        if user:
            return user, "USER_INPUT"
        if measured_units:
            return measured_units * s, ("DETECTED" if cands else "ESTIMATED")
        return None, "UNKNOWN"

    width, width_src = dim(W, W_u)
    height, height_src = dim(H, H_u)
    length, length_src = dim(L, B_u)
    if conflict:
        if "width" in residuals and abs(residuals["width"]) > CONFLICT_TOLERANCE:
            width_src = "CONFLICT"
        if "height" in residuals and abs(residuals["height"]) > CONFLICT_TOLERANCE:
            height_src = "CONFLICT"
        if length_conflict is not None:
            length_src = "CONFLICT"

    # ---- Box placement: room origin in the camera frame (units) --------------
    x0 = box["left_x"]
    if x0 is None and box["right_x"] is not None and width:
        x0 = box["right_x"] - width / s
    z0 = None
    if B_u is not None and length:
        z0 = B_u - length / s
    if z0 is None and B_u is not None:
        z0 = 0.0                                  # front wall unknown: at the camera

    camera = {
        "K": K.tolist(), "focal_px": float(K[0, 0]),
        "principal_point": [float(K[0, 2]), float(K[1, 2])],
        # columns: room X, Y, Z in camera coordinates (x right, y down, z forward)
        "R_room_to_camera": np.column_stack([X, Y, Z]).tolist(),
        "pitch_deg": math.degrees(math.asin(max(-1.0, min(1.0, -Y[2])))),
        "yaw_deg": math.degrees(math.atan2(float(Z[0]), float(Z[2]))),
        "roll_deg": 0.0, "roll_source": "level (no plumb evidence)",
    }
    vert = fr.get("vertical_vp") or {}
    if vert.get("reliable"):
        up_v = np.array(vert["up_camera"])
        camera["roll_deg"] = float(vert["roll_deg"])
        camera["roll_source"] = "vertical vanishing point"
        camera["vertical_vs_floor_deg"] = math.degrees(math.acos(min(1.0, abs(float(up_v @ Y)))))
    if x0 is not None and z0 is not None:
        camera["position_mm"] = [float(-x0 * s), float(s), float(-z0 * s)]
    camera["height_mm"] = s

    # ---- VPs in the room's own terms -----------------------------------------
    def vp_of(D):
        v = K @ D
        at_inf = abs(v[2]) < 1e-9 * max(abs(v[0]), abs(v[1]), 1.0)
        return {"homogeneous": [float(x) for x in v], "at_infinity": bool(at_inf),
                "image": None if at_inf else [float(v[0] / v[2]), float(v[1] / v[2])]}

    vanishing = {"VP_X": vp_of(X), "VP_Z": vp_of(Z), "VP_Y": vp_of(Y),
                 "VP_Y_detected": {k: vert.get(k) for k in ("reliable", "confidence", "image",
                                                            "at_infinity", "inliers", "lines")}}

    # ---- 3D corners, projected -------------------------------------------------
    corners_3d, projected = {}, {}
    if x0 is not None and z0 is not None and width and length and height:
        O = x0 * X - 1.0 * Y + z0 * Z                 # room origin, camera frame, units
        for name, (px, py, pz) in {
            "floor_front_left": (0, 0, 0), "floor_front_right": (width, 0, 0),
            "floor_back_left": (0, 0, length), "floor_back_right": (width, 0, length),
            "ceiling_front_left": (0, height, 0), "ceiling_front_right": (width, height, 0),
            "ceiling_back_left": (0, height, length), "ceiling_back_right": (width, height, length),
        }.items():
            corners_3d[name] = [float(px), float(py), float(pz)]
            P = O + (px * X + py * Y + pz * Z) / s
            projected[name] = _project(K, P)

    errors = {}
    for name, det in (fr.get("detected_corners_2d") or {}).items():
        pr = projected.get(name)
        if pr is not None and det is not None:
            errors[name] = float(math.hypot(pr[0] - det[0], pr[1] - det[1]))

    # ---- Wall planes in mm, camera frame (what the renderer uses) -------------
    planes_mm = {}
    for wid, e in fr["walls"].items():
        if e.get("plane_units"):
            a, b, c, d = e["plane_units"]
            planes_mm[wid] = [a, b, c, d * s]
    floor_plane_mm = [float(Y[0]), float(Y[1]), float(Y[2]), float(s)]

    # ---- Status and confidence ------------------------------------------------
    warnings = []
    lo, hi = CAMERA_HEIGHT_RANGE_MM
    if not lo <= s <= hi:
        warnings.append(f"camera height {s:.0f} mm is outside {lo:.0f}-{hi:.0f} mm")
    measured_dims = sum(v is not None for v in (W_u, H_u, B_u))
    if conflict:
        status = "CONFLICT"
    elif cands:
        status = "OK"
    elif measured_dims:
        status = "ESTIMATED"
    else:
        status = "INSUFFICIENT"
    err_vals = list(errors.values())
    conf = float(fr.get("floor_vp_confidence") or 0.5)
    conf *= 1.0 if cands else 0.5
    if err_vals:
        conf *= 1.0 / (1.0 + max(err_vals) / 20.0)
    if conflict:
        conf *= 0.3

    room.update({
        "status": status,
        "width_mm": width, "length_mm": length, "height_mm": height,
        "width_source": width_src, "length_source": length_src, "height_source": height_src,
        "length_is_lower_bound": bool(not L and B_u is not None),
        "camera_height_mm": s, "camera_height_source": s_source,
        "measured_units": {"width": W_u, "height": H_u, "back_depth": B_u},
        "residuals": {**{f"{k}_residual": v for k, v in residuals.items()},
                      **({"length_residual": length_conflict} if length_conflict is not None else {})},
        "camera": camera,
        "vanishing_points": vanishing,
        "floor_plane_mm": floor_plane_mm,
        "wall_planes_mm": planes_mm,
        "walls": {wid: {k: e.get(k) for k in ("class", "junction_columns", "top_height")}
                  for wid, e in fr["walls"].items()},
        "corners_3d": corners_3d,
        "projected_corners_2d": projected,
        "detected_corners_2d": fr.get("detected_corners_2d") or {},
        "reprojection_error_px": errors,
        "reprojection_error_mean_px": float(np.mean(err_vals)) if err_vals else None,
        "reprojection_error_max_px": float(np.max(err_vals)) if err_vals else None,
        "confidence": round(conf, 3),
        "warnings": warnings,
    })
    return room
