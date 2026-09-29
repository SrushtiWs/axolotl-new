"""
Geometry debug overlay -- validation only, never part of a production render.

Draws on the clean room:
  * the structural lines RoomGeometry used: each wall's floor junction (green)
    and top edge (cyan), extended across its wall;
  * the horizon (white) and VP_X / VP_Y / VP_Z (arrows when off the image);
  * detected room corners (green circles) vs projected RoomGeometry corners
    (red crosses) with the pixel error, and the projected room box edges;
  * a panel: room size and sources, scale, camera, status.
"""

from __future__ import annotations

import cv2
import numpy as np

BOX_EDGES = [
    ("floor_front_left", "floor_front_right"), ("floor_front_right", "floor_back_right"),
    ("floor_back_right", "floor_back_left"), ("floor_back_left", "floor_front_left"),
    ("ceiling_front_left", "ceiling_front_right"), ("ceiling_front_right", "ceiling_back_right"),
    ("ceiling_back_right", "ceiling_back_left"), ("ceiling_back_left", "ceiling_front_left"),
    ("floor_front_left", "ceiling_front_left"), ("floor_front_right", "ceiling_front_right"),
    ("floor_back_left", "ceiling_back_left"), ("floor_back_right", "ceiling_back_right"),
]


def _draw_line_eq(img, l, color, thickness=1):
    h, w = img.shape[:2]
    a, b, c = l
    pts = []
    if abs(b) > 1e-9:
        pts = [(0, -c / b), (w, -(a * w + c) / b)]
    elif abs(a) > 1e-9:
        pts = [(-c / a, 0), (-c / a, h)]
    if pts:
        cv2.line(img, tuple(int(round(v)) for v in pts[0]), tuple(int(round(v)) for v in pts[1]),
                 color, thickness, cv2.LINE_AA)


def _vp_marker(img, vp, color, label):
    h, w = img.shape[:2]
    v = np.array(vp["homogeneous"], float)
    c = np.array([w / 2.0, h / 2.0])
    if not vp["at_infinity"] and vp["image"] is not None:
        x, y = vp["image"]
        if -0.1 * w < x < 1.1 * w and -0.1 * h < y < 1.1 * h:
            cv2.drawMarker(img, (int(x), int(y)), color, cv2.MARKER_TILTED_CROSS, 28, 3)
            cv2.putText(img, label, (int(x) + 10, int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            return
        d = np.array([x, y]) - c
    else:
        d = v[:2]
    d = d / (np.linalg.norm(d) or 1.0) * min(w, h) * 0.22
    tip = c + d
    cv2.arrowedLine(img, tuple(int(t) for t in c), tuple(int(t) for t in tip), color, 3, tipLength=0.2)
    cv2.putText(img, label + " (off image)", tuple(int(t) for t in tip + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def draw(clean_bgr: np.ndarray, fr: dict, room: dict) -> np.ndarray:
    img = (clean_bgr.astype(np.float32) * 0.6).astype(np.uint8)
    h, w = img.shape[:2]

    for e in (fr.get("walls") or {}).values():
        if e.get("junction_line"):
            _draw_line_eq(img, e["junction_line"], (0, 220, 0), 1)
        if e.get("top_line"):
            _draw_line_eq(img, e["top_line"], (255, 220, 0), 1)

    vps = room.get("vanishing_points") or {}
    if vps and room.get("camera"):
        K = np.array(room["camera"]["K"], float)
        Y = np.array(room["camera"]["R_room_to_camera"], float)[:, 1]
        _draw_line_eq(img, np.linalg.inv(K).T @ Y, (255, 255, 255), 1)       # horizon
        _vp_marker(img, vps["VP_Z"], (0, 0, 255), "VP_Z depth")
        _vp_marker(img, vps["VP_X"], (255, 128, 0), "VP_X width")
        _vp_marker(img, vps["VP_Y"], (255, 0, 255), "VP_Y up")

    proj = room.get("projected_corners_2d") or {}
    for a, b in BOX_EDGES:
        pa, pb = proj.get(a), proj.get(b)
        if pa and pb:
            cv2.line(img, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), (60, 60, 255), 2, cv2.LINE_AA)
    for name, det in (room.get("detected_corners_2d") or {}).items():
        cv2.circle(img, (int(det[0]), int(det[1])), 9, (0, 255, 0), 2)
        pr = proj.get(name)
        if pr:
            cv2.drawMarker(img, (int(pr[0]), int(pr[1])), (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
            err = (room.get("reprojection_error_px") or {}).get(name)
            if err is not None:
                cv2.putText(img, f"{err:.1f}px", (int(det[0]) + 12, int(det[1]) + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    cam = room.get("camera") or {}
    fmt = lambda v, n=0: "-" if v is None else f"{v:.{n}f}"  # noqa: E731
    lines = [
        f"status {room.get('status')}   confidence {room.get('confidence')}",
        f"W {fmt(room.get('width_mm'))} mm ({room.get('width_source')})  "
        f"L {fmt(room.get('length_mm'))} mm ({room.get('length_source')}{', lower bound' if room.get('length_is_lower_bound') else ''})  "
        f"H {fmt(room.get('height_mm'))} mm ({room.get('height_source')})",
        f"camera height {fmt(room.get('camera_height_mm'))} mm ({room.get('camera_height_source')})",
        f"f {fmt(cam.get('focal_px'))} px  pitch {fmt(cam.get('pitch_deg'), 1)}  yaw {fmt(cam.get('yaw_deg'), 1)}  "
        f"roll {fmt(cam.get('roll_deg'), 1)} ({cam.get('roll_source')})",
        f"reprojection mean {fmt(room.get('reprojection_error_mean_px'), 1)} px  max {fmt(room.get('reprojection_error_max_px'), 1)} px",
        f"residuals {room.get('residuals')}",
    ]
    panel_h = 22 * len(lines) + 12
    cv2.rectangle(img, (0, h - panel_h), (w, h), (0, 0, 0), -1)
    for i, t in enumerate(lines):
        cv2.putText(img, t, (10, h - panel_h + 24 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img
