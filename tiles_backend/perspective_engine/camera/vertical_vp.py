"""
VP_Y: the vertical vanishing point, from the room's plumb structural lines.

Wall corners, door and window jambs and the sides of built-in units are
vertical in the world, so their images meet at one point -- at infinity when
the camera is level, above or below the frame when it is pitched. Its
direction is gravity in camera coordinates, which fixes the camera's roll.

Reuses the existing line detector (camera.vp_from_mask.detect_lines_lsd) and
the existing homogeneous RANSAC fit (surface.wall.geometry._fit), so a point
at infinity is representable and no second VP implementation exists.

A result is marked `reliable` only with enough agreeing lines; otherwise the
caller keeps its existing behaviour (roll 0, gravity from the floor plane).
"""

from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np

from .vp_from_mask import detect_lines_lsd

#: A line within this of image-vertical is a candidate plumb line.
NEAR_VERTICAL_DEG = 20.0

#: Candidate lines shorter than this share of the image diagonal are noise.
MIN_LENGTH_FRACTION = 0.04

#: Share of a line's length that must lie on the (grown) wall mask: structure,
#: not furniture legs or curtain folds in front of the room.
ON_WALL_SHARE = 0.6

#: Agreeing lines, and their length share, for the result to be relied on.
MIN_INLIERS = 4
MIN_SUPPORT = 0.5

#: Camera roll is APPLIED (wall tile grids, RoomGeometry's camera) only when
#: VP_Y is reliable and its confidence -- support x min(1, inliers / 8) -- is
#: at least this: 80% of the plumb-line length agreeing, over 7+ lines.
#: Below it, roll stays 0 and the reason is reported.
HIGH_CONFIDENCE = 0.8


def roll_gate(vertical: Optional[dict]) -> tuple[bool, str]:
    """(apply roll?, why) for a detect() result; the one gate every roll use shares."""
    if not vertical:
        return False, "no VP_Y for this room"
    conf = float(vertical.get("confidence") or 0.0)
    if not vertical.get("reliable"):
        return False, f"VP_Y not reliable ({vertical.get('inliers', 0)} agreeing lines)"
    if conf < HIGH_CONFIDENCE:
        return False, f"VP_Y confidence {conf:.2f} < {HIGH_CONFIDENCE:.2f}"
    return True, f"VP_Y confidence {conf:.2f} >= {HIGH_CONFIDENCE:.2f}"


def _share_on(mask: np.ndarray, seg) -> float:
    h, w = mask.shape
    t = np.linspace(0.0, 1.0, 16)
    xs = np.clip(np.round(seg[0] + t * (seg[2] - seg[0])).astype(int), 0, w - 1)
    ys = np.clip(np.round(seg[1] + t * (seg[3] - seg[1])).astype(int), 0, h - 1)
    return float(mask[ys, xs].mean())


def detect(room_bgr, wall_mask: np.ndarray, f: float, cx: float, cy: float) -> dict:
    """
    {
      "reliable": bool, "confidence": 0..1, "lines": n, "inliers": n,
      "homogeneous": [x, y, w] | None, "image": [x, y] | None, "at_infinity": bool,
      "up_camera": [x, y, z] | None,   # gravity, unit, camera frame (y down)
      "roll_deg": float | None, "pitch_deg": float | None,
    }
    """
    from ..surface.wall.geometry import _fit

    out = {"reliable": False, "confidence": 0.0, "lines": 0, "inliers": 0,
           "homogeneous": None, "image": None, "at_infinity": None,
           "up_camera": None, "roll_deg": None, "pitch_deg": None}

    h, w = wall_mask.shape
    gray = cv2.cvtColor(room_bgr, cv2.COLOR_BGR2GRAY)
    lines = detect_lines_lsd(gray)
    if lines is None or not len(lines):
        return out
    lines = np.asarray(lines, np.float64).reshape(-1, 4)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    grown = cv2.dilate(wall_mask.astype(np.uint8), k).astype(bool)
    min_len = MIN_LENGTH_FRACTION * float(np.hypot(h, w))

    keep = []
    for s in lines:
        dx, dy = s[2] - s[0], s[3] - s[1]
        if math.hypot(dx, dy) < min_len:
            continue
        if math.degrees(math.atan2(abs(dx), abs(dy))) > NEAR_VERTICAL_DEG:
            continue
        if _share_on(grown, s) < ON_WALL_SHARE:
            continue
        keep.append(s)
    out["lines"] = len(keep)
    if len(keep) < MIN_INLIERS:
        return out

    segs = np.array(keep)
    v, inliers = _fit(segs, np.ones(len(segs)))
    if v is None:
        return out

    lengths = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    support = float(lengths[inliers].sum() / lengths.sum())
    n_in = int(inliers.sum())

    D = np.array([(v[0] - cx * v[2]) / f, (v[1] - cy * v[2]) / f, v[2]])
    D /= np.linalg.norm(D) or 1.0
    if D[1] > 0:                       # gravity points to the image top (-y)
        D = -D

    at_inf = abs(v[2]) < 1e-6 * max(abs(v[0]), abs(v[1]), 1.0)
    out.update({
        "inliers": n_in,
        "support": round(support, 3),
        "homogeneous": [float(x) for x in v],
        "at_infinity": bool(at_inf),
        "image": None if at_inf else [float(v[0] / v[2]), float(v[1] / v[2])],
        "up_camera": [float(x) for x in D],
        # roll: gravity's lean in the image plane; pitch: its tilt toward the lens
        "roll_deg": math.degrees(math.atan2(D[0], -D[1])),
        "pitch_deg": math.degrees(math.asin(max(-1.0, min(1.0, -D[2])))),
        "confidence": round(support * min(1.0, n_in / 8.0), 3),
        "reliable": bool(n_in >= MIN_INLIERS and support >= MIN_SUPPORT),
    })
    return out
