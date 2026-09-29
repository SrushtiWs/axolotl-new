"""
Wall-specific vanishing points.

Reuses the shared line infrastructure -- detect_lines_lsd and
filter_lines_by_floor_mask from camera/vp_from_mask (the latter is
mask-generic despite its name), and estimate_vp_ransac from
camera/vanishing_points. What is wall-specific is the FILTERING and the
INTERPRETATION:

  * a floor wants two horizontal directions and a horizon;
  * a wall wants the plumb direction (vertical edges: door jambs, corners,
    window reveals) and the one horizontal direction that runs ALONG the wall
    (skirting, picture rail, cornice, window heads).

The plumb VP is the useful one. On a wall the tile grid should be square to
gravity, so the vertical lines tell us how much the CAMERA is rolled, and that
roll is the only rotation the grid should pick up from the image. The
along-wall VP is computed too, but it is used as corroboration rather than
allowed to steer the grid -- see constraints.py for why.
"""

import math

import numpy as np
import cv2

from ...camera.vp_from_mask import (
    compute_vanishing_points,
    detect_lines_lsd,
    filter_lines_by_floor_mask,
)
from ..base import GeometryEvidence

#: Fixed seed for the VP RANSAC.
#:
#: Without it the wall plane was different on every render -- measured at
#: n=(-0.42,0,-0.91), then (-0.98,0,-0.19), then (-0.82,0,-0.57) for the SAME
#: wall across three consecutive renders that differed only in tile rotation.
#: A wall's orientation cannot depend on which tile angle the user picked.
VP_SEED = 1337

#: Line-filter tolerance for walls. filter_lines_by_floor_mask defaults to
#: requiring 80% of a segment inside the mask, which is right for a floor
#: (where a line crossing the region is furniture) but wrong for a wall, whose
#: most informative lines -- skirting, cornice, window reveals -- run along the
#: mask BOUNDARY and lose samples to both sides.
WALL_INSIDE_RATIO = 0.55

#: A segment counts as plumb when its angle from image-vertical is under this.
VERTICAL_TOLERANCE_DEG = 25.0
#: ...and as along-wall when its angle from image-horizontal is under this.
HORIZONTAL_TOLERANCE_DEG = 35.0
#: Minimum segment length, as a fraction of image diagonal.
MIN_LINE_FRACTION = 0.04


def _split_by_orientation(lines):
    """Partition segments into (near-vertical, near-horizontal, other)."""
    vert, horiz = [], []
    for x1, y1, x2, y2 in lines:
        dx, dy = float(x2 - x1), float(y2 - y1)
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            continue
        angle_from_vertical = math.degrees(math.atan2(abs(dx), abs(dy)))
        angle_from_horizontal = math.degrees(math.atan2(abs(dy), abs(dx)))
        if angle_from_vertical <= VERTICAL_TOLERANCE_DEG:
            vert.append((x1, y1, x2, y2))
        elif angle_from_horizontal <= HORIZONTAL_TOLERANCE_DEG:
            horiz.append((x1, y1, x2, y2))
    return (np.array(vert, dtype=np.float32).reshape(-1, 4),
            np.array(horiz, dtype=np.float32).reshape(-1, 4))


def _camera_roll_from_verticals(vert_lines) -> float:
    """
    Camera roll, in radians, from the plumb lines' median image angle.

    A plumb line in the world images as a line whose deviation from image
    vertical IS the camera's roll (plus a perspective term that vanishes near
    the image centre). The median is used rather than a VP fit because roll is
    a single robust scalar and a vertical VP sits near infinity for a level
    camera, where a RANSAC fit is numerically worst.
    """
    if vert_lines is None or len(vert_lines) == 0:
        return 0.0
    angles = []
    for x1, y1, x2, y2 in vert_lines:
        dx, dy = float(x2 - x1), float(y2 - y1)
        if dy < 0:
            dx, dy = -dx, -dy
        if abs(dy) < 1e-6:
            continue
        angles.append(math.atan2(dx, dy))
    if not angles:
        return 0.0
    return float(np.median(angles))


def _wall_lines(room_bgr, mask, min_len):
    """Line segments running ALONG the wall region, boundary lines included."""
    gray = cv2.cvtColor(room_bgr, cv2.COLOR_BGR2GRAY)
    lines = detect_lines_lsd(gray)
    if lines is None or len(lines) == 0:
        return np.empty((0, 4), np.float32), 0

    h, w = mask.shape[:2]
    grow = max(2, int(round(0.008 * np.hypot(h, w))))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
    grown = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)

    on_wall = filter_lines_by_floor_mask(
        lines, grown, min_length=min_len, inside_ratio=WALL_INSIDE_RATIO
    )
    if on_wall is None:
        on_wall = np.empty((0, 4), np.float32)
    return on_wall, int(len(lines))


def room_horizontal_vps(room_bgr, wall_mask):
    """
    The room's horizontal vanishing directions, seen from the wall region.

    A rectangular room has exactly TWO horizontal directions, and every wall in
    it contains one of them. Recovering both once, from all the wall pixels
    together, is far more robust than asking each wall fragment for its own
    vanishing point -- a fragment often carries no horizontal line at all
    (measured: 0 and 1 horizontal lines on two of three walls), which is why
    per-instance detection kept returning nothing.

    Clustering and fitting are the shared ones from camera/vp_from_mask, run
    with a fixed rng so the result is identical on every render.

    Returns a list of up to two (vx, vy) points, and an info dict.
    """
    h, w = wall_mask.shape[:2]
    min_len = MIN_LINE_FRACTION * float(np.hypot(h, w))
    on_wall, total = _wall_lines(room_bgr, wall_mask, min_len)

    _vert, horiz = _split_by_orientation(on_wall)
    info = {"total_lines": total, "wall_lines": int(len(on_wall)),
            "horizontal_lines": int(len(horiz))}

    if len(horiz) < 4:
        info["stage"] = "too-few-horizontal-lines"
        return [], info

    rng = np.random.default_rng(VP_SEED)
    try:
        r1, r2 = compute_vanishing_points(horiz, iterations=300, threshold=3.0, rng=rng)
    except Exception as exc:                      # noqa: BLE001 - detection is best-effort
        info["stage"] = f"vp-error:{exc}"
        return [], info

    vps = []
    for r in (r1, r2):
        if not r:
            continue
        vp = r[0]
        if vp is None:
            continue
        vx, vy = float(vp[0]), float(vp[1])
        if not (np.isfinite(vx) and np.isfinite(vy)):
            continue
        vps.append((vx, vy))

    info["stage"] = "ok" if vps else "no-vp"
    info["vps"] = vps
    return vps, info


def resolve(room_bgr, instance_mask, opts, cx, cy) -> GeometryEvidence:
    """
    Wall direction evidence for ONE wall instance.

    Confidence is deliberately conservative: it rises with the number of plumb
    segments actually found on this wall and how tightly they agree. Anything
    below opts.vp_confidence_threshold is reported but not acted on, exactly as
    the floor path treats a weak detection.
    """
    ev = GeometryEvidence(source="wall-none")

    h, w = instance_mask.shape[:2]
    gray = cv2.cvtColor(room_bgr, cv2.COLOR_BGR2GRAY)
    lines = detect_lines_lsd(gray)
    if lines is None or len(lines) == 0:
        ev.info = {"stage": "no-lines", "total_lines": 0}
        return ev

    min_len = MIN_LINE_FRACTION * float(np.hypot(h, w))

    # Filter against a DILATED mask.
    #
    # The most informative lines on a wall are its junctions -- the skirting
    # where it meets the floor, the cornice where it meets the ceiling, the
    # reveal of a window. Every one of them lies exactly ON the mask boundary,
    # and filter_lines_by_floor_mask requires 80% of a segment's samples to
    # fall INSIDE the mask, so all of them were being thrown away: a real room
    # yielded 3, 0 and 1 horizontal lines for its three walls, which is not
    # enough to fit an along-wall vanishing point at all. Growing the mask by a
    # little under 1% of the diagonal keeps the junctions while still rejecting
    # lines that merely cross the wall.
    grow = max(2, int(round(0.008 * np.hypot(h, w))))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
    grown = cv2.dilate(instance_mask.astype(np.uint8), kernel).astype(bool)

    # Mask-generic despite the floor-era name: keeps only segments lying ALONG
    # this wall instead of merely crossing it.
    on_wall = filter_lines_by_floor_mask(lines, grown, min_length=min_len)
    if on_wall is None or len(on_wall) == 0:
        ev.info = {"stage": "no-lines-on-wall", "total_lines": int(len(lines))}
        return ev

    vert, horiz = _split_by_orientation(on_wall)
    roll_rad = _camera_roll_from_verticals(vert)

    # Along-wall VP: corroboration only.
    along_vp = None
    if len(horiz) >= 2:
        try:
            vp = estimate_vp_ransac(horiz, iterations=200, threshold=3.0)
            if vp is not None and np.all(np.isfinite(vp[:2])):
                along_vp = (float(vp[0]), float(vp[1]))
        except Exception:
            along_vp = None

    # Confidence: enough plumb evidence, and consistent.
    if len(vert) >= 3:
        angles = []
        for x1, y1, x2, y2 in vert:
            dx, dy = float(x2 - x1), float(y2 - y1)
            if dy < 0:
                dx, dy = -dx, -dy
            if abs(dy) > 1e-6:
                angles.append(math.atan2(dx, dy))
        spread = float(np.std(angles)) if angles else math.pi
        count_term = min(1.0, len(vert) / 8.0)
        agree_term = math.exp(-spread / math.radians(6.0))
        ev.confidence = float(np.clip(0.5 * count_term + 0.5 * agree_term, 0.0, 1.0))
        ev.source = "wall-plumb"
    else:
        ev.confidence = 0.0
        ev.source = "wall-insufficient-verticals"

    if along_vp is not None:
        ev.vp_x, ev.vp_y = along_vp
        ev.vp1_raw = along_vp

    ev.info = {
        "stage": "ok",
        "total_lines": int(len(lines)),
        "wall_lines": int(len(on_wall)),
        "vertical_lines": int(len(vert)),
        "horizontal_lines": int(len(horiz)),
        "camera_roll_deg": math.degrees(roll_rad),
        "along_wall_vp": along_vp,
    }
    # Stashed for constraints.py, which decides how much of it to trust.
    ev.info["roll_rad"] = roll_rad
    return ev
