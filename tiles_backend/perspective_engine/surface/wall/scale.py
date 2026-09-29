"""
Millimetres per plane unit, for a wall.

The floor's anchor does not transfer. For a floor, |plane_d| is the camera's
perpendicular distance to the plane, i.e. its height, and a person
photographing a room holds a phone at a fairly predictable height. For a wall,
|plane_d| is the camera's distance to the WALL, which is anything from 0.5 m
in a corridor to 8 m across a hall. There is no prior, so guessing one would
be inventing precision.

Four anchors, tried in order of how much they are actually known:

  1. USER DIMENSIONS. Room height is the strongest: it spans the wall's whole
     v axis, it is the number people know best about their own room, and it is
     already collected by the Room Size panel in the frontend. Room width or
     length anchors the u axis when the wall's full horizontal run is visible.

  2. THE FLOOR IN THE SAME PHOTO. The floor and the wall are reconstructed in
     ONE camera frame, so millimetres-per-unit is a property of that frame,
     not of the surface. If the floor plane resolved its own scale from the
     camera-height anchor, the wall can use the identical factor -- no new
     assumption at all. This is the best automatic option and needs no input.

  3. A DOOR. Internal door leaves are ~2100 mm by building convention, which
     is a far tighter prior than anything about a wall's distance. Uses
     metric_scale.mm_per_unit_from_reference, which exists for exactly this.

  4. Refuse. Raising beats rendering a wall at a silently invented scale,
     where every tile is the wrong size and nothing on screen says so.
"""

import numpy as np

from ...camera.metric_scale import (
    MetricScaleError,
    mm_per_unit_from_reference,
)


def _extent_units(values, valid):
    if values is None or not np.any(valid):
        return 0.0
    vals = values[valid]
    return float(vals.max() - vals.min())


def resolve(instance, opts, *, u_units=None, v_units=None, visible=None,
            floor_mm_per_unit=None, door_height_units=None):
    """
    Returns (mm_per_unit, info).

    u_units / v_units are the wall's plane coordinates BEFORE any millimetre
    conversion; visible is the mask of pixels that both belong to this wall and
    produced a valid ray. floor_mm_per_unit, when supplied, is the factor the
    floor path resolved for the same photograph.
    """
    info = {"source": None, "candidates": []}

    # ---- 1. user-supplied room dimensions ----
    if v_units is not None and visible is not None and opts.room_height_mm:
        v_extent = _extent_units(v_units, visible)
        info["candidates"].append({"source": "room_height", "extent_units": v_extent})
        if v_extent > 1e-6:
            mm_per_unit, ref_info = mm_per_unit_from_reference(
                v_extent, float(opts.room_height_mm)
            )
            if mm_per_unit is not None:
                info.update({
                    "source": "room_height",
                    "room_height_mm": float(opts.room_height_mm),
                    "wall_v_extent_units": v_extent,
                    "reference": ref_info,
                    # The wall mask rarely runs floor-to-ceiling exactly (a
                    # sofa hides the skirting, a cornice is its own class), so
                    # this reads slightly small when the wall is clipped.
                    "caveat": "assumes the visible wall spans the full room height",
                })
                return mm_per_unit, info

    horizontal_mm = opts.room_width_mm or opts.room_length_mm
    if u_units is not None and visible is not None and horizontal_mm:
        u_extent = _extent_units(u_units, visible)
        info["candidates"].append({"source": "room_width", "extent_units": u_extent})
        if u_extent > 1e-6:
            mm_per_unit, ref_info = mm_per_unit_from_reference(
                u_extent, float(horizontal_mm)
            )
            if mm_per_unit is not None:
                info.update({
                    "source": "room_width",
                    "room_width_mm": float(horizontal_mm),
                    "wall_u_extent_units": u_extent,
                    "reference": ref_info,
                    "caveat": "assumes the visible wall spans the full room width",
                })
                return mm_per_unit, info

    # ---- 2. the floor's factor, from the same camera frame ----
    if floor_mm_per_unit is not None and floor_mm_per_unit > 0:
        info.update({
            "source": "floor_frame",
            "mm_per_unit": float(floor_mm_per_unit),
            "note": "floor and wall share one camera frame, so the factor transfers exactly",
        })
        return float(floor_mm_per_unit), info

    # ---- 3. architectural prior: door height ----
    if door_height_units and door_height_units > 1e-6:
        mm_per_unit, ref_info = mm_per_unit_from_reference(
            float(door_height_units), float(opts.door_height_mm)
        )
        if mm_per_unit is not None:
            info.update({
                "source": "door_prior",
                "door_height_mm": float(opts.door_height_mm),
                "reference": ref_info,
            })
            return mm_per_unit, info

    # ---- 4. refuse ----
    raise MetricScaleError(
        "wall scale: no physical anchor available. Supply the room height "
        "(Room Size panel), render with the floor visible so its scale can be "
        "reused, or include a door in frame. Rendering without an anchor would "
        "size every tile arbitrarily."
    )


def door_height_in_units(instance_mask, openings_mask, v_units, valid):
    """
    Vertical extent, in plane units, of the tallest door-like opening touching
    this wall -- the measurement the door prior needs.

    Returns None when no opening is a plausible door: an opening is only used
    when it is clearly taller than wide, which rejects windows and pictures.
    """
    if openings_mask is None or v_units is None:
        return None

    import cv2

    candidate = (openings_mask & _dilate(instance_mask)) & valid
    if not np.any(candidate):
        return None

    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate.astype(np.uint8), connectivity=8
    )
    best = None
    for i in range(1, num):
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        if h < 1.4 * w:      # doors are tall; windows and pictures are not
            continue
        sel = (labels == i) & valid
        if not np.any(sel):
            continue
        extent = float(v_units[sel].max() - v_units[sel].min())
        if extent > 0 and (best is None or extent > best):
            best = extent
    return best


def _dilate(mask_bool, radius: int = 3):
    import cv2
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(mask_bool.astype(np.uint8), kernel).astype(bool)
