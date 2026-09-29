"""
Plane coordinates -> tile grid coordinates.

The invariant this module exists to protect:

    PERSPECTIVE NEVER CHANGES A TILE'S PHYSICAL SIZE.

u and v arrive in plane units, are converted to millimetres ONCE by a scale
factor that came from a physical anchor, and are then divided by
tile_width_mm / tile_height_mm. Rotation happens in millimetre space, before
the division, so an 800x1600 tile is 800x1600 mm at 0 degrees, at 45 degrees,
near the camera and at the far corner. Foreshortening is entirely a property
of the ray-plane intersection that produced u and v; nothing here rescales a
tile to "look right".
"""

import math
from typing import Tuple

import numpy as np

#: Orientations the UI offers. The grid repeats every 180 degrees, so these
#: four cover every distinct lay.
SUPPORTED_ORIENTATIONS = (0, 45, 90, 135)

_AXIS_PROJECTION = {
    0: (1.0, 0.0),
    45: (math.sqrt(0.5), math.sqrt(0.5)),
    90: (0.0, 1.0),
    135: (math.sqrt(0.5), math.sqrt(0.5)),
}


def snap_orientation(deg: float) -> int:
    """
    Fold any angle onto one of 0/45/90/135.

    A tile grid repeats every 180 degrees -- a tile laid at 135 covers a
    surface exactly like one laid at -45 -- so the angle is folded into
    [0, 180) before snapping. Reporting only; the render itself uses the
    caller's exact tile_rotation, which need not be one of the four.
    """
    folded = ((float(deg) % 180.0) + 180.0) % 180.0
    snapped = int(round(folded / 45.0) * 45)
    return 0 if snapped == 180 else snapped


def effective_footprint_mm(tile_w_mm: float, tile_h_mm: float,
                           orientation_deg: float) -> Tuple[float, float]:
    """
    The tile's axis-aligned footprint on the surface at a given orientation:

        across = w*|cos| + h*|sin|
        deep   = w*|sin| + h*|cos|

    At 0 it is the tile; at 90 it is an exact width/height swap; at 45 and 135
    it is the bounding box of the rotated tile, (w + h)/sqrt(2) on both axes.
    The tile itself is unchanged in every case -- this is the box it occupies,
    used for reporting tiles-across x tiles-deep, never for rendering.
    """
    cos_a, sin_a = _AXIS_PROJECTION[snap_orientation(orientation_deg)]
    return (tile_w_mm * cos_a + tile_h_mm * sin_a,
            tile_w_mm * sin_a + tile_h_mm * cos_a)


def rotate(u, v, rad: float):
    """Rotate plane coordinates by `rad`, turning the grid, not the tiles."""
    cos_r, sin_r = math.cos(rad), math.sin(rad)
    return u * cos_r - v * sin_r, u * sin_r + v * cos_r


def to_grid_fractions(u_mm, v_mm, tile_w_mm: float, tile_h_mm: float,
                      flip_x: bool = False, flip_y: bool = False):
    """
    Millimetre coordinates -> position within the current tile, in [0, 1).

    The doubled modulo is deliberate: np.mod already returns a non-negative
    result for a positive divisor, but coordinates behind the anchor can be
    large negatives where floating point lands exactly on -0.0, and the second
    pass normalises that to +0.0 rather than letting it sample the far edge of
    the texture.
    """
    tile_w_mm = max(tile_w_mm, 1e-3)
    tile_h_mm = max(tile_h_mm, 1e-3)

    grid_u = u_mm / tile_w_mm
    grid_v = v_mm / tile_h_mm

    frac_u = np.mod(np.mod(grid_u, 1.0) + 1.0, 1.0)
    frac_v = np.mod(np.mod(grid_v, 1.0) + 1.0, 1.0)

    if flip_x:
        frac_u = 1.0 - frac_u
    if flip_y:
        frac_v = 1.0 - frac_v

    return frac_u, frac_v, grid_u, grid_v


def anchor_offset(values_mm: np.ndarray, tile_pitch_mm: float,
                  mm_per_unit: float) -> float:
    """
    Offset, in PLANE UNITS, that moves the grid so a tile edge falls exactly on
    `values_mm`.

    Used by every surface's anchor: the floor lines its grid up with the
    back-wall junction, a wall lines its up with the skirting and its corner.
    Returns 0.0 when there is nothing usable to anchor to.
    """
    if values_mm is None or len(values_mm) == 0:
        return 0.0
    pitch = max(float(tile_pitch_mm), 1e-3)
    if abs(mm_per_unit) < 1e-12:
        return 0.0
    anchor_mm = float(np.median(values_mm))
    if not math.isfinite(anchor_mm):
        return 0.0
    return -(anchor_mm % pitch) / mm_per_unit


def surface_extent_mm(u_mm, v_mm, visible) -> Tuple[float, float]:
    """Reconstructed width and height of the visible surface, in millimetres."""
    if not np.any(visible):
        return 0.0, 0.0
    return (float(u_mm[visible].max() - u_mm[visible].min()),
            float(v_mm[visible].max() - v_mm[visible].min()))
