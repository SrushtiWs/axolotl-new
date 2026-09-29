"""
What a 3D view needs to redraw a surface exactly as the 2D render drew it.

The 2D pipelines decide everything: camera, plane, grid basis, rotation,
anchor, metric scale, grout and lighting. This module only RECORDS those
decisions, once per rendered surface, in a form a GPU renderer can replay:

    pixel (x, y) -> camera ray -> plane -> (u, v) -> rotate -> + offset -> * mm_per_unit
                 = (u_mm, v_mm), the tile grid in millimetres

(u_mm, v_mm) is an affine function of the 3D point on the plane, so a quad on
that plane carrying (u_mm, v_mm) at its corners reproduces the grid at every
pixel under perspective-correct interpolation. Nothing here feeds back into
the render; the pipelines' pixels are unchanged.

Units: camera frame is the pipelines' (x right, y down, z forward, principal
point at the image centre). `meters_per_unit` converts plane units to metres,
so 1 unit of the 3D scene is 1 m and a 600 x 1200 mm tile is 0.6 x 1.2.
"""

from __future__ import annotations

import numpy as np

from .core.composite import GROUT_MAX_PX, GROUT_MIN_PX, parse_hex_color
from .core.uv import rotate


def _grout(u_mm, v_mm, visible, tile_w_mm, tile_h_mm, grout_mm):
    """
    The scalars core.composite.apply_grout derives before its per-pixel pass:
    the grout's on-screen width and ink, fixed from the median mm-per-pixel
    over the visible surface. The per-pixel part (local mm-per-pixel) is
    recomputed on the GPU from screen-space derivatives.
    """
    du_dy, du_dx = np.gradient(u_mm)
    dv_dy, dv_dx = np.gradient(v_mm)
    per_px_u = np.maximum(np.hypot(du_dx, du_dy), 1e-6)
    per_px_v = np.maximum(np.hypot(dv_dx, dv_dy), 1e-6)

    if np.any(visible):
        ref_u = float(np.median(per_px_u[visible]))
        ref_v = float(np.median(per_px_v[visible]))
    else:
        ref_u = ref_v = 1.0

    want_u = grout_mm / max(ref_u, 1e-6)
    want_v = grout_mm / max(ref_v, 1e-6)
    width_u = float(np.clip(want_u, GROUT_MIN_PX, GROUT_MAX_PX))
    width_v = float(np.clip(want_v, GROUT_MIN_PX, GROUT_MAX_PX))

    return {
        "width_px": [width_u, width_v],
        "ink": [min(1.0, want_u / width_u) if width_u > 0 else 1.0,
                min(1.0, want_v / width_v) if width_v > 0 else 1.0],
        "max_half_fraction": 0.40,
    }


def record(*, kind, f, cx, cy, image_size, plane, e_u, e_v, rad, offset_units,
           mm_per_unit, opts, tile_size_px, height_stretch, u_mm, v_mm, u_raw, v_raw,
           visible, average_brightness):
    """
    One surface's replayable render state.

    u_raw / v_raw are the unrotated plane coordinates (units) the pipeline
    computed; their range over `visible` fixes the quad the 3D view draws.
    """
    a, b, c, d = (float(x) for x in plane)
    n = np.array([a, b, c], np.float64)
    norm = float(np.linalg.norm(n)) or 1.0
    n, d = n / norm, d / norm
    origin = -d * n                                  # the plane point nearest the camera

    eu = np.asarray(e_u, np.float64)
    ev = np.asarray(e_v, np.float64)

    if np.any(visible):
        # A little past the visible extent, so no pixel is lost to rasterisation.
        u0, u1 = float(u_raw[visible].min()), float(u_raw[visible].max())
        v0, v1 = float(v_raw[visible].min()), float(v_raw[visible].max())
        pad_u, pad_v = 0.02 * (u1 - u0) + 1e-6, 0.02 * (v1 - v0) + 1e-6
        u0, u1, v0, v1 = u0 - pad_u, u1 + pad_u, v0 - pad_v, v1 + pad_v
    else:
        u0 = u1 = v0 = v1 = 0.0

    corners_uv = [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]
    corners, grid = [], []

    for u, v in corners_uv:
        # project_to_plane_uv measures u = X . e_u, v = X . e_v; origin is along n,
        # so X = origin + u e_u + v e_v lies on the plane with exactly those (u, v).
        X = origin + u * eu + v * ev
        corners.append([float(x) for x in X])
        ur, vr = rotate(np.float64(u), np.float64(v), rad)
        grid.append([float((ur + offset_units[0]) * mm_per_unit),
                     float((vr + offset_units[1]) * mm_per_unit)])

    tile_w_mm = max(float(opts.tile_width_mm), 1e-3)
    tile_h_mm = max(float(opts.tile_height_mm), 1e-3)

    grout = None
    if opts.enable_grout and opts.grout_width_mm > 0:
        grout = _grout(u_mm, v_mm, visible, tile_w_mm, tile_h_mm, float(opts.grout_width_mm))
        grout["color_rgb"] = list(parse_hex_color(opts.grout_color))
        grout["width_mm"] = float(opts.grout_width_mm)

    return {
        "kind": kind,
        "camera": {"focal_px": float(f), "cx": float(cx), "cy": float(cy),
                   "image_size": [int(image_size[0]), int(image_size[1])]},
        "plane": {"normal": [float(x) for x in n], "d_units": float(d),
                  "e_u": [float(x) for x in eu], "e_v": [float(x) for x in ev]},
        "meters_per_unit": float(mm_per_unit) / 1000.0,
        # Quad corners in camera-frame units, and the tile grid (mm) at each.
        "quad_units": corners,
        "quad_grid_mm": grid,
        "grid": {"rotation_rad": float(rad), "offset_units": [float(o) for o in offset_units],
                 "mm_per_unit": float(mm_per_unit)},
        "tile": {"width_mm": tile_w_mm, "height_mm": tile_h_mm,
                 "size_px": [int(tile_size_px[0]), int(tile_size_px[1])],
                 "flip": [bool(opts.flip_tile_x), bool(opts.flip_tile_y)],
                 "height_stretch": float(height_stretch)},
        "grout": grout,
        "lighting": {"blend": float(opts.lighting_blend),
                     "average_brightness": float(average_brightness),
                     "opacity": float(opts.tile_opacity)},
    }
