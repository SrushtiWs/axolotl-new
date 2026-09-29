"""
Where the floor's tile grid starts.

Lifted unchanged from the renderer's auto_anchor_to_wall block: find the
topmost floor pixel in each column (the floor-to-wall junction at the back of
the room) and the leftmost in each row (the junction at the side), and shift
the grid so a tile edge lands on each. Tiling from a wall rather than from an
arbitrary origin is what stops the back row being a random sliver.
"""

import numpy as np

from ...core.uv import anchor_offset


def offsets(floor_bool, u_rot, v_rot, valid_ray, mm_per_unit, opts):
    """Returns (offset_u, offset_v) in plane units. (0, 0) when disabled."""
    auto_offset_u = 0.0
    auto_offset_v = 0.0

    if not opts.auto_anchor_to_wall:
        return auto_offset_u, auto_offset_v

    col_has_floor = floor_bool.any(axis=0)
    valid_cols = np.where(col_has_floor)[0]
    if len(valid_cols) > 10:
        top_ys = np.argmax(floor_bool[:, valid_cols], axis=0)
        wall_v_vals = v_rot[top_ys, valid_cols]
        wall_ok = valid_ray[top_ys, valid_cols]
        wall_v_vals = wall_v_vals[wall_ok]
        if len(wall_v_vals) > 5:
            auto_offset_v = anchor_offset(
                wall_v_vals * mm_per_unit, opts.tile_height_mm, mm_per_unit
            )

    row_has_floor = floor_bool.any(axis=1)
    valid_rows = np.where(row_has_floor)[0]
    if len(valid_rows) > 10:
        left_xs = np.argmax(floor_bool[valid_rows, :], axis=1)
        wall_u_vals = u_rot[valid_rows, left_xs]
        wall_ok_u = valid_ray[valid_rows, left_xs]
        wall_u_vals = wall_u_vals[wall_ok_u]
        if len(wall_u_vals) > 5:
            auto_offset_u = anchor_offset(
                wall_u_vals * mm_per_unit, opts.tile_width_mm, mm_per_unit
            )

    return auto_offset_u, auto_offset_v
