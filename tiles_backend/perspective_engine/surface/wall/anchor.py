"""
Where a wall's tile grid starts.

Tilers do not start a wall in the middle. They start at the bottom, on the
floor line or the top of the skirting, so the first course is a full tile and
any cut lands at the ceiling where it is least visible; and they start at a
corner, so the vertical joint lines up with the room's edge.

So this anchors:

  v  to the FLOOR-WALL JUNCTION -- the lowest wall pixel in each column. This
     is the same idea as the floor's back-wall anchor (floor/anchor.py), read
     from the other side of the same line.

  u  to the WALL CORNER -- the instance's leading vertical edge, which
     instances.py has already placed on the real corner seam by snapping it to
     a detected vertical line.

Both fall back to no offset when the evidence is missing, which is better than
anchoring to a wall edge that is really the image border.
"""

import numpy as np

from ...core.uv import anchor_offset

#: Columns whose lowest wall pixel sits within this fraction of the image
#: bottom are ignored: the wall runs out of frame there, so its "junction" is
#: the photo's edge, not the floor.
BOTTOM_EDGE_GUARD = 0.01

#: Same idea for the corner: an instance starting at column 0 is clipped by
#: the frame, not bounded by a corner.
SIDE_EDGE_GUARD = 0.01


def offsets(instance_mask, u_rot, v_rot, valid_ray, mm_per_unit, opts):
    """Returns (offset_u, offset_v) in plane units."""
    offset_u = 0.0
    offset_v = 0.0

    if not opts.auto_anchor_to_wall:
        return offset_u, offset_v

    h, w = instance_mask.shape[:2]
    bottom_guard = int(round((1.0 - BOTTOM_EDGE_GUARD) * h))
    side_guard = int(round(SIDE_EDGE_GUARD * w))

    # ---- v: the floor-wall junction (skirting line) ----
    col_has_wall = instance_mask.any(axis=0)
    valid_cols = np.where(col_has_wall)[0]
    if len(valid_cols) > 10:
        # Lowest wall pixel per column = where the wall meets the floor.
        flipped = instance_mask[::-1, :]
        bottom_ys = (h - 1) - np.argmax(flipped[:, valid_cols], axis=0)
        keep = (bottom_ys < bottom_guard) & valid_ray[bottom_ys, valid_cols]
        junction_v = v_rot[bottom_ys, valid_cols][keep]
        if len(junction_v) > 5:
            offset_v = anchor_offset(
                junction_v * mm_per_unit, opts.tile_height_mm, mm_per_unit
            )

    # ---- u: the wall's corner ----
    row_has_wall = instance_mask.any(axis=1)
    valid_rows = np.where(row_has_wall)[0]
    if len(valid_rows) > 10:
        left_xs = np.argmax(instance_mask[valid_rows, :], axis=1)
        keep_u = (left_xs > side_guard) & valid_ray[valid_rows, left_xs]
        corner_u = u_rot[valid_rows, left_xs][keep_u]
        if len(corner_u) > 5:
            offset_u = anchor_offset(
                corner_u * mm_per_unit, opts.tile_width_mm, mm_per_unit
            )

    return offset_u, offset_v
