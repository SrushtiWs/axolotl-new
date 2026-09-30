"""
Tile count and edge cuts, from the room's size in mm and the tile's size in mm.

Reporting only: nothing here feeds the renderer.

Per axis:
    footprint  = the tile's extent along that axis (core/uv.effective_footprint_mm;
                 at 90 degrees width and height swap)
    full       = floor(span / footprint)          -- no rounding of span or tile first
    edge_cut   = span - full * footprint          -- the width of the one cut piece
    count      = full + (1 if edge_cut > 0 else 0)

The grid pitch is the tile itself: grout is painted inside each tile, as the
renderer's `fract(grid_mm / tile_mm)` does, so it does not change the count.

The count assumes the grid starts at one edge. The render anchors its grid
on its own; where it starts mid-surface, the same cut is split across both
edges. The number of cut tiles along an axis is then at most one more.

Diagonal lays (45 / 135) are not reported: along an edge they give a row of
triangular cuts, which a whole-tile division does not describe.
"""

from __future__ import annotations

import math
from typing import Optional

from ..core.uv import effective_footprint_mm, snap_orientation

#: Below this a remainder is float noise (e.g. 3600 - 6 * 600), not a cut.
EDGE_EPS_MM = 1e-6

#: Which room dimension a wall runs along, by its RoomGeometry class.
WALL_AXIS = {"back": ("x", "width"), "side": ("z", "length")}


def _axis(span_mm: Optional[float], footprint_mm: float) -> dict:
    if span_mm is None or not span_mm > 0:
        return {"span_mm": None, "full": None, "edge_cut_mm": None, "count": None}
    ratio = span_mm / footprint_mm
    full = math.floor(ratio)
    cut = span_mm - full * footprint_mm
    if cut < EDGE_EPS_MM:
        cut = 0.0
    elif footprint_mm - cut < EDGE_EPS_MM:          # ratio fell a hair short of an integer
        full, cut = full + 1, 0.0
    # Computed from the exact span; only the reported values are rounded (0.001 mm).
    return {"span_mm": round(span_mm, 3), "full": full, "edge_cut_mm": round(cut, 3),
            "count": full + (1 if cut > 0 else 0)}


def surface_layout(axes: dict, tile_w_mm: float, tile_h_mm: float, rotation_deg: float) -> dict:
    """
    `axes` = {"x": (span_mm, source), "z": (...)}, two axes in the order
    (across, deep) -- across takes the tile's width at 0 degrees.
    """
    names = list(axes)
    orientation = snap_orientation(rotation_deg)
    out: dict = {"axes": names, "tile_mm": [tile_w_mm, tile_h_mm], "rotation_deg": orientation}

    if orientation in (45, 135):
        out.update(status="UNAVAILABLE", reason="diagonal lay: edge cuts are triangles, not a whole-tile division")
        return out

    across, deep = effective_footprint_mm(tile_w_mm, tile_h_mm, orientation)
    out["footprint_mm"] = [across, deep]
    per = {name: _axis(axes[name][0], pitch) for name, pitch in zip(names, (across, deep))}

    for name in names:
        out[f"span_{name}_mm"] = per[name]["span_mm"]
        out[f"span_{name}_source"] = axes[name][1]
        out[f"tile_count_{name}"] = per[name]["count"]
        out[f"full_{name}"] = per[name]["full"]
        out[f"edge_cut_{name}_mm"] = per[name]["edge_cut_mm"]

    missing = [n for n in names if per[n]["count"] is None]
    if missing:
        out.update(status="UNAVAILABLE", reason="no size for " + ", ".join(missing))
        return out

    a, b = names
    total = per[a]["count"] * per[b]["count"]
    full = per[a]["full"] * per[b]["full"]
    sources = {axes[n][1] for n in names}
    out.update(
        status="ESTIMATED" if sources - {"USER_INPUT"} else "USER_INPUT",
        total_tiles=total, full_tiles=full, partial_tiles=total - full,
    )
    return out


def room_layout(room: dict, tile_w_mm: float, tile_h_mm: float, rotation_deg: float) -> dict:
    """
    Floor (x = width, z = length) and every wall RoomGeometry classified
    (back: x x y, side: z x y), from a RoomGeometry dict.
    """
    def dim(name):
        return room.get(f"{name}_mm"), room.get(f"{name}_source") or "UNKNOWN"

    width, length, height = dim("width"), dim("length"), dim("height")
    by_name = {"width": width, "length": length}

    surfaces = {"floor": surface_layout({"x": width, "z": length}, tile_w_mm, tile_h_mm, rotation_deg)}
    if room.get("length_is_lower_bound") and surfaces["floor"].get("span_z_mm") is not None:
        surfaces["floor"]["note"] = "length is a lower bound (the back wall is the farthest point seen)"

    for wid, entry in sorted((room.get("walls") or {}).items()):
        cls = (entry or {}).get("class")
        if cls not in WALL_AXIS:
            surfaces[wid] = {"status": "UNAVAILABLE",
                             "reason": f"wall class {cls!r} runs along no room axis"}
            continue
        axis, name = WALL_AXIS[cls]
        surfaces[wid] = surface_layout({axis: by_name[name], "y": height}, tile_w_mm, tile_h_mm, rotation_deg)
        surfaces[wid]["wall_class"] = cls

    return {
        "room_status": room.get("status"),
        "room_confidence": room.get("confidence"),
        "grid_anchor": "counts assume the grid starts at one edge; the render's own anchor may split a cut across both edges",
        "surfaces": surfaces,
    }
