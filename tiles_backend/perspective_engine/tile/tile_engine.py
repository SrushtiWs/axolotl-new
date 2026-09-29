"""
tile_engine.py -- backward-compatible entry point.

This module used to contain the whole floor renderer. That renderer now lives
in pipeline/floor_pipeline.py, built out of the shared core/ stages and the
floor's own surface/floor/ decisions, so the wall pipeline could reuse the
same rendering core instead of copying it.

Nothing a caller relies on has moved:

    from perspective_engine.tile.tile_engine import RenderOptions, render_tile_full

still works and still renders a floor, because render_tile_full defaults to
surface="floor". Pass surface="wall" (or set opts.surface) to route the same
call to the wall pipeline.

The original file is kept beside this one as tile_engine_original.py.bak for
diffing if a floor render ever looks different.
"""

from ..camera.metric_scale import DEFAULT_CAMERA_HEIGHT_MM, MetricScaleError  # noqa: F401
from ..core import VP_DEBUG_INFO, debug as _debug  # noqa: F401
from ..core.composite import (  # noqa: F401
    GROUT_MAX_PX,
    GROUT_MIN_PX,
    decode_depth_array,
    parse_hex_color,
)
from ..core.options import SurfaceRenderOptions
from ..core.plane_fit import (  # noqa: F401
    PLANE_FIT_POLISH_ROUNDS,
    PLANE_FIT_RESTARTS,
    PLANE_FIT_SEED,
)
from ..surface.base import SurfaceKind

#: The options class every existing caller imports. Same field names, same
#: defaults; wall-only fields were appended with defaults, so an existing
#: construction is unchanged.
RenderOptions = SurfaceRenderOptions


def render_tile_full(room_bgr, depth_bgr, mask_bgra, tile_bgra, opts,
                     surface=None, **kwargs):
    """
    Render a tile texture onto a segmented surface.

    room_bgr:  HxWx3 uint8 (BGR, cv2 convention)
    depth_bgr: HxWx3 uint8 (BGR) -- same size as room
    mask_bgra: HxWx4 uint8 (BGRA) -- alpha/brightness define surface coverage
    tile_bgra: hxwx4 uint8 (BGRA) tile texture, alpha used for blending

    surface: "floor" (default) or "wall". Falls back to opts.surface, which
    itself defaults to "floor" -- so a call written before walls existed takes
    the identical path it always did.

    Extra keyword arguments are forwarded to the wall pipeline
    (floor_mm_per_unit, openings_mask) and ignored by the floor.
    """
    kind = SurfaceKind.parse(
        surface if surface is not None else getattr(opts, "surface", "floor")
    )

    if kind is SurfaceKind.WALL:
        from ..pipeline.wall_pipeline import render_wall
        return render_wall(room_bgr, depth_bgr, mask_bgra, tile_bgra, opts, **kwargs)

    from ..pipeline.floor_pipeline import render_floor
    return render_floor(room_bgr, depth_bgr, mask_bgra, tile_bgra, opts)


def render_tile_full_with_info(room_bgr, depth_bgr, mask_bgra, tile_bgra, opts,
                               surface=None, **kwargs):
    """
    As render_tile_full, but returns (image, info).

    info reports what the render actually decided -- planes, metric anchor,
    reconstructed surface size, tiles across/deep, and for a wall the
    per-instance breakdown. Added rather than folded into render_tile_full so
    the existing single-return signature keeps working.
    """
    kind = SurfaceKind.parse(
        surface if surface is not None else getattr(opts, "surface", "floor")
    )

    if kind is SurfaceKind.WALL:
        from ..pipeline.wall_pipeline import render_wall_with_info
        return render_wall_with_info(
            room_bgr, depth_bgr, mask_bgra, tile_bgra, opts, **kwargs
        )

    from ..pipeline.floor_pipeline import render_floor_with_info
    return render_floor_with_info(room_bgr, depth_bgr, mask_bgra, tile_bgra, opts)
