"""
perspective_engine — tile placement and perspective rendering.

    CLEAN_ROOM.png + FLOOR_MASK.png + WALL_MASK.png
            |
      engine.render_room
            |
      the reference project's floor and wall pipelines (migrated unchanged):
      vanishing points, focal length, plane, metric scale, rotation, anchor,
      millimetre grid, cut tiles, grout, OpenCV sampling and lighting
            |
      compositing.clip — no tile pixel outside its mask
            |
      final tiled room

The reference project's `perspective_engine/__init__.py` imported a legacy
DeepLSD path (PyTorch) that the renderer never uses; it was not migrated, so
importing this package needs only numpy, OpenCV and PIL. The wall pipeline
additionally needs the MiDaS depth model — see depth.py.

See README.md.
"""

from .engine import (  # noqa: F401
    RoomRender,
    TileRequest,
    detect_room_geometry,
    detect_surfaces,
    interior_point,
    render_room,
    wall_instances,
)
from .core.options import SurfaceRenderOptions  # noqa: F401
from .tile.tile_engine import render_tile_full, render_tile_full_with_info  # noqa: F401
