"""
perspective_engine — the backend's connection to the tile engine.

    CLEAN_ROOM.png -> floor_wall.py -> FLOOR_MASK.png + WALL_MASK.png
                                              |
                                      perspective_engine (this package)
                                              |
                     tiles_backend/perspective_engine (the tile engine)
                                              |
                                  strict mask clipping -> final tiled room

Detection stays in `floor_wall.py`; this package only consumes its two files.
See README.md for which code owns which responsibility.
"""

from perspective_engine.compositing import allowed, clip
from perspective_engine.masks import MaskError, load, prepare
from perspective_engine.render import Rendered, describe, ensure_geometry, render_tiled_room, surfaces

__all__ = [
    "MaskError",
    "Rendered",
    "allowed",
    "clip",
    "describe",
    "ensure_geometry",
    "load",
    "prepare",
    "render_tiled_room",
    "surfaces",
]
