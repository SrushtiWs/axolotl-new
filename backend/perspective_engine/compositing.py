"""
The final guarantee: no tile pixel outside its white mask.

`engine.render` already paints only the pixels its scene's masks hold —
`_project_floor` walks `scene.floor`, `_project_walls` walks `scene.wall`, and
relighting touches only what they covered — and `render.render_tiled_room`
binds those scene masks to `FLOOR_MASK.png` and `WALL_MASK.png`. So clipping is
already true by construction, and this module is the check that says so.

It is still a real clip, not only an assertion. If a pixel outside the allowed
mask were ever claimed — by a future change to the renderer, say — it is put
back to the untiled room before anything leaves this package, the objects are
restored over the result again, and the count is reported. The measured count
is 0; the point is that it stays 0 whatever changes upstream.
"""

from __future__ import annotations

import numpy as np

import engine
from engine import Result
from scene import Scene


def allowed(surface: str, floor: np.ndarray, wall: np.ndarray) -> np.ndarray:
    """Where a tile may land for this request: the mask of each surface asked for."""
    if surface == "floor":
        return floor

    if surface == "wall":
        return wall

    return floor | wall


def clip(result: Result, scene: Scene, permitted: np.ndarray) -> dict:
    """
    Remove any tile pixel outside `permitted`, in place, and report what it found.

    `raw` and `lit` are reset to `result.underlying` — the clean room the tiles
    were projected onto — wherever a tile is forbidden, and the composite is
    rebuilt from the clipped `lit` through the renderer's own object
    restoration, so the objects still sit on top exactly as before.
    """
    outside = result.target & ~permitted

    violations = int(outside.sum())

    if violations:
        result.raw[outside] = result.underlying[outside]
        result.lit[outside] = result.underlying[outside]

        result.target &= permitted

        result.composite = engine._restore_props(scene, result.lit)

    return {
        "tile_pixels": int(result.target.sum()),
        "outside_mask_before_clip": violations,
        "outside_mask_after_clip": int((result.target & ~permitted).sum()),
    }
