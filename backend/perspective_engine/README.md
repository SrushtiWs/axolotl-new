# backend/perspective_engine

The backend's connection to the tile engine. It takes a Clean Room job's masks and hands
them, with the room, to [`tiles_backend/perspective_engine`](../../tiles_backend/perspective_engine/README.md),
which places the tiles.

```text
1. Object removal                     extraction/                     (unchanged)
        ↓
2. CLEAN_ROOM.png
        ↓
3. Floor/Wall detection               floor_wall.py                   (unchanged)
        ↓
4. FLOOR_MASK.png + WALL_MASK.png
        ↓
5. this package                       masks.py: load + validate 0/255
        ↓                             render.py: estimated camera → fallback VP, wall choice
6. tiles_backend/perspective_engine   reference tile engine, clipped to each mask
        ↓
7. objects back on top                engine._restore_props (from the original photo)
        ↓
8. strict clip, checked again         compositing.py
        ↓
   final tiled room
```

There is one tile engine. This package does not place tiles; it prepares inputs and checks
outputs.

## Files

| File | Responsibility |
|---|---|
| `masks.py` | Reads `FLOOR_MASK.png` / `WALL_MASK.png` from a job, refuses anything that isn't strictly 0/255, resizes to the render size, keeps them disjoint |
| `render.py` | `render_tiled_room(...)`: estimated camera, fallback VP, wall choice, tile engine, objects back, result in the `engine.Result` shape `/generate` writes |
| `compositing.py` | The final clip and its report. After the tile engine's own clip, this re-checks the finished image, objects included |

## What the backend adds to the tile engine

- **The masks.** Handed over verbatim as the clipping boundary.
- **A fallback vanishing point.** `live_scene` estimates this room's camera from the floor's far edge and the room length. Its horizon, straight ahead, is given to the tile engine, which uses it only when it rejects its own vanishing point.
- **Which walls.** Left / right / back are this backend's room-box walls. When only some are chosen, the pixels `engine._project_walls` assigns to them become the tile engine's `wall_region`. Every wall is still rendered with one shared scale, as in the reference; only the chosen walls' pixels are kept.
- **The objects.** Put back over the tiles from the original photograph through their soft alpha.

## Where it runs

`POST /generate`, whenever the request names a Clean Room `job_id` whose `segments/` folder
has both masks. The response shows `timings.perspective_engine: true`,
`geometry.renderer: "tiles_backend.perspective_engine"`, the floor's and walls' own
diagnostics under `geometry.floor` / `geometry.wall`, and `geometry.mask_clip`.

These paths are unchanged: the calibrated room (fitted stage08g3 camera, `engine.py`), an
upload with no Clean Room run, and a job from before the floor/wall stage existed.
