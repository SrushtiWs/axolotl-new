# tiles_backend/perspective_engine

The tile engine: a cleaned room and its two masks in, a tiled room out.

```text
Original room
   ↓  object removal                         backend/extraction/        (unchanged)
CLEAN_ROOM.png
   ↓  floor/wall detection                   backend/floor_wall.py      (unchanged)
FLOOR_MASK.png + WALL_MASK.png
   ↓
tiles_backend/perspective_engine/
   ├─ depth.py         MiDaS DPT-Hybrid (ONNX), as the reference /depth produced it
   ├─ pipeline/        the reference floor and wall pipelines        (migrated unchanged)
   │    camera/  vanishing points, focal length, metric scale
   │    surface/ floor + wall decisions: plane, anchor, instances
   │    core/    rays, plane basis, mm grid, rotation, grout, OpenCV sampling, lighting
   ├─ compositing.py   strict mask clip
   └─ engine.py        render_room(): builds the inputs, runs the pipelines, merges
   ↓
final tiled room  →  backend puts the objects back from the original photograph
```

Detection is **not** here. The masks come from the backend's floor/wall stage and are the
source of truth for where a tile may land.

## Where it came from

The reference project is `/Users/mac/srushti/Automatic tile mapping - Retailer 2 copy`. Its
tile renderer is `perspective_engine/tile/tile_engine.render_tile_full`, reached through
`POST /render-tile-full` in its `app.py`. The 32 modules that renderer actually loads were
copied into this package, in the same layout, so their relative imports work unchanged.
Two `__init__.py` files differ (the reference's imported a legacy DeepLSD path, PyTorch,
that the renderer never touches). Two pipelines carry **optional** hooks for stored
geometry — `floor_pipeline.render_floor_with_info(evidence=, focal=)` and
`wall_pipeline.render_wall_with_info(focal=, plane_normals=)`. With the hooks unused they
are the reference code's behaviour exactly: re-verified pixel-identical against the
reference project on real rooms after adding them. The other 30 are byte-identical.

**Not migrated:** `segmentation/` (its own SegFormer, MiDaS and DeepLSD, all PyTorch),
`surface/*/segment.py` (its floor/wall detector), `pipeline/perspective_engine.py` (legacy),
`app.py` (HTTP). Detection stays with the backend's SegFormer-B4 masks.

## Who does what

| Responsibility | Module (migrated unchanged unless noted) |
|---|---|
| Floor vanishing points: floor lines plus the mask's floor-to-wall junction, RANSAC, 0.55 confidence gate | `camera/vp_from_mask.py`, `camera/floor_boundary.py`, `surface/floor/vp.py` |
| Focal length: EXIF, then two-VP calibration, then a 26 mm-equivalent prior | `camera/focal_estimate.py` |
| Floor plane: analytic from the horizon when the VP is accepted, RANSAC on depth otherwise | `surface/floor/constraints.py`, `core/plane_fit.py` |
| Metric scale: floor from camera height (1500 mm), wall from room height | `camera/metric_scale.py`, `surface/floor/scale.py`, `surface/wall/scale.py` |
| Wall planes: room's horizontal VPs, one plane per wall, plumb | `surface/wall/vp.py`, `surface/wall/constraints.py` |
| Wall split: separate walls by facing direction and depth | `surface/wall/instances.py`, `surface/wall/depth_mask.py`, `layout.py`, `boundaries.py` |
| Rays to plane, plane basis, grid direction from the VP | `core/raycast.py` |
| Tile size in mm, 0/45/90/135 layouts, rotation composed on the VP direction | `core/uv.py` |
| Fractional and cut tiles: the mm grid runs to the mask edge, each pixel samples its place in its tile | `core/uv.py` `to_grid_fractions` |
| Grid anchor: a tile edge on the floor-to-wall junction; on a wall, the skirting and corner | `surface/floor/anchor.py`, `surface/wall/anchor.py` |
| Grout: constant on-screen width, faded below one pixel | `core/composite.py` `apply_grout` |
| OpenCV sampling and room lighting | `core/composite.py` `sample_tile`, `composite` |
| Depth, as the reference `/depth` produced it | `depth.py` — **new** |
| Strict mask clip | `compositing.py` — **new** |
| Inputs, surfaces, merge | `engine.py` — **new** |

## Floor and wall geometry — detected once, used for every tile

```text
CLEAN_ROOM.png + FLOOR_MASK + WALL_MASK
   ↓  engine.detect_room_geometry
floor   surface/floor/geometry.py   the floor pipeline's own detection: floor lines + the
                                    mask's floor/wall junction → vanishing points (RANSAC,
                                    confidence gate) → focal length (EXIF → two-VP → prior)
                                    → horizon, pitch, floor direction, plane, coverage
camera  one focal length for floor and walls — the floor's calibration
walls   the wall pipeline's own split into walls, with that camera; then per wall,
        surface/wall/geometry.py: its own along-wall vanishing point from its own lines
        and its floor junction, fitted in homogeneous coordinates so a wall facing the
        camera (vanishing point at infinity) is representable → its plumb orientation
   ↓
segments/floor/  floor_mask.png  floor_geometry.json  floor_vanishing_points.json
segments/wall/   wall_mask.png   wall_geometry.json   wall_vanishing_points.json  walls/wall-<i>.png
   ↓  render_room(geometry=…)
tiles projected with exactly that geometry
```

Why each wall gets its own direction: the reference engine orients every wall from one of
the room's two global horizontal vanishing points. A room showing only one converging
direction hands that direction to every wall — the back wall included — and a wall facing
the camera is then projected edge-on: measured on an atrium, the back wall's whole face
held one quarter of one tile. From its own lines the same wall faces the camera and shows
a proper grid. A wall with too little line evidence keeps the reference choice.

Floor `status`: `detected` (used as-is), `out-of-view` (accepted by the gate but its
horizon puts the plane off the mask — a staircase's curved edges once scored 0.74 this
way) or `rejected`. The last two are resolved at render time from the backend's estimated
camera, which knows the room's dimensions.

## The clipping rule

```text
mask == 255  →  tile allowed
mask == 0    →  tile forbidden; the clean-room pixel is kept exactly
```

The reference compositor already keeps the base pixel wherever the mask factor is under 0.1,
and a 0/255 mask gives exactly 0 or 1. `compositing.clip` then enforces it independently:
anything outside the allowed mask is reset, and the count is reported. Floor tiles are
clipped to `FLOOR_MASK`, wall tiles to `WALL_MASK` (and to the chosen walls, when only some
were asked for). The mask is used pixel for pixel; no box or polygon is derived from it.

## Use

```python
from tiles_backend.perspective_engine import TileRequest, render_room

out = render_room(
    clean_rgb, floor_mask, wall_mask, tile_rgb, "both",
    TileRequest(tile_width_mm=600, tile_height_mm=1200, rotation_deg=45, grout_mm=3,
                room_width_mm=3658, room_length_mm=4267, room_height_mm=2743),
    photo_bytes=original_upload,     # EXIF focal length, read first as the reference did
    wall_region=None,                # or a bool mask: keep only these wall pixels
    fallback_vp=(x, y),              # used only when the engine rejects its own VP
)
out.image          # (H, W, 3) RGB, the clean room tiled inside the masks
out.info           # per surface: VP source, focal, plane, scale, tiles across/deep, clip report
```

In the app, `backend/perspective_engine/render.py` calls this for `/generate` whenever the
request names a Clean Room job that has both masks, then puts the objects back over the
tiles from the original photograph.

## Verified

- **Pixel-identical to the reference project's own code:** the reference `perspective_engine`
  loaded from its own folder and this package, given the same room, masks, depth, tile and
  options. Floor and wall, 0/45/90/135°, on real rooms: every pixel identical.
- **The reference test suites, run against this package:** `test_tile_projection` all pass;
  `test_surface_parity` 67 pass, and the 6 that fail test the reference's floor/wall *detector*
  and its `app.py`, which were deliberately not migrated.
- **Strict clipping:** 0 tile pixels outside the allowed mask in every render, checked
  independently of the engine's own report. A deliberately broken render was cut back to 0.
- **Depth:** on the reference's own input/output pair (`input_vp/room5.png` →
  `depth_heatmap.png`), mean decoded-depth difference 0.0043 on 0..1, correlation 0.99983.

## Where this differs from the reference app, and why

The engine code is the reference code. What differs is what it is handed.

1. **Depth runs on ONNX.** PyTorch cannot run on this machine. Same network
   (`Intel/dpt-hybrid-midas`, via `Xenova/dpt-hybrid-midas`, 533 MB in `backend/models/`),
   same preprocessing including the reference's uint8 quirk. The export only takes 384×384
   where the reference kept the aspect ratio; the measured cost is above.
2. **No fixed 1500 px focal length.** The reference frontend sent `focal_length=1500` on
   every request, and the engine checks a plausible caller value before its own prior, so any
   photo without EXIF or two usable VPs got 1500 px whatever its width. On a 540 px room that
   is a 20° lens and a 13.4 m deep floor; the engine's own prior gives 3.6 m. The engine's
   documented order (EXIF, two-VP, prior) now decides. `TileRequest.focal_px` overrides.
3. **A fallback vanishing point.** The engine uses the caller's manual VP when its own
   detection is rejected. The reference frontend never sent one in automatic mode, so a
   rejected detection fell through to a plane fitted on depth: on one room a 600 mm tile came
   out over a metre across and the floor reconstructed 2.1 × 0.7 m. The backend passes its own
   estimated camera's horizon; that floor reconstructs 5.1 × 4.1 m. An accepted detection
   still wins. Walls get no fallback, as in the reference.
4. **Floor and wall in one request.** The reference rendered one surface per request; `both`
   renders each on the clean room and merges them through their own masks, which is the same
   thing. If one surface fails, the other is still returned and the failure is reported.
5. **Rendered on the clean room.** The reference drew on the original photo and its lighting
   read the objects in it; here tiles go onto `CLEAN_ROOM.png` and the backend puts the
   objects back on top afterwards.

## Known limits (the reference engine's own)

- **Floor VP accepted but wrong.** On a staircase room the detector passed its gate with a
  horizon that puts the floor plane out of view; no floor tile lands. Reported, not hidden.
- **Wall anchor failure.** When the wall used for the metric anchor produces no valid ray,
  the reference wall pipeline fails as a whole. Seen on 1 of 6 rooms. Reported.
- **Walls can share a plane.** When the room's horizontal VPs are weak, the engine can give
  several walls the same orientation.
- **Speed:** walls take about 6–16 s at 1600 px (vanishing points and wall splitting at the
  reference's 600 RANSAC iterations), plus about 1.2 s for depth; a floor takes about 2–5 s.
