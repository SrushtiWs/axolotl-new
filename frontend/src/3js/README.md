# 3js: the Three.js tile layer

Everything Three.js in the frontend is in this folder. It adds an optional **3D View** that redraws the tiles of the selected surfaces with Three.js.

The 2D renderer (OpenCV, `tiles_backend/perspective_engine`) is still the default view. It is not changed by anything here.

Three.js is only a **drawing layer**. It does no room detection, and it does not estimate vanishing points, camera angle, room size or tile size. It replays exactly what the 2D engine decided for each render.

## Files

| File | What it does |
|---|---|
| `index.ts` | The folder's public exports. The app imports only from `../3js`. |
| `threeTiles.ts` | Core renderer: the `three.json` types, `loadThreeJob`, the camera built from the engine's intrinsics (`projectionFromIntrinsics`), the surface plane mesh (`surfaceGeometry`), the tile shader (`surfaceMaterial`), and `TileLayerRenderer`. |
| `ThreeTileLayer.tsx` | React component that stacks photo → tiles → objects in the image's own box. |
| `three-layer.css` | Positioning for that stack. |
| `README.md` | This file. |

**Used by:** `pages/Studio.tsx`, through the **3D View** button next to Compare. The button is enabled once a surface is tiled.

**Dependency:** `three@0.169.0`, with `@types/three@0.169.0` for development (in `frontend/package.json`).

## Where the data comes from (backend, outside this folder)

Python can't import a package whose name starts with a digit, so the backend half keeps its place:

| Backend file | Role |
|---|---|
| `tiles_backend/perspective_engine/three_layer.py` | `record(...)`: captures one rendered surface's state. Read-only; it never changes pixels. |
| `tiles_backend/perspective_engine/pipeline/floor_pipeline.py` | Adds `info["three"]` after the floor render. |
| `tiles_backend/perspective_engine/pipeline/wall_pipeline.py` | Adds `"three"` to each wall's entry after it is rendered. |
| `backend/perspective_engine/render.py` | Carries the records on `Rendered.three`, keyed `floor` or `wall-<i>`. |
| `backend/app.py` | `_save_three(...)` writes the files below. `/generate` returns `three_url`. |

Each tile render job folder (`backend/jobs/<job>/`) gets:

```text
three.json            the render state of every surface in this render
three/<surface>.png   that surface's tiled pixels, white: the clip mask
three/objects.png     the room's objects (RGBA, alpha = object coverage)
original.png          the photograph (underlay)
CLEAN_ROOM.png        the render's base image (source of the lighting)
tile.png              the tile artwork
```

## The room (Phase 3)

`three.json` also carries `room`: the canonical RoomGeometry the render used
(`tiles_backend/perspective_engine/room/geometry.py`). The floor and every wall
of that render share its single metric scale (the camera height), so the per-surface
`meters_per_unit` values below already agree with each other and with `room`; the
3D view adds no scale of its own.

## `three.json`, per surface

```jsonc
{
  "kind": "floor | wall",
  "camera": { "focal_px", "cx", "cy", "image_size": [w, h] },  // the engine's intrinsics
  "plane":  { "normal", "d_units", "e_u", "e_v" },               // plane + its tile axes
  "meters_per_unit": 0.0026,                                     // plane units -> metres
  "quad_units":   [[x,y,z] x 4],   // corners of the visible plane region (camera frame, units)
  "quad_grid_mm": [[u,v]   x 4],   // the tile grid (mm) at those corners
  "grid":  { "rotation_rad", "offset_units", "mm_per_unit" },  // room angle + preset, anchor
  "tile":  { "width_mm", "height_mm", "size_px", "flip", "height_stretch" },
  "grout": { "width_px", "ink", "max_half_fraction", "color_rgb", "width_mm" } | null,
  "lighting": { "blend", "average_brightness", "opacity" },
  "mask": "three/floor.png"
}
```

## Units and coordinates

- **1 scene unit = 1 metre.** A 600×1200 mm tile is 0.6 × 1.2 units, and 1200×2400 mm is 1.2 × 2.4. Tiles are never resized.
- The engine's camera frame is x right, y down, z forward. Three.js uses (x, −y, −z).
- The camera sits at the origin. Its projection matrix is built directly from the engine's `focal_px` and `(cx, cy)`, so a 3D point lands on the same image pixel the engine projected it to.
- There is a half-pixel shift (`cx + 0.5`, `cy + 0.5`). The engine evaluates pixel *i* at coordinate *i*, but a WebGL fragment's centre is at *i + 0.5*.
- Each surface is drawn with its own render's camera, so floor and wall renders line up even if their intrinsics differ.

## What the shader does per pixel (same steps as the 2D engine)

1. **Clip.** The pixel must be white in `three/<surface>.png`, or it is discarded. No tile can land on black.
2. **Grid.** `fract(grid_mm / tile_mm)`, with flips (`core/uv.py`). The grid in mm is an affine function of the 3D point, so perspective-correct interpolation is exact.
3. **Texture.** Bilinear with wrap, and the floor's texture-space v squash (`height_stretch`), as `cv2.remap` in `core/composite.sample_tile`.
4. **Grout.** Constant on-screen width, faded below 1 px (`core/composite.apply_grout`). Local mm-per-pixel comes from screen derivatives.
5. **Lighting.** From the render's base image brightness (`core/composite.composite`).
6. **Opacity.** The render's tile opacity.

Then `objects.png` is laid on top, which is how the 2D composite puts furniture back over the tiles.

## Verified (3D layer vs the 2D engine, pixel by pixel)

| Check | Result |
|---|---|
| Surfaces | floor and 4 walls; rotations 0/45/90/135; tiles 1200×1800, 600×1200, 1200×2400 |
| 3D pixels outside the white mask | **0** on every test |
| White mask pixels not drawn | **0** on every test |
| Colour vs 2D | 98–99.7% of pixels within 2 levels, **100% within 8** (8-bit rounding) |
| 1 unit = 1 m | quad edge lengths in the scene equal tile-grid lengths exactly |
| 2D output after the backend additions | byte-identical to before |
| Real app | upload → Clean Room → select surface → 3D View on/off, no errors |

## Use

1. Restart the backend and frontend (`./dev.sh`).
2. Select a surface.
3. Click **3D View**.

A surface rendered before `three.json` existed shows "3D view unavailable". Deselect it and select it again to re-render.

## Limits (current)

- The view is the photo's own camera. There is no free orbit yet.
- The tile artwork is assumed opaque, as with the catalogue's JPGs. With a transparent tile, the 3D view blends it over the photograph, while the 2D engine blends it over the clean room.
- The 3D view shows only what the backend has already rendered. Changing tile, size, grout or rotation still goes through the backend render first.
