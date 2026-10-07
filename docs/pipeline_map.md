# Pipeline Map — Stage 0 DISCOVER

> **Read-only.** No code was changed to produce this document.

---

## 1. Repository Layout

```
axolotl copy/
├── backend/               ← Python backend (Flask/FastAPI app)
│   ├── app.py             ← HTTP entry point (/generate, /compose, etc.)
│   ├── engine.py          ← Metric tile projection (engine — DO NOT TOUCH)
│   ├── floor_wall.py      ← Floor/wall mask detection (segmentation stage)
│   ├── live_scene.py      ← Estimated camera for uploaded photos
│   ├── surfaces.py        ← SegFormer-B4 ADE20K semantic segmenter
│   ├── matting.py         ← Guided-filter edge snap / alpha matting
│   ├── layers.py          ← Layer compositing helpers
│   ├── scene.py           ← Scene dataclass (Room, WallPlane, …)
│   ├── room_data.py       ← Room metadata, job paths
│   ├── reuse.py           ← Load a job folder (room, clean, props, …)
│   ├── pipeline_assets.py ← Asset paths for each room/job
│   ├── weights.py         ← Model weight download helpers
│   ├── deeplsd_runner.py  ← Subprocess wrapper for DeepLSD (geocalib-env)
│   ├── geocalib_runner.py ← Subprocess wrapper for GeoCalib
│   ├── object_completion.py ← Object inpainting helpers
│   ├── perspective_engine/ ← Backend connector to the tile engine
│   │   ├── __init__.py    ← Public API: load, prepare, render_tiled_room, …
│   │   ├── compositing.py ← Final guarantee: no tile outside its mask
│   │   ├── geometry.py    ← Persist/load floor & wall geometry to disk
│   │   ├── masks.py       ← Read FLOOR_MASK.png / WALL_MASK.png as bool
│   │   ├── render.py      ← Orchestrate tiles_backend render → Result
│   │   └── strong_lines.py← L1 strong-line detection (engine — DO NOT TOUCH)
│   └── extraction/        ← Object detection & mask cleanup
│       ├── cleanup.py     ← close → open → drop_small → fill_holes → smooth_contours
│       ├── config.py      ← ExtractionConfig with per-stage parameters
│       ├── extractor.py   ← GroundingDINO + SAM 2 extraction pipeline
│       ├── filtering.py   ← Post-detection filtering (structural, floor, …)
│       ├── inpaint.py     ← Object inpainting
│       └── residual.py    ← SegFormer residual regions (missed objects)
├── tiles_backend/         ← Migrated tile engine (ALL ENGINE — DO NOT TOUCH)
│   └── perspective_engine/
│       ├── engine.py      ← render_room, TileRequest, GEOMETRY_VERSION=5
│       ├── camera/
│       │   ├── vp_from_mask.py   ← Floor VP detection + confidence scoring
│       │   ├── floor_boundary.py ← Floor-to-wall junction segments
│       │   ├── focal_estimate.py ← EXIF → two-VP → FOV-prior focal length
│       │   ├── metric_scale.py   ← Metric scale from camera height / room dims
│       │   ├── vertical_vp.py    ← Vertical VP detection
│       │   └── boundary_vp.py    ← Horizontal VP from wall boundaries
│       ├── pipeline/
│       │   ├── floor_pipeline.py ← render_floor_with_info
│       │   └── wall_pipeline.py  ← render_wall_with_info
│       ├── surface/wall/
│       │   ├── corner_cut.py     ← Pillar / wall corner detection
│       │   └── junction_layout.py← Wall junction points
│       └── room/                 ← RoomGeometry, canonical room frame
├── tests/
│   ├── regression/
│   │   ├── baseline_rooms.py    ← Pixel baseline save/check (6 rooms)
│   │   ├── l1_lines.py          ← L1 report entry script
│   │   ├── l1_source_compare.py ← L1 source comparison
│   │   ├── camera_evidence.py   ← lsd(), seg_angle() helpers
│   │   └── wall_direction_diagnose.py ← fit_vp(), vp_image() helpers
│   └── validation/
│       └── validate_rooms.py    ← Full room validation (VP, camera, seam)
├── scripts/               ← 190+ numbered stage scripts (mostly GPU/frozen)
├── outputs/mask_v2/       ← Current mask outputs (8 rooms: floor + wall only)
├── backups/
│   ├── deeplsd-cache/     ← Cached DeepLSD JSON per image hash
│   ├── L1-report/         ← l1_lines.py outputs
│   ├── L1-check/          ← baseline_rooms.py check outputs
│   └── baselines-*/       ← Pixel baseline archives
├── room-data/             ← Room metadata index
└── run_pipeline.py        ← One-command runner for test07/test08 scripts
```

---

## 2. Stage-by-Stage Map

### Stage L1 — Line Quality
| Item | Detail |
|---|---|
| **Entry script** | `tests/regression/l1_lines.py` |
| **Inputs** | Job folder (CLEAN_ROOM.png, FLOOR_MASK.png, WALL_MASK.png, ALL_OBJECTS.png), baseline `.npz` (wall regions) |
| **Engine called** | `backend/perspective_engine/strong_lines.py → strong_lines()` |
| **Outputs** | `<out_dir>/<room>__l1.json`, `<room>__l1.png` |
| **Backup location** | `backups/L1-report/` |

### Stage L2 — Camera / Vanishing Points
| Item | Detail |
|---|---|
| **Entry script** | `tests/validation/validate_rooms.py` |
| **Inputs** | Job `segments/` (CLEAN_ROOM.png, FLOOR_MASK.png, WALL_MASK.png, geometry JSON) |
| **Engine called** | `tiles_backend/perspective_engine/engine.py → render_room() / detect_room_geometry()` |
| **Camera detection** | `camera/vp_from_mask.py → detect_floor_vp()`, `floor_boundary.py`, `focal_estimate.py`, `metric_scale.py` |
| **Outputs** | `tests/validation/report.md`, `report.json` |

### Floor/Wall Mask Stage
| Item | Detail |
|---|---|
| **Entry script** | `backend/floor_wall.py → detect_floor_wall()` |
| **Inputs** | CLEAN_ROOM.png |
| **Model** | SegFormer-B4 ADE20K (ONNX, shared session via `surfaces.py`) |
| **Outputs** | `FLOOR_MASK.png`, `WALL_MASK.png`, overlay PNGs |
| **Current output** | `outputs/mask_v2/<room>/` |

### Texture / Compositing Stage
| Item | Detail |
|---|---|
| **Entry script** | `backend/perspective_engine/render.py → render_tiled_room()` |
| **Inputs** | CLEAN_ROOM.png, FLOOR_MASK, WALL_MASK, tile artwork, `TileRequest` |
| **Engine** | `tiles_backend/perspective_engine/engine.py → render_room()` |
| **Outputs** | `engine.Result` (raw, lit, composite arrays) |
| **Clip check** | `perspective_engine/compositing.py → clip()` |

### Geometry Persistence
| Item | Detail |
|---|---|
| **Module** | `backend/perspective_engine/geometry.py` |
| **Writes** | `segments/floor/floor_geometry.json`, `floor_vanishing_points.json`, `segments/wall/wall_geometry.json`, `wall_vanishing_points.json`, `segments/wall/walls/wall-<i>.png`, `segments/room/room_frame.json` |
| **Re-detection trigger** | `version < GEOMETRY_VERSION (5)` → re-runs detection |

---

## 3. Engine Files — DO NOT TOUCH Without Diff + Approval

| File / Directory | Reason locked |
|---|---|
| `backend/engine.py` | Metric tile projection, illumination transfer, gloss |
| `backend/perspective_engine/strong_lines.py` | L1 line detection, fusion, merge, drop |
| `backend/perspective_engine/compositing.py` | Final mask-clip guarantee |
| `backend/perspective_engine/geometry.py` | Geometry persistence schema |
| `backend/perspective_engine/masks.py` | Strict binary mask reader |
| `backend/live_scene.py` | Synthesised camera for uploaded photos |
| `tiles_backend/perspective_engine/engine.py` | Tile engine, GEOMETRY_VERSION |
| `tiles_backend/perspective_engine/camera/` | **(Entire folder locked)** VP detection, floor boundary, focal estimate, metric scale, `vertical_vp.py`, `boundary_vp.py` |
| `tiles_backend/perspective_engine/pipeline/` | **(Entire folder locked)** Floor and wall render pipelines |
| `tiles_backend/perspective_engine/surface/` | **(Entire folder locked)** `corner_cut.py`, `junction_layout.py`, `vp.py` |
| `tiles_backend/perspective_engine/room/` | **(Entire folder locked)** `RoomGeometry` and canonical room frame |

---

## 4. L1 Verdict Rule (exact)

Source: `tests/regression/l1_lines.py`, `line_quality()`, lines 133–157.

```python
insufficient = []
for f in ("vertical", "horizontal-A", "horizontal-B"):
    # cnt = lines in this family; coverage = share of image width covered
    if cnt < 3 or coverage < 0.3:
        insufficient.append(f"{f}: {cnt} lines, {100*coverage:.0f}% of the width")

verdict = "insufficient lines -> needs_fix at L2" if insufficient else "sufficient"
```

**A family fails when it has fewer than 3 lines OR covers less than 30 % of the image width.**
All three families must pass for the verdict to be `"sufficient"`.

`gradient_supported_whole_length_share` = share of kept lines where ≥ 90 % of sample points
have gradient magnitude ≥ `strong` (photo's median × 2.0).
**Currently: computed and reported in JSON but NOT part of pass/fail verdict.**

`furniture_share` = share of kept lines whose midpoint falls within 10 px of the object mask.
**Currently: computed and reported but NOT part of pass/fail verdict.**

---

## 5. L2 Rules (a)–(e) and needs_fix

Source: `tiles_backend/perspective_engine/camera/vp_from_mask.py` and `vertical_vp.py`.

> **Note on Rule (e):** The rule "at least 8 vertical lines over 30% of image HEIGHT" is **not** currently implemented exactly like that. The actual code in `vertical_vp.py` and `vp.py` uses `len(vert) >= 3` (minimum 3 plumb lines) or `n_in >= MIN_INLIERS (4)`. There is no 30% image height check. 
> The rules listed below are what the codebase *actually* implements today.

| Rule | Constant / location | Condition |
|---|---|---|
| **(a) Inlier count** | `score_vp_confidence()` | Too few RANSAC inliers → low confidence |
| **(b) Floor line coverage** | `score_vp_confidence()` | Insufficient floor lines to work with |
| **(c) Restart agreement** | `MAX_VP_DIRECTION_SPREAD_DEG = 6.0` | VP direction swings > 6° across seeds → FAILED |
| **(d) Both directions populated** | `_select_depth_vp()` | Both horizontal families must have support |
| **(e) Wall junction agreement** | `BOUNDARY_LINE_WEIGHT = 3.0` | Junction lines weighted 3× in vote; disagreement lowers confidence |

### Three Distinct needs_fix Contexts (found in task-72 grep)

| File | Line(s) | What it means |
|---|---|---|
| `backend/room_data.py:386` | Profile status | A wall dot is outside its own wall mask **or** a corner is uncertain → profile `status = "needs_fix"` (reported; room is never auto-moved). Controlled by `STATUSES = ("auto", "needs_fix", "approved")` |
| `backend/perspective_engine/render.py:363` | Wall straight-edge RANSAC | A wall's ceiling/floor junction has too few inliers or too short a span → `report["needs_fix"].append(name)` and the clipping is skipped for that wall |
| `tiles_backend/perspective_engine/camera/vp_from_mask.py` | VP detection | Auto-detected VP rejected → caller falls back to `fallback_vp` (from `live_scene`) |
| `tests/regression/l1_lines.py` | L1 Report Label | The string `"insufficient lines -> needs_fix at L2"` is only written to the JSON report. It is **not** consumed by L2 or any other code. |

**Two "Keep current camera" fallbacks:**
1. **`GEOMETRY_VERSION >= 5` fallback:** In `tiles_backend/perspective_engine/engine.py` (via `geometry.py`), if the saved geometry JSON exists and has version >= 5, detection is skipped completely and the stored `focal_px` and VPs are loaded verbatim. This applies to rooms marked `needs_fix` (they are not auto-moved, so their geometry stays >= 5).
2. **`fallback_vp` fallback:** When detection *does* run (geometry v<5 or absent) but confidence is below threshold, `vp_from_mask` uses the caller-provided `fallback_vp` (horizon from `live_scene`).

**"Keep current camera" fallback:**
`geometry.load()` loads the stored geometry if `version >= GEOMETRY_VERSION (5)`.
Stored focal + VPs are used directly — no re-detection.
Geometry older than version 5 triggers re-detection.

---

## 6. Speckle / Dots Origin and Removal Order

### Origin
Speckle in surface masks originates from **SegFormer logit interpolation**: the model
decides at a 128×128 internal grid; bilinear upsampling to full resolution leaves stray
pixels at class boundaries wherever the argmax flips.

### Floor/wall mask cleanup order (`backend/floor_wall.py → _tidy()`)

```
SegFormer logits
  → softmax → group (floor / wall / rival)
  → resize to full resolution (float, bilinear)
  → surfaces.snap()            [guided-filter, moves boundary onto photo edges]
  → fill_small_holes()         [HOLE_FRACTION = 0.02 of mask area; skip room-surface gaps]
  → drop_small_components()    [SPECK_FRACTION = 0.001 of mask area; always keeps largest]
```

### Object mask cleanup order (`backend/extraction/cleanup.py → clean()`)

```
SAM 2 mask (full resolution)
  → MORPH_CLOSE (radius 2)      [seal hairline breaks]
  → MORPH_OPEN  (radius 1)      [remove single-pixel spurs; kept only if ≤ 3 % area loss]
  → drop_small_components()     [< min_component_fraction (0.02) of area; always keeps largest]
  → fill_small_holes()          [< max_hole_fraction of area; skip gaps > 50 % room surface]
  → smooth_contours()           [redraw from outer contours, discard detached fringe]
```

### Where upscaling happens

**Object masks from SAM 2 are produced at full render resolution — no low-res upscale.**
The only resize is `perspective_engine/masks.py → _to_shape()`: FLOOR_MASK / WALL_MASK
are resized as float (bilinear) + thresholded at 0.5 when the render resolution differs
from the stored mask resolution. This is for the tile engine, not for object masks.

---

## 7. What EXISTS vs. MISSING for Stages 1–6

### What EXISTS

| Stage | Status | Location |
|---|---|---|
| **Stage 1 (L1)** | ✅ Runs on 8 rooms | `tests/regression/l1_lines.py` + `backups/L1-report/` |
| **Stage 2 (Masks)** | ⚠️ Partial — floor + combined wall only | `outputs/mask_v2/<room>/` |
| **Stage 3 (Camera/VPs)** | ⚠️ Partial — validation script exists; no new-format report | `tests/validation/validate_rooms.py` |
| **Stage 4 (Geometry)** | ⚠️ Partial — geometry stored inside render; no standalone overlay output | `tiles_backend/perspective_engine/` |
| **Stage 5 (Texture)** | ✅ Full metric tile render + illumination transfer | `backend/perspective_engine/render.py` |
| **Stage 6 (QA)** | ⚠️ Partial — 6-room baseline + validation, no all-8 table | `tests/regression/` + `tests/validation/` |

### What is MISSING (by stage)

| Stage | Missing items |
|---|---|
| **Stage 1** | `weak_support` flag (gradient support < 0.70); report real min-max furniture_share across all rooms |
| **Stage 2** | Entire stage: per-wall masks, pillar masks, ceiling mask, glass_mirror class, full-res SAM/SAM2 object masks, polygon fit, soft alpha. **Note on pillar/walls:** The 3 walls + 1 pillar in the current pillar room is produced by `corner_cut.py` (which runs *inside* the locked rendering engine), not by the `mask_v2` stage! |
| **Stage 3** | New L2 report format per room; wall/pillar mask as extra VP evidence. **Note on VPs:** L2 runs its own `detect_lines_lsd(gray)` and does **not** use the `strong_lines` from L1. |
| **Stage 4** | Standalone grid-overlay output per surface; global parameterised tile grid; "approximate" label |
| **Stage 5** | Depends on Stage 2 masks and Stage 4 homographies; seam-free corner behaviour |
| **Stage 6** | Single all-8-rooms command; summary table; ≥ 2 new-room overfitting test; hard-case list |

---

## 8. L1 Results — All 8 Rooms (from backups/L1-report/)

| Room | Kept lines | Gradient support | Furniture share | Verdict |
|---|---|---|---|---|
| pillar | 146 | 0.986 | 0.000 | ❌ **needs_fix** (4 vertical lines, 12% width) |
| sunset_living | 195 | **0.456** | 0.123 | ✅ sufficient |
| kitchen | 104 | **0.654** | 0.048 | ✅ sufficient |
| bedroom | 146 | 0.979 | 0.048 | ✅ sufficient |
| living | 275 | 0.887 | 0.022 | ✅ sufficient |
| empty | 296 | 0.970 | 0.000 | ✅ sufficient |
| wide_angle | 398 | 0.965 | 0.040 | ✅ sufficient |
| pink_frontal | 56 | 1.000 | 0.161 | ✅ sufficient |

> sunset_living (0.456) and kitchen (0.654) have the lowest gradient support — these are
> the target rooms for the `weak_support` flag in Stage 1.

---

## 9. Room → Job Path Mapping

| Room | Job path (in baseline_rooms.py) |
|---|---|
| wide_angle | `backend/jobs/37a3d22ab0e9` |
| pillar | `backend/jobs/47080a8bc81d` |
| sunset_living | `backend/jobs/897dc4295929` |
| bedroom | `tests/fixtures/rooms/bedroom` |
| living | `tests/fixtures/rooms/living` |
| empty | `tests/fixtures/rooms/empty` |
| **kitchen** | `backend/jobs/a795d77f6fe1` (and others with identical sizes) |
| **pink_frontal** | `backend/jobs/63705139a5ee` |

*(Note: How did L1 run for kitchen/pink_frontal without a baseline `.npz`? `l1_lines.py` uses `dict(np.load(base / f"{name}.npz")) if (base / f"{name}.npz").exists() else {}`. It safely falls back to an empty dict if the .npz is absent, allowing L1 checks to run.)*

---

## 10. Additional Pipeline Details

- **Object Mask Subtraction:** Object masks are **not** subtracted in `backend/floor_wall.py` relative to `snap_to_edges` / `fill_small_holes`. `floor_wall.py` operates directly on `CLEAN_ROOM.png` (where objects have already been painted out by LaMa/diffusion). Thus, `fill_small_holes` only seals model speckles, it does not accidentally refill objects because the objects aren't in the clean room to begin with.
- **Mask Resolution:** Confirmed that SAM 2 outputs run at full resolution. For `wide_angle`, both `FLOOR_MASK.png` and `ALL_OBJECTS.png` are exactly **2560 x 1920**.
- **Engine.py usage:** L2 validation and compositing (`tests/validation/validate_rooms.py`) directly import and use the engine from `tiles_backend/perspective_engine/engine.py`.

---

*End of Stage 0 DISCOVER*

**Waiting for: next**
