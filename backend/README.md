# Backend — AI Tile Visualizer

FastAPI service implementing the `POST /generate` contract the frontend expects.

## Run

```bash
./dev.sh                 # from the project root: backend + frontend together
```

or separately:

```bash
backend/.venv/bin/uvicorn app:app --app-dir backend --reload --port 8000
cd frontend && npm run dev
```

The frontend reads `frontend/.env.local`:

```
VITE_API_BASE_URL=http://localhost:8000
```

Vite only reads that file at startup — restart the dev server after changing it.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Whether the geometry assets are present, and the calibrated room's metric size |
| `GET` | `/pipeline/flow` | The nine stages in order, each with a preview image from the committed run |
| `GET` | `/pipeline/segments` | The calibrated room's committed props + mirror, as the same two layers |
| `POST` | `/segment` | Extract **any** uploaded room photo into `ALL_OBJECTS.png` + `MIRRORS_ONLY.png`, and rebuild the room behind them as `CLEAN_ROOM.png` |
| `POST` | `/floor-wall` | Split a **cleaned** room into `FLOOR_MASK.png` + `WALL_MASK.png` |
| `POST` | `/generate` | Render a tile visualization |
| `GET` | `/jobs/{job_id}/{file}` | Rendered output (`result.png`, `raw.png`, `lit.png`, `original.png`, `state.json`) |
| `GET` | `/assets/{path}` | Read-only mount of `test07/production_pipeline/` |
| `GET` | `/generated/{path}` | Object and surface cut-outs produced by `pipeline_assets.py` |

`POST /generate` takes multipart fields `room_image`, `tile_image`, `room_width`,
`room_length`, `room_height`, `tile_width`, `tile_height`, `rotation`, `surface`
and returns `{ job_id, original_image_url, result_image_url, raw_image_url,
clean_room_url, stats, notes }`. `clean_room_url` is the emptied room the tiles
were projected onto, or `null` on the calibrated path, which has its own.

### The clean room

`extraction/inpaint.py` fills the object masks with LaMa (`big-lama`, ONNX, CPU,
fetched once into `models/lama_fp32.onnx`). The export's input is a fixed
512×512, so a room is not pushed through it whole: each group of holes is filled
in its own context window, capped at 640 px, and every hole is written by
exactly one window. The fill is composited back with the mask as a stencil, so
every pixel outside it is bit-for-bit the original photograph — `objects.json`
records `clean_room.checks.outside_changed`, which is that count, and it is 0.

The mask is grown by `grow_px` (2–8 px, proportional to the photo) before
filling. The written alpha is cut at the object's own boundary, and the ring
just outside it still holds the object's edge colour and contact shadow; leaving
it would leave a silhouette.

`/generate` then detects floor and wall on the *clean* room rather than the
photograph. A chair hides the floor it stands on and SegFormer cannot label what
it cannot see — measured on one 1408×768 room, wall coverage went from 0.662 to
0.889 of the frame and floor from 0.1058 to 0.1107. Those are the pixels the
objects were hiding; the objects are composited back over the tiles from the
original photograph immediately afterwards.

## What it actually does

`engine.py` is stage **08h1** (true metric tile projection) followed by **08h2**
(props over metric floor), re-expressed as one parameterised, vectorised
function. Every pixel is back-projected through the fitted camera, intersected
with the floor plane `Y=0` or a wall plane, and the metric hit point indexes a
tile grid with `tile + grout` pitch. Illumination and the gloss response are
carried over from the empty-room render unchanged.

Verified against the committed pipeline output
(`stage08_tile_application/08h2_props_over_metric_floor/02_metric_floor_plus_physical_props.png`)
for the locked spec — 600 × 1200 mm, 5 mm grout, rotation 0, both surfaces:
**mean absolute difference 0.19/255, maximum 7/255**, from nearest-neighbour
rounding order. No pixel differs by more than 8.

What a request genuinely controls: tile artwork, tile size in mm, rotation
(0/45/90/135) of the floor layout, and target surface (floor / wall skirting /
both).

## Object extraction — any uploaded room

`POST /segment` takes any room photo and returns **two** transparent PNGs:

| File | Holds |
| --- | --- |
| `all_objects.png` | every object in the room — furniture, rugs, lamps, plants, paintings, wall decorations, **and the mirrors** |
| `mirrors.png` | only the mirrors |

Not one PNG per object. Both layers are the full size of the uploaded photo and
keep every object at that photo's own coordinates, so compositing them back over
a retiled room is a plain alpha blend with no placement maths. The mirrors are a
strict subset of the objects layer, enforced in the bytes that reach the file.

No calibrated geometry is needed, so this works where `/generate` cannot.

Wall, floor and ceiling are excluded from both layers — they come back
separately under `surfaces`, which is what `/generate` tiles. An instance is
also trimmed against the eroded wall/floor/ceiling map before it joins a layer:
a point prompt on a kitchen cabinet can lead SAM out across the polished floor
it stands on, and a patch of old floor inside an object mask would paint itself
back over the new tiles.

A mirror reflects the room, and SegFormer will happily find furniture inside
that reflection. Any non-mirror instance lying 70% or more inside a mirror is
dropped as a reflection — `detections` says so, with the percentage.

`detections` reports every instance the segmentation found and what became of it
(`object`, `mirror`, `excluded`, `skipped`, `failed`) so the per-object view is
still there for debugging. A mask that cannot be placed is logged and skipped;
it does not fail the request.

`GET /pipeline/segments` returns the same two layers for the calibrated room,
composed from the committed stage 06F4 props and the stage 03 mirror — the
pipeline had already split the mirror off as its own layer. `pipeline_assets.objects()`
still cuts the six props individually and is unchanged; it is simply not what
the endpoint hands the UI any more.

`torch` is **not installable on this machine**: PyTorch stopped publishing macOS
x86_64 wheels after 2.2, and 2.2 has no cp313 build. Everything below is
therefore `onnxruntime` on the CPU. Weights download once into `backend/models/`
on first use.

[`precise.py`](precise.py) runs the stage-06 idea with the models that install
here — propose, then localise, then matte:

| Step | Model | Answers |
| --- | --- | --- |
| Proposals | SegFormer-B4 ADE20K, 150 classes (246 MB) | what is in the room, roughly where |
| Boundary | SAM ViT-B, encoder + mask decoder (376 MB) | which pixels are actually the object |
| Edge | colour guided filter (He, Sun & Tang) | how much of each edge pixel is the object |

Each SegFormer component becomes a prompt: up to 3 positive points at the
distance transform's peaks, and up to 4 negative points on the ring around it
that belong to another class — those are what stop SAM swallowing the wall
behind a picture. SAM returns three candidates (part, subpart, whole) and the
one agreeing best with the proposal wins, its own IoU estimate breaking ties.

Instances that overlap by more than 60% of the smaller one are the same thing
found twice and collapse — a window split by its mullions into four
"windowpane" components comes back as one window. Containment merges only
*within* a label, so a pillow on a bed survives as its own object.

The instance masks are unioned into a layer *before* the matting pass, not
after. Matting each object separately and stacking the results leaves a seam
wherever two objects touch — the pillow's edge against the bed it lies on gets
feathered as though the bed were background. One union, one matte, and an
internal boundary like that is not an edge at all. `layers.refine_mask` cleans
the union first with OpenCV: closing at radius 1, connected-component filtering
for specks, and hole filling capped at 0.2% of the canvas so the gap between
chair legs stays open. There is no erosion and no opening anywhere in it — both
take pixels away, and the first pixels they take are leaf tips, lamp flex, chair
legs and the thin inner edge of a mirror frame.

The matting pass exists because a binary mask cannot be composited. It
staircases on every diagonal, and it carries a one-pixel rim of the old
background that reads as an outline over a new floor. `matting.refine` re-draws
the boundary as a soft alpha that follows the photo's own colour edges, moving
it at most 2 px from where SAM put it; `matting.decontaminate` then inpaints
every partially transparent pixel from the object's own interior, so what gets
blended is the object's colour and not the room it was cut from.

Measured on an i5-10400: **about 11 seconds** per photo — SegFormer ~3 s, the
SAM encoder ~6 s once, then ~30 ms per object for the decoder and the matte. A
bedroom returns 18 objects, a kitchen 24, a bathroom 8. Inference runs in a
worker thread so it does not stall the event loop.

Known limits: the instance split comes from SegFormer's components, so two
adjacent paintings that it merged into one region stay one object — SAM is only
asked to refine regions, not to find ones nobody proposed. ADE20K also has no
indoor word for a lot of furniture and reaches for the nearest outdoor one (an
ottoman as "rock"); those names are reported as `object` rather than the wrong
noun. `/generate` still uses the plain `segmentation.segment` path below.

## Fast segmentation — what `/generate` uses

[`segmentation.py`](segmentation.py) is the original single-model path:
SegFormer-B4 plus connected components, no SAM, no matting, about 3 seconds. It
is what `live_scene.py` consumes when `/generate` is handed an uncalibrated
photo, where the masks decide only which pixels to *protect* from tiling — a
job a binary mask does fine.

## Committed segmentation — the calibrated room

`pipeline_assets.py` reads the stage **06F4** result — the six main props the
pipeline settled on, with their names, source stages and pixel counts — and cuts
each one into a transparent PNG using its committed mask plus the **stage01
master RGB**. That follows the pipeline's own frozen rule: *final prop RGB must
come from the master input*, never from a generated image.

The six: complete vanity system, complete toilet system, shower fixture, wall
electrical plate, toilet paper holder, ceiling light.

Surfaces come from the stage02 three-model consensus (SegFormer + OneFormer +
Mask2Former, 2-of-3 majority) and are cut the same way.

Cut-outs are written once into `backend/generated/` and reused. Delete that
directory to force a rebuild.

## Floor and wall masks

    CLEAN_ROOM.png -> Floor + Wall detection -> FLOOR_MASK.png + WALL_MASK.png

`floor_wall.py` answers one question about a cleaned room: which pixels are
floor, and which are wall. **Two classes, two files.** It detects nothing else
and writes nothing else — no ceiling mask, no background mask, no per-class
anything. Both outputs are strictly binary, 0 or 255, at the source image's
exact resolution:

    CLEAN_ROOM.png
      -> ADE20K semantic segmentation      (surfaces.py, the session already open)
      -> floor and wall probabilities      the only two output classes
      -> each must beat every other class  mutually exclusive by construction
      -> guided-filter edge snap           (surfaces.snap)
      -> minimal cleanup                   specks out, enclosed pinholes in
      -> boundary refinement               band-limited graph cut onto the image edge
      -> FLOOR_MASK.png + WALL_MASK.png

`FLOOR_MASK.png` is white on floor pixels and black everywhere else;
`WALL_MASK.png` is white on wall pixels and black everywhere else. No pixel is
white in both.

**No new model and no new dependency.** `surfaces.py` is already an ADE20K
semantic segmenter, and this calls its cached ONNX session rather than loading a
second network.

It is not `surfaces.detect(rgb)["floor"]`. That takes a plain argmax over all
150 ADE20K classes, which is right for deciding which detections to reject and
loses floor wherever the network splits its confidence: a pixel scoring `floor`
0.31, `rug` 0.22, `wall` 0.34 is 53% floor and a plain argmax calls it wall.
Summing each surface's classes first fixes that, and makes the two masks
disjoint by construction rather than by a subtraction afterwards.

A pixel is then floor when floor wins outright and wall when wall wins
outright — against the other surface, and against the strongest single class
that is neither. That last comparison is the whole of how a ceiling stays out of
the wall mask: ADE20K still predicts `ceiling` on a ceiling and `door` on a
door, those predictions still win their own pixels, and a pixel they win is in
neither mask. Nothing needs a ceiling group, and nothing produces a ceiling
output.

Holding a surface against the strongest rival rather than the sum of all 149
others is what makes *complete* floor true: a floor at 0.40 against a long tail
of ten classes at 0.06 is floor, not a tie.

The two groups are short by measurement. Across the cleaned rooms in
`backend/jobs/`, ADE20K spends 93% of a cleaned room on three labels — wall
0.556, floor 0.268, ceiling 0.107. `curtain` (0.031) and `door` (0.013) are the
next two. All three are background here, along with every other ADE20K class.

Run on the **clean** room, not the photograph, for the reason `/generate`
already detects surfaces there: a chair standing on the floor is floor the model
cannot label, and `CLEAN_ROOM.png` is that floor rebuilt.

No erosion, no blur, no opening, no polygon fitting, no colour thresholding, and
no assumption about where in the frame a surface sits or which way a wall runs.
Cleanup is two proportional passes — enclosed gaps up to 2% of the mask sealed,
detached components under 0.1% dropped — both the same limits `live_scene` uses
on the surface masks it feeds the renderer.

Measured over 20 real rooms from 358×447 to 4000×2668: floor and wall found in
every one, zero overlapping pixels, every mask exactly the source resolution.
About 2 seconds for an ordinary room, 10 for a 10-megapixel one.

`POST /segment` writes both files into the job's `segments/` folder alongside
the layers and reports them as `floor_mask_png` / `wall_mask_png`.
`POST /floor-wall` runs only this stage on one uploaded image, which is the fast
way to look at a mask. Only the two masks are written. `FLOOR_OVERLAY.png` and `WALL_OVERLAY.png` —
the mask tinted over a copy of the source — are opt-in debug output
(`overlays=true` on `/floor-wall`, `overlays=True` on `floor_wall.write`). The source image
is never modified.

### Boundary refinement

The class map decides *what* is floor and wall; the refinement decides *where*
the line between them runs. The guided-filter snap puts a boundary on the image
edge but pixel by pixel, so it comes out ragged and dotted. The last step
re-cuts each boundary with OpenCV's GrabCut inside a narrow band around it
(1.2% of the long edge, 4–24 px). Pixels beyond the band are locked to the side
they're already on, and the other surface's core is locked out. So the boundary
can move onto the real edge, but the region can't be redrawn and can't grow into
its neighbour. The cut's colour model separates brown floor from a white
skirting, its contrast term lands it on the edge, and its length penalty keeps
it continuous.

It runs on 768 px tiles, re-cuts each tile seam, and finishes with a 3×3
majority vote inside the band plus the same proportional speck/pinhole limits
`_tidy` uses. It is deterministic (fixed seed).

Across 21 rooms: roughness 1.043 → 1.020 (1.000 is a clean line), edge
strength along the boundary 358 → 460, no room worse on either, largest area
change 2.7%. Cost: about 1–2 s per ordinary room, about 9 s at 4000×2668.
`detect_floor_wall(..., refine=False)` returns the masks exactly as before.

### What consumes the masks

[`perspective_engine/`](perspective_engine/README.md) does. It hands
`FLOOR_MASK.png` and `WALL_MASK.png`, as the strict clipping boundary, to the
tile engine in [`tiles_backend/perspective_engine/`](../tiles_backend/perspective_engine/README.md):
the reference project's tile renderer, migrated unchanged. `/generate` uses it
when the request names a Clean Room job that has both masks; the calibrated
room and uploads without a Clean Room run still render through `engine.py`,
exactly as before.

### Mask2Former

Mask2Former was the first choice and cannot run on this machine. PyTorch 2.2.2
is the last release with a macOS x86_64 wheel; it was compiled against NumPy 1.x
and raises `Numpy is not available` on every array conversion under the NumPy 2.x
this backend runs on, while the installed `opencv-python-headless` 5.0 requires
`numpy>=2`. Supporting it means downgrading NumPy and OpenCV underneath the whole
extraction pipeline. `detect_floor_wall(..., backend="mask2former")` raises that
explanation rather than silently substituting something else.


## Two geometry paths

`POST /generate` picks one automatically, by grayscale correlation against the
calibrated room (threshold 0.90):

**`fitted`** — the calibrated room. Uses the camera the pipeline's stage08g3
actually fitted. Verified byte-for-byte against the committed output; the room
dimensions on the form cannot override a measurement.

**`estimated`** — any other photo. [`live_scene.py`](live_scene.py) segments the
upload, takes the floor mask, and synthesises a camera: a level, unrolled
pinhole at 67° horizontal field of view, 1500 mm above the floor, with its pitch
solved by bisection so the predicted floor/wall junction lands on the photo's
own far floor edge (2nd percentile of floor rows). The room dimensions you enter
drive that solve, so they set the tile scale. The segmented objects are
composited back over the tiled floor. Renders at up to 1600 px on the longest
edge.

`engine.py` is untouched by this — the estimated path builds the same `Scene`
dataclass and runs the identical projection. So does nothing under `scripts/`.

The estimate assumes a level camera, a rectilinear lens, and a photographer
standing at one end of the room. A tilted or wide-angle shot will skew. Wall
tiling is rejected on this path (`surface` must be `floor`) because no
wall-plane geometry is estimated.

Measured end to end on an i5-10400: about **4 seconds** per uploaded room.

## Limits, stated plainly

**Only the calibrated room gets measured geometry.** Every other photo is tiled
on an *estimated* camera, as described above. It is a plausible reconstruction,
not a measurement: the pipeline's GPU stages (Qwen empty room, SAM2 masks,
stage08g3 fit) are what produce the real thing, and they need CUDA.

**The reconstruction behind an object is a reconstruction.** Both paths now tile
an emptied room — the calibrated one a Qwen render, an upload a LaMa fill inside
the object masks — and both put the objects back afterwards. What LaMa writes is
a plausible continuation of the surrounding floor and wall, not a photograph of
what was really there, and it is only as good as the mask it was given: a thin
leaf or a lamp rod the segmentation missed is not in the mask, so it is not
filled, and a faint trace of it survives in `CLEAN_ROOM.png`. A shadow the model
did not label as part of its object stays where it is for the same reason.
Measured `outside_changed` is 0 on every room tested — nothing outside the
masks is ever rewritten.

**`surface=wall` tiles the skirting course only**, and only on the calibrated
path (up to 180 mm, the extent of `08b2_wall_screeding_mask`). Full wall tiling
would need a wall-plane mask, which stage 08b2 does not produce. On the
estimated path, wall and both are rejected.

**Progress is still simulated.** The frontend's five steps run on a fixed 700 ms
timer in `TileVisualizer.tsx` and are not driven by the server.

## Jobs

Each request writes `backend/jobs/<job_id>/` with `original.png`, `tile.png`,
`raw.png` (exact texture, no relighting), `lit.png` (relit, no props),
`result.png` (final composite) and `state.json`. Nothing prunes them.
