"""
HTTP layer for the AI Tile Visualizer.

Implements the contract the frontend already expects:

    POST /generate   multipart: room_image, tile_image,
                                room_width, room_length, room_height,
                                tile_width, tile_height, rotation, surface
    ->               { original_image_url, result_image_url, job_id }

Run it with:

    backend/.venv/bin/uvicorn app:app --app-dir backend --reload --port 8000
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import re
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

import floor_wall
import live_scene
import perspective_engine
from tiles_backend.perspective_engine import depth as perspective_engine_depth
import pipeline_assets
import reuse
import scene as scene_module
import surfaces
from engine import GROUT_MM, MissingGeometryError, TileSpec, render
from extraction import config as extraction_config
from extraction import debug as extraction_debug
from extraction import dino, inpaint, output, sam2
from extraction.extractor import ExtractionError, extract

JOBS_DIR = Path(__file__).resolve().parent / "jobs"
JOBS_DIR.mkdir(exist_ok=True)

pipeline_assets.GENERATED.mkdir(exist_ok=True)

ALLOWED_ROTATIONS = {0, 45, 90, 135}
ALLOWED_SURFACES = {"floor", "wall", "both"}

# The walls `live_scene` builds for an uploaded room, by name.
ALLOWED_WALLS = {"left", "right", "back"}

# Correlation above which an uploaded room is taken to *be* the calibrated room.
SCENE_MATCH_THRESHOLD = 0.90

MM_PER_FOOT = 304.8

# Longest edge an uploaded photo is rendered at. The projection is per-pixel, so
# this bounds both the time and the memory a large upload can cost.
MAX_RENDER_EDGE = 1600

# Largest upload accepted at all, in pixels. Refusing early beats running three
# models over something that will exhaust memory in the matting pass.
MAX_UPLOAD_PIXELS = 40_000_000

app = FastAPI(title="AI Tile Visualizer", version="1.0.0")

# Which browser origins may call this API.
#
# Local development is always allowed, unchanged. A Cloudflare Tunnel mints a
# fresh `*.trycloudflare.com` hostname on every run, so the frontend's origin
# cannot be written down here — the pattern admits that whole domain instead,
# and `CORS_ORIGIN_REGEX` overrides it entirely for a named tunnel on your own
# domain. Nothing external is allowed unless a tunnel is actually running.
LOCAL_ORIGINS = r"http://(localhost|127\.0\.0\.1)(:\d+)?"

CLOUDFLARE_ORIGINS = r"https://[a-z0-9-]+\.trycloudflare\.com"

CORS_ORIGIN_REGEX = os.environ.get(
    "CORS_ORIGIN_REGEX", f"({LOCAL_ORIGINS}|{CLOUDFLARE_ORIGINS})"
)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=CORS_ORIGIN_REGEX,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/jobs", StaticFiles(directory=JOBS_DIR), name="jobs")

# Committed stage outputs, read-only, so the UI can show the pipeline's own work.
app.mount("/assets", StaticFiles(directory=scene_module.PROD), name="assets")

# Object and surface cut-outs produced by pipeline_assets.
app.mount("/generated", StaticFiles(directory=pipeline_assets.GENERATED), name="generated")


def _decode(upload: UploadFile, data: bytes, field: str) -> np.ndarray:
    if not data:
        raise HTTPException(422, f"{field} is empty.")

    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            422, f"{field} could not be decoded as an image (got {upload.filename!r})."
        ) from None

    # Phones record orientation in EXIF rather than rotating the pixels. Left
    # uncorrected, a portrait photo reaches the models on its side: every box
    # and every mask then comes back rotated with respect to what the user sees,
    # and no amount of care further down recovers the alignment. Applying it
    # here means every stage after this point works in display orientation.
    try:
        image = ImageOps.exif_transpose(image)
    except Exception:  # a malformed EXIF block must not lose the upload
        pass

    if image.width * image.height > MAX_UPLOAD_PIXELS:
        raise HTTPException(
            422,
            f"{field} is {image.width}x{image.height}, larger than this server accepts "
            f"({MAX_UPLOAD_PIXELS // 1_000_000} megapixels).",
        )

    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _fit_within(rgb: np.ndarray, longest_edge: int) -> np.ndarray:
    """Downscale so the longest edge is at most `longest_edge`. No upscaling."""
    height, width = rgb.shape[:2]

    scale = longest_edge / max(height, width)

    if scale >= 1.0:
        return rgb

    resized = Image.fromarray(rgb).resize(
        (max(1, round(width * scale)), max(1, round(height * scale))), Image.LANCZOS
    )

    return np.asarray(resized, dtype=np.uint8)


def _positive(value: float, field: str) -> float:
    if not np.isfinite(value) or value <= 0:
        raise HTTPException(422, f"{field} must be greater than zero.")

    return float(value)


def _wall_note(geometry: dict) -> str:
    """What the response says about wall tiling. Identical on both paths."""
    walls = geometry.get("walls") or []

    if not walls:
        return "No wall was segmented in this photo, so only the floor could be tiled."

    return (
        "Walls are tiled on the estimated room box — "
        + ", ".join(plane["label"] for plane in walls)
        + " — each on its own plane, so the grid follows that wall's own direction. "
        "Their size comes from the room dimensions you entered "
        f"({geometry['room_mm'][0] / MM_PER_FOOT:.0f} × "
        f"{geometry['room_mm'][1] / MM_PER_FOOT:.0f} × "
        f"{geometry['room_mm'][2] / MM_PER_FOOT:.0f} ft), so those matter more for "
        "wall tiling than for floor tiling."
    )


def _tile_engine_notes(geometry: dict, clip: dict, job_id: str) -> list[str]:
    """What the response says when tiles_backend.perspective_engine placed them."""
    notes = []

    floor = geometry.get("floor")

    if floor:
        vp = {
            "auto": "detected from the room's own floor lines",
            "manual-low-confidence": "the estimated camera's horizon — the detected one "
            f"scored {floor['vp_confidence']:.2f}, under the engine's 0.55 gate",
            "fallback-after-out-of-view": "the estimated camera's horizon — the detected "
            "one passed the engine's gate but put the floor out of view",
        }.get(floor["vp_source"], floor["vp_source"])

        room = geometry.get("room") or {}
        scale_note = (
            f"scale from the room geometry's camera height, {room['camera_height_mm']:.0f} mm "
            f"({room.get('camera_height_source')})"
            if room.get("camera_height_mm") else f"scale from a {1500:.0f} mm camera height"
        )
        notes.append(
            f"Floor: vanishing point {vp}; focal length {floor['focal_px']:.0f} px "
            f"({floor['focal_source']}); {scale_note}. "
            f"The visible floor reconstructs {floor['surface_width_mm'] / 1000:.2f} × "
            f"{floor['surface_depth_mm'] / 1000:.2f} m — "
            f"{floor['tiles_across']:.1f} × {floor['tiles_deep']:.1f} tiles."
        )

    wall = geometry.get("wall")

    if wall:
        described = ", ".join(
            f"{entry['wall_width_mm'] / 1000:.2f} × {entry['wall_height_mm'] / 1000:.2f} m"
            for entry in wall["walls"]
        )

        chosen = geometry.get("walls_selected")

        placed = (wall.get("scale_source") == "room-geometry")
        notes.append(
            f"{len(wall['walls'])} wall(s), each on its own plane"
            + (", placed from the room geometry on the same scale as the floor"
               if placed else " and scaled from the room height you entered")
            + f": {described or 'none rendered'}."
            + (f" Kept: {', '.join(chosen)}." if isinstance(chosen, list) else "")
        )

        if wall.get("failures"):
            notes.append(
                "Skipped walls: "
                + "; ".join(f"{item['instance']}: {item['error']}" for item in wall["failures"])
            )

    room = geometry.get("room")
    if room and room.get("status") != "INSUFFICIENT":
        mm = lambda v: "?" if v is None else f"{v / 1000:.2f} m"  # noqa: E731
        err = room.get("reprojection_error_mean_px")
        status_text = {
            "OK": "Room geometry agrees with the room size you entered",
            "ESTIMATED": "Room size estimated from the photo (no size entered to fix the scale)",
            "CONFLICT": "The room size you entered disagrees with the photo's geometry",
        }.get(room["status"], room["status"])
        notes.append(
            f"{status_text}: width {mm(room.get('width_mm'))} ({room.get('width_source')}), "
            f"length {mm(room.get('length_mm'))} ({room.get('length_source')}"
            + (", at least" if room.get("length_is_lower_bound") else "") + "), "
            f"height {mm(room.get('height_mm'))} ({room.get('height_source')}); camera "
            f"{room['camera_height_mm'] / 1000:.2f} m high"
            + (f"; corners reproject within {err:.0f} px on average" if err is not None else "")
            + (f"; residuals {room.get('residuals')}" if room["status"] == "CONFLICT" else "")
            + "."
        )

    for name, reason in (geometry.get("skipped") or {}).items():
        notes.append(f"{name.capitalize()} not tiled: {reason}.")

    notes.append(
        f"{geometry['object_count']} objects from this room's Clean Room run (job {job_id}) "
        "were put back over the tiles from the original photograph. Tiles were clipped "
        f"to FLOOR_MASK.png and WALL_MASK.png: {clip['outside_mask_after_clip']} tile "
        "pixels fall outside them."
    )

    return notes


def _accepted_union(job_dir: Path) -> np.ndarray:
    """
    Every pixel the two written layers claim, at the photo's own size.

    Taken at `alpha > 0` rather than at the alpha cutoff: a pixel a layer holds
    even weakly is a pixel that will be composited back over the tiles, so the
    renderer has to stop the tiles there too.
    """
    union: np.ndarray | None = None

    for name in (output.ALL_OBJECTS_FILENAME, output.MIRRORS_FILENAME):
        layer = np.array(Image.open(job_dir / name).convert("RGBA"))

        claimed = layer[..., 3] > 0

        union = claimed if union is None else (union | claimed)

    return union if union is not None else np.zeros((1, 1), dtype=bool)


def _render_clean(job_dir: Path, render_rgb: np.ndarray) -> np.ndarray:
    """
    The clean room at render resolution, or the photo if there is no clean room.

    Falling back to the photograph is the pre-inpainting behaviour and is always
    safe: tiles then land on the room as photographed, which is what this
    endpoint did before the fill existed.
    """
    path = job_dir / output.CLEAN_ROOM_FILENAME

    if not path.exists():
        return render_rgb

    clean = np.array(Image.open(path).convert("RGB"))

    if clean.shape[:2] == render_rgb.shape[:2]:
        return clean

    return reuse.fit(clean, max(render_rgb.shape[:2]))


def _detect_floor_wall(job_dir: Path, rgb: np.ndarray) -> dict:
    """
    Run the floor/wall stage on this job's cleaned room and write its masks.

    The cleaned room is the right input and the reason is the whole point of
    running it here: a chair standing on the floor is floor the model cannot
    label, and `CLEAN_ROOM.png` is that floor rebuilt. It is read at its own
    full resolution — the same size as the uploaded photo — so `FLOOR_MASK.png`
    and `WALL_MASK.png` come out pixel-aligned with every other file in the
    folder.

    Falls back to the photograph when the fill could not run, and reports a
    failure rather than raising one: this stage is additive, and a room that
    cannot be split into floor and wall is still a room that segmented, cleaned
    and rendered correctly.
    """
    source_path = job_dir / output.CLEAN_ROOM_FILENAME

    cleaned = source_path.exists()

    source = np.array(Image.open(source_path).convert("RGB")) if cleaned else rgb

    try:
        result = floor_wall.detect_floor_wall(source)
    except (floor_wall.FloorWallUnavailable, ValueError) as error:
        return {"available": False, "reason": str(error)}

    written = floor_wall.write(source, result, job_dir)

    return {
        "available": True,
        "source": output.CLEAN_ROOM_FILENAME if cleaned else "original photograph",
        "on_clean_room": cleaned,
        **written,
    }


def _detect_geometry(job_dir: Path, render_rgb: np.ndarray, render_clean: np.ndarray, payload: bytes) -> dict:
    """Detect and store the room's floor/wall geometry. Reports, never raises."""
    try:
        surface_masks = perspective_engine.load(job_dir)

        if surface_masks is None:
            return {"available": False, "reason": "no FLOOR_MASK.png / WALL_MASK.png"}

        perspective_engine.ensure_geometry(
            job_dir, render_rgb, surface_masks[0], surface_masks[1],
            clean=render_clean, photo_bytes=payload,
        )
    except Exception as error:  # additive stage: never fail the clean room
        return {"available": False, "reason": f"{type(error).__name__}: {error}"}

    return {"available": True}


def _geometry_urls(info: dict, prefix: str) -> dict:
    """The floor/ and wall/ files for the response."""
    if not info.get("available"):
        return {"available": False, "reason": info.get("reason")}

    return {
        "available": True,
        "floor": {
            "mask": f"{prefix}/floor/floor_mask.png",
            "geometry": f"{prefix}/floor/floor_geometry.json",
            "vanishing_points": f"{prefix}/floor/floor_vanishing_points.json",
        },
        "wall": {
            "mask": f"{prefix}/wall/wall_mask.png",
            "geometry": f"{prefix}/wall/wall_geometry.json",
            "vanishing_points": f"{prefix}/wall/wall_vanishing_points.json",
        },
    }


def _to_shape(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """
    A boolean mask at another resolution.

    Resized as a float and thresholded at 0.5, never nearest-neighbour: this is
    the same rule `extraction.to_original` uses on the way up, so a boundary
    survives the round trip without gaining the grid's staircase.
    """
    if mask.shape == shape:
        return mask

    resized = cv2.resize(
        mask.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR
    )

    return resized >= 0.5


def _clean_for_render(rgb: np.ndarray, objects: list) -> tuple[np.ndarray | None, dict]:
    """
    The emptied room for the render path, or `None` if it could not be made.

    `/segment` builds its `CLEAN_ROOM.png` from the written layers; this builds
    the same thing from the masks in hand, because `/generate` never writes
    those files. Same module, same fill, same rule — inside the object masks and
    nowhere else.

    Returning `None` rather than raising is deliberate: an unavailable
    inpainting model must not take tile rendering down with it. The caller falls
    back to the photograph, which is exactly what this endpoint did before.
    """
    if not objects:
        return None, {"available": False, "reason": "no objects were detected"}

    hole = np.zeros(rgb.shape[:2], dtype=bool)

    for item in objects:
        if item.mask.shape == hole.shape:
            hole |= item.mask

    if not hole.any():
        return None, {"available": False, "reason": "no object pixels to fill"}

    try:
        clean, info = inpaint.clean_room(rgb, hole)
    except inpaint.InpaintUnavailable as error:
        return None, {"available": False, "reason": str(error)}
    except Exception as error:  # onnxruntime raises its own failure types
        return None, {"available": False, "reason": f"inpainting failed: {error}"}

    info["available"] = True

    return clean, info


@app.get("/health")
def health() -> dict:
    """Report whether this server can actually render, and for which room."""
    missing = [str(path.relative_to(scene_module.PROJECT_ROOT)) for path in scene_module.missing_assets()]

    payload: dict = {
        "status": "ok" if not missing else "degraded",
        "missing_assets": missing,
        "object_extraction": {
            "flow": [
                "Grounding DINO tiny (ONNX, CPU) — detection + boxes",
                "SAM 2.1 Hiera-large (ONNX, CPU) — box-prompted masks",
                "OpenCV — deterministic mask cleanup",
                "colour guided filter — alpha + foreground decontamination",
                "LaMa (ONNX, CPU) — rebuild the room inside the object masks",
            ],
            "weights_present": dino.available() and sam2.available(),
            "detection_weights_present": dino.available(),
            "segmentation_weights_present": sam2.available(),
            "inpainting_weights_present": inpaint.available(),
        },
        "surfaces": {
            "model": "SegFormer-B4 ADE20K (ONNX, CPU)",
            "role": "wall/floor/ceiling only — not an object detector",
            "weights_present": surfaces.model_available(),
        },
        "tile_engine": {
            "package": "tiles_backend.perspective_engine",
            "source": "reference project's tile renderer, migrated unchanged",
            "used_by": "POST /generate with a Clean Room job that has FLOOR_MASK + WALL_MASK",
            "depth_model": "MiDaS DPT-Hybrid-384 (ONNX, CPU)",
            "depth_weights_present": perspective_engine_depth.available(),
        },
        "floor_wall": {
            "model": "SegFormer-B4 ADE20K (ONNX, CPU) — the same session, grouped by class",
            "role": "FLOOR_MASK.png + WALL_MASK.png for a cleaned room",
            "endpoint": "POST /floor-wall",
            "weights_present": surfaces.model_available(),
        },
    }

    if not missing:
        loaded = scene_module.load_scene()

        payload["calibrated_scene"] = {
            "canvas": [loaded.width, loaded.height],
            "room_mm": [loaded.room_u_mm, loaded.room_v_mm, loaded.room_height_mm],
            "source": "test07/production_pipeline (stage07 empty room + stage08g3 camera fit)",
        }

    return payload


@app.get("/pipeline/flow")
def pipeline_flow(request: Request) -> dict:
    """The nine stages in order, each with a preview image where one exists."""
    base = str(request.base_url).rstrip("/")

    stages = []

    for stage in pipeline_assets.flow():
        preview = stage.pop("preview_path")

        stage["preview_url"] = f"{base}/assets/{preview}" if preview else None

        stages.append(stage)

    return {"stages": stages}


@app.get("/pipeline/segments")
def pipeline_segments(request: Request) -> dict:
    """
    The calibrated room's committed segmentation, as the same two layers
    `/segment` returns: `ALL_OBJECTS.png` and `MIRRORS_ONLY.png`, plus the
    wall/floor/ceiling surface split.

    The props come from stage 06F4 and the mirror from stage 03, merged into the
    two layers. No per-prop PNG is written or served.
    """
    base = str(request.base_url).rstrip("/")

    def expand(entry: dict) -> dict:
        # A copy: object_layers() is cached, so its dicts outlive the request.
        expanded = {key: value for key, value in entry.items() if key != "mask_path"}

        expanded["cutout_url"] = f"{base}/generated/{expanded.pop('cutout_path')}"

        if entry.get("mask_path"):
            expanded["mask_url"] = f"{base}/assets/{entry['mask_path']}"

        return expanded

    composed, detections = pipeline_assets.object_layers()

    objects = [expand(item) for item in composed]

    # Local name kept distinct from the `surfaces` module imported above.
    surface_entries = [expand(item) for item in pipeline_assets.surfaces()]

    return {
        "room_url": f"{base}/assets/stage01_master/00_master_input.png",
        "objects": objects,
        "surfaces": surface_entries,
        "object_count": len(objects),
        "detections": detections,
        "source": "pipeline",
    }


# ---------------------------------------------------------------------------
# Long-running segmentation, without holding the connection open.
#
# `/segment` does 90-500 seconds of CPU work. Behind a reverse proxy that is a
# problem the code cannot solve from inside: Cloudflare gives up on any request
# after about 100 seconds and answers 524 itself. Measured on this backend, a
# public POST /segment returned `HTTP 524 after 125.6s` while the run completed
# normally server-side in 94.69s — the work succeeded, the answer had nowhere
# to go, and the browser reported "Failed to fetch".
#
# So the wait is moved out of the request. `/segment/start` hands back a token
# immediately, the work proceeds in the background, and `/segment/status/{id}`
# is polled. Every request is then milliseconds long and no proxy timeout can
# apply. The original synchronous `/segment` is untouched below and still works
# for local use and any existing caller.
#
# The registry is in-process and deliberately small: results are the same job
# folders already written to disk, so a restart costs a re-run, not data.
SEGMENT_JOBS: dict[str, dict] = {}

# How long a finished result stays available to be collected.
SEGMENT_JOB_TTL_S = 3600


def _forget_old_segment_jobs() -> None:
    """Drop results nobody collected, so the registry cannot grow forever."""
    cutoff = time.time() - SEGMENT_JOB_TTL_S

    for token in [k for k, v in SEGMENT_JOBS.items() if v.get("finished", 0) and v["finished"] < cutoff]:
        SEGMENT_JOBS.pop(token, None)


async def _segment_core(payload: bytes, rgb: np.ndarray, debug: bool, base: str) -> dict:
    """
    Extract an uploaded room photo into two full-canvas union layers.

        Grounding DINO   what objects are present, and where  (boxes + scores)
             |
        SAM 2.1          which exact pixels belong to each box (box-prompted)
             |
        OpenCV           deterministic mask cleanup
             |
        matting          alpha that follows the real edge, background colour
                         taken back out of it so nothing shows a rim
             |
        ALL_OBJECTS.png  every non-mirror object, full canvas
        MIRRORS_ONLY.png every mirror, full canvas

    Needs no calibrated geometry, so it works on any photo. Every RGB pixel in
    every output is a pixel of the upload at its original coordinates — nothing
    here generates, repaints, moves or resizes an object.

    Pass `debug=true` to also write the per-stage renders under `debug/`, which
    is the way to see whether a bad cut-out came from the detector, the
    segmenter or the cleanup.
    """
    started = time.perf_counter()

    config = extraction_config.load(debug=debug or None)

    job_id = uuid.uuid4().hex[:12]

    job_dir = JOBS_DIR / job_id / "segments"

    def analyse():
        # Inference is tens of seconds of blocking CPU work, so it goes to a
        # worker thread rather than stalling every other request on the loop.
        result = extract(rgb, config)

        metadata = output.write(rgb, result, config, job_dir)

        # ---- the handoff to /generate -----------------------------------
        #
        # Built from the files just written rather than from `result.objects`,
        # because `output.write` is where a mask is finally accepted: it
        # revalidates, drops duplicates and settles the two layers against each
        # other. What landed in those two PNGs is what "the accepted masks"
        # means, so that is what the renderer is given.
        clock = time.perf_counter()

        accepted = _accepted_union(job_dir)

        render_rgb = reuse.fit(rgb, MAX_RENDER_EDGE)

        render_clean = _render_clean(job_dir, render_rgb)

        props = _to_shape(accepted, render_rgb.shape[:2])

        # Surfaces are detected on the *clean* room, which is the one place
        # this costs an extra forward pass — and it is the pass `/generate` no
        # longer has to run, not an additional one.
        found = surfaces.instances(render_clean)

        metadata["timings"]["handoff_s"] = round(time.perf_counter() - clock, 2)

        # ---- floor + wall, on the room the objects just left ---------------
        #
        # Additive: nothing above or below reads this, and the two masks are
        # written beside the layers rather than into them. It runs here because
        # this is the thread that already has the cleaned room on disk, and
        # because doing it anywhere else would mean a second forward pass.
        clock = time.perf_counter()

        metadata["floor_wall"] = _detect_floor_wall(job_dir, rgb)

        metadata["timings"]["floor_wall_s"] = round(time.perf_counter() - clock, 2)

        # ---- floor + wall geometry, from the masks just written ------------
        #
        # The floor's vanishing points and plane, and every wall's own
        # orientation, detected once on the render-resolution clean room and
        # stored in segments/floor/ and segments/wall/. Every tile render of
        # this room then projects with this geometry. Additive: a failure here
        # is reported and the room stays usable — the render detects it then.
        clock = time.perf_counter()

        metadata["geometry"] = _detect_geometry(job_dir, render_rgb, render_clean, payload)

        metadata["timings"]["geometry_s"] = round(time.perf_counter() - clock, 2)

        reuse.save(
            JOBS_DIR / job_id,
            render_rgb,
            render_clean,
            props,
            found,
            {
                "source": reuse.digest(payload),
                "object_count": metadata["counts"]["accepted"],
                "clean_room": {
                    key: value
                    for key, value in (metadata.get("clean_room") or {}).items()
                    if key != "checks"
                },
                "timings": metadata["timings"],
            },
        )

        return result, metadata

    try:
        result, metadata = await run_in_threadpool(analyse)
    except ExtractionError as error:
        # A missing model or an unusable image: the reason is fit to show.
        raise HTTPException(503, str(error)) from error
    except MemoryError as error:
        raise HTTPException(
            507, "Ran out of memory extracting this photo. Try a smaller image."
        ) from error
    except Exception as error:  # onnxruntime raises its own failure types
        raise HTTPException(500, f"Object extraction failed: {error}") from error

    Image.fromarray(rgb).save(JOBS_DIR / job_id / "original.png")

    prefix = f"{base}/jobs/{job_id}/segments"

    debug_files: list[str] = []

    if config.debug:
        work = _fit_within(rgb, config.max_inference_edge)

        debug_dir = job_dir / "debug"

        debug_files = await run_in_threadpool(extraction_debug.write, work, result, debug_dir)

        debug_files += extraction_debug.write_layer_previews(
            debug_dir, job_dir, (output.ALL_OBJECTS_FILENAME, output.MIRRORS_FILENAME)
        )

    height, width = rgb.shape[:2]

    # Every detection the pipeline considered, accepted or not, in the shape the
    # panel's detection list already renders. A rejected detection carries the
    # reason it was rejected, which is the whole point of showing them.
    detections = [
        {
            "name": record["label"],
            "status": "mirror" if record["is_mirror"] else "object",
            "pixels": record["alpha_area"],
            "bbox": record["bbox"],
            "score": record["confidence"],
            "layers": [record["layer"]],
        }
        for record in metadata["objects"]
    ] + [
        {
            "name": entry.get("label", entry.get("id", "unknown")),
            "status": "excluded",
            "score": entry.get("confidence"),
            "bbox": entry.get("box"),
            "reason": entry.get("reason"),
        }
        for entry in metadata["rejected"]
    ]

    stage = "GROUNDING_DINO + SAM2.1 + OPENCV + GUIDED_FILTER_MATTE"

    # Exactly two entries, whatever the photo holds — the same shape this
    # endpoint has always returned. There is deliberately no per-object image
    # here: objects are composited into the layer they belong to, and their
    # individual extents are reported in `detections` instead.
    layer_entries = [
        {
            "id": "all_objects",
            "name": "All objects",
            "filename": output.ALL_OBJECTS_FILENAME,
            "source_stage": stage,
            "pixels": sum(
                r["alpha_area"] for r in metadata["objects"] if not r["is_mirror"]
            ),
            "bbox": [0, 0, width, height],
            "cutout_url": f"{prefix}/{output.ALL_OBJECTS_FILENAME}",
            "members": [r["label"] for r in metadata["objects"] if not r["is_mirror"]],
        },
        {
            "id": "mirrors",
            "name": "Mirrors",
            "filename": output.MIRRORS_FILENAME,
            "source_stage": stage,
            "pixels": sum(r["alpha_area"] for r in metadata["objects"] if r["is_mirror"]),
            "bbox": [0, 0, width, height],
            "cutout_url": f"{prefix}/{output.MIRRORS_FILENAME}",
            "members": [r["label"] for r in metadata["objects"] if r["is_mirror"]],
        },
    ]

    # The wall/floor/ceiling split the extraction already computed, reported
    # rather than discarded. No extra inference runs for this: `extract` needed
    # these masks for structural rejection and they are already in hand. The
    # `surfaces` field has always been part of this response — it was simply
    # being returned empty.
    #
    # Coverage is what a caller needs: a photograph with no floor in it is not
    # a room, and this is the only reliable way to know before rendering.
    surface_entries = []

    for label, mask in sorted(result.surface_masks.items()):
        pixels = int(mask.sum())

        if pixels == 0:
            continue

        surface_entries.append(
            {
                "id": label,
                "name": label.capitalize(),
                "source_stage": "SEGFORMER_B4_ADE20K",
                "pixels": pixels,
                "coverage": round(pixels / float(mask.size), 5),
                "bbox": [0, 0, int(width), int(height)],
            }
        )

    # The emptied room, when the inpainting stage produced one. It is additive:
    # `clean_room_png` is null if the fill could not run, and every other field
    # in this response is exactly what it was before.
    clean_room = dict(metadata.get("clean_room") or {})

    clean_room_png = (
        f"{prefix}/{output.CLEAN_ROOM_FILENAME}" if clean_room.get("available") else None
    )

    if clean_room_png:
        clean_room["url"] = clean_room_png

    # The floor/wall split of that cleaned room. Every field here is new; the
    # rest of this response is byte-for-byte what it was before the stage
    # existed, so an existing caller sees no change at all.
    surface_masks = dict(metadata.get("floor_wall") or {})

    files = surface_masks.pop("files", {})

    for key, filename in files.items():
        surface_masks[f"{key}_url"] = f"{prefix}/{filename}"

    floor_mask_png = surface_masks.get("floor_mask_url")
    wall_mask_png = surface_masks.get("wall_mask_url")

    metadata["timings"]["total_s"] = round(time.perf_counter() - started, 2)

    print(f"[segment {job_id}] {metadata['timings']}", flush=True)

    return {
        "success": True,
        "job_id": job_id,
        "room_url": f"{base}/jobs/{job_id}/original.png",
        "objects": layer_entries,
        "surfaces": surface_entries,
        "object_count": len(layer_entries),
        "all_objects_png": f"{prefix}/{output.ALL_OBJECTS_FILENAME}",
        "mirrors_only_png": f"{prefix}/{output.MIRRORS_FILENAME}",
        "clean_room_png": clean_room_png,
        "clean_room": clean_room,
        "floor_mask_png": floor_mask_png,
        "wall_mask_png": wall_mask_png,
        "floor_wall": surface_masks,
        # The floor and wall geometry folders (null when detection failed —
        # `geometry.reason` says why).
        "geometry": _geometry_urls(metadata.get("geometry") or {}, prefix),
        "detections": detections,
        "rejected": metadata["rejected"],
        "counts": metadata["counts"],
        "timings": metadata["timings"],
        "metadata_url": f"{prefix}/objects.json",
        "debug": {
            "enabled": config.debug,
            "files": [f"{prefix}/debug/{name}" for name in debug_files],
        },
        "source": "live",
    }


@app.post("/segment")
async def segment_room(
    request: Request,
    room_image: UploadFile = File(...),
    debug: bool = Form(False),
) -> dict:
    """The original synchronous call, unchanged in behaviour and response."""
    payload = await room_image.read()

    rgb = _decode(room_image, payload, "room_image")

    return await _segment_core(payload, rgb, debug, str(request.base_url).rstrip("/"))


@app.post("/segment/start")
async def segment_start(
    request: Request,
    room_image: UploadFile = File(...),
    debug: bool = Form(False),
) -> dict:
    """
    Begin a segmentation and return a token at once.

    The upload is decoded here rather than in the worker so an unusable image
    is still rejected with a 422 the caller can read, exactly as the
    synchronous endpoint does.
    """
    payload = await room_image.read()

    rgb = _decode(room_image, payload, "room_image")

    _forget_old_segment_jobs()

    token = uuid.uuid4().hex[:12]

    SEGMENT_JOBS[token] = {"status": "running", "started": time.time(), "finished": 0}

    base = str(request.base_url).rstrip("/")

    async def run() -> None:
        try:
            result = await _segment_core(payload, rgb, debug, base)

            SEGMENT_JOBS[token].update(status="done", result=result)
        except HTTPException as error:
            SEGMENT_JOBS[token].update(status="error", detail=str(error.detail), code=error.status_code)
        except Exception as error:  # a worker must never die silently
            SEGMENT_JOBS[token].update(status="error", detail=str(error), code=500)
        finally:
            SEGMENT_JOBS[token]["finished"] = time.time()

    asyncio.create_task(run())

    return {"job_id": token, "status": "running"}


@app.get("/segment/status/{token}")
def segment_status(token: str) -> dict:
    """
    Where a background segmentation has got to.

    `running` carries the elapsed seconds so a caller can show progress;
    `done` carries the identical payload the synchronous endpoint returns, so
    nothing downstream needs to know which route was used.
    """
    job = SEGMENT_JOBS.get(token)

    if job is None:
        raise HTTPException(404, f"No segmentation job {token!r}. It may have expired.")

    if job["status"] == "running":
        return {"status": "running", "elapsed_s": round(time.time() - job["started"], 1)}

    if job["status"] == "error":
        raise HTTPException(job.get("code", 500), job.get("detail", "Segmentation failed."))

    return {"status": "done", **job["result"]}


@app.post("/floor-wall")
async def floor_wall_masks(
    request: Request,
    room_image: UploadFile = File(...),
    overlays: bool = Form(False),
) -> dict:
    """
    Floor and wall masks for one room image, on their own.

    Hand it a `CLEAN_ROOM.png` — the room with its objects already removed — and
    it returns `FLOOR_MASK.png` and `WALL_MASK.png`: white where the surface is,
    black everywhere else, at the uploaded image's exact resolution.

    Nothing else runs. No detection, no extraction, no inpainting, no render —
    one forward pass of the ADE20K segmentation the backend already loads. That
    makes it about three seconds rather than the minute and a half `/segment`
    costs, which is what makes it usable for checking a mask.

    `POST /segment` produces the same two files for the room it cleans, under
    the same names; this endpoint exists for running the stage against an image
    that has already been cleaned, or against anything else worth testing.
    """
    payload = await room_image.read()

    rgb = _decode(room_image, payload, "room_image")

    job_id = uuid.uuid4().hex[:12]

    job_dir = JOBS_DIR / job_id / "segments"

    started = time.perf_counter()

    def analyse():
        result = floor_wall.detect_floor_wall(rgb)

        return result, floor_wall.write(rgb, result, job_dir, overlays=overlays)

    try:
        result, written = await run_in_threadpool(analyse)
    except floor_wall.FloorWallUnavailable as error:
        raise HTTPException(503, str(error)) from error
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    except MemoryError as error:
        raise HTTPException(
            507, "Ran out of memory segmenting this image. Try a smaller one."
        ) from error
    except Exception as error:
        raise HTTPException(500, f"Floor/wall detection failed: {error}") from error

    Image.fromarray(rgb).save(JOBS_DIR / job_id / "original.png")

    prefix = f"{str(request.base_url).rstrip('/')}/jobs/{job_id}/segments"

    urls = {f"{key}_url": f"{prefix}/{name}" for key, name in written["files"].items()}

    return {
        "success": True,
        "job_id": job_id,
        "room_url": f"{str(request.base_url).rstrip('/')}/jobs/{job_id}/original.png",
        "floor_mask_png": urls["floor_mask_url"],
        "wall_mask_png": urls["wall_mask_url"],
        **urls,
        "stats": written["stats"],
        "elapsed_s": round(time.perf_counter() - started, 2),
    }

# ---------------------------------------------------------------------------
# Surface selection: tile a surface by selecting it, untile it by deselecting.
#
#   POST /surfaces   what a cleaned room offers: the floor, and every wall the
#                    tile engine will tile, each with a select dot
#   POST /generate   renders one surface kind (floor, or every wall) and keeps
#                    each surface's tiled pixels as a layer
#   POST /compose    the photograph, with only the selected surfaces' tiles
#
# `/compose` starts from the original photograph and copies tiled pixels in
# only inside the selected surfaces' own regions. So a deselected surface is
# the photograph, exactly, and switching one wall off cannot touch another
# wall or the floor — each is its own region. No tile is rendered to compose:
# selecting and deselecting is instant once a surface has been rendered.

LAYERS_FILENAME = "layers.npz"

JOB_ID_PATTERN = re.compile(r"^[0-9a-f]{12}$")


def _save_layers(job_dir: Path, regions: dict) -> None:
    """Every surface region of a render, bit-packed, keyed by surface id."""
    shape = next(iter(regions.values())).shape

    np.savez_compressed(
        job_dir / LAYERS_FILENAME,
        shape=np.array(shape),
        **{key: np.packbits(mask.astype(bool), axis=None) for key, mask in regions.items()},
    )


THREE_FILENAME = "three.json"
ROOM_DEBUG_FILENAME = "room_debug.png"


def _save_three(job_dir: Path, rendered) -> None:
    """
    What the optional 3D view needs to redraw this render's tiles, next to it:

      three.json               per surface, the render state the tile engine used
      three/<surface>.png      that surface's tiled pixels (white) — the clip mask
      three/objects.png        the room's objects, over the tiles, as the 2D
                               composite restores them (alpha = object coverage)

    Written from the same render that produced result.png; nothing is recomputed.
    """
    if not rendered.three:
        return

    folder = job_dir / "three"
    folder.mkdir(exist_ok=True)

    files = {}

    for key, mask in rendered.regions.items():
        if key not in rendered.three:
            continue
        Image.fromarray((mask.astype(np.uint8) * 255)).save(folder / f"{key}.png")
        files[key] = f"three/{key}.png"

    scene = rendered.scene
    alpha = (
        np.clip(scene.props_alpha, 0.0, 1.0)
        if scene.props_alpha is not None
        else scene.props.astype(np.float32)
    )
    objects = np.dstack([scene.master_input, np.round(alpha * 255).astype(np.uint8)])
    Image.fromarray(objects, "RGBA").save(folder / "objects.png")

    height, width = scene.master_input.shape[:2]

    payload = {
        "version": 1,
        "units": "1 scene unit = 1 metre",
        "image_size": [int(width), int(height)],
        "photo": "original.png",
        "base": output.CLEAN_ROOM_FILENAME if (job_dir / output.CLEAN_ROOM_FILENAME).exists() else "original.png",
        "tile": "tile.png",
        "objects": "three/objects.png",
        # The canonical room this render used: same camera, same millimetres.
        "room": (rendered.geometry or {}).get("room"),
        "surfaces": {
            key: {**record, "mask": files[key]}
            for key, record in rendered.three.items()
            if key in files
        },
    }

    (job_dir / THREE_FILENAME).write_text(json.dumps(payload), encoding="utf-8")


def _load_layers(job_dir: Path) -> dict:
    with np.load(job_dir / LAYERS_FILENAME) as data:
        shape = tuple(int(v) for v in data["shape"])
        count = shape[0] * shape[1]

        return {
            key: np.unpackbits(data[key], count=count).astype(bool).reshape(shape)
            for key in data.files
            if key != "shape"
        }


def _render_job(job_id: str) -> Path:
    """A `/generate` job folder that holds surface layers, or a 404."""
    if not JOB_ID_PATTERN.match(job_id or ""):
        raise HTTPException(422, f"Not a job id: {job_id!r}.")

    job_dir = JOBS_DIR / job_id

    if not (job_dir / LAYERS_FILENAME).exists() or not (job_dir / "result.png").exists():
        raise HTTPException(
            404, f"Job {job_id} has no surface layers — it was not rendered by the tile engine."
        )

    return job_dir


def _compose_layers(request: Request, layers: str) -> dict:
    """
    `/compose` for a list of (render job, surface) pairs.

    Starts from the photograph and copies each listed surface's tiled pixels
    from its own render, inside that surface's own region. Regions of
    different surfaces never overlap, so the order does not matter, and every
    pixel outside the listed surfaces is the photograph's.
    """
    pairs = []

    for item in layers.split(","):
        if not item.strip():
            continue

        job, _, key = item.strip().partition(":")

        if not key:
            raise HTTPException(422, f"Layer {item!r} must look like 'job:surface'.")

        pairs.append((_render_job(job), key))

    if not pairs:
        raise HTTPException(422, "No layers given.")

    original = np.asarray(Image.open(pairs[0][0] / "original.png").convert("RGB"))

    composed = original.copy()
    selected = np.zeros(original.shape[:2], dtype=bool)

    cache: dict = {}

    for job_dir, key in pairs:
        if job_dir.name not in cache:
            photo = np.asarray(Image.open(job_dir / "original.png").convert("RGB"))

            if not np.array_equal(photo, original):
                raise HTTPException(409, "The layers are renders of different photographs.")

            cache[job_dir.name] = (
                _load_layers(job_dir),
                np.asarray(Image.open(job_dir / "result.png").convert("RGB")),
            )

        regions, tiled = cache[job_dir.name]

        if key not in regions:
            raise HTTPException(422, f"No surface {key!r} in render {job_dir.name}.")

        region = regions[key]

        composed[region] = tiled[region]
        selected |= region

    outside = int((np.any(composed != original, axis=2) & ~selected).sum())

    digest = hashlib.sha1(
        ",".join(f"{job_dir.name}:{key}" for job_dir, key in sorted(pairs, key=lambda p: p[1])).encode()
    ).hexdigest()[:12]

    owner = pairs[0][0]

    name = f"composed_{digest}.png"

    if not (owner / name).exists():
        Image.fromarray(composed).save(owner / name)

    base = str(request.base_url).rstrip("/")

    return {
        "result_image_url": f"{base}/jobs/{owner.name}/{name}",
        "original_image_url": f"{base}/jobs/{owner.name}/original.png",
        "selection": {
            "floor": any(key == "floor" for _, key in pairs),
            "walls": [key for _, key in pairs if key.startswith("wall-")],
        },
        "tiled_pixels": int(selected.sum()),
        "outside_selection_pixels": outside,
    }


@app.post("/surfaces")
async def room_surfaces(
    request: Request,
    room_image: UploadFile = File(...),
    job_id: str = Form(...),
) -> dict:
    """
    The surfaces a cleaned room offers, each with a select dot, before any tile.

    `job_id` is the room's Clean Room run and `room_image` the same photograph
    it was run on — checked byte for byte, as `/generate` does. The walls are
    WALL_MASK split exactly as the tile engine splits it when it tiles them, so
    a wall's dot switches exactly that wall's tiles.
    """
    if not JOB_ID_PATTERN.match(job_id or ""):
        raise HTTPException(422, f"Not a job id: {job_id!r}.")

    payload = await room_image.read()

    bundle = reuse.load(JOBS_DIR / job_id, reuse.digest(payload))

    if bundle is None:
        raise HTTPException(409, f"Job {job_id} is not a Clean Room run of this photograph.")

    try:
        surface_masks = perspective_engine.load(JOBS_DIR / job_id / "segments")
    except perspective_engine.MaskError as error:
        raise HTTPException(500, f"Floor/wall masks for job {job_id} are unusable: {error}") from error

    if surface_masks is None:
        raise HTTPException(
            409, f"Job {job_id} has no FLOOR_MASK.png / WALL_MASK.png. Run Clean Room again."
        )

    try:
        def listing():
            stored = perspective_engine.ensure_geometry(
                JOBS_DIR / job_id / "segments", bundle.room, surface_masks[0], surface_masks[1],
                clean=bundle.clean, photo_bytes=payload,
            )
            return perspective_engine.surfaces(
                bundle.room, surface_masks[0], surface_masks[1],
                clean=bundle.clean, photo_bytes=payload, geometry=stored,
            )

        found = await run_in_threadpool(listing)
    except Exception as error:
        raise HTTPException(500, f"Could not detect this room's surfaces: {error}") from error

    height, width = bundle.room.shape[:2]

    return {"job_id": job_id, "canvas": [int(width), int(height)], "surfaces": found}


@app.post("/compose")
async def compose_surfaces(
    request: Request,
    floor_job: str = Form(""),
    wall_job: str = Form(""),
    floor: bool = Form(False),
    walls: str = Form(""),
    # "job:surface,job:surface" — each surface from its own render. Lets every
    # surface carry the tile it was given, whatever the others carry.
    layers: str = Form(""),
) -> dict:
    """
    The photograph with only the selected surfaces tiled.

    `floor_job` is a `/generate` run with surface=floor, `wall_job` one with
    surface=wall, both from the same Clean Room run. `floor` selects the floor;
    `walls` lists the selected wall ids ("wall-0,wall-2"). Every pixel outside
    the selected regions is the photograph's own.
    """
    wall_ids = [item.strip() for item in walls.split(",") if item.strip()]

    if layers:
        return _compose_layers(request, layers)

    jobs = {}

    if floor:
        if not floor_job:
            raise HTTPException(422, "floor is selected but no floor_job was given.")
        jobs["floor"] = _render_job(floor_job)

    if wall_ids:
        if not wall_job:
            raise HTTPException(422, "walls are selected but no wall_job was given.")
        jobs["wall"] = _render_job(wall_job)

    if not jobs:
        raise HTTPException(422, "Nothing is selected, so there is nothing to compose.")

    originals = {
        kind: np.asarray(Image.open(job_dir / "original.png").convert("RGB"))
        for kind, job_dir in jobs.items()
    }

    original = next(iter(originals.values()))

    if any(not np.array_equal(image, original) for image in originals.values()):
        raise HTTPException(409, "The floor and wall renders are of different photographs.")

    composed = original.copy()
    selected = np.zeros(original.shape[:2], dtype=bool)

    def paint(job_dir: Path, keys: list[str]) -> None:
        layers = _load_layers(job_dir)

        tiled = np.asarray(Image.open(job_dir / "result.png").convert("RGB"))

        for key in keys:
            if key not in layers:
                raise HTTPException(422, f"No surface {key!r} in render {job_dir.name}.")

            region = layers[key]

            if region.shape != composed.shape[:2] or tiled.shape != composed.shape:
                raise HTTPException(409, f"Render {job_dir.name} is not the same size as the photo.")

            composed[region] = tiled[region]
            selected[region] = True

    if "floor" in jobs:
        paint(jobs["floor"], ["floor"])

    if "wall" in jobs:
        paint(jobs["wall"], wall_ids)

    # The guarantee, checked rather than assumed: nothing outside the selected
    # surfaces differs from the photograph.
    outside = int((np.any(composed != original, axis=2) & ~selected).sum())

    owner = jobs.get("wall") or jobs["floor"]

    key = hashlib.sha1(
        f"{floor_job}|{wall_job}|{floor}|{','.join(sorted(wall_ids))}".encode()
    ).hexdigest()[:12]

    name = f"composed_{key}.png"

    Image.fromarray(composed).save(owner / name)

    base = str(request.base_url).rstrip("/")

    return {
        "result_image_url": f"{base}/jobs/{owner.name}/{name}",
        "original_image_url": f"{base}/jobs/{owner.name}/original.png",
        "selection": {"floor": bool(floor), "walls": wall_ids},
        "tiled_pixels": int(selected.sum()),
        "outside_selection_pixels": outside,
    }


@app.post("/generate")
async def generate(
    request: Request,
    room_image: UploadFile = File(...),
    tile_image: UploadFile = File(...),
    room_width: float = Form(...),
    room_length: float = Form(...),
    room_height: float = Form(...),
    tile_width: float = Form(...),
    tile_height: float = Form(...),
    rotation: int = Form(...),
    surface: str = Form(...),
    # Optional so the existing contract is unchanged: a caller that does not
    # send it gets exactly the grout the renderer has always used. `TileSpec`
    # already carried this field — only the HTTP layer was withholding it.
    grout_mm: float = Form(GROUT_MM),
    # Which walls to tile, comma-separated ("left,back"). Empty means every
    # wall the room has, which is what this endpoint has always done.
    walls: str = Form(""),
    # The job a previous POST /segment produced for this exact photograph.
    # When it is given and matches, the accepted object masks, the clean room
    # and the surfaces come from that run instead of being recomputed — which
    # is both the only way the preview PNGs and the final render can agree, and
    # the difference between a two-second re-render and a ninety-second one.
    # Omitted, unknown or from a different photo: the full pipeline runs.
    job_id: str = Form(""),
    # One wall on its own, as POST /surfaces lists it ("wall-2"). Only used
    # with surface=wall on a Clean Room job; each wall is then rendered as its
    # own scale anchor, so one wall's tiles never depend on another's.
    wall_id: str = Form(""),
) -> dict:
    if rotation not in ALLOWED_ROTATIONS:
        raise HTTPException(422, f"rotation must be one of {sorted(ALLOWED_ROTATIONS)}.")

    if surface not in ALLOWED_SURFACES:
        raise HTTPException(422, f"surface must be one of {sorted(ALLOWED_SURFACES)}.")

    requested_walls = tuple(
        part.strip().lower() for part in walls.split(",") if part.strip()
    )

    wall_index: int | None = None

    if wall_id:
        match_wall = re.fullmatch(r"wall-(\d+)", wall_id.strip())

        if not match_wall or surface != "wall":
            raise HTTPException(422, "wall_id must look like 'wall-2' and needs surface=wall.")

        wall_index = int(match_wall.group(1))

    unknown = [label for label in requested_walls if label not in ALLOWED_WALLS]

    if unknown:
        raise HTTPException(
            422, f"walls must be drawn from {sorted(ALLOWED_WALLS)}; got {unknown}."
        )

    for value, field in (
        (room_width, "room_width"),
        (room_length, "room_length"),
        (room_height, "room_height"),
        (tile_width, "tile_width"),
        (tile_height, "tile_height"),
    ):
        _positive(value, field)

    started = time.perf_counter()

    room_payload = await room_image.read()

    room_rgb = _decode(room_image, room_payload, "room_image")
    tile_rgb = _decode(tile_image, await tile_image.read(), "tile_image")

    # The accepted masks from this photo's own /segment run, if the caller named
    # a job and the bytes match. `reuse.load` returns None for anything it is
    # not certain about, and None simply means the pipeline runs as before.
    bundle = reuse.load(JOBS_DIR / job_id, reuse.digest(room_payload)) if job_id else None

    if scene_module.missing_assets():
        raise HTTPException(
            503,
            "Pipeline geometry assets are missing from test07/production_pipeline; "
            "this server cannot render. See GET /health.",
        )

    match = scene_module.signature_match(room_rgb)

    # Set by the estimated path below. The calibrated scene already ships with a
    # real empty room of its own, so it needs no fill.
    clean_rgb: np.ndarray | None = None
    fill_info: dict = {"available": False, "reason": "calibrated scene — not needed"}

    # Set only when perspective_engine rendered this request against the job's
    # FLOOR_MASK.png / WALL_MASK.png; otherwise the render below runs as before.
    rendered: perspective_engine.Rendered | None = None
    rendered_s = 0.0

    spec = TileSpec(
        artwork=tile_rgb,
        width_mm=float(tile_width),
        height_mm=float(tile_height),
        rotation_deg=float(rotation),
        grout_mm=float(grout_mm),
    )

    if match >= SCENE_MATCH_THRESHOLD:
        # The calibrated room: a camera the pipeline actually fitted to it.
        target = scene_module.load_scene()

        geometry = {
            "method": "fitted",
            "source": "stage08g3 metric camera fit",
            "room_mm": [target.room_u_mm, target.room_v_mm, target.room_height_mm],
        }

        notes = [
            "Room dimensions are recorded but not applied: the scene's metric scale comes "
            f"from the fitted camera, which measures this room at {target.room_u_mm:.0f} × "
            f"{target.room_v_mm:.0f} mm.",
            "Tile artwork, tile size, rotation and target surface are applied for real.",
        ]
    elif bundle is not None:
        # ---- the fast path: one extraction, already done -----------------
        #
        # Every mask here is the one /segment accepted and wrote. `ALL_OBJECTS`,
        # `MIRRORS_ONLY`, `CLEAN_ROOM` and the restoration below therefore all
        # come from a single source of truth, and no model runs in this request
        # at all — the whole call is the tile projection.
        room_rgb = bundle.room

        clean_rgb = bundle.clean

        props = bundle.props

        fill_info = dict(bundle.meta.get("clean_room") or {})
        fill_info["reused_from"] = job_id

        # FLOOR_MASK.png + WALL_MASK.png from this job's floor/wall stage, when
        # it wrote them. They are the strict clip for the render, which then
        # goes through perspective_engine. A job from before that stage has no
        # masks and takes the path below, unchanged.
        try:
            surface_masks = perspective_engine.load(JOBS_DIR / job_id / "segments")
        except perspective_engine.MaskError as error:
            raise HTTPException(500, f"Floor/wall masks for job {job_id} are unusable: {error}") from error

        try:
            if surface_masks is not None:
                clock = time.perf_counter()

                # The room's detected geometry: the single source of truth for
                # where the tiles go (detected now for a room cleaned before
                # geometry was stored, and kept for next time).
                room_geometry = await run_in_threadpool(
                    perspective_engine.ensure_geometry,
                    JOBS_DIR / job_id / "segments",
                    room_rgb,
                    surface_masks[0],
                    surface_masks[1],
                    clean=clean_rgb,
                    photo_bytes=room_payload,
                )

                rendered = await run_in_threadpool(
                    perspective_engine.render_tiled_room,
                    room_rgb,
                    surface_masks[0],
                    surface_masks[1],
                    spec,
                    surface,
                    (room_width, room_length, room_height),
                    clean=clean_rgb,
                    props=props,
                    wall_labels=requested_walls or None,
                    object_count=int(bundle.meta.get("object_count") or 0),
                    # The original upload, whose EXIF focal length the tile
                    # engine reads first, as the reference project did.
                    photo_bytes=room_payload,
                    wall_index=wall_index,
                    geometry=room_geometry,
                )

                rendered_s = round(time.perf_counter() - clock, 2)

                target, geometry = rendered.scene, rendered.geometry

                geometry["mask_clip"] = rendered.clip
            else:
                target, geometry = live_scene.build(
                    room_rgb,
                    [],
                    bundle.surfaces,
                    room_width,
                    room_length,
                    room_height,
                    wall_labels=requested_walls or None,
                    clean=clean_rgb,
                    props=props,
                    object_count=int(bundle.meta.get("object_count") or 0),
                )
        except MissingGeometryError as error:
            raise HTTPException(422, str(error)) from error
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

        geometry["masks"] = f"reused from segment job {job_id}"
        geometry["clean_room"] = fill_info

        if rendered is not None:
            notes = _tile_engine_notes(geometry, rendered.clip, job_id)
        else:
            notes = [
                "The camera for this photo is estimated, not measured: a level lens at "
                f"{geometry['horizontal_fov_deg']}° field of view, "
                f"{geometry['camera_height_mm']:.0f} mm above the floor, pitched "
                f"{geometry['camera_pitch_deg']}° down so the floor meets the back wall where "
                "the photo shows it. Tile scale follows the room dimensions you entered.",
                f"{geometry['object_count']} objects came from this room's own Clean Room run "
                f"(job {job_id}) rather than being segmented again, so the objects restored "
                "here are cut along exactly the boundaries ALL_OBJECTS.png was written from.",
            ]

            if surface in ("wall", "both"):
                notes.append(_wall_note(geometry))
    else:
        # Any other photo: segment it, then estimate a camera for its room box.
        room_rgb = _fit_within(room_rgb, MAX_RENDER_EDGE)

        # The same extraction the /segment panel runs, so the props protected
        # from the tile projection are the cut-to-cut masks and not a coarser
        # second opinion. The photo is already within the working resolution, so
        # the masks come back at exactly its shape.
        config = extraction_config.load()

        def analyse():
            extraction = extract(room_rgb, config)

            # The room behind the objects, rebuilt. Surfaces are then detected
            # on *that* rather than on the raw photograph: a chair hides the
            # floor it stands on and a plant hides the wall behind it, and
            # SegFormer cannot label a surface it cannot see. Measured on a
            # 1408x768 room, detecting on the clean room raised wall coverage
            # from 0.662 to 0.889 of the frame and floor from 0.1058 to 0.1107,
            # which is the area the objects were hiding.
            #
            # The objects are composited straight back on top afterwards, from
            # the original photograph, so nothing that was hidden stays visible.
            clean, fill = _clean_for_render(room_rgb, extraction.objects)

            found_surfaces = surfaces.instances(clean if clean is not None else room_rgb)

            return extraction, found_surfaces, clean, fill

        try:
            extraction, found_surfaces, clean_rgb, fill_info = await run_in_threadpool(analyse)
        except ExtractionError as error:
            raise HTTPException(503, str(error)) from error
        except Exception as error:
            raise HTTPException(500, f"Object extraction failed: {error}") from error

        try:
            target, geometry = live_scene.build(
                room_rgb,
                extraction.objects,
                found_surfaces,
                room_width,
                room_length,
                room_height,
                # Empty means "every wall", preserving the previous behaviour.
                wall_labels=requested_walls or None,
                clean=clean_rgb,
            )
        except MissingGeometryError as error:
            raise HTTPException(422, str(error)) from error

        geometry["detected_objects"] = [
            {"label": item.label, "confidence": round(item.confidence, 3)}
            for item in extraction.objects
        ]

        geometry["clean_room"] = fill_info

        notes = [
            "The camera for this photo is estimated, not measured: a level lens at "
            f"{geometry['horizontal_fov_deg']}° field of view, "
            f"{geometry['camera_height_mm']:.0f} mm above the floor, pitched "
            f"{geometry['camera_pitch_deg']}° down so the floor meets the back wall where "
            "the photo shows it. Tile scale follows the room dimensions you entered.",
            f"{geometry['object_count']} segmented objects were composited back over the "
            "tiled floor so nothing is painted over.",
        ]

        if fill_info.get("available"):
            notes.append(
                "The objects were removed and the floor and wall behind them rebuilt "
                f"with LaMa before anything was tiled — {fill_info.get('filled_px', 0):,} "
                "pixels, filled only inside the object masks. Tiles are projected onto "
                "that emptied room, so the grid runs behind furniture instead of "
                "stopping at it, and the objects are then put back from the original "
                "photograph at their own coordinates."
            )
        else:
            notes.append(
                "The room could not be emptied before tiling "
                f"({fill_info.get('reason', 'unknown reason')}), so tiles were projected "
                "onto the photograph as it is."
            )

        if surface in ("wall", "both"):
            notes.append(_wall_note(geometry))

    clock = time.perf_counter()

    if rendered is not None:
        # Already rendered, and clipped to FLOOR_MASK/WALL_MASK, by perspective_engine.
        result = rendered.result
    else:
        try:
            result = await run_in_threadpool(render, target, spec, surface)
        except MissingGeometryError as error:
            raise HTTPException(422, str(error)) from error
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

        rendered_s = round(time.perf_counter() - clock, 2)

    timings = {
        "reused_masks": bundle is not None,
        "source_job": job_id if bundle is not None else None,
        "perspective_engine": rendered is not None,
        "render_s": rendered_s,
        "total_s": round(time.perf_counter() - started, 2),
    }

    job_id = uuid.uuid4().hex[:12]

    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True)

    Image.fromarray(room_rgb).save(job_dir / "original.png")
    Image.fromarray(tile_rgb).save(job_dir / "tile.png")

    if clean_rgb is not None:
        Image.fromarray(clean_rgb).save(job_dir / output.CLEAN_ROOM_FILENAME)

    Image.fromarray(result.raw).save(job_dir / "raw.png")
    Image.fromarray(result.lit).save(job_dir / "lit.png")
    Image.fromarray(result.composite).save(job_dir / "result.png")

    # Each surface's tiled pixels, so `/compose` can show or hide one surface
    # at a time from this render without rendering again.
    if rendered is not None and rendered.regions:
        _save_layers(job_dir, rendered.regions)

    # The optional 3D view's inputs, from this same render (read-only).
    if rendered is not None and rendered.regions:
        _save_three(job_dir, rendered)

    # Room-geometry validation overlay (debug only; not part of the result).
    if rendered is not None and rendered.room_debug is not None:
        cv2.imwrite(str(job_dir / ROOM_DEBUG_FILENAME), rendered.room_debug)

    submitted = {
        "room_ft": [room_width, room_length, room_height],
        "room_mm": [
            round(room_width * MM_PER_FOOT, 1),
            round(room_length * MM_PER_FOOT, 1),
            round(room_height * MM_PER_FOOT, 1),
        ],
        "tile_mm": [tile_width, tile_height],
        "grout_mm": grout_mm,
        "walls": list(requested_walls),
        "rotation": rotation,
        "surface": surface,
    }

    state = {
        "job_id": job_id,
        "submitted": submitted,
        "geometry": geometry,
        "render": result.stats,
    }

    state["timings"] = timings

    (job_dir / "state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")

    print(f"[generate {job_id}] {timings}", flush=True)

    base = str(request.base_url).rstrip("/")

    return {
        "job_id": job_id,
        "original_image_url": f"{base}/jobs/{job_id}/original.png",
        "result_image_url": f"{base}/jobs/{job_id}/result.png",
        "raw_image_url": f"{base}/jobs/{job_id}/raw.png",
        "clean_room_url": (
            f"{base}/jobs/{job_id}/{output.CLEAN_ROOM_FILENAME}"
            if clean_rgb is not None
            else None
        ),
        "stats": result.stats,
        "geometry": geometry,
        "timings": timings,
        "notes": notes,
        # The surfaces this render tiled, each with its select dot. Null when
        # the tile engine did not render this request.
        "surfaces": perspective_engine.describe(rendered.regions) if rendered is not None else None,
        # Room-geometry validation overlay; null when this render had no room geometry.
        "room_debug_url": (
            f"{base}/jobs/{job_id}/{ROOM_DEBUG_FILENAME}"
            if (job_dir / ROOM_DEBUG_FILENAME).exists()
            else None
        ),
        # The optional 3D view's render state for this job; null when none was written.
        "three_url": (
            f"{base}/jobs/{job_id}/{THREE_FILENAME}"
            if (job_dir / THREE_FILENAME).exists()
            else None
        ),
    }
