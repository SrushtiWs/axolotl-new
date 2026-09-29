"""
Pipeline flow and segmentation outputs.

Two things the frontend wants to show:

  * the stage flow — what the pipeline does, in order, with a representative
    image from each stage that actually ran;
  * the segmentation output — the objects the pipeline cut out of the room,
    merged into two transparent room-size PNGs, plus the wall/floor/ceiling
    surface split.

The layers are composed here on demand from the committed masks and the stage01
master RGB (the frozen rule: "Final prop RGB must come from the master input")
and written into backend/generated/layers/. There is no per-prop PNG: every
prop mask is unioned into ALL_OBJECTS.png and every mirror mask into
MIRRORS_ONLY.png, which is the only object output this module produces.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

import layers
from scene import PROD, PROJECT_ROOT, MASTER_INPUT_PATH

GENERATED = Path(__file__).resolve().parent / "generated"

DEFINITION_PATH = PROD / "PRODUCTION_PIPELINE_DEFINITION.json"

PROP_STATE_PATH = PROD / "stage06_prop_layer" / "06f4_final_six_prop_layer" / "00_stage06f4_result.json"

# Stage 03 keeps the mirror as its own layer, separate from the props, which is
# exactly the split the two output layers need.
MIRROR_MASK_PATH = PROD / "stage03_mirror" / "00_mirror_mask.png"

# The stage scripts record absolute paths from the GPU box they ran on.
REMOTE_ROOT = "/workspace/axolotl"


@dataclass(frozen=True)
class StagePreview:
    """One pipeline stage and the committed image that best represents it."""

    stage_id: str
    title: str
    preview: str | None  # path relative to the production_pipeline directory
    status: str  # done | live | unavailable


# Ordered exactly as the pipeline runs. stage08 is the only one this server
# recomputes per request; the rest are frozen outputs from the GPU run.
STAGE_PREVIEWS: tuple[StagePreview, ...] = (
    StagePreview("stage01_master", "Master input", "stage01_master/00_master_input.png", "done"),
    StagePreview(
        "stage02_structure",
        "Room structure",
        "stage02_structure/20_structure_preview.png",
        "done",
    ),
    StagePreview("stage03_mirror", "Mirror layer", "stage03_mirror/02_mirror_bbox_preview.png", "done"),
    StagePreview(
        "stage04_glass", "Glass layer", "stage04_glass/01_original_glass_reference_rgba.png", "done"
    ),
    StagePreview(
        "stage05_clean_room_with_props",
        "Clean room with props",
        "stage05_clean_room_with_props/07_clean_room_with_exact_props.png",
        "done",
    ),
    StagePreview(
        "stage06_prop_layer",
        "Prop layer",
        "stage06_prop_layer/06f4_final_six_prop_layer/04_stage05f_rgb_checkerboard.png",
        "done",
    ),
    StagePreview(
        "stage07_empty_room",
        "Empty room + shadows",
        "stage07_empty_room/07a_qwen_empty_room/00_stage07_empty_room_candidate.png",
        "done",
    ),
    StagePreview(
        "stage08_tile_application",
        "Tile application",
        "stage08_tile_application/08h2_props_over_metric_floor/02_metric_floor_plus_physical_props.png",
        "live",
    ),
    StagePreview(
        "stage09_final_composite",
        "Final composite",
        "stage08_tile_application/08h2_props_over_metric_floor/03_stage08h2_props_composite_audit.png",
        "done",
    ),
)

# Surface masks from the three-model consensus in stage02.
SURFACE_MASKS: tuple[tuple[str, str, str], ...] = (
    ("floor", "Floor", "stage02_structure/17_floor_majority_2of3.png"),
    ("wall", "Wall", "stage02_structure/16_wall_majority_2of3.png"),
    ("ceiling", "Ceiling", "stage02_structure/18_ceiling_majority_2of3.png"),
)


def _localise(path_string: str) -> Path:
    """Map a path recorded on the GPU box onto this checkout."""
    if path_string.startswith(REMOTE_ROOT):
        return PROJECT_ROOT / path_string[len(REMOTE_ROOT) :].lstrip("/")

    return Path(path_string)


def _load_definition() -> dict:
    if not DEFINITION_PATH.exists():
        return {}

    return json.loads(DEFINITION_PATH.read_text(encoding="utf-8"))


def _bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(mask)

    if ys.size == 0:
        return None

    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _cut_out(mask: np.ndarray, rgb: np.ndarray, destination: Path) -> list[int] | None:
    """
    Write a transparent PNG holding only the masked pixels, cropped to the
    object's bounding box. Returns the bbox, or None for an empty mask.
    """
    box = _bbox(mask)

    if box is None:
        return None

    if destination.exists():
        return box

    x0, y0, x1, y1 = box

    rgba = np.zeros((y1 - y0, x1 - x0, 4), dtype=np.uint8)
    rgba[..., :3] = rgb[y0:y1, x0:x1]
    rgba[..., 3] = mask[y0:y1, x0:x1].astype(np.uint8) * 255

    destination.parent.mkdir(parents=True, exist_ok=True)

    Image.fromarray(rgba, mode="RGBA").save(destination)

    return box


def _read_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 127


def flow() -> list[dict]:
    """The nine pipeline stages, in order, each with a preview if one exists."""
    purposes = {
        stage_id: entry.get("purpose", "")
        for stage_id, entry in _load_definition().get("stages", {}).items()
    }

    stages: list[dict] = []

    for index, stage in enumerate(STAGE_PREVIEWS, start=1):
        available = stage.preview is not None and (PROD / stage.preview).exists()

        stages.append(
            {
                "step": index,
                "id": stage.stage_id,
                "title": stage.title,
                "purpose": purposes.get(stage.stage_id, ""),
                "status": stage.status if available else "unavailable",
                "preview_path": stage.preview if available else None,
            }
        )

    return stages


@lru_cache(maxsize=1)
def object_layers() -> tuple[list[dict], list[dict]]:
    """
    The committed room as the same two layers `/segment` returns.

    The props come from stage 06F4 and the mirror from stage 03, which kept it
    as its own layer — so the split the two outputs need is one the pipeline
    already made. Both are composed from the stage01 master RGB, under the
    frozen rule that final prop RGB comes from the master input.

    Returns (layers, detections). A mask that cannot be read is reported in
    `detections` and skipped, rather than failing the whole response.
    """
    if not PROP_STATE_PATH.exists() or not MASTER_INPUT_PATH.exists():
        return [], []

    rgb = np.asarray(Image.open(MASTER_INPUT_PATH).convert("RGB"), dtype=np.uint8)

    shape = rgb.shape[:2]

    state = json.loads(PROP_STATE_PATH.read_text(encoding="utf-8"))

    pixels = state.get("per_prop_pixels", {})

    detections: list[dict] = []

    object_union = np.zeros(shape, dtype=bool)
    mirror_union = np.zeros(shape, dtype=bool)

    object_members: list[str] = []
    mirror_members: list[str] = []

    sources: list[tuple[str, Path, bool]] = [
        (entry["name"], _localise(entry["mask"]), False)
        for _, entry in sorted(state.get("selected_masks", {}).items())
    ]

    if MIRROR_MASK_PATH.exists():
        sources.append(("mirror", MIRROR_MASK_PATH, True))

    for name, path, mirror in sources:
        if not path.exists():
            detections.append(
                {"name": name, "status": "failed", "reason": f"mask missing: {path.name}"}
            )
            continue

        try:
            mask = _read_mask(path)
        except Exception as error:
            detections.append({"name": name, "status": "failed", "reason": str(error)})
            continue

        if mask.shape != shape:
            detections.append(
                {
                    "name": name,
                    "status": "failed",
                    "reason": f"mask is {mask.shape[::-1]}, master is {shape[::-1]}",
                }
            )
            continue

        object_union |= mask
        object_members.append(name)

        if mirror:
            mirror_union |= mask
            mirror_members.append(name)

        detections.append(
            {
                "name": name,
                "status": "mirror" if mirror else "object",
                "pixels": int(pixels.get(name, int(mask.sum()))),
                "layers": ["all_objects", "mirrors"] if mirror else ["all_objects"],
            }
        )

    object_mask = layers.refine_mask(object_union)
    mirror_mask = layers.refine_mask(mirror_union) & object_mask

    all_objects = layers.compose(rgb, object_mask, "all_objects", object_members)

    mirrors = layers.compose(
        rgb, mirror_mask, "mirrors", mirror_members, limit=all_objects.rgba[..., 3]
    )

    height, width = shape

    composed: list[dict] = []

    # Two files, same as the live path: the props and the mirror are already
    # merged into the two masks above, and only those two get written.
    for layer, title in ((all_objects, "All objects"), (mirrors, "Mirrors")):
        filename = f"layers/{layer.filename}"

        destination = GENERATED / filename

        destination.parent.mkdir(parents=True, exist_ok=True)

        layer.to_image().save(destination)

        composed.append(
            {
                "id": layer.name,
                "name": title,
                "filename": layer.filename,
                "source_stage": "STAGE06F4_PROPS + STAGE03_MIRROR",
                "pixels": layer.pixels,
                "bbox": [0, 0, width, height],
                "cutout_path": filename,
                "members": layer.members,
            }
        )

    return composed, detections


def surfaces() -> list[dict]:
    """The wall/floor/ceiling split, cut from the master RGB the same way."""
    if not MASTER_INPUT_PATH.exists():
        return []

    rgb = np.asarray(Image.open(MASTER_INPUT_PATH).convert("RGB"), dtype=np.uint8)

    found: list[dict] = []

    for surface_id, label, relative in SURFACE_MASKS:
        mask_path = PROD / relative

        if not mask_path.exists():
            continue

        mask = _read_mask(mask_path)

        filename = f"surfaces/{surface_id}.png"

        box = _cut_out(mask, rgb, GENERATED / filename)

        if box is None:
            continue

        found.append(
            {
                "id": surface_id,
                "name": label,
                "source_stage": "STAGE02_THREE_MODEL_CONSENSUS",
                "pixels": int(mask.sum()),
                "bbox": box,
                "cutout_path": filename,
                "mask_path": relative,
            }
        )

    return found
