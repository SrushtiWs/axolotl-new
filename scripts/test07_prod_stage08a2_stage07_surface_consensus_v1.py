# ============================================================
# AUTO-GENERATED TEST07 STAGE08A2 ADAPTER
#
# SOURCE:
# /workspace/axolotl/scripts/test07_prod_stage02_structure_v1.py
#
# INPUT OVERRIDE:
# /workspace/axolotl/test07/production_pipeline/stage07_empty_room/07a_qwen_empty_room/00_stage07_empty_room_candidate.png
#
# OUTPUT OVERRIDE:
# /workspace/axolotl/test07/production_pipeline/stage08_tile_application/08a2_stage07_surface_consensus
#
# Frozen Stage02 model/inference/voting logic is unchanged.
# ============================================================


from pathlib import Path
from PIL import Image, ImageDraw
from IPython.display import display

import gc
import hashlib
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation,
    OneFormerProcessor,
    OneFormerForUniversalSegmentation,
    Mask2FormerImageProcessor,
    Mask2FormerForUniversalSegmentation,
)


# ============================================================
# 1. PATHS
# ============================================================

BASE = Path("/workspace/axolotl")

PROJECT = (
    BASE
    / "test07"
)

PROD = (
    PROJECT
    / "production_pipeline"
)

STAGE01 = (
    PROD
    / "stage01_master"
)

OUT = (
    PROD
    / "stage08_tile_application"
    / "08a2_stage07_surface_consensus"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


MASTER_PATH = (
    PROD
    / "stage07_empty_room"
    / "07a_qwen_empty_room"
    / "00_stage07_empty_room_candidate.png"
)

STAGE01_REPORT = (
    STAGE01
    / "00_stage01_master_report.json"
)


MODEL_CACHE = Path(
    "/workspace/data/huggingface-cache"
)

MODEL_CACHE.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. MODEL IDS
# ============================================================

SEGFORMER_ID = (
    "nvidia/"
    "segformer-b5-finetuned-ade-640-640"
)

ONEFORMER_ID = (
    "shi-labs/"
    "oneformer_ade20k_swin_large"
)

MASK2FORMER_ID = (
    "facebook/"
    "mask2former-swin-large-ade-semantic"
)


# ============================================================
# 3. HELPERS
# ============================================================

def normalize_label(text):

    return (
        str(text)
        .lower()
        .strip()
        .replace("-", " ")
        .replace("_", " ")
    )


def sha256_file(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:

        while True:

            chunk = f.read(
                1024 * 1024
            )

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def save_mask(mask, path):

    Image.fromarray(
        mask.astype(np.uint8) * 255,
        mode="L"
    ).save(path)


def save_vote_map(votes, path):

    # 0 votes = 0
    # 1 vote  = 85
    # 2 votes = 170
    # 3 votes = 255

    visual = (
        votes.astype(np.uint8)
        * 85
    )

    Image.fromarray(
        visual,
        mode="L"
    ).save(path)


def structural_ids(
    id2label
):

    wall_ids = []
    floor_ids = []
    ceiling_ids = []

    for key, label in id2label.items():

        class_id = int(key)

        normalized = normalize_label(
            label
        )

        if normalized == "wall":

            wall_ids.append(
                class_id
            )

        elif normalized == "floor":

            floor_ids.append(
                class_id
            )

        elif normalized == "ceiling":

            ceiling_ids.append(
                class_id
            )

    return (
        wall_ids,
        floor_ids,
        ceiling_ids
    )


def semantic_to_masks(
    semantic_map,
    wall_ids,
    floor_ids,
    ceiling_ids
):

    wall = np.zeros(
        semantic_map.shape,
        dtype=bool
    )

    floor = np.zeros(
        semantic_map.shape,
        dtype=bool
    )

    ceiling = np.zeros(
        semantic_map.shape,
        dtype=bool
    )

    for class_id in wall_ids:
        wall |= (
            semantic_map == class_id
        )

    for class_id in floor_ids:
        floor |= (
            semantic_map == class_id
        )

    for class_id in ceiling_ids:
        ceiling |= (
            semantic_map == class_id
        )

    return (
        wall,
        floor,
        ceiling
    )


def tint(
    image,
    mask,
    color,
    alpha=0.52
):

    result = image.copy().astype(
        np.float32
    )

    color = np.array(
        color,
        dtype=np.float32
    )

    result[mask] = (
        result[mask]
        * (1.0 - alpha)
        +
        color
        * alpha
    )

    return np.clip(
        result,
        0,
        255
    ).astype(np.uint8)


def release_model(
    *objects
):

    for obj in objects:

        try:
            del obj
        except Exception:
            pass

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


# ============================================================
# 4. START
# ============================================================

print()
print("=" * 110)
print("TEST07 PRODUCTION")
print("STAGE 08A2 - STAGE07A SURFACE CONSENSUS")
print("=" * 110)


if not MASTER_PATH.exists():

    raise FileNotFoundError(
        f"Stage01 master missing: {MASTER_PATH}"
    )


master_pil = Image.open(
    MASTER_PATH
).convert("RGB")

master = np.array(
    master_pil
)

H, W = master.shape[:2]


print()
print(
    "MASTER:",
    MASTER_PATH
)

print(
    "SIZE:",
    W,
    "x",
    H
)

print(
    "SHA256:",
    sha256_file(
        MASTER_PATH
    )
)


# Keep exact production master reference in Stage02
master_reference_path = (
    OUT
    / "00_master_reference.png"
)

master_pil.save(
    master_reference_path
)


# ============================================================
# 5. DEVICE
# ============================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print()
print(
    "DEVICE:",
    DEVICE
)


if DEVICE == "cuda":

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    free, total = torch.cuda.mem_get_info()

    print(
        "GPU FREE GB:",
        round(
            free / 1024**3,
            2
        )
    )

    print(
        "GPU TOTAL GB:",
        round(
            total / 1024**3,
            2
        )
    )


# ============================================================
# 6. SEGFORMER
# ============================================================

print()
print("=" * 110)
print("MODEL 1/3 - SEGFORMER B5")
print("=" * 110)


seg_load_start = time.time()


seg_processor = (
    SegformerImageProcessor
    .from_pretrained(
        SEGFORMER_ID,
        cache_dir=str(
            MODEL_CACHE
        )
    )
)


seg_model = (
    SegformerForSemanticSegmentation
    .from_pretrained(
        SEGFORMER_ID,
        cache_dir=str(
            MODEL_CACHE
        )
    )
)


seg_model = seg_model.to(
    DEVICE
)

seg_model.eval()


seg_load_sec = (
    time.time()
    -
    seg_load_start
)


seg_ids = structural_ids(
    seg_model.config.id2label
)

seg_wall_ids = seg_ids[0]
seg_floor_ids = seg_ids[1]
seg_ceiling_ids = seg_ids[2]


print(
    "STRUCTURE IDS:",
    {
        "wall": seg_wall_ids,
        "floor": seg_floor_ids,
        "ceiling": seg_ceiling_ids
    }
)


seg_inputs = seg_processor(
    images=master_pil,
    return_tensors="pt"
)


seg_inputs = {
    k:
        v.to(DEVICE)
        if torch.is_tensor(v)
        else v

    for k, v
    in seg_inputs.items()
}


seg_inf_start = time.time()


with torch.inference_mode():

    seg_outputs = seg_model(
        **seg_inputs
    )


seg_logits = F.interpolate(
    seg_outputs.logits,
    size=(
        H,
        W
    ),
    mode="bilinear",
    align_corners=False
)


seg_map = (
    seg_logits
    .argmax(dim=1)[0]
    .detach()
    .cpu()
    .numpy()
    .astype(np.int32)
)


seg_inf_sec = (
    time.time()
    -
    seg_inf_start
)


(
    seg_wall,
    seg_floor,
    seg_ceiling
) = semantic_to_masks(
    seg_map,
    seg_wall_ids,
    seg_floor_ids,
    seg_ceiling_ids
)


save_mask(
    seg_wall,
    OUT
    / "01_segformer_wall.png"
)

save_mask(
    seg_floor,
    OUT
    / "02_segformer_floor.png"
)

save_mask(
    seg_ceiling,
    OUT
    / "03_segformer_ceiling.png"
)


print(
    "✅ SEGFORMER COMPLETE"
)

print(
    "LOAD SEC:",
    round(
        seg_load_sec,
        2
    )
)

print(
    "INFERENCE SEC:",
    round(
        seg_inf_sec,
        3
    )
)


del seg_outputs
del seg_logits
del seg_inputs
del seg_model
del seg_processor

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()


# ============================================================
# 7. ONEFORMER
# ============================================================

print()
print("=" * 110)
print("MODEL 2/3 - ONEFORMER")
print("=" * 110)


one_load_start = time.time()


one_processor = (
    OneFormerProcessor
    .from_pretrained(
        ONEFORMER_ID,
        cache_dir=str(
            MODEL_CACHE
        )
    )
)


one_model = (
    OneFormerForUniversalSegmentation
    .from_pretrained(
        ONEFORMER_ID,
        cache_dir=str(
            MODEL_CACHE
        )
    )
)


one_model = one_model.to(
    DEVICE
)

one_model.eval()


one_load_sec = (
    time.time()
    -
    one_load_start
)


one_ids = structural_ids(
    one_model.config.id2label
)

one_wall_ids = one_ids[0]
one_floor_ids = one_ids[1]
one_ceiling_ids = one_ids[2]


print(
    "STRUCTURE IDS:",
    {
        "wall": one_wall_ids,
        "floor": one_floor_ids,
        "ceiling": one_ceiling_ids
    }
)


one_inputs = one_processor(
    images=master_pil,
    task_inputs=[
        "semantic"
    ],
    return_tensors="pt"
)


one_inputs = {
    k:
        v.to(DEVICE)
        if torch.is_tensor(v)
        else v

    for k, v
    in one_inputs.items()
}


one_inf_start = time.time()


with torch.inference_mode():

    one_outputs = one_model(
        **one_inputs
    )


one_semantic = (
    one_processor
    .post_process_semantic_segmentation(
        one_outputs,
        target_sizes=[
            (
                H,
                W
            )
        ]
    )[0]
    .detach()
    .cpu()
    .numpy()
    .astype(np.int32)
)


one_inf_sec = (
    time.time()
    -
    one_inf_start
)


(
    one_wall,
    one_floor,
    one_ceiling
) = semantic_to_masks(
    one_semantic,
    one_wall_ids,
    one_floor_ids,
    one_ceiling_ids
)


save_mask(
    one_wall,
    OUT
    / "04_oneformer_wall.png"
)

save_mask(
    one_floor,
    OUT
    / "05_oneformer_floor.png"
)

save_mask(
    one_ceiling,
    OUT
    / "06_oneformer_ceiling.png"
)


print(
    "✅ ONEFORMER COMPLETE"
)

print(
    "LOAD SEC:",
    round(
        one_load_sec,
        2
    )
)

print(
    "INFERENCE SEC:",
    round(
        one_inf_sec,
        3
    )
)


del one_outputs
del one_inputs
del one_model
del one_processor

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()


# ============================================================
# 8. MASK2FORMER
# ============================================================

print()
print("=" * 110)
print("MODEL 3/3 - MASK2FORMER")
print("=" * 110)


mask_load_start = time.time()


mask_processor = (
    Mask2FormerImageProcessor
    .from_pretrained(
        MASK2FORMER_ID,
        cache_dir=str(
            MODEL_CACHE
        )
    )
)


mask_model = (
    Mask2FormerForUniversalSegmentation
    .from_pretrained(
        MASK2FORMER_ID,
        cache_dir=str(
            MODEL_CACHE
        )
    )
)


mask_model = mask_model.to(
    DEVICE
)

mask_model.eval()


mask_load_sec = (
    time.time()
    -
    mask_load_start
)


mask_ids = structural_ids(
    mask_model.config.id2label
)

mask_wall_ids = mask_ids[0]
mask_floor_ids = mask_ids[1]
mask_ceiling_ids = mask_ids[2]


print(
    "STRUCTURE IDS:",
    {
        "wall": mask_wall_ids,
        "floor": mask_floor_ids,
        "ceiling": mask_ceiling_ids
    }
)


mask_inputs = mask_processor(
    images=master_pil,
    return_tensors="pt"
)


mask_inputs = {
    k:
        v.to(DEVICE)
        if torch.is_tensor(v)
        else v

    for k, v
    in mask_inputs.items()
}


mask_inf_start = time.time()


with torch.inference_mode():

    mask_outputs = mask_model(
        **mask_inputs
    )


mask_semantic = (
    mask_processor
    .post_process_semantic_segmentation(
        mask_outputs,
        target_sizes=[
            (
                H,
                W
            )
        ]
    )[0]
    .detach()
    .cpu()
    .numpy()
    .astype(np.int32)
)


mask_inf_sec = (
    time.time()
    -
    mask_inf_start
)


(
    mask_wall,
    mask_floor,
    mask_ceiling
) = semantic_to_masks(
    mask_semantic,
    mask_wall_ids,
    mask_floor_ids,
    mask_ceiling_ids
)


save_mask(
    mask_wall,
    OUT
    / "07_mask2former_wall.png"
)

save_mask(
    mask_floor,
    OUT
    / "08_mask2former_floor.png"
)

save_mask(
    mask_ceiling,
    OUT
    / "09_mask2former_ceiling.png"
)


print(
    "✅ MASK2FORMER COMPLETE"
)

print(
    "LOAD SEC:",
    round(
        mask_load_sec,
        2
    )
)

print(
    "INFERENCE SEC:",
    round(
        mask_inf_sec,
        3
    )
)


del mask_outputs
del mask_inputs
del mask_model
del mask_processor

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()


# ============================================================
# 9. VOTE MAPS
# ============================================================

wall_votes = (
    seg_wall.astype(np.uint8)
    +
    one_wall.astype(np.uint8)
    +
    mask_wall.astype(np.uint8)
)


floor_votes = (
    seg_floor.astype(np.uint8)
    +
    one_floor.astype(np.uint8)
    +
    mask_floor.astype(np.uint8)
)


ceiling_votes = (
    seg_ceiling.astype(np.uint8)
    +
    one_ceiling.astype(np.uint8)
    +
    mask_ceiling.astype(np.uint8)
)


save_vote_map(
    wall_votes,
    OUT
    / "10_wall_vote_count.png"
)

save_vote_map(
    floor_votes,
    OUT
    / "11_floor_vote_count.png"
)

save_vote_map(
    ceiling_votes,
    OUT
    / "12_ceiling_vote_count.png"
)


# ============================================================
# 10. 3/3 CONSENSUS
# ============================================================

wall_3of3 = (
    wall_votes == 3
)

floor_3of3 = (
    floor_votes == 3
)

ceiling_3of3 = (
    ceiling_votes == 3
)


save_mask(
    wall_3of3,
    OUT
    / "13_wall_consensus_3of3.png"
)

save_mask(
    floor_3of3,
    OUT
    / "14_floor_consensus_3of3.png"
)

save_mask(
    ceiling_3of3,
    OUT
    / "15_ceiling_consensus_3of3.png"
)


# ============================================================
# 11. >= 2/3 MAJORITY
# ============================================================

wall_majority = (
    wall_votes >= 2
)

floor_majority = (
    floor_votes >= 2
)

ceiling_majority = (
    ceiling_votes >= 2
)


save_mask(
    wall_majority,
    OUT
    / "16_wall_majority_2of3.png"
)

save_mask(
    floor_majority,
    OUT
    / "17_floor_majority_2of3.png"
)

save_mask(
    ceiling_majority,
    OUT
    / "18_ceiling_majority_2of3.png"
)


# ============================================================
# 12. DISAGREEMENT / UNCERTAINTY
# ============================================================
#
# A pixel is uncertain if at least one model identifies
# wall/floor/ceiling there, but the three models do not all
# agree on that structural interpretation.
# ============================================================

any_structure = (
    (wall_votes > 0)
    |
    (floor_votes > 0)
    |
    (ceiling_votes > 0)
)


full_surface_agreement = (
    wall_3of3
    |
    floor_3of3
    |
    ceiling_3of3
)


surface_disagreement = (
    any_structure
    &
    ~full_surface_agreement
)


save_mask(
    surface_disagreement,
    OUT
    / "19_surface_disagreement_mask.png"
)


# ============================================================
# 13. COLORED DISAGREEMENT MAP
# ============================================================
#
# Yellow  = wall disagreement
# Cyan    = floor disagreement
# Magenta = ceiling disagreement
# White   = overlap of uncertainty categories
# ============================================================

disagreement_rgb = master.copy()


wall_uncertain = (
    (wall_votes > 0)
    &
    (wall_votes < 3)
)

floor_uncertain = (
    (floor_votes > 0)
    &
    (floor_votes < 3)
)

ceiling_uncertain = (
    (ceiling_votes > 0)
    &
    (ceiling_votes < 3)
)


disagreement_rgb = tint(
    disagreement_rgb,
    wall_uncertain,
    [255, 220, 0],
    0.68
)

disagreement_rgb = tint(
    disagreement_rgb,
    floor_uncertain,
    [0, 255, 255],
    0.68
)

disagreement_rgb = tint(
    disagreement_rgb,
    ceiling_uncertain,
    [255, 0, 255],
    0.68
)


disagreement_path = (
    OUT
    / "19_surface_disagreement_map.png"
)


Image.fromarray(
    disagreement_rgb
).save(
    disagreement_path
)


# ============================================================
# 14. STRUCTURE PREVIEW
# ============================================================
#
# BLUE  = majority wall
# GREEN = majority floor
# WHITE = majority ceiling
# RED   = unresolved/disagreement
# ============================================================

structure_preview = master.copy()


structure_preview = tint(
    structure_preview,
    wall_majority,
    [0, 102, 255],
    0.52
)

structure_preview = tint(
    structure_preview,
    floor_majority,
    [0, 200, 0],
    0.58
)

structure_preview = tint(
    structure_preview,
    ceiling_majority,
    [255, 255, 255],
    0.48
)

structure_preview = tint(
    structure_preview,
    surface_disagreement,
    [255, 0, 0],
    0.35
)


structure_preview_path = (
    OUT
    / "20_structure_preview.png"
)


Image.fromarray(
    structure_preview
).save(
    structure_preview_path
)


# ============================================================
# 15. 3-PANEL CONFIDENCE VIEW
# ============================================================

LABEL_H = 42


high_conf_preview = master.copy()

high_conf_preview = tint(
    high_conf_preview,
    wall_3of3,
    [0, 102, 255],
    0.60
)

high_conf_preview = tint(
    high_conf_preview,
    floor_3of3,
    [0, 200, 0],
    0.65
)

high_conf_preview = tint(
    high_conf_preview,
    ceiling_3of3,
    [255, 255, 255],
    0.55
)


majority_preview = master.copy()

majority_preview = tint(
    majority_preview,
    wall_majority,
    [0, 102, 255],
    0.60
)

majority_preview = tint(
    majority_preview,
    floor_majority,
    [0, 200, 0],
    0.65
)

majority_preview = tint(
    majority_preview,
    ceiling_majority,
    [255, 255, 255],
    0.55
)


panel = Image.new(
    "RGB",
    (
        W * 4,
        H + LABEL_H
    ),
    (
        255,
        255,
        255
    )
)


panel.paste(
    master_pil,
    (
        0,
        LABEL_H
    )
)

panel.paste(
    Image.fromarray(
        high_conf_preview
    ),
    (
        W,
        LABEL_H
    )
)

panel.paste(
    Image.fromarray(
        majority_preview
    ),
    (
        W * 2,
        LABEL_H
    )
)

panel.paste(
    Image.fromarray(
        disagreement_rgb
    ),
    (
        W * 3,
        LABEL_H
    )
)


draw = ImageDraw.Draw(
    panel
)


labels = [
    (
        8,
        "MASTER"
    ),
    (
        W + 8,
        "3/3 CONSENSUS"
    ),
    (
        W * 2 + 8,
        ">=2/3 MAJORITY"
    ),
    (
        W * 3 + 8,
        "DISAGREEMENT"
    ),
]


for x, text in labels:

    draw.text(
        (
            x,
            11
        ),
        text,
        fill=(
            0,
            0,
            0
        )
    )


comparison_path = (
    OUT
    / "21_structure_confidence_comparison.png"
)


panel.save(
    comparison_path
)


# ============================================================
# 16. COUNTS
# ============================================================

print()
print("=" * 110)
print("STAGE 02 STRUCTURE COUNTS")
print("=" * 110)


print()
print("SEGFORMER")

print(
    "Wall   :",
    int(seg_wall.sum())
)

print(
    "Floor  :",
    int(seg_floor.sum())
)

print(
    "Ceiling:",
    int(seg_ceiling.sum())
)


print()
print("ONEFORMER")

print(
    "Wall   :",
    int(one_wall.sum())
)

print(
    "Floor  :",
    int(one_floor.sum())
)

print(
    "Ceiling:",
    int(one_ceiling.sum())
)


print()
print("MASK2FORMER")

print(
    "Wall   :",
    int(mask_wall.sum())
)

print(
    "Floor  :",
    int(mask_floor.sum())
)

print(
    "Ceiling:",
    int(mask_ceiling.sum())
)


print()
print("3/3 CONSENSUS")

print(
    "Wall   :",
    int(wall_3of3.sum())
)

print(
    "Floor  :",
    int(floor_3of3.sum())
)

print(
    "Ceiling:",
    int(ceiling_3of3.sum())
)


print()
print(">=2/3 MAJORITY")

print(
    "Wall   :",
    int(wall_majority.sum())
)

print(
    "Floor  :",
    int(floor_majority.sum())
)

print(
    "Ceiling:",
    int(ceiling_majority.sum())
)


print()
print(
    "DISAGREEMENT PIXELS:",
    int(
        surface_disagreement.sum()
    )
)


# ============================================================
# 17. STAGE STATE
# ============================================================

report = {

    "stage":
        "PRODUCTION_STAGE02_STRUCTURE",

    "status":
        "COMPLETED_INITIAL_PRIOR",

    "important_note":
        (
            "Stage02 masks are initial structural priors only. "
            "They are not final production wall/floor/ceiling masks."
        ),

    "input": {

        "master":
            str(
                MASTER_PATH
            ),

        "width":
            W,

        "height":
            H,

        "sha256":
            sha256_file(
                MASTER_PATH
            ),
    },

    "models": {

        "segformer": {
            "model_id":
                SEGFORMER_ID,

            "load_seconds":
                round(
                    seg_load_sec,
                    3
                ),

            "inference_seconds":
                round(
                    seg_inf_sec,
                    3
                ),

            "wall_ids":
                seg_wall_ids,

            "floor_ids":
                seg_floor_ids,

            "ceiling_ids":
                seg_ceiling_ids,
        },

        "oneformer": {
            "model_id":
                ONEFORMER_ID,

            "load_seconds":
                round(
                    one_load_sec,
                    3
                ),

            "inference_seconds":
                round(
                    one_inf_sec,
                    3
                ),

            "wall_ids":
                one_wall_ids,

            "floor_ids":
                one_floor_ids,

            "ceiling_ids":
                one_ceiling_ids,
        },

        "mask2former": {
            "model_id":
                MASK2FORMER_ID,

            "load_seconds":
                round(
                    mask_load_sec,
                    3
                ),

            "inference_seconds":
                round(
                    mask_inf_sec,
                    3
                ),

            "wall_ids":
                mask_wall_ids,

            "floor_ids":
                mask_floor_ids,

            "ceiling_ids":
                mask_ceiling_ids,
        },
    },

    "counts": {

        "segformer": {
            "wall":
                int(seg_wall.sum()),

            "floor":
                int(seg_floor.sum()),

            "ceiling":
                int(seg_ceiling.sum()),
        },

        "oneformer": {
            "wall":
                int(one_wall.sum()),

            "floor":
                int(one_floor.sum()),

            "ceiling":
                int(one_ceiling.sum()),
        },

        "mask2former": {
            "wall":
                int(mask_wall.sum()),

            "floor":
                int(mask_floor.sum()),

            "ceiling":
                int(mask_ceiling.sum()),
        },

        "consensus_3of3": {
            "wall":
                int(wall_3of3.sum()),

            "floor":
                int(floor_3of3.sum()),

            "ceiling":
                int(ceiling_3of3.sum()),
        },

        "majority_2of3": {
            "wall":
                int(wall_majority.sum()),

            "floor":
                int(floor_majority.sum()),

            "ceiling":
                int(ceiling_majority.sum()),
        },

        "surface_disagreement":
            int(
                surface_disagreement.sum()
            ),
    },

    "interpretation": {

        "3of3":
            "High-confidence semantic seed.",

        "2of3":
            "Majority structural prior.",

        "disagreement":
            (
                "Region requiring later special-layer, "
                "reconstruction, or geometric reasoning."
            ),
    },

    "outputs": {

        "master_reference":
            str(
                master_reference_path
            ),

        "wall_consensus_3of3":
            str(
                OUT
                / "13_wall_consensus_3of3.png"
            ),

        "floor_consensus_3of3":
            str(
                OUT
                / "14_floor_consensus_3of3.png"
            ),

        "ceiling_consensus_3of3":
            str(
                OUT
                / "15_ceiling_consensus_3of3.png"
            ),

        "wall_majority":
            str(
                OUT
                / "16_wall_majority_2of3.png"
            ),

        "floor_majority":
            str(
                OUT
                / "17_floor_majority_2of3.png"
            ),

        "ceiling_majority":
            str(
                OUT
                / "18_ceiling_majority_2of3.png"
            ),

        "disagreement":
            str(
                disagreement_path
            ),

        "preview":
            str(
                structure_preview_path
            ),

        "comparison":
            str(
                comparison_path
            ),
    },

    "frozen_rules": [

        "Production Stage01 master remains geometry source of truth.",

        "Stage02 does not alter the master image.",

        "3/3 agreement means high model agreement, not guaranteed ground truth.",

        ">=2/3 masks are coarse structural priors, not final masks.",

        "Mirror and glass are handled in dedicated later stages.",

        "No generated image is used in this stage.",
    ],
}


REPORT_PATH = (
    OUT
    / "00_stage02_structure_state.json"
)


REPORT_PATH.write_text(
    json.dumps(
        report,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# ============================================================
# 18. FINISH
# ============================================================

print()
print("=" * 110)
print("PRODUCTION STAGE 02 COMPLETE")
print("=" * 110)

print()
print(
    "STATE:",
    REPORT_PATH
)

print(
    "STRUCTURE PREVIEW:",
    structure_preview_path
)

print(
    "CONFIDENCE COMPARISON:",
    comparison_path
)


print()
print("STRUCTURE CONFIDENCE COMPARISON")

display(
    panel
)


print()
print("STRUCTURE PREVIEW")

display(
    Image.open(
        structure_preview_path
    )
)
