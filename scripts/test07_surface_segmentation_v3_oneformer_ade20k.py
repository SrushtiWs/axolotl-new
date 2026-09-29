
# ============================================================
# TEST07
# SURFACE SEGMENTATION V3
#
# ONEFORMER ADE20K RAW CAPABILITY BENCHMARK
#
# MODEL:
# shi-labs/oneformer_ade20k_swin_large
#
# PURPOSE:
# Compare OneFormer against existing SegFormer B5 ADE20K
# for:
# - wall
# - floor
# - ceiling
#
# RAW SEMANTIC ABILITY FIRST.
#
# No aggressive morphology or recovery logic is used here.
# ============================================================


# ============================================================
# 1. IMPORTS
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

from transformers import (
    OneFormerProcessor,
    OneFormerForUniversalSegmentation,
)


# ============================================================
# 2. IDENTIFIERS
# ============================================================

STAGE_NAME = (
    "TEST07_SURFACE_SEGMENTATION_V3_"
    "ONEFORMER_ADE20K"
)

SCRIPT_NAME = (
    "test07_surface_segmentation_v3_"
    "oneformer_ade20k.py"
)

MODEL_ID = (
    "shi-labs/"
    "oneformer_ade20k_swin_large"
)


# ============================================================
# 3. PATHS
# ============================================================

BASE = Path(
    "/workspace/axolotl"
)

ROOT = (
    BASE
    / "test07"
    / "runs"
    / "room03_bathroom"
    / "stages"
)

OUT = (
    ROOT
    / "surface_segmentation_v3_oneformer_ade20k"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


ORIGINAL_PATH = (
    ROOT
    / "01_structure_inverse_base"
    / "00_input_resized.png"
)


OLD_WALL_PATH = (
    ROOT
    / "01_structure_inverse_base"
    / "02_wall_mask.png"
)


OLD_CEILING_PATH = (
    ROOT
    / "01_structure_inverse_base"
    / "03_ceiling_mask.png"
)


OLD_FLOOR_PATH = (
    ROOT
    / "01_structure_inverse_base"
    / "04_floor_mask.png"
)


SEGFORMER_V2_WALL_PATH = (
    ROOT
    / "surface_segmentation_v2_confidence_aware"
    / "14_final_wall_mask_v2.png"
)


SEGFORMER_V2_CEILING_PATH = (
    ROOT
    / "surface_segmentation_v2_confidence_aware"
    / "15_final_ceiling_mask_v2.png"
)


SEGFORMER_V2_FLOOR_PATH = (
    ROOT
    / "surface_segmentation_v2_confidence_aware"
    / "16_final_floor_mask_v2.png"
)


PROP_MASK_PATH = (
    ROOT
    / "01y3_frozen_canonical_prop_protection"
    / "00_canonical_prop_protection_mask.png"
)


MODEL_CACHE = Path(
    "/workspace/data/huggingface-cache"
)

MODEL_CACHE.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 4. HELPERS
# ============================================================

def normalize_label(text):

    return (
        str(text)
        .lower()
        .strip()
        .replace("-", " ")
        .replace("_", " ")
    )


def load_mask(
    path,
    size
):

    img = Image.open(
        path
    ).convert(
        "L"
    )

    if img.size != size:

        img = img.resize(
            size,
            Image.Resampling.NEAREST
        )

    return (
        np.array(img) > 0
    )


def save_mask(
    mask,
    path
):

    Image.fromarray(
        mask.astype(
            np.uint8
        )
        * 255,
        mode="L"
    ).save(
        path
    )


def sha256_file(path):

    h = hashlib.sha256()

    with open(
        path,
        "rb"
    ) as f:

        while True:

            chunk = f.read(
                1024 * 1024
            )

            if not chunk:

                break

            h.update(
                chunk
            )

    return h.hexdigest()


def tint(
    image,
    mask,
    color,
    alpha
):

    result = image.copy().astype(
        np.float32
    )

    c = np.array(
        color,
        dtype=np.float32
    )

    result[
        mask
    ] = (

        result[
            mask
        ]
        *
        (
            1.0
            -
            alpha
        )

        +

        c
        *
        alpha
    )

    return np.clip(
        result,
        0,
        255
    ).astype(
        np.uint8
    )


# ============================================================
# 5. START / VALIDATION
# ============================================================

print()
print("=" * 110)
print(STAGE_NAME)
print("=" * 110)

print()
print(
    "MODEL:",
    MODEL_ID
)

print(
    "SCRIPT:",
    SCRIPT_NAME
)

print(
    "OUTPUT:",
    OUT
)


required = {

    "ORIGINAL":
        ORIGINAL_PATH,

    "OLD WALL":
        OLD_WALL_PATH,

    "OLD FLOOR":
        OLD_FLOOR_PATH,

    "OLD CEILING":
        OLD_CEILING_PATH,

    "SEG V2 WALL":
        SEGFORMER_V2_WALL_PATH,

    "SEG V2 FLOOR":
        SEGFORMER_V2_FLOOR_PATH,

    "SEG V2 CEILING":
        SEGFORMER_V2_CEILING_PATH,

    "PROPS":
        PROP_MASK_PATH,
}


print()
print("INPUT CHECK")
print("-" * 110)


for name, path in required.items():

    ok = path.exists()

    print(
        f"{name:20s}",
        "✅"
        if ok
        else "❌",
        path
    )

    if not ok:

        raise FileNotFoundError(
            path
        )


# ============================================================
# 6. LOAD ORIGINAL
# ============================================================

original_pil = Image.open(
    ORIGINAL_PATH
).convert(
    "RGB"
)

original = np.array(
    original_pil
)

H, W = original.shape[:2]

SIZE = (
    W,
    H
)


print()
print(
    "MASTER SIZE:",
    W,
    "x",
    H
)


# ============================================================
# 7. LOAD REFERENCES
# ============================================================

old_wall = load_mask(
    OLD_WALL_PATH,
    SIZE
)

old_floor = load_mask(
    OLD_FLOOR_PATH,
    SIZE
)

old_ceiling = load_mask(
    OLD_CEILING_PATH,
    SIZE
)


seg_v2_wall = load_mask(
    SEGFORMER_V2_WALL_PATH,
    SIZE
)

seg_v2_floor = load_mask(
    SEGFORMER_V2_FLOOR_PATH,
    SIZE
)

seg_v2_ceiling = load_mask(
    SEGFORMER_V2_CEILING_PATH,
    SIZE
)


props = load_mask(
    PROP_MASK_PATH,
    SIZE
)


# ============================================================
# 8. DEVICE
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
        torch.cuda.get_device_name(
            0
        )
    )

    free, total = (
        torch.cuda.mem_get_info()
    )

    print(
        "GPU FREE GB:",
        round(
            free
            /
            1024**3,
            2
        )
    )

    print(
        "GPU TOTAL GB:",
        round(
            total
            /
            1024**3,
            2
        )
    )


# ============================================================
# 9. MEMORY CLEANUP
# ============================================================

gc.collect()


if torch.cuda.is_available():

    torch.cuda.empty_cache()

    torch.cuda.ipc_collect()


# ============================================================
# 10. LOAD ONEFORMER
# ============================================================

print()
print("=" * 110)
print("LOADING ONEFORMER ADE20K")
print("=" * 110)

print()
print(
    "First run may download model files."
)


load_start = time.time()


processor = (
    OneFormerProcessor
    .from_pretrained(
        MODEL_ID,
        cache_dir=str(
            MODEL_CACHE
        )
    )
)


model = (
    OneFormerForUniversalSegmentation
    .from_pretrained(
        MODEL_ID,
        cache_dir=str(
            MODEL_CACHE
        )
    )
)


model = model.to(
    DEVICE
)


model.eval()


load_seconds = (
    time.time()
    -
    load_start
)


print()
print(
    "✅ ONEFORMER READY"
)

print(
    "MODEL LOAD SEC:",
    round(
        load_seconds,
        2
    )
)


# ============================================================
# 11. LABEL MAP
# ============================================================

id2label = {

    int(k): v

    for k, v

    in model.config.id2label.items()
}


wall_ids = []

floor_ids = []

ceiling_ids = []


for class_id, label in (
    id2label.items()
):

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


print()
print("=" * 110)
print("ONEFORMER ADE20K STRUCTURAL IDS")
print("=" * 110)

print(
    "WALL:",
    wall_ids
)

print(
    "FLOOR:",
    floor_ids
)

print(
    "CEILING:",
    ceiling_ids
)


if not wall_ids:

    raise RuntimeError(
        "OneFormer ADE20K wall class not found."
    )


if not floor_ids:

    raise RuntimeError(
        "OneFormer ADE20K floor class not found."
    )


# ============================================================
# 12. PREPROCESS
# ============================================================
#
# OneFormer is task-conditioned.
#
# We explicitly request semantic segmentation.
# ============================================================

task_inputs = [
    "semantic"
]


inputs = processor(
    images=original_pil,
    task_inputs=task_inputs,
    return_tensors="pt"
)


inputs = {

    key:
        value.to(
            DEVICE
        )

    if torch.is_tensor(
        value
    )

    else value

    for key, value

    in inputs.items()
}


# ============================================================
# 13. INFERENCE
# ============================================================

print()
print("=" * 110)
print("ONEFORMER SEMANTIC INFERENCE")
print("=" * 110)


inference_start = (
    time.time()
)


with torch.inference_mode():

    outputs = model(
        **inputs
    )


inference_seconds = (
    time.time()
    -
    inference_start
)


print()
print(
    "✅ ONEFORMER INFERENCE COMPLETE"
)

print(
    "INFERENCE SEC:",
    round(
        inference_seconds,
        3
    )
)


# ============================================================
# 14. POST-PROCESS SEMANTIC MAP
# ============================================================
#
# Use official OneFormer processor post-processing.
# This avoids guessing query-mask fusion ourselves.
# ============================================================

target_sizes = [
    (
        H,
        W
    )
]


semantic_results = (
    processor.post_process_semantic_segmentation(
        outputs,
        target_sizes=target_sizes
    )
)


semantic_map = (
    semantic_results[0]
    .detach()
    .cpu()
    .numpy()
    .astype(
        np.int32
    )
)


print()
print(
    "SEMANTIC MAP SHAPE:",
    semantic_map.shape
)


# ============================================================
# 15. SAVE RAW SEMANTIC LABEL MAP
# ============================================================

semantic_save = semantic_map.astype(
    np.uint16
)


Image.fromarray(
    semantic_save
).save(
    OUT
    / "00_oneformer_semantic_label_map.png"
)


# ============================================================
# 16. RAW STRUCTURAL MASKS
# ============================================================

wall_raw = np.zeros(
    (
        H,
        W
    ),
    dtype=bool
)


floor_raw = np.zeros(
    (
        H,
        W
    ),
    dtype=bool
)


ceiling_raw = np.zeros(
    (
        H,
        W
    ),
    dtype=bool
)


for class_id in wall_ids:

    wall_raw |= (
        semantic_map
        ==
        class_id
    )


for class_id in floor_ids:

    floor_raw |= (
        semantic_map
        ==
        class_id
    )


for class_id in ceiling_ids:

    ceiling_raw |= (
        semantic_map
        ==
        class_id
    )


# ============================================================
# 17. SAVE TRUE RAW MASKS
# ============================================================
#
# These are intentionally BEFORE prop subtraction.
# They show exactly what OneFormer predicts.
# ============================================================

save_mask(
    wall_raw,
    OUT
    / "01_raw_wall_mask.png"
)


save_mask(
    floor_raw,
    OUT
    / "02_raw_floor_mask.png"
)


save_mask(
    ceiling_raw,
    OUT
    / "03_raw_ceiling_mask.png"
)


# ============================================================
# 18. PROP-SAFE MASKS
# ============================================================
#
# This is not semantic correction.
# It merely applies our existing frozen physical-prop protection
# so comparisons are useful for the final pipeline.
# ============================================================

wall_safe = (
    wall_raw
    &
    ~props
)


floor_safe = (
    floor_raw
    &
    ~props
)


ceiling_safe = (
    ceiling_raw
    &
    ~props
)


save_mask(
    wall_safe,
    OUT
    / "04_prop_safe_wall_mask.png"
)


save_mask(
    floor_safe,
    OUT
    / "05_prop_safe_floor_mask.png"
)


save_mask(
    ceiling_safe,
    OUT
    / "06_prop_safe_ceiling_mask.png"
)


# ============================================================
# 19. ONEFORMER RAW OVERLAY
# ============================================================

overlay_raw = original.copy()


overlay_raw = tint(
    overlay_raw,
    wall_raw,
    [
        0,
        102,
        255
    ],
    0.50
)


overlay_raw = tint(
    overlay_raw,
    ceiling_raw,
    [
        255,
        255,
        255
    ],
    0.45
)


overlay_raw = tint(
    overlay_raw,
    floor_raw,
    [
        0,
        200,
        0
    ],
    0.55
)


overlay_raw_path = (
    OUT
    / "07_oneformer_raw_surface_overlay.png"
)


Image.fromarray(
    overlay_raw
).save(
    overlay_raw_path
)


# ============================================================
# 20. PROP-SAFE OVERLAY
# ============================================================

overlay_safe = original.copy()


overlay_safe = tint(
    overlay_safe,
    wall_safe,
    [
        0,
        102,
        255
    ],
    0.50
)


overlay_safe = tint(
    overlay_safe,
    ceiling_safe,
    [
        255,
        255,
        255
    ],
    0.45
)


overlay_safe = tint(
    overlay_safe,
    floor_safe,
    [
        0,
        200,
        0
    ],
    0.55
)


overlay_safe = tint(
    overlay_safe,
    props,
    [
        255,
        0,
        255
    ],
    0.45
)


overlay_safe_path = (
    OUT
    / "08_oneformer_prop_safe_surface_overlay.png"
)


Image.fromarray(
    overlay_safe
).save(
    overlay_safe_path
)


# ============================================================
# 21. FLOOR DIFFERENCE VS OLD SEGFORMER
# ============================================================

floor_added_vs_old = (
    floor_safe
    &
    ~old_floor
)


floor_old_only = (
    old_floor
    &
    ~floor_safe
)


floor_added_vs_v2 = (
    floor_safe
    &
    ~seg_v2_floor
)


floor_v2_only = (
    seg_v2_floor
    &
    ~floor_safe
)


save_mask(
    floor_added_vs_old,
    OUT
    / "09_floor_added_vs_old_segformer.png"
)


save_mask(
    floor_old_only,
    OUT
    / "10_old_segformer_floor_only.png"
)


save_mask(
    floor_added_vs_v2,
    OUT
    / "11_floor_added_vs_segformer_v2.png"
)


save_mask(
    floor_v2_only,
    OUT
    / "12_segformer_v2_floor_only.png"
)


# ============================================================
# 22. FLOOR AUDIT OVERLAY
# ============================================================
#
# GREEN = OneFormer
# CYAN  = OneFormer added beyond SegFormer V2
# RED   = SegFormer V2 floor absent in OneFormer
# MAGENTA = props
# ============================================================

floor_audit = original.copy()


floor_audit = tint(
    floor_audit,
    floor_safe,
    [
        0,
        200,
        0
    ],
    0.45
)


floor_audit = tint(
    floor_audit,
    floor_added_vs_v2,
    [
        0,
        255,
        255
    ],
    0.65
)


floor_audit = tint(
    floor_audit,
    floor_v2_only,
    [
        255,
        0,
        0
    ],
    0.55
)


floor_audit = tint(
    floor_audit,
    props,
    [
        255,
        0,
        255
    ],
    0.45
)


floor_audit_path = (
    OUT
    / "13_oneformer_floor_audit_vs_segformer_v2.png"
)


Image.fromarray(
    floor_audit
).save(
    floor_audit_path
)


# ============================================================
# 23. WALL DIFFERENCES
# ============================================================

wall_added_vs_v2 = (
    wall_safe
    &
    ~seg_v2_wall
)


wall_v2_only = (
    seg_v2_wall
    &
    ~wall_safe
)


save_mask(
    wall_added_vs_v2,
    OUT
    / "14_wall_added_vs_segformer_v2.png"
)


save_mask(
    wall_v2_only,
    OUT
    / "15_segformer_v2_wall_only.png"
)


# ============================================================
# 24. CEILING DIFFERENCES
# ============================================================

ceiling_added_vs_v2 = (
    ceiling_safe
    &
    ~seg_v2_ceiling
)


ceiling_v2_only = (
    seg_v2_ceiling
    &
    ~ceiling_safe
)


save_mask(
    ceiling_added_vs_v2,
    OUT
    / "16_ceiling_added_vs_segformer_v2.png"
)


save_mask(
    ceiling_v2_only,
    OUT
    / "17_segformer_v2_ceiling_only.png"
)


# ============================================================
# 25. SEGFORMER V2 OVERLAY FOR FAIR COMPARISON
# ============================================================

seg_overlay = original.copy()


seg_overlay = tint(
    seg_overlay,
    seg_v2_wall,
    [
        0,
        102,
        255
    ],
    0.50
)


seg_overlay = tint(
    seg_overlay,
    seg_v2_ceiling,
    [
        255,
        255,
        255
    ],
    0.45
)


seg_overlay = tint(
    seg_overlay,
    seg_v2_floor,
    [
        0,
        200,
        0
    ],
    0.55
)


seg_overlay = tint(
    seg_overlay,
    props,
    [
        255,
        0,
        255
    ],
    0.45
)


# ============================================================
# 26. FULL MODEL COMPARISON
# ============================================================

LABEL_H = 42


comparison = Image.new(
    "RGB",
    (
        W * 3,
        H + LABEL_H
    ),
    (
        255,
        255,
        255
    )
)


comparison.paste(
    original_pil,
    (
        0,
        LABEL_H
    )
)


comparison.paste(
    Image.fromarray(
        seg_overlay
    ),
    (
        W,
        LABEL_H
    )
)


comparison.paste(
    Image.fromarray(
        overlay_safe
    ),
    (
        W * 2,
        LABEL_H
    )
)


draw = ImageDraw.Draw(
    comparison
)


draw.text(
    (
        8,
        10
    ),
    "ORIGINAL",
    fill=(
        0,
        0,
        0
    )
)


draw.text(
    (
        W + 8,
        10
    ),
    "SEGFORMER B5 V2",
    fill=(
        0,
        0,
        0
    )
)


draw.text(
    (
        W * 2 + 8,
        10
    ),
    "ONEFORMER ADE20K",
    fill=(
        0,
        0,
        0
    )
)


comparison_path = (
    OUT
    / "18_segformer_v2_vs_oneformer_comparison.png"
)


comparison.save(
    comparison_path
)


# ============================================================
# 27. FLOOR-ONLY COMPARISON
# ============================================================

seg_floor_overlay = tint(
    original,
    seg_v2_floor,
    [
        0,
        200,
        0
    ],
    0.60
)


one_floor_overlay = tint(
    original,
    floor_safe,
    [
        0,
        200,
        0
    ],
    0.60
)


floor_compare = Image.new(
    "RGB",
    (
        W * 3,
        H + LABEL_H
    ),
    (
        255,
        255,
        255
    )
)


floor_compare.paste(
    original_pil,
    (
        0,
        LABEL_H
    )
)


floor_compare.paste(
    Image.fromarray(
        seg_floor_overlay
    ),
    (
        W,
        LABEL_H
    )
)


floor_compare.paste(
    Image.fromarray(
        one_floor_overlay
    ),
    (
        W * 2,
        LABEL_H
    )
)


floor_draw = ImageDraw.Draw(
    floor_compare
)


floor_draw.text(
    (
        8,
        10
    ),
    "ORIGINAL",
    fill=(
        0,
        0,
        0
    )
)


floor_draw.text(
    (
        W + 8,
        10
    ),
    "SEGFORMER FLOOR",
    fill=(
        0,
        0,
        0
    )
)


floor_draw.text(
    (
        W * 2 + 8,
        10
    ),
    "ONEFORMER FLOOR",
    fill=(
        0,
        0,
        0
    )
)


floor_compare_path = (
    OUT
    / "19_floor_only_comparison.png"
)


floor_compare.save(
    floor_compare_path
)


# ============================================================
# 28. WALL-ONLY COMPARISON
# ============================================================

seg_wall_overlay = tint(
    original,
    seg_v2_wall,
    [
        0,
        102,
        255
    ],
    0.60
)


one_wall_overlay = tint(
    original,
    wall_safe,
    [
        0,
        102,
        255
    ],
    0.60
)


wall_compare = Image.new(
    "RGB",
    (
        W * 3,
        H + LABEL_H
    ),
    (
        255,
        255,
        255
    )
)


wall_compare.paste(
    original_pil,
    (
        0,
        LABEL_H
    )
)


wall_compare.paste(
    Image.fromarray(
        seg_wall_overlay
    ),
    (
        W,
        LABEL_H
    )
)


wall_compare.paste(
    Image.fromarray(
        one_wall_overlay
    ),
    (
        W * 2,
        LABEL_H
    )
)


wall_draw = ImageDraw.Draw(
    wall_compare
)


wall_draw.text(
    (
        8,
        10
    ),
    "ORIGINAL",
    fill=(
        0,
        0,
        0
    )
)


wall_draw.text(
    (
        W + 8,
        10
    ),
    "SEGFORMER WALL",
    fill=(
        0,
        0,
        0
    )
)


wall_draw.text(
    (
        W * 2 + 8,
        10
    ),
    "ONEFORMER WALL",
    fill=(
        0,
        0,
        0
    )
)


wall_compare_path = (
    OUT
    / "20_wall_only_comparison.png"
)


wall_compare.save(
    wall_compare_path
)


# ============================================================
# 29. COUNTS
# ============================================================

print()
print("=" * 110)
print("ONEFORMER VS SEGFORMER COUNTS")
print("=" * 110)


print()
print("FLOOR")

print(
    "SegFormer V2:",
    int(
        seg_v2_floor.sum()
    )
)

print(
    "OneFormer:",
    int(
        floor_safe.sum()
    )
)

print(
    "OneFormer added:",
    int(
        floor_added_vs_v2.sum()
    )
)

print(
    "SegFormer only:",
    int(
        floor_v2_only.sum()
    )
)


print()
print("WALL")

print(
    "SegFormer V2:",
    int(
        seg_v2_wall.sum()
    )
)

print(
    "OneFormer:",
    int(
        wall_safe.sum()
    )
)

print(
    "OneFormer added:",
    int(
        wall_added_vs_v2.sum()
    )
)

print(
    "SegFormer only:",
    int(
        wall_v2_only.sum()
    )
)


print()
print("CEILING")

print(
    "SegFormer V2:",
    int(
        seg_v2_ceiling.sum()
    )
)

print(
    "OneFormer:",
    int(
        ceiling_safe.sum()
    )
)


# ============================================================
# 30. SAVE REPORT
# ============================================================

report = {

    "stage":
        STAGE_NAME,

    "status":
        "COMPLETED",

    "script":
        SCRIPT_NAME,

    "model":
        MODEL_ID,

    "task":
        "semantic segmentation",

    "purpose":
        (
            "Raw semantic benchmark of OneFormer ADE20K "
            "against SegFormer B5 for architectural "
            "wall/floor/ceiling segmentation."
        ),

    "device":
        DEVICE,

    "model_load_seconds":
        round(
            load_seconds,
            3
        ),

    "inference_seconds":
        round(
            inference_seconds,
            3
        ),

    "structural_ids": {

        "wall":
            wall_ids,

        "floor":
            floor_ids,

        "ceiling":
            ceiling_ids,
    },

    "counts": {

        "oneformer_raw_wall":
            int(
                wall_raw.sum()
            ),

        "oneformer_raw_floor":
            int(
                floor_raw.sum()
            ),

        "oneformer_raw_ceiling":
            int(
                ceiling_raw.sum()
            ),

        "oneformer_safe_wall":
            int(
                wall_safe.sum()
            ),

        "oneformer_safe_floor":
            int(
                floor_safe.sum()
            ),

        "oneformer_safe_ceiling":
            int(
                ceiling_safe.sum()
            ),

        "segformer_v2_wall":
            int(
                seg_v2_wall.sum()
            ),

        "segformer_v2_floor":
            int(
                seg_v2_floor.sum()
            ),

        "segformer_v2_ceiling":
            int(
                seg_v2_ceiling.sum()
            ),

        "oneformer_floor_added_vs_segformer_v2":
            int(
                floor_added_vs_v2.sum()
            ),

        "segformer_v2_floor_only":
            int(
                floor_v2_only.sum()
            ),

        "oneformer_wall_added_vs_segformer_v2":
            int(
                wall_added_vs_v2.sum()
            ),

        "segformer_v2_wall_only":
            int(
                wall_v2_only.sum()
            ),
    },

    "sources": {

        "original":
            str(
                ORIGINAL_PATH
            ),

        "segformer_v2_wall":
            str(
                SEGFORMER_V2_WALL_PATH
            ),

        "segformer_v2_floor":
            str(
                SEGFORMER_V2_FLOOR_PATH
            ),

        "segformer_v2_ceiling":
            str(
                SEGFORMER_V2_CEILING_PATH
            ),

        "props":
            str(
                PROP_MASK_PATH
            ),
    },

    "hashes": {

        "original":
            sha256_file(
                ORIGINAL_PATH
            ),

        "segformer_v2_floor":
            sha256_file(
                SEGFORMER_V2_FLOOR_PATH
            ),
    },

    "outputs": {

        "semantic_label_map":
            str(
                OUT
                / "00_oneformer_semantic_label_map.png"
            ),

        "raw_wall":
            str(
                OUT
                / "01_raw_wall_mask.png"
            ),

        "raw_floor":
            str(
                OUT
                / "02_raw_floor_mask.png"
            ),

        "raw_ceiling":
            str(
                OUT
                / "03_raw_ceiling_mask.png"
            ),

        "safe_wall":
            str(
                OUT
                / "04_prop_safe_wall_mask.png"
            ),

        "safe_floor":
            str(
                OUT
                / "05_prop_safe_floor_mask.png"
            ),

        "safe_ceiling":
            str(
                OUT
                / "06_prop_safe_ceiling_mask.png"
            ),

        "raw_overlay":
            str(
                overlay_raw_path
            ),

        "safe_overlay":
            str(
                overlay_safe_path
            ),

        "floor_audit":
            str(
                floor_audit_path
            ),

        "full_comparison":
            str(
                comparison_path
            ),

        "floor_comparison":
            str(
                floor_compare_path
            ),

        "wall_comparison":
            str(
                wall_compare_path
            ),
    },

    "benchmark": {

        "SegFormer_B5":
            "baseline established",

        "OneFormer_ADE20K":
            "pending visual review",

        "Mask2Former_ADE20K":
            "test only if OneFormer insufficient",
    },
}


report_path = (
    OUT
    / "00_surface_segmentation_v3_oneformer_report.json"
)


report_path.write_text(
    json.dumps(
        report,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# ============================================================
# 31. RELEASE MODEL
# ============================================================

try:
    del outputs
except Exception:
    pass

try:
    del model
except Exception:
    pass

try:
    del processor
except Exception:
    pass


gc.collect()


if torch.cuda.is_available():

    torch.cuda.empty_cache()


# ============================================================
# 32. FINAL
# ============================================================

print()
print("=" * 110)
print("ONEFORMER SURFACE BENCHMARK COMPLETE")
print("=" * 110)

print()
print(
    "RAW OVERLAY:",
    overlay_raw_path
)

print(
    "PROP-SAFE OVERLAY:",
    overlay_safe_path
)

print(
    "FLOOR COMPARISON:",
    floor_compare_path
)

print(
    "WALL COMPARISON:",
    wall_compare_path
)

print(
    "REPORT:",
    report_path
)


# ============================================================
# 33. DISPLAY
# ============================================================

print()
print(
    "ONEFORMER RAW SURFACE OUTPUT"
)

display(
    Image.open(
        overlay_raw_path
    )
)


print()
print(
    "ONEFORMER PROP-SAFE OUTPUT"
)

display(
    Image.open(
        overlay_safe_path
    )
)


print()
print(
    "SEGFORMER V2 VS ONEFORMER"
)

display(
    comparison
)


print()
print(
    "FLOOR ONLY: SEGFORMER VS ONEFORMER"
)

display(
    floor_compare
)


print()
print(
    "WALL ONLY: SEGFORMER VS ONEFORMER"
)

display(
    wall_compare
)


print()
print(
    "ONEFORMER FLOOR DIFFERENCE AUDIT"
)

display(
    Image.open(
        floor_audit_path
    )
)
