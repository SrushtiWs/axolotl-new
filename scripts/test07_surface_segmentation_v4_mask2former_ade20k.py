
# ============================================================
# TEST07
# SURFACE SEGMENTATION V4
#
# MASK2FORMER ADE20K RAW BENCHMARK
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
    Mask2FormerImageProcessor,
    Mask2FormerForUniversalSegmentation,
)


# ============================================================
# 1. IDENTIFIERS
# ============================================================

STAGE_NAME = (
    "TEST07_SURFACE_SEGMENTATION_V4_"
    "MASK2FORMER_ADE20K"
)

SCRIPT_NAME = (
    "test07_surface_segmentation_v4_"
    "mask2former_ade20k.py"
)

MODEL_ID = (
    "facebook/"
    "mask2former-swin-large-ade-semantic"
)


# ============================================================
# 2. PATHS
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
    / "surface_segmentation_v4_mask2former_ade20k"
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


SEGFORMER_WALL_PATH = (
    ROOT
    / "surface_segmentation_v2_confidence_aware"
    / "14_final_wall_mask_v2.png"
)

SEGFORMER_FLOOR_PATH = (
    ROOT
    / "surface_segmentation_v2_confidence_aware"
    / "16_final_floor_mask_v2.png"
)

SEGFORMER_CEILING_PATH = (
    ROOT
    / "surface_segmentation_v2_confidence_aware"
    / "15_final_ceiling_mask_v2.png"
)


ONEFORMER_WALL_PATH = (
    ROOT
    / "surface_segmentation_v3_oneformer_ade20k"
    / "04_prop_safe_wall_mask.png"
)

ONEFORMER_FLOOR_PATH = (
    ROOT
    / "surface_segmentation_v3_oneformer_ade20k"
    / "05_prop_safe_floor_mask.png"
)

ONEFORMER_CEILING_PATH = (
    ROOT
    / "surface_segmentation_v3_oneformer_ade20k"
    / "06_prop_safe_ceiling_mask.png"
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


def load_mask(path, size):

    img = Image.open(
        path
    ).convert("L")

    if img.size != size:

        img = img.resize(
            size,
            Image.Resampling.NEAREST
        )

    return np.array(img) > 0


def save_mask(mask, path):

    Image.fromarray(
        mask.astype(np.uint8) * 255,
        mode="L"
    ).save(path)


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

    result[mask] = (
        result[mask] * (1.0 - alpha)
        +
        c * alpha
    )

    return np.clip(
        result,
        0,
        255
    ).astype(np.uint8)


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


# ============================================================
# 4. START
# ============================================================

print()
print("=" * 110)
print(STAGE_NAME)
print("=" * 110)

print()
print("MODEL :", MODEL_ID)
print("SCRIPT:", SCRIPT_NAME)
print("OUTPUT:", OUT)


required = {
    "ORIGINAL": ORIGINAL_PATH,
    "SEG WALL": SEGFORMER_WALL_PATH,
    "SEG FLOOR": SEGFORMER_FLOOR_PATH,
    "SEG CEILING": SEGFORMER_CEILING_PATH,
    "ONE WALL": ONEFORMER_WALL_PATH,
    "ONE FLOOR": ONEFORMER_FLOOR_PATH,
    "ONE CEILING": ONEFORMER_CEILING_PATH,
    "PROPS": PROP_MASK_PATH,
}


print()
print("INPUT CHECK")
print("-" * 110)

for name, path in required.items():

    ok = path.exists()

    print(
        f"{name:18s}",
        "✅" if ok else "❌",
        path
    )

    if not ok:

        raise FileNotFoundError(
            path
        )


# ============================================================
# 5. LOAD ORIGINAL / REFERENCES
# ============================================================

original_pil = Image.open(
    ORIGINAL_PATH
).convert("RGB")

original = np.array(
    original_pil
)

H, W = original.shape[:2]

SIZE = (
    W,
    H
)


seg_wall = load_mask(
    SEGFORMER_WALL_PATH,
    SIZE
)

seg_floor = load_mask(
    SEGFORMER_FLOOR_PATH,
    SIZE
)

seg_ceiling = load_mask(
    SEGFORMER_CEILING_PATH,
    SIZE
)


one_wall = load_mask(
    ONEFORMER_WALL_PATH,
    SIZE
)

one_floor = load_mask(
    ONEFORMER_FLOOR_PATH,
    SIZE
)

one_ceiling = load_mask(
    ONEFORMER_CEILING_PATH,
    SIZE
)


props = load_mask(
    PROP_MASK_PATH,
    SIZE
)


print()
print(
    "MASTER SIZE:",
    W,
    "x",
    H
)


# ============================================================
# 6. DEVICE
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


gc.collect()

if torch.cuda.is_available():

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


# ============================================================
# 7. LOAD MASK2FORMER
# ============================================================

print()
print("=" * 110)
print("LOADING MASK2FORMER ADE20K")
print("=" * 110)

print()
print(
    "First run may download model files."
)


load_start = time.time()


processor = (
    Mask2FormerImageProcessor
    .from_pretrained(
        MODEL_ID,
        cache_dir=str(
            MODEL_CACHE
        )
    )
)


model = (
    Mask2FormerForUniversalSegmentation
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
    "✅ MASK2FORMER READY"
)

print(
    "MODEL LOAD SEC:",
    round(
        load_seconds,
        2
    )
)


# ============================================================
# 8. STRUCTURAL CLASS IDS
# ============================================================

id2label = {
    int(k): v
    for k, v
    in model.config.id2label.items()
}


wall_ids = []
floor_ids = []
ceiling_ids = []


for class_id, label in id2label.items():

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
print("MASK2FORMER ADE20K STRUCTURAL IDS")
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
        "Mask2Former wall class not found."
    )


if not floor_ids:

    raise RuntimeError(
        "Mask2Former floor class not found."
    )


# ============================================================
# 9. PROCESS INPUT
# ============================================================

inputs = processor(
    images=original_pil,
    return_tensors="pt"
)


inputs = {
    key:
        value.to(
            DEVICE
        )
    if torch.is_tensor(value)
    else value

    for key, value
    in inputs.items()
}


# ============================================================
# 10. INFERENCE
# ============================================================

print()
print("=" * 110)
print("MASK2FORMER SEMANTIC INFERENCE")
print("=" * 110)


inference_start = time.time()


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
    "✅ MASK2FORMER INFERENCE COMPLETE"
)

print(
    "INFERENCE SEC:",
    round(
        inference_seconds,
        3
    )
)


# ============================================================
# 11. OFFICIAL SEMANTIC POST-PROCESS
# ============================================================

semantic_results = (
    processor.post_process_semantic_segmentation(
        outputs,
        target_sizes=[
            (
                H,
                W
            )
        ]
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
# 12. SAVE LABEL MAP
# ============================================================

Image.fromarray(
    semantic_map.astype(
        np.uint16
    )
).save(
    OUT
    / "00_mask2former_semantic_label_map.png"
)


# ============================================================
# 13. RAW MASKS
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
        semantic_map == class_id
    )


for class_id in floor_ids:

    floor_raw |= (
        semantic_map == class_id
    )


for class_id in ceiling_ids:

    ceiling_raw |= (
        semantic_map == class_id
    )


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
# 14. PROP-SAFE MASKS
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
# 15. RAW OVERLAY
# ============================================================

raw_overlay = original.copy()

raw_overlay = tint(
    raw_overlay,
    wall_raw,
    [0, 102, 255],
    0.50
)

raw_overlay = tint(
    raw_overlay,
    ceiling_raw,
    [255, 255, 255],
    0.45
)

raw_overlay = tint(
    raw_overlay,
    floor_raw,
    [0, 200, 0],
    0.55
)


raw_overlay_path = (
    OUT
    / "07_mask2former_raw_surface_overlay.png"
)


Image.fromarray(
    raw_overlay
).save(
    raw_overlay_path
)


# ============================================================
# 16. PROP-SAFE OVERLAY
# ============================================================

safe_overlay = original.copy()

safe_overlay = tint(
    safe_overlay,
    wall_safe,
    [0, 102, 255],
    0.50
)

safe_overlay = tint(
    safe_overlay,
    ceiling_safe,
    [255, 255, 255],
    0.45
)

safe_overlay = tint(
    safe_overlay,
    floor_safe,
    [0, 200, 0],
    0.55
)

safe_overlay = tint(
    safe_overlay,
    props,
    [255, 0, 255],
    0.45
)


safe_overlay_path = (
    OUT
    / "08_mask2former_prop_safe_surface_overlay.png"
)


Image.fromarray(
    safe_overlay
).save(
    safe_overlay_path
)


# ============================================================
# 17. MASK DIFFERENCE AUDITS
# ============================================================

mask_floor_added_vs_seg = (
    floor_safe
    &
    ~seg_floor
)

seg_floor_only = (
    seg_floor
    &
    ~floor_safe
)


mask_floor_added_vs_one = (
    floor_safe
    &
    ~one_floor
)

one_floor_only = (
    one_floor
    &
    ~floor_safe
)


save_mask(
    mask_floor_added_vs_seg,
    OUT
    / "09_floor_added_vs_segformer.png"
)

save_mask(
    seg_floor_only,
    OUT
    / "10_segformer_floor_only.png"
)

save_mask(
    mask_floor_added_vs_one,
    OUT
    / "11_floor_added_vs_oneformer.png"
)

save_mask(
    one_floor_only,
    OUT
    / "12_oneformer_floor_only.png"
)


# ============================================================
# 18. FLOOR AUDIT OVERLAY
# ============================================================
#
# GREEN = Mask2Former floor
# CYAN  = Mask2Former added beyond OneFormer
# RED   = OneFormer floor absent in Mask2Former
# MAGENTA = props
# ============================================================

floor_audit = original.copy()


floor_audit = tint(
    floor_audit,
    floor_safe,
    [0, 200, 0],
    0.45
)


floor_audit = tint(
    floor_audit,
    mask_floor_added_vs_one,
    [0, 255, 255],
    0.65
)


floor_audit = tint(
    floor_audit,
    one_floor_only,
    [255, 0, 0],
    0.55
)


floor_audit = tint(
    floor_audit,
    props,
    [255, 0, 255],
    0.45
)


floor_audit_path = (
    OUT
    / "13_mask2former_floor_audit_vs_oneformer.png"
)


Image.fromarray(
    floor_audit
).save(
    floor_audit_path
)


# ============================================================
# 19. BUILD SEGFORMER OVERLAY
# ============================================================

seg_overlay = original.copy()

seg_overlay = tint(
    seg_overlay,
    seg_wall,
    [0, 102, 255],
    0.50
)

seg_overlay = tint(
    seg_overlay,
    seg_ceiling,
    [255, 255, 255],
    0.45
)

seg_overlay = tint(
    seg_overlay,
    seg_floor,
    [0, 200, 0],
    0.55
)

seg_overlay = tint(
    seg_overlay,
    props,
    [255, 0, 255],
    0.45
)


# ============================================================
# 20. BUILD ONEFORMER OVERLAY
# ============================================================

one_overlay = original.copy()

one_overlay = tint(
    one_overlay,
    one_wall,
    [0, 102, 255],
    0.50
)

one_overlay = tint(
    one_overlay,
    one_ceiling,
    [255, 255, 255],
    0.45
)

one_overlay = tint(
    one_overlay,
    one_floor,
    [0, 200, 0],
    0.55
)

one_overlay = tint(
    one_overlay,
    props,
    [255, 0, 255],
    0.45
)


# ============================================================
# 21. FULL 4-PANEL COMPARISON
# ============================================================

LABEL_H = 42


comparison = Image.new(
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
        one_overlay
    ),
    (
        W * 2,
        LABEL_H
    )
)


comparison.paste(
    Image.fromarray(
        safe_overlay
    ),
    (
        W * 3,
        LABEL_H
    )
)


draw = ImageDraw.Draw(
    comparison
)


for x, text in [
    (8, "ORIGINAL"),
    (W + 8, "SEGFORMER B5"),
    (W * 2 + 8, "ONEFORMER"),
    (W * 3 + 8, "MASK2FORMER"),
]:

    draw.text(
        (
            x,
            10
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
    / "14_three_model_surface_comparison.png"
)


comparison.save(
    comparison_path
)


# ============================================================
# 22. FLOOR-ONLY THREE-MODEL COMPARISON
# ============================================================

seg_floor_overlay = tint(
    original,
    seg_floor,
    [0, 200, 0],
    0.60
)


one_floor_overlay = tint(
    original,
    one_floor,
    [0, 200, 0],
    0.60
)


mask_floor_overlay = tint(
    original,
    floor_safe,
    [0, 200, 0],
    0.60
)


floor_compare = Image.new(
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


floor_compare.paste(
    Image.fromarray(
        mask_floor_overlay
    ),
    (
        W * 3,
        LABEL_H
    )
)


floor_draw = ImageDraw.Draw(
    floor_compare
)


for x, text in [
    (8, "ORIGINAL"),
    (W + 8, "SEGFORMER FLOOR"),
    (W * 2 + 8, "ONEFORMER FLOOR"),
    (W * 3 + 8, "MASK2FORMER FLOOR"),
]:

    floor_draw.text(
        (
            x,
            10
        ),
        text,
        fill=(
            0,
            0,
            0
        )
    )


floor_compare_path = (
    OUT
    / "15_three_model_floor_comparison.png"
)


floor_compare.save(
    floor_compare_path
)


# ============================================================
# 23. WALL-ONLY THREE-MODEL COMPARISON
# ============================================================

seg_wall_overlay = tint(
    original,
    seg_wall,
    [0, 102, 255],
    0.60
)


one_wall_overlay = tint(
    original,
    one_wall,
    [0, 102, 255],
    0.60
)


mask_wall_overlay = tint(
    original,
    wall_safe,
    [0, 102, 255],
    0.60
)


wall_compare = Image.new(
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


wall_compare.paste(
    Image.fromarray(
        mask_wall_overlay
    ),
    (
        W * 3,
        LABEL_H
    )
)


wall_draw = ImageDraw.Draw(
    wall_compare
)


for x, text in [
    (8, "ORIGINAL"),
    (W + 8, "SEGFORMER WALL"),
    (W * 2 + 8, "ONEFORMER WALL"),
    (W * 3 + 8, "MASK2FORMER WALL"),
]:

    wall_draw.text(
        (
            x,
            10
        ),
        text,
        fill=(
            0,
            0,
            0
        )
    )


wall_compare_path = (
    OUT
    / "16_three_model_wall_comparison.png"
)


wall_compare.save(
    wall_compare_path
)


# ============================================================
# 24. COUNTS
# ============================================================

print()
print("=" * 110)
print("THREE-MODEL COUNTS")
print("=" * 110)

print()
print("FLOOR")

print(
    "SegFormer :",
    int(
        seg_floor.sum()
    )
)

print(
    "OneFormer :",
    int(
        one_floor.sum()
    )
)

print(
    "Mask2Former:",
    int(
        floor_safe.sum()
    )
)


print()
print("WALL")

print(
    "SegFormer :",
    int(
        seg_wall.sum()
    )
)

print(
    "OneFormer :",
    int(
        one_wall.sum()
    )
)

print(
    "Mask2Former:",
    int(
        wall_safe.sum()
    )
)


print()
print("CEILING")

print(
    "SegFormer :",
    int(
        seg_ceiling.sum()
    )
)

print(
    "OneFormer :",
    int(
        one_ceiling.sum()
    )
)

print(
    "Mask2Former:",
    int(
        ceiling_safe.sum()
    )
)


# ============================================================
# 25. REPORT
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

    "purpose":
        (
            "Raw Mask2Former ADE20K benchmark against "
            "SegFormer B5 and OneFormer for architectural "
            "surface segmentation."
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

        "segformer_floor":
            int(
                seg_floor.sum()
            ),

        "oneformer_floor":
            int(
                one_floor.sum()
            ),

        "mask2former_floor":
            int(
                floor_safe.sum()
            ),

        "segformer_wall":
            int(
                seg_wall.sum()
            ),

        "oneformer_wall":
            int(
                one_wall.sum()
            ),

        "mask2former_wall":
            int(
                wall_safe.sum()
            ),

        "segformer_ceiling":
            int(
                seg_ceiling.sum()
            ),

        "oneformer_ceiling":
            int(
                one_ceiling.sum()
            ),

        "mask2former_ceiling":
            int(
                ceiling_safe.sum()
            ),

        "mask2_floor_added_vs_seg":
            int(
                mask_floor_added_vs_seg.sum()
            ),

        "seg_floor_only":
            int(
                seg_floor_only.sum()
            ),

        "mask2_floor_added_vs_one":
            int(
                mask_floor_added_vs_one.sum()
            ),

        "one_floor_only":
            int(
                one_floor_only.sum()
            ),
    },

    "sources": {

        "original":
            str(
                ORIGINAL_PATH
            ),

        "segformer_floor":
            str(
                SEGFORMER_FLOOR_PATH
            ),

        "oneformer_floor":
            str(
                ONEFORMER_FLOOR_PATH
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

        "segformer_floor":
            sha256_file(
                SEGFORMER_FLOOR_PATH
            ),

        "oneformer_floor":
            sha256_file(
                ONEFORMER_FLOOR_PATH
            ),
    },

    "outputs": {

        "semantic_map":
            str(
                OUT
                / "00_mask2former_semantic_label_map.png"
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
                raw_overlay_path
            ),

        "safe_overlay":
            str(
                safe_overlay_path
            ),

        "floor_audit":
            str(
                floor_audit_path
            ),

        "surface_comparison":
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

        "SegFormer":
            "baseline",

        "OneFormer":
            "similar / small gains",

        "Mask2Former":
            "pending visual review",
    },
}


report_path = (
    OUT
    / "00_surface_segmentation_v4_mask2former_report.json"
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
# 26. RELEASE MODEL
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
# 27. FINAL
# ============================================================

print()
print("=" * 110)
print("MASK2FORMER SURFACE BENCHMARK COMPLETE")
print("=" * 110)

print()
print(
    "RAW OVERLAY:",
    raw_overlay_path
)

print(
    "SAFE OVERLAY:",
    safe_overlay_path
)

print(
    "THREE-MODEL FLOOR:",
    floor_compare_path
)

print(
    "THREE-MODEL WALL:",
    wall_compare_path
)

print(
    "REPORT:",
    report_path
)


# ============================================================
# 28. DISPLAY
# ============================================================

print()
print(
    "MASK2FORMER RAW OUTPUT"
)

display(
    Image.open(
        raw_overlay_path
    )
)


print()
print(
    "THREE MODEL SURFACE COMPARISON"
)

display(
    comparison
)


print()
print(
    "THREE MODEL FLOOR COMPARISON"
)

display(
    floor_compare
)


print()
print(
    "THREE MODEL WALL COMPARISON"
)

display(
    wall_compare
)


print()
print(
    "MASK2FORMER FLOOR AUDIT"
)

display(
    Image.open(
        floor_audit_path
    )
)
