
# ============================================================
# TEST07
# SURFACE SEGMENTATION V2
#
# CONFIDENCE-AWARE SEGFORMER B5 ADE20K
#
# MODEL:
# nvidia/segformer-b5-finetuned-ade-640-640
#
# PURPOSE:
# Recover wall / floor / ceiling pixels that were discarded by
# the previous hard-argmax Stage01 implementation.
#
# IMPORTANT:
# This stage does NOT modify any previous TEST07 result.
# This is a new independent surface-segmentation benchmark.
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
import cv2
import torch

try:
    from transformers.models.segformer.image_processing_segformer import (
        SegformerImageProcessor
    )

    from transformers.models.segformer.modeling_segformer import (
        SegformerForSemanticSegmentation
    )

except Exception as e:

    raise RuntimeError(
        "Transformers / SegFormer import failed.\n"
        "Install transformers first.\n"
        + str(e)
    )


# ============================================================
# 2. IDENTIFIERS
# ============================================================

STAGE_NAME = (
    "TEST07_SURFACE_SEGMENTATION_V2_"
    "CONFIDENCE_AWARE"
)

SCRIPT_NAME = (
    "test07_surface_segmentation_v2_"
    "confidence_aware.py"
)

MODEL_ID = (
    "nvidia/"
    "segformer-b5-finetuned-ade-640-640"
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
    / "surface_segmentation_v2_confidence_aware"
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
# 4. SETTINGS
# ============================================================

# ------------------------------------------------------------
# A pixel may be recovered for a structural class if:
#
# 1. that class has at least this absolute probability
# 2. it is reasonably close to the pixel's winning class
#
# These are deliberately conservative.
# ------------------------------------------------------------

RECOVERY_MIN_PROB = 0.20

RECOVERY_MAX_GAP_FROM_WINNER = 0.14


# Higher confidence used as trusted class seeds.
TRUSTED_SEED_PROB = 0.50


# Connected components smaller than this are discarded unless
# already present in the hard argmax mask.
MIN_COMPONENT_PIXELS = 40


# Pixel class IDs:
CLASS_OTHER = 0
CLASS_WALL = 1
CLASS_CEILING = 2
CLASS_FLOOR = 3


# Display colors only.
WALL_COLOR = np.array(
    [0, 102, 255],
    dtype=np.uint8
)

CEILING_COLOR = np.array(
    [255, 255, 255],
    dtype=np.uint8
)

FLOOR_COLOR = np.array(
    [0, 200, 0],
    dtype=np.uint8
)


# ============================================================
# 5. HELPERS
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

    return (
        np.array(img) > 0
    )


def save_mask(mask, path):

    Image.fromarray(
        mask.astype(np.uint8) * 255,
        mode="L"
    ).save(path)


def save_heatmap_gray(probability, path):

    p = np.clip(
        probability,
        0.0,
        1.0
    )

    img = (
        p * 255.0
    ).round().astype(
        np.uint8
    )

    Image.fromarray(
        img,
        mode="L"
    ).save(path)


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


def component_filter_with_seed(
    candidate,
    trusted_seed,
    hard_mask,
    min_area=40
):

    candidate_u8 = (
        candidate.astype(np.uint8)
    )

    n, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            candidate_u8,
            connectivity=8
        )
    )

    out = np.zeros_like(
        candidate,
        dtype=bool
    )

    component_audit = []

    for component_id in range(
        1,
        n
    ):

        component = (
            labels
            ==
            component_id
        )

        area = int(
            stats[
                component_id,
                cv2.CC_STAT_AREA
            ]
        )

        touches_trusted = bool(
            np.any(
                component
                &
                trusted_seed
            )
        )

        touches_hard = bool(
            np.any(
                component
                &
                hard_mask
            )
        )

        keep = (
            touches_trusted
            or
            touches_hard
            or
            area >= min_area
        )

        if keep:

            out |= component

        component_audit.append(
            {
                "component_id":
                    int(component_id),

                "area":
                    area,

                "touches_trusted_seed":
                    touches_trusted,

                "touches_hard_argmax":
                    touches_hard,

                "kept":
                    keep,
            }
        )

    return (
        out,
        component_audit
    )


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
# 6. VALIDATE INPUTS
# ============================================================

required = {

    "ORIGINAL":
        ORIGINAL_PATH,

    "OLD WALL":
        OLD_WALL_PATH,

    "OLD CEILING":
        OLD_CEILING_PATH,

    "OLD FLOOR":
        OLD_FLOOR_PATH,

    "PROP MASK":
        PROP_MASK_PATH,
}


print()
print("=" * 110)
print(STAGE_NAME)
print("=" * 110)

print()
print(
    "SCRIPT:",
    SCRIPT_NAME
)

print(
    "MODEL:",
    MODEL_ID
)

print(
    "OUTPUT:",
    OUT
)

print()


for name, path in required.items():

    exists = (
        path.exists()
    )

    print(
        f"{name:18s}",
        "✅" if exists else "❌",
        path
    )

    if not exists:

        raise FileNotFoundError(
            path
        )


# ============================================================
# 7. LOAD ORIGINAL
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
# 8. LOAD OLD MASKS / PROP MASK
# ============================================================

old_wall = load_mask(
    OLD_WALL_PATH,
    SIZE
)

old_ceiling = load_mask(
    OLD_CEILING_PATH,
    SIZE
)

old_floor = load_mask(
    OLD_FLOOR_PATH,
    SIZE
)

props = load_mask(
    PROP_MASK_PATH,
    SIZE
)


# ============================================================
# 9. DEVICE
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


# ============================================================
# 10. CLEAN MEMORY
# ============================================================

gc.collect()

if torch.cuda.is_available():

    torch.cuda.empty_cache()

    torch.cuda.ipc_collect()


# ============================================================
# 11. LOAD SEGFORMER
# ============================================================

print()
print("=" * 110)
print("LOADING SEGFORMER B5 ADE20K")
print("=" * 110)


load_start = time.time()


processor = (
    SegformerImageProcessor
    .from_pretrained(
        MODEL_ID,
        cache_dir=str(
            MODEL_CACHE
        )
    )
)


model = (
    SegformerForSemanticSegmentation
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
    "✅ SEGFORMER READY"
)

print(
    "MODEL LOAD SEC:",
    round(
        load_seconds,
        2
    )
)


# ============================================================
# 12. FIND ADE20K STRUCTURAL CLASS IDS
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
print("ADE20K STRUCTURAL IDS")
print("-" * 110)

print(
    "WALL   :",
    wall_ids
)

print(
    "FLOOR  :",
    floor_ids
)

print(
    "CEILING:",
    ceiling_ids
)


if not wall_ids:

    raise RuntimeError(
        "ADE20K wall class not found."
    )


if not floor_ids:

    raise RuntimeError(
        "ADE20K floor class not found."
    )


if not ceiling_ids:

    print(
        "⚠️ Ceiling class not found."
    )


# ============================================================
# 13. SEGFORMER INFERENCE
# ============================================================

print()
print("=" * 110)
print("SEGFORMER INFERENCE")
print("=" * 110)


inputs = processor(
    images=original_pil,
    return_tensors="pt"
)


inputs = {

    key:
        value.to(
            DEVICE
        )

    for key, value
    in inputs.items()
}


inference_start = (
    time.time()
)


with torch.inference_mode():

    outputs = model(
        **inputs
    )


logits = outputs.logits


upsampled_logits = (
    torch.nn.functional.interpolate(
        logits,
        size=(
            H,
            W
        ),
        mode="bilinear",
        align_corners=False
    )
)


inference_seconds = (
    time.time()
    -
    inference_start
)


print(
    "✅ LOGITS READY"
)

print(
    "LOGITS:",
    tuple(
        upsampled_logits.shape
    )
)

print(
    "INFERENCE SEC:",
    round(
        inference_seconds,
        3
    )
)


# ============================================================
# 14. HARD ARGMAX MAP
# ============================================================

semantic_map = (
    upsampled_logits
    .argmax(
        dim=1
    )[0]
    .detach()
    .cpu()
    .numpy()
    .astype(
        np.int32
    )
)


# ============================================================
# 15. FULL PROBABILITY MAP
# ============================================================
#
# THIS IS THE IMPORTANT V2 CHANGE.
# ============================================================

probabilities = torch.softmax(
    upsampled_logits,
    dim=1
)[0]


# ============================================================
# 16. STRUCTURAL PROBABILITIES
# ============================================================

def aggregate_probability(
    probs,
    class_ids
):

    if not class_ids:

        return torch.zeros(
            (
                H,
                W
            ),
            device=probs.device,
            dtype=probs.dtype
        )

    return probs[
        class_ids
    ].sum(
        dim=0
    )


wall_prob_t = aggregate_probability(
    probabilities,
    wall_ids
)

floor_prob_t = aggregate_probability(
    probabilities,
    floor_ids
)

ceiling_prob_t = aggregate_probability(
    probabilities,
    ceiling_ids
)


wall_prob = (
    wall_prob_t
    .detach()
    .cpu()
    .numpy()
    .astype(
        np.float32
    )
)


floor_prob = (
    floor_prob_t
    .detach()
    .cpu()
    .numpy()
    .astype(
        np.float32
    )
)


ceiling_prob = (
    ceiling_prob_t
    .detach()
    .cpu()
    .numpy()
    .astype(
        np.float32
    )
)


# ============================================================
# 17. WINNER PROBABILITY
# ============================================================

winner_prob = (
    probabilities
    .max(
        dim=0
    )
    .values
    .detach()
    .cpu()
    .numpy()
    .astype(
        np.float32
    )
)


# ============================================================
# 18. SAVE PROBABILITY ARRAYS
# ============================================================

np.save(
    OUT
    / "00_wall_probability.npy",
    wall_prob
)

np.save(
    OUT
    / "01_floor_probability.npy",
    floor_prob
)

np.save(
    OUT
    / "02_ceiling_probability.npy",
    ceiling_prob
)

np.save(
    OUT
    / "03_winner_probability.npy",
    winner_prob
)


save_heatmap_gray(
    wall_prob,
    OUT
    / "04_wall_probability.png"
)

save_heatmap_gray(
    floor_prob,
    OUT
    / "05_floor_probability.png"
)

save_heatmap_gray(
    ceiling_prob,
    OUT
    / "06_ceiling_probability.png"
)

save_heatmap_gray(
    winner_prob,
    OUT
    / "07_winner_probability.png"
)


# ============================================================
# 19. RECREATE HARD ARGMAX STRUCTURAL MASKS
# ============================================================

hard_wall = np.zeros(
    (H, W),
    dtype=bool
)

hard_floor = np.zeros(
    (H, W),
    dtype=bool
)

hard_ceiling = np.zeros(
    (H, W),
    dtype=bool
)


for class_id in wall_ids:

    hard_wall |= (
        semantic_map
        ==
        class_id
    )


for class_id in floor_ids:

    hard_floor |= (
        semantic_map
        ==
        class_id
    )


for class_id in ceiling_ids:

    hard_ceiling |= (
        semantic_map
        ==
        class_id
    )


# ============================================================
# 20. TRUSTED HIGH-CONFIDENCE SEEDS
# ============================================================

trusted_wall = (
    wall_prob
    >=
    TRUSTED_SEED_PROB
)


trusted_floor = (
    floor_prob
    >=
    TRUSTED_SEED_PROB
)


trusted_ceiling = (
    ceiling_prob
    >=
    TRUSTED_SEED_PROB
)


# ============================================================
# 21. CONFIDENCE RECOVERY CANDIDATES
# ============================================================
#
# Example:
#
# floor = 0.42
# wall  = 0.43
#
# Old Stage01:
#     floor lost
#
# V2:
#     floor remains a possible recovery candidate.
# ============================================================

wall_gap = (
    winner_prob
    -
    wall_prob
)


floor_gap = (
    winner_prob
    -
    floor_prob
)


ceiling_gap = (
    winner_prob
    -
    ceiling_prob
)


wall_recovery_candidate = (
    (
        wall_prob
        >=
        RECOVERY_MIN_PROB
    )
    &
    (
        wall_gap
        <=
        RECOVERY_MAX_GAP_FROM_WINNER
    )
)


floor_recovery_candidate = (
    (
        floor_prob
        >=
        RECOVERY_MIN_PROB
    )
    &
    (
        floor_gap
        <=
        RECOVERY_MAX_GAP_FROM_WINNER
    )
)


ceiling_recovery_candidate = (
    (
        ceiling_prob
        >=
        RECOVERY_MIN_PROB
    )
    &
    (
        ceiling_gap
        <=
        RECOVERY_MAX_GAP_FROM_WINNER
    )
)


# ============================================================
# 22. INITIAL SOFT STRUCTURAL MASKS
# ============================================================

wall_candidate = (
    hard_wall
    |
    wall_recovery_candidate
)


floor_candidate = (
    hard_floor
    |
    floor_recovery_candidate
)


ceiling_candidate = (
    hard_ceiling
    |
    ceiling_recovery_candidate
)


# ============================================================
# 23. PROP EXCLUSION
# ============================================================
#
# Our frozen physical-prop mask is already trusted.
# Do not let structural recovery overwrite these objects.
# ============================================================

wall_candidate &= ~props

floor_candidate &= ~props

ceiling_candidate &= ~props


# ============================================================
# 24. COMPONENT / SEED FILTERING
# ============================================================

wall_filtered, wall_components = (
    component_filter_with_seed(
        wall_candidate,
        trusted_wall,
        hard_wall,
        min_area=MIN_COMPONENT_PIXELS
    )
)


floor_filtered, floor_components = (
    component_filter_with_seed(
        floor_candidate,
        trusted_floor,
        hard_floor,
        min_area=MIN_COMPONENT_PIXELS
    )
)


ceiling_filtered, ceiling_components = (
    component_filter_with_seed(
        ceiling_candidate,
        trusted_ceiling,
        hard_ceiling,
        min_area=MIN_COMPONENT_PIXELS
    )
)


# ============================================================
# 25. STRUCTURAL CLASS CONFLICT RESOLUTION
# ============================================================
#
# A pixel may be a near-tie candidate for multiple classes.
#
# In that case choose whichever structural probability is
# highest at that pixel.
# ============================================================

structural_probs = np.stack(
    [
        wall_prob,
        ceiling_prob,
        floor_prob
    ],
    axis=-1
)


allowed = np.stack(
    [
        wall_filtered,
        ceiling_filtered,
        floor_filtered
    ],
    axis=-1
)


masked_scores = np.where(
    allowed,
    structural_probs,
    -1.0
)


best_surface = np.argmax(
    masked_scores,
    axis=-1
)


has_surface = np.any(
    allowed,
    axis=-1
)


final_wall = (
    has_surface
    &
    (
        best_surface == 0
    )
)


final_ceiling = (
    has_surface
    &
    (
        best_surface == 1
    )
)


final_floor = (
    has_surface
    &
    (
        best_surface == 2
    )
)


# Props always excluded.
final_wall &= ~props
final_ceiling &= ~props
final_floor &= ~props


# ============================================================
# 26. OVERLAP ASSERTION
# ============================================================

assert not np.any(
    final_wall
    &
    final_floor
)

assert not np.any(
    final_wall
    &
    final_ceiling
)

assert not np.any(
    final_floor
    &
    final_ceiling
)


# ============================================================
# 27. SAVE HARD + RECOVERY + FINAL MASKS
# ============================================================

save_mask(
    hard_wall,
    OUT
    / "08_hard_wall_mask.png"
)

save_mask(
    hard_ceiling,
    OUT
    / "09_hard_ceiling_mask.png"
)

save_mask(
    hard_floor,
    OUT
    / "10_hard_floor_mask.png"
)


save_mask(
    wall_recovery_candidate,
    OUT
    / "11_wall_recovery_candidate.png"
)

save_mask(
    ceiling_recovery_candidate,
    OUT
    / "12_ceiling_recovery_candidate.png"
)

save_mask(
    floor_recovery_candidate,
    OUT
    / "13_floor_recovery_candidate.png"
)


save_mask(
    final_wall,
    OUT
    / "14_final_wall_mask_v2.png"
)

save_mask(
    final_ceiling,
    OUT
    / "15_final_ceiling_mask_v2.png"
)

save_mask(
    final_floor,
    OUT
    / "16_final_floor_mask_v2.png"
)


# ============================================================
# 28. COMPARE AGAINST OLD STAGE01
# ============================================================

wall_added = (
    final_wall
    &
    ~old_wall
)

wall_lost = (
    old_wall
    &
    ~final_wall
)


floor_added = (
    final_floor
    &
    ~old_floor
)

floor_lost = (
    old_floor
    &
    ~final_floor
)


ceiling_added = (
    final_ceiling
    &
    ~old_ceiling
)

ceiling_lost = (
    old_ceiling
    &
    ~final_ceiling
)


# ============================================================
# 29. SAVE DIFFERENCE MASKS
# ============================================================

save_mask(
    floor_added,
    OUT
    / "17_floor_added_vs_stage01.png"
)

save_mask(
    floor_lost,
    OUT
    / "18_floor_lost_vs_stage01.png"
)

save_mask(
    wall_added,
    OUT
    / "19_wall_added_vs_stage01.png"
)

save_mask(
    wall_lost,
    OUT
    / "20_wall_lost_vs_stage01.png"
)


# ============================================================
# 30. FINAL CLASS LABEL MAP
# ============================================================

surface_label_map = np.zeros(
    (
        H,
        W
    ),
    dtype=np.uint8
)


surface_label_map[
    final_wall
] = CLASS_WALL


surface_label_map[
    final_ceiling
] = CLASS_CEILING


surface_label_map[
    final_floor
] = CLASS_FLOOR


Image.fromarray(
    surface_label_map,
    mode="L"
).save(
    OUT
    / "21_surface_label_map_v2.png"
)


# ============================================================
# 31. SURFACE OVERLAY
# ============================================================

overlay = original.copy()


overlay = tint(
    overlay,
    final_wall,
    WALL_COLOR,
    0.50
)


overlay = tint(
    overlay,
    final_ceiling,
    CEILING_COLOR,
    0.45
)


overlay = tint(
    overlay,
    final_floor,
    FLOOR_COLOR,
    0.55
)


# Props highlighted magenta.
overlay = tint(
    overlay,
    props,
    [255, 0, 255],
    0.45
)


overlay_path = (
    OUT
    / "22_surface_overlay_v2.png"
)


Image.fromarray(
    overlay
).save(
    overlay_path
)


# ============================================================
# 32. FLOOR RECOVERY AUDIT OVERLAY
# ============================================================
#
# GREEN = final V2 floor
# CYAN  = V2 pixels newly recovered beyond old Stage01
# RED   = old floor pixels missing in V2
# MAGENTA = frozen props
# ============================================================

floor_audit = original.copy()


floor_audit = tint(
    floor_audit,
    final_floor,
    [0, 200, 0],
    0.45
)


floor_audit = tint(
    floor_audit,
    floor_added,
    [0, 255, 255],
    0.70
)


floor_audit = tint(
    floor_audit,
    floor_lost,
    [255, 0, 0],
    0.60
)


floor_audit = tint(
    floor_audit,
    props,
    [255, 0, 255],
    0.45
)


floor_audit_path = (
    OUT
    / "23_floor_recovery_audit_overlay.png"
)


Image.fromarray(
    floor_audit
).save(
    floor_audit_path
)


# ============================================================
# 33. OLD VS V2 FLOOR COMPARISON
# ============================================================

old_overlay = original.copy()

old_overlay = tint(
    old_overlay,
    old_floor,
    [0, 200, 0],
    0.55
)


new_overlay = original.copy()

new_overlay = tint(
    new_overlay,
    final_floor,
    [0, 200, 0],
    0.55
)


LABEL_H = 40


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
        old_overlay
    ),
    (
        W,
        LABEL_H
    )
)


comparison.paste(
    Image.fromarray(
        new_overlay
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
    "OLD STAGE01 HARD FLOOR",
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
    "SEGFORMER V2 CONFIDENCE FLOOR",
    fill=(
        0,
        0,
        0
    )
)


comparison_path = (
    OUT
    / "24_old_vs_v2_floor_comparison.png"
)


comparison.save(
    comparison_path
)


# ============================================================
# 34. PROBABILITY DIAGNOSTIC
# ============================================================

heat_w = Image.open(
    OUT
    / "04_wall_probability.png"
).convert(
    "RGB"
)


heat_f = Image.open(
    OUT
    / "05_floor_probability.png"
).convert(
    "RGB"
)


heat_c = Image.open(
    OUT
    / "06_ceiling_probability.png"
).convert(
    "RGB"
)


prob_compare = Image.new(
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


prob_compare.paste(
    heat_w,
    (
        0,
        LABEL_H
    )
)


prob_compare.paste(
    heat_f,
    (
        W,
        LABEL_H
    )
)


prob_compare.paste(
    heat_c,
    (
        W * 2,
        LABEL_H
    )
)


prob_draw = ImageDraw.Draw(
    prob_compare
)


prob_draw.text(
    (8, 10),
    "P(WALL)",
    fill=(0, 0, 0)
)


prob_draw.text(
    (W + 8, 10),
    "P(FLOOR)",
    fill=(0, 0, 0)
)


prob_draw.text(
    (W * 2 + 8, 10),
    "P(CEILING)",
    fill=(0, 0, 0)
)


prob_compare_path = (
    OUT
    / "25_structural_probability_comparison.png"
)


prob_compare.save(
    prob_compare_path
)


# ============================================================
# 35. PRINT COUNTS
# ============================================================

print()
print("=" * 110)
print("SURFACE SEGMENTATION V2 COUNTS")
print("=" * 110)

print()

print(
    "OLD WALL:",
    int(
        old_wall.sum()
    )
)

print(
    "V2 WALL :",
    int(
        final_wall.sum()
    )
)

print(
    "WALL ADDED:",
    int(
        wall_added.sum()
    )
)

print(
    "WALL LOST :",
    int(
        wall_lost.sum()
    )
)


print()

print(
    "OLD FLOOR:",
    int(
        old_floor.sum()
    )
)

print(
    "V2 FLOOR :",
    int(
        final_floor.sum()
    )
)

print(
    "FLOOR ADDED:",
    int(
        floor_added.sum()
    )
)

print(
    "FLOOR LOST :",
    int(
        floor_lost.sum()
    )
)


print()

print(
    "OLD CEILING:",
    int(
        old_ceiling.sum()
    )
)

print(
    "V2 CEILING :",
    int(
        final_ceiling.sum()
    )
)


# ============================================================
# 36. PROBABILITY STATISTICS
# ============================================================

def stats_dict(array):

    return {
        "min":
            float(
                array.min()
            ),

        "max":
            float(
                array.max()
            ),

        "mean":
            float(
                array.mean()
            ),

        "p50":
            float(
                np.percentile(
                    array,
                    50
                )
            ),

        "p90":
            float(
                np.percentile(
                    array,
                    90
                )
            ),

        "p95":
            float(
                np.percentile(
                    array,
                    95
                )
            ),
    }


# ============================================================
# 37. REPORT
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

    "architecture":
        (
            "SegFormer B5 ADE20K logits -> softmax probabilities "
            "-> hard seeds + near-winner confidence recovery "
            "-> connected component filtering "
            "-> probability-based structural conflict resolution"
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

    "thresholds": {

        "recovery_min_probability":
            RECOVERY_MIN_PROB,

        "recovery_max_gap_from_winner":
            RECOVERY_MAX_GAP_FROM_WINNER,

        "trusted_seed_probability":
            TRUSTED_SEED_PROB,

        "minimum_component_pixels":
            MIN_COMPONENT_PIXELS,
    },

    "probability_statistics": {

        "wall":
            stats_dict(
                wall_prob
            ),

        "floor":
            stats_dict(
                floor_prob
            ),

        "ceiling":
            stats_dict(
                ceiling_prob
            ),

        "winner":
            stats_dict(
                winner_prob
            ),
    },

    "counts": {

        "old_wall":
            int(
                old_wall.sum()
            ),

        "v2_wall":
            int(
                final_wall.sum()
            ),

        "wall_added":
            int(
                wall_added.sum()
            ),

        "wall_lost":
            int(
                wall_lost.sum()
            ),

        "old_floor":
            int(
                old_floor.sum()
            ),

        "v2_floor":
            int(
                final_floor.sum()
            ),

        "floor_added":
            int(
                floor_added.sum()
            ),

        "floor_lost":
            int(
                floor_lost.sum()
            ),

        "old_ceiling":
            int(
                old_ceiling.sum()
            ),

        "v2_ceiling":
            int(
                final_ceiling.sum()
            ),

        "ceiling_added":
            int(
                ceiling_added.sum()
            ),

        "ceiling_lost":
            int(
                ceiling_lost.sum()
            ),

        "prop_pixels":
            int(
                props.sum()
            ),
    },

    "components": {

        "wall":
            wall_components,

        "floor":
            floor_components,

        "ceiling":
            ceiling_components,
    },

    "sources": {

        "original":
            str(
                ORIGINAL_PATH
            ),

        "old_wall":
            str(
                OLD_WALL_PATH
            ),

        "old_floor":
            str(
                OLD_FLOOR_PATH
            ),

        "old_ceiling":
            str(
                OLD_CEILING_PATH
            ),

        "prop_mask":
            str(
                PROP_MASK_PATH
            ),
    },

    "hashes": {

        "original":
            sha256_file(
                ORIGINAL_PATH
            ),

        "old_wall":
            sha256_file(
                OLD_WALL_PATH
            ),

        "old_floor":
            sha256_file(
                OLD_FLOOR_PATH
            ),

        "old_ceiling":
            sha256_file(
                OLD_CEILING_PATH
            ),
    },

    "outputs": {

        "wall_probability":
            str(
                OUT
                / "00_wall_probability.npy"
            ),

        "floor_probability":
            str(
                OUT
                / "01_floor_probability.npy"
            ),

        "ceiling_probability":
            str(
                OUT
                / "02_ceiling_probability.npy"
            ),

        "final_wall_mask":
            str(
                OUT
                / "14_final_wall_mask_v2.png"
            ),

        "final_ceiling_mask":
            str(
                OUT
                / "15_final_ceiling_mask_v2.png"
            ),

        "final_floor_mask":
            str(
                OUT
                / "16_final_floor_mask_v2.png"
            ),

        "surface_overlay":
            str(
                overlay_path
            ),

        "floor_audit":
            str(
                floor_audit_path
            ),

        "old_vs_v2_floor":
            str(
                comparison_path
            ),

        "probability_comparison":
            str(
                prob_compare_path
            ),
    },

    "benchmark_status": {
        "old_stage01":
            "hard argmax reference",

        "surface_segmentation_v2":
            "pending visual review",
    },
}


report_path = (
    OUT
    / "00_surface_segmentation_v2_report.json"
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
# 38. RELEASE MODEL
# ============================================================

try:
    del outputs
except Exception:
    pass

try:
    del logits
except Exception:
    pass

try:
    del upsampled_logits
except Exception:
    pass

try:
    del probabilities
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
# 39. FINAL
# ============================================================

print()
print("=" * 110)
print("SURFACE SEGMENTATION V2 COMPLETE")
print("=" * 110)

print()

print(
    "FINAL WALL MASK:"
)

print(
    OUT
    / "14_final_wall_mask_v2.png"
)


print()

print(
    "FINAL FLOOR MASK:"
)

print(
    OUT
    / "16_final_floor_mask_v2.png"
)


print()

print(
    "FINAL CEILING MASK:"
)

print(
    OUT
    / "15_final_ceiling_mask_v2.png"
)


print()

print(
    "REPORT:"
)

print(
    report_path
)


# ============================================================
# 40. DISPLAY
# ============================================================

print()
print(
    "STRUCTURAL PROBABILITY MAPS"
)

display(
    prob_compare
)


print()
print(
    "SURFACE SEGMENTATION V2 OVERLAY"
)

display(
    Image.open(
        overlay_path
    )
)


print()
print(
    "FLOOR RECOVERY AUDIT"
)

display(
    Image.open(
        floor_audit_path
    )
)


print()
print(
    "OLD STAGE01 FLOOR VS CONFIDENCE-AWARE V2"
)

display(
    comparison
)
