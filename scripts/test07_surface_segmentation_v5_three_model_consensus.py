
# ============================================================
# TEST07
# SURFACE SEGMENTATION V5
#
# THREE-MODEL CONSENSUS + UNCERTAINTY MAP
#
# MODELS:
# - SegFormer B5 ADE20K
# - OneFormer ADE20K
# - Mask2Former ADE20K
#
# GOAL:
# Produce coarse semantic priors for later geometry refinement.
#
# 3/3 agreement = trusted seed
# 2/3 agreement = probable surface
# 1/3 only      = uncertain/disagreement
# ============================================================

from pathlib import Path
from PIL import Image, ImageDraw
from IPython.display import display

import json
import hashlib
import numpy as np


# ============================================================
# 1. IDENTIFIERS / PATHS
# ============================================================

STAGE_NAME = (
    "TEST07_SURFACE_SEGMENTATION_V5_"
    "THREE_MODEL_CONSENSUS"
)

SCRIPT_NAME = (
    "test07_surface_segmentation_v5_"
    "three_model_consensus.py"
)

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
    / "surface_segmentation_v5_three_model_consensus"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. SOURCE PATHS
# ============================================================

ORIGINAL_PATH = (
    ROOT
    / "01_structure_inverse_base"
    / "00_input_resized.png"
)


# ------------------------------------------------------------
# SEGFORMER V2
# ------------------------------------------------------------

SEG_WALL_PATH = (
    ROOT
    / "surface_segmentation_v2_confidence_aware"
    / "14_final_wall_mask_v2.png"
)

SEG_CEILING_PATH = (
    ROOT
    / "surface_segmentation_v2_confidence_aware"
    / "15_final_ceiling_mask_v2.png"
)

SEG_FLOOR_PATH = (
    ROOT
    / "surface_segmentation_v2_confidence_aware"
    / "16_final_floor_mask_v2.png"
)


# ------------------------------------------------------------
# ONEFORMER
# ------------------------------------------------------------

ONE_WALL_PATH = (
    ROOT
    / "surface_segmentation_v3_oneformer_ade20k"
    / "04_prop_safe_wall_mask.png"
)

ONE_CEILING_PATH = (
    ROOT
    / "surface_segmentation_v3_oneformer_ade20k"
    / "06_prop_safe_ceiling_mask.png"
)

ONE_FLOOR_PATH = (
    ROOT
    / "surface_segmentation_v3_oneformer_ade20k"
    / "05_prop_safe_floor_mask.png"
)


# ------------------------------------------------------------
# MASK2FORMER
# ------------------------------------------------------------

MASK2_WALL_PATH = (
    ROOT
    / "surface_segmentation_v4_mask2former_ade20k"
    / "04_prop_safe_wall_mask.png"
)

MASK2_CEILING_PATH = (
    ROOT
    / "surface_segmentation_v4_mask2former_ade20k"
    / "06_prop_safe_ceiling_mask.png"
)

MASK2_FLOOR_PATH = (
    ROOT
    / "surface_segmentation_v4_mask2former_ade20k"
    / "05_prop_safe_floor_mask.png"
)


# ------------------------------------------------------------
# FROZEN PROP MASK
# ------------------------------------------------------------

PROP_MASK_PATH = (
    ROOT
    / "01y3_frozen_canonical_prop_protection"
    / "00_canonical_prop_protection_mask.png"
)


# ============================================================
# 3. HELPERS
# ============================================================

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


def tint(
    image,
    mask,
    color,
    alpha
):

    out = image.copy().astype(
        np.float32
    )

    c = np.array(
        color,
        dtype=np.float32
    )

    out[mask] = (
        out[mask] * (1.0 - alpha)
        +
        c * alpha
    )

    return np.clip(
        out,
        0,
        255
    ).astype(np.uint8)


def agreement_maps(
    m1,
    m2,
    m3
):

    votes = (
        m1.astype(np.uint8)
        +
        m2.astype(np.uint8)
        +
        m3.astype(np.uint8)
    )

    agree_3 = (
        votes == 3
    )

    agree_2 = (
        votes == 2
    )

    agree_1 = (
        votes == 1
    )

    any_surface = (
        votes >= 1
    )

    majority = (
        votes >= 2
    )

    return (
        votes,
        agree_3,
        agree_2,
        agree_1,
        majority,
        any_surface,
    )


# ============================================================
# 4. VALIDATE
# ============================================================

required = {
    "ORIGINAL": ORIGINAL_PATH,

    "SEG WALL": SEG_WALL_PATH,
    "SEG FLOOR": SEG_FLOOR_PATH,
    "SEG CEILING": SEG_CEILING_PATH,

    "ONE WALL": ONE_WALL_PATH,
    "ONE FLOOR": ONE_FLOOR_PATH,
    "ONE CEILING": ONE_CEILING_PATH,

    "MASK2 WALL": MASK2_WALL_PATH,
    "MASK2 FLOOR": MASK2_FLOOR_PATH,
    "MASK2 CEILING": MASK2_CEILING_PATH,

    "PROPS": PROP_MASK_PATH,
}


print()
print("=" * 110)
print(STAGE_NAME)
print("=" * 110)

print()
print("SCRIPT:", SCRIPT_NAME)
print("OUTPUT:", OUT)

print()
print("INPUT CHECK")
print("-" * 110)


for name, path in required.items():

    ok = path.exists()

    print(
        f"{name:20s}",
        "✅" if ok else "❌",
        path
    )

    if not ok:

        raise FileNotFoundError(
            path
        )


# ============================================================
# 5. LOAD ORIGINAL
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


print()
print(
    "MASTER SIZE:",
    W,
    "x",
    H
)


# ============================================================
# 6. LOAD MASKS
# ============================================================

seg_wall = load_mask(
    SEG_WALL_PATH,
    SIZE
)

seg_floor = load_mask(
    SEG_FLOOR_PATH,
    SIZE
)

seg_ceiling = load_mask(
    SEG_CEILING_PATH,
    SIZE
)


one_wall = load_mask(
    ONE_WALL_PATH,
    SIZE
)

one_floor = load_mask(
    ONE_FLOOR_PATH,
    SIZE
)

one_ceiling = load_mask(
    ONE_CEILING_PATH,
    SIZE
)


mask2_wall = load_mask(
    MASK2_WALL_PATH,
    SIZE
)

mask2_floor = load_mask(
    MASK2_FLOOR_PATH,
    SIZE
)

mask2_ceiling = load_mask(
    MASK2_CEILING_PATH,
    SIZE
)


props = load_mask(
    PROP_MASK_PATH,
    SIZE
)


# ============================================================
# 7. FORCE PROP EXCLUSION FOR FAIR CONSENSUS
# ============================================================

for m in [
    seg_wall,
    seg_floor,
    seg_ceiling,
    one_wall,
    one_floor,
    one_ceiling,
    mask2_wall,
    mask2_floor,
    mask2_ceiling,
]:

    m &= ~props


# ============================================================
# 8. FLOOR CONSENSUS
# ============================================================

(
    floor_votes,
    floor_3,
    floor_2,
    floor_1,
    floor_majority,
    floor_any,
) = agreement_maps(
    seg_floor,
    one_floor,
    mask2_floor
)


# ============================================================
# 9. WALL CONSENSUS
# ============================================================

(
    wall_votes,
    wall_3,
    wall_2,
    wall_1,
    wall_majority,
    wall_any,
) = agreement_maps(
    seg_wall,
    one_wall,
    mask2_wall
)


# ============================================================
# 10. CEILING CONSENSUS
# ============================================================

(
    ceiling_votes,
    ceiling_3,
    ceiling_2,
    ceiling_1,
    ceiling_majority,
    ceiling_any,
) = agreement_maps(
    seg_ceiling,
    one_ceiling,
    mask2_ceiling
)


# ============================================================
# 11. SAVE FLOOR MAPS
# ============================================================

Image.fromarray(
    floor_votes.astype(np.uint8),
    mode="L"
).save(
    OUT
    / "00_floor_vote_count.png"
)

save_mask(
    floor_3,
    OUT
    / "01_floor_agree_3of3.png"
)

save_mask(
    floor_2,
    OUT
    / "02_floor_agree_2of3.png"
)

save_mask(
    floor_1,
    OUT
    / "03_floor_agree_1of3.png"
)

save_mask(
    floor_majority,
    OUT
    / "04_floor_majority_2plus.png"
)

save_mask(
    floor_any,
    OUT
    / "05_floor_any_model.png"
)


# ============================================================
# 12. SAVE WALL MAPS
# ============================================================

Image.fromarray(
    wall_votes.astype(np.uint8),
    mode="L"
).save(
    OUT
    / "06_wall_vote_count.png"
)

save_mask(
    wall_3,
    OUT
    / "07_wall_agree_3of3.png"
)

save_mask(
    wall_2,
    OUT
    / "08_wall_agree_2of3.png"
)

save_mask(
    wall_1,
    OUT
    / "09_wall_agree_1of3.png"
)

save_mask(
    wall_majority,
    OUT
    / "10_wall_majority_2plus.png"
)


# ============================================================
# 13. SAVE CEILING MAPS
# ============================================================

Image.fromarray(
    ceiling_votes.astype(np.uint8),
    mode="L"
).save(
    OUT
    / "11_ceiling_vote_count.png"
)

save_mask(
    ceiling_3,
    OUT
    / "12_ceiling_agree_3of3.png"
)

save_mask(
    ceiling_2,
    OUT
    / "13_ceiling_agree_2of3.png"
)

save_mask(
    ceiling_1,
    OUT
    / "14_ceiling_agree_1of3.png"
)

save_mask(
    ceiling_majority,
    OUT
    / "15_ceiling_majority_2plus.png"
)


# ============================================================
# 14. CONSENSUS FLOOR OVERLAY
# ============================================================
#
# DARK GREEN = 3/3 agreement
# YELLOW     = 2/3 agreement
# RED        = 1/3 only
# MAGENTA    = frozen props
# ============================================================

floor_overlay = original.copy()


floor_overlay = tint(
    floor_overlay,
    floor_3,
    [0, 170, 0],
    0.65
)


floor_overlay = tint(
    floor_overlay,
    floor_2,
    [255, 220, 0],
    0.70
)


floor_overlay = tint(
    floor_overlay,
    floor_1,
    [255, 0, 0],
    0.70
)


floor_overlay = tint(
    floor_overlay,
    props,
    [255, 0, 255],
    0.45
)


floor_overlay_path = (
    OUT
    / "16_floor_consensus_overlay.png"
)


Image.fromarray(
    floor_overlay
).save(
    floor_overlay_path
)


# ============================================================
# 15. WALL CONSENSUS OVERLAY
# ============================================================
#
# BLUE   = 3/3
# CYAN   = 2/3
# RED    = 1/3
# MAGENTA = props
# ============================================================

wall_overlay = original.copy()


wall_overlay = tint(
    wall_overlay,
    wall_3,
    [0, 102, 255],
    0.65
)


wall_overlay = tint(
    wall_overlay,
    wall_2,
    [0, 255, 255],
    0.65
)


wall_overlay = tint(
    wall_overlay,
    wall_1,
    [255, 0, 0],
    0.70
)


wall_overlay = tint(
    wall_overlay,
    props,
    [255, 0, 255],
    0.45
)


wall_overlay_path = (
    OUT
    / "17_wall_consensus_overlay.png"
)


Image.fromarray(
    wall_overlay
).save(
    wall_overlay_path
)


# ============================================================
# 16. CEILING CONSENSUS OVERLAY
# ============================================================

ceiling_overlay = original.copy()


ceiling_overlay = tint(
    ceiling_overlay,
    ceiling_3,
    [255, 255, 255],
    0.65
)


ceiling_overlay = tint(
    ceiling_overlay,
    ceiling_2,
    [255, 220, 0],
    0.70
)


ceiling_overlay = tint(
    ceiling_overlay,
    ceiling_1,
    [255, 0, 0],
    0.70
)


ceiling_overlay = tint(
    ceiling_overlay,
    props,
    [255, 0, 255],
    0.45
)


ceiling_overlay_path = (
    OUT
    / "18_ceiling_consensus_overlay.png"
)


Image.fromarray(
    ceiling_overlay
).save(
    ceiling_overlay_path
)


# ============================================================
# 17. MAJORITY SURFACE MAP
# ============================================================
#
# This is our coarse semantic prior:
#
# floor >= 2 votes
# wall >= 2 votes
# ceiling >= 2 votes
#
# Resolve accidental cross-class conflicts by highest vote count.
# ============================================================

vote_stack = np.stack(
    [
        wall_votes,
        ceiling_votes,
        floor_votes,
    ],
    axis=-1
)


majority_allowed = (
    vote_stack >= 2
)


masked_votes = np.where(
    majority_allowed,
    vote_stack,
    -1
)


best_class = np.argmax(
    masked_votes,
    axis=-1
)


has_majority_surface = np.any(
    majority_allowed,
    axis=-1
)


canonical_wall = (
    has_majority_surface
    &
    (
        best_class == 0
    )
)


canonical_ceiling = (
    has_majority_surface
    &
    (
        best_class == 1
    )
)


canonical_floor = (
    has_majority_surface
    &
    (
        best_class == 2
    )
)


canonical_wall &= ~props
canonical_ceiling &= ~props
canonical_floor &= ~props


save_mask(
    canonical_wall,
    OUT
    / "19_canonical_wall_majority.png"
)

save_mask(
    canonical_ceiling,
    OUT
    / "20_canonical_ceiling_majority.png"
)

save_mask(
    canonical_floor,
    OUT
    / "21_canonical_floor_majority.png"
)


# ============================================================
# 18. UNCERTAINTY MAP
# ============================================================
#
# 0 = no structural model predicts surface
# 1 = disagreement / one model only
# 2 = moderate confidence / 2 models
# 3 = high confidence / all 3 models
#
# We use the strongest structural vote at each pixel.
# ============================================================

max_votes = np.max(
    vote_stack,
    axis=-1
).astype(
    np.uint8
)


Image.fromarray(
    max_votes,
    mode="L"
).save(
    OUT
    / "22_structural_uncertainty_vote_map.png"
)


# ============================================================
# 19. COLOR UNCERTAINTY OVERLAY
# ============================================================
#
# GREEN = 3-vote trusted
# YELLOW = 2-vote probable
# RED = 1-vote disagreement
# ============================================================

uncertainty_overlay = original.copy()


uncertainty_overlay = tint(
    uncertainty_overlay,
    max_votes == 3,
    [0, 200, 0],
    0.45
)


uncertainty_overlay = tint(
    uncertainty_overlay,
    max_votes == 2,
    [255, 220, 0],
    0.55
)


uncertainty_overlay = tint(
    uncertainty_overlay,
    max_votes == 1,
    [255, 0, 0],
    0.65
)


uncertainty_overlay = tint(
    uncertainty_overlay,
    props,
    [255, 0, 255],
    0.45
)


uncertainty_overlay_path = (
    OUT
    / "23_structural_uncertainty_overlay.png"
)


Image.fromarray(
    uncertainty_overlay
).save(
    uncertainty_overlay_path
)


# ============================================================
# 20. CANONICAL PRIOR OVERLAY
# ============================================================

canonical_overlay = original.copy()


canonical_overlay = tint(
    canonical_overlay,
    canonical_wall,
    [0, 102, 255],
    0.50
)


canonical_overlay = tint(
    canonical_overlay,
    canonical_ceiling,
    [255, 255, 255],
    0.45
)


canonical_overlay = tint(
    canonical_overlay,
    canonical_floor,
    [0, 200, 0],
    0.55
)


canonical_overlay = tint(
    canonical_overlay,
    props,
    [255, 0, 255],
    0.45
)


canonical_overlay_path = (
    OUT
    / "24_canonical_majority_surface_overlay.png"
)


Image.fromarray(
    canonical_overlay
).save(
    canonical_overlay_path
)


# ============================================================
# 21. 4-PANEL SUMMARY
# ============================================================

LABEL_H = 42


summary = Image.new(
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


summary.paste(
    original_pil,
    (
        0,
        LABEL_H
    )
)


summary.paste(
    Image.fromarray(
        floor_overlay
    ),
    (
        W,
        LABEL_H
    )
)


summary.paste(
    Image.fromarray(
        wall_overlay
    ),
    (
        W * 2,
        LABEL_H
    )
)


summary.paste(
    Image.fromarray(
        uncertainty_overlay
    ),
    (
        W * 3,
        LABEL_H
    )
)


draw = ImageDraw.Draw(
    summary
)


for x, text in [
    (8, "ORIGINAL"),
    (W + 8, "FLOOR CONSENSUS"),
    (W * 2 + 8, "WALL CONSENSUS"),
    (W * 3 + 8, "UNCERTAINTY"),
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


summary_path = (
    OUT
    / "25_consensus_summary.png"
)


summary.save(
    summary_path
)


# ============================================================
# 22. COUNTS
# ============================================================

print()
print("=" * 110)
print("THREE-MODEL CONSENSUS COUNTS")
print("=" * 110)


print()
print("FLOOR")
print(
    "3/3 agree:",
    int(
        floor_3.sum()
    )
)
print(
    "2/3 agree:",
    int(
        floor_2.sum()
    )
)
print(
    "1/3 only :",
    int(
        floor_1.sum()
    )
)
print(
    "Majority :",
    int(
        floor_majority.sum()
    )
)


print()
print("WALL")
print(
    "3/3 agree:",
    int(
        wall_3.sum()
    )
)
print(
    "2/3 agree:",
    int(
        wall_2.sum()
    )
)
print(
    "1/3 only :",
    int(
        wall_1.sum()
    )
)
print(
    "Majority :",
    int(
        wall_majority.sum()
    )
)


print()
print("CEILING")
print(
    "3/3 agree:",
    int(
        ceiling_3.sum()
    )
)
print(
    "2/3 agree:",
    int(
        ceiling_2.sum()
    )
)
print(
    "1/3 only :",
    int(
        ceiling_1.sum()
    )
)
print(
    "Majority :",
    int(
        ceiling_majority.sum()
    )
)


# ============================================================
# 23. REPORT
# ============================================================

report = {

    "stage":
        STAGE_NAME,

    "status":
        "COMPLETED",

    "script":
        SCRIPT_NAME,

    "purpose":
        (
            "Fuse SegFormer, OneFormer and Mask2Former into "
            "high-confidence structural semantic priors and "
            "explicit disagreement/uncertainty maps."
        ),

    "models": [
        "SegFormer B5 ADE20K",
        "OneFormer ADE20K",
        "Mask2Former ADE20K",
    ],

    "consensus_rule": {

        "high_confidence":
            "3 of 3 models agree",

        "medium_confidence":
            "2 of 3 models agree",

        "disagreement":
            "exactly 1 model predicts surface",

        "canonical_surface":
            "majority vote >= 2 models",
    },

    "counts": {

        "floor": {
            "agree_3":
                int(
                    floor_3.sum()
                ),

            "agree_2":
                int(
                    floor_2.sum()
                ),

            "agree_1":
                int(
                    floor_1.sum()
                ),

            "majority":
                int(
                    floor_majority.sum()
                ),
        },

        "wall": {
            "agree_3":
                int(
                    wall_3.sum()
                ),

            "agree_2":
                int(
                    wall_2.sum()
                ),

            "agree_1":
                int(
                    wall_1.sum()
                ),

            "majority":
                int(
                    wall_majority.sum()
                ),
        },

        "ceiling": {
            "agree_3":
                int(
                    ceiling_3.sum()
                ),

            "agree_2":
                int(
                    ceiling_2.sum()
                ),

            "agree_1":
                int(
                    ceiling_1.sum()
                ),

            "majority":
                int(
                    ceiling_majority.sum()
                ),
        },
    },

    "sources": {

        "original":
            str(
                ORIGINAL_PATH
            ),

        "segformer_wall":
            str(
                SEG_WALL_PATH
            ),

        "segformer_floor":
            str(
                SEG_FLOOR_PATH
            ),

        "segformer_ceiling":
            str(
                SEG_CEILING_PATH
            ),

        "oneformer_wall":
            str(
                ONE_WALL_PATH
            ),

        "oneformer_floor":
            str(
                ONE_FLOOR_PATH
            ),

        "oneformer_ceiling":
            str(
                ONE_CEILING_PATH
            ),

        "mask2former_wall":
            str(
                MASK2_WALL_PATH
            ),

        "mask2former_floor":
            str(
                MASK2_FLOOR_PATH
            ),

        "mask2former_ceiling":
            str(
                MASK2_CEILING_PATH
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

        "seg_floor":
            sha256_file(
                SEG_FLOOR_PATH
            ),

        "one_floor":
            sha256_file(
                ONE_FLOOR_PATH
            ),

        "mask2_floor":
            sha256_file(
                MASK2_FLOOR_PATH
            ),
    },

    "outputs": {

        "floor_3of3":
            str(
                OUT
                / "01_floor_agree_3of3.png"
            ),

        "floor_2of3":
            str(
                OUT
                / "02_floor_agree_2of3.png"
            ),

        "floor_disagreement":
            str(
                OUT
                / "03_floor_agree_1of3.png"
            ),

        "canonical_floor":
            str(
                OUT
                / "21_canonical_floor_majority.png"
            ),

        "canonical_wall":
            str(
                OUT
                / "19_canonical_wall_majority.png"
            ),

        "canonical_ceiling":
            str(
                OUT
                / "20_canonical_ceiling_majority.png"
            ),

        "floor_overlay":
            str(
                floor_overlay_path
            ),

        "wall_overlay":
            str(
                wall_overlay_path
            ),

        "uncertainty_overlay":
            str(
                uncertainty_overlay_path
            ),

        "canonical_overlay":
            str(
                canonical_overlay_path
            ),

        "summary":
            str(
                summary_path
            ),
    },

    "next_stage":
        (
            "Use trusted consensus structural priors as input "
            "to depth/3D geometry boundary refinement."
        ),
}


report_path = (
    OUT
    / "00_surface_segmentation_v5_consensus_report.json"
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
# 24. FINAL
# ============================================================

print()
print("=" * 110)
print("SURFACE SEGMENTATION V5 COMPLETE")
print("=" * 110)

print()
print(
    "FLOOR CONSENSUS:",
    floor_overlay_path
)

print(
    "WALL CONSENSUS:",
    wall_overlay_path
)

print(
    "UNCERTAINTY:",
    uncertainty_overlay_path
)

print(
    "CANONICAL PRIOR:",
    canonical_overlay_path
)

print(
    "REPORT:",
    report_path
)


# ============================================================
# 25. DISPLAY
# ============================================================

print()
print(
    "FLOOR CONSENSUS"
)

display(
    Image.open(
        floor_overlay_path
    )
)


print()
print(
    "WALL CONSENSUS"
)

display(
    Image.open(
        wall_overlay_path
    )
)


print()
print(
    "STRUCTURAL UNCERTAINTY"
)

display(
    Image.open(
        uncertainty_overlay_path
    )
)


print()
print(
    "CANONICAL MAJORITY SURFACE PRIOR"
)

display(
    Image.open(
        canonical_overlay_path
    )
)


print()
print(
    "SUMMARY"
)

display(
    summary
)
