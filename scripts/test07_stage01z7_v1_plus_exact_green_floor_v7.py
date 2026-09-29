
# ============================================================
# TEST07 - ROOM03
# STAGE01Z7
#
# V1 CLEANROOM + EXACT GREEN FLOOR
#
# PURPOSE
# ------------------------------------------------------------
# Preserve the successful V1 result for:
#
# - wall cleanup
# - ceiling coherence
# - mirror removal
# - shower-glass removal
# - reconstructed hidden architecture
#
# Change ONLY:
#
# FLOOR -> exact solid #00C800
#
# Genuine props are restored exactly from ORIGINAL RGB.
#
# NO NEW QWEN RUN.
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

from pathlib import Path
from PIL import Image, ImageDraw
from IPython.display import display

import hashlib
import json
import numpy as np


# ============================================================
# 2. IDENTIFIERS
# ============================================================

STAGE_NAME = (
    "TEST07_STAGE01Z7_"
    "V1_PLUS_EXACT_GREEN_FLOOR_V7"
)

SCRIPT_NAME = (
    "test07_stage01z7_"
    "v1_plus_exact_green_floor_v7.py"
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
    / "01z7_v1_plus_exact_green_floor_v7"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# ORIGINAL
# ------------------------------------------------------------

ORIGINAL_PATH = (
    ROOT
    / "01_structure_inverse_base"
    / "00_input_resized.png"
)


# ------------------------------------------------------------
# EXISTING FLOOR MASK
# ------------------------------------------------------------

FLOOR_MASK_PATH = (
    ROOT
    / "01_structure_inverse_base"
    / "04_floor_mask.png"
)


# ------------------------------------------------------------
# FLOOR AREA THAT WAS BEHIND GLASS
# ------------------------------------------------------------

GLASS_TO_FLOOR_PATH = (
    ROOT
    / "01w2_v3_glass_background_classification"
    / "01_glass_to_floor.png"
)


# ------------------------------------------------------------
# FROZEN REAL-PROP MASK
# ------------------------------------------------------------

PROP_MASK_PATH = (
    ROOT
    / "01y3_frozen_canonical_prop_protection"
    / "00_canonical_prop_protection_mask.png"
)


# ------------------------------------------------------------
# SUCCESSFUL V1 CLEAN ROOM
#
# IMPORTANT:
# Use SAFE V1 rather than raw V1.
# ------------------------------------------------------------

V1_SAFE_PATH = (
    ROOT
    / "01z1_qwen_coherent_cleanroom_v1"
    / "02_safe_props_restored.png"
)


# ============================================================
# 4. TARGET FLOOR COLOR
# ============================================================

FLOOR_HEX = "#00C800"

FLOOR_RGB = np.array(
    [
        0,
        200,
        0
    ],
    dtype=np.uint8
)


# ============================================================
# 5. HELPERS
# ============================================================

def load_rgb(path):

    return np.array(
        Image.open(
            path
        ).convert(
            "RGB"
        )
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
        np.array(
            img
        ) > 0
    )


def save_mask(
    mask,
    path
):

    Image.fromarray(
        (
            mask.astype(
                np.uint8
            )
            * 255
        ),
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


# ============================================================
# 6. START
# ============================================================

print()
print("=" * 100)
print(STAGE_NAME)
print("=" * 100)

print()
print(
    "SCRIPT:",
    SCRIPT_NAME
)

print(
    "OUTPUT:",
    OUT
)

print()
print(
    "NO NEW QWEN GENERATION"
)

print(
    "USING SUCCESSFUL V1 CLEAN ROOM AS BASE"
)


# ============================================================
# 7. VALIDATE INPUTS
# ============================================================

required = {

    "ORIGINAL":
        ORIGINAL_PATH,

    "V1 SAFE":
        V1_SAFE_PATH,

    "FLOOR MASK":
        FLOOR_MASK_PATH,

    "GLASS->FLOOR":
        GLASS_TO_FLOOR_PATH,

    "PROP MASK":
        PROP_MASK_PATH,
}


print()
print("INPUT CHECK")
print("-" * 100)

for name, path in required.items():

    ok = (
        path.exists()
    )

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
# 8. LOAD ORIGINAL
# ============================================================

original_pil = Image.open(
    ORIGINAL_PATH
).convert(
    "RGB"
)

original = np.array(
    original_pil
)

H, W = (
    original.shape[:2]
)

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
# 9. LOAD V1 SAFE RESULT
# ============================================================

v1_pil = Image.open(
    V1_SAFE_PATH
).convert(
    "RGB"
)

if v1_pil.size != SIZE:

    v1_pil = v1_pil.resize(
        SIZE,
        Image.Resampling.LANCZOS
    )

v1 = np.array(
    v1_pil
)


# ============================================================
# 10. LOAD MASKS
# ============================================================

floor = load_mask(
    FLOOR_MASK_PATH,
    SIZE
)

glass_floor = load_mask(
    GLASS_TO_FLOOR_PATH,
    SIZE
)

props = load_mask(
    PROP_MASK_PATH,
    SIZE
)


print()
print("SOURCE MASK PIXELS")
print("-" * 100)

print(
    "Stage01 floor:",
    int(
        floor.sum()
    )
)

print(
    "glass->floor:",
    int(
        glass_floor.sum()
    )
)

print(
    "frozen props:",
    int(
        props.sum()
    )
)


# ============================================================
# 11. CREATE FINAL FLOOR MASK
# ============================================================
#
# Floor visible normally
# +
# floor revealed after glass removal
#
# Then exclude all frozen physical props.
# ============================================================

floor_augmented = (
    floor
    |
    glass_floor
)


floor_editable = (
    floor_augmented
    &
    ~props
)


print()
print(
    "AUGMENTED FLOOR:",
    int(
        floor_augmented.sum()
    )
)

print(
    "EDITABLE FLOOR:",
    int(
        floor_editable.sum()
    )
)


# ============================================================
# 12. SAVE FLOOR MASKS
# ============================================================

save_mask(
    floor,
    OUT
    / "00_stage01_floor_mask.png"
)

save_mask(
    glass_floor,
    OUT
    / "01_glass_to_floor_mask.png"
)

save_mask(
    floor_augmented,
    OUT
    / "02_augmented_floor_mask.png"
)

save_mask(
    floor_editable,
    OUT
    / "03_final_editable_floor_mask.png"
)

save_mask(
    props,
    OUT
    / "04_frozen_prop_mask.png"
)


# ============================================================
# 13. CREATE V7
# ============================================================
#
# Start EXACTLY from V1 safe result.
#
# Do not touch:
# - V1 walls
# - V1 ceiling
# - V1 mirror reconstruction
# - V1 glass reconstruction
# - any non-floor background
# ============================================================

v7 = (
    v1.copy()
)


# ============================================================
# 14. SET FLOOR TO EXACT #00C800
# ============================================================

v7[
    floor_editable
] = FLOOR_RGB


# ============================================================
# 15. RESTORE ORIGINAL PROPS AGAIN
# ============================================================
#
# Even though V1 was already safe,
# restore once more after floor compositing.
# ============================================================

v7[
    props
] = original[
    props
]


# ============================================================
# 16. SAVE FINAL V7
# ============================================================

v7_pil = Image.fromarray(
    v7
)

v7_path = (
    OUT
    / "05_v7_final_surface_template.png"
)

v7_pil.save(
    v7_path
)


# ============================================================
# 17. FLOOR EXACT-COLOR VERIFICATION
# ============================================================

floor_pixels = (
    v7[
        floor_editable
    ]
)


if floor_pixels.size:

    exact_floor_pixels = np.all(
        floor_pixels
        ==
        FLOOR_RGB,
        axis=1
    )

    floor_exact_ratio = float(
        exact_floor_pixels.mean()
    )

else:

    floor_exact_ratio = 1.0


# ============================================================
# 18. PROP RGB VERIFICATION
# ============================================================

prop_diff = np.abs(

    v7.astype(
        np.int16
    )

    -

    original.astype(
        np.int16
    )

)[
    props
]


if prop_diff.size:

    max_prop_rgb_error = int(
        prop_diff.max()
    )

    mean_prop_rgb_error = float(
        prop_diff.mean()
    )

else:

    max_prop_rgb_error = 0

    mean_prop_rgb_error = 0.0


# ============================================================
# 19. NON-FLOOR V1 PRESERVATION VERIFICATION
# ============================================================
#
# Outside:
# - editable floor
# - frozen props
#
# V7 must be IDENTICAL to V1.
# ============================================================

must_match_v1 = (
    ~floor_editable
    &
    ~props
)


v1_diff = np.abs(

    v7.astype(
        np.int16
    )

    -

    v1.astype(
        np.int16
    )

)[
    must_match_v1
]


if v1_diff.size:

    max_nonfloor_v1_error = int(
        v1_diff.max()
    )

else:

    max_nonfloor_v1_error = 0


# ============================================================
# 20. PRINT VERIFICATION
# ============================================================

print()
print("=" * 100)
print("V7 VERIFICATION")
print("=" * 100)

print(
    "FLOOR EXACT #00C800 %:",
    round(
        floor_exact_ratio
        * 100,
        6
    )
)

print(
    "MAX PROP RGB ERROR:",
    max_prop_rgb_error
)

print(
    "MEAN PROP RGB ERROR:",
    mean_prop_rgb_error
)

print(
    "MAX NON-FLOOR CHANGE VS V1:",
    max_nonfloor_v1_error
)


if floor_exact_ratio == 1.0:

    print(
        "✅ FLOOR IS EXACT #00C800"
    )


if max_prop_rgb_error == 0:

    print(
        "✅ PROP RGB EXACTLY MATCHES ORIGINAL"
    )


if max_nonfloor_v1_error == 0:

    print(
        "✅ ALL NON-FLOOR NON-PROP PIXELS "
        "ARE EXACTLY UNCHANGED FROM V1"
    )


# ============================================================
# 21. FLOOR OVERLAY ON ORIGINAL
# ============================================================

overlay = (
    original.copy()
    .astype(
        np.float32
    )
)


overlay[
    floor_editable
] = (

    overlay[
        floor_editable
    ]
    *
    0.35

    +

    FLOOR_RGB.astype(
        np.float32
    )
    *
    0.65
)


# Show protected props in magenta for debug
overlay[
    props
] = (

    overlay[
        props
    ]
    *
    0.45

    +

    np.array(
        [
            255,
            0,
            255
        ],
        dtype=np.float32
    )
    *
    0.55
)


overlay = np.clip(
    overlay,
    0,
    255
).astype(
    np.uint8
)


overlay_path = (
    OUT
    / "06_floor_and_prop_overlay.png"
)

Image.fromarray(
    overlay
).save(
    overlay_path
)


# ============================================================
# 22. THREE-PANEL COMPARISON
# ============================================================

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
    v1_pil,
    (
        W,
        LABEL_H
    )
)


comparison.paste(
    v7_pil,
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
    "V1 GOOD REFERENCE",
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
    "V7 = V1 + EXACT GREEN FLOOR",
    fill=(
        0,
        0,
        0
    )
)


comparison_path = (
    OUT
    / "07_original_v1_v7_comparison.png"
)

comparison.save(
    comparison_path
)


# ============================================================
# 23. V1 VS V7
# ============================================================

compare_v1_v7 = Image.new(
    "RGB",
    (
        W * 2,
        H + LABEL_H
    ),
    (
        255,
        255,
        255
    )
)


compare_v1_v7.paste(
    v1_pil,
    (
        0,
        LABEL_H
    )
)


compare_v1_v7.paste(
    v7_pil,
    (
        W,
        LABEL_H
    )
)


draw2 = ImageDraw.Draw(
    compare_v1_v7
)


draw2.text(
    (
        8,
        10
    ),
    "V1",
    fill=(
        0,
        0,
        0
    )
)


draw2.text(
    (
        W + 8,
        10
    ),
    "V7",
    fill=(
        0,
        0,
        0
    )
)


compare_v1_v7_path = (
    OUT
    / "08_v1_vs_v7.png"
)

compare_v1_v7.save(
    compare_v1_v7_path
)


# ============================================================
# 24. ORIGINAL VS V7
# ============================================================

compare_original_v7 = Image.new(
    "RGB",
    (
        W * 2,
        H + LABEL_H
    ),
    (
        255,
        255,
        255
    )
)


compare_original_v7.paste(
    original_pil,
    (
        0,
        LABEL_H
    )
)


compare_original_v7.paste(
    v7_pil,
    (
        W,
        LABEL_H
    )
)


draw3 = ImageDraw.Draw(
    compare_original_v7
)


draw3.text(
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


draw3.text(
    (
        W + 8,
        10
    ),
    "V7 FINAL",
    fill=(
        0,
        0,
        0
    )
)


compare_original_v7_path = (
    OUT
    / "09_original_vs_v7.png"
)

compare_original_v7.save(
    compare_original_v7_path
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

    "new_qwen_generation":
        False,

    "strategy":
        (
            "Use successful V1 clean-room reconstruction "
            "unchanged and deterministically replace only "
            "the editable floor with exact #00C800."
        ),

    "sources": {

        "original":
            str(
                ORIGINAL_PATH
            ),

        "v1_safe":
            str(
                V1_SAFE_PATH
            ),

        "floor_mask":
            str(
                FLOOR_MASK_PATH
            ),

        "glass_to_floor":
            str(
                GLASS_TO_FLOOR_PATH
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

        "v1_safe":
            sha256_file(
                V1_SAFE_PATH
            ),

        "floor_mask":
            sha256_file(
                FLOOR_MASK_PATH
            ),

        "glass_to_floor":
            sha256_file(
                GLASS_TO_FLOOR_PATH
            ),

        "prop_mask":
            sha256_file(
                PROP_MASK_PATH
            ),
    },

    "floor": {

        "target_hex":
            FLOOR_HEX,

        "target_rgb":
            FLOOR_RGB.tolist(),

        "stage01_floor_pixels":
            int(
                floor.sum()
            ),

        "glass_to_floor_pixels":
            int(
                glass_floor.sum()
            ),

        "augmented_floor_pixels":
            int(
                floor_augmented.sum()
            ),

        "editable_floor_pixels":
            int(
                floor_editable.sum()
            ),

        "exact_target_ratio":
            floor_exact_ratio,
    },

    "verification": {

        "max_prop_rgb_error":
            max_prop_rgb_error,

        "mean_prop_rgb_error":
            mean_prop_rgb_error,

        "max_nonfloor_nonprop_error_vs_v1":
            max_nonfloor_v1_error,
    },

    "rules": [

        "V1 remains untouched",

        "No new Qwen generation",

        "Do not modify V1 walls",

        "Do not modify V1 ceiling",

        "Do not modify V1 mirror reconstruction",

        "Do not modify V1 glass reconstruction",

        "Only editable floor becomes #00C800",

        "Frozen physical props restored exactly from original RGB",
    ],

    "outputs": {

        "stage01_floor_mask":
            str(
                OUT
                / "00_stage01_floor_mask.png"
            ),

        "glass_to_floor":
            str(
                OUT
                / "01_glass_to_floor_mask.png"
            ),

        "augmented_floor":
            str(
                OUT
                / "02_augmented_floor_mask.png"
            ),

        "final_floor_mask":
            str(
                OUT
                / "03_final_editable_floor_mask.png"
            ),

        "v7":
            str(
                v7_path
            ),

        "overlay":
            str(
                overlay_path
            ),

        "original_v1_v7":
            str(
                comparison_path
            ),

        "v1_vs_v7":
            str(
                compare_v1_v7_path
            ),

        "original_vs_v7":
            str(
                compare_original_v7_path
            ),
    },

    "benchmark": {

        "V1":
            "GOOD REFERENCE",

        "V2":
            "FAILED LONG PROMPT",

        "V3":
            "GOOD GENERATIVE WHITE/GREEN TEST",

        "V4":
            "DIAGNOSTIC CLASS TEST",

        "V5":
            "HYBRID FULL-SURFACE TEST",

        "V6":
            "WALL MASK REFINEMENT TEST",

        "V7":
            "V1 + ONLY EXACT GREEN FLOOR",
    },
}


report_path = (
    OUT
    / "00_stage01z7_report.json"
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
# 26. FINAL STATUS
# ============================================================

print()
print("=" * 100)
print("STAGE01Z7 COMPLETE")
print("=" * 100)

print()
print(
    "NO QWEN GENERATION REQUIRED"
)

print()
print(
    "V7 RESULT:"
)

print(
    v7_path
)

print()
print(
    "ORIGINAL / V1 / V7:"
)

print(
    comparison_path
)

print()
print(
    "REPORT:"
)

print(
    report_path
)

print()
print(
    "FLOOR EXACT #00C800:",
    round(
        floor_exact_ratio
        * 100,
        6
    ),
    "%"
)

print(
    "MAX PROP RGB ERROR:",
    max_prop_rgb_error
)

print(
    "MAX NON-FLOOR CHANGE VS V1:",
    max_nonfloor_v1_error
)


# ============================================================
# 27. DISPLAY
# ============================================================

print()
print(
    "FLOOR / PROP MASK OVERLAY"
)

display(
    Image.open(
        overlay_path
    )
)

print()
print(
    "ORIGINAL / V1 / V7"
)

display(
    comparison
)

print()
print(
    "V1 VS V7"
)

display(
    compare_v1_v7
)
