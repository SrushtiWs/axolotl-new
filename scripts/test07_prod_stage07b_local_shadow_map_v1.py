
from pathlib import Path
import json

import numpy as np
from PIL import Image

import cv2
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE = Path("/workspace/axolotl")

PROD = (
    BASE
    / "test07"
    / "production_pipeline"
)

STAGE05F = (
    PROD
    / "stage05_clean_room_with_props"
    / "12_final_canonical_prop_detection_master.png"
)

STAGE07A = (
    PROD
    / "stage07_empty_room"
    / "07a_qwen_empty_room"
    / "00_stage07_empty_room_candidate.png"
)

STAGE06_MASK = (
    PROD
    / "stage06_prop_layer"
    / "06f4_final_six_prop_layer"
    / "01_final_six_prop_union_mask.png"
)

# Architectural surface priors
WALL_MASK = (
    PROD
    / "stage02_structure"
    / "16_wall_majority_2of3.png"
)

FLOOR_MASK = (
    PROD
    / "stage02_structure"
    / "17_floor_majority_2of3.png"
)

CEILING_MASK = (
    PROD
    / "stage02_structure"
    / "18_ceiling_majority_2of3.png"
)

OUT = (
    PROD
    / "stage07_empty_room"
    / "07b_shadow_map"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

# Search distance around props.
DILATION_RADIUS = 28

# Ignore tiny brightness changes from Qwen reconstruction.
MIN_LUMA_DROP = 5.0

# Stage05F must be at least this much darker than Stage07.
MIN_DARK_RATIO = 0.97

# Avoid unrealistically strong shadow transfer.
MIN_MULTIPLIER = 0.52

# Smooth transition.
GAUSSIAN_SIGMA = 2.2


# ============================================================
# VALIDATE
# ============================================================

for p in [
    STAGE05F,
    STAGE07A,
    STAGE06_MASK,
    WALL_MASK,
    FLOOR_MASK,
    CEILING_MASK,
]:
    if not p.exists():
        raise FileNotFoundError(p)


# ============================================================
# LOAD
# ============================================================

img_with_props = Image.open(
    STAGE05F
).convert("RGB")

img_empty = Image.open(
    STAGE07A
).convert("RGB")

if img_with_props.size != img_empty.size:
    raise RuntimeError(
        f"Image size mismatch: "
        f"{img_with_props.size} vs {img_empty.size}"
    )

W, H = img_with_props.size


a = np.asarray(
    img_with_props
).astype(np.float32)

b = np.asarray(
    img_empty
).astype(np.float32)


prop_mask = (
    np.asarray(
        Image.open(
            STAGE06_MASK
        ).convert("L")
    )
    > 127
)


# ============================================================
# ARCHITECTURAL SURFACE MASK
# ============================================================

def load_bool(path):
    return (
        np.asarray(
            Image.open(path).convert("L")
        )
        > 127
    )


wall = load_bool(
    WALL_MASK
)

floor = load_bool(
    FLOOR_MASK
)

ceiling = load_bool(
    CEILING_MASK
)


surface_mask = (
    wall
    |
    floor
    |
    ceiling
)


# ============================================================
# LOCAL SEARCH REGION
#
# Only around physical props.
# ============================================================

kernel_size = (
    DILATION_RADIUS * 2
    +
    1
)

kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (
        kernel_size,
        kernel_size
    )
)


dilated = cv2.dilate(
    prop_mask.astype(np.uint8),
    kernel,
    iterations=1
).astype(bool)


shadow_search_region = (
    dilated
    &
    ~prop_mask
    &
    surface_mask
)


# ============================================================
# LUMINANCE
# ============================================================

def luminance(rgb):

    return (
        0.2126 * rgb[..., 0]
        +
        0.7152 * rgb[..., 1]
        +
        0.0722 * rgb[..., 2]
    )


lum_props = luminance(
    a
)

lum_empty = luminance(
    b
)


# ============================================================
# DARKENING TEST
#
# A visible cast shadow must satisfy:
#
# Stage05F is darker than Stage07A.
# ============================================================

luma_drop = (
    lum_empty
    -
    lum_props
)


safe_empty = np.maximum(
    lum_empty,
    1.0
)


ratio = (
    lum_props
    /
    safe_empty
)


shadow_candidate = (
    shadow_search_region
    &
    (luma_drop >= MIN_LUMA_DROP)
    &
    (ratio <= MIN_DARK_RATIO)
)


# ============================================================
# RAW SHADOW MULTIPLIER
#
# Example:
# ratio 0.70 means new tile should retain 70% brightness.
# ============================================================

raw_multiplier = np.ones(
    (
        H,
        W
    ),
    dtype=np.float32
)


raw_multiplier[
    shadow_candidate
] = np.clip(
    ratio[
        shadow_candidate
    ],
    MIN_MULTIPLIER,
    1.0
)


# ============================================================
# CONVERT TO SHADOW STRENGTH
#
# 0 = no shadow
# 1 = strongest allowed shadow
# ============================================================

shadow_strength = (
    1.0
    -
    raw_multiplier
)


# Smooth only the shadow field.
shadow_strength = cv2.GaussianBlur(
    shadow_strength,
    (0, 0),
    sigmaX=GAUSSIAN_SIGMA,
    sigmaY=GAUSSIAN_SIGMA
)


# Keep shadow strictly local to the search region.
shadow_strength *= (
    shadow_search_region.astype(
        np.float32
    )
)


# Reconstruct final smooth multiplier.
shadow_multiplier = (
    1.0
    -
    shadow_strength
)


shadow_multiplier = np.clip(
    shadow_multiplier,
    MIN_MULTIPLIER,
    1.0
)


# ============================================================
# SAVE SEARCH REGION
# ============================================================

SEARCH_PATH = (
    OUT
    / "01_local_shadow_search_region.png"
)


Image.fromarray(
    shadow_search_region.astype(np.uint8)
    *
    255
).save(
    SEARCH_PATH
)


# ============================================================
# SAVE RAW SHADOW CANDIDATE
# ============================================================

CANDIDATE_PATH = (
    OUT
    / "02_raw_shadow_candidate.png"
)


Image.fromarray(
    shadow_candidate.astype(np.uint8)
    *
    255
).save(
    CANDIDATE_PATH
)


# ============================================================
# SAVE SHADOW STRENGTH
#
# black = no shadow
# white = strong shadow
# ============================================================

STRENGTH_PATH = (
    OUT
    / "03_shadow_strength.png"
)


Image.fromarray(
    (
        shadow_strength
        *
        255
    ).astype(np.uint8)
).save(
    STRENGTH_PATH
)


# ============================================================
# SAVE MULTIPLIER
#
# white = unchanged
# dark = shadow
# ============================================================

MULTIPLIER_PATH = (
    OUT
    / "04_shadow_multiplier.png"
)


Image.fromarray(
    (
        shadow_multiplier
        *
        255
    ).astype(np.uint8)
).save(
    MULTIPLIER_PATH
)


# ============================================================
# TEST APPLICATION
#
# Apply derived shadow back onto Stage07 empty room.
#
# This is only a diagnostic:
# Stage07 * shadow_multiplier should roughly recreate
# visible cast shadows without restoring the objects.
# ============================================================

test_shadowed = (
    b
    *
    shadow_multiplier[
        ...,
        None
    ]
)


test_shadowed = np.clip(
    test_shadowed,
    0,
    255
).astype(np.uint8)


TEST_PATH = (
    OUT
    / "05_stage07_with_reapplied_shadow_test.png"
)


Image.fromarray(
    test_shadowed
).save(
    TEST_PATH
)


# ============================================================
# 5-PANEL AUDIT
# ============================================================

fig, axes = plt.subplots(
    1,
    5,
    figsize=(
        22,
        7
    )
)


axes[0].imshow(
    img_with_props
)

axes[0].set_title(
    "1. Stage05F\nProps + Original Shadows"
)

axes[0].axis(
    "off"
)


axes[1].imshow(
    img_empty
)

axes[1].set_title(
    "2. Stage07A\nEmpty / No Prop Shadows"
)

axes[1].axis(
    "off"
)


axes[2].imshow(
    shadow_search_region,
    cmap="gray",
    vmin=0,
    vmax=1
)

axes[2].set_title(
    "3. Local Shadow Search Region"
)

axes[2].axis(
    "off"
)


axes[3].imshow(
    shadow_strength,
    cmap="gray",
    vmin=0,
    vmax=0.5
)

axes[3].set_title(
    "4. Derived Shadow Strength"
)

axes[3].axis(
    "off"
)


axes[4].imshow(
    test_shadowed
)

axes[4].set_title(
    "5. Stage07 + Reapplied Shadows"
)

axes[4].axis(
    "off"
)


plt.tight_layout()


AUDIT_PATH = (
    OUT
    / "06_stage07b_shadow_audit.png"
)


plt.savefig(
    AUDIT_PATH,
    dpi=150,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# STATS
# ============================================================

search_pixels = int(
    shadow_search_region.sum()
)

candidate_pixels = int(
    shadow_candidate.sum()
)

active_soft_pixels = int(
    (
        shadow_strength
        >
        0.01
    ).sum()
)


if active_soft_pixels > 0:

    mean_multiplier_active = float(
        shadow_multiplier[
            shadow_strength
            >
            0.01
        ].mean()
    )

else:

    mean_multiplier_active = 1.0


# ============================================================
# STATE
# ============================================================

STATE = {

    "stage":
        "07B",

    "approach":
        "LOCAL_LUMINANCE_RATIO_SHADOW_TRANSFER",

    "inputs": {

        "stage05f":
            str(STAGE05F),

        "stage07a":
            str(STAGE07A),

        "stage06_prop_mask":
            str(STAGE06_MASK),

        "wall_mask":
            str(WALL_MASK),

        "floor_mask":
            str(FLOOR_MASK),

        "ceiling_mask":
            str(CEILING_MASK),
    },

    "image_size": [
        W,
        H
    ],

    "settings": {

        "dilation_radius":
            DILATION_RADIUS,

        "min_luma_drop":
            MIN_LUMA_DROP,

        "min_dark_ratio":
            MIN_DARK_RATIO,

        "min_multiplier":
            MIN_MULTIPLIER,

        "gaussian_sigma":
            GAUSSIAN_SIGMA
    },

    "statistics": {

        "search_pixels":
            search_pixels,

        "raw_shadow_candidate_pixels":
            candidate_pixels,

        "active_soft_shadow_pixels":
            active_soft_pixels,

        "mean_multiplier_active":
            mean_multiplier_active
    },

    "outputs": {

        "search_region":
            str(SEARCH_PATH),

        "raw_shadow_candidate":
            str(CANDIDATE_PATH),

        "shadow_strength":
            str(STRENGTH_PATH),

        "shadow_multiplier":
            str(MULTIPLIER_PATH),

        "reapplied_shadow_test":
            str(TEST_PATH),

        "audit":
            str(AUDIT_PATH)
    },

    "important_rule":
        (
            "Shadow is stored as illumination strength, "
            "not copied RGB. This allows the future tile "
            "texture to remain visible underneath the shadow."
        ),

    "status":
        "REQUIRES_VISUAL_SHADOW_AUDIT"
}


STATE_PATH = (
    OUT
    / "00_stage07b_result.json"
)


STATE_PATH.write_text(
    json.dumps(
        STATE,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# ============================================================
# PRINT
# ============================================================

print()
print("=" * 110)
print("STAGE 07B RESULT")
print("=" * 110)

print()

print(
    "SEARCH PIXELS:",
    search_pixels
)

print(
    "RAW SHADOW CANDIDATE PIXELS:",
    candidate_pixels
)

print(
    "ACTIVE SOFT SHADOW PIXELS:",
    active_soft_pixels
)

print(
    "MEAN ACTIVE MULTIPLIER:",
    round(
        mean_multiplier_active,
        4
    )
)

print()
print(
    "SHADOW STRENGTH:",
    STRENGTH_PATH
)

print(
    "SHADOW MULTIPLIER:",
    MULTIPLIER_PATH
)

print(
    "REAPPLIED TEST:",
    TEST_PATH
)

print(
    "AUDIT:",
    AUDIT_PATH
)

print(
    "STATE:",
    STATE_PATH
)

print()
print(
    "NO AI MODEL WAS RUN IN 07B."
)
