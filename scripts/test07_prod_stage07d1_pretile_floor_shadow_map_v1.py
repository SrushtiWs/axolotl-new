
from pathlib import Path
import json

import cv2
import numpy as np
from PIL import Image
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

ORIGINAL_PATH = (
    PROD
    / "stage01_master"
    / "00_master_input.png"
)

FLOOR_MASK_PATH = (
    PROD
    / "stage08_tile_application"
    / "08b2_floor_plus_screeding_target"
    / "01_main_floor_mask.png"
)

PROP_MASK_PATH = (
    PROD
    / "stage06_prop_layer"
    / "06f4_final_six_prop_layer"
    / "01_final_six_prop_union_mask.png"
)

OUT = (
    PROD
    / "stage07_empty_room"
    / "07d1_pretile_floor_shadow_map"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD
# ============================================================

for p in [
    ORIGINAL_PATH,
    FLOOR_MASK_PATH,
    PROP_MASK_PATH,
]:
    if not p.exists():
        raise FileNotFoundError(p)


original = np.asarray(
    Image.open(ORIGINAL_PATH).convert("RGB")
).astype(np.float32)

floor = (
    np.asarray(
        Image.open(FLOOR_MASK_PATH).convert("L")
    ) > 127
)

props = (
    np.asarray(
        Image.open(PROP_MASK_PATH).convert("L")
    ) > 127
)

H, W = floor.shape


# ============================================================
# LUMINANCE
# ============================================================

luma = (
    0.2126 * original[..., 0]
    +
    0.7152 * original[..., 1]
    +
    0.0722 * original[..., 2]
)


# ============================================================
# FLOOR-SAFE MASK
#
# Remove prop RGB completely.
# ============================================================

safe_floor = (
    floor
    &
    (~props)
)


# ============================================================
# PROP-CONTACT SEARCH REGION
#
# Shadows we currently care about are spatially near
# floor-touching props.
#
# Do not treat the entire room illumination as a shadow.
# ============================================================

prop_u8 = (
    props.astype(np.uint8) * 255
)

kernel_near = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (45, 45)
)

near_props = (
    cv2.dilate(
        prop_u8,
        kernel_near
    ) > 0
)

candidate_region = (
    floor
    &
    near_props
    &
    (~props)
)


# ============================================================
# ESTIMATE LOCAL UNSHADOWED FLOOR BRIGHTNESS
#
# We estimate the underlying illumination with a large
# edge-preserving-ish local smooth field.
#
# Prop pixels are filled before smoothing so black holes
# do not generate fake dark halos.
# ============================================================

filled = luma.copy()


# Replace non-safe pixels with nearest/smooth floor estimate
# using normalized Gaussian filtering.

safe_float = safe_floor.astype(np.float32)

weighted_luma = (
    luma * safe_float
)

sigma = 18.0

num = cv2.GaussianBlur(
    weighted_luma,
    (0, 0),
    sigmaX=sigma,
    sigmaY=sigma
)

den = cv2.GaussianBlur(
    safe_float,
    (0, 0),
    sigmaX=sigma,
    sigmaY=sigma
)

base_luma = (
    num
    /
    np.maximum(
        den,
        1e-5
    )
)


# ============================================================
# RAW SHADOW RATIO
#
# observed / estimated_unshadowed
#
# < 1 = darker than local expected floor
# ============================================================

ratio = (
    luma
    /
    np.maximum(
        base_luma,
        1.0
    )
)

ratio = np.clip(
    ratio,
    0.40,
    1.20
)


# ============================================================
# DARKENING ONLY
#
# Reject brightening.
# ============================================================

shadow_strength = np.clip(
    1.0 - ratio,
    0.0,
    0.55
)


# ============================================================
# ONLY LOCAL PROP-RELATED FLOOR SHADOW CANDIDATES
# ============================================================

shadow_strength[
    ~candidate_region
] = 0.0


# ============================================================
# SUPPRESS TINY TEXTURE / TILE VARIATION
#
# Very weak darkening is more likely texture/noise.
# ============================================================

MIN_SHADOW = 0.045

shadow_strength[
    shadow_strength < MIN_SHADOW
] = 0.0


# ============================================================
# CLEAN SMALL ISLANDS
# ============================================================

binary = (
    shadow_strength > 0
).astype(np.uint8) * 255

binary = cv2.morphologyEx(
    binary,
    cv2.MORPH_OPEN,
    cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )
)

binary = cv2.morphologyEx(
    binary,
    cv2.MORPH_CLOSE,
    cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )
)

shadow_strength[
    binary == 0
] = 0.0


# ============================================================
# LIGHT SOFTENING
#
# Keep contact structure but remove pixel noise.
# ============================================================

shadow_strength = cv2.GaussianBlur(
    shadow_strength.astype(np.float32),
    (0, 0),
    sigmaX=1.3,
    sigmaY=1.3
)

shadow_strength[
    ~candidate_region
] = 0.0


# ============================================================
# CONVERT TO MULTIPLIER
#
# 1.0 = unchanged
# 0.5 = strong darkening
# ============================================================

shadow_multiplier = (
    1.0 - shadow_strength
)

shadow_multiplier = np.clip(
    shadow_multiplier,
    0.50,
    1.00
)

shadow_multiplier[
    ~floor
] = 1.0


# ============================================================
# SAVE 16-BIT MULTIPLIER
#
# Preserve precision for later rendering.
# ============================================================

mult16 = (
    shadow_multiplier
    *
    65535.0
).round().astype(np.uint16)

MULT16_PATH = (
    OUT
    / "01_floor_shadow_multiplier_16bit.png"
)

Image.fromarray(
    mult16,
    mode="I;16"
).save(
    MULT16_PATH
)


# ============================================================
# SAVE 8-BIT VISUAL MATTE
#
# black = no shadow
# white = strongest shadow
# ============================================================

matte8 = (
    np.clip(
        shadow_strength / 0.55,
        0.0,
        1.0
    )
    * 255
).astype(np.uint8)

MATTE_PATH = (
    OUT
    / "02_floor_shadow_strength_visual.png"
)

Image.fromarray(
    matte8
).save(
    MATTE_PATH
)


# ============================================================
# TEST ONLY:
# APPLY MULTIPLIER TO A NEUTRAL COLOR
#
# This verifies that the map itself contains no original
# floor RGB / beige texture.
# ============================================================

neutral = np.full(
    (H, W, 3),
    210.0,
    dtype=np.float32
)

neutral_shadowed = neutral.copy()

neutral_shadowed[floor] *= (
    shadow_multiplier[
        floor,
        None
    ]
)

neutral_shadowed = np.clip(
    neutral_shadowed,
    0,
    255
).astype(np.uint8)


# ============================================================
# ORIGINAL OVERLAY
# ============================================================

overlay = original.copy()

heat = np.zeros_like(
    overlay
)

heat[..., 0] = 255

alpha = np.clip(
    shadow_strength / 0.40,
    0.0,
    0.65
)

overlay = (
    overlay * (1.0 - alpha[..., None])
    +
    heat * alpha[..., None]
)

overlay = np.clip(
    overlay,
    0,
    255
).astype(np.uint8)


# ============================================================
# 5-PANEL AUDIT
# ============================================================

fig, axes = plt.subplots(
    1,
    5,
    figsize=(22, 7)
)


axes[0].imshow(
    original.astype(np.uint8)
)
axes[0].set_title(
    "1. Original Stage01"
)
axes[0].axis("off")


axes[1].imshow(
    candidate_region,
    cmap="gray"
)
axes[1].set_title(
    "2. Floor Shadow Search Region"
)
axes[1].axis("off")


axes[2].imshow(
    matte8,
    cmap="gray",
    vmin=0,
    vmax=255
)
axes[2].set_title(
    "3. Extracted Shadow Strength"
)
axes[2].axis("off")


axes[3].imshow(
    overlay
)
axes[3].set_title(
    "4. Shadow Location Audit"
)
axes[3].axis("off")


axes[4].imshow(
    neutral_shadowed
)
axes[4].set_title(
    "5. Applied to Neutral Surface\n"
    "NO ORIGINAL FLOOR RGB"
)
axes[4].axis("off")


plt.tight_layout()


AUDIT_PATH = (
    OUT
    / "03_stage07d1_pretile_shadow_audit.png"
)

plt.savefig(
    AUDIT_PATH,
    dpi=150,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# STATISTICS
# ============================================================

active = (
    shadow_strength > 0.01
)

active_pixels = int(
    active.sum()
)

candidate_pixels = int(
    candidate_region.sum()
)

if active_pixels > 0:

    mean_multiplier = float(
        shadow_multiplier[
            active
        ].mean()
    )

    strongest_multiplier = float(
        shadow_multiplier[
            active
        ].min()
    )

else:

    mean_multiplier = 1.0
    strongest_multiplier = 1.0


# ============================================================
# STATE
# ============================================================

STATE = {

    "stage":
        "07D1",

    "architecture":
        "PRE_TILE_SHADOW_EXTRACTION",

    "input_rgb":
        str(ORIGINAL_PATH),

    "floor_mask":
        str(FLOOR_MASK_PATH),

    "prop_mask":
        str(PROP_MASK_PATH),

    "rules": {

        "runs_before_tile_selection":
            True,

        "contains_original_floor_rgb":
            False,

        "stored_as":
            "SHADOW_MULTIPLIER",

        "no_shadow_value":
            1.0
    },

    "statistics": {

        "candidate_pixels":
            candidate_pixels,

        "active_shadow_pixels":
            active_pixels,

        "mean_active_multiplier":
            mean_multiplier,

        "strongest_multiplier":
            strongest_multiplier
    },

    "outputs": {

        "multiplier_16bit":
            str(MULT16_PATH),

        "strength_visual":
            str(MATTE_PATH),

        "audit":
            str(AUDIT_PATH)
    },

    "status":
        "RND_REQUIRES_VISUAL_AUDIT"
}


STATE_PATH = (
    OUT
    / "00_stage07d1_result.json"
)

STATE_PATH.write_text(
    json.dumps(
        STATE,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


print()
print("=" * 110)
print("STAGE 07D1 RESULT")
print("=" * 110)

print()
print(
    "CANDIDATE PIXELS:",
    candidate_pixels
)

print(
    "ACTIVE SHADOW PIXELS:",
    active_pixels
)

print(
    "MEAN ACTIVE MULTIPLIER:",
    round(
        mean_multiplier,
        4
    )
)

print(
    "STRONGEST MULTIPLIER:",
    round(
        strongest_multiplier,
        4
    )
)

print()
print(
    "16-BIT SHADOW MAP:",
    MULT16_PATH
)

print(
    "AUDIT:",
    AUDIT_PATH
)

print()
print(
    "IMPORTANT:"
)

print(
    "This shadow map was generated BEFORE tile selection."
)

print(
    "No original floor RGB is stored in the multiplier."
)
