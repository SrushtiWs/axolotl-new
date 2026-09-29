
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

STAGE07A = (
    PROD
    / "stage07_empty_room"
    / "07a_qwen_empty_room"
    / "00_stage07_empty_room_candidate.png"
)

B2_CANDIDATE = (
    PROD
    / "stage07_empty_room"
    / "07b2_conservative_floor_shadow"
    / "02_clean_shadow_candidate.png"
)

B2_STRENGTH = (
    PROD
    / "stage07_empty_room"
    / "07b2_conservative_floor_shadow"
    / "03_floor_shadow_strength.png"
)

OUT = (
    PROD
    / "stage07_empty_room"
    / "07b3_soft_shadow_refinement"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

# Reduce overall shadow darkness.
GLOBAL_STRENGTH_SCALE = 0.72

# Maximum darkness after refinement.
MAX_SHADOW_STRENGTH = 0.30

# Boundary feather.
BOUNDARY_BLUR_SIGMA = 5.0

# Secondary wide falloff.
WIDE_BLUR_SIGMA = 10.0

# Mix local core and wide soft shadow.
CORE_WEIGHT = 0.72
WIDE_WEIGHT = 0.28


# ============================================================
# LOAD
# ============================================================

for p in [
    STAGE07A,
    B2_CANDIDATE,
    B2_STRENGTH,
]:
    if not p.exists():
        raise FileNotFoundError(p)


stage07 = Image.open(
    STAGE07A
).convert("RGB")

W, H = stage07.size


candidate = (
    np.asarray(
        Image.open(
            B2_CANDIDATE
        ).convert("L")
    ).astype(np.float32)
    /
    255.0
)


strength_b2 = (
    np.asarray(
        Image.open(
            B2_STRENGTH
        ).convert("L")
    ).astype(np.float32)
    /
    255.0
)


# ============================================================
# LIMIT TO ACCEPTED B2 SHAPE
# ============================================================

base_strength = (
    strength_b2
    *
    (candidate > 0.5).astype(np.float32)
)


# ============================================================
# CORE SOFTENING
# ============================================================

core = cv2.GaussianBlur(
    base_strength,
    (0, 0),
    sigmaX=BOUNDARY_BLUR_SIGMA,
    sigmaY=BOUNDARY_BLUR_SIGMA
)


# ============================================================
# WIDE CAST-SHADOW FALLOFF
# ============================================================

wide = cv2.GaussianBlur(
    base_strength,
    (0, 0),
    sigmaX=WIDE_BLUR_SIGMA,
    sigmaY=WIDE_BLUR_SIGMA
)


# ============================================================
# COMBINE
# ============================================================

refined_strength = (
    CORE_WEIGHT * core
    +
    WIDE_WEIGHT * wide
)


refined_strength *= (
    GLOBAL_STRENGTH_SCALE
)


refined_strength = np.clip(
    refined_strength,
    0.0,
    MAX_SHADOW_STRENGTH
)


# Remove tiny numerical haze
refined_strength[
    refined_strength < 0.006
] = 0.0


# ============================================================
# SHADOW MULTIPLIER
# ============================================================

multiplier = (
    1.0
    -
    refined_strength
)


# ============================================================
# APPLY TO STAGE07
# ============================================================

rgb = np.asarray(
    stage07
).astype(np.float32)


shadowed = (
    rgb
    *
    multiplier[
        ...,
        None
    ]
)


shadowed = np.clip(
    shadowed,
    0,
    255
).astype(np.uint8)


# ============================================================
# SAVE
# ============================================================

STRENGTH_PATH = (
    OUT
    / "01_refined_shadow_strength.png"
)


Image.fromarray(
    (
        refined_strength
        *
        255
    ).astype(np.uint8)
).save(
    STRENGTH_PATH
)


MULTIPLIER_PATH = (
    OUT
    / "02_refined_shadow_multiplier.png"
)


Image.fromarray(
    (
        multiplier
        *
        255
    ).astype(np.uint8)
).save(
    MULTIPLIER_PATH
)


RESULT_PATH = (
    OUT
    / "03_stage07_with_refined_shadows.png"
)


Image.fromarray(
    shadowed
).save(
    RESULT_PATH
)


# ============================================================
# 4-PANEL AUDIT
# ============================================================

fig, axes = plt.subplots(
    1,
    4,
    figsize=(18, 7)
)


axes[0].imshow(
    stage07
)

axes[0].set_title(
    "1. Stage07A — No Shadows"
)

axes[0].axis("off")


axes[1].imshow(
    strength_b2,
    cmap="gray",
    vmin=0,
    vmax=0.42
)

axes[1].set_title(
    "2. 07B2 Shadow Strength"
)

axes[1].axis("off")


axes[2].imshow(
    refined_strength,
    cmap="gray",
    vmin=0,
    vmax=0.30
)

axes[2].set_title(
    "3. 07B3 Refined Soft Shadow"
)

axes[2].axis("off")


axes[3].imshow(
    shadowed
)

axes[3].set_title(
    "4. Stage07 + Refined Shadows"
)

axes[3].axis("off")


plt.tight_layout()


AUDIT_PATH = (
    OUT
    / "04_stage07b3_audit.png"
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
    refined_strength
    >
    0.01
)


active_pixels = int(
    active.sum()
)


if active_pixels:

    mean_strength = float(
        refined_strength[
            active
        ].mean()
    )

    max_strength = float(
        refined_strength.max()
    )

else:

    mean_strength = 0.0
    max_strength = 0.0


# ============================================================
# STATE
# ============================================================

STATE = {

    "stage":
        "07B3",

    "approach":
        "SOFTEN_EXISTING_07B2_SHADOWS",

    "settings": {

        "global_strength_scale":
            GLOBAL_STRENGTH_SCALE,

        "max_shadow_strength":
            MAX_SHADOW_STRENGTH,

        "boundary_blur_sigma":
            BOUNDARY_BLUR_SIGMA,

        "wide_blur_sigma":
            WIDE_BLUR_SIGMA,

        "core_weight":
            CORE_WEIGHT,

        "wide_weight":
            WIDE_WEIGHT,
    },

    "statistics": {

        "active_pixels":
            active_pixels,

        "mean_strength":
            mean_strength,

        "max_strength":
            max_strength,
    },

    "outputs": {

        "strength":
            str(
                STRENGTH_PATH
            ),

        "multiplier":
            str(
                MULTIPLIER_PATH
            ),

        "shadowed_empty_room":
            str(
                RESULT_PATH
            ),

        "audit":
            str(
                AUDIT_PATH
            ),
    },

    "status":
        "REQUIRES_FINAL_SHADOW_VISUAL_AUDIT"
}


STATE_PATH = (
    OUT
    / "00_stage07b3_result.json"
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
print("STAGE 07B3 RESULT")
print("=" * 110)

print()
print(
    "ACTIVE SHADOW PIXELS:",
    active_pixels
)

print(
    "MEAN STRENGTH:",
    round(
        mean_strength,
        4
    )
)

print(
    "MAX STRENGTH:",
    round(
        max_strength,
        4
    )
)

print()
print(
    "REFINED MULTIPLIER:",
    MULTIPLIER_PATH
)

print(
    "AUDIT:",
    AUDIT_PATH
)

print(
    "STATE:",
    STATE_PATH
)
