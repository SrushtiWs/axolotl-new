
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

B2_STRENGTH = (
    PROD
    / "stage07_empty_room"
    / "07b2_conservative_floor_shadow"
    / "03_floor_shadow_strength.png"
)

B2_CANDIDATE = (
    PROD
    / "stage07_empty_room"
    / "07b2_conservative_floor_shadow"
    / "02_clean_shadow_candidate.png"
)

OUT = (
    PROD
    / "stage07_empty_room"
    / "07b4_conservative_shadow_refinement"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

# Very small blur.
# We only want anti-aliased / slightly feathered boundaries.
EDGE_SIGMA = 1.35

# Preserve most of 07B2's original darkness.
STRENGTH_SCALE = 0.92

# Prevent excessive darkness on new tiles.
MAX_STRENGTH = 0.38

# Allow feathering only a few pixels outside detected shape.
FEATHER_EXPANSION = 4

# Remove negligible haze.
MIN_ACTIVE_STRENGTH = 0.004


# ============================================================
# VALIDATE
# ============================================================

for path in [
    STAGE05F,
    STAGE07A,
    B2_STRENGTH,
    B2_CANDIDATE,
]:

    if not path.exists():
        raise FileNotFoundError(path)


# ============================================================
# LOAD
# ============================================================

stage05f = Image.open(
    STAGE05F
).convert("RGB")

stage07 = Image.open(
    STAGE07A
).convert("RGB")


if stage05f.size != stage07.size:
    raise RuntimeError(
        "Stage05F / Stage07A size mismatch."
    )


W, H = stage07.size


strength_b2 = (
    np.asarray(
        Image.open(
            B2_STRENGTH
        ).convert("L")
    ).astype(np.float32)
    /
    255.0
)


candidate = (
    np.asarray(
        Image.open(
            B2_CANDIDATE
        ).convert("L")
    )
    >
    127
)


# ============================================================
# FEATHER ALLOWANCE REGION
#
# Tiny expansion only.
# This prevents Gaussian blur from spreading across the floor.
# ============================================================

kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (
        FEATHER_EXPANSION * 2 + 1,
        FEATHER_EXPANSION * 2 + 1
    )
)


feather_region = cv2.dilate(
    candidate.astype(np.uint8),
    kernel,
    iterations=1
).astype(bool)


# ============================================================
# SLIGHT EDGE FEATHER
# ============================================================

refined = cv2.GaussianBlur(
    strength_b2,
    (0, 0),
    sigmaX=EDGE_SIGMA,
    sigmaY=EDGE_SIGMA
)


# Keep shape local.
refined *= (
    feather_region.astype(
        np.float32
    )
)


# Preserve most of original intensity.
refined *= (
    STRENGTH_SCALE
)


refined = np.clip(
    refined,
    0.0,
    MAX_STRENGTH
)


refined[
    refined < MIN_ACTIVE_STRENGTH
] = 0.0


# ============================================================
# MULTIPLIER
# ============================================================

multiplier = (
    1.0
    -
    refined
)


# ============================================================
# APPLY TO STAGE07
# ============================================================

stage07_np = np.asarray(
    stage07
).astype(np.float32)


shadowed = (
    stage07_np
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
    / "01_final_shadow_strength_candidate.png"
)


Image.fromarray(
    (
        refined
        *
        255
    ).astype(np.uint8)
).save(
    STRENGTH_PATH
)


MULTIPLIER_PATH = (
    OUT
    / "02_final_shadow_multiplier_candidate.png"
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


SHADOWED_PATH = (
    OUT
    / "03_stage07_with_07b4_shadows.png"
)


Image.fromarray(
    shadowed
).save(
    SHADOWED_PATH
)


# ============================================================
# 5-PANEL AUDIT
# ============================================================

fig, axes = plt.subplots(
    1,
    5,
    figsize=(22, 7)
)


axes[0].imshow(
    stage05f
)

axes[0].set_title(
    "1. Stage05F\nOriginal Shadow Reference"
)

axes[0].axis(
    "off"
)


axes[1].imshow(
    stage07
)

axes[1].set_title(
    "2. Stage07A\nNo Shadows"
)

axes[1].axis(
    "off"
)


axes[2].imshow(
    strength_b2,
    cmap="gray",
    vmin=0,
    vmax=0.42
)

axes[2].set_title(
    "3. 07B2\nDetected Shadow Strength"
)

axes[2].axis(
    "off"
)


axes[3].imshow(
    refined,
    cmap="gray",
    vmin=0,
    vmax=0.42
)

axes[3].set_title(
    "4. 07B4\nConservative Refinement"
)

axes[3].axis(
    "off"
)


axes[4].imshow(
    shadowed
)

axes[4].set_title(
    "5. Stage07 + 07B4 Shadows"
)

axes[4].axis(
    "off"
)


plt.tight_layout()


AUDIT_PATH = (
    OUT
    / "04_stage07b4_audit.png"
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
    refined
    >
    0.01
)


active_pixels = int(
    active.sum()
)


if active_pixels > 0:

    mean_strength = float(
        refined[
            active
        ].mean()
    )

else:

    mean_strength = 0.0


max_strength = float(
    refined.max()
)


# ============================================================
# STATE
# ============================================================

STATE = {

    "stage":
        "07B4",

    "approach":
        "CONSERVATIVE_EDGE_FEATHER_OF_07B2",

    "source_shadow_stage":
        "07B2",

    "settings": {

        "edge_sigma":
            EDGE_SIGMA,

        "strength_scale":
            STRENGTH_SCALE,

        "max_strength":
            MAX_STRENGTH,

        "feather_expansion":
            FEATHER_EXPANSION,

        "min_active_strength":
            MIN_ACTIVE_STRENGTH
    },

    "statistics": {

        "active_shadow_pixels":
            active_pixels,

        "mean_strength":
            mean_strength,

        "max_strength":
            max_strength
    },

    "outputs": {

        "shadow_strength":
            str(
                STRENGTH_PATH
            ),

        "shadow_multiplier":
            str(
                MULTIPLIER_PATH
            ),

        "shadowed_empty_room":
            str(
                SHADOWED_PATH
            ),

        "audit":
            str(
                AUDIT_PATH
            )
    },

    "status":
        "REQUIRES_FINAL_SHADOW_VISUAL_AUDIT"
}


STATE_PATH = (
    OUT
    / "00_stage07b4_result.json"
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
print("STAGE 07B4 RESULT")
print("=" * 110)

print()
print(
    "ACTIVE SHADOW PIXELS:",
    active_pixels
)

print(
    "MEAN SHADOW STRENGTH:",
    round(
        mean_strength,
        4
    )
)

print(
    "MAX SHADOW STRENGTH:",
    round(
        max_strength,
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
    "AUDIT:",
    AUDIT_PATH
)

print(
    "STATE:",
    STATE_PATH
)

print()
print(
    "NO NEW SHADOW DETECTION WAS RUN."
)
