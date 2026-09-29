
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


P01_ROI_PATH = (
    PROD
    / "stage07_empty_room"
    / "07e1b_qwen_coordinate_fix"
    / "01_p01_shadow_mask.png"
)


P02_ROI_PATH = (
    PROD
    / "stage07_empty_room"
    / "07e1b_qwen_coordinate_fix"
    / "02_p02_shadow_mask.png"
)


OUT = (
    PROD
    / "stage07_empty_room"
    / "07e3_inpainted_shadow_multiplier"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# VALIDATE
# ============================================================

for p in [
    ORIGINAL_PATH,
    FLOOR_MASK_PATH,
    P01_ROI_PATH,
    P02_ROI_PATH,
]:

    if not p.exists():
        raise FileNotFoundError(p)


# ============================================================
# LOAD
# ============================================================

rgb = np.asarray(
    Image.open(
        ORIGINAL_PATH
    ).convert("RGB")
).astype(
    np.float32
)


floor = (
    np.asarray(
        Image.open(
            FLOOR_MASK_PATH
        ).convert("L")
    ) > 127
)


p01_roi = (
    np.asarray(
        Image.open(
            P01_ROI_PATH
        ).convert("L")
    ) > 127
)


p02_roi = (
    np.asarray(
        Image.open(
            P02_ROI_PATH
        ).convert("L")
    ) > 127
)


H, W = floor.shape


# ============================================================
# LUMINANCE
# ============================================================

def luminance(x):

    return (
        0.2126 * x[..., 0]
        +
        0.7152 * x[..., 1]
        +
        0.0722 * x[..., 2]
    )


luma = luminance(
    rgb
).astype(
    np.float32
)


# ============================================================
# LOW-FREQUENCY OBSERVED ILLUMINATION
#
# Suppress high-frequency floor material pattern first.
#
# This is important:
# we want lighting/shadow, not wood/tile texture.
# ============================================================

observed_low = cv2.GaussianBlur(
    luma,
    (0, 0),
    sigmaX=5.0,
    sigmaY=5.0
)


# ============================================================
# BUILD INPAINT MASK
#
# Expand the Qwen ROI slightly so shadow-edge pixels do not
# contaminate our reconstruction of unshadowed illumination.
# ============================================================

def expanded_roi(
    roi,
    radius
):

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            radius * 2 + 1,
            radius * 2 + 1
        )
    )


    return (
        cv2.dilate(
            roi.astype(
                np.uint8
            ) * 255,
            kernel
        ) > 0
    )


p01_hole = expanded_roi(
    p01_roi,
    radius=6
)


p02_hole = expanded_roi(
    p02_roi,
    radius=5
)


# Only floor is relevant.
p01_hole &= floor
p02_hole &= floor


# ============================================================
# INPAINT UN-SHADOWED LOW-FREQUENCY LUMINANCE
#
# OpenCV inpaint expects uint8.
# ============================================================

low_u8 = np.clip(
    observed_low,
    0,
    255
).astype(
    np.uint8
)


def estimate_unshadowed(
    hole_mask,
    radius
):

    hole_u8 = (
        hole_mask.astype(
            np.uint8
        ) * 255
    )


    repaired = cv2.inpaint(
        low_u8,
        hole_u8,
        float(radius),
        cv2.INPAINT_TELEA
    ).astype(
        np.float32
    )


    # Smooth reconstructed illumination slightly.
    repaired = cv2.GaussianBlur(
        repaired,
        (0, 0),
        sigmaX=3.0,
        sigmaY=3.0
    )


    return repaired


p01_expected = estimate_unshadowed(
    p01_hole,
    radius=9
)


p02_expected = estimate_unshadowed(
    p02_hole,
    radius=8
)


# ============================================================
# SHADOW ATTENUATION
#
# observed / expected
#
# A true shadow should be below 1.0.
# ============================================================

def estimate_strength(
    roi,
    observed,
    expected,
    min_darkening,
    max_strength
):

    ratio = (
        observed
        /
        np.maximum(
            expected,
            1.0
        )
    )


    ratio = np.clip(
        ratio,
        0.45,
        1.10
    )


    darkening = np.clip(
        1.0
        -
        ratio,
        0.0,
        max_strength
    )


    darkening[
        darkening
        <
        min_darkening
    ] = 0.0


    # --------------------------------------------------------
    # Soft semantic support
    #
    # Qwen polygon is coarse. We do not want hard polygon edges.
    # --------------------------------------------------------

    roi_u8 = (
        roi.astype(
            np.uint8
        ) * 255
    )


    distance = cv2.distanceTransform(
        roi_u8,
        cv2.DIST_L2,
        5
    )


    semantic_weight = np.clip(
        distance / 7.0,
        0.0,
        1.0
    )


    semantic_weight = (
        0.30
        +
        0.70
        *
        semantic_weight
    )


    semantic_weight[
        ~roi
    ] = 0.0


    strength = (
        darkening
        *
        semantic_weight
    )


    # --------------------------------------------------------
    # Feather a little outside coarse semantic polygon
    # --------------------------------------------------------

    strength = cv2.GaussianBlur(
        strength.astype(
            np.float32
        ),
        (0, 0),
        sigmaX=2.7,
        sigmaY=2.7
    )


    support = expanded_roi(
        roi,
        radius=5
    )


    strength[
        ~support
    ] = 0.0


    strength[
        ~floor
    ] = 0.0


    return np.clip(
        strength,
        0.0,
        max_strength
    )


p01_strength = estimate_strength(
    p01_roi,
    observed_low,
    p01_expected,
    min_darkening=0.018,
    max_strength=0.40
)


p02_strength = estimate_strength(
    p02_roi,
    observed_low,
    p02_expected,
    min_darkening=0.020,
    max_strength=0.45
)


# ============================================================
# COMBINE
# ============================================================

combined_strength = np.maximum(
    p01_strength,
    p02_strength
)


# ============================================================
# FINAL REUSABLE MULTIPLIER
#
# 1.0 = no shadow
# lower values = darker
# ============================================================

multiplier = np.clip(
    1.0
    -
    combined_strength,
    0.55,
    1.00
)


multiplier[
    ~floor
] = 1.0


# ============================================================
# SAVE 16-BIT PRODUCTION-CANDIDATE MULTIPLIER
# ============================================================

MULT_PATH = (
    OUT
    / "01_floor_shadow_multiplier_16bit.png"
)


Image.fromarray(
    (
        multiplier
        *
        65535.0
    ).round().astype(
        np.uint16
    ),
    mode="I;16"
).save(
    MULT_PATH
)


# ============================================================
# VISUAL HELPERS
# ============================================================

def strength8(x):

    return (
        np.clip(
            x / 0.45,
            0.0,
            1.0
        )
        *
        255
    ).astype(
        np.uint8
    )


# ============================================================
# SAVE INDIVIDUAL
# ============================================================

P01_PATH = (
    OUT
    / "02_p01_vanity_strength.png"
)

P02_PATH = (
    OUT
    / "03_p02_toilet_strength.png"
)

COMBINED_PATH = (
    OUT
    / "04_combined_shadow_strength.png"
)


Image.fromarray(
    strength8(
        p01_strength
    )
).save(
    P01_PATH
)


Image.fromarray(
    strength8(
        p02_strength
    )
).save(
    P02_PATH
)


Image.fromarray(
    strength8(
        combined_strength
    )
).save(
    COMBINED_PATH
)


# ============================================================
# RECONSTRUCTION DIAGNOSTIC
# ============================================================

expected_vis = observed_low.copy()


expected_vis[
    p01_roi
] = p01_expected[
    p01_roi
]


expected_vis[
    p02_roi
] = p02_expected[
    p02_roi
]


# ============================================================
# LOCATION AUDIT
# ============================================================

overlay = rgb.copy()


p01_alpha = np.clip(
    p01_strength / 0.30,
    0.0,
    0.65
)


overlay[
    ...
] = (
    overlay
    *
    (
        1.0
        -
        p01_alpha[..., None]
    )
    +
    np.array(
        [255, 0, 0],
        dtype=np.float32
    )
    *
    p01_alpha[..., None]
)


p02_alpha = np.clip(
    p02_strength / 0.30,
    0.0,
    0.65
)


overlay = (
    overlay
    *
    (
        1.0
        -
        p02_alpha[..., None]
    )
    +
    np.array(
        [0, 255, 255],
        dtype=np.float32
    )
    *
    p02_alpha[..., None]
)


overlay = np.clip(
    overlay,
    0,
    255
).astype(
    np.uint8
)


# ============================================================
# MATERIAL-INDEPENDENCE TEST
# ============================================================

neutral = np.full(
    (
        H,
        W,
        3
    ),
    210.0,
    dtype=np.float32
)


yy, xx = np.indices(
    (
        H,
        W
    )
)


pattern = np.zeros(
    (
        H,
        W,
        3
    ),
    dtype=np.float32
)


pattern[..., 0] = (
    80
    +
    55
    *
    (
        np.sin(
            xx / 13.0
        )
        +
        1.0
    )
    /
    2.0
)


pattern[..., 1] = (
    140
    +
    60
    *
    (
        np.sin(
            (
                xx + yy
            )
            /
            19.0
        )
        +
        1.0
    )
    /
    2.0
)


pattern[..., 2] = (
    190
    +
    45
    *
    (
        np.cos(
            yy / 15.0
        )
        +
        1.0
    )
    /
    2.0
)


neutral_result = neutral.copy()
pattern_result = pattern.copy()


neutral_result[
    floor
] *= multiplier[
    floor,
    None
]


pattern_result[
    floor
] *= multiplier[
    floor,
    None
]


neutral_result = np.clip(
    neutral_result,
    0,
    255
).astype(
    np.uint8
)


pattern_result = np.clip(
    pattern_result,
    0,
    255
).astype(
    np.uint8
)


# ============================================================
# 10-PANEL AUDIT
# ============================================================

fig, axes = plt.subplots(
    1,
    10,
    figsize=(
        42,
        7
    )
)


axes[0].imshow(
    rgb.astype(
        np.uint8
    )
)

axes[0].set_title(
    "1. Original"
)

axes[0].axis("off")


axes[1].imshow(
    observed_low,
    cmap="gray"
)

axes[1].set_title(
    "2. Low-Frequency\nObserved Luma"
)

axes[1].axis("off")


axes[2].imshow(
    expected_vis,
    cmap="gray"
)

axes[2].set_title(
    "3. Estimated\nUnshadowed Luma"
)

axes[2].axis("off")


axes[3].imshow(
    p01_roi,
    cmap="gray"
)

axes[3].set_title(
    "4. Qwen P01 ROI"
)

axes[3].axis("off")


axes[4].imshow(
    strength8(
        p01_strength
    ),
    cmap="gray",
    vmin=0,
    vmax=255
)

axes[4].set_title(
    "5. P01 Strength"
)

axes[4].axis("off")


axes[5].imshow(
    p02_roi,
    cmap="gray"
)

axes[5].set_title(
    "6. Qwen P02 ROI"
)

axes[5].axis("off")


axes[6].imshow(
    strength8(
        p02_strength
    ),
    cmap="gray",
    vmin=0,
    vmax=255
)

axes[6].set_title(
    "7. P02 Strength"
)

axes[6].axis("off")


axes[7].imshow(
    overlay
)

axes[7].set_title(
    "8. Strength Location"
)

axes[7].axis("off")


axes[8].imshow(
    neutral_result
)

axes[8].set_title(
    "9. Neutral Transfer"
)

axes[8].axis("off")


axes[9].imshow(
    pattern_result
)

axes[9].set_title(
    "10. Pattern Transfer"
)

axes[9].axis("off")


plt.tight_layout()


AUDIT_PATH = (
    OUT
    / "05_stage07e3_inpainted_shadow_audit.png"
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

p01_active = (
    p01_strength
    >
    0.01
)


p02_active = (
    p02_strength
    >
    0.01
)


combined_active = (
    combined_strength
    >
    0.01
)


print()
print("=" * 110)
print("STAGE 07E3 RESULT")
print("=" * 110)

print()

print(
    "P01 ACTIVE:",
    int(
        p01_active.sum()
    )
)

print(
    "P02 ACTIVE:",
    int(
        p02_active.sum()
    )
)

print(
    "COMBINED ACTIVE:",
    int(
        combined_active.sum()
    )
)


if combined_active.any():

    print(
        "MEAN MULTIPLIER:",
        round(
            float(
                multiplier[
                    combined_active
                ].mean()
            ),
            4
        )
    )


    print(
        "STRONGEST MULTIPLIER:",
        round(
            float(
                multiplier[
                    combined_active
                ].min()
            ),
            4
        )
    )


# ============================================================
# STATE
# ============================================================

STATE = {

    "stage":
        "07E3",

    "architecture":
        "PRE_TILE_SEMANTIC_GUIDED_INPAINTED_SHADOW_MULTIPLIER",

    "semantic_source":
        "07E1B_QWEN",

    "strength_method": [
        "low-frequency luminance",
        "semantic ROI expansion",
        "local illumination inpainting",
        "observed/expected attenuation",
        "soft semantic confidence",
        "Gaussian feathering",
    ],

    "runs_before_tile_selection":
        True,

    "model_run":
        False,

    "contains_original_surface_rgb":
        False,

    "runtime_after_tile_selection":
        "multiply_selected_surface_RGB_by_precomputed_map",

    "outputs": {

        "multiplier":
            str(
                MULT_PATH
            ),

        "p01":
            str(
                P01_PATH
            ),

        "p02":
            str(
                P02_PATH
            ),

        "combined":
            str(
                COMBINED_PATH
            ),

        "audit":
            str(
                AUDIT_PATH
            )
    },

    "status":
        "RND_REQUIRES_VISUAL_AUDIT"
}


STATE_PATH = (
    OUT
    / "00_stage07e3_result.json"
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

print(
    "MULTIPLIER:",
    MULT_PATH
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
    "NO MODEL WAS RUN."
)

print(
    "NO TILE WAS SELECTED."
)
