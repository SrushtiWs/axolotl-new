
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

P01_MASK_PATH = (
    PROD
    / "stage06_prop_layer"
    / "06f3_vanity_internal_completion"
    / "01_p01_completed_vanity_mask.png"
)

P02_MASK_PATH = (
    PROD
    / "stage06_prop_layer"
    / "06d2c1_complete_main_prop_sam2_multimask_audit"
    / "objects"
    / "P02_candidate_3_mask.png"
)

OUT = (
    PROD
    / "stage07_empty_room"
    / "07d4_chromatic_shadow_decomposition"
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
    P01_MASK_PATH,
    P02_MASK_PATH,
]:
    if not p.exists():
        raise FileNotFoundError(p)


rgb = np.asarray(
    Image.open(
        ORIGINAL_PATH
    ).convert("RGB")
).astype(np.float32)


floor = (
    np.asarray(
        Image.open(
            FLOOR_MASK_PATH
        ).convert("L")
    )
    > 127
)


p01 = (
    np.asarray(
        Image.open(
            P01_MASK_PATH
        ).convert("L")
    )
    > 127
)


p02 = (
    np.asarray(
        Image.open(
            P02_MASK_PATH
        ).convert("L")
    )
    > 127
)


H, W = floor.shape


# ============================================================
# FLOOR CONTACT
# ============================================================

def contact_seed(
    prop,
    floor_mask,
    radius=5
):
    k = 2 * radius + 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (k, k)
    )

    dil = (
        cv2.dilate(
            prop.astype(np.uint8) * 255,
            kernel
        )
        > 0
    )

    return (
        dil
        &
        floor_mask
        &
        (~prop)
    )


p01_contact = contact_seed(
    p01,
    floor,
    5
)

p02_contact = contact_seed(
    p02,
    floor,
    5
)


# ============================================================
# DISTANCE-GROWN REGIONS
# ============================================================

def distance_region(
    seed,
    floor_mask,
    radius
):
    inv = (
        ~seed
    ).astype(np.uint8)

    dist = cv2.distanceTransform(
        inv,
        cv2.DIST_L2,
        5
    )

    region = (
        (dist <= radius)
        &
        floor_mask
    )

    return region, dist


p01_region, p01_dist = distance_region(
    p01_contact,
    floor,
    58
)

p02_region, p02_dist = distance_region(
    p02_contact,
    floor,
    42
)


p01_region &= ~p01
p02_region &= ~p02


# ============================================================
# SAFE FLOOR FOR REFERENCE ESTIMATION
# ============================================================

safe = (
    floor
    &
    (~p01)
    &
    (~p02)
)

safe_f = safe.astype(np.float32)


# ============================================================
# LOCAL RGB REFERENCE
#
# Estimate what the floor RGB would approximately be without
# localized darkening.
# ============================================================

SIGMA = 24.0

reference_rgb = np.zeros_like(
    rgb,
    dtype=np.float32
)


for c in range(3):

    numerator = cv2.GaussianBlur(
        rgb[..., c] * safe_f,
        (0, 0),
        sigmaX=SIGMA,
        sigmaY=SIGMA
    )

    denominator = cv2.GaussianBlur(
        safe_f,
        (0, 0),
        sigmaX=SIGMA,
        sigmaY=SIGMA
    )

    reference_rgb[..., c] = (
        numerator
        /
        np.maximum(
            denominator,
            1e-5
        )
    )


# ============================================================
# LUMINANCE DARKENING
# ============================================================

def luminance(x):

    return (
        0.2126 * x[..., 0]
        +
        0.7152 * x[..., 1]
        +
        0.0722 * x[..., 2]
    )


obs_luma = luminance(
    rgb
)

ref_luma = luminance(
    reference_rgb
)


luma_ratio = (
    obs_luma
    /
    np.maximum(
        ref_luma,
        1.0
    )
)


luma_ratio = np.clip(
    luma_ratio,
    0.40,
    1.25
)


raw_darkening = np.clip(
    1.0
    -
    luma_ratio,
    0.0,
    0.55
)


# ============================================================
# TEST 1:
# RGB MULTIPLICATIVE CONSISTENCY
#
# True approximate shadow:
#
#    observed_R/reference_R
# ~= observed_G/reference_G
# ~= observed_B/reference_B
#
# Material/color changes tend to produce unequal ratios.
# ============================================================

channel_ratio = (
    rgb
    /
    np.maximum(
        reference_rgb,
        5.0
    )
)


ratio_mean = np.mean(
    channel_ratio,
    axis=2
)


ratio_std = np.std(
    channel_ratio,
    axis=2
)


ratio_cv = (
    ratio_std
    /
    np.maximum(
        ratio_mean,
        0.05
    )
)


# Low CV is more shadow-like.
rgb_consistency = np.exp(
    -(
        ratio_cv
        /
        0.115
    ) ** 2
)


# ============================================================
# TEST 2:
# NORMALIZED RGB CHROMATICITY
#
# Shadow should generally preserve relative color better than
# a genuine material/design transition.
# ============================================================

obs_sum = np.sum(
    rgb,
    axis=2,
    keepdims=True
)


ref_sum = np.sum(
    reference_rgb,
    axis=2,
    keepdims=True
)


obs_chroma = (
    rgb
    /
    np.maximum(
        obs_sum,
        1.0
    )
)


ref_chroma = (
    reference_rgb
    /
    np.maximum(
        ref_sum,
        1.0
    )
)


chroma_distance = np.sqrt(
    np.sum(
        (
            obs_chroma
            -
            ref_chroma
        ) ** 2,
        axis=2
    )
)


chroma_confidence = np.exp(
    -(
        chroma_distance
        /
        0.055
    ) ** 2
)


# ============================================================
# COMBINED SHADOW-LIKENESS
# ============================================================

shadow_likeness = (
    rgb_consistency
    *
    chroma_confidence
)


# ============================================================
# DISTANCE CONFIDENCE
# ============================================================

def distance_confidence(
    dist,
    max_dist
):
    c = (
        1.0
        -
        dist
        /
        float(max_dist)
    )

    return np.clip(
        c,
        0.0,
        1.0
    )


p01_distance_conf = distance_confidence(
    p01_dist,
    58
)

p02_distance_conf = distance_confidence(
    p02_dist,
    42
)


# ============================================================
# EXTRACT FUNCTION
# ============================================================

def extract_shadow(
    region,
    distance_conf,
    darkening_threshold,
    likeness_threshold,
    min_component_area,
    blur_sigma
):

    candidate = (
        region
        &
        (
            raw_darkening
            >=
            darkening_threshold
        )
        &
        (
            shadow_likeness
            >=
            likeness_threshold
        )
    )


    # Strength comes from luminance darkening.
    # Confidence only controls whether we believe it.
    strength = (
        raw_darkening
        *
        (
            0.65
            +
            0.35
            *
            distance_conf
        )
        *
        shadow_likeness
    )


    strength[
        ~candidate
    ] = 0.0


    # ------------------------------------------
    # Remove isolated texture fragments
    # ------------------------------------------

    binary = (
        strength > 0
    ).astype(np.uint8)


    n, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )
    )


    clean = np.zeros_like(
        binary
    )


    for i in range(1, n):

        area = int(
            stats[
                i,
                cv2.CC_STAT_AREA
            ]
        )

        if area >= min_component_area:

            clean[
                labels == i
            ] = 1


    strength[
        clean == 0
    ] = 0.0


    # Soft shadow falloff
    strength = cv2.GaussianBlur(
        strength.astype(np.float32),
        (0, 0),
        sigmaX=blur_sigma,
        sigmaY=blur_sigma
    )


    strength[
        ~region
    ] = 0.0


    return strength


# ============================================================
# OBJECT-SPECIFIC
# ============================================================

p01_shadow = extract_shadow(
    p01_region,
    p01_distance_conf,
    darkening_threshold=0.035,
    likeness_threshold=0.38,
    min_component_area=12,
    blur_sigma=1.5
)


p02_shadow = extract_shadow(
    p02_region,
    p02_distance_conf,
    darkening_threshold=0.040,
    likeness_threshold=0.35,
    min_component_area=9,
    blur_sigma=1.3
)


# ============================================================
# COMBINE
# ============================================================

combined = np.maximum(
    p01_shadow,
    p02_shadow
)


combined = np.clip(
    combined,
    0.0,
    0.50
)


multiplier = np.clip(
    1.0
    -
    combined,
    0.50,
    1.00
)


multiplier[
    ~floor
] = 1.0


# ============================================================
# SAVE 16-BIT MULTIPLIER
# ============================================================

mult16 = (
    multiplier
    *
    65535.0
).round().astype(np.uint16)


MULT_PATH = (
    OUT
    / "01_floor_shadow_multiplier_16bit.png"
)


Image.fromarray(
    mult16,
    mode="I;16"
).save(
    MULT_PATH
)


# ============================================================
# VIS HELPERS
# ============================================================

def norm01(x, hi):

    return np.clip(
        x / hi,
        0.0,
        1.0
    )


def strength8(x):

    return (
        norm01(
            x,
            0.50
        )
        *
        255
    ).astype(np.uint8)


# ============================================================
# DIAGNOSTICS
# ============================================================

darkening_vis = norm01(
    raw_darkening,
    0.40
)


likeness_vis = np.clip(
    shadow_likeness,
    0.0,
    1.0
)


# ============================================================
# LOCATION OVERLAY
# ============================================================

overlay = rgb.copy()


a1 = np.clip(
    p01_shadow / 0.35,
    0.0,
    0.65
)


red = np.zeros_like(
    overlay
)

red[..., 0] = 255


overlay = (
    overlay
    *
    (
        1.0
        -
        a1[..., None]
    )
    +
    red
    *
    a1[..., None]
)


a2 = np.clip(
    p02_shadow / 0.35,
    0.0,
    0.65
)


cyan = np.zeros_like(
    overlay
)

cyan[..., 1] = 255
cyan[..., 2] = 255


overlay = (
    overlay
    *
    (
        1.0
        -
        a2[..., None]
    )
    +
    cyan
    *
    a2[..., None]
)


overlay = np.clip(
    overlay,
    0,
    255
).astype(np.uint8)


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


# Synthetic patterned surface, deliberately different from
# original floor. This is better than a flat-color test.
yy, xx = np.indices(
    (H, W)
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
    70
    +
    45
    *
    (
        np.sin(
            xx / 11.0
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
    55
    *
    (
        np.sin(
            (xx + yy) / 17.0
        )
        +
        1.0
    )
    /
    2.0
)


pattern[..., 2] = (
    195
    +
    40
    *
    (
        np.cos(
            yy / 13.0
        )
        +
        1.0
    )
    /
    2.0
)


neutral_shadowed = neutral.copy()
pattern_shadowed = pattern.copy()


neutral_shadowed[
    floor
] *= multiplier[
    floor,
    None
]


pattern_shadowed[
    floor
] *= multiplier[
    floor,
    None
]


neutral_shadowed = np.clip(
    neutral_shadowed,
    0,
    255
).astype(np.uint8)


pattern_shadowed = np.clip(
    pattern_shadowed,
    0,
    255
).astype(np.uint8)


# ============================================================
# SAVE MAPS
# ============================================================

P01_PATH = (
    OUT
    / "02_p01_vanity_shadow.png"
)

P02_PATH = (
    OUT
    / "03_p02_toilet_shadow.png"
)

COMBINED_PATH = (
    OUT
    / "04_combined_shadow.png"
)

LIKELINESS_PATH = (
    OUT
    / "05_shadow_likeness.png"
)


Image.fromarray(
    strength8(
        p01_shadow
    )
).save(
    P01_PATH
)


Image.fromarray(
    strength8(
        p02_shadow
    )
).save(
    P02_PATH
)


Image.fromarray(
    strength8(
        combined
    )
).save(
    COMBINED_PATH
)


Image.fromarray(
    (
        likeness_vis
        *
        255
    ).astype(np.uint8)
).save(
    LIKELINESS_PATH
)


# ============================================================
# 9-PANEL AUDIT
# ============================================================

fig, axes = plt.subplots(
    1,
    9,
    figsize=(
        38,
        7
    )
)


axes[0].imshow(
    rgb.astype(np.uint8)
)

axes[0].set_title(
    "1. Original"
)

axes[0].axis(
    "off"
)


axes[1].imshow(
    darkening_vis,
    cmap="gray",
    vmin=0,
    vmax=1
)

axes[1].set_title(
    "2. Raw Darkening"
)

axes[1].axis(
    "off"
)


axes[2].imshow(
    likeness_vis,
    cmap="gray",
    vmin=0,
    vmax=1
)

axes[2].set_title(
    "3. Shadow-Likeness\n"
    "RGB + Chromaticity"
)

axes[2].axis(
    "off"
)


axes[3].imshow(
    strength8(
        p01_shadow
    ),
    cmap="gray",
    vmin=0,
    vmax=255
)

axes[3].set_title(
    "4. P01 Vanity"
)

axes[3].axis(
    "off"
)


axes[4].imshow(
    strength8(
        p02_shadow
    ),
    cmap="gray",
    vmin=0,
    vmax=255
)

axes[4].set_title(
    "5. P02 Toilet"
)

axes[4].axis(
    "off"
)


axes[5].imshow(
    strength8(
        combined
    ),
    cmap="gray",
    vmin=0,
    vmax=255
)

axes[5].set_title(
    "6. Combined"
)

axes[5].axis(
    "off"
)


axes[6].imshow(
    overlay
)

axes[6].set_title(
    "7. Location Audit"
)

axes[6].axis(
    "off"
)


axes[7].imshow(
    neutral_shadowed
)

axes[7].set_title(
    "8. Neutral Transfer"
)

axes[7].axis(
    "off"
)


axes[8].imshow(
    pattern_shadowed
)

axes[8].set_title(
    "9. New Pattern Transfer"
)

axes[8].axis(
    "off"
)


plt.tight_layout()


AUDIT_PATH = (
    OUT
    / "06_stage07d4_chromatic_shadow_audit.png"
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

p01_active = (
    p01_shadow
    >
    0.01
)

p02_active = (
    p02_shadow
    >
    0.01
)

active = (
    combined
    >
    0.01
)


print()
print("=" * 110)
print("STAGE 07D4 RESULT")
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
        active.sum()
    )
)


if active.any():

    print(
        "MEAN MULTIPLIER:",
        round(
            float(
                multiplier[
                    active
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
                    active
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
        "07D4",

    "architecture":
        "PRE_TILE_INTRINSIC_SHADOW_DECOMPOSITION",

    "method": [
        "prop-floor contact geometry",
        "local RGB reference",
        "luminance darkening",
        "RGB attenuation consistency",
        "normalized chromaticity consistency",
        "material-independent multiplier"
    ],

    "runs_before_tile_selection":
        True,

    "contains_original_floor_rgb":
        False,

    "runtime_ai_after_tile_selection":
        False,

    "outputs": {

        "multiplier_16bit":
            str(
                MULT_PATH
            ),

        "p01_shadow":
            str(
                P01_PATH
            ),

        "p02_shadow":
            str(
                P02_PATH
            ),

        "combined":
            str(
                COMBINED_PATH
            ),

        "shadow_likeness":
            str(
                LIKELINESS_PATH
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
    / "00_stage07d4_result.json"
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
    "AUDIT:",
    AUDIT_PATH
)

print(
    "MULTIPLIER:",
    MULT_PATH
)

print(
    "STATE:",
    STATE_PATH
)

print()

print(
    "NO TILE WAS SELECTED."
)

print(
    "NO GENERATIVE MODEL WAS USED."
)
