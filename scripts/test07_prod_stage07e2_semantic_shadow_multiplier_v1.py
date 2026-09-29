
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


P01_SEMANTIC_PATH = (
    PROD
    / "stage07_empty_room"
    / "07e1b_qwen_coordinate_fix"
    / "01_p01_shadow_mask.png"
)


P02_SEMANTIC_PATH = (
    PROD
    / "stage07_empty_room"
    / "07e1b_qwen_coordinate_fix"
    / "02_p02_shadow_mask.png"
)


OUT = (
    PROD
    / "stage07_empty_room"
    / "07e2_semantic_shadow_multiplier"
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
    P01_SEMANTIC_PATH,
    P02_SEMANTIC_PATH,
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
            P01_SEMANTIC_PATH
        ).convert("L")
    ) > 127
)


p02_roi = (
    np.asarray(
        Image.open(
            P02_SEMANTIC_PATH
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
)


# ============================================================
# BUILD LOCAL REFERENCE RINGS
#
# Instead of comparing shadow ROI against the whole floor,
# estimate nearby floor brightness surrounding each semantic
# shadow region.
# ============================================================

def make_reference_ring(
    roi,
    floor_mask,
    inner_radius,
    outer_radius
):

    roi_u8 = (
        roi.astype(
            np.uint8
        ) * 255
    )


    inner_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            inner_radius * 2 + 1,
            inner_radius * 2 + 1
        )
    )


    outer_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            outer_radius * 2 + 1,
            outer_radius * 2 + 1
        )
    )


    inner = (
        cv2.dilate(
            roi_u8,
            inner_kernel
        ) > 0
    )


    outer = (
        cv2.dilate(
            roi_u8,
            outer_kernel
        ) > 0
    )


    ring = (
        outer
        &
        (~inner)
        &
        floor_mask
    )


    return ring


p01_ring = make_reference_ring(
    p01_roi,
    floor,
    inner_radius=5,
    outer_radius=32
)


p02_ring = make_reference_ring(
    p02_roi,
    floor,
    inner_radius=4,
    outer_radius=26
)


# ============================================================
# LOCAL REFERENCE RGB
#
# Use robust median from local nearby floor.
#
# This reference is used only for estimating illumination
# attenuation. It is NOT copied into final output.
# ============================================================

def robust_reference_rgb(
    ring
):

    pixels = rgb[
        ring
    ]


    if len(
        pixels
    ) < 20:

        raise RuntimeError(
            "Reference ring too small."
        )


    return np.median(
        pixels,
        axis=0
    )


p01_ref_rgb = robust_reference_rgb(
    p01_ring
)


p02_ref_rgb = robust_reference_rgb(
    p02_ring
)


p01_ref_luma = float(
    luminance(
        p01_ref_rgb.reshape(
            1,
            1,
            3
        )
    )[0, 0]
)


p02_ref_luma = float(
    luminance(
        p02_ref_rgb.reshape(
            1,
            1,
            3
        )
    )[0, 0]
)


print("=" * 110)
print("STAGE 07E2 — SEMANTIC SHADOW MULTIPLIER")
print("=" * 110)

print()

print(
    "P01 REFERENCE RGB:",
    [
        round(
            float(v),
            2
        )
        for v in p01_ref_rgb
    ]
)

print(
    "P01 REFERENCE LUMA:",
    round(
        p01_ref_luma,
        2
    )
)

print()

print(
    "P02 REFERENCE RGB:",
    [
        round(
            float(v),
            2
        )
        for v in p02_ref_rgb
    ]
)

print(
    "P02 REFERENCE LUMA:",
    round(
        p02_ref_luma,
        2
    )
)


# ============================================================
# RAW LOCAL ATTENUATION
#
# shadow multiplier approximately:
#
# observed_luma / nearby_unshadowed_luma
#
# But ONLY inside semantic ROI.
# ============================================================

def raw_multiplier_from_reference(
    roi,
    reference_luma
):

    mult = np.ones(
        (
            H,
            W
        ),
        dtype=np.float32
    )


    ratio = (
        luma
        /
        max(
            reference_luma,
            1.0
        )
    )


    ratio = np.clip(
        ratio,
        0.50,
        1.05
    )


    mult[
        roi
    ] = ratio[
        roi
    ]


    return mult


p01_raw_mult = raw_multiplier_from_reference(
    p01_roi,
    p01_ref_luma
)


p02_raw_mult = raw_multiplier_from_reference(
    p02_roi,
    p02_ref_luma
)


# ============================================================
# CHROMATICITY CONSISTENCY
#
# Strong color/material differences should be trusted less
# than approximately achromatic illumination attenuation.
# ============================================================

def chroma_similarity_to_reference(
    reference_rgb
):

    eps = 1.0


    obs_sum = np.sum(
        rgb,
        axis=2,
        keepdims=True
    )


    obs_chr = (
        rgb
        /
        np.maximum(
            obs_sum,
            eps
        )
    )


    ref = np.asarray(
        reference_rgb,
        dtype=np.float32
    )


    ref_chr = (
        ref
        /
        max(
            float(
                ref.sum()
            ),
            eps
        )
    )


    distance = np.sqrt(
        np.sum(
            (
                obs_chr
                -
                ref_chr[
                    None,
                    None,
                    :
                ]
            ) ** 2,
            axis=2
        )
    )


    similarity = np.exp(
        -(
            distance
            /
            0.070
        ) ** 2
    )


    return np.clip(
        similarity,
        0.0,
        1.0
    )


p01_chroma = chroma_similarity_to_reference(
    p01_ref_rgb
)


p02_chroma = chroma_similarity_to_reference(
    p02_ref_rgb
)


# ============================================================
# SEMANTIC ROI EDGE FALL-OFF
#
# Hard Qwen polygon boundaries must NOT become hard shadows.
#
# Distance to ROI boundary gives a soft semantic confidence.
# ============================================================

def interior_confidence(
    roi
):

    u8 = (
        roi.astype(
            np.uint8
        ) * 255
    )


    dist = cv2.distanceTransform(
        u8,
        cv2.DIST_L2,
        5
    )


    if dist.max() <= 0:

        return np.zeros_like(
            dist,
            dtype=np.float32
        )


    conf = np.clip(
        dist / 10.0,
        0.0,
        1.0
    )


    # Don't completely remove polygon boundary;
    # retain minimum support.
    conf = (
        0.30
        +
        0.70
        *
        conf
    )


    conf[
        ~roi
    ] = 0.0


    return conf.astype(
        np.float32
    )


p01_sem_conf = interior_confidence(
    p01_roi
)


p02_sem_conf = interior_confidence(
    p02_roi
)


# ============================================================
# BUILD SHADOW STRENGTH
#
# Strength = amount darker than reference.
#
# Semantic ROI determines WHERE.
# Chromaticity reduces floor-design contamination.
# ============================================================

def build_strength(
    raw_mult,
    roi,
    chroma_conf,
    semantic_conf,
    min_darkening
):

    darkening = np.clip(
        1.0
        -
        raw_mult,
        0.0,
        0.50
    )


    # Ignore extremely weak differences.
    darkening[
        darkening
        <
        min_darkening
    ] = 0.0


    strength = (
        darkening
        *
        (
            0.45
            +
            0.55
            *
            chroma_conf
        )
        *
        semantic_conf
    )


    strength[
        ~roi
    ] = 0.0


    # Soft physical falloff
    strength = cv2.GaussianBlur(
        strength.astype(
            np.float32
        ),
        (0, 0),
        sigmaX=2.3,
        sigmaY=2.3
    )


    # Permit slight feathering just outside Qwen polygon.
    expanded = (
        cv2.dilate(
            roi.astype(
                np.uint8
            ) * 255,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (
                    9,
                    9
                )
            )
        ) > 0
    )


    strength[
        ~expanded
    ] = 0.0


    strength[
        ~floor
    ] = 0.0


    return np.clip(
        strength,
        0.0,
        0.45
    )


p01_strength = build_strength(
    p01_raw_mult,
    p01_roi,
    p01_chroma,
    p01_sem_conf,
    min_darkening=0.025
)


p02_strength = build_strength(
    p02_raw_mult,
    p02_roi,
    p02_chroma,
    p02_sem_conf,
    min_darkening=0.025
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
# 1.00 = no shadow
# lower = darker
# ============================================================

multiplier = (
    1.0
    -
    combined_strength
)


multiplier = np.clip(
    multiplier,
    0.55,
    1.00
)


multiplier[
    ~floor
] = 1.0


# ============================================================
# SAVE 16-BIT MULTIPLIER
# ============================================================

MULT16_PATH = (
    OUT
    / "01_floor_shadow_multiplier_16bit.png"
)


mult16 = (
    multiplier
    *
    65535.0
).round().astype(
    np.uint16
)


Image.fromarray(
    mult16,
    mode="I;16"
).save(
    MULT16_PATH
)


# ============================================================
# SAVE SHADOW STRENGTH MAPS
# ============================================================

def strength8(
    x
):

    return (
        np.clip(
            x
            /
            0.45,
            0.0,
            1.0
        )
        *
        255
    ).astype(
        np.uint8
    )


P01_PATH = (
    OUT
    / "02_p01_vanity_shadow_strength.png"
)


P02_PATH = (
    OUT
    / "03_p02_toilet_shadow_strength.png"
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
# REFERENCE-RING AUDIT
# ============================================================

ring_vis = np.zeros(
    (
        H,
        W,
        3
    ),
    dtype=np.uint8
)


ring_vis[
    p01_ring
] = [
    255,
    0,
    0
]


ring_vis[
    p02_ring
] = [
    0,
    255,
    255
]


ring_vis[
    p01_ring
    &
    p02_ring
] = [
    255,
    255,
    0
]


# ============================================================
# LOCATION OVERLAY
# ============================================================

overlay = rgb.copy()


a1 = np.clip(
    p01_strength
    /
    0.35,
    0.0,
    0.70
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
    p02_strength
    /
    0.35,
    0.0,
    0.70
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
).astype(
    np.uint8
)


# ============================================================
# MATERIAL-INDEPENDENCE TESTS
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


# Deliberately colorful synthetic pattern.
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
    135
    +
    65
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
    rgb.astype(
        np.uint8
    )
)

axes[0].set_title(
    "1. Original"
)

axes[0].axis(
    "off"
)


axes[1].imshow(
    ring_vis
)

axes[1].set_title(
    "2. Local Reference Rings\n"
    "RED=P01 CYAN=P02"
)

axes[1].axis(
    "off"
)


axes[2].imshow(
    p01_roi,
    cmap="gray"
)

axes[2].set_title(
    "3. Qwen P01 ROI"
)

axes[2].axis(
    "off"
)


axes[3].imshow(
    strength8(
        p01_strength
    ),
    cmap="gray",
    vmin=0,
    vmax=255
)

axes[3].set_title(
    "4. P01 Soft Strength"
)

axes[3].axis(
    "off"
)


axes[4].imshow(
    p02_roi,
    cmap="gray"
)

axes[4].set_title(
    "5. Qwen P02 ROI"
)

axes[4].axis(
    "off"
)


axes[5].imshow(
    strength8(
        p02_strength
    ),
    cmap="gray",
    vmin=0,
    vmax=255
)

axes[5].set_title(
    "6. P02 Soft Strength"
)

axes[5].axis(
    "off"
)


axes[6].imshow(
    overlay
)

axes[6].set_title(
    "7. Strength Location Audit"
)

axes[6].axis(
    "off"
)


axes[7].imshow(
    neutral_result
)

axes[7].set_title(
    "8. Neutral Transfer"
)

axes[7].axis(
    "off"
)


axes[8].imshow(
    pattern_result
)

axes[8].set_title(
    "9. Pattern Transfer"
)

axes[8].axis(
    "off"
)


plt.tight_layout()


AUDIT_PATH = (
    OUT
    / "05_stage07e2_semantic_shadow_multiplier_audit.png"
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
print("STAGE 07E2 RESULT")
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
        "07E2",

    "architecture":
        "PRE_TILE_SEMANTIC_GUIDED_SHADOW_MULTIPLIER",

    "pipeline": [

        "Qwen semantic shadow localization",

        "local nearby floor reference",

        "luminance attenuation estimation",

        "chromaticity consistency weighting",

        "soft semantic boundary",

        "16-bit reusable multiplier"
    ],

    "runs_before_tile_selection":
        True,

    "model_run":
        False,

    "contains_original_floor_rgb":
        False,

    "runtime_after_tile_selection":
        "simple_pixel_multiplier_only",

    "outputs": {

        "multiplier_16bit":
            str(
                MULT16_PATH
            ),

        "p01_strength":
            str(
                P01_PATH
            ),

        "p02_strength":
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
    / "00_stage07e2_result.json"
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
    MULT16_PATH
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
    "NO MODEL WAS RUN IN 07E2."
)

print(
    "NO TILE WAS SELECTED."
)

print(
    "THIS MAP CAN BE REUSED ON FUTURE TILE RGB."
)
