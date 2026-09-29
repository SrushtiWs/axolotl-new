
from pathlib import Path
import json

import cv2
import numpy as np

from PIL import Image

import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE = Path(
    "/workspace/axolotl"
)

PROD = (
    BASE
    / "test07"
    / "production_pipeline"
)


MASTER_PATH = (
    PROD
    / "stage01_master"
    / "00_master_input.png"
)


FLOOR_PATH = (
    PROD
    / "stage08_tile_application"
    / "08b2_floor_plus_screeding_target"
    / "01_main_floor_mask.png"
)


FUSION_DIR = (
    PROD
    / "stage07_empty_room"
    / "07f6_fsd_qwen_floor_fusion"
)


P01_FUSED_PATH = (
    FUSION_DIR
    / "07_p01_expanded_fusion.png"
)


P02_FUSED_PATH = (
    FUSION_DIR
    / "08_p02_expanded_fusion.png"
)


P01_SOFT_PATH = (
    FUSION_DIR
    / "01_p01_fsd_soft_fused.npy"
)


P02_SOFT_PATH = (
    FUSION_DIR
    / "02_p02_fsd_soft_fused.npy"
)


OUT = (
    PROD
    / "stage07_empty_room"
    / "07f7_fsd_soft_multiplier_candidates"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# VALIDATE
# ============================================================

for p in [
    MASTER_PATH,
    FLOOR_PATH,
    P01_FUSED_PATH,
    P02_FUSED_PATH,
    P01_SOFT_PATH,
    P02_SOFT_PATH,
]:

    if not p.exists():

        raise FileNotFoundError(
            p
        )


# ============================================================
# LOAD
# ============================================================

master = np.asarray(
    Image.open(
        MASTER_PATH
    ).convert("RGB")
)


floor = (
    np.asarray(
        Image.open(
            FLOOR_PATH
        ).convert("L")
    ) > 127
)


p01_mask = (
    np.asarray(
        Image.open(
            P01_FUSED_PATH
        ).convert("L")
    ) > 127
)


p02_mask = (
    np.asarray(
        Image.open(
            P02_FUSED_PATH
        ).convert("L")
    ) > 127
)


p01_soft = np.load(
    P01_SOFT_PATH
).astype(
    np.float32
)


p02_soft = np.load(
    P02_SOFT_PATH
).astype(
    np.float32
)


H, W = floor.shape


# ============================================================
# COMBINE RAW FSD SOFT SIGNAL
# ============================================================

raw_soft = np.maximum(
    p01_soft,
    p02_soft
)


support = (
    p01_mask
    |
    p02_mask
)


raw_soft[
    ~support
] = 0.0


# ============================================================
# NORMALIZE SIGNAL
#
# IMPORTANT:
#
# We do NOT normalize using min/max of this bathroom.
#
# FSD soft output already has approximately 0..1 behavior.
# Clamp to valid signal range only.
# ============================================================

signal = np.clip(
    raw_soft,
    0.0,
    1.0
)


# ============================================================
# REMOVE VERY WEAK MODEL RESPONSE
#
# This is signal cleanup, not floor-RGB thresholding.
#
# Below 0.15 = very weak FSD evidence.
# Use smooth ramp rather than binary cutoff.
# ============================================================

SIGNAL_START = 0.15


signal_clean = np.clip(
    (
        signal
        -
        SIGNAL_START
    )
    /
    (
        1.0
        -
        SIGNAL_START
    ),
    0.0,
    1.0
)


# ============================================================
# SPATIAL FEATHER
#
# FSD provides detailed response but binary fusion boundaries
# can still be sharp.
#
# Blur only the STRENGTH field and constrain it to slightly
# expanded support.
# ============================================================

support_expanded = (
    cv2.dilate(
        support.astype(
            np.uint8
        ) * 255,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                7,
                7
            )
        )
    ) > 0
)


signal_softened = cv2.GaussianBlur(
    signal_clean,
    (0, 0),
    sigmaX=1.6,
    sigmaY=1.6
)


signal_softened[
    ~support_expanded
] = 0.0


signal_softened[
    ~floor
] = 0.0


# ============================================================
# CANDIDATE A — CONSERVATIVE
#
# Maximum darkening:
# about 28%
# ============================================================

strength_A = (
    signal_softened
    *
    0.28
)


mult_A = np.clip(
    1.0
    -
    strength_A,
    0.72,
    1.0
)


# ============================================================
# CANDIDATE B — BALANCED
#
# Slight nonlinear emphasis.
# Maximum darkening:
# about 36%
# ============================================================

strength_B = (
    np.power(
        signal_softened,
        0.85
    )
    *
    0.36
)


mult_B = np.clip(
    1.0
    -
    strength_B,
    0.64,
    1.0
)


# ============================================================
# CANDIDATE C — CONTACT-WEIGHTED
#
# Weak/medium shadow remains restrained.
# Strong FSD response becomes distinctly darker.
#
# Maximum darkening:
# about 43%
# ============================================================

strength_C = (
    np.power(
        signal_softened,
        1.35
    )
    *
    0.43
)


mult_C = np.clip(
    1.0
    -
    strength_C,
    0.57,
    1.0
)


# ============================================================
# OUTSIDE FLOOR = NO SHADOW OPERATION
# ============================================================

for mult in [
    mult_A,
    mult_B,
    mult_C
]:

    mult[
        ~floor
    ] = 1.0


# ============================================================
# SAVE 16-BIT MULTIPLIERS
# ============================================================

def save_multiplier(
    name,
    mult
):

    path = (
        OUT
        / name
    )

    arr = (
        np.clip(
            mult,
            0.0,
            1.0
        )
        *
        65535.0
    ).round().astype(
        np.uint16
    )

    Image.fromarray(
        arr,
        mode="I;16"
    ).save(
        path
    )

    return path


A_PATH = save_multiplier(
    "01_multiplier_A_conservative_16bit.png",
    mult_A
)


B_PATH = save_multiplier(
    "02_multiplier_B_balanced_16bit.png",
    mult_B
)


C_PATH = save_multiplier(
    "03_multiplier_C_contact_weighted_16bit.png",
    mult_C
)


# ============================================================
# SAVE RAW SIGNAL
# ============================================================

np.save(
    OUT
    / "04_fsd_shadow_signal.npy",
    signal_softened.astype(
        np.float32
    )
)


# ============================================================
# MATERIAL-INDEPENDENCE TEST SURFACES
# ============================================================

neutral = np.full(
    (
        H,
        W,
        3
    ),
    215.0,
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
    90
    +
    70
    *
    (
        np.sin(
            xx / 14.0
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
            (
                xx + yy
            )
            /
            21.0
        )
        +
        1.0
    )
    /
    2.0
)


pattern[..., 2] = (
    185
    +
    55
    *
    (
        np.cos(
            yy / 17.0
        )
        +
        1.0
    )
    /
    2.0
)


# ============================================================
# APPLY MULTIPLIER
# ============================================================

def apply_multiplier(
    surface,
    multiplier
):

    result = surface.copy()

    result[
        floor
    ] *= multiplier[
        floor,
        None
    ]

    return np.clip(
        result,
        0,
        255
    ).astype(
        np.uint8
    )


neutral_A = apply_multiplier(
    neutral,
    mult_A
)

neutral_B = apply_multiplier(
    neutral,
    mult_B
)

neutral_C = apply_multiplier(
    neutral,
    mult_C
)


pattern_A = apply_multiplier(
    pattern,
    mult_A
)

pattern_B = apply_multiplier(
    pattern,
    mult_B
)

pattern_C = apply_multiplier(
    pattern,
    mult_C
)


# ============================================================
# STRENGTH DISPLAY
# ============================================================

def strength_display(
    multiplier
):

    strength = (
        1.0
        -
        multiplier
    )

    return np.clip(
        strength
        /
        0.45,
        0.0,
        1.0
    )


disp_A = strength_display(
    mult_A
)

disp_B = strength_display(
    mult_B
)

disp_C = strength_display(
    mult_C
)


# ============================================================
# LOCATION OVERLAY USING BALANCED CANDIDATE
# ============================================================

alpha = np.clip(
    (
        1.0
        -
        mult_B
    )
    /
    0.36,
    0.0,
    1.0
) * 0.65


overlay = (
    master.astype(
        np.float32
    )
    *
    (
        1.0
        -
        alpha[
            ...,
            None
        ]
    )
    +
    np.array(
        [
            255,
            0,
            0
        ],
        dtype=np.float32
    )
    *
    alpha[
        ...,
        None
    ]
)


overlay = np.clip(
    overlay,
    0,
    255
).astype(
    np.uint8
)


# ============================================================
# METRICS
# ============================================================

active_A = (
    mult_A
    <
    0.99
)


active_B = (
    mult_B
    <
    0.99
)


active_C = (
    mult_C
    <
    0.99
)


def metrics(
    multiplier,
    active
):

    if not active.any():

        return {
            "pixels": 0,
            "mean": 1.0,
            "min": 1.0,
        }

    return {

        "pixels":
            int(
                active.sum()
            ),

        "mean":
            float(
                multiplier[
                    active
                ].mean()
            ),

        "min":
            float(
                multiplier[
                    active
                ].min()
            ),
    }


metrics_A = metrics(
    mult_A,
    active_A
)


metrics_B = metrics(
    mult_B,
    active_B
)


metrics_C = metrics(
    mult_C,
    active_C
)


# ============================================================
# 12-PANEL AUDIT
# ============================================================

fig, axes = plt.subplots(
    2,
    6,
    figsize=(
        28,
        12
    )
)


# ROW 1
axes[0, 0].imshow(
    master
)

axes[0, 0].set_title(
    "1. Original"
)

axes[0, 0].axis(
    "off"
)


axes[0, 1].imshow(
    signal_softened,
    cmap="gray",
    vmin=0,
    vmax=1
)

axes[0, 1].set_title(
    "2. FSD Fused\nSoft Signal"
)

axes[0, 1].axis(
    "off"
)


axes[0, 2].imshow(
    disp_A,
    cmap="gray",
    vmin=0,
    vmax=1
)

axes[0, 2].set_title(
    "3. A Conservative\nStrength"
)

axes[0, 2].axis(
    "off"
)


axes[0, 3].imshow(
    disp_B,
    cmap="gray",
    vmin=0,
    vmax=1
)

axes[0, 3].set_title(
    "4. B Balanced\nStrength"
)

axes[0, 3].axis(
    "off"
)


axes[0, 4].imshow(
    disp_C,
    cmap="gray",
    vmin=0,
    vmax=1
)

axes[0, 4].set_title(
    "5. C Contact-Weighted\nStrength"
)

axes[0, 4].axis(
    "off"
)


axes[0, 5].imshow(
    overlay
)

axes[0, 5].set_title(
    "6. Balanced\nLocation Audit"
)

axes[0, 5].axis(
    "off"
)


# ROW 2

axes[1, 0].imshow(
    neutral_A
)

axes[1, 0].set_title(
    "7. Neutral A"
)

axes[1, 0].axis(
    "off"
)


axes[1, 1].imshow(
    neutral_B
)

axes[1, 1].set_title(
    "8. Neutral B"
)

axes[1, 1].axis(
    "off"
)


axes[1, 2].imshow(
    neutral_C
)

axes[1, 2].set_title(
    "9. Neutral C"
)

axes[1, 2].axis(
    "off"
)


axes[1, 3].imshow(
    pattern_A
)

axes[1, 3].set_title(
    "10. Pattern A"
)

axes[1, 3].axis(
    "off"
)


axes[1, 4].imshow(
    pattern_B
)

axes[1, 4].set_title(
    "11. Pattern B"
)

axes[1, 4].axis(
    "off"
)


axes[1, 5].imshow(
    pattern_C
)

axes[1, 5].set_title(
    "12. Pattern C"
)

axes[1, 5].axis(
    "off"
)


plt.tight_layout()


AUDIT_PATH = (
    OUT
    / "05_stage07f7_multiplier_candidates_audit.png"
)


plt.savefig(
    AUDIT_PATH,
    dpi=150,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# RESULT
# ============================================================

print()
print("=" * 110)
print("07F7 RESULT")
print("=" * 110)

print()

print(
    "RAW SIGNAL MIN/MAX:",
    round(
        float(
            signal[
                support
            ].min()
        ),
        6
    ),
    "/",
    round(
        float(
            signal[
                support
            ].max()
        ),
        6
    )
)


print()

print(
    "A CONSERVATIVE:"
)

print(
    "  ACTIVE:",
    metrics_A[
        "pixels"
    ]
)

print(
    "  MEAN MULTIPLIER:",
    round(
        metrics_A[
            "mean"
        ],
        4
    )
)

print(
    "  STRONGEST:",
    round(
        metrics_A[
            "min"
        ],
        4
    )
)


print()

print(
    "B BALANCED:"
)

print(
    "  ACTIVE:",
    metrics_B[
        "pixels"
    ]
)

print(
    "  MEAN MULTIPLIER:",
    round(
        metrics_B[
            "mean"
        ],
        4
    )
)

print(
    "  STRONGEST:",
    round(
        metrics_B[
            "min"
        ],
        4
    )
)


print()

print(
    "C CONTACT-WEIGHTED:"
)

print(
    "  ACTIVE:",
    metrics_C[
        "pixels"
    ]
)

print(
    "  MEAN MULTIPLIER:",
    round(
        metrics_C[
            "mean"
        ],
        4
    )
)

print(
    "  STRONGEST:",
    round(
        metrics_C[
            "min"
        ],
        4
    )
)


# ============================================================
# STATE
# ============================================================

STATE = {

    "stage":
        "07F7",

    "architecture":
        "FSD_SOFT_RESPONSE_TO_REUSABLE_MATERIAL_INDEPENDENT_MULTIPLIER",

    "model_run":
        False,

    "uses_original_floor_rgb_for_strength":
        False,

    "source":
        "07F6 fused FSD soft response",

    "signal_start":
        SIGNAL_START,

    "candidates": {

        "A_conservative": {

            "max_nominal_darkening":
                0.28,

            "metrics":
                metrics_A,

            "path":
                str(
                    A_PATH
                ),
        },

        "B_balanced": {

            "max_nominal_darkening":
                0.36,

            "metrics":
                metrics_B,

            "path":
                str(
                    B_PATH
                ),
        },

        "C_contact_weighted": {

            "max_nominal_darkening":
                0.43,

            "metrics":
                metrics_C,

            "path":
                str(
                    C_PATH
                ),
        },
    },

    "audit":
        str(
            AUDIT_PATH
        ),

    "status":
        "RND_COMPARE_CANDIDATE_MAPPINGS",

    "next_if_pass":
        "APPLY_SELECTED_MULTIPLIER_TO_08H1_TRUE_METRIC_TILE",
}


STATE_PATH = (
    OUT
    / "00_stage07f7_result.json"
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
    "STATE:",
    STATE_PATH
)

print()

print(
    "NO MODEL WAS RUN."
)

print(
    "NO ORIGINAL FLOOR RGB WAS USED TO DERIVE STRENGTH."
)

print(
    "NO CANDIDATE IS FROZEN YET."
)
