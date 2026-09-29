
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


QWEN_P01_PATH = (
    PROD
    / "stage07_empty_room"
    / "07e1b_qwen_coordinate_fix"
    / "01_p01_shadow_mask.png"
)


QWEN_P02_PATH = (
    PROD
    / "stage07_empty_room"
    / "07e1b_qwen_coordinate_fix"
    / "02_p02_shadow_mask.png"
)


FSD_DIR = (
    PROD
    / "stage07_empty_room"
    / "07f5_raw_fsd_inference"
)


FSD_BINARY_RAW_PATH = (
    FSD_DIR
    / "01_binary_output_raw.npy"
)


FSD_SOFT_RAW_PATH = (
    FSD_DIR
    / "02_soft_output_raw.npy"
)


OUT = (
    PROD
    / "stage07_empty_room"
    / "07f6_fsd_qwen_floor_fusion"
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
    QWEN_P01_PATH,
    QWEN_P02_PATH,
    FSD_BINARY_RAW_PATH,
    FSD_SOFT_RAW_PATH,
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


qwen_p01 = (
    np.asarray(
        Image.open(
            QWEN_P01_PATH
        ).convert("L")
    ) > 127
)


qwen_p02 = (
    np.asarray(
        Image.open(
            QWEN_P02_PATH
        ).convert("L")
    ) > 127
)


fsd_binary_score = np.load(
    FSD_BINARY_RAW_PATH
).astype(
    np.float32
)


fsd_soft = np.load(
    FSD_SOFT_RAW_PATH
).astype(
    np.float32
)


H, W = floor.shape


for name, arr in [
    (
        "master",
        master
    ),
    (
        "fsd_binary",
        fsd_binary_score
    ),
    (
        "fsd_soft",
        fsd_soft
    ),
]:

    if arr.shape[:2] != (
        H,
        W
    ):

        raise RuntimeError(
            f"{name} size mismatch: {arr.shape}"
        )


print("=" * 110)
print("07F6 — FSD + QWEN + FLOOR FUSION")
print("=" * 110)

print()

print(
    "IMAGE:",
    W,
    "x",
    H
)


# ============================================================
# OFFICIAL FSD BINARY INTERPRETATION
#
# Official demo:
#   binary_mask > 0.5
# ============================================================

fsd_binary = (
    fsd_binary_score
    >
    0.5
)


fsd_floor = (
    fsd_binary
    &
    floor
)


# ============================================================
# QWEN STRICT ROIs
# ============================================================

qwen_p01 &= floor
qwen_p02 &= floor


# ============================================================
# STRICT FUSION
#
# Qwen polygon exactly as returned.
# ============================================================

p01_strict = (
    fsd_floor
    &
    qwen_p01
)


p02_strict = (
    fsd_floor
    &
    qwen_p02
)


strict_union = (
    p01_strict
    |
    p02_strict
)


# ============================================================
# EXPANDED SEMANTIC SUPPORT
#
# Qwen decides approximate ownership/region.
#
# FSD still decides which pixels inside that neighborhood
# actually exhibit shadow evidence.
#
# Expansion is deliberately modest and then clipped to floor.
# ============================================================

def semantic_expand(
    mask,
    radius
):

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            radius * 2 + 1,
            radius * 2 + 1
        )
    )


    expanded = (
        cv2.dilate(
            mask.astype(
                np.uint8
            ) * 255,
            kernel
        ) > 0
    )


    return (
        expanded
        &
        floor
    )


p01_support = semantic_expand(
    qwen_p01,
    radius=16
)


p02_support = semantic_expand(
    qwen_p02,
    radius=14
)


# ============================================================
# EXPANDED BINARY FUSION
# ============================================================

p01_fused = (
    fsd_floor
    &
    p01_support
)


p02_fused = (
    fsd_floor
    &
    p02_support
)


# Resolve overlap conservatively.
#
# If both semantic supports contain a pixel, assign it to the
# closer original Qwen ROI.
# ============================================================

overlap = (
    p01_fused
    &
    p02_fused
)


if overlap.any():

    p01_src = (
        qwen_p01.astype(
            np.uint8
        ) * 255
    )


    p02_src = (
        qwen_p02.astype(
            np.uint8
        ) * 255
    )


    p01_dist = cv2.distanceTransform(
        255 - p01_src,
        cv2.DIST_L2,
        5
    )


    p02_dist = cv2.distanceTransform(
        255 - p02_src,
        cv2.DIST_L2,
        5
    )


    assign_p01 = (
        overlap
        &
        (
            p01_dist
            <=
            p02_dist
        )
    )


    assign_p02 = (
        overlap
        &
        (
            p02_dist
            <
            p01_dist
        )
    )


    p01_fused[
        overlap
    ] = False


    p02_fused[
        overlap
    ] = False


    p01_fused |= assign_p01
    p02_fused |= assign_p02


fused_union = (
    p01_fused
    |
    p02_fused
)


# ============================================================
# FSD SOFT RESPONSE INSIDE FUSED REGIONS
#
# Do NOT reinterpret as calibrated probability.
# Preserve raw model response.
# ============================================================

p01_soft_fused = np.zeros(
    (
        H,
        W
    ),
    dtype=np.float32
)


p02_soft_fused = np.zeros(
    (
        H,
        W
    ),
    dtype=np.float32
)


p01_soft_fused[
    p01_fused
] = fsd_soft[
    p01_fused
]


p02_soft_fused[
    p02_fused
] = fsd_soft[
    p02_fused
]


combined_soft = np.maximum(
    p01_soft_fused,
    p02_soft_fused
)


# ============================================================
# SAVE RAW FLOAT FUSION
# ============================================================

np.save(
    OUT
    / "01_p01_fsd_soft_fused.npy",
    p01_soft_fused
)


np.save(
    OUT
    / "02_p02_fsd_soft_fused.npy",
    p02_soft_fused
)


np.save(
    OUT
    / "03_combined_fsd_soft_fused.npy",
    combined_soft
)


# ============================================================
# SAVE MASKS
# ============================================================

def save_mask(
    path,
    mask
):

    Image.fromarray(
        mask.astype(
            np.uint8
        ) * 255
    ).save(
        path
    )


save_mask(
    OUT
    / "04_fsd_floor_binary.png",
    fsd_floor
)


save_mask(
    OUT
    / "05_p01_strict_fusion.png",
    p01_strict
)


save_mask(
    OUT
    / "06_p02_strict_fusion.png",
    p02_strict
)


save_mask(
    OUT
    / "07_p01_expanded_fusion.png",
    p01_fused
)


save_mask(
    OUT
    / "08_p02_expanded_fusion.png",
    p02_fused
)


save_mask(
    OUT
    / "09_expanded_fusion_union.png",
    fused_union
)


# ============================================================
# NORMALIZED SOFT DISPLAY
#
# Display only.
# Raw arrays remain untouched.
# ============================================================

soft_max = float(
    fsd_soft.max()
)


if soft_max > 0:

    soft_display = np.clip(
        fsd_soft
        /
        soft_max,
        0.0,
        1.0
    )

else:

    soft_display = np.zeros_like(
        fsd_soft
    )


combined_soft_display = np.zeros_like(
    combined_soft
)


if fused_union.any():

    local_max = float(
        combined_soft[
            fused_union
        ].max()
    )


    if local_max > 0:

        combined_soft_display = np.clip(
            combined_soft
            /
            local_max,
            0.0,
            1.0
        )


# ============================================================
# SEMANTIC SUPPORT OVERLAY
# ============================================================

semantic_overlay = master.astype(
    np.float32
).copy()


semantic_overlay[
    p01_support
] = (
    semantic_overlay[
        p01_support
    ]
    *
    0.65
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
    0.35
)


semantic_overlay[
    p02_support
] = (
    semantic_overlay[
        p02_support
    ]
    *
    0.65
    +
    np.array(
        [
            0,
            255,
            255
        ],
        dtype=np.float32
    )
    *
    0.35
)


semantic_overlay = np.clip(
    semantic_overlay,
    0,
    255
).astype(
    np.uint8
)


# ============================================================
# FUSED LOCATION OVERLAY
#
# RED  = P01 vanity-associated FSD shadow
# CYAN = P02 toilet-associated FSD shadow
# ============================================================

fused_overlay = master.astype(
    np.float32
).copy()


fused_overlay[
    p01_fused
] = (
    fused_overlay[
        p01_fused
    ]
    *
    0.35
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
    0.65
)


fused_overlay[
    p02_fused
] = (
    fused_overlay[
        p02_fused
    ]
    *
    0.35
    +
    np.array(
        [
            0,
            255,
            255
        ],
        dtype=np.float32
    )
    *
    0.65
)


fused_overlay = np.clip(
    fused_overlay,
    0,
    255
).astype(
    np.uint8
)


FUSED_OVERLAY_PATH = (
    OUT
    / "10_fused_shadow_location_overlay.png"
)


Image.fromarray(
    fused_overlay
).save(
    FUSED_OVERLAY_PATH
)


# ============================================================
# SOFT-FUSION OVERLAY
# ============================================================

alpha = (
    combined_soft_display
    *
    0.65
)


soft_overlay = (
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
)


red_layer = np.zeros_like(
    master,
    dtype=np.float32
)

red_layer[..., 0] = 255


cyan_layer = np.zeros_like(
    master,
    dtype=np.float32
)

cyan_layer[..., 1] = 255
cyan_layer[..., 2] = 255


p01_alpha = (
    np.where(
        p01_fused,
        combined_soft_display,
        0
    )
    *
    0.65
)


p02_alpha = (
    np.where(
        p02_fused,
        combined_soft_display,
        0
    )
    *
    0.65
)


soft_overlay = master.astype(
    np.float32
)


soft_overlay = (
    soft_overlay
    *
    (
        1.0
        -
        p01_alpha[
            ...,
            None
        ]
    )
    +
    red_layer
    *
    p01_alpha[
        ...,
        None
    ]
)


soft_overlay = (
    soft_overlay
    *
    (
        1.0
        -
        p02_alpha[
            ...,
            None
        ]
    )
    +
    cyan_layer
    *
    p02_alpha[
        ...,
        None
    ]
)


soft_overlay = np.clip(
    soft_overlay,
    0,
    255
).astype(
    np.uint8
)


# ============================================================
# METRICS
# ============================================================

def safe_mean(
    arr,
    mask
):

    if not mask.any():

        return 0.0

    return float(
        arr[
            mask
        ].mean()
    )


def safe_max(
    arr,
    mask
):

    if not mask.any():

        return 0.0

    return float(
        arr[
            mask
        ].max()
    )


stats = {

    "fsd_binary_all":
        int(
            fsd_binary.sum()
        ),

    "fsd_binary_floor":
        int(
            fsd_floor.sum()
        ),

    "qwen_p01":
        int(
            qwen_p01.sum()
        ),

    "qwen_p02":
        int(
            qwen_p02.sum()
        ),

    "p01_strict":
        int(
            p01_strict.sum()
        ),

    "p02_strict":
        int(
            p02_strict.sum()
        ),

    "strict_union":
        int(
            strict_union.sum()
        ),

    "p01_support":
        int(
            p01_support.sum()
        ),

    "p02_support":
        int(
            p02_support.sum()
        ),

    "p01_fused":
        int(
            p01_fused.sum()
        ),

    "p02_fused":
        int(
            p02_fused.sum()
        ),

    "fused_union":
        int(
            fused_union.sum()
        ),

    "p01_soft_mean":
        safe_mean(
            fsd_soft,
            p01_fused
        ),

    "p01_soft_max":
        safe_max(
            fsd_soft,
            p01_fused
        ),

    "p02_soft_mean":
        safe_mean(
            fsd_soft,
            p02_fused
        ),

    "p02_soft_max":
        safe_max(
            fsd_soft,
            p02_fused
        ),
}


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
    master
)

axes[0].set_title(
    "1. Original"
)

axes[0].axis(
    "off"
)


axes[1].imshow(
    fsd_binary,
    cmap="gray"
)

axes[1].set_title(
    "2. Raw FSD Binary"
)

axes[1].axis(
    "off"
)


axes[2].imshow(
    fsd_floor,
    cmap="gray"
)

axes[2].set_title(
    "3. FSD ∩ Floor"
)

axes[2].axis(
    "off"
)


qwen_vis = np.zeros(
    (
        H,
        W,
        3
    ),
    dtype=np.uint8
)

qwen_vis[
    qwen_p01
] = [
    255,
    0,
    0
]

qwen_vis[
    qwen_p02
] = [
    0,
    255,
    255
]


axes[3].imshow(
    qwen_vis
)

axes[3].set_title(
    "4. Qwen ROIs\n"
    "RED=P01 CYAN=P02"
)

axes[3].axis(
    "off"
)


strict_vis = np.zeros(
    (
        H,
        W,
        3
    ),
    dtype=np.uint8
)

strict_vis[
    p01_strict
] = [
    255,
    0,
    0
]

strict_vis[
    p02_strict
] = [
    0,
    255,
    255
]


axes[4].imshow(
    strict_vis
)

axes[4].set_title(
    "5. STRICT Fusion"
)

axes[4].axis(
    "off"
)


axes[5].imshow(
    semantic_overlay
)

axes[5].set_title(
    "6. Expanded Semantic\nSupport"
)

axes[5].axis(
    "off"
)


fused_vis = np.zeros(
    (
        H,
        W,
        3
    ),
    dtype=np.uint8
)

fused_vis[
    p01_fused
] = [
    255,
    0,
    0
]

fused_vis[
    p02_fused
] = [
    0,
    255,
    255
]


axes[6].imshow(
    fused_vis
)

axes[6].set_title(
    "7. Expanded FSD Fusion"
)

axes[6].axis(
    "off"
)


axes[7].imshow(
    fused_overlay
)

axes[7].set_title(
    "8. Fused Location Audit"
)

axes[7].axis(
    "off"
)


axes[8].imshow(
    combined_soft_display,
    cmap="gray",
    vmin=0,
    vmax=1
)

axes[8].set_title(
    "9. FSD Soft Response\ninside Fusion"
)

axes[8].axis(
    "off"
)


axes[9].imshow(
    soft_overlay
)

axes[9].set_title(
    "10. Soft Fusion Overlay"
)

axes[9].axis(
    "off"
)


plt.tight_layout()


AUDIT_PATH = (
    OUT
    / "11_stage07f6_fsd_qwen_floor_fusion_audit.png"
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
print("07F6 RESULT")
print("=" * 110)

print()

print(
    "RAW FSD SHADOW PIXELS:",
    stats[
        "fsd_binary_all"
    ]
)


print(
    "FSD FLOOR SHADOW PIXELS:",
    stats[
        "fsd_binary_floor"
    ]
)


print()

print(
    "P01 QWEN ROI:",
    stats[
        "qwen_p01"
    ]
)

print(
    "P01 STRICT FUSION:",
    stats[
        "p01_strict"
    ]
)

print(
    "P01 EXPANDED FUSION:",
    stats[
        "p01_fused"
    ]
)


print()

print(
    "P02 QWEN ROI:",
    stats[
        "qwen_p02"
    ]
)

print(
    "P02 STRICT FUSION:",
    stats[
        "p02_strict"
    ]
)

print(
    "P02 EXPANDED FUSION:",
    stats[
        "p02_fused"
    ]
)


print()

print(
    "FUSED UNION:",
    stats[
        "fused_union"
    ]
)


print()

print(
    "P01 SOFT MEAN/MAX:",
    round(
        stats[
            "p01_soft_mean"
        ],
        6
    ),
    "/",
    round(
        stats[
            "p01_soft_max"
        ],
        6
    )
)


print(
    "P02 SOFT MEAN/MAX:",
    round(
        stats[
            "p02_soft_mean"
        ],
        6
    ),
    "/",
    round(
        stats[
            "p02_soft_max"
        ],
        6
    )
)


# ============================================================
# SAVE STATE
# ============================================================

STATE = {

    "stage":
        "07F6",

    "architecture":
        "FSD_VISUAL_EVIDENCE_PLUS_QWEN_OWNERSHIP_PLUS_FLOOR_GEOMETRY",

    "model_run":
        False,

    "sources": {

        "fsd_binary":
            str(
                FSD_BINARY_RAW_PATH
            ),

        "fsd_soft":
            str(
                FSD_SOFT_RAW_PATH
            ),

        "qwen_p01":
            str(
                QWEN_P01_PATH
            ),

        "qwen_p02":
            str(
                QWEN_P02_PATH
            ),

        "floor":
            str(
                FLOOR_PATH
            ),
    },

    "semantic_expansion_pixels": {

        "p01":
            16,

        "p02":
            14,
    },

    "statistics":
        stats,

    "outputs": {

        "fused_overlay":
            str(
                FUSED_OVERLAY_PATH
            ),

        "audit":
            str(
                AUDIT_PATH
            ),
    },

    "status":
        "RND_REQUIRES_VISUAL_FUSION_AUDIT",

    "next_if_pass":
        "07F7_DERIVE_REUSABLE_SHADOW_STRENGTH_FROM_FSD_SOFT_RESPONSE",
}


STATE_PATH = (
    OUT
    / "00_stage07f6_result.json"
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
    "NO SHADOW MULTIPLIER HAS BEEN GENERATED YET."
)
