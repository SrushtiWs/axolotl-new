
from pathlib import Path
import sys
import json
import gc
import hashlib

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from PIL import Image

import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE = Path(
    "/workspace/axolotl"
)

FSD_DIR = (
    BASE
    / "test07"
    / "models"
    / "FSD"
)

CKPT_PATH = (
    FSD_DIR
    / "ckpt"
    / "FSD_best.ckpt"
)

MASTER_PATH = (
    BASE
    / "test07"
    / "production_pipeline"
    / "stage01_master"
    / "00_master_input.png"
)

OUT = (
    BASE
    / "test07"
    / "production_pipeline"
    / "stage07_empty_room"
    / "07f5_raw_fsd_inference"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# VALIDATE
# ============================================================

for p in [
    CKPT_PATH,
    MASTER_PATH,
]:

    if not p.exists():

        raise FileNotFoundError(
            p
        )


# ============================================================
# IMPORT FSD
# ============================================================

fsd_string = str(
    FSD_DIR
)

if fsd_string not in sys.path:

    sys.path.insert(
        0,
        fsd_string
    )


from networks.fdrnet import FDRNet


# ============================================================
# LOAD IMAGE
# ============================================================

image_pil = Image.open(
    MASTER_PATH
).convert("RGB")


image_np = np.asarray(
    image_pil
)


H, W = image_np.shape[:2]


print("=" * 110)
print("07F5 — RAW FSD SHADOW INFERENCE")
print("=" * 110)

print()

print(
    "MASTER:",
    MASTER_PATH
)

print(
    "ORIGINAL SIZE:",
    W,
    "x",
    H
)


# ============================================================
# IMAGE -> TENSOR
#
# Official demo:
#   cv2 BGR -> RGB
#   TF.to_tensor()
#
# PIL RGB + TF.to_tensor gives the same [0,1] RGB tensor.
# ============================================================

image_tensor = TF.to_tensor(
    image_pil
).unsqueeze(
    0
)


print(
    "INPUT RANGE:",
    float(
        image_tensor.min()
    ),
    "to",
    float(
        image_tensor.max()
    )
)


# ============================================================
# PAD TO MULTIPLE OF 32
#
# Preserve original image geometry.
# Do NOT resize/stretch the room.
#
# Right/bottom padding only.
# Reflection padding minimizes artificial border content.
# ============================================================

MULTIPLE = 32


pad_w = (
    MULTIPLE
    -
    (
        W
        %
        MULTIPLE
    )
) % MULTIPLE


pad_h = (
    MULTIPLE
    -
    (
        H
        %
        MULTIPLE
    )
) % MULTIPLE


print()

print(
    "PAD RIGHT:",
    pad_w
)

print(
    "PAD BOTTOM:",
    pad_h
)


if (
    pad_w > 0
    or
    pad_h > 0
):

    image_input = F.pad(
        image_tensor,
        (
            0,
            pad_w,
            0,
            pad_h
        ),
        mode="reflect"
    )

else:

    image_input = image_tensor


INPUT_H = image_input.shape[-2]
INPUT_W = image_input.shape[-1]


print(
    "NETWORK INPUT SIZE:",
    INPUT_W,
    "x",
    INPUT_H
)


# ============================================================
# BUILD EXACT FSD MODEL
#
# Same architecture as official demo.
#
# use_pretrained=False because the verified FSD checkpoint
# already contains backbone weights.
# ============================================================

model = FDRNet(
    backbone="efficientnet-b3",
    proj_planes=16,
    pred_planes=32,
    use_pretrained=False,
    fix_backbone=False,
    has_se=False,
    dropout_2d=0,
    normalize=True,
    mu_init=0.5,
    reweight_mode="manual",
)


# ============================================================
# LOAD VERIFIED CHECKPOINT
# ============================================================

print()
print(
    "Loading official FSD checkpoint..."
)


ckpt = torch.load(
    CKPT_PATH,
    map_location="cpu",
    weights_only=True
)


model.load_state_dict(
    ckpt["model"],
    strict=True
)


print(
    "✅ CHECKPOINT LOADED STRICTLY"
)


# ============================================================
# CUDA
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


model = model.to(
    DEVICE
)

model.eval()


image_input = image_input.to(
    DEVICE
)


print(
    "DEVICE:",
    DEVICE
)


if torch.cuda.is_available():

    torch.cuda.reset_peak_memory_stats()


# ============================================================
# ONE MODEL FORWARD PASS
#
# Official demo calls model(image) separately for:
#   binary_mask
#   soft_feat
#
# We call it ONCE and read both outputs from that same pass.
# ============================================================

print()
print(
    "Running FSD inference..."
)


with torch.inference_mode():

    output = model(
        image_input
    )


print(
    "✅ FSD FORWARD PASS COMPLETE"
)


# ============================================================
# INSPECT RAW RETURN
# ============================================================

print()
print("=" * 110)
print("RAW MODEL OUTPUT")
print("=" * 110)


print(
    "OUTPUT TYPE:",
    type(
        output
    )
)


if not isinstance(
    output,
    dict
):

    raise RuntimeError(
        "FSD output is not a dictionary."
    )


print(
    "OUTPUT KEYS:",
    list(
        output.keys()
    )
)


if "binary_mask" not in output:

    raise RuntimeError(
        "binary_mask missing from FSD output."
    )


if "soft_feat" not in output:

    raise RuntimeError(
        "soft_feat missing from FSD output."
    )


binary_raw = output[
    "binary_mask"
]


soft_raw = output[
    "soft_feat"
]


print()

print(
    "binary_mask shape:",
    tuple(
        binary_raw.shape
    )
)

print(
    "binary_mask range:",
    float(
        binary_raw.min()
    ),
    "to",
    float(
        binary_raw.max()
    )
)


print()

print(
    "soft_feat shape:",
    tuple(
        soft_raw.shape
    )
)

print(
    "soft_feat range:",
    float(
        soft_raw.min()
    ),
    "to",
    float(
        soft_raw.max()
    )
)


# ============================================================
# RESIZE MODEL OUTPUTS TO NETWORK INPUT SIZE
#
# Following official demo behavior.
# ============================================================

binary_full = F.interpolate(
    binary_raw,
    size=(
        INPUT_H,
        INPUT_W
    ),
    mode="bilinear",
    align_corners=False
)


soft_full = F.interpolate(
    soft_raw,
    size=(
        INPUT_H,
        INPUT_W
    ),
    mode="bilinear",
    align_corners=False
)


# ============================================================
# CROP BACK TO EXACT ORIGINAL STAGE01 SIZE
# ============================================================

binary_crop = binary_full[
    ...,
    :H,
    :W
]


soft_crop = soft_full[
    ...,
    :H,
    :W
]


# ============================================================
# MOVE TO CPU
# ============================================================

binary_np = (
    binary_crop[
        0,
        0
    ]
    .detach()
    .float()
    .cpu()
    .numpy()
)


soft_np = (
    soft_crop[
        0,
        0
    ]
    .detach()
    .float()
    .cpu()
    .numpy()
)


# ============================================================
# RAW STATISTICS
# ============================================================

print()
print("=" * 110)
print("CROPPED OUTPUT STATISTICS")
print("=" * 110)

print()

print(
    "BINARY SIZE:",
    binary_np.shape
)

print(
    "BINARY MIN:",
    round(
        float(
            binary_np.min()
        ),
        6
    )
)

print(
    "BINARY MAX:",
    round(
        float(
            binary_np.max()
        ),
        6
    )
)

print(
    "BINARY MEAN:",
    round(
        float(
            binary_np.mean()
        ),
        6
    )
)


print()

print(
    "SOFT SIZE:",
    soft_np.shape
)

print(
    "SOFT MIN:",
    round(
        float(
            soft_np.min()
        ),
        6
    )
)

print(
    "SOFT MAX:",
    round(
        float(
            soft_np.max()
        ),
        6
    )
)

print(
    "SOFT MEAN:",
    round(
        float(
            soft_np.mean()
        ),
        6
    )
)


# ============================================================
# OFFICIAL DEMO BINARY THRESHOLD
#
# demo.py:
#
# pred = (pred_logit > 0.5)
#
# We preserve that exact interpretation here.
# ============================================================

binary_mask = (
    binary_np
    >
    0.5
)


print()

print(
    "OFFICIAL >0.5 SHADOW PIXELS:",
    int(
        binary_mask.sum()
    )
)

print(
    "IMAGE PIXELS:",
    H * W
)

print(
    "SHADOW COVERAGE:",
    round(
        float(
            binary_mask.mean()
        ),
        6
    )
)


# ============================================================
# SAVE RAW NUMERICAL ARRAYS
#
# These are important for future analysis.
# Do not quantize our only copies.
# ============================================================

BINARY_NPY = (
    OUT
    / "01_binary_output_raw.npy"
)


SOFT_NPY = (
    OUT
    / "02_soft_output_raw.npy"
)


np.save(
    BINARY_NPY,
    binary_np.astype(
        np.float32
    )
)


np.save(
    SOFT_NPY,
    soft_np.astype(
        np.float32
    )
)


# ============================================================
# VISUALIZATION NORMALIZATION
#
# For diagnostic display ONLY.
#
# Raw .npy arrays remain untouched.
# ============================================================

def normalize_for_display(
    arr
):

    lo = float(
        np.nanmin(
            arr
        )
    )

    hi = float(
        np.nanmax(
            arr
        )
    )


    if (
        not np.isfinite(
            lo
        )
        or
        not np.isfinite(
            hi
        )
        or
        hi <= lo
    ):

        return np.zeros_like(
            arr,
            dtype=np.float32
        )


    return np.clip(
        (
            arr
            -
            lo
        )
        /
        (
            hi
            -
            lo
        ),
        0.0,
        1.0
    ).astype(
        np.float32
    )


binary_display = normalize_for_display(
    binary_np
)


soft_display = normalize_for_display(
    soft_np
)


# ============================================================
# SAVE 8-BIT DIAGNOSTIC MAPS
# ============================================================

BINARY_SCORE_PATH = (
    OUT
    / "03_binary_score_display.png"
)


SOFT_PATH = (
    OUT
    / "04_soft_output_display.png"
)


BINARY_MASK_PATH = (
    OUT
    / "05_official_threshold_binary_mask.png"
)


Image.fromarray(
    (
        binary_display
        *
        255
    ).round().astype(
        np.uint8
    )
).save(
    BINARY_SCORE_PATH
)


Image.fromarray(
    (
        soft_display
        *
        255
    ).round().astype(
        np.uint8
    )
).save(
    SOFT_PATH
)


Image.fromarray(
    binary_mask.astype(
        np.uint8
    )
    *
    255
).save(
    BINARY_MASK_PATH
)


# ============================================================
# BINARY OVERLAY
#
# Red = FSD binary shadow according to official >0.5 threshold.
# ============================================================

binary_overlay = image_np.astype(
    np.float32
).copy()


binary_overlay[
    binary_mask
] = (
    binary_overlay[
        binary_mask
    ]
    *
    0.40
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
    0.60
)


binary_overlay = np.clip(
    binary_overlay,
    0,
    255
).astype(
    np.uint8
)


BINARY_OVERLAY_PATH = (
    OUT
    / "06_binary_shadow_overlay.png"
)


Image.fromarray(
    binary_overlay
).save(
    BINARY_OVERLAY_PATH
)


# ============================================================
# SOFT HEAT OVERLAY
#
# Diagnostic only.
# High normalized soft response = stronger red overlay.
# ============================================================

soft_alpha = (
    soft_display
    *
    0.65
)


soft_overlay = (
    image_np.astype(
        np.float32
    )
    *
    (
        1.0
        -
        soft_alpha[
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
    soft_alpha[
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


SOFT_OVERLAY_PATH = (
    OUT
    / "07_soft_response_overlay.png"
)


Image.fromarray(
    soft_overlay
).save(
    SOFT_OVERLAY_PATH
)


# ============================================================
# 6-PANEL RAW AUDIT
# ============================================================

fig, axes = plt.subplots(
    1,
    6,
    figsize=(
        26,
        7
    )
)


axes[0].imshow(
    image_np
)

axes[0].set_title(
    "1. Original Stage01"
)

axes[0].axis(
    "off"
)


axes[1].imshow(
    binary_display,
    cmap="gray",
    vmin=0,
    vmax=1
)

axes[1].set_title(
    "2. FSD Binary Score\n"
    "display-normalized"
)

axes[1].axis(
    "off"
)


axes[2].imshow(
    binary_mask,
    cmap="gray"
)

axes[2].set_title(
    "3. Official Binary\n"
    "> 0.5"
)

axes[2].axis(
    "off"
)


axes[3].imshow(
    binary_overlay
)

axes[3].set_title(
    "4. Binary Overlay\n"
    "RED=shadow"
)

axes[3].axis(
    "off"
)


axes[4].imshow(
    soft_display,
    cmap="gray",
    vmin=0,
    vmax=1
)

axes[4].set_title(
    "5. FSD Soft Output\n"
    "display-normalized"
)

axes[4].axis(
    "off"
)


axes[5].imshow(
    soft_overlay
)

axes[5].set_title(
    "6. Soft Response Overlay"
)

axes[5].axis(
    "off"
)


plt.tight_layout()


AUDIT_PATH = (
    OUT
    / "08_stage07f5_raw_fsd_audit.png"
)


plt.savefig(
    AUDIT_PATH,
    dpi=150,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# VRAM
# ============================================================

peak_vram = None


if torch.cuda.is_available():

    peak_vram = (
        torch.cuda.max_memory_allocated()
        /
        1024**3
    )


    print()
    print(
        "PEAK CUDA ALLOCATED GB:",
        round(
            float(
                peak_vram
            ),
            3
        )
    )


# ============================================================
# RESULT STATE
# ============================================================

STATE = {

    "stage":
        "07F5",

    "purpose":
        "RAW_FSD_SHADOW_DETECTION_STAGE01",

    "model":
        "FSD / FDRNet",

    "backbone":
        "efficientnet-b3",

    "checkpoint":
        str(
            CKPT_PATH
        ),

    "input": {

        "path":
            str(
                MASTER_PATH
            ),

        "original_width":
            W,

        "original_height":
            H,

        "network_width":
            INPUT_W,

        "network_height":
            INPUT_H,

        "pad_right":
            pad_w,

        "pad_bottom":
            pad_h,
    },

    "postprocessing": {

        "qwen_roi_used":
            False,

        "floor_mask_used":
            False,

        "prop_mask_used":
            False,

        "custom_cleanup_used":
            False,

        "official_binary_threshold":
            0.5,
    },

    "raw_statistics": {

        "binary_min":
            float(
                binary_np.min()
            ),

        "binary_max":
            float(
                binary_np.max()
            ),

        "binary_mean":
            float(
                binary_np.mean()
            ),

        "soft_min":
            float(
                soft_np.min()
            ),

        "soft_max":
            float(
                soft_np.max()
            ),

        "soft_mean":
            float(
                soft_np.mean()
            ),

        "binary_shadow_pixels":
            int(
                binary_mask.sum()
            ),

        "binary_shadow_coverage":
            float(
                binary_mask.mean()
            ),

        "peak_cuda_allocated_gb":
            (
                float(
                    peak_vram
                )
                if peak_vram is not None
                else None
            ),
    },

    "outputs": {

        "binary_raw_npy":
            str(
                BINARY_NPY
            ),

        "soft_raw_npy":
            str(
                SOFT_NPY
            ),

        "binary_score_display":
            str(
                BINARY_SCORE_PATH
            ),

        "soft_display":
            str(
                SOFT_PATH
            ),

        "binary_mask":
            str(
                BINARY_MASK_PATH
            ),

        "binary_overlay":
            str(
                BINARY_OVERLAY_PATH
            ),

        "soft_overlay":
            str(
                SOFT_OVERLAY_PATH
            ),

        "audit":
            str(
                AUDIT_PATH
            ),
    },

    "status":
        "RND_REQUIRES_RAW_VISUAL_AUDIT",

    "next_if_useful":
        "07F6_FSD_PLUS_FLOOR_PLUS_QWEN_SEMANTIC_FUSION",
}


STATE_PATH = (
    OUT
    / "00_stage07f5_result.json"
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
# FINAL
# ============================================================

print()
print("=" * 110)
print("07F5 RESULT")
print("=" * 110)

print()

print(
    "BINARY MIN/MAX:",
    round(
        float(
            binary_np.min()
        ),
        6
    ),
    "/",
    round(
        float(
            binary_np.max()
        ),
        6
    )
)


print(
    "SOFT MIN/MAX:",
    round(
        float(
            soft_np.min()
        ),
        6
    ),
    "/",
    round(
        float(
            soft_np.max()
        ),
        6
    )
)


print(
    "OFFICIAL >0.5 SHADOW PIXELS:",
    int(
        binary_mask.sum()
    )
)


print(
    "SHADOW COVERAGE:",
    round(
        float(
            binary_mask.mean()
        ),
        6
    )
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
    "IMPORTANT:"
)

print(
    "This is RAW FSD output."
)

print(
    "No Qwen semantic ROI was used."
)

print(
    "No floor mask was used."
)

print(
    "Do not judge the final production shadow system yet."
)


# ============================================================
# CLEANUP
# ============================================================

del output
del binary_raw
del soft_raw
del binary_full
del soft_full
del ckpt
del model
del image_input

gc.collect()


if torch.cuda.is_available():

    torch.cuda.empty_cache()
