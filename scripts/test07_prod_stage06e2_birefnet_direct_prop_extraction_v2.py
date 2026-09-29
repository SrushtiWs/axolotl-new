
from pathlib import Path
import json
import gc

import numpy as np
import torch
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation


# ============================================================
# PATHS
# ============================================================

BASE = Path("/workspace/axolotl")

PROD = BASE / "test07" / "production_pipeline"

STAGE05F_PATH = (
    PROD
    / "stage05_clean_room_with_props"
    / "12_final_canonical_prop_detection_master.png"
)

OUT = (
    PROD
    / "stage06_prop_layer"
    / "06e2_birefnet_direct_prop_extraction_v2"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_ID = "ZhengPeng7/BiRefNet"

CACHE = "/workspace/data/huggingface-cache"


# ============================================================
# INPUT
# ============================================================

if not STAGE05F_PATH.exists():
    raise FileNotFoundError(STAGE05F_PATH)


image = Image.open(
    STAGE05F_PATH
).convert("RGB")

W, H = image.size


print("=" * 110)
print("PRODUCTION EXPERIMENT 06E2 V2")
print("BIREFNET DIRECT FULL-PROP EXTRACTION")
print("=" * 110)

print()
print("INPUT:", STAGE05F_PATH)
print("SIZE :", f"{W} × {H}")


# ============================================================
# LOAD BIREFNET
# ============================================================

print()
print("Loading BiRefNet...")


model = AutoModelForImageSegmentation.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    cache_dir=CACHE,
)


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


model = model.to(device)
model.eval()


MODEL_DTYPE = next(model.parameters()).dtype


print("✅ BIREFNET READY")
print("DEVICE:", device)
print("MODEL DTYPE:", MODEL_DTYPE)


# ============================================================
# PREPROCESS
# ============================================================

MODEL_SIZE = 1024


transform = transforms.Compose([

    transforms.Resize(
        (MODEL_SIZE, MODEL_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    ),
])


tensor = transform(
    image
).unsqueeze(0)


# CRITICAL FIX:
# input dtype must match BiRefNet weights
tensor = tensor.to(
    device=device,
    dtype=MODEL_DTYPE
)


print()
print(
    "INPUT TENSOR:",
    tuple(tensor.shape),
    tensor.dtype,
    tensor.device
)


# ============================================================
# INFERENCE
# ============================================================

print()
print("Running BiRefNet inference...")


with torch.inference_mode():

    output = model(
        tensor
    )


# VERIFIED STRUCTURE:
# output = [Tensor(1,1,1024,1024)]

if not isinstance(output, list):

    raise RuntimeError(
        f"Expected list output, got {type(output)}"
    )


if len(output) != 1:

    raise RuntimeError(
        f"Expected list length 1, got {len(output)}"
    )


pred = output[0]


if not torch.is_tensor(pred):

    raise RuntimeError(
        f"Expected Tensor in output[0], got {type(pred)}"
    )


print(
    "OUTPUT:",
    tuple(pred.shape),
    pred.dtype
)


# ============================================================
# LOGITS -> SOFT ALPHA
# ============================================================

pred = torch.sigmoid(
    pred
)


alpha_model = (
    pred[0, 0]
    .detach()
    .float()
    .cpu()
    .numpy()
)


alpha_model = np.clip(
    alpha_model,
    0.0,
    1.0
)


# ============================================================
# RESTORE EXACT STAGE05F DIMENSIONS
# ============================================================

alpha_img = Image.fromarray(
    (
        alpha_model
        *
        255.0
    ).astype(np.uint8)
)


alpha_img = alpha_img.resize(
    (W, H),
    Image.Resampling.LANCZOS
)


ALPHA_PATH = (
    OUT
    / "01_birefnet_soft_alpha.png"
)

alpha_img.save(
    ALPHA_PATH
)


# ============================================================
# BINARY MASK
# ============================================================

alpha_np = (
    np.asarray(
        alpha_img
    ).astype(np.float32)
    /
    255.0
)


BINARY_THRESHOLD = 0.50


binary = (
    alpha_np
    >=
    BINARY_THRESHOLD
)


BINARY_PATH = (
    OUT
    / "02_birefnet_binary_050.png"
)


Image.fromarray(
    binary.astype(np.uint8) * 255
).save(
    BINARY_PATH
)


# ============================================================
# SAME-SIZE TRANSPARENT PNG
#
# RGB = Stage05F
# ALPHA = BiRefNet
# ============================================================

rgb = np.asarray(
    image
)


rgba = np.zeros(
    (H, W, 4),
    dtype=np.uint8
)


rgba[..., :3] = rgb

rgba[..., 3] = np.asarray(
    alpha_img
)


RGBA_PATH = (
    OUT
    / "03_stage05f_birefnet_transparent.png"
)


Image.fromarray(
    rgba,
    mode="RGBA"
).save(
    RGBA_PATH
)


# ============================================================
# CHECKERBOARD PREVIEW
# ============================================================

tile = 16

yy, xx = np.indices(
    (H, W)
)


pattern = (
    (
        xx // tile
        +
        yy // tile
    )
    %
    2
)


checker_gray = np.where(
    pattern == 0,
    220,
    180
).astype(np.uint8)


checker = np.stack(
    [
        checker_gray,
        checker_gray,
        checker_gray
    ],
    axis=2
)


a = alpha_np[..., None]


preview = (
    rgb.astype(np.float32) * a
    +
    checker.astype(np.float32) * (1.0 - a)
).astype(np.uint8)


PREVIEW_PATH = (
    OUT
    / "04_birefnet_checkerboard_preview.png"
)


Image.fromarray(
    preview
).save(
    PREVIEW_PATH
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
    image
)

axes[0].set_title(
    "1. Stage05F Input"
)

axes[0].axis("off")


axes[1].imshow(
    alpha_np,
    cmap="gray",
    vmin=0,
    vmax=1
)

axes[1].set_title(
    "2. BiRefNet Soft Alpha"
)

axes[1].axis("off")


axes[2].imshow(
    binary,
    cmap="gray",
    vmin=0,
    vmax=1
)

axes[2].set_title(
    "3. Binary Alpha >= 0.50"
)

axes[2].axis("off")


axes[3].imshow(
    preview
)

axes[3].set_title(
    "4. Stage05F + BiRefNet Alpha"
)

axes[3].axis("off")


plt.tight_layout()


COMPARISON_PATH = (
    OUT
    / "05_stage06e2_v2_comparison.png"
)


plt.savefig(
    COMPARISON_PATH,
    dpi=150,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# STATS
# ============================================================

total_pixels = int(
    W * H
)

foreground_pixels = int(
    binary.sum()
)

foreground_fraction = float(
    foreground_pixels
    /
    total_pixels
)

mean_alpha = float(
    alpha_np.mean()
)


# ============================================================
# SAVE RESULT STATE
# ============================================================

STATE = {

    "stage":
        "06E2_V2",

    "approach":
        "BIREFNET_DIRECT_FULL_FOREGROUND",

    "input":
        str(STAGE05F_PATH),

    "stage01_used":
        False,

    "model":
        MODEL_ID,

    "verified_model_output":
        "list_len_1_tensor_1x1x1024x1024",

    "model_dtype":
        str(MODEL_DTYPE),

    "image_size": [
        W,
        H
    ],

    "model_input_size": [
        MODEL_SIZE,
        MODEL_SIZE
    ],

    "binary_threshold":
        BINARY_THRESHOLD,

    "statistics": {

        "total_pixels":
            total_pixels,

        "foreground_pixels":
            foreground_pixels,

        "foreground_fraction":
            foreground_fraction,

        "mean_soft_alpha":
            mean_alpha
    },

    "outputs": {

        "soft_alpha":
            str(ALPHA_PATH),

        "binary_mask":
            str(BINARY_PATH),

        "transparent_png":
            str(RGBA_PATH),

        "checkerboard_preview":
            str(PREVIEW_PATH),

        "comparison":
            str(COMPARISON_PATH)
    },

    "status":
        "REQUIRES_VISUAL_FOREGROUND_AUDIT"
}


STATE_PATH = (
    OUT
    / "00_stage06e2_v2_result.json"
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
print("PRODUCTION EXPERIMENT 06E2 V2 RESULT")
print("=" * 110)

print()
print(
    "SIZE:",
    f"{W} × {H}"
)

print(
    "MODEL DTYPE:",
    MODEL_DTYPE
)

print(
    "TOTAL PIXELS:",
    total_pixels
)

print(
    "FOREGROUND PIXELS:",
    foreground_pixels
)

print(
    "FOREGROUND FRACTION:",
    round(
        foreground_fraction,
        4
    )
)

print(
    "MEAN SOFT ALPHA:",
    round(
        mean_alpha,
        4
    )
)

print()
print(
    "SOFT ALPHA:",
    ALPHA_PATH
)

print(
    "BINARY MASK:",
    BINARY_PATH
)

print(
    "TRANSPARENT PNG:",
    RGBA_PATH
)

print(
    "CHECKERBOARD:",
    PREVIEW_PATH
)

print(
    "COMPARISON:",
    COMPARISON_PATH
)

print(
    "STATE:",
    STATE_PATH
)

print()
print(
    "NO FLORENCE / DINO / SAM2 / QWEN WAS RUN."
)


# ============================================================
# CLEANUP
# ============================================================

del model
del tensor
del output
del pred

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()
