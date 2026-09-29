
from pathlib import Path
import json
import gc

import numpy as np
import torch

from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

from transformers import AutoModelForImageSegmentation


# ============================================================
# PATHS
# ============================================================

BASE = Path("/workspace/axolotl")

PROD = (
    BASE
    / "test07"
    / "production_pipeline"
)

STAGE05F_PATH = (
    PROD
    / "stage05_clean_room_with_props"
    / "12_final_canonical_prop_detection_master.png"
)

OUT = (
    PROD
    / "stage06_prop_layer"
    / "06e2_birefnet_direct_prop_extraction"
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

    raise FileNotFoundError(
        STAGE05F_PATH
    )


image = Image.open(
    STAGE05F_PATH
).convert("RGB")


W, H = image.size


print("=" * 110)
print("PRODUCTION EXPERIMENT 06E2")
print("BIREFNET DIRECT FULL-PROP EXTRACTION")
print("=" * 110)

print()
print("INPUT:", STAGE05F_PATH)
print("SIZE:", f"{W} × {H}")
print("CUDA:", torch.cuda.is_available())


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("Loading BiRefNet...")


model = AutoModelForImageSegmentation.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    cache_dir=CACHE,
)


device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


model = model.to(device)
model.eval()


print("✅ BIREFNET READY")


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
        ],
    ),
])


tensor = transform(
    image
).unsqueeze(0).to(device)


# ============================================================
# INFERENCE
# ============================================================

print()
print("Running BiRefNet inference...")


with torch.inference_mode():

    output = model(
        tensor
    )


# ============================================================
# INSPECT OUTPUT STRUCTURE
# ============================================================

print()
print("RAW OUTPUT TYPE:", type(output))


# Common BiRefNet return patterns
if isinstance(output, (list, tuple)):

    pred = output[-1]

elif isinstance(output, dict):

    if "logits" in output:
        pred = output["logits"]

    elif "pred" in output:
        pred = output["pred"]

    else:
        raise RuntimeError(
            f"Unknown BiRefNet output dict keys: {list(output.keys())}"
        )

elif hasattr(output, "logits"):

    pred = output.logits

else:

    pred = output


# Unwrap nested lists / tuples
while isinstance(pred, (list, tuple)):

    pred = pred[-1]


print(
    "PRED TYPE:",
    type(pred)
)

if hasattr(pred, "shape"):

    print(
        "PRED SHAPE:",
        tuple(pred.shape)
    )


# ============================================================
# SIGMOID -> SOFT ALPHA
# ============================================================

pred = torch.sigmoid(
    pred
)


if pred.ndim == 4:

    pred = pred[0, 0]

elif pred.ndim == 3:

    pred = pred[0]


alpha_model = (
    pred
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
# RESTORE EXACT INPUT SIZE
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
    binary.astype(np.uint8)
    *
    255
).save(
    BINARY_PATH
)


# ============================================================
# TRANSPARENT PNG
#
# RGB = Stage05F
# Alpha = BiRefNet
# ============================================================

rgb = np.asarray(
    image
)


rgba = np.zeros(
    (
        H,
        W,
        4
    ),
    dtype=np.uint8
)


rgba[
    ...,
    :3
] = rgb


rgba[
    ...,
    3
] = np.asarray(
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


checker_pattern = (
    (
        xx // tile
        +
        yy // tile
    )
    %
    2
)


checker_gray = np.where(
    checker_pattern == 0,
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


alpha3 = (
    alpha_np[
        ...,
        None
    ]
)


preview = (
    rgb.astype(np.float32)
    *
    alpha3

    +

    checker.astype(np.float32)
    *
    (
        1.0
        -
        alpha3
    )
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

axes[0].axis(
    "off"
)


axes[1].imshow(
    alpha_np,
    cmap="gray",
    vmin=0,
    vmax=1
)

axes[1].set_title(
    "2. BiRefNet Soft Alpha"
)

axes[1].axis(
    "off"
)


axes[2].imshow(
    binary,
    cmap="gray",
    vmin=0,
    vmax=1
)

axes[2].set_title(
    "3. Binary Alpha >= 0.50"
)

axes[2].axis(
    "off"
)


axes[3].imshow(
    preview
)

axes[3].set_title(
    "4. Same-size Transparent Preview"
)

axes[3].axis(
    "off"
)


plt.tight_layout()


COMPARISON_PATH = (
    OUT
    / "05_stage06e2_comparison.png"
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
    W
    *
    H
)


binary_pixels = int(
    binary.sum()
)


binary_fraction = float(
    binary_pixels
    /
    total_pixels
)


mean_soft_alpha = float(
    alpha_np.mean()
)


# ============================================================
# STATE
# ============================================================

STATE = {

    "stage":
        "06E2",

    "approach":
        "BIREFNET_DIRECT_FULL_FOREGROUND",

    "input":
        str(
            STAGE05F_PATH
        ),

    "stage01_used":
        False,

    "model":
        MODEL_ID,

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

        "binary_foreground_pixels":
            binary_pixels,

        "binary_foreground_fraction":
            binary_fraction,

        "mean_soft_alpha":
            mean_soft_alpha,
    },

    "outputs": {

        "soft_alpha":
            str(
                ALPHA_PATH
            ),

        "binary_mask":
            str(
                BINARY_PATH
            ),

        "transparent_png":
            str(
                RGBA_PATH
            ),

        "checkerboard_preview":
            str(
                PREVIEW_PATH
            ),

        "comparison":
            str(
                COMPARISON_PATH
            ),
    },

    "status":
        "REQUIRES_VISUAL_FOREGROUND_AUDIT",
}


STATE_PATH = (
    OUT
    / "00_stage06e2_result.json"
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
print("PRODUCTION EXPERIMENT 06E2 RESULT")
print("=" * 110)

print()
print(
    "SIZE:",
    f"{W} × {H}"
)

print(
    "STAGE01 USED:",
    False
)

print(
    "TOTAL PIXELS:",
    total_pixels
)

print(
    "BINARY FOREGROUND PIXELS:",
    binary_pixels
)

print(
    "BINARY FOREGROUND FRACTION:",
    round(
        binary_fraction,
        4
    )
)

print(
    "MEAN SOFT ALPHA:",
    round(
        mean_soft_alpha,
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
