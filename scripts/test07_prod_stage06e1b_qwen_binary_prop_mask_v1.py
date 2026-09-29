
from pathlib import Path
import gc
import json

import numpy as np
import torch

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

STAGE05F_PATH = (
    PROD
    / "stage05_clean_room_with_props"
    / "12_final_canonical_prop_detection_master.png"
)

OUT = (
    PROD
    / "stage06_prop_layer"
    / "06e1b_qwen_binary_prop_mask"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_ID = (
    "Qwen/Qwen-Image-Edit-2509"
)

CACHE = (
    "/workspace/data/huggingface-cache"
)


# ============================================================
# QWEN SETTINGS
# ============================================================

SEED = 12345

STEPS = 20

TRUE_CFG_SCALE = 4.0


# ============================================================
# MASK SETTINGS
# ============================================================

# White threshold.
#
# We deliberately require high brightness because the desired
# AI output is pure white props on pure black background.
WHITE_THRESHOLD = 180


# ============================================================
# VALIDATE
# ============================================================

if not STAGE05F_PATH.exists():

    raise FileNotFoundError(
        STAGE05F_PATH
    )


stage05f = Image.open(
    STAGE05F_PATH
).convert(
    "RGB"
)

W, H = stage05f.size


print("=" * 110)
print("PRODUCTION EXPERIMENT 06E1B")
print("STAGE05F -> QWEN BINARY PROP MASK")
print("=" * 110)

print()
print(
    "INPUT:",
    STAGE05F_PATH
)

print(
    "SIZE:",
    f"{W} × {H}"
)

print(
    "STAGE01 USED:",
    False
)


# ============================================================
# PROMPT
# ============================================================

PROMPT = """
Convert this room image into a precise binary foreground mask.

This is NOT a normal image edit.
This is a semantic object-isolation mask.

============================================================
OUTPUT FORMAT
============================================================

Create an image with ONLY TWO COLORS:

PURE WHITE:
RGB 255,255,255
HEX #FFFFFF

PURE BLACK:
RGB 0,0,0
HEX #000000

No gray.
No color.
No shading.
No texture.
No gradients.
No shadows.
No realistic rendering.

The output must remain exactly aligned with the input image.

Keep exactly the same:
- canvas
- framing
- viewpoint
- perspective
- object position
- object size
- object shape

Do not crop.
Do not zoom.
Do not move anything.
Do not resize anything.

============================================================
WHITE = ALL PHYSICAL PROPS TO KEEP
============================================================

Paint PURE WHITE every visible non-architectural physical
object, prop, furniture item, fixture or accessory.

Preserve complete physical objects.

If multiple non-architectural physical items:
- touch each other
- rest directly on each other
- are attached to each other
- are integrated together
- form one continuous physical assembly

paint the complete assembly white.

For this image, important retained physical objects include:

- complete vanity system
  including cabinet/body, countertop, sink/basin,
  faucet, bottle, attached drawers/doors/handles/knobs

- complete toilet system
  including body, seat and lid

- shower arm and shower head

- wall electrical plate

- wall-mounted toilet paper holder

- recessed ceiling light

Also preserve any other visible discrete physical prop that
really exists in the image.

Small and thin fixtures must not be missed.

============================================================
BLACK = REMOVE ARCHITECTURE / BACKGROUND
============================================================

Paint PURE BLACK all architectural/background regions:

- walls
- floor
- ceiling
- baseboards
- skirting
- room corners
- wall surfaces
- floor surfaces
- ceiling surfaces
- shadows belonging only to room architecture
- empty background

Do NOT restore the removed mirror.

Do NOT restore the removed glass partition.

Do NOT restore glass-enclosure hardware that was removed with
the glass system.

============================================================
CRITICAL
============================================================

Do not create photographic content.

Do not preserve original colors.

Do not generate a green background.

Do not add objects.

Do not hallucinate.

Do not turn shadows into objects.

The result must be a clean black-and-white segmentation mask:

WHITE = physical props only
BLACK = everything else
"""


# ============================================================
# LOAD QWEN
# ============================================================

print()
print(
    "Loading Qwen Image Edit..."
)


from diffusers import (
    QwenImageEditPlusPipeline,
)


pipe = (
    QwenImageEditPlusPipeline
    .from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        cache_dir=CACHE,
    )
)


pipe.enable_sequential_cpu_offload()


print(
    "✅ QWEN READY"
)


# ============================================================
# RUN
# ============================================================

generator = torch.Generator(
    device="cpu"
).manual_seed(
    SEED
)


print()
print(
    "Generating binary semantic prop mask..."
)


result = pipe(
    image=stage05f,
    prompt=PROMPT,
    num_inference_steps=STEPS,
    true_cfg_scale=TRUE_CFG_SCALE,
    generator=generator,
)


qwen_output = result.images[0].convert(
    "RGB"
)


# ============================================================
# FORCE SAME CANVAS SIZE
# ============================================================

if qwen_output.size != (
    W,
    H
):

    print(
        "QWEN OUTPUT SIZE:",
        qwen_output.size
    )

    print(
        "Returning binary reference to:",
        (
            W,
            H
        )
    )

    qwen_output = qwen_output.resize(
        (
            W,
            H
        ),
        Image.Resampling.NEAREST
    )


RAW_PATH = (
    OUT
    / "00_qwen_binary_reference_raw.png"
)


qwen_output.save(
    RAW_PATH
)


# ============================================================
# CONVERT AI REFERENCE TO GRAYSCALE
# ============================================================

qwen_np = np.asarray(
    qwen_output
).astype(
    np.float32
)


gray = (
    0.299
    *
    qwen_np[
        ...,
        0
    ]

    +

    0.587
    *
    qwen_np[
        ...,
        1
    ]

    +

    0.114
    *
    qwen_np[
        ...,
        2
    ]
)


# ============================================================
# BINARY MASK
# ============================================================

prop_mask = (
    gray
    >=
    WHITE_THRESHOLD
)


MASK_PATH = (
    OUT
    / "01_ai_binary_prop_mask.png"
)


Image.fromarray(
    prop_mask.astype(
        np.uint8
    )
    *
    255
).save(
    MASK_PATH
)


# ============================================================
# BUILD TRANSPARENT PNG USING STAGE05F RGB
#
# IMPORTANT:
# Stage01 is intentionally NOT used in this experiment.
# ============================================================

stage05f_np = np.asarray(
    stage05f
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
] = stage05f_np


rgba[
    ...,
    3
] = (
    prop_mask.astype(
        np.uint8
    )
    *
    255
)


TRANSPARENT_PATH = (
    OUT
    / "02_stage05f_props_transparent.png"
)


Image.fromarray(
    rgba,
    mode="RGBA"
).save(
    TRANSPARENT_PATH
)


# ============================================================
# CHECKERBOARD PREVIEW
# ============================================================

checker = np.zeros(
    (
        H,
        W,
        3
    ),
    dtype=np.uint8
)

tile = 16


for y in range(
    H
):

    for x in range(
        W
    ):

        value = (
            220
            if
            (
                (
                    x // tile
                    +
                    y // tile
                )
                %
                2
                ==
                0
            )
            else
            180
        )

        checker[
            y,
            x
        ] = [
            value,
            value,
            value
        ]


alpha = (
    rgba[
        ...,
        3:4
    ].astype(
        np.float32
    )
    /
    255.0
)


preview_np = (

    stage05f_np.astype(
        np.float32
    )
    *
    alpha

    +

    checker.astype(
        np.float32
    )
    *
    (
        1.0
        -
        alpha
    )
).astype(
    np.uint8
)


CHECKERBOARD_PATH = (
    OUT
    / "03_stage05f_transparent_checkerboard.png"
)


Image.fromarray(
    preview_np
).save(
    CHECKERBOARD_PATH
)


# ============================================================
# 4-PANEL AUDIT
# ============================================================

fig, axes = plt.subplots(
    1,
    4,
    figsize=(
        18,
        7
    )
)


axes[0].imshow(
    stage05f
)

axes[0].set_title(
    "1. Stage05F Input"
)

axes[0].axis(
    "off"
)


axes[1].imshow(
    qwen_output
)

axes[1].set_title(
    "2. Raw Qwen Binary Reference"
)

axes[1].axis(
    "off"
)


axes[2].imshow(
    prop_mask,
    cmap="gray",
    vmin=0,
    vmax=1
)

axes[2].set_title(
    "3. Thresholded AI Mask"
)

axes[2].axis(
    "off"
)


axes[3].imshow(
    preview_np
)

axes[3].set_title(
    "4. Stage05F RGB + AI Alpha"
)

axes[3].axis(
    "off"
)


plt.tight_layout()


COMPARISON_PATH = (
    OUT
    / "04_stage06e1b_comparison.png"
)


plt.savefig(
    COMPARISON_PATH,
    dpi=150,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# PIXEL REPORT
# ============================================================

total_pixels = int(
    W
    *
    H
)

prop_pixels = int(
    prop_mask.sum()
)

background_pixels = (
    total_pixels
    -
    prop_pixels
)

prop_fraction = float(
    prop_pixels
    /
    total_pixels
)


# ============================================================
# SAVE STATE
# ============================================================

STATE = {

    "stage":
        "06E1B",

    "approach":
        "QWEN_BINARY_SEMANTIC_PROP_MASK",

    "input":
        str(
            STAGE05F_PATH
        ),

    "stage01_used":
        False,

    "image_size": [
        W,
        H
    ],

    "model":
        MODEL_ID,

    "qwen_config": {

        "seed":
            SEED,

        "steps":
            STEPS,

        "true_cfg_scale":
            TRUE_CFG_SCALE,
    },

    "mask_threshold":
        WHITE_THRESHOLD,

    "pixel_counts": {

        "total":
            total_pixels,

        "prop":
            prop_pixels,

        "background":
            background_pixels,

        "prop_fraction":
            prop_fraction,
    },

    "outputs": {

        "raw_qwen_binary_reference":
            str(
                RAW_PATH
            ),

        "binary_prop_mask":
            str(
                MASK_PATH
            ),

        "stage05f_transparent_png":
            str(
                TRANSPARENT_PATH
            ),

        "checkerboard_preview":
            str(
                CHECKERBOARD_PATH
            ),

        "comparison":
            str(
                COMPARISON_PATH
            ),
    },

    "status":
        "REQUIRES_VISUAL_BINARY_MASK_AUDIT",
}


STATE_PATH = (
    OUT
    / "00_stage06e1b_result.json"
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
print("PRODUCTION EXPERIMENT 06E1B RESULT")
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
    "PROP PIXELS:",
    prop_pixels
)

print(
    "BACKGROUND PIXELS:",
    background_pixels
)

print(
    "PROP FRACTION:",
    round(
        prop_fraction,
        4
    )
)


print()
print(
    "RAW QWEN:",
    RAW_PATH
)

print(
    "AI MASK:",
    MASK_PATH
)

print(
    "TRANSPARENT PNG:",
    TRANSPARENT_PATH
)

print(
    "CHECKERBOARD:",
    CHECKERBOARD_PATH
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
    "NO STAGE01 RGB WAS USED."
)

print(
    "NO INDIVIDUAL PROP DETECTION WAS RUN."
)

print(
    "NO FLORENCE / DINO / SAM2 WAS RUN."
)


# ============================================================
# CLEANUP
# ============================================================

del pipe
del result

gc.collect()

if torch.cuda.is_available():

    torch.cuda.empty_cache()
