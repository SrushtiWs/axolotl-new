
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


# ------------------------------------------------------------
# AI segmentation / semantic-isolation input
# ------------------------------------------------------------

DETECTION_MASTER_PATH = (
    PROD
    / "stage05_clean_room_with_props"
    / "12_final_canonical_prop_detection_master.png"
)


# ------------------------------------------------------------
# FINAL exact RGB source
# ------------------------------------------------------------

STAGE01_MASTER_PATH = (
    PROD
    / "stage01_master"
    / "00_master_input.png"
)


OUT = (
    PROD
    / "stage06_prop_layer"
    / "06e1_qwen_full_prop_isolation"
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
# CHROMA BACKGROUND
#
# Current experiment only.
#
# Qwen should replace ALL architecture/background pixels with
# a highly saturated artificial green.
# ============================================================

CHROMA_RGB = np.array(
    [0, 255, 0],
    dtype=np.int16
)


# ------------------------------------------------------------
# Chroma distance threshold.
#
# Qwen may not generate mathematically exact RGB 0,255,0
# around anti-aliased boundaries, therefore use tolerance.
# ------------------------------------------------------------

CHROMA_DISTANCE_THRESHOLD = 115.0


# ============================================================
# VALIDATE
# ============================================================

for path in [
    DETECTION_MASTER_PATH,
    STAGE01_MASTER_PATH,
]:

    if not path.exists():

        raise FileNotFoundError(
            path
        )


detection_master = Image.open(
    DETECTION_MASTER_PATH
).convert(
    "RGB"
)


stage01_master = Image.open(
    STAGE01_MASTER_PATH
).convert(
    "RGB"
)


if detection_master.size != stage01_master.size:

    raise RuntimeError(
        (
            "Stage05F and Stage01 dimensions differ: "
            f"{detection_master.size} vs "
            f"{stage01_master.size}"
        )
    )


W, H = stage01_master.size


print("=" * 110)
print("PRODUCTION EXPERIMENT 06E1")
print("QWEN AI-FIRST FULL PROP ISOLATION")
print("=" * 110)

print()
print(
    "IMAGE SIZE:",
    f"{W} × {H}"
)

print(
    "DETECTION INPUT:",
    DETECTION_MASTER_PATH
)

print(
    "EXACT RGB SOURCE:",
    STAGE01_MASTER_PATH
)


# ============================================================
# PROMPT
# ============================================================

PROMPT = """
Create a precise foreground-object isolation image from this room image.

CRITICAL GEOMETRY RULES:

Keep the exact same:
- canvas size
- camera viewpoint
- perspective
- framing
- object position
- object size
- object orientation
- object shape

Do not crop.
Do not zoom.
Do not move anything.
Do not resize anything.
Do not rearrange anything.

============================================================
KEEP
============================================================

Preserve every visible non-architectural physical prop,
fixture, furniture item, appliance, accessory, and object.

Preserve the COMPLETE visible physical object.

If multiple non-architectural items physically:
- touch each other,
- rest directly on each other,
- attach to each other,
- are integrated together,
- or form one continuous physical assembly,

preserve the complete connected assembly.

For example, if a vanity contains a cabinet, countertop,
basin, faucet, bottle, drawers and handles that physically
touch or belong to the same visible assembly, preserve all of
them.

Preserve small and thin fixtures too.

Examples include:
- furniture
- cabinets
- counters
- sinks/basins
- faucets
- bottles
- toilets
- toilet seats/lids
- wall-mounted fixtures
- electrical plates
- ceiling lights
- shower heads and arms
- handles
- knobs
- mounted accessories

============================================================
REMOVE
============================================================

Remove ALL architectural/background surfaces and room
structure, including:

- walls
- wall finish
- wall tiles
- floor
- floor tiles
- ceiling
- baseboards
- skirting
- room corners
- structural planes
- shadows belonging only to removed room surfaces

The mirror has already been removed.
Do not recreate it.

The glass partition/enclosure has already been removed.
Do not recreate it.

Do not recreate glass hardware that belongs to the removed
glass enclosure.

============================================================
BACKGROUND
============================================================

Every removed/background pixel must become one flat,
perfectly uniform, solid chroma green:

RGB 0, 255, 0
HEX #00FF00

No texture.
No gradient.
No shadows.
No lighting variation.
No reflections.
No background details.

============================================================
PRESERVATION
============================================================

Do not add new objects.

Do not hallucinate missing objects.

Do not stylize.

Do not redesign.

Do not beautify.

Do not replace props.

Do not change colors or materials of retained props.

Do not modify their placement.

The output must look like the original props remain in their
original pixel locations while the entire room/background has
been replaced with flat #00FF00 green.
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
# RUN QWEN
# ============================================================

generator = torch.Generator(
    device="cpu"
).manual_seed(
    SEED
)


print()
print(
    "Generating AI prop-isolation reference..."
)


result = pipe(

    image=detection_master,

    prompt=PROMPT,

    num_inference_steps=STEPS,

    true_cfg_scale=TRUE_CFG_SCALE,

    generator=generator,
)


qwen_image = result.images[0].convert(
    "RGB"
)


# ============================================================
# FORCE BACK TO MASTER SIZE
#
# Qwen may output a model-native size.
# We require exact Stage01 coordinates.
#
# This resize does NOT define object semantics.
# It only returns AI reference to canonical master dimensions.
# ============================================================

if qwen_image.size != (
    W,
    H
):

    print(
        "QWEN OUTPUT SIZE:",
        qwen_image.size
    )

    print(
        "Resizing AI semantic reference back to:",
        (
            W,
            H
        )
    )


    qwen_image = qwen_image.resize(
        (
            W,
            H
        ),
        Image.Resampling.LANCZOS
    )


RAW_QWEN_PATH = (
    OUT
    / "00_qwen_chroma_prop_isolation_raw.png"
)


qwen_image.save(
    RAW_QWEN_PATH
)


# ============================================================
# CHROMA → ALPHA
# ============================================================

qwen_np = np.asarray(
    qwen_image
).astype(
    np.int16
)


# Euclidean RGB distance from perfect chroma green.
distance = np.sqrt(
    np.sum(
        (
            qwen_np
            -
            CHROMA_RGB.reshape(
                1,
                1,
                3
            )
        ) ** 2,
        axis=2
    )
)


# ------------------------------------------------------------
# Close enough to chroma green = background.
# Everything else = potential retained prop.
# ------------------------------------------------------------

background_mask = (
    distance
    <=
    CHROMA_DISTANCE_THRESHOLD
)


prop_mask = (
    ~background_mask
)


# ============================================================
# SAVE RAW AI-DERIVED MASK
# ============================================================

MASK_PATH = (
    OUT
    / "01_ai_prop_mask.png"
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
# CREATE EXACT-RGB TRANSPARENT PNG
#
# RGB is NEVER taken from Qwen.
#
# Only alpha comes from AI semantic isolation.
# ============================================================

stage01_np = np.asarray(
    stage01_master
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
] = stage01_np


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


FINAL_RGBA_PATH = (
    OUT
    / "02_exact_rgb_props_transparent.png"
)


Image.fromarray(
    rgba,
    mode="RGBA"
).save(
    FINAL_RGBA_PATH
)


# ============================================================
# TRANSPARENCY CHECKERBOARD PREVIEW
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

        if (
            (
                x // tile
                +
                y // tile
            )
            %
            2
            ==
            0
        ):

            checker[
                y,
                x
            ] = [
                220,
                220,
                220
            ]

        else:

            checker[
                y,
                x
            ] = [
                180,
                180,
                180
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

    stage01_np.astype(
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


PREVIEW_PATH = (
    OUT
    / "03_transparent_props_checkerboard_preview.png"
)


Image.fromarray(
    preview_np
).save(
    PREVIEW_PATH
)


# ============================================================
# SIDE-BY-SIDE AUDIT
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(
        15,
        7
    )
)


axes[0].imshow(
    detection_master
)

axes[0].set_title(
    "Stage05F Input"
)

axes[0].axis(
    "off"
)


axes[1].imshow(
    qwen_image
)

axes[1].set_title(
    "Qwen Chroma Isolation"
)

axes[1].axis(
    "off"
)


axes[2].imshow(
    preview_np
)

axes[2].set_title(
    "Exact Stage01 RGB + AI Alpha"
)

axes[2].axis(
    "off"
)


plt.tight_layout()


COMPARISON_PATH = (
    OUT
    / "04_ai_isolation_comparison.png"
)


plt.savefig(
    COMPARISON_PATH,
    dpi=150,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# RESULT STATE
# ============================================================

prop_pixels = int(
    prop_mask.sum()
)

background_pixels = int(
    background_mask.sum()
)

total_pixels = int(
    W
    *
    H
)


FINAL_STATE = {

    "stage":
        "06E1",

    "approach":
        "QWEN_AI_FIRST_FULL_PROP_ISOLATION",

    "input_detection_master":
        str(
            DETECTION_MASTER_PATH
        ),

    "exact_rgb_source":
        str(
            STAGE01_MASTER_PATH
        ),

    "image_size": [
        W,
        H
    ],

    "qwen_model":
        MODEL_ID,

    "qwen_config": {

        "seed":
            SEED,

        "steps":
            STEPS,

        "true_cfg_scale":
            TRUE_CFG_SCALE,
    },

    "chroma_background": {

        "rgb": [
            0,
            255,
            0
        ],

        "distance_threshold":
            CHROMA_DISTANCE_THRESHOLD,
    },

    "pixel_counts": {

        "total":
            total_pixels,

        "prop":
            prop_pixels,

        "background":
            background_pixels,

        "prop_fraction":
            float(
                prop_pixels
                /
                total_pixels
            ),
    },

    "outputs": {

        "raw_qwen_chroma":
            str(
                RAW_QWEN_PATH
            ),

        "ai_prop_mask":
            str(
                MASK_PATH
            ),

        "exact_rgb_transparent_png":
            str(
                FINAL_RGBA_PATH
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

    "important_rule":
        (
            "Qwen generated RGB is never used in the final "
            "prop layer. Only AI-derived alpha is used. "
            "Final prop RGB comes from Stage01 master."
        ),

    "status":
        "REQUIRES_VISUAL_AI_ISOLATION_AUDIT",
}


STATE_PATH = (
    OUT
    / "00_stage06e1_result.json"
)


STATE_PATH.write_text(
    json.dumps(
        FINAL_STATE,
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
print("PRODUCTION EXPERIMENT 06E1 RESULT")
print("=" * 110)

print()
print(
    "SIZE:",
    f"{W} × {H}"
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
        prop_pixels
        /
        total_pixels,
        4
    )
)


print()
print(
    "RAW QWEN:",
    RAW_QWEN_PATH
)

print(
    "AI MASK:",
    MASK_PATH
)

print(
    "TRANSPARENT PNG:",
    FINAL_RGBA_PATH
)

print(
    "CHECKERBOARD PREVIEW:",
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
    "NO INDIVIDUAL PROP DETECTION WAS RUN."
)

print(
    "NO FLORENCE / DINO / SAM2 WAS RUN."
)

print(
    "FINAL RGB COMES FROM STAGE01."
)


# ============================================================
# CLEANUP
# ============================================================

del pipe
del result

gc.collect()

if torch.cuda.is_available():

    torch.cuda.empty_cache()
