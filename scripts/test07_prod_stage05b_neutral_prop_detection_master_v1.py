
from pathlib import Path
import json
import gc

import torch

from PIL import Image

import matplotlib.pyplot as plt

from diffusers import (
    QwenImageEditPlusPipeline,
)


# ============================================================
# PATHS
# ============================================================

BASE = Path("/workspace/axolotl")

PROD = (
    BASE
    / "test07"
    / "production_pipeline"
)


SOURCE = (
    PROD
    / "stage05_clean_room_with_props"
    / "07_clean_room_with_exact_props.png"
)


OUT = (
    PROD
    / "stage05_clean_room_with_props"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


OUTPUT_PATH = (
    OUT
    / "08_neutral_prop_detection_master.png"
)


STATE_PATH = (
    OUT
    / "08_neutral_prop_detection_master_state.json"
)


CACHE = (
    "/workspace/data/huggingface-cache"
)


MODEL_ID = (
    "Qwen/Qwen-Image-Edit-2509"
)


# ============================================================
# CONFIG
# ============================================================

SEED = 12345

STEPS = 20

TRUE_CFG_SCALE = 4.0


# ============================================================
# VALIDATE
# ============================================================

if not SOURCE.exists():

    raise FileNotFoundError(
        SOURCE
    )


# ============================================================
# GPU CLEANUP
# ============================================================

gc.collect()

if torch.cuda.is_available():

    torch.cuda.empty_cache()


# ============================================================
# LOAD SOURCE
# ============================================================

source = Image.open(
    SOURCE
).convert(
    "RGB"
)


print("=" * 110)
print("PRODUCTION STAGE 05B")
print("NEUTRAL PROP-DETECTION MASTER")
print("=" * 110)

print()
print(
    "SOURCE:",
    SOURCE
)

print(
    "SIZE:",
    source.size
)


# ============================================================
# PROMPT
# ============================================================

PROMPT = """
Edit this room image for object-detection preprocessing.

Preserve the exact room geometry, camera perspective,
lighting direction, and placement of every remaining physical
object and fixture.

Do not add, remove, move, resize, redesign, replace, or
rearrange any physical prop.

The mirror and glass partition are already removed. Keep them
removed.

Change ONLY architectural surface appearance:

- remove every decorative wall tile, wall pattern, stone
  pattern, wallpaper pattern, colored wall finish, and wall
  texture;
- make all wall surfaces plain, smooth, neutral off-white;
- remove every floor tile, floor pattern, wood pattern,
  colored floor finish, and decorative floor texture;
- make the entire visible floor plain, smooth, neutral light
  warm gray / light beige;
- make the ceiling plain, smooth, neutral off-white;
- do not introduce green, blue, red, or other diagnostic
  surface colors;
- do not create grout lines;
- do not create tile seams;
- do not create new architectural details.

Preserve ALL existing physical props exactly where they are,
including small fixtures and objects resting on larger props.

Examples of things that must remain unchanged if present:
furniture, cabinets, drawers, countertops, sinks, basins,
faucets, bottles, dispensers, toilets, toilet seats, toilet
paper holders, electrical plates, switches, outlets, ceiling
lights, handles, knobs, rails, appliances, plants, decor,
and every other discrete physical object.

Do not simplify props.
Do not erase small props.
Do not merge props into the background.
Do not invent new props.

The final image should look like the same exact room with all
props still present, but with plain neutral wall, floor, and
ceiling surfaces suitable for computer-vision object
detection.
"""


# ============================================================
# LOAD QWEN IMAGE EDIT
# ============================================================

print()
print(
    "Loading Qwen Image Edit..."
)


pipe = (
    QwenImageEditPlusPipeline
    .from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        cache_dir=CACHE
    )
)


pipe.enable_sequential_cpu_offload()


print(
    "✅ QWEN READY"
)


# ============================================================
# GENERATE
# ============================================================

generator = torch.Generator(
    device="cpu"
).manual_seed(
    SEED
)


print()
print(
    "Generating neutral detection master..."
)


result = pipe(

    image=source,

    prompt=PROMPT,

    generator=generator,

    num_inference_steps=STEPS,

    true_cfg_scale=TRUE_CFG_SCALE,
)


edited = result.images[0]


# ============================================================
# FORCE MASTER SIZE
# ============================================================

if edited.size != source.size:

    edited = edited.resize(
        source.size,
        Image.Resampling.LANCZOS
    )


edited.save(
    OUTPUT_PATH
)


# ============================================================
# STATE
# ============================================================

state = {

    "stage":
        "05B",

    "purpose":
        "canonical neutral prop-detection master",

    "source":
        str(
            SOURCE
        ),

    "output":
        str(
            OUTPUT_PATH
        ),

    "size":
        list(
            edited.size
        ),

    "model":
        MODEL_ID,

    "seed":
        SEED,

    "steps":
        STEPS,

    "true_cfg_scale":
        TRUE_CFG_SCALE,

    "usage_rule": {

        "stage06_detection_source":
            str(
                OUTPUT_PATH
            ),

        "final_rgb_source":
            str(
                PROD
                / "stage01_master"
                / "00_master_input.png"
            ),
    },

    "surface_rules": {

        "wall":
            "plain neutral",

        "floor":
            "plain neutral",

        "ceiling":
            "plain neutral",

        "mirror":
            "removed",

        "glass_partition":
            "removed",
    },

    "status":
        "REQUIRES_VISUAL_AUDIT",
}


STATE_PATH.write_text(
    json.dumps(
        state,
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
print("PRODUCTION STAGE 05B RESULT")
print("=" * 110)

print(
    "OUTPUT:",
    OUTPUT_PATH
)

print(
    "SIZE:",
    edited.size
)

print(
    "STATE:",
    STATE_PATH
)

print()
print(
    "STAGE06 DETECTION SOURCE:",
    OUTPUT_PATH
)

print(
    "FINAL RGB SOURCE:",
    PROD
    / "stage01_master"
    / "00_master_input.png"
)


# ============================================================
# INLINE VISUAL AUDIT
# ============================================================

print()
print("=" * 110)
print("INLINE VISUAL VERIFICATION")
print("=" * 110)


plt.figure(
    figsize=(8, 9)
)

plt.imshow(
    source
)

plt.title(
    "BEFORE — Existing Stage05"
)

plt.axis(
    "off"
)

plt.show()


plt.figure(
    figsize=(8, 9)
)

plt.imshow(
    edited
)

plt.title(
    "AFTER — Stage05B Neutral Prop-Detection Master"
)

plt.axis(
    "off"
)

plt.show()


# ============================================================
# CLEANUP
# ============================================================

del pipe

gc.collect()

if torch.cuda.is_available():

    torch.cuda.empty_cache()
