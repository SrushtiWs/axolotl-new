
from pathlib import Path
import json
import gc

import torch
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


INPUT_PATH = (
    PROD
    / "stage08_tile_application"
    / "08h2_props_over_metric_floor"
    / "02_metric_floor_plus_physical_props.png"
)


OUT = (
    PROD
    / "stage08_tile_application"
    / "08i1_ai_shadow_reference"
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
# SETTINGS
# ============================================================

SEED = 41783

STEPS = 24

TRUE_CFG_SCALE = 4.0


# ============================================================
# VALIDATE / LOAD
# ============================================================

if not INPUT_PATH.exists():

    raise FileNotFoundError(
        INPUT_PATH
    )


source = Image.open(
    INPUT_PATH
).convert("RGB")


W, H = source.size


print("=" * 110)
print("STAGE 08I1 — AI SHADOW REFERENCE")
print("=" * 110)

print()
print(
    "INPUT:",
    INPUT_PATH
)

print(
    "SIZE:",
    f"{W} × {H}"
)


# ============================================================
# PROMPT
# ============================================================

PROMPT = """
Edit this exact bathroom image by adding only physically
realistic cast shadows and contact shadows for the objects that
are already present.

This is a shadow-generation task only.

============================================================
DO NOT CHANGE ANY EXISTING RGB CONTENT EXCEPT SHADOWING
============================================================

Preserve exactly:

- the blue, white and gold floor tile artwork
- every tile position
- tile perspective
- tile scale
- grout lines
- wall screeding/skirting tile
- vanity cabinet
- countertop
- sink
- faucet
- bottle
- toilet
- shower fixture
- electrical plate
- toilet-paper holder
- ceiling light
- walls
- ceiling
- room geometry
- camera
- framing

Do not redesign, repaint, move, resize, remove or regenerate
any object.

============================================================
ADD REALISTIC SHADOWS
============================================================

Add physically plausible shadows caused by the existing
objects and the existing room light.

Highest priority:

1. complete vanity system:
   add realistic contact shadow where the vanity meets or
   approaches surrounding surfaces, and its cast shadow on
   the tiled floor where physically visible.

2. complete toilet:
   add a realistic soft contact/cast shadow on the tiled floor
   underneath and beside the toilet.

Also preserve subtle physically plausible contact shading for
the smaller fixtures where visible.

============================================================
SHADOW BEHAVIOR
============================================================

The shadows must:

- follow the existing lighting direction
- remain translucent
- allow the blue/white/gold tile artwork to remain visible
  underneath
- have natural soft falloff
- be darker near physical contact
- become softer farther from the object
- follow the floor perspective
- not become solid black shapes
- not look like generic graphic drop shadows

Do not add random dark patches.

Do not shadow areas where no object could physically cast a
shadow.

============================================================
SPECIAL LAYERS
============================================================

Do NOT add a mirror.

Do NOT add glass partition or glass hardware.

Those are separate production layers.

============================================================
FINAL TARGET
============================================================

Return the exact same image with the only meaningful change
being realistic cast/contact shadows from the existing props.
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
    "Generating shadow-reference image..."
)


result = pipe(
    image=source,
    prompt=PROMPT,
    num_inference_steps=STEPS,
    true_cfg_scale=TRUE_CFG_SCALE,
    generator=generator,
)


shadow_reference = (
    result.images[0]
    .convert("RGB")
)


if shadow_reference.size != (
    W,
    H
):

    print(
        "QWEN OUTPUT SIZE:",
        shadow_reference.size
    )

    shadow_reference = (
        shadow_reference.resize(
            (
                W,
                H
            ),
            Image.Resampling.LANCZOS
        )
    )


# ============================================================
# SAVE REFERENCE
# ============================================================

REFERENCE_PATH = (
    OUT
    / "00_ai_shadow_reference.png"
)


shadow_reference.save(
    REFERENCE_PATH
)


# ============================================================
# DIFFERENCE DIAGNOSTICS
#
# This is not yet the production shadow map.
# ============================================================

base_np = np.asarray(
    source
).astype(
    np.float32
)


ref_np = np.asarray(
    shadow_reference
).astype(
    np.float32
)


abs_diff = np.mean(
    np.abs(
        base_np
        -
        ref_np
    ),
    axis=2
)


diff_vis = np.clip(
    abs_diff
    /
    70.0,
    0.0,
    1.0
)


DIFF_PATH = (
    OUT
    / "01_ai_reference_absolute_difference.png"
)


Image.fromarray(
    (
        diff_vis
        *
        255
    ).astype(
        np.uint8
    )
).save(
    DIFF_PATH
)


# ============================================================
# DARKENING-ONLY DIAGNOSTIC
# ============================================================

def luminance(rgb):

    return (
        0.2126 * rgb[..., 0]
        +
        0.7152 * rgb[..., 1]
        +
        0.0722 * rgb[..., 2]
    )


base_luma = luminance(
    base_np
)


ref_luma = luminance(
    ref_np
)


darkening = np.clip(
    base_luma
    -
    ref_luma,
    0.0,
    None
)


darkening_vis = np.clip(
    darkening
    /
    60.0,
    0.0,
    1.0
)


DARK_PATH = (
    OUT
    / "02_ai_reference_darkening_only.png"
)


Image.fromarray(
    (
        darkening_vis
        *
        255
    ).astype(
        np.uint8
    )
).save(
    DARK_PATH
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
    source
)

axes[0].set_title(
    "1. 08H2\nExact Composite / No New Shadows"
)

axes[0].axis(
    "off"
)


axes[1].imshow(
    shadow_reference
)

axes[1].set_title(
    "2. 08I1\nAI Shadow Reference"
)

axes[1].axis(
    "off"
)


axes[2].imshow(
    diff_vis,
    cmap="gray",
    vmin=0,
    vmax=1
)

axes[2].set_title(
    "3. Absolute Difference"
)

axes[2].axis(
    "off"
)


axes[3].imshow(
    darkening_vis,
    cmap="gray",
    vmin=0,
    vmax=1
)

axes[3].set_title(
    "4. Darkening-Only Difference"
)

axes[3].axis(
    "off"
)


plt.tight_layout()


AUDIT_PATH = (
    OUT
    / "03_stage08i1_shadow_reference_audit.png"
)


plt.savefig(
    AUDIT_PATH,
    dpi=150,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# STATE
# ============================================================

STATE = {

    "stage":
        "08I1",

    "purpose":
        "AI_SHADOW_REFERENCE_ONLY",

    "input":
        str(
            INPUT_PATH
        ),

    "model":
        MODEL_ID,

    "settings": {

        "seed":
            SEED,

        "steps":
            STEPS,

        "true_cfg_scale":
            TRUE_CFG_SCALE
    },

    "important_rule":
        (
            "The AI output is NOT accepted as final RGB. "
            "It is only evidence for deriving a shadow "
            "multiplier over the exact 08H2 image."
        ),

    "outputs": {

        "shadow_reference":
            str(
                REFERENCE_PATH
            ),

        "absolute_difference":
            str(
                DIFF_PATH
            ),

        "darkening_only":
            str(
                DARK_PATH
            ),

        "audit":
            str(
                AUDIT_PATH
            )
    },

    "status":
        "REQUIRES_AI_SHADOW_REFERENCE_VISUAL_AUDIT",

    "next_if_pass":
        "08I2_EXTRACT_LOCAL_SHADOW_MULTIPLIER"
}


STATE_PATH = (
    OUT
    / "00_stage08i1_result.json"
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
print("STAGE 08I1 RESULT")
print("=" * 110)

print()

print(
    "SHADOW REFERENCE:",
    REFERENCE_PATH
)

print(
    "DARKENING DIAGNOSTIC:",
    DARK_PATH
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
    "QWEN RGB HAS NOT BEEN ACCEPTED AS FINAL RGB."
)

print(
    "NEXT STEP, IF SHADOWS LOOK USEFUL:"
)

print(
    "08I2 — derive local multiplier and apply it to exact 08H2."
)


# ============================================================
# CLEANUP
# ============================================================

del pipe
del result

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()
