
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

STAGE05F_PATH = (
    PROD
    / "stage05_clean_room_with_props"
    / "12_final_canonical_prop_detection_master.png"
)

STAGE06_MASK_PATH = (
    PROD
    / "stage06_prop_layer"
    / "06f4_final_six_prop_layer"
    / "01_final_six_prop_union_mask.png"
)

OUT = (
    PROD
    / "stage07_empty_room"
    / "07a_qwen_empty_room"
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

SEED = 23091

STEPS = 24

TRUE_CFG_SCALE = 4.0


# ============================================================
# VALIDATE
# ============================================================

for path in [
    STAGE05F_PATH,
    STAGE06_MASK_PATH,
]:

    if not path.exists():

        raise FileNotFoundError(
            path
        )


stage05f = Image.open(
    STAGE05F_PATH
).convert("RGB")


stage06_mask = Image.open(
    STAGE06_MASK_PATH
).convert("L")


W, H = stage05f.size


if stage06_mask.size != (
    W,
    H
):

    raise RuntimeError(
        "Stage06 mask size does not match Stage05F."
    )


print("=" * 110)
print("STAGE 07A — EMPTY ROOM RECONSTRUCTION")
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
    "STAGE06 MASK:",
    STAGE06_MASK_PATH
)


# ============================================================
# PROMPT
# ============================================================

PROMPT = """
Create the exact same room after removing every remaining
physical prop and fixture from the image.
Keep the original wall, floor, ceiling, and architectural surfaces exactly as they are.
Only remove the objects and their shadows. Do NOT change surface colors or textures.

This is an empty-room reconstruction task.

============================================================
REMOVE COMPLETELY
============================================================

Remove these visible physical objects:

1. the complete vanity system:
   - cabinet/body
   - drawers/doors
   - handles
   - countertop
   - sink/basin
   - faucet
   - bottle

2. the complete toilet system:
   - toilet body
   - seat
   - lid
   - attached toilet components

3. the shower fixture:
   - shower arm
   - shower head

4. the wall electrical plate

5. the toilet paper holder

6. the recessed ceiling light

Also remove every cast shadow, contact shadow, dark patch,
reflection, or lighting artifact caused specifically by these
removed objects.

============================================================
RECONSTRUCT HIDDEN ROOM SURFACES
============================================================

Where objects were removed, reconstruct the real architectural
surface that continues behind them:

- wall
- floor
- ceiling
- skirting/baseboard
- room corner
- architectural edges

Reconstruct these surfaces naturally and continuously.

The reconstructed surfaces must match the surrounding room
geometry and material.

============================================================
CRITICAL GEOMETRY LOCK
============================================================

Keep exactly the same:

- image dimensions
- camera viewpoint
- perspective
- field of view
- room proportions
- wall positions
- floor plane
- ceiling plane
- wall-floor boundaries
- wall-ceiling boundaries
- room corners
- baseboard/skirting geometry
- doorway/opening geometry
- overall illumination direction

Do NOT crop.
Do NOT zoom.
Do NOT rotate.
Do NOT shift the camera.
Do NOT redesign the room.
Do NOT change room dimensions.

============================================================
IMPORTANT SPECIAL-LAYER RULE
============================================================

The mirror has already been removed.

Do NOT restore a mirror.

The glass partition/enclosure has already been removed.

Do NOT restore:
- glass partition
- glass door
- glass rail
- glass hardware

These remain separate production layers.

============================================================
FINAL TARGET
============================================================

The result must look like the exact same bathroom after every
physical prop has been removed.

Only architectural room structure should remain:

- walls
- floor
- ceiling
- skirting/baseboards
- openings
- architectural surfaces

No furniture.
No sanitary fixtures.
No accessories.
No electrical plate.
No toilet.
No vanity.
No sink.
No faucet.
No bottle.
No shower head.
No toilet-paper holder.
No ceiling light.
No shadows from removed props.

Do not add any new objects.
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
    "Generating Stage07 empty room..."
)


result = pipe(
    image=stage05f,
    prompt=PROMPT,
    num_inference_steps=STEPS,
    true_cfg_scale=TRUE_CFG_SCALE,
    generator=generator,
)


empty_room = result.images[0].convert(
    "RGB"
)


# ============================================================
# RESTORE CANONICAL SIZE
# ============================================================

if empty_room.size != (
    W,
    H
):

    print(
        "QWEN OUTPUT SIZE:",
        empty_room.size
    )

    print(
        "Returning to canonical size:",
        (
            W,
            H
        )
    )

    empty_room = empty_room.resize(
        (
            W,
            H
        ),
        Image.Resampling.LANCZOS
    )


# ============================================================
# SAVE
# ============================================================

EMPTY_PATH = (
    OUT
    / "00_stage07_empty_room_candidate.png"
)


empty_room.save(
    EMPTY_PATH
)


# ============================================================
# DIFFERENCE DIAGNOSTIC
#
# This is NOT yet the final shadow map.
#
# It simply shows where Stage07 differs from Stage05F.
# ============================================================

a = np.asarray(
    stage05f
).astype(
    np.float32
)


b = np.asarray(
    empty_room
).astype(
    np.float32
)


diff = np.mean(
    np.abs(
        a - b
    ),
    axis=2
)


diff_norm = np.clip(
    diff
    /
    80.0,
    0.0,
    1.0
)


DIFF_PATH = (
    OUT
    / "01_stage05f_vs_stage07_difference.png"
)


Image.fromarray(
    (
        diff_norm
        *
        255
    ).astype(
        np.uint8
    )
).save(
    DIFF_PATH
)


# ============================================================
# STAGE06 MASK OVERLAY DIAGNOSTIC
#
# Purpose:
# show which differences are expected inside removed props
# and which differences happen outside them.
# ============================================================

mask_np = (
    np.asarray(
        stage06_mask
    )
    >
    127
)


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
    "1. Stage05F — Props Present"
)

axes[0].axis(
    "off"
)


axes[1].imshow(
    empty_room
)

axes[1].set_title(
    "2. Stage07A — Empty Room Candidate"
)

axes[1].axis(
    "off"
)


axes[2].imshow(
    diff_norm,
    cmap="gray",
    vmin=0,
    vmax=1
)

axes[2].set_title(
    "3. Stage05F ↔ Stage07 Difference"
)

axes[2].axis(
    "off"
)


axes[3].imshow(
    stage05f
)


overlay = np.zeros(
    (
        H,
        W,
        4
    ),
    dtype=np.float32
)


overlay[
    mask_np,
    0
] = 1.0


overlay[
    mask_np,
    3
] = 0.45


axes[3].imshow(
    overlay
)

axes[3].set_title(
    "4. Stage06 Physical-Prop Mask"
)

axes[3].axis(
    "off"
)


plt.tight_layout()


AUDIT_PATH = (
    OUT
    / "02_stage07a_visual_audit.png"
)


plt.savefig(
    AUDIT_PATH,
    dpi=150,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# SAVE STATE
# ============================================================

STATE = {

    "stage":
        "07A",

    "approach":
        "QWEN_EMPTY_ROOM_RECONSTRUCTION",

    "input":
        str(
            STAGE05F_PATH
        ),

    "stage06_mask_reference":
        str(
            STAGE06_MASK_PATH
        ),

    "image_size": [
        W,
        H
    ],

    "qwen_model":
        MODEL_ID,

    "settings": {

        "seed":
            SEED,

        "steps":
            STEPS,

        "true_cfg_scale":
            TRUE_CFG_SCALE
    },

    "outputs": {

        "empty_room_candidate":
            str(
                EMPTY_PATH
            ),

        "difference_map":
            str(
                DIFF_PATH
            ),

        "visual_audit":
            str(
                AUDIT_PATH
            )
    },

    "status":
        "REQUIRES_STAGE07_VISUAL_AUDIT",

    "next_if_pass":
        "07B_SHADOW_MAP_EXTRACTION"
}


STATE_PATH = (
    OUT
    / "00_stage07a_result.json"
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
print("STAGE 07A RESULT")
print("=" * 110)

print()
print(
    "EMPTY ROOM:",
    EMPTY_PATH
)

print(
    "DIFFERENCE MAP:",
    DIFF_PATH
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
    "NEXT IF VISUALLY ACCEPTED:"
)

print(
    "07B — derive cast/contact shadow-strength map "
    "from Stage05F vs Stage07."
)


# ============================================================
# CLEANUP
# ============================================================

del pipe
del result

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()
