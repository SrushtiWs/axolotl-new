
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

STAGE05F = (
    PROD
    / "stage05_clean_room_with_props"
    / "12_final_canonical_prop_detection_master.png"
)

STAGE07A = (
    PROD
    / "stage07_empty_room"
    / "07a_qwen_empty_room"
    / "00_stage07_empty_room_candidate.png"
)

OUT = (
    PROD
    / "stage07_empty_room"
    / "07c1_qwen_shadow_preserved_empty"
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


SEED = 31427
STEPS = 24
TRUE_CFG_SCALE = 4.0


# ============================================================
# VALIDATE
# ============================================================

for p in [
    STAGE05F,
    STAGE07A,
]:
    if not p.exists():
        raise FileNotFoundError(p)


source = Image.open(
    STAGE05F
).convert("RGB")

clean_empty = Image.open(
    STAGE07A
).convert("RGB")


W, H = source.size


if clean_empty.size != (
    W,
    H
):
    raise RuntimeError(
        "Stage05F / Stage07A size mismatch"
    )


print("=" * 110)
print("STAGE 07C1 — QWEN SHADOW-PRESERVED EMPTY ROOM")
print("=" * 110)

print()
print(
    "INPUT:",
    STAGE05F
)

print(
    "SIZE:",
    f"{W} × {H}"
)


# ============================================================
# PROMPT
# ============================================================

PROMPT = """
Edit this exact bathroom image.

The goal is to create the same empty room after removing the
physical objects, BUT preserve the visible cast shadows and
contact shadows originally produced by those objects.

============================================================
REMOVE ONLY THE PHYSICAL OBJECTS
============================================================

Remove completely:

1. complete vanity system:
   cabinet/body,
   drawers/doors,
   handles,
   countertop,
   sink/basin,
   faucet,
   bottle

2. complete toilet system:
   toilet body,
   seat,
   lid,
   attached components

3. shower arm and shower head

4. wall electrical plate

5. toilet-paper holder

6. recessed ceiling light

============================================================
VERY IMPORTANT — PRESERVE SHADOWS
============================================================

Do NOT erase the visible cast shadows or contact shadows that
those removed objects created on the room surfaces.

Preserve the original shadow:

- position
- direction
- shape
- softness
- darkness gradient
- contact region
- falloff

Especially preserve:

- the vanity cast/contact shadow visible on the floor
- the toilet cast/contact shadow visible on the floor

The physical objects themselves must disappear, but their
existing visible shadows must remain naturally on the room
surfaces.

Do NOT invent new shadows.

Do NOT convert shadows into solid black shapes.

Do NOT move shadows.

============================================================
RECONSTRUCT ONLY THE OCCLUDED SURFACE
============================================================

Where an opaque object physically covered the wall or floor,
reconstruct the architectural surface behind the object.

But wherever its original visible cast shadow exists outside
the physical object silhouette, preserve that shadow.

============================================================
GEOMETRY LOCK
============================================================

Keep exactly the same:

- image dimensions
- framing
- camera position
- perspective
- field of view
- wall geometry
- floor geometry
- ceiling geometry
- room corners
- baseboards
- openings
- lighting direction

Do not crop.
Do not zoom.
Do not shift the camera.
Do not redesign the room.

============================================================
SPECIAL LAYERS
============================================================

The mirror was already removed.
Do NOT restore a mirror.

The glass partition was already removed.
Do NOT restore:
- glass
- glass door
- glass rail
- glass hardware

============================================================
FINAL RESULT
============================================================

The room must contain no physical props.

It should look like the props became invisible while the real
cast/contact shadows they originally produced remain visible
at their exact original locations.

Do not add anything else.
"""


# ============================================================
# LOAD QWEN
# ============================================================

print()
print("Loading Qwen Image Edit...")


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


print("✅ QWEN READY")


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
    "Generating shadow-preserved empty-room candidate..."
)


result = pipe(
    image=source,
    prompt=PROMPT,
    num_inference_steps=STEPS,
    true_cfg_scale=TRUE_CFG_SCALE,
    generator=generator,
)


shadow_empty = result.images[0].convert(
    "RGB"
)


if shadow_empty.size != (
    W,
    H
):

    print(
        "QWEN SIZE:",
        shadow_empty.size
    )

    shadow_empty = shadow_empty.resize(
        (
            W,
            H
        ),
        Image.Resampling.LANCZOS
    )


# ============================================================
# SAVE
# ============================================================

SHADOW_EMPTY_PATH = (
    OUT
    / "00_shadow_preserved_empty_room_candidate.png"
)


shadow_empty.save(
    SHADOW_EMPTY_PATH
)


# ============================================================
# DIFFERENCE VS CLEAN STAGE07
#
# Diagnostic only.
# If Qwen preserved useful shadows, they should appear here.
# ============================================================

shadow_np = np.asarray(
    shadow_empty
).astype(np.float32)


clean_np = np.asarray(
    clean_empty
).astype(np.float32)


# luminance
def luminance(rgb):

    return (
        0.2126 * rgb[..., 0]
        +
        0.7152 * rgb[..., 1]
        +
        0.0722 * rgb[..., 2]
    )


lum_shadow = luminance(
    shadow_np
)

lum_clean = luminance(
    clean_np
)


darkening = np.clip(
    lum_clean
    -
    lum_shadow,
    0,
    None
)


darkening_vis = np.clip(
    darkening
    /
    60.0,
    0.0,
    1.0
)


DIFF_PATH = (
    OUT
    / "01_shadow_candidate_darkening_vs_stage07.png"
)


Image.fromarray(
    (
        darkening_vis
        *
        255
    ).astype(np.uint8)
).save(
    DIFF_PATH
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
    "1. Stage05F\nProps + Real Shadows"
)

axes[0].axis(
    "off"
)


axes[1].imshow(
    clean_empty
)

axes[1].set_title(
    "2. Stage07A\nClean Empty Room"
)

axes[1].axis(
    "off"
)


axes[2].imshow(
    shadow_empty
)

axes[2].set_title(
    "3. 07C1\nAI Shadow-Preserved Empty"
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
    "4. 07C1 vs Stage07A\nDarkening Difference"
)

axes[3].axis(
    "off"
)


plt.tight_layout()


AUDIT_PATH = (
    OUT
    / "02_stage07c1_audit.png"
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
        "07C1",

    "approach":
        "QWEN_REMOVE_PROPS_PRESERVE_CAST_SHADOWS",

    "input":
        str(
            STAGE05F
        ),

    "clean_empty_reference":
        str(
            STAGE07A
        ),

    "image_size": [
        W,
        H
    ],

    "settings": {

        "seed":
            SEED,

        "steps":
            STEPS,

        "true_cfg_scale":
            TRUE_CFG_SCALE
    },

    "outputs": {

        "shadow_preserved_empty":
            str(
                SHADOW_EMPTY_PATH
            ),

        "darkening_difference":
            str(
                DIFF_PATH
            ),

        "audit":
            str(
                AUDIT_PATH
            )
    },

    "status":
        "REQUIRES_VISUAL_SHADOW_PRESERVATION_AUDIT"
}


STATE_PATH = (
    OUT
    / "00_stage07c1_result.json"
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
print("=" * 110)
print("STAGE 07C1 RESULT")
print("=" * 110)

print()
print(
    "SHADOW-PRESERVED EMPTY:",
    SHADOW_EMPTY_PATH
)

print(
    "DARKENING DIFF:",
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


# ============================================================
# CLEANUP
# ============================================================

del pipe
del result

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()
