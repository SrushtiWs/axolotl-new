
from pathlib import Path
from PIL import Image, ImageDraw
from IPython.display import display

import os
import gc
import json
import time
import hashlib

import numpy as np
import torch

from diffusers import (
    QwenImageEditPlusPipeline
)


# ============================================================
# 1. IDENTIFIERS
# ============================================================

STAGE_NAME = (
    "PRODUCTION_STAGE04B_"
    "QWEN_GLASS_RECREATION"
)

MODEL_ID = (
    "Qwen/Qwen-Image-Edit-2509"
)


# ============================================================
# 2. PATHS
# ============================================================

BASE = Path(
    "/workspace/axolotl"
)

PROJECT = (
    BASE
    / "test07"
)

PROD = (
    PROJECT
    / "production_pipeline"
)

STAGE01 = (
    PROD
    / "stage01_master"
)

STAGE04 = (
    PROD
    / "stage04_glass"
)

STAGE04.mkdir(
    parents=True,
    exist_ok=True
)


MASTER_PATH = (
    STAGE01
    / "00_master_input.png"
)


GLASS_MASK_PATH = (
    STAGE04
    / "00_glass_mask.png"
)


GLASS_REFERENCE_PATH = (
    STAGE04
    / "01_original_glass_reference_rgba.png"
)


GLASS_STATE_PATH = (
    STAGE04
    / "00_stage04_glass_state.json"
)


CACHE_DIR = Path(
    "/workspace/data/huggingface-cache"
)

CACHE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


os.environ[
    "HF_HOME"
] = str(
    CACHE_DIR
)

os.environ[
    "HUGGINGFACE_HUB_CACHE"
] = str(
    CACHE_DIR
)


# ============================================================
# 3. GENERATION SETTINGS
# ============================================================

SEED = 12345

NUM_INFERENCE_STEPS = 20

TRUE_CFG_SCALE = 4.0

DTYPE = torch.bfloat16


# ============================================================
# 4. PROMPT
# ============================================================

PROMPT = r"""
Edit this exact bathroom photograph.

ABSOLUTE PRIORITY:

Keep the exact same room, camera position, camera angle,
perspective, framing, room dimensions, wall boundaries,
floor boundaries, ceiling boundaries, object positions,
object scale, lighting and composition.

Do NOT redesign the room.

Focus ONLY on the existing shower glass partition on the
right side of the photograph.

Recreate the SAME shower glass enclosure cleanly and
realistically in exactly the same physical location and
perspective.

The shower enclosure must remain the same two-panel system.

Preserve:

- the same overall glass-panel boundaries
- the same perspective and vertical alignment
- the same main panel
- the same narrow right-side panel
- the same clear transparent upper glass character
- the same frosted/translucent lower glass character
- the same top rail/frame arrangement
- the same vertical frame members
- the same horizontal handle position
- the same enclosure scale
- the same relationship with the wall and floor

The glass must look like realistic architectural shower glass.

Do not move the enclosure.

Do not enlarge it.

Do not shrink it.

Do not add another panel.

Do not remove a panel.

Do not change the room behind it.

Do not change the shower fixtures.

Do not change the toilet.

Do not change the basin.

Do not change the vanity.

Do not change the mirror.

Do not change walls, floor or ceiling.

Do not add any new object or decoration.

The task is only to recreate the existing shower glass
partition cleanly and realistically while leaving the rest
of the photograph unchanged.
"""


# ============================================================
# 5. NEGATIVE PROMPT
# ============================================================

NEGATIVE_PROMPT = r"""
different room,
different bathroom,
different camera,
changed camera angle,
changed perspective,
zoom,
crop,
warped room,
shifted wall,
shifted floor,
shifted ceiling,
moved shower enclosure,
larger shower enclosure,
smaller shower enclosure,
extra glass panel,
missing glass panel,
different glass shape,
frameless redesign,
different handle,
different rail,
different frame,
new shower door,
new furniture,
new decoration,
new accessories,
moved toilet,
changed toilet,
moved basin,
changed basin,
changed vanity,
changed wall,
changed floor,
changed ceiling,
changed mirror,
hallucinated object,
duplicated object,
missing fixture,
redesigned bathroom
"""


# ============================================================
# 6. OUTPUT PATHS
# ============================================================

INPUT_SNAPSHOT_PATH = (
    STAGE04
    / "04b_00_qwen_input.png"
)

PROMPT_PATH = (
    STAGE04
    / "04b_01_prompt.txt"
)

NEGATIVE_PROMPT_PATH = (
    STAGE04
    / "04b_02_negative_prompt.txt"
)

RAW_QWEN_PATH = (
    STAGE04
    / "04b_03_raw_qwen_glass_recreation.png"
)

MASTER_SIZE_RAW_PATH = (
    STAGE04
    / "04b_04_raw_master_size.png"
)

MASK_LOCKED_PATH = (
    STAGE04
    / "04b_05_mask_locked_glass_recreation.png"
)

GLASS_REGION_RGBA_PATH = (
    STAGE04
    / "04b_06_ai_glass_region_reference_rgba.png"
)

COMPARISON_PATH = (
    STAGE04
    / "04b_07_original_vs_ai_glass_comparison.png"
)

REPORT_PATH = (
    STAGE04
    / "04b_00_qwen_glass_recreation_report.json"
)


# ============================================================
# 7. HELPERS
# ============================================================

def sha256_file(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:

        while True:

            chunk = f.read(
                1024 * 1024
            )

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


# ============================================================
# 8. INPUT CHECK
# ============================================================

print()
print("=" * 110)
print(STAGE_NAME)
print("=" * 110)


required = {
    "MASTER":
        MASTER_PATH,

    "GLASS MASK":
        GLASS_MASK_PATH,

    "GLASS REFERENCE":
        GLASS_REFERENCE_PATH,

    "GLASS STATE":
        GLASS_STATE_PATH,
}


print()
print("INPUT CHECK")
print("-" * 110)


for name, path in required.items():

    exists = path.exists()

    print(
        f"{name:24s}",
        "✅" if exists else "❌",
        path
    )

    if not exists:

        raise FileNotFoundError(
            path
        )


# ============================================================
# 9. LOAD MASTER
# ============================================================

original_pil = Image.open(
    MASTER_PATH
).convert(
    "RGB"
)


original = np.array(
    original_pil
)


H, W = original.shape[:2]


print()
print(
    "MASTER SIZE:",
    W,
    "x",
    H
)


# ============================================================
# 10. LOAD GLASS MASK
# ============================================================

mask_pil = Image.open(
    GLASS_MASK_PATH
).convert(
    "L"
)


if mask_pil.size != (
    W,
    H
):

    raise RuntimeError(
        "Glass mask does not match production master."
    )


glass_mask = (
    np.array(
        mask_pil
    ) > 0
)


glass_pixels = int(
    glass_mask.sum()
)


print(
    "GLASS MASK PIXELS:",
    glass_pixels
)


# ============================================================
# 11. SAVE INPUT/PROMPTS
# ============================================================

original_pil.save(
    INPUT_SNAPSHOT_PATH
)


PROMPT_PATH.write_text(
    PROMPT,
    encoding="utf-8"
)


NEGATIVE_PROMPT_PATH.write_text(
    NEGATIVE_PROMPT,
    encoding="utf-8"
)


# ============================================================
# 12. MEMORY CLEAN
# ============================================================

gc.collect()


if torch.cuda.is_available():

    torch.cuda.empty_cache()

    try:

        torch.cuda.ipc_collect()

    except Exception:

        pass


if torch.cuda.is_available():

    free_before, total_before = (
        torch.cuda.mem_get_info()
    )

    print()
    print(
        "GPU FREE BEFORE GENERATION GB:",
        round(
            free_before / 1024**3,
            2
        )
    )


# ============================================================
# 13. REUSE QWEN PIPE IF ALREADY LOADED
# ============================================================

reuse_existing_pipe = False


if "pipe" in globals():

    try:

        if isinstance(
            globals()["pipe"],
            QwenImageEditPlusPipeline
        ):

            pipe = globals()["pipe"]

            reuse_existing_pipe = True

    except Exception:

        reuse_existing_pipe = False


if reuse_existing_pipe:

    print()
    print(
        "✅ REUSING EXISTING QWEN PIPELINE"
    )

    model_load_seconds = 0.0


else:

    print()
    print("=" * 110)
    print("LOADING QWEN IMAGE EDIT")
    print("=" * 110)


    load_start = time.time()


    pipe = (
        QwenImageEditPlusPipeline
        .from_pretrained(
            MODEL_ID,
            torch_dtype=DTYPE,
            cache_dir=str(
                CACHE_DIR
            ),
        )
    )


    pipe.enable_sequential_cpu_offload()


    model_load_seconds = (
        time.time()
        -
        load_start
    )


    print()
    print(
        "✅ QWEN PIPELINE LOADED"
    )

    print(
        "MODEL LOAD SEC:",
        round(
            model_load_seconds,
            2
        )
    )


# ============================================================
# 14. FORCE LOW-VRAM MODE
# ============================================================

try:

    pipe.enable_sequential_cpu_offload()

    print(
        "✅ Sequential CPU offload active"
    )

except Exception as e:

    print(
        "CPU OFFLOAD STATUS:",
        e
    )


# ============================================================
# 15. GENERATOR
# ============================================================

generator = torch.Generator(
    device="cpu"
).manual_seed(
    SEED
)


# ============================================================
# 16. GENERATE
# ============================================================

print()
print("=" * 110)
print("STARTING GLASS RECREATION")
print("=" * 110)

print(
    "SEED :",
    SEED
)

print(
    "STEPS:",
    NUM_INFERENCE_STEPS
)

print(
    "CFG  :",
    TRUE_CFG_SCALE
)


generation_start = time.time()


with torch.inference_mode():

    result = pipe(
        image=original_pil,
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        true_cfg_scale=TRUE_CFG_SCALE,
        num_inference_steps=NUM_INFERENCE_STEPS,
        generator=generator,
        output_type="pil",
    )


generation_seconds = (
    time.time()
    -
    generation_start
)


print()
print(
    "GENERATION TIME SEC:",
    round(
        generation_seconds,
        2
    )
)


# ============================================================
# 17. RAW OUTPUT
# ============================================================

raw_qwen = (
    result
    .images[0]
    .convert("RGB")
)


raw_original_size = (
    raw_qwen.size
)


raw_qwen.save(
    RAW_QWEN_PATH
)


print(
    "RAW QWEN SIZE:",
    raw_original_size
)


# ============================================================
# 18. FORCE EXACT MASTER SIZE
# ============================================================

if raw_qwen.size != (
    W,
    H
):

    raw_master_size = raw_qwen.resize(
        (
            W,
            H
        ),
        Image.Resampling.LANCZOS
    )

else:

    raw_master_size = raw_qwen.copy()


raw_master_size.save(
    MASTER_SIZE_RAW_PATH
)


generated = np.array(
    raw_master_size
)


# ============================================================
# 19. STRICT GEOMETRY LOCK
# ============================================================
#
# Qwen is allowed to contribute pixels ONLY inside
# the verified Stage04A glass mask.
#
# Everything outside the glass mask comes EXACTLY from
# Stage01 production master.
# ============================================================

locked = original.copy()


locked[
    glass_mask
] = generated[
    glass_mask
]


Image.fromarray(
    locked
).save(
    MASK_LOCKED_PATH
)


# ============================================================
# 20. VERIFY OUTSIDE MASK EXACTNESS
# ============================================================

outside = (
    ~glass_mask
)


outside_error = np.abs(
    locked.astype(np.int16)
    -
    original.astype(np.int16)
)


max_outside_rgb_error = int(
    outside_error[
        outside
    ].max()
)


print()
print(
    "MAX RGB ERROR OUTSIDE GLASS MASK:",
    max_outside_rgb_error
)


if max_outside_rgb_error != 0:

    raise RuntimeError(
        "Geometry lock failed: pixels outside glass changed."
    )


# ============================================================
# 21. AI GLASS REGION REFERENCE RGBA
# ============================================================
#
# IMPORTANT:
# This is an APPEARANCE REFERENCE only.
#
# Its alpha currently equals the Stage04A geometry mask.
# It is NOT yet considered physically correct transparent glass.
# ============================================================

glass_rgba = np.zeros(
    (
        H,
        W,
        4
    ),
    dtype=np.uint8
)


glass_rgba[
    glass_mask,
    :3
] = generated[
    glass_mask
]


glass_rgba[
    glass_mask,
    3
] = 255


Image.fromarray(
    glass_rgba,
    mode="RGBA"
).save(
    GLASS_REGION_RGBA_PATH
)


# ============================================================
# 22. COMPARISON
# ============================================================

LABEL_H = 42


comparison = Image.new(
    "RGB",
    (
        W * 3,
        H + LABEL_H
    ),
    (
        255,
        255,
        255
    )
)


comparison.paste(
    original_pil,
    (
        0,
        LABEL_H
    )
)


comparison.paste(
    raw_master_size,
    (
        W,
        LABEL_H
    )
)


comparison.paste(
    Image.fromarray(
        locked
    ),
    (
        W * 2,
        LABEL_H
    )
)


draw = ImageDraw.Draw(
    comparison
)


for x, text in [
    (
        8,
        "ORIGINAL"
    ),
    (
        W + 8,
        "RAW QWEN"
    ),
    (
        W * 2 + 8,
        "MASK-LOCKED GLASS"
    ),
]:

    draw.text(
        (
            x,
            11
        ),
        text,
        fill=(
            0,
            0,
            0
        )
    )


comparison.save(
    COMPARISON_PATH
)


# ============================================================
# 23. REPORT
# ============================================================

report = {

    "stage":
        STAGE_NAME,

    "status":
        "GENERATED_PENDING_VISUAL_REVIEW",

    "model":
        MODEL_ID,

    "pipeline":
        "QwenImageEditPlusPipeline",

    "purpose":
        (
            "Generate a controlled AI appearance reference "
            "for the existing verified shower glass system."
        ),

    "important_limitation":
        (
            "The generated RGBA is not yet the final optical "
            "transparent glass asset because RGB inside a "
            "transparent panel may contain background appearance."
        ),

    "input": {

        "master":
            str(
                MASTER_PATH
            ),

        "master_sha256":
            sha256_file(
                MASTER_PATH
            ),

        "glass_mask":
            str(
                GLASS_MASK_PATH
            ),

        "glass_pixels":
            glass_pixels,
    },

    "generation": {

        "seed":
            SEED,

        "steps":
            NUM_INFERENCE_STEPS,

        "true_cfg_scale":
            TRUE_CFG_SCALE,

        "dtype":
            str(
                DTYPE
            ),

        "sequential_cpu_offload":
            True,

        "reused_existing_pipe":
            reuse_existing_pipe,

        "model_load_seconds":
            round(
                model_load_seconds,
                3
            ),

        "generation_seconds":
            round(
                generation_seconds,
                3
            ),

        "raw_output_size":
            list(
                raw_original_size
            ),
    },

    "verification": {

        "geometry_authority":
            "Stage04A canonical glass mask",

        "pixels_outside_mask_restored_from_master":
            True,

        "max_rgb_error_outside_glass_mask":
            max_outside_rgb_error,
    },

    "outputs": {

        "raw_qwen":
            str(
                RAW_QWEN_PATH
            ),

        "master_size_raw":
            str(
                MASTER_SIZE_RAW_PATH
            ),

        "mask_locked":
            str(
                MASK_LOCKED_PATH
            ),

        "ai_glass_region_reference_rgba":
            str(
                GLASS_REGION_RGBA_PATH
            ),

        "comparison":
            str(
                COMPARISON_PATH
            ),
    },

    "frozen_rules": [

        "Stage04A geometry remains authoritative.",

        "Qwen cannot modify pixels outside the canonical glass mask.",

        "AI glass result is currently an appearance reference.",

        "Final transparent glass must not contain baked old-room background.",

        "Final optical glass construction occurs after a clean background exists.",
    ],
}


REPORT_PATH.write_text(
    json.dumps(
        report,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# ============================================================
# 24. FINAL
# ============================================================

print()
print("=" * 110)
print("PRODUCTION STAGE 04B COMPLETE")
print("=" * 110)

print()

print(
    "RAW QWEN:",
    RAW_QWEN_PATH
)

print(
    "MASK-LOCKED:",
    MASK_LOCKED_PATH
)

print(
    "AI GLASS REFERENCE:",
    GLASS_REGION_RGBA_PATH
)

print(
    "COMPARISON:",
    COMPARISON_PATH
)

print(
    "REPORT:",
    REPORT_PATH
)


print()
print(
    "ORIGINAL vs RAW QWEN vs MASK-LOCKED"
)

display(
    comparison
)


print()
print(
    "AI GLASS REGION REFERENCE"
)

display(
    Image.open(
        GLASS_REGION_RGBA_PATH
    )
)
