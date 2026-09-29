
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
    "PRODUCTION_STAGE05_"
    "CLEAN_ROOM_WITH_PROPS"
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

STAGE03 = (
    PROD
    / "stage03_mirror"
)

STAGE04 = (
    PROD
    / "stage04_glass"
)

STAGE05 = (
    PROD
    / "stage05_clean_room_with_props"
)

STAGE05.mkdir(
    parents=True,
    exist_ok=True
)


MASTER_PATH = (
    STAGE01
    / "00_master_input.png"
)


MIRROR_MASK_PATH = (
    STAGE03
    / "00_mirror_mask.png"
)


GLASS_MASK_PATH = (
    STAGE04
    / "00_glass_mask.png"
)


# ------------------------------------------------------------
# Existing verified Room03 physical-prop protection
# ------------------------------------------------------------

FROZEN_PROP_MASK_PATH = (
    PROJECT
    / "runs"
    / "room03_bathroom"
    / "stages"
    / "01y3_frozen_canonical_prop_protection"
    / "00_canonical_prop_protection_mask.png"
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
Edit this exact room photograph into a clean architectural
detection room.

ABSOLUTE PRIORITY:

Keep the exact same room, camera position, camera angle,
perspective, framing, room geometry, wall boundaries,
floor boundaries, ceiling boundaries, object positions,
object size, scale and lighting.

DO NOT redesign the room.

Make ONLY these changes:

1. REMOVE the wall mirror completely.
   Remove the mirror itself, its frame and all reflections.
   Reconstruct the real wall naturally behind it.

2. REMOVE the complete shower glass partition.
   Remove all clear glass, frosted glass, panels, rails,
   frames, handles, hinges and glass-partition hardware.
   Preserve all genuine physical shower fixtures and other
   real objects that are behind or near the glass.
   Reconstruct only the hidden wall, floor and ceiling.

3. ALL visible architectural WALL surfaces must become
   plain, smooth, uniform, matte WHITE.
   Remove tile designs, patterns, textures, grout, joints,
   decorative finishes and stone appearance from walls.

4. ALL visible CEILING surfaces must become
   plain, smooth, uniform, matte WHITE.
   Preserve genuine ceiling fixtures and lights.

5. ALL visible FLOOR surfaces must become
   plain, smooth, uniform solid GREEN #00C800.
   Remove all floor tile patterns, texture, joints and grout.

6. PRESERVE EVERY genuine physical object in the room
   except the mirror and shower glass partition.

Preserve exactly:

- toilet
- basin
- vanity
- cabinet
- countertop
- faucet
- showerhead
- towel bars
- switches
- outlets
- bottles
- dispensers
- lights
- door
- handles
- plumbing fixtures
- all other genuine physical props

Do not move any object.

Do not resize any object.

Do not add anything.

Do not remove any genuine object except the mirror
and shower glass partition.

The final image must look like the SAME real bathroom
with simplified architectural surfaces and all genuine
props preserved in exactly the same position.
"""


# ============================================================
# 5. NEGATIVE PROMPT
# ============================================================

NEGATIVE_PROMPT = r"""
different room,
different camera,
changed camera angle,
changed perspective,
zoom,
crop,
warped geometry,
shifted walls,
shifted corners,
shifted floor,
shifted ceiling,
moved object,
resized object,
missing object,
duplicated object,
new furniture,
new shelf,
new picture,
new frame,
new decoration,
new accessory,
hallucinated object,
redesigned bathroom,
changed vanity,
changed toilet,
changed basin,
changed cabinet,
changed countertop,
changed faucet,
changed showerhead,
changed switch,
changed outlet,
changed door,
green walls,
green ceiling,
green furniture,
green cabinet,
green fixture,
white furniture,
white toilet,
white vanity replacement,
new mirror,
remaining mirror,
mirror reflection,
remaining shower glass,
remaining frosted glass,
remaining glass frame,
remaining glass handle,
floor tile pattern,
wall tile pattern,
grout,
decorative wall texture
"""


# ============================================================
# 6. OUTPUT PATHS
# ============================================================

INPUT_SNAPSHOT_PATH = (
    STAGE05
    / "00_input_master.png"
)

PROP_MASK_SNAPSHOT_PATH = (
    STAGE05
    / "01_protected_props_mask.png"
)

EDITABLE_MASK_PATH = (
    STAGE05
    / "02_editable_architecture_mask.png"
)

PROMPT_PATH = (
    STAGE05
    / "03_prompt.txt"
)

NEGATIVE_PROMPT_PATH = (
    STAGE05
    / "04_negative_prompt.txt"
)

RAW_QWEN_PATH = (
    STAGE05
    / "05_raw_qwen.png"
)

RAW_MASTER_SIZE_PATH = (
    STAGE05
    / "06_raw_qwen_master_size.png"
)

SAFE_OUTPUT_PATH = (
    STAGE05
    / "07_clean_room_with_exact_props.png"
)

PROP_LAYER_PATH = (
    STAGE05
    / "08_original_protected_props_reference_rgba.png"
)

COMPARISON_PATH = (
    STAGE05
    / "09_original_vs_raw_vs_safe.png"
)

REPORT_PATH = (
    STAGE05
    / "00_stage05_clean_room_report.json"
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


def load_mask(path, expected_size):

    img = Image.open(
        path
    ).convert("L")

    if img.size != expected_size:

        raise RuntimeError(
            f"Mask coordinate mismatch: {path}\n"
            f"Expected {expected_size}, got {img.size}"
        )

    return (
        np.array(img) > 0
    )


# ============================================================
# 8. START
# ============================================================

print()
print("=" * 110)
print(STAGE_NAME)
print("=" * 110)


required = {

    "MASTER":
        MASTER_PATH,

    "MIRROR MASK":
        MIRROR_MASK_PATH,

    "GLASS MASK":
        GLASS_MASK_PATH,

    "PROP PROTECTION":
        FROZEN_PROP_MASK_PATH,
}


print()
print("INPUT CHECK")
print("-" * 110)


for name, path in required.items():

    ok = path.exists()

    print(
        f"{name:24s}",
        "✅" if ok else "❌",
        path
    )

    if not ok:

        raise FileNotFoundError(
            path
        )


# ============================================================
# 9. LOAD MASTER
# ============================================================

original_pil = Image.open(
    MASTER_PATH
).convert("RGB")

original = np.array(
    original_pil
)

H, W = original.shape[:2]

SIZE = (
    W,
    H
)


print()
print(
    "MASTER SIZE:",
    W,
    "x",
    H
)


# ============================================================
# 10. LOAD MASKS
# ============================================================

mirror_mask = load_mask(
    MIRROR_MASK_PATH,
    SIZE
)

glass_mask = load_mask(
    GLASS_MASK_PATH,
    SIZE
)

prop_mask = load_mask(
    FROZEN_PROP_MASK_PATH,
    SIZE
)


# ------------------------------------------------------------
# Critical safety:
# mirror/glass MUST NOT be reintroduced through prop restore.
# ------------------------------------------------------------

special_layer_mask = (
    mirror_mask
    |
    glass_mask
)


safe_prop_mask = (
    prop_mask
    &
    ~special_layer_mask
)


protected_pixels = int(
    safe_prop_mask.sum()
)


print()
print(
    "RAW PROP MASK PIXELS:",
    int(
        prop_mask.sum()
    )
)

print(
    "SAFE PROTECTED PROP PIXELS:",
    protected_pixels
)

print(
    "MIRROR PIXELS:",
    int(
        mirror_mask.sum()
    )
)

print(
    "GLASS PIXELS:",
    int(
        glass_mask.sum()
    )
)


# ============================================================
# 11. EDITABLE ARCHITECTURE MASK
# ============================================================
#
# Everything not explicitly protected as a genuine prop
# is allowed to be reconstructed by Qwen.
#
# This is diagnostic metadata only; Qwen Image Edit does not
# consume this mask directly in this pipeline.
# ============================================================

editable_mask = (
    ~safe_prop_mask
)


Image.fromarray(
    editable_mask.astype(
        np.uint8
    ) * 255,
    mode="L"
).save(
    EDITABLE_MASK_PATH
)


# ============================================================
# 12. SAVE INPUT / MASK / PROMPTS
# ============================================================

original_pil.save(
    INPUT_SNAPSHOT_PATH
)


Image.fromarray(
    safe_prop_mask.astype(
        np.uint8
    ) * 255,
    mode="L"
).save(
    PROP_MASK_SNAPSHOT_PATH
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
# 13. CLEAN MEMORY
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
# 14. REUSE QWEN PIPE IF AVAILABLE
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
# 15. LOW-VRAM MODE
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
# 16. GENERATOR
# ============================================================

generator = torch.Generator(
    device="cpu"
).manual_seed(
    SEED
)


# ============================================================
# 17. QWEN GENERATION
# ============================================================

print()
print("=" * 110)
print("STARTING PRODUCTION STAGE05 GENERATION")
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
# 18. RAW RESULT
# ============================================================

raw_qwen = (
    result
    .images[0]
    .convert("RGB")
)


raw_qwen.save(
    RAW_QWEN_PATH
)


print(
    "RAW QWEN SIZE:",
    raw_qwen.size
)


# ============================================================
# 19. FORCE MASTER SIZE
# ============================================================

if raw_qwen.size != SIZE:

    raw_master_size = raw_qwen.resize(
        SIZE,
        Image.Resampling.LANCZOS
    )

else:

    raw_master_size = raw_qwen.copy()


raw_master_size.save(
    RAW_MASTER_SIZE_PATH
)


generated = np.array(
    raw_master_size
)


# ============================================================
# 20. RESTORE EXACT ORIGINAL PROP RGB
# ============================================================
#
# This is the critical V1/V3 safety mechanism.
#
# Qwen provides:
# - reconstructed architecture
# - simplified surfaces
#
# Stage01 master provides:
# - exact genuine prop RGB
# ============================================================

safe = generated.copy()


safe[
    safe_prop_mask
] = original[
    safe_prop_mask
]


Image.fromarray(
    safe
).save(
    SAFE_OUTPUT_PATH
)


# ============================================================
# 21. VERIFY PROP RGB
# ============================================================

prop_error = np.abs(
    safe.astype(
        np.int16
    )
    -
    original.astype(
        np.int16
    )
)


if safe_prop_mask.any():

    max_prop_rgb_error = int(
        prop_error[
            safe_prop_mask
        ].max()
    )

    mean_prop_rgb_error = float(
        prop_error[
            safe_prop_mask
        ].mean()
    )

else:

    max_prop_rgb_error = 0

    mean_prop_rgb_error = 0.0


print()
print(
    "MAX RGB ERROR INSIDE PROTECTED PROPS:",
    max_prop_rgb_error
)

print(
    "MEAN RGB ERROR INSIDE PROTECTED PROPS:",
    round(
        mean_prop_rgb_error,
        6
    )
)


if max_prop_rgb_error != 0:

    raise RuntimeError(
        "Protected prop RGB restoration failed."
    )


# ============================================================
# 22. SAVE ORIGINAL PROP REFERENCE RGBA
# ============================================================

prop_rgba = np.zeros(
    (
        H,
        W,
        4
    ),
    dtype=np.uint8
)


prop_rgba[
    safe_prop_mask,
    :3
] = original[
    safe_prop_mask
]


prop_rgba[
    safe_prop_mask,
    3
] = 255


Image.fromarray(
    prop_rgba,
    mode="RGBA"
).save(
    PROP_LAYER_PATH
)


# ============================================================
# 23. COMPARISON
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
        safe
    ),
    (
        W * 2,
        LABEL_H
    )
)


draw = ImageDraw.Draw(
    comparison
)


for x, label in [

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
        "SAFE EXACT-PROP RESTORE"
    ),
]:

    draw.text(
        (
            x,
            11
        ),
        label,
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
# 24. REPORT
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

    "goal":
        (
            "Create plain-surface room with mirror/glass "
            "removed while preserving genuine physical props."
        ),

    "surface_target": {

        "wall":
            "plain smooth matte white",

        "ceiling":
            "plain smooth matte white",

        "floor":
            "#00C800",
    },

    "input": {

        "master":
            str(
                MASTER_PATH
            ),

        "master_sha256":
            sha256_file(
                MASTER_PATH
            ),

        "mirror_mask":
            str(
                MIRROR_MASK_PATH
            ),

        "glass_mask":
            str(
                GLASS_MASK_PATH
            ),

        "prop_protection_source":
            str(
                FROZEN_PROP_MASK_PATH
            ),
    },

    "counts": {

        "mirror_pixels":
            int(
                mirror_mask.sum()
            ),

        "glass_pixels":
            int(
                glass_mask.sum()
            ),

        "raw_prop_mask_pixels":
            int(
                prop_mask.sum()
            ),

        "safe_protected_prop_pixels":
            protected_pixels,
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
    },

    "verification": {

        "max_rgb_error_inside_protected_props":
            max_prop_rgb_error,

        "mean_rgb_error_inside_protected_props":
            mean_prop_rgb_error,

        "prop_rgb_source":
            "Stage01 master",

        "mirror_reintroduced_by_prop_restore":
            False,

        "glass_reintroduced_by_prop_restore":
            False,
    },

    "outputs": {

        "input":
            str(
                INPUT_SNAPSHOT_PATH
            ),

        "protected_prop_mask":
            str(
                PROP_MASK_SNAPSHOT_PATH
            ),

        "editable_architecture_mask":
            str(
                EDITABLE_MASK_PATH
            ),

        "raw_qwen":
            str(
                RAW_QWEN_PATH
            ),

        "raw_master_size":
            str(
                RAW_MASTER_SIZE_PATH
            ),

        "clean_room_with_exact_props":
            str(
                SAFE_OUTPUT_PATH
            ),

        "prop_reference_rgba":
            str(
                PROP_LAYER_PATH
            ),

        "comparison":
            str(
                COMPARISON_PATH
            ),
    },

    "frozen_rules": [

        "Stage01 master remains the RGB and geometry source of truth.",

        "Mirror must remain removed from Stage05 output.",

        "Glass partition must remain removed from Stage05 output.",

        "All genuine non-special-layer props must remain present.",

        "Protected prop RGB is restored exactly from Stage01 master.",

        "Stage05 output is the input to Production Stage06 prop detection.",

        "Stage05 is not yet the empty room.",
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
# 25. FINAL
# ============================================================

print()
print("=" * 110)
print("PRODUCTION STAGE 05 COMPLETE")
print("=" * 110)

print()

print(
    "SAFE OUTPUT:",
    SAFE_OUTPUT_PATH
)

print(
    "PROTECTED PROPS:",
    protected_pixels
)

print(
    "MAX PROP RGB ERROR:",
    max_prop_rgb_error
)

print(
    "REPORT:",
    REPORT_PATH
)


print()
print("ORIGINAL vs RAW QWEN vs SAFE")

display(
    comparison
)


print()
print(
    "FINAL STAGE05 CLEAN ROOM WITH PROPS"
)

display(
    Image.open(
        SAFE_OUTPUT_PATH
    )
)


print()
print(
    "PROTECTED PROP REFERENCE"
)

display(
    Image.open(
        PROP_LAYER_PATH
    )
)
