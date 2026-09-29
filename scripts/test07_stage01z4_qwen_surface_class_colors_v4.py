
# ============================================================
# TEST07 - ROOM03
# STAGE01Z4
#
# QWEN DIAGNOSTIC SURFACE-CLASS COLORS V4
#
# MODEL:
# Qwen/Qwen-Image-Edit-2509
#
# PURPOSE:
# Determine whether Qwen can correctly distinguish:
#
# WALL    -> BLUE  #0066FF
# CEILING -> WHITE
# FLOOR   -> GREEN #00C800
#
# while:
# - preserving geometry/camera/perspective
# - removing large wall mirror
# - removing internal shower glass partition
# - preserving genuine physical props
#
# FINAL prop pixels are restored exactly from original RGB.
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

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

from diffusers import QwenImageEditPlusPipeline


# ============================================================
# 2. IDENTIFIERS / PATHS
# ============================================================

STAGE_NAME = (
    "TEST07_STAGE01Z4_"
    "QWEN_SURFACE_CLASS_COLORS_V4"
)

SCRIPT_NAME = (
    "test07_stage01z4_"
    "qwen_surface_class_colors_v4.py"
)

MODEL_ID = "Qwen/Qwen-Image-Edit-2509"

BASE = Path(
    "/workspace/axolotl"
)

ROOT = (
    BASE
    / "test07"
    / "runs"
    / "room03_bathroom"
    / "stages"
)

OUT = (
    ROOT
    / "01z4_qwen_surface_class_colors_v4"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


ORIGINAL_PATH = (
    ROOT
    / "01_structure_inverse_base"
    / "00_input_resized.png"
)

PROP_MASK_PATH = (
    ROOT
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

os.environ["HF_HOME"] = str(
    CACHE_DIR
)

os.environ["HUGGINGFACE_HUB_CACHE"] = str(
    CACHE_DIR
)


# ============================================================
# 3. SETTINGS
# ============================================================

SEED = 12345

NUM_INFERENCE_STEPS = 20

TRUE_CFG_SCALE = 4.0

DTYPE = torch.bfloat16


# ============================================================
# 4. TARGET COLORS
# ============================================================

WALL_HEX = "#0066FF"
WALL_RGB = [0, 102, 255]

CEILING_HEX = "#FFFFFF"
CEILING_RGB = [255, 255, 255]

FLOOR_HEX = "#00C800"
FLOOR_RGB = [0, 200, 0]


# ============================================================
# 5. SHORT DIAGNOSTIC PROMPT
# ============================================================

PROMPT = r"""
Edit this exact room photograph into a diagnostic architectural
surface-class template.

ABSOLUTE PRIORITY:
Keep the exact same camera, viewpoint, perspective, framing,
room geometry, corners, wall boundaries, ceiling boundaries,
floor boundaries, object positions, object scale and lighting.

Make ONLY these changes:

1. WALL SURFACES
Detect every actual visible editable WALL plane.

Replace ONLY wall surfaces with one clean solid uniform BLUE color:

HEX #0066FF
RGB 0, 102, 255

The blue must follow the exact original wall planes, perspective,
boundaries, corners, depth and occlusions.

Do NOT paint the ceiling blue.
Do NOT paint the floor blue.
Do NOT paint doors, windows, furniture, cabinets, fixtures or props blue.


2. CEILING SURFACES
Detect every actual visible CEILING plane.

Replace ONLY ceiling surfaces with clean uniform WHITE:

HEX #FFFFFF
RGB 255, 255, 255

Preserve ceiling geometry and ceiling-mounted objects such as lights.

Do NOT paint walls white.
Do NOT paint floor white.
Do NOT paint physical props white.


3. FLOOR SURFACE
Detect the COMPLETE visible FLOOR plane.

Replace ONLY the actual floor with one solid uniform GREEN:

HEX #00C800
RGB 0, 200, 0

The floor must contain:
NO tile pattern,
NO grout,
NO joints,
NO texture,
NO decorative markings.

The green must follow the exact floor plane, perspective, boundaries,
edges, depth and occlusions.

Do NOT paint walls green.
Do NOT paint ceiling green.
Do NOT paint furniture, cabinets, toilet, basin, fixtures or props green.


4. LARGE WALL MIRROR
Remove the large wall mirror completely.

Remove reflected scene content inside it.

The area behind the removed mirror must become the SAME BLUE as
the surrounding wall because it is part of the wall plane.

Do not leave:
- mirror frame
- reflection
- rectangular white patch
- decorative object
- shelf
- picture
- new opening


5. INTERNAL SHOWER GLASS PARTITION
Remove the complete transparent internal shower glass partition.

Remove:
- clear glass
- frosted glass
- glass panels
- partition rails
- partition frames
- handles
- hinges
- supports
- partition hardware

Preserve everything genuinely visible through the glass.

After glass removal:
- visible wall areas behind it must become BLUE
- visible floor areas behind it must become GREEN
- visible ceiling areas behind it must remain WHITE
- real physical fixtures behind it must remain


6. PRESERVE REAL PHYSICAL OBJECTS
Preserve all genuine physical objects and fixtures exactly where they are.

Examples in this room include:
- vanity cabinet
- countertop
- basin
- faucet
- soap dispenser
- toilet
- wall electrical/flush plate
- showerhead
- horizontal towel/shower bar
- ceiling light
- door
- all other real physical props

Do NOT move them.
Do NOT resize them.
Do NOT redesign them.
Do NOT replace them.
Do NOT add anything new.


7. GEOMETRY
Do NOT:
- change camera angle
- change camera height
- change perspective
- change field of view
- zoom
- crop
- extend image
- move corners
- move wall boundaries
- move floor boundaries
- move ceiling boundaries
- straighten perspective
- redesign architecture


FINAL DIAGNOSTIC TARGET:

WALLS:
solid uniform BLUE #0066FF

CEILING:
solid uniform WHITE #FFFFFF

FLOOR:
solid uniform GREEN #00C800

LARGE WALL MIRROR:
removed, exposing BLUE wall

INTERNAL SHOWER GLASS PARTITION:
removed

REAL PROPS:
preserved

ROOM GEOMETRY:
preserved

CAMERA:
preserved

Do not add shelves, pictures, frames, artwork, furniture,
decorations or architectural structures.

This is a surface-class diagnostic image, not an interior design.
"""


# ============================================================
# 6. NEGATIVE PROMPT
# ============================================================

NEGATIVE_PROMPT = r"""
different room,
changed camera,
changed perspective,
zoom,
crop,
warped walls,
warped floor,
warped ceiling,
shifted corners,
moved objects,
resized objects,
missing objects,
duplicated objects,
new shelf,
new picture,
new frame,
new decoration,
new furniture,
hallucinated objects,
blue ceiling,
blue floor,
blue furniture,
blue cabinet,
blue toilet,
blue basin,
blue fixtures,
green walls,
green ceiling,
green furniture,
green cabinet,
green toilet,
green fixtures,
white walls,
white floor,
white furniture,
remaining wall mirror,
mirror reflection,
remaining shower glass partition,
glass rail,
glass frame,
wall texture,
wall tiles,
floor tiles,
floor grout,
floor texture,
artificial blocks,
collage,
cutout,
flattened depth
"""


# ============================================================
# 7. START
# ============================================================

print()
print("=" * 100)
print(STAGE_NAME)
print("=" * 100)

print("SCRIPT :", SCRIPT_NAME)
print("OUTPUT :", OUT)
print("MODEL  :", MODEL_ID)

print()
print("TARGET COLORS:")
print("WALL   :", WALL_HEX, WALL_RGB)
print("CEILING:", CEILING_HEX, CEILING_RGB)
print("FLOOR  :", FLOOR_HEX, FLOOR_RGB)


# ============================================================
# 8. INPUT VALIDATION
# ============================================================

for name, path in {
    "ORIGINAL": ORIGINAL_PATH,
    "PROP MASK": PROP_MASK_PATH,
}.items():

    status = (
        "✅ FOUND"
        if path.exists()
        else "❌ MISSING"
    )

    print(
        f"{name:12s}",
        status,
        path
    )

    if not path.exists():

        raise FileNotFoundError(
            path
        )


# ============================================================
# 9. LOAD ORIGINAL
# ============================================================

original_pil = Image.open(
    ORIGINAL_PATH
).convert(
    "RGB"
)

original = np.array(
    original_pil
)

H, W = (
    original.shape[:2]
)

print()
print(
    "MASTER SIZE:",
    W,
    "x",
    H
)


# ============================================================
# 10. LOAD FROZEN PROP MASK
# ============================================================

prop_mask_pil = Image.open(
    PROP_MASK_PATH
).convert(
    "L"
)

if prop_mask_pil.size != (
    W,
    H
):

    prop_mask_pil = (
        prop_mask_pil.resize(
            (W, H),
            Image.Resampling.NEAREST
        )
    )


prop_mask = (
    np.array(
        prop_mask_pil
    )
    > 0
)

protected_pixels = int(
    prop_mask.sum()
)

print(
    "FROZEN PROP PIXELS:",
    protected_pixels
)


# ============================================================
# 11. SAVE EXACT INPUT SNAPSHOTS
# ============================================================

input_snapshot = (
    OUT
    / "00_input_original.png"
)

original_pil.save(
    input_snapshot
)

mask_snapshot = (
    OUT
    / "01_frozen_prop_mask.png"
)

prop_mask_pil.save(
    mask_snapshot
)


# ============================================================
# 12. SAVE PROMPTS UTF-8
# ============================================================

prompt_path = (
    OUT
    / "02_prompt.txt"
)

prompt_path.write_text(
    PROMPT,
    encoding="utf-8"
)

negative_prompt_path = (
    OUT
    / "03_negative_prompt.txt"
)

negative_prompt_path.write_text(
    NEGATIVE_PROMPT,
    encoding="utf-8"
)


# ============================================================
# 13. MEMORY CLEANUP
# ============================================================

gc.collect()

if torch.cuda.is_available():

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    free_before, total_before = (
        torch.cuda.mem_get_info()
    )

    print()
    print(
        "GPU FREE BEFORE GENERATION GB:",
        round(
            free_before
            / 1024**3,
            2
        )
    )


# ============================================================
# 14. REUSE EXISTING PIPE IF AVAILABLE
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
        "✅ Reusing existing QwenImageEditPlusPipeline"
    )

    model_load_seconds = 0.0

else:

    print()
    print("=" * 100)
    print("LOADING QWEN IMAGE EDIT")
    print("=" * 100)

    load_start = (
        time.time()
    )

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
        - load_start
    )

    print(
        "✅ Pipeline loaded"
    )

    print(
        "MODEL LOAD TIME SEC:",
        round(
            model_load_seconds,
            2
        )
    )


# ============================================================
# 15. ENSURE LOW-VRAM MODE
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

generator = (
    torch.Generator(
        device="cpu"
    )
    .manual_seed(
        SEED
    )
)


# ============================================================
# 17. GENERATE
# ============================================================

print()
print("=" * 100)
print("STARTING STAGE01Z4 GENERATION")
print("=" * 100)

print(
    "SEED:",
    SEED
)

print(
    "STEPS:",
    NUM_INFERENCE_STEPS
)

print(
    "CFG:",
    TRUE_CFG_SCALE
)

generation_start = (
    time.time()
)

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
    - generation_start
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
# 18. RAW QWEN OUTPUT
# ============================================================

raw_qwen = (
    result.images[0]
    .convert(
        "RGB"
    )
)

raw_original_size = (
    raw_qwen.size
)

print(
    "RAW QWEN SIZE:",
    raw_original_size
)

raw_path = (
    OUT
    / "04_raw_qwen_v4.png"
)

raw_qwen.save(
    raw_path
)


# ============================================================
# 19. NORMALIZE TO MASTER SIZE
# ============================================================

if raw_qwen.size != (
    W,
    H
):

    qwen_master = (
        raw_qwen.resize(
            (W, H),
            Image.Resampling.LANCZOS
        )
    )

else:

    qwen_master = (
        raw_qwen.copy()
    )


master_path = (
    OUT
    / "05_qwen_v4_master_size.png"
)

qwen_master.save(
    master_path
)

generated = np.array(
    qwen_master
)


# ============================================================
# 20. RESTORE ORIGINAL PROP RGB
# ============================================================

safe = (
    generated.copy()
)

safe[
    prop_mask
] = original[
    prop_mask
]

safe_pil = (
    Image.fromarray(
        safe
    )
)

safe_path = (
    OUT
    / "06_safe_v4_original_props_restored.png"
)

safe_pil.save(
    safe_path
)


# ============================================================
# 21. VERIFY EXACT PROP RGB
# ============================================================

diff = np.abs(

    safe.astype(
        np.int16
    )

    -

    original.astype(
        np.int16
    )
)

protected_diff = (
    diff[
        prop_mask
    ]
)

if protected_diff.size:

    max_prop_rgb_error = int(
        protected_diff.max()
    )

    mean_prop_rgb_error = float(
        protected_diff.mean()
    )

else:

    max_prop_rgb_error = 0

    mean_prop_rgb_error = 0.0


print()
print("=" * 100)
print("FROZEN PROP RGB VERIFICATION")
print("=" * 100)

print(
    "MAX RGB ERROR:",
    max_prop_rgb_error
)

print(
    "MEAN RGB ERROR:",
    mean_prop_rgb_error
)

if max_prop_rgb_error == 0:

    print(
        "✅ EXACT ORIGINAL PROP RGB PRESERVED"
    )


# ============================================================
# 22. THREE-PANEL COMPARISON
# ============================================================

LABEL_H = 36

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
    qwen_master,
    (
        W,
        LABEL_H
    )
)

comparison.paste(
    safe_pil,
    (
        W * 2,
        LABEL_H
    )
)

draw = ImageDraw.Draw(
    comparison
)

draw.text(
    (8, 10),
    "ORIGINAL",
    fill=(0, 0, 0)
)

draw.text(
    (W + 8, 10),
    "RAW QWEN V4",
    fill=(0, 0, 0)
)

draw.text(
    (W * 2 + 8, 10),
    "SAFE V4 + ORIGINAL PROPS",
    fill=(0, 0, 0)
)

comparison_path = (
    OUT
    / "07_original_raw_safe_v4_comparison.png"
)

comparison.save(
    comparison_path
)


# ============================================================
# 23. ORIGINAL VS FINAL V4
# ============================================================

final_compare = Image.new(

    "RGB",

    (
        W * 2,
        H + LABEL_H
    ),

    (
        255,
        255,
        255
    )
)

final_compare.paste(
    original_pil,
    (
        0,
        LABEL_H
    )
)

final_compare.paste(
    safe_pil,
    (
        W,
        LABEL_H
    )
)

draw2 = ImageDraw.Draw(
    final_compare
)

draw2.text(
    (8, 10),
    "ORIGINAL",
    fill=(0, 0, 0)
)

draw2.text(
    (W + 8, 10),
    "STAGE01Z4 FINAL",
    fill=(0, 0, 0)
)

final_compare_path = (
    OUT
    / "08_original_vs_final_v4.png"
)

final_compare.save(
    final_compare_path
)


# ============================================================
# 24. PROP RGBA
# ============================================================

rgba = np.zeros(
    (
        H,
        W,
        4
    ),
    dtype=np.uint8
)

rgba[..., :3] = original

rgba[..., 3] = (
    prop_mask.astype(
        np.uint8
    )
    * 255
)

rgba_path = (
    OUT
    / "09_original_props_rgba.png"
)

Image.fromarray(
    rgba,
    mode="RGBA"
).save(
    rgba_path
)


# ============================================================
# 25. HASH HELPER
# ============================================================

def sha256_file(path):

    h = hashlib.sha256()

    with open(
        path,
        "rb"
    ) as f:

        while True:

            chunk = (
                f.read(
                    1024 * 1024
                )
            )

            if not chunk:

                break

            h.update(
                chunk
            )

    return (
        h.hexdigest()
    )


# ============================================================
# 26. REPORT
# ============================================================

report = {

    "stage":
        STAGE_NAME,

    "status":
        "COMPLETED",

    "script":
        SCRIPT_NAME,

    "model":
        MODEL_ID,

    "pipeline":
        "QwenImageEditPlusPipeline",

    "purpose":
        (
            "Diagnostic verification of wall vs ceiling vs floor "
            "surface-class understanding."
        ),

    "target_colors": {

        "wall": {
            "hex":
                WALL_HEX,

            "rgb":
                WALL_RGB,
        },

        "ceiling": {
            "hex":
                CEILING_HEX,

            "rgb":
                CEILING_RGB,
        },

        "floor": {
            "hex":
                FLOOR_HEX,

            "rgb":
                FLOOR_RGB,
        },
    },

    "input": {

        "original":
            str(
                ORIGINAL_PATH
            ),

        "original_sha256":
            sha256_file(
                ORIGINAL_PATH
            ),

        "prop_mask":
            str(
                PROP_MASK_PATH
            ),

        "prop_mask_sha256":
            sha256_file(
                PROP_MASK_PATH
            ),
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

        "protected_prop_pixels":
            protected_pixels,

        "max_rgb_error_inside_frozen_props":
            max_prop_rgb_error,

        "mean_rgb_error_inside_frozen_props":
            mean_prop_rgb_error,
    },

    "expected_surface_classes": {

        "walls":
            "BLUE #0066FF",

        "ceiling":
            "WHITE #FFFFFF",

        "floor":
            "GREEN #00C800",

        "mirror":
            "removed and replaced by blue wall",

        "internal_shower_glass":
            "removed",

        "real_props":
            "preserved",
    },

    "outputs": {

        "input":
            str(
                input_snapshot
            ),

        "mask":
            str(
                mask_snapshot
            ),

        "prompt":
            str(
                prompt_path
            ),

        "negative_prompt":
            str(
                negative_prompt_path
            ),

        "raw":
            str(
                raw_path
            ),

        "master_size":
            str(
                master_path
            ),

        "safe":
            str(
                safe_path
            ),

        "three_panel":
            str(
                comparison_path
            ),

        "original_vs_final":
            str(
                final_compare_path
            ),

        "props_rgba":
            str(
                rgba_path
            ),
    },

    "benchmark_versions": {

        "V1":
            "KEEP - GOOD REFERENCE",

        "V2":
            "KEEP - FAILED LONG PROMPT",

        "V3":
            "KEEP - WHITE WALL / GREEN FLOOR TEST",

        "V4":
            "PENDING SURFACE-CLASS REVIEW",
    },
}


report_path = (
    OUT
    / "00_stage01z4_report.json"
)

report_path.write_text(
    json.dumps(
        report,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# ============================================================
# 27. FINAL STATUS
# ============================================================

print()
print("=" * 100)
print("STAGE01Z4 COMPLETE")
print("=" * 100)

print(
    "SAFE RESULT:",
    safe_path
)

print(
    "3-PANEL:",
    comparison_path
)

print(
    "REPORT:",
    report_path
)

print(
    "MAX PROP RGB ERROR:",
    max_prop_rgb_error
)


# ============================================================
# 28. DISPLAY
# ============================================================

print()
print(
    "ORIGINAL / RAW QWEN V4 / SAFE V4"
)

display(
    comparison
)

print()
print(
    "ORIGINAL VS FINAL V4"
)

display(
    final_compare
)
