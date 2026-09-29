
# ============================================================
# TEST07 - ROOM03
# STAGE01Z3
#
# SHORT PRIORITY SURFACE-TEMPLATE PROMPT V3
#
# MODEL:
# Qwen/Qwen-Image-Edit-2509
#
# TARGET:
# SAME ORIGINAL ROOM
# - remove large wall mirror
# - remove internal shower glass partition
# - walls = plain matte white
# - ceiling = plain matte white
# - floor = solid #00C800
# - preserve genuine props
# - preserve geometry/camera/perspective
#
# IMPORTANT:
# V1 and V2 are NOT overwritten.
# Frozen prop pixels are restored exactly from original RGB.
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
# 2. PATHS
# ============================================================

STAGE_NAME = "TEST07_STAGE01Z3_QWEN_SHORT_SURFACE_TEMPLATE_V3"

SCRIPT_NAME = "test07_stage01z3_qwen_short_surface_template_v3.py"

MODEL_ID = "Qwen/Qwen-Image-Edit-2509"

BASE = Path("/workspace/axolotl")

ROOT = (
    BASE
    / "test07"
    / "runs"
    / "room03_bathroom"
    / "stages"
)

OUT = (
    ROOT
    / "01z3_qwen_short_surface_template_v3"
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

os.environ["HF_HOME"] = str(CACHE_DIR)

os.environ["HUGGINGFACE_HUB_CACHE"] = str(CACHE_DIR)


# ============================================================
# 3. GENERATION SETTINGS
# ============================================================

SEED = 12345

NUM_INFERENCE_STEPS = 20

TRUE_CFG_SCALE = 4.0

DTYPE = torch.bfloat16


# ============================================================
# 4. SHORT HIGH-PRIORITY PROMPT
# ============================================================

PROMPT = r"""
Edit this exact room photograph into a clean architectural surface template.

ABSOLUTE PRIORITY:
Keep the exact same room, camera position, camera angle, perspective,
framing, room geometry, wall boundaries, floor boundaries, ceiling
boundaries, object positions, scale and lighting.

Make ONLY these changes:

1. REMOVE the large wall mirror completely.
   Remove everything reflected inside it.
   Continue the real wall naturally behind it as plain smooth matte white.
   Do not leave a frame, rectangular patch, reflection or new object.

2. REMOVE the complete transparent internal shower glass partition.
   Remove the glass panels, frosted glass, clear glass, rails, frames,
   handles, hinges and partition hardware.
   Preserve the real wall, floor and physical fixtures already visible
   through the glass.
   Reconstruct only the minimum hidden architecture necessary.

3. ALL editable visible wall surfaces:
   plain, smooth, uniform, matte WHITE.
   Remove wall tiles, patterns, texture, stone, decorative finish,
   grout and joints.

4. ALL editable visible ceiling surfaces:
   plain, smooth, uniform, matte WHITE.
   Preserve ceiling lights and mounted fixtures.

5. COMPLETE visible floor surface:
   replace with ONE solid uniform GREEN color:
   HEX #00C800
   RGB 0, 200, 0.

   The floor must contain:
   NO tile pattern,
   NO grout,
   NO joints,
   NO texture,
   NO decorative markings.

   Green must appear ONLY on the actual floor plane.
   Do not paint walls, furniture, cabinets, fixtures or objects green.

6. Preserve ALL genuine physical objects and fixtures in their exact
   original positions, including:
   vanity cabinet, countertop, basin, faucet, soap dispenser, toilet,
   wall plate, showerhead, towel bar, lights, doors and all other
   real physical props.

7. Do NOT add anything.
   Do NOT create shelves, pictures, frames, decoration, furniture,
   accessories or architectural structures.

8. Do NOT redesign or beautify the room.
   Do NOT move, resize, replace or restyle objects.
   Do NOT change the camera or perspective.
   Do NOT crop, zoom or extend the image.

FINAL TARGET:

Same exact original scene.

WALLS = plain matte white.
CEILING = plain matte white.
FLOOR = solid uniform #00C800 green.
LARGE WALL MIRROR = removed.
OBSTRUCTING INTERNAL SHOWER GLASS PARTITION = removed.
REAL OBJECTS AND FIXTURES = preserved.
CAMERA AND GEOMETRY = preserved.

The result must look like the same real photograph after surface
preparation, not a redesigned or newly generated room.
"""


# ============================================================
# 5. NEGATIVE PROMPT
# ============================================================

NEGATIVE_PROMPT = r"""
different room,
different camera,
changed perspective,
zoom,
crop,
warped geometry,
shifted walls,
shifted corners,
moved objects,
resized objects,
missing objects,
duplicated objects,
new furniture,
new shelf,
new picture,
new frame,
new decoration,
new accessories,
hallucinated objects,
redesigned bathroom,
changed vanity,
changed toilet,
changed basin,
changed door,
green walls,
green furniture,
green cabinets,
green fixtures,
white furniture,
white-painted objects,
remaining wall tiles,
remaining wall pattern,
remaining floor tiles,
remaining floor texture,
floor grout,
mirror reflection,
remaining wall mirror,
remaining shower glass partition,
glass rails,
glass frame,
artificial blocks,
collage,
cutout,
flattened depth
"""


# ============================================================
# 6. START
# ============================================================

print()
print("=" * 100)
print(STAGE_NAME)
print("=" * 100)

print("SCRIPT :", SCRIPT_NAME)
print("OUTPUT :", OUT)
print("MODEL  :", MODEL_ID)


# ============================================================
# 7. INPUT CHECK
# ============================================================

for name, path in {
    "ORIGINAL": ORIGINAL_PATH,
    "PROP MASK": PROP_MASK_PATH,
}.items():

    print(
        name,
        "✅ FOUND" if path.exists() else "❌ MISSING",
        path
    )

    if not path.exists():
        raise FileNotFoundError(path)


# ============================================================
# 8. LOAD ORIGINAL
# ============================================================

original_pil = Image.open(
    ORIGINAL_PATH
).convert("RGB")

original = np.array(
    original_pil
)

H, W = original.shape[:2]

print()
print("MASTER SIZE:", W, "x", H)


# ============================================================
# 9. LOAD FROZEN PROP MASK
# ============================================================

prop_mask_pil = Image.open(
    PROP_MASK_PATH
).convert("L")

if prop_mask_pil.size != (W, H):

    prop_mask_pil = prop_mask_pil.resize(
        (W, H),
        Image.Resampling.NEAREST
    )

prop_mask = (
    np.array(prop_mask_pil) > 0
)

protected_pixels = int(
    prop_mask.sum()
)

print(
    "FROZEN PROP PIXELS:",
    protected_pixels
)


# ============================================================
# 10. SAVE EXACT INPUTS
# ============================================================

input_snapshot = OUT / "00_input_original.png"

original_pil.save(
    input_snapshot
)

mask_snapshot = OUT / "01_frozen_prop_mask.png"

prop_mask_pil.save(
    mask_snapshot
)


# ============================================================
# 11. SAVE PROMPTS - UTF8
# ============================================================

prompt_path = OUT / "02_prompt.txt"

prompt_path.write_text(
    PROMPT,
    encoding="utf-8"
)

negative_prompt_path = OUT / "03_negative_prompt.txt"

negative_prompt_path.write_text(
    NEGATIVE_PROMPT,
    encoding="utf-8"
)


# ============================================================
# 12. CLEAN MEMORY
# ============================================================

gc.collect()

if torch.cuda.is_available():

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


if torch.cuda.is_available():

    free_before, total_before = torch.cuda.mem_get_info()

    print()
    print(
        "GPU FREE BEFORE GENERATION GB:",
        round(
            free_before / 1024**3,
            2
        )
    )


# ============================================================
# 13. REUSE EXISTING QWEN PIPE IF AVAILABLE
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

    load_start = time.time()

    pipe = (
        QwenImageEditPlusPipeline
        .from_pretrained(
            MODEL_ID,
            torch_dtype=DTYPE,
            cache_dir=str(CACHE_DIR),
        )
    )

    pipe.enable_sequential_cpu_offload()

    model_load_seconds = (
        time.time() - load_start
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
# 14. ENSURE LOW-VRAM MODE
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
# 16. QWEN GENERATION
# ============================================================

print()
print("=" * 100)
print("STARTING STAGE01Z3 GENERATION")
print("=" * 100)

print("SEED :", SEED)
print("STEPS:", NUM_INFERENCE_STEPS)
print("CFG  :", TRUE_CFG_SCALE)

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
    time.time() - generation_start
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

raw_qwen = result.images[0].convert(
    "RGB"
)

raw_original_size = raw_qwen.size

print(
    "RAW QWEN SIZE:",
    raw_original_size
)

raw_path = OUT / "04_raw_qwen_v3.png"

raw_qwen.save(
    raw_path
)


# ============================================================
# 18. NORMALIZE TO ORIGINAL SIZE
# ============================================================

if raw_qwen.size != (W, H):

    qwen_master = raw_qwen.resize(
        (W, H),
        Image.Resampling.LANCZOS
    )

else:

    qwen_master = raw_qwen.copy()


master_path = OUT / "05_qwen_v3_master_size.png"

qwen_master.save(
    master_path
)


generated = np.array(
    qwen_master
)


# ============================================================
# 19. EXACT ORIGINAL PROP RESTORATION
# ============================================================

safe = generated.copy()

safe[prop_mask] = original[prop_mask]

safe_pil = Image.fromarray(
    safe
)

safe_path = (
    OUT
    / "06_safe_v3_original_props_restored.png"
)

safe_pil.save(
    safe_path
)


# ============================================================
# 20. VERIFY EXACT RGB
# ============================================================

diff = np.abs(
    safe.astype(np.int16)
    -
    original.astype(np.int16)
)

protected_diff = diff[
    prop_mask
]

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
# 21. THREE-PANEL COMPARISON
# ============================================================

LABEL_H = 36

comparison = Image.new(
    "RGB",
    (
        W * 3,
        H + LABEL_H
    ),
    (255, 255, 255)
)

comparison.paste(
    original_pil,
    (0, LABEL_H)
)

comparison.paste(
    qwen_master,
    (W, LABEL_H)
)

comparison.paste(
    safe_pil,
    (W * 2, LABEL_H)
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
    "RAW QWEN V3",
    fill=(0, 0, 0)
)

draw.text(
    (W * 2 + 8, 10),
    "SAFE V3 + ORIGINAL PROPS",
    fill=(0, 0, 0)
)

comparison_path = (
    OUT
    / "07_original_raw_safe_v3_comparison.png"
)

comparison.save(
    comparison_path
)


# ============================================================
# 22. ORIGINAL VS FINAL V3
# ============================================================

final_compare = Image.new(
    "RGB",
    (
        W * 2,
        H + LABEL_H
    ),
    (255, 255, 255)
)

final_compare.paste(
    original_pil,
    (0, LABEL_H)
)

final_compare.paste(
    safe_pil,
    (W, LABEL_H)
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
    "STAGE01Z3 FINAL",
    fill=(0, 0, 0)
)

final_compare_path = (
    OUT
    / "08_original_vs_final_v3.png"
)

final_compare.save(
    final_compare_path
)


# ============================================================
# 23. SAVE ORIGINAL-RGB PROP RGBA
# ============================================================

rgba = np.zeros(
    (H, W, 4),
    dtype=np.uint8
)

rgba[..., :3] = original

rgba[..., 3] = (
    prop_mask.astype(np.uint8) * 255
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
# 24. HASH HELPER
# ============================================================

def sha256_file(path):

    h = hashlib.sha256()

    with open(
        path,
        "rb"
    ) as f:

        while True:

            chunk = f.read(
                1024 * 1024
            )

            if not chunk:
                break

            h.update(
                chunk
            )

    return h.hexdigest()


# ============================================================
# 25. REPORT
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

    "version_goal":
        (
            "Retain V1 coherent reconstruction behavior "
            "while explicitly forcing solid #00C800 floor."
        ),

    "input": {
        "original":
            str(ORIGINAL_PATH),

        "original_sha256":
            sha256_file(
                ORIGINAL_PATH
            ),

        "prop_mask":
            str(PROP_MASK_PATH),

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
            str(DTYPE),

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

    "target": {
        "walls":
            "plain smooth matte white",

        "ceiling":
            "plain smooth matte white",

        "floor":
            "#00C800",

        "large_wall_mirror":
            "removed",

        "internal_shower_glass_partition":
            "removed",

        "props":
            "preserved",

        "geometry":
            "preserved",
    },

    "outputs": {
        "input":
            str(input_snapshot),

        "mask":
            str(mask_snapshot),

        "prompt":
            str(prompt_path),

        "negative_prompt":
            str(negative_prompt_path),

        "raw":
            str(raw_path),

        "master_size":
            str(master_path),

        "safe":
            str(safe_path),

        "three_panel":
            str(comparison_path),

        "original_vs_final":
            str(final_compare_path),

        "props_rgba":
            str(rgba_path),
    },

    "comparison_status": {
        "V1":
            "KEEP - GOOD REFERENCE",

        "V2":
            "KEEP - FAILED FULL PROMPT BENCHMARK",

        "V3":
            "PENDING VISUAL REVIEW",
    },
}


report_path = (
    OUT
    / "00_stage01z3_report.json"
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
# 26. FINAL OUTPUT
# ============================================================

print()
print("=" * 100)
print("STAGE01Z3 COMPLETE")
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
# 27. DISPLAY
# ============================================================

print()
print(
    "ORIGINAL / RAW QWEN V3 / SAFE V3"
)

display(
    comparison
)

print()
print(
    "ORIGINAL VS FINAL V3"
)

display(
    final_compare
)
