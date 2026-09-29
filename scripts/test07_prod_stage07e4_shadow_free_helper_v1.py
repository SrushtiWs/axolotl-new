
from pathlib import Path
import json
import gc

import cv2
import numpy as np
import torch

from PIL import Image
import matplotlib.pyplot as plt

from diffusers import QwenImageEditPlusPipeline


# ============================================================
# PATHS
# ============================================================

BASE = Path("/workspace/axolotl")

PROD = (
    BASE
    / "test07"
    / "production_pipeline"
)

MASTER_PATH = (
    PROD
    / "stage01_master"
    / "00_master_input.png"
)

P01_ROI_PATH = (
    PROD
    / "stage07_empty_room"
    / "07e1b_qwen_coordinate_fix"
    / "01_p01_shadow_mask.png"
)

P02_ROI_PATH = (
    PROD
    / "stage07_empty_room"
    / "07e1b_qwen_coordinate_fix"
    / "02_p02_shadow_mask.png"
)

OUT = (
    PROD
    / "stage07_empty_room"
    / "07e4_shadow_free_helper"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# MODEL
# ============================================================

MODEL_ID = "Qwen/Qwen-Image-Edit-2509"

CACHE = "/workspace/data/huggingface-cache"

SEED = 62041
STEPS = 28
TRUE_CFG_SCALE = 4.5


# ============================================================
# VALIDATE
# ============================================================

for p in [
    MASTER_PATH,
    P01_ROI_PATH,
    P02_ROI_PATH,
]:
    if not p.exists():
        raise FileNotFoundError(p)


# ============================================================
# LOAD
# ============================================================

master = Image.open(
    MASTER_PATH
).convert("RGB")

master_np = np.asarray(
    master
)

W, H = master.size


p01_roi = (
    np.asarray(
        Image.open(
            P01_ROI_PATH
        ).convert("L")
    ) > 127
)


p02_roi = (
    np.asarray(
        Image.open(
            P02_ROI_PATH
        ).convert("L")
    ) > 127
)


combined_roi = (
    p01_roi
    |
    p02_roi
)


# ============================================================
# GUIDANCE IMAGE
#
# RED     = vanity-shadow analysis region
# CYAN    = toilet-shadow analysis region
#
# This is only guidance for the model.
# ============================================================

guide = master_np.astype(
    np.float32
).copy()


guide[
    p01_roi
] = (
    guide[
        p01_roi
    ] * 0.25
    +
    np.array(
        [255, 0, 0],
        dtype=np.float32
    ) * 0.75
)


guide[
    p02_roi
] = (
    guide[
        p02_roi
    ] * 0.25
    +
    np.array(
        [0, 255, 255],
        dtype=np.float32
    ) * 0.75
)


guide = np.clip(
    guide,
    0,
    255
).astype(
    np.uint8
)


GUIDE_PATH = (
    OUT
    / "00_shadow_roi_guidance.png"
)


Image.fromarray(
    guide
).save(
    GUIDE_PATH
)


# ============================================================
# PROMPT
# ============================================================

PROMPT = """
You are creating an analytical SHADOW-FREE REFERENCE for a
room-visualization preprocessing system.

The first supplied image is the original room.

The second supplied image is a guidance copy of the same room.

In the guidance image:

RED marks the approximate floor-shadow region associated with
the vanity.

CYAN marks the approximate floor-shadow region associated with
the toilet.

============================================================
YOUR ONLY EDIT
============================================================

Remove the visible cast/contact FLOOR SHADOWS caused by:

1. the vanity
2. the toilet

Reconstruct what the original floor surface would naturally
look like if those two shadows were absent.

============================================================
PRESERVE EVERYTHING ELSE
============================================================

Do NOT:

- remove the vanity
- remove the toilet
- move any object
- resize any object
- change the room geometry
- change camera perspective
- change floor layout
- change floor material
- replace floor material
- change floor color intentionally
- redesign floor texture
- change walls
- change ceiling
- alter mirror
- alter glass
- add new objects
- add new shadows
- modify fixtures
- crop
- zoom
- change framing

The purpose is NOT beautification.

The purpose is NOT redesign.

The purpose is ONLY to estimate the SAME ORIGINAL floor
appearance underneath the existing vanity/toilet shadows.

============================================================
IMPORTANT
============================================================

Dark floor material/design is NOT shadow.

Preserve genuine material variation, seams, grain, texture and
color.

Remove only illumination darkening physically caused by the
vanity and toilet.

Keep all props exactly where they are.

Return the same scene with those floor shadows removed as
naturally and locally as possible.
"""


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 110)
print("STAGE 07E4 — PRE-TILE SHADOW-FREE HELPER")
print("=" * 110)

print()
print("MASTER:", MASTER_PATH)
print("GUIDANCE:", GUIDE_PATH)

print()
print("Loading Qwen Image Edit...")


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
    "Generating PRE-TILE shadow-free helper..."
)


result = pipe(
    image=[
        master,
        Image.fromarray(guide)
    ],
    prompt=PROMPT,
    num_inference_steps=STEPS,
    true_cfg_scale=TRUE_CFG_SCALE,
    generator=generator,
)


helper = (
    result.images[0]
    .convert("RGB")
)


print(
    "QWEN OUTPUT SIZE:",
    helper.size
)


# ============================================================
# ALIGN SIZE
#
# This does NOT establish exact registration yet.
# 07E5 will deal with local registration if 07E4 is useful.
# ============================================================

if helper.size != (
    W,
    H
):

    helper = helper.resize(
        (
            W,
            H
        ),
        Image.Resampling.LANCZOS
    )


HELPER_PATH = (
    OUT
    / "01_shadow_free_helper.png"
)


helper.save(
    HELPER_PATH
)


# ============================================================
# DIFFERENCE — DIAGNOSTIC ONLY
# ============================================================

helper_np = np.asarray(
    helper
).astype(
    np.float32
)


master_f = master_np.astype(
    np.float32
)


abs_diff = np.mean(
    np.abs(
        helper_np
        -
        master_f
    ),
    axis=2
)


# Only show ROI-specific difference separately.
roi_diff = abs_diff.copy()

roi_diff[
    ~combined_roi
] = 0.0


ABS_DIFF_VIS = np.clip(
    abs_diff / 70.0,
    0.0,
    1.0
)


ROI_DIFF_VIS = np.clip(
    roi_diff / 70.0,
    0.0,
    1.0
)


# ============================================================
# SAVE DIFFERENCE DIAGNOSTICS
# ============================================================

Image.fromarray(
    (
        ABS_DIFF_VIS
        *
        255
    ).astype(
        np.uint8
    )
).save(
    OUT
    / "02_global_difference_diagnostic.png"
)


Image.fromarray(
    (
        ROI_DIFF_VIS
        *
        255
    ).astype(
        np.uint8
    )
).save(
    OUT
    / "03_roi_difference_diagnostic.png"
)


# ============================================================
# OVERLAY ROI ON HELPER
# ============================================================

helper_roi_audit = helper_np.copy()


helper_roi_audit[
    p01_roi
] = (
    helper_roi_audit[
        p01_roi
    ] * 0.70
    +
    np.array(
        [255, 0, 0],
        dtype=np.float32
    ) * 0.30
)


helper_roi_audit[
    p02_roi
] = (
    helper_roi_audit[
        p02_roi
    ] * 0.70
    +
    np.array(
        [0, 255, 255],
        dtype=np.float32
    ) * 0.30
)


helper_roi_audit = np.clip(
    helper_roi_audit,
    0,
    255
).astype(
    np.uint8
)


# ============================================================
# 6-PANEL AUDIT
# ============================================================

fig, axes = plt.subplots(
    1,
    6,
    figsize=(
        26,
        7
    )
)


axes[0].imshow(
    master
)

axes[0].set_title(
    "1. Original Stage01"
)

axes[0].axis("off")


axes[1].imshow(
    guide
)

axes[1].set_title(
    "2. Shadow ROI Guidance\n"
    "RED=P01 | CYAN=P02"
)

axes[1].axis("off")


axes[2].imshow(
    helper
)

axes[2].set_title(
    "3. Qwen Shadow-Free Helper"
)

axes[2].axis("off")


axes[3].imshow(
    helper_roi_audit
)

axes[3].set_title(
    "4. Helper + Analysis ROIs"
)

axes[3].axis("off")


axes[4].imshow(
    ABS_DIFF_VIS,
    cmap="gray",
    vmin=0,
    vmax=1
)

axes[4].set_title(
    "5. Global Difference"
)

axes[4].axis("off")


axes[5].imshow(
    ROI_DIFF_VIS,
    cmap="gray",
    vmin=0,
    vmax=1
)

axes[5].set_title(
    "6. ROI Difference Only"
)

axes[5].axis("off")


plt.tight_layout()


AUDIT_PATH = (
    OUT
    / "04_stage07e4_shadow_free_helper_audit.png"
)


plt.savefig(
    AUDIT_PATH,
    dpi=150,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# DIFFERENCE STATISTICS
# ============================================================

global_mean_diff = float(
    abs_diff.mean()
)


if combined_roi.any():

    roi_mean_diff = float(
        abs_diff[
            combined_roi
        ].mean()
    )

else:

    roi_mean_diff = 0.0


outside = (
    ~combined_roi
)


outside_mean_diff = float(
    abs_diff[
        outside
    ].mean()
)


print()
print("=" * 110)
print("STAGE 07E4 RESULT")
print("=" * 110)

print()

print(
    "GLOBAL MEAN RGB DIFFERENCE:",
    round(
        global_mean_diff,
        3
    )
)

print(
    "ROI MEAN RGB DIFFERENCE:",
    round(
        roi_mean_diff,
        3
    )
)

print(
    "OUTSIDE-ROI MEAN RGB DIFFERENCE:",
    round(
        outside_mean_diff,
        3
    )
)

print()

print(
    "HELPER:",
    HELPER_PATH
)

print(
    "AUDIT:",
    AUDIT_PATH
)


# ============================================================
# STATE
# ============================================================

STATE = {

    "stage":
        "07E4",

    "purpose":
        "PRE_TILE_AI_SHADOW_FREE_ANALYTICAL_HELPER",

    "model":
        MODEL_ID,

    "runs_before_tile_selection":
        True,

    "used_as_final_rgb":
        False,

    "user_runtime_ai":
        False,

    "statistics": {

        "global_mean_rgb_difference":
            global_mean_diff,

        "roi_mean_rgb_difference":
            roi_mean_diff,

        "outside_roi_mean_rgb_difference":
            outside_mean_diff
    },

    "outputs": {

        "guidance":
            str(
                GUIDE_PATH
            ),

        "helper":
            str(
                HELPER_PATH
            ),

        "audit":
            str(
                AUDIT_PATH
            )
    },

    "status":
        "RND_REQUIRES_HELPER_VISUAL_AUDIT",

    "next_if_pass":
        "07E5_LOCAL_REGISTER_AND_DERIVE_SHADOW_MULTIPLIER"
}


STATE_PATH = (
    OUT
    / "00_stage07e4_result.json"
)


STATE_PATH.write_text(
    json.dumps(
        STATE,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


print(
    "STATE:",
    STATE_PATH
)

print()
print(
    "IMPORTANT:"
)

print(
    "This helper is NOT production/final RGB."
)

print(
    "It exists only to estimate reusable shadows before tile selection."
)


# ============================================================
# CLEANUP
# ============================================================

del pipe
del result

gc.collect()

if torch.cuda.is_available():

    torch.cuda.empty_cache()
