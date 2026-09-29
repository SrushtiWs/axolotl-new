
from pathlib import Path
import json
import re
import gc

import numpy as np
import torch

from PIL import (
    Image,
    ImageDraw
)

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


MASTER_PATH = (
    PROD
    / "stage01_master"
    / "00_master_input.png"
)


FLOOR_MASK_PATH = (
    PROD
    / "stage08_tile_application"
    / "08b2_floor_plus_screeding_target"
    / "01_main_floor_mask.png"
)


P01_MASK_PATH = (
    PROD
    / "stage06_prop_layer"
    / "06f3_vanity_internal_completion"
    / "01_p01_completed_vanity_mask.png"
)


P02_MASK_PATH = (
    PROD
    / "stage06_prop_layer"
    / "06d2c1_complete_main_prop_sam2_multimask_audit"
    / "objects"
    / "P02_candidate_3_mask.png"
)


OUT = (
    PROD
    / "stage07_empty_room"
    / "07e1_qwen_semantic_shadow_localization"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


GUIDANCE_PATH = (
    OUT
    / "00_shadow_guidance_image.png"
)


RAW_RESPONSE_PATH = (
    OUT
    / "01_qwen_shadow_raw.txt"
)


PARSED_PATH = (
    OUT
    / "02_qwen_shadow_parsed.json"
)


# ============================================================
# MODEL
# ============================================================

MODEL_ID = (
    "Qwen/Qwen2.5-VL-7B-Instruct"
)


CACHE = (
    "/workspace/data/huggingface-cache"
)


MAX_NEW_TOKENS = 1200


# ============================================================
# VALIDATE
# ============================================================

for p in [
    MASTER_PATH,
    FLOOR_MASK_PATH,
    P01_MASK_PATH,
    P02_MASK_PATH,
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


floor = (
    np.asarray(
        Image.open(
            FLOOR_MASK_PATH
        ).convert("L")
    )
    >
    127
)


p01 = (
    np.asarray(
        Image.open(
            P01_MASK_PATH
        ).convert("L")
    )
    >
    127
)


p02 = (
    np.asarray(
        Image.open(
            P02_MASK_PATH
        ).convert("L")
    )
    >
    127
)


# ============================================================
# CREATE GUIDANCE IMAGE
#
# Qwen receives:
#
# IMAGE 1:
#   original room
#
# IMAGE 2:
#   same room with:
#
#   green = selectable floor
#   red   = P01 vanity
#   cyan  = P02 toilet
#
# This gives semantic + spatial evidence WITHOUT manual input.
# ============================================================

guide = master_np.astype(
    np.float32
).copy()


# Floor green
guide[
    floor
] = (
    guide[
        floor
    ] * 0.60
    +
    np.array(
        [0, 255, 0],
        dtype=np.float32
    ) * 0.40
)


# P01 red
guide[
    p01
] = (
    guide[
        p01
    ] * 0.25
    +
    np.array(
        [255, 0, 0],
        dtype=np.float32
    ) * 0.75
)


# P02 cyan
guide[
    p02
] = (
    guide[
        p02
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


Image.fromarray(
    guide
).save(
    GUIDANCE_PATH
)


# ============================================================
# PROMPT
# ============================================================

PROMPT = r"""
You are performing PRE-TILE shadow understanding for a room
visualization system.

You receive TWO images of the exact same room.

IMAGE 1:
The original room photograph.

IMAGE 2:
A guidance image.

In IMAGE 2:

GREEN = confirmed visible FLOOR pixels.

RED = P01, the complete vanity system.

CYAN = P02, the complete toilet system.

Your job is ONLY to identify visible FLOOR CAST SHADOWS and
FLOOR CONTACT SHADOWS physically caused by P01 and P02.

============================================================
CRITICAL DISTINCTION
============================================================

A dark region is NOT automatically a shadow.

DO NOT classify any of these as shadow:

- floor material design
- wood grain
- tile pattern
- natural color variation
- grout
- floor seams
- general room illumination gradient
- reflections
- specular highlights
- architectural edges
- wall shadows
- baseboard or screeding
- prop pixels themselves
- glass effects
- mirror reflections

Only identify darkness that is spatially and physically
consistent with a cast/contact shadow from the specified prop.

============================================================
P01
============================================================

P01 is the RED vanity system.

Identify only the visible shadow it casts onto the GREEN floor.

Do not include dark floor material merely because it lies near
the vanity.

If the true visible P01 shadow cannot be confidently separated
from floor design, be conservative.

============================================================
P02
============================================================

P02 is the CYAN toilet system.

Identify the visible soft contact/cast floor shadow underneath
and beside the toilet.

Again, include only actual shadow.

============================================================
COORDINATE SYSTEM
============================================================

Return polygon coordinates normalized to the range 0..1000.

x=0 is left edge.
x=1000 is right edge.

y=0 is top edge.
y=1000 is bottom edge.

Polygons must describe ONLY visible shadow regions.

Multiple polygons are allowed.

============================================================
OUTPUT
============================================================

Return STRICT JSON ONLY.

No markdown.
No explanation before or after JSON.

Use exactly this structure:

{
  "P01": {
    "shadow_visible": true,
    "confidence": 0.0,
    "polygons": [
      [[x,y],[x,y],[x,y]]
    ],
    "reason": "short reason"
  },
  "P02": {
    "shadow_visible": true,
    "confidence": 0.0,
    "polygons": [
      [[x,y],[x,y],[x,y]]
    ],
    "reason": "short reason"
  }
}

If a shadow is not confidently visible:

"shadow_visible": false
"polygons": []

Do not invent polygons.
"""


# ============================================================
# LOAD QWEN
#
# Uses exact proven server pattern.
# ============================================================

from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)

from qwen_vl_utils import (
    process_vision_info,
)


print("=" * 110)
print("STAGE 07E1 — QWEN SEMANTIC SHADOW LOCALIZATION")
print("=" * 110)

print()
print(
    "MASTER:",
    MASTER_PATH
)

print(
    "GUIDANCE:",
    GUIDANCE_PATH
)

print()
print(
    "Loading Qwen2.5-VL-7B..."
)


processor = (
    AutoProcessor
    .from_pretrained(
        MODEL_ID,
        cache_dir=CACHE
    )
)


model = (
    Qwen2_5_VLForConditionalGeneration
    .from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=CACHE,
        low_cpu_mem_usage=True,
    )
)


model.eval()


print(
    "✅ QWEN READY"
)


# ============================================================
# INPUT
# ============================================================

messages = [
    {
        "role": "user",

        "content": [

            {
                "type": "image",
                "image": str(
                    MASTER_PATH
                ),
            },

            {
                "type": "image",
                "image": str(
                    GUIDANCE_PATH
                ),
            },

            {
                "type": "text",
                "text": PROMPT,
            },
        ],
    }
]


chat_text = (
    processor
    .apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
)


image_inputs, video_inputs = (
    process_vision_info(
        messages
    )
)


inputs = processor(
    text=[
        chat_text
    ],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt"
)


inputs = inputs.to(
    model.device
)


# ============================================================
# GENERATE
# ============================================================

print()
print(
    "Running semantic shadow localization..."
)


with torch.inference_mode():

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        repetition_penalty=1.03,
    )


trimmed_ids = [
    output_ids[
        len(input_ids):
    ]

    for input_ids, output_ids

    in zip(
        inputs.input_ids,
        generated_ids
    )
]


response = processor.batch_decode(
    trimmed_ids,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)[0]


RAW_RESPONSE_PATH.write_text(
    response,
    encoding="utf-8"
)


print()
print("=" * 110)
print("RAW QWEN RESPONSE")
print("=" * 110)

print(
    response
)


# ============================================================
# PARSE JSON ROBUSTLY
# ============================================================

def extract_json(text):

    text = text.strip()


    # Remove markdown fences if model ignored instruction.
    text = re.sub(
        r"^```(?:json)?",
        "",
        text,
        flags=re.I
    ).strip()


    text = re.sub(
        r"```$",
        "",
        text
    ).strip()


    start = text.find("{")
    end = text.rfind("}")


    if start < 0 or end < 0:

        raise RuntimeError(
            "Qwen response contains no JSON object."
        )


    candidate = text[
        start:end+1
    ]


    return json.loads(
        candidate
    )


parsed = extract_json(
    response
)


PARSED_PATH.write_text(
    json.dumps(
        parsed,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# ============================================================
# NORMALIZED POLYGON -> IMAGE MASK
# ============================================================

def polygons_to_mask(
    record
):

    canvas = Image.new(
        "L",
        (
            W,
            H
        ),
        0
    )


    draw = ImageDraw.Draw(
        canvas
    )


    if not record.get(
        "shadow_visible",
        False
    ):

        return np.zeros(
            (
                H,
                W
            ),
            dtype=bool
        )


    polygons = record.get(
        "polygons",
        []
    )


    for poly in polygons:

        if not isinstance(
            poly,
            list
        ):

            continue


        pts = []


        for point in poly:

            if (
                not isinstance(
                    point,
                    list
                )
                or
                len(point) < 2
            ):
                continue


            try:
                nx = float(
                    point[0]
                )

                ny = float(
                    point[1]
                )

            except Exception:
                continue


            nx = np.clip(
                nx,
                0.0,
                1000.0
            )


            ny = np.clip(
                ny,
                0.0,
                1000.0
            )


            x = int(
                round(
                    nx
                    /
                    1000.0
                    *
                    (W - 1)
                )
            )


            y = int(
                round(
                    ny
                    /
                    1000.0
                    *
                    (H - 1)
                )
            )


            pts.append(
                (
                    x,
                    y
                )
            )


        if len(
            pts
        ) >= 3:

            draw.polygon(
                pts,
                fill=255
            )


    mask = (
        np.asarray(
            canvas
        )
        >
        127
    )


    # Absolutely never allow shadow outside floor.
    mask &= floor


    return mask


p01_shadow = polygons_to_mask(
    parsed.get(
        "P01",
        {}
    )
)


p02_shadow = polygons_to_mask(
    parsed.get(
        "P02",
        {}
    )
)


# ============================================================
# REMOVE PROP PIXELS
# ============================================================

p01_shadow &= ~p01
p01_shadow &= ~p02

p02_shadow &= ~p01
p02_shadow &= ~p02


# ============================================================
# GEOMETRY SAFETY FILTER
#
# AI gives semantic shadow location.
#
# We still require proximity to the object.
#
# This protects against Qwen accidentally selecting a distant
# dark region.
# ============================================================

def proximity_region(
    prop,
    radius
):

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            radius * 2 + 1,
            radius * 2 + 1
        )
    )


    region = (
        cv2.dilate(
            prop.astype(
                np.uint8
            )
            *
            255,
            kernel
        )
        >
        0
    )


    region &= floor


    return region


import cv2


p01_allowed = proximity_region(
    p01,
    radius=70
)


p02_allowed = proximity_region(
    p02,
    radius=55
)


p01_shadow &= p01_allowed

p02_shadow &= p02_allowed


# ============================================================
# CLEAN POLYGON EDGES SLIGHTLY
#
# We are NOT estimating final shadow softness yet.
# This stage only localizes semantic support.
# ============================================================

def clean_mask(
    mask
):

    u8 = (
        mask.astype(
            np.uint8
        )
        *
        255
    )


    u8 = cv2.morphologyEx(
        u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                5,
                5
            )
        )
    )


    return (
        u8
        >
        0
    )


p01_shadow = clean_mask(
    p01_shadow
)


p02_shadow = clean_mask(
    p02_shadow
)


combined = (
    p01_shadow
    |
    p02_shadow
)


# ============================================================
# SAVE MASKS
# ============================================================

P01_OUT = (
    OUT
    / "03_p01_qwen_shadow_mask.png"
)


P02_OUT = (
    OUT
    / "04_p02_qwen_shadow_mask.png"
)


COMBINED_OUT = (
    OUT
    / "05_combined_qwen_shadow_mask.png"
)


Image.fromarray(
    p01_shadow.astype(
        np.uint8
    )
    *
    255
).save(
    P01_OUT
)


Image.fromarray(
    p02_shadow.astype(
        np.uint8
    )
    *
    255
).save(
    P02_OUT
)


Image.fromarray(
    combined.astype(
        np.uint8
    )
    *
    255
).save(
    COMBINED_OUT
)


# ============================================================
# OVERLAY
# ============================================================

overlay = master_np.astype(
    np.float32
).copy()


overlay[
    p01_shadow
] = (
    overlay[
        p01_shadow
    ]
    *
    0.38
    +
    np.array(
        [
            255,
            0,
            0
        ],
        dtype=np.float32
    )
    *
    0.62
)


overlay[
    p02_shadow
] = (
    overlay[
        p02_shadow
    ]
    *
    0.38
    +
    np.array(
        [
            0,
            255,
            255
        ],
        dtype=np.float32
    )
    *
    0.62
)


overlay = np.clip(
    overlay,
    0,
    255
).astype(
    np.uint8
)


OVERLAY_OUT = (
    OUT
    / "06_qwen_shadow_location_overlay.png"
)


Image.fromarray(
    overlay
).save(
    OVERLAY_OUT
)


# ============================================================
# NEUTRAL TRANSFER TEST
#
# Binary semantic mask only at this stage.
#
# Use fixed provisional multiplier purely to visualize whether
# the detected regions are spatially correct.
# ============================================================

test_multiplier = np.ones(
    (
        H,
        W
    ),
    dtype=np.float32
)


test_multiplier[
    p01_shadow
] = 0.78


test_multiplier[
    p02_shadow
] = 0.72


neutral = np.full(
    (
        H,
        W,
        3
    ),
    210.0,
    dtype=np.float32
)


neutral[
    floor
] *= test_multiplier[
        floor,
        None
    ]


neutral = np.clip(
    neutral,
    0,
    255
).astype(
    np.uint8
)


# ============================================================
# 7-PANEL AUDIT
# ============================================================

fig, axes = plt.subplots(
    1,
    7,
    figsize=(
        30,
        7
    )
)


axes[0].imshow(
    master_np
)

axes[0].set_title(
    "1. Original"
)

axes[0].axis(
    "off"
)


axes[1].imshow(
    guide
)

axes[1].set_title(
    "2. AI Guidance\n"
    "GREEN=floor RED=P01 CYAN=P02"
)

axes[1].axis(
    "off"
)


axes[2].imshow(
    p01_shadow,
    cmap="gray"
)

axes[2].set_title(
    "3. Qwen P01\nVanity Shadow"
)

axes[2].axis(
    "off"
)


axes[3].imshow(
    p02_shadow,
    cmap="gray"
)

axes[3].set_title(
    "4. Qwen P02\nToilet Shadow"
)

axes[3].axis(
    "off"
)


axes[4].imshow(
    combined,
    cmap="gray"
)

axes[4].set_title(
    "5. Combined"
)

axes[4].axis(
    "off"
)


axes[5].imshow(
    overlay
)

axes[5].set_title(
    "6. Location Audit"
)

axes[5].axis(
    "off"
)


axes[6].imshow(
    neutral
)

axes[6].set_title(
    "7. Neutral Transfer\n"
    "Position Test Only"
)

axes[6].axis(
    "off"
)


plt.tight_layout()


AUDIT_OUT = (
    OUT
    / "07_stage07e1_qwen_shadow_audit.png"
)


plt.savefig(
    AUDIT_OUT,
    dpi=150,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# STATE
# ============================================================

p01_record = parsed.get(
    "P01",
    {}
)


p02_record = parsed.get(
    "P02",
    {}
)


STATE = {

    "stage":
        "07E1",

    "architecture":
        "PRE_TILE_SEMANTIC_SHADOW_LOCALIZATION",

    "model":
        MODEL_ID,

    "runs_before_tile_selection":
        True,

    "modifies_room_rgb":
        False,

    "qwen_role":
        "SEMANTIC_SHADOW_LOCATION_ONLY",

    "deterministic_postprocessing": [
        "normalized polygon conversion",
        "floor-mask clipping",
        "prop-mask exclusion",
        "object-proximity safety filtering",
        "small morphological cleanup"
    ],

    "qwen": {

        "P01": {

            "shadow_visible":
                p01_record.get(
                    "shadow_visible"
                ),

            "confidence":
                p01_record.get(
                    "confidence"
                ),

            "reason":
                p01_record.get(
                    "reason"
                ),
        },

        "P02": {

            "shadow_visible":
                p02_record.get(
                    "shadow_visible"
                ),

            "confidence":
                p02_record.get(
                    "confidence"
                ),

            "reason":
                p02_record.get(
                    "reason"
                ),
        },
    },

    "statistics": {

        "p01_shadow_pixels":
            int(
                p01_shadow.sum()
            ),

        "p02_shadow_pixels":
            int(
                p02_shadow.sum()
            ),

        "combined_shadow_pixels":
            int(
                combined.sum()
            ),
    },

    "outputs": {

        "guidance":
            str(
                GUIDANCE_PATH
            ),

        "raw_qwen":
            str(
                RAW_RESPONSE_PATH
            ),

        "parsed":
            str(
                PARSED_PATH
            ),

        "p01_mask":
            str(
                P01_OUT
            ),

        "p02_mask":
            str(
                P02_OUT
            ),

        "combined":
            str(
                COMBINED_OUT
            ),

        "overlay":
            str(
                OVERLAY_OUT
            ),

        "audit":
            str(
                AUDIT_OUT
            ),
    },

    "status":
        "RND_REQUIRES_VISUAL_AUDIT",

    "next_if_location_passes":
        (
            "07E2 derive continuous shadow-strength "
            "multiplier inside semantic regions"
        ),
}


STATE_PATH = (
    OUT
    / "00_stage07e1_result.json"
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
# PRINT RESULT
# ============================================================

print()
print("=" * 110)
print("STAGE 07E1 RESULT")
print("=" * 110)

print()

print(
    "P01 VISIBLE:",
    p01_record.get(
        "shadow_visible"
    )
)

print(
    "P01 CONFIDENCE:",
    p01_record.get(
        "confidence"
    )
)

print(
    "P01 PIXELS:",
    int(
        p01_shadow.sum()
    )
)

print()

print(
    "P02 VISIBLE:",
    p02_record.get(
        "shadow_visible"
    )
)

print(
    "P02 CONFIDENCE:",
    p02_record.get(
        "confidence"
    )
)

print(
    "P02 PIXELS:",
    int(
        p02_shadow.sum()
    )
)

print()

print(
    "COMBINED PIXELS:",
    int(
        combined.sum()
    )
)

print()

print(
    "AUDIT:",
    AUDIT_OUT
)

print(
    "RAW RESPONSE:",
    RAW_RESPONSE_PATH
)

print(
    "STATE:",
    STATE_PATH
)

print()
print(
    "NO IMAGE EDIT MODEL WAS USED."
)

print(
    "QWEN PROVIDED SEMANTIC LOCATION ONLY."
)

print(
    "NO TILE WAS SELECTED OR MODIFIED."
)


# ============================================================
# CLEANUP
# ============================================================

del model
del processor
del inputs
del generated_ids

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()
