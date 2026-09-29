
from pathlib import Path
import json
import re

import torch
from PIL import Image, ImageDraw

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)

from qwen_vl_utils import process_vision_info


BASE = Path("/workspace/axolotl")

PROD = (
    BASE
    / "test07"
    / "production_pipeline"
)

STAGE06 = (
    PROD
    / "stage06_prop_layer"
)

MASTER = (
    PROD
    / "stage01_master"
    / "00_master_input.png"
)

INVENTORY_JSON = (
    STAGE06
    / "06c2_lost_instance_existence_audit"
    / "00a_physical_inventory_bridge.json"
)

OUT = (
    STAGE06
    / "06c9b_multicandidate_grid_relocalization"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_CACHE = Path(
    "/workspace/data/huggingface-cache"
)

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

TARGET_ID = 21

ROWS = 4
COLS = 4

CROP_EXPANSION = 0.20


# ============================================================
# HELPERS
# ============================================================

def grid_boxes(W, H):

    result = []

    n = 1

    for r in range(ROWS):

        for c in range(COLS):

            x1 = W * c / COLS
            y1 = H * r / ROWS

            x2 = W * (c + 1) / COLS
            y2 = H * (r + 1) / ROWS

            result.append({
                "cell": n,
                "bbox": [
                    x1,
                    y1,
                    x2,
                    y2
                ]
            })

            n += 1

    return result


def draw_grid(image):

    image = image.copy()

    draw = ImageDraw.Draw(image)

    W, H = image.size

    boxes = grid_boxes(
        W,
        H
    )

    for row in boxes:

        n = row["cell"]

        x1, y1, x2, y2 = row["bbox"]

        draw.rectangle(
            [
                x1,
                y1,
                x2,
                y2
            ],
            outline="red",
            width=3
        )

        draw.rectangle(
            [
                x1 + 3,
                y1 + 3,
                x1 + 32,
                y1 + 24
            ],
            fill="white",
            outline="red"
        )

        draw.text(
            (
                x1 + 8,
                y1 + 5
            ),
            str(n),
            fill="red"
        )

    return image, boxes


def expand_box(
    bbox,
    ratio,
    W,
    H
):

    x1, y1, x2, y2 = bbox

    bw = x2 - x1
    bh = y2 - y1

    return [
        max(
            0,
            x1 - bw * ratio
        ),
        max(
            0,
            y1 - bh * ratio
        ),
        min(
            W,
            x2 + bw * ratio
        ),
        min(
            H,
            y2 + bh * ratio
        ),
    ]


def parse_top3(text):

    nums = [
        int(x)
        for x in re.findall(
            r"\b(?:[1-9]|1[0-6])\b",
            text
        )
    ]

    unique = []

    for n in nums:

        if n not in unique:
            unique.append(n)

    return unique[:3]


def qwen(
    images,
    prompt,
    model,
    processor
):

    content = []

    for img in images:

        content.append({
            "type": "image",
            "image": str(img)
        })

    content.append({
        "type": "text",
        "text": prompt
    })


    messages = [{
        "role": "user",
        "content": content
    }]


    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


    image_inputs, video_inputs = (
        process_vision_info(
            messages
        )
    )


    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )


    inputs = inputs.to(
        model.device
    )


    with torch.inference_mode():

        output = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False
        )


    trimmed = [

        out[
            len(inp):
        ]

        for inp, out
        in zip(
            inputs.input_ids,
            output
        )
    ]


    return processor.batch_decode(
        trimmed,
        skip_special_tokens=True
    )[0]


# ============================================================
# INPUT
# ============================================================

master = Image.open(
    MASTER
).convert("RGB")

W, H = master.size


inventory = json.loads(
    INVENTORY_JSON.read_text(
        encoding="utf-8"
    )
)["objects"]


obj = next(
    r
    for r in inventory
    if int(r["id"]) == TARGET_ID
)


name = obj.get(
    "name",
    "drawer handle"
)

phrase = obj.get(
    "grounding_phrase",
    ""
)


# ============================================================
# MODEL
# ============================================================

dtype = (
    torch.float16
    if torch.cuda.is_available()
    else torch.float32
)


model = (
    Qwen2_5_VLForConditionalGeneration
    .from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto",
        cache_dir=str(MODEL_CACHE)
    )
)

model.eval()


processor = (
    AutoProcessor
    .from_pretrained(
        MODEL_ID,
        cache_dir=str(MODEL_CACHE)
    )
)


# ============================================================
# PASS 1 — TOP 3 CELLS
# ============================================================

grid_img, boxes = draw_grid(
    master
)

GRID_PATH = (
    OUT
    / "021_drawer_handle_top3_grid.png"
)

grid_img.save(
    GRID_PATH
)


prompt1 = f"""
The target physical object definitely exists in this room.

TARGET:
{name}

DESCRIPTION:
{phrase}

The image is divided into 16 numbered cells.

Identify the THREE cells most likely to contain the actual
drawer handle.

Important:
- Find the small physical handle attached to a drawer/cabinet.
- Do not choose the whole cabinet merely because it contains
  drawers.
- Do not confuse it with a faucet, basin, bottle, towel bar,
  door handle, or other fixture.
- Rank the three best possibilities.

Return exactly:

CELLS: <best>, <second>, <third>
"""


raw1 = qwen(
    [GRID_PATH],
    prompt1,
    model,
    processor
)


top3 = parse_top3(
    raw1
)


print("=" * 100)
print("06C9B DRAWER HANDLE")
print("=" * 100)

print(
    "QWEN RAW TOP3:",
    raw1
)

print(
    "TOP3 CELLS:",
    top3
)


if len(top3) < 3:

    raise RuntimeError(
        "Could not parse three candidate cells."
    )


# ============================================================
# MAKE THREE HIGH-CONTEXT CROPS
# ============================================================

crop_paths = []

candidate_regions = []


for rank, cell in enumerate(
    top3,
    start=1
):

    bbox = boxes[
        cell - 1
    ][
        "bbox"
    ]


    region = expand_box(
        bbox,
        CROP_EXPANSION,
        W,
        H
    )


    x1, y1, x2, y2 = [
        int(round(v))
        for v in region
    ]


    crop = master.crop(
        (
            x1,
            y1,
            x2,
            y2
        )
    )


    draw = ImageDraw.Draw(
        crop
    )

    draw.rectangle(
        [
            2,
            2,
            48,
            30
        ],
        fill="white",
        outline="red"
    )

    draw.text(
        (
            10,
            7
        ),
        str(rank),
        fill="red"
    )


    path = (
        OUT
        /
        f"021_candidate_{rank}_cell_{cell}.png"
    )


    crop.save(
        path
    )


    crop_paths.append(
        path
    )


    candidate_regions.append({

        "candidate_rank":
            rank,

        "source_cell":
            cell,

        "region":
            region,

        "image":
            str(path)
    })


# ============================================================
# PASS 2 — COMPARE THREE CROPS
# ============================================================

prompt2 = f"""
The target is:

{name}

DESCRIPTION:
{phrase}

You are shown THREE candidate room crops in order:
IMAGE 1, IMAGE 2, IMAGE 3.

Choose which image most clearly contains the actual small
drawer handle itself.

A drawer handle is the physical knob, pull, grip, or handle
attached to the front face of a drawer or cabinet door.

Do NOT select an image merely because it contains a large
cabinet.

Do NOT confuse:
- sink or basin
- faucet
- bottle
- door
- towel bar
- countertop

Return exactly:

IMAGE: 1

or

IMAGE: 2

or

IMAGE: 3
"""


raw2 = qwen(
    crop_paths,
    prompt2,
    model,
    processor
)


match = re.search(
    r"IMAGE:\s*([123])",
    raw2.upper()
)


selected = (
    int(
        match.group(1)
    )
    if match
    else None
)


print()
print(
    "QWEN RAW COMPARISON:",
    raw2
)

print(
    "SELECTED IMAGE:",
    selected
)


selected_region = None

if selected is not None:

    selected_region = (
        candidate_regions[
            selected - 1
        ]
    )


# ============================================================
# SAVE
# ============================================================

result = {

    "inventory_id":
        TARGET_ID,

    "inventory_name":
        name,

    "grounding_phrase":
        phrase,

    "top3_cells":
        top3,

    "comparison_raw":
        raw2,

    "selected_candidate":
        selected,

    "selected_region":
        selected_region,

    "candidate_regions":
        candidate_regions,
}


RESULT_PATH = (
    OUT
    / "00_multicandidate_result.json"
)


RESULT_PATH.write_text(
    json.dumps(
        result,
        indent=2
    ),
    encoding="utf-8"
)


print()
print("=" * 100)
print("PRODUCTION STAGE 06C9B RESULT")
print("=" * 100)

print(
    "TOP3:",
    top3
)

print(
    "SELECTED:",
    selected
)

print(
    "SELECTED CELL:",
    (
        selected_region[
            "source_cell"
        ]
        if selected_region
        else None
    )
)

print(
    "SELECTED REGION:",
    (
        [
            round(v, 1)
            for v
            in selected_region[
                "region"
            ]
        ]
        if selected_region
        else None
    )
)

print()
print(
    "RESULT:",
    RESULT_PATH
)
