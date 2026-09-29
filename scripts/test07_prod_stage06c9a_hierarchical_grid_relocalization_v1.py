
from pathlib import Path
import json
import re

import torch
from PIL import Image, ImageDraw

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)

from qwen_vl_utils import (
    process_vision_info
)


# ============================================================
# PATHS
# ============================================================

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
    / "06c9a_hierarchical_grid_relocalization"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

GRID_DIR = (
    OUT
    / "grid_images"
)

GRID_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_CACHE = Path(
    "/workspace/data/huggingface-cache"
)


# ============================================================
# CONFIG
# ============================================================

MODEL_ID = (
    "Qwen/Qwen2.5-VL-7B-Instruct"
)

MAX_NEW_TOKENS = 40


TARGET_IDS = [
    10,
    21,
    25,
]


COARSE_ROWS = 4
COARSE_COLS = 4

FINE_ROWS = 3
FINE_COLS = 3

COARSE_CONTEXT_EXPANSION = 0.20


# ============================================================
# PROMPTS
# ============================================================

COARSE_PROMPT = """
The target physical object is already confirmed to exist
somewhere in this real room image.

TARGET:
{name}

DESCRIPTION:
{phrase}

The complete image is divided into numbered grid cells.

Choose the ONE numbered cell that contains the target object
or contains the largest/most useful visible part of it.

Rules:

1. Identify the actual physical object, not a nearby object.
2. A sink is not a faucet.
3. A cabinet is not a drawer handle.
4. Use complete scene context.
5. Do not invent coordinates.
6. If the object crosses cells, choose the cell containing its
   most distinctive or largest visible portion.
7. Return exactly:

CELL: <number>
""".strip()


FINE_PROMPT = """
The target physical object is already confirmed to exist.

TARGET:
{name}

DESCRIPTION:
{phrase}

This image is a localized portion of the original room.
It is divided into numbered cells.

Choose the ONE numbered cell that most directly contains the
target physical object.

Rules:

1. Select the target itself, not nearby furniture or fixtures.
2. A sink is not a faucet.
3. A cabinet is not a drawer handle.
4. Do not invent another location.
5. Return exactly:

CELL: <number>
""".strip()


# ============================================================
# HELPERS
# ============================================================

def safe_name(text):

    return "".join(
        c
        if c.isalnum() or c in "-_"
        else "_"
        for c in str(text)
    )[:60]


def parse_cell(raw, maximum):

    m = re.search(
        r"CELL:\s*(\d+)",
        raw.upper()
    )

    if not m:
        return None

    value = int(
        m.group(1)
    )

    if (
        value < 1
        or
        value > maximum
    ):
        return None

    return value


def grid_boxes(
    bbox,
    rows,
    cols,
):

    x1, y1, x2, y2 = [
        float(v)
        for v in bbox
    ]

    width = (
        x2 - x1
    )

    height = (
        y2 - y1
    )

    boxes = []

    number = 1

    for r in range(rows):

        for c in range(cols):

            bx1 = (
                x1
                +
                width
                *
                c
                /
                cols
            )

            by1 = (
                y1
                +
                height
                *
                r
                /
                rows
            )

            bx2 = (
                x1
                +
                width
                *
                (
                    c + 1
                )
                /
                cols
            )

            by2 = (
                y1
                +
                height
                *
                (
                    r + 1
                )
                /
                rows
            )

            boxes.append({

                "cell":
                    number,

                "bbox": [
                    bx1,
                    by1,
                    bx2,
                    by2
                ],
            })

            number += 1

    return boxes


def draw_grid(
    image,
    region_bbox,
    rows,
    cols,
):

    image = image.copy()

    draw = ImageDraw.Draw(
        image
    )

    boxes = grid_boxes(
        region_bbox,
        rows,
        cols
    )

    for item in boxes:

        number = item[
            "cell"
        ]

        x1, y1, x2, y2 = item[
            "bbox"
        ]

        draw.rectangle(
            [
                x1,
                y1,
                x2,
                y2
            ],
            outline="red",
            width=2
        )

        label_x = int(
            x1 + 3
        )

        label_y = int(
            y1 + 3
        )

        draw.rectangle(
            [
                label_x,
                label_y,
                label_x + 24,
                label_y + 18
            ],
            fill="white",
            outline="red"
        )

        draw.text(
            (
                label_x + 5,
                label_y + 2
            ),
            str(number),
            fill="red"
        )

    return image, boxes


def expand_bbox(
    bbox,
    ratio,
    W,
    H
):

    x1, y1, x2, y2 = [
        float(v)
        for v in bbox
    ]

    bw = (
        x2 - x1
    )

    bh = (
        y2 - y1
    )

    return [

        max(
            0.0,
            x1
            -
            bw * ratio
        ),

        max(
            0.0,
            y1
            -
            bh * ratio
        ),

        min(
            float(W),
            x2
            +
            bw * ratio
        ),

        min(
            float(H),
            y2
            +
            bh * ratio
        ),
    ]


def run_qwen_choice(
    image_path,
    prompt,
    processor,
    model,
    maximum,
):

    messages = [
        {
            "role":
                "user",

            "content": [
                {
                    "type":
                        "image",

                    "image":
                        str(image_path)
                },

                {
                    "type":
                        "text",

                    "text":
                        prompt
                }
            ]
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


    with torch.inference_mode():

        generated = model.generate(
            **inputs,
            max_new_tokens=
                MAX_NEW_TOKENS,
            do_sample=False
        )


    trimmed = [

        output_ids[
            len(
                input_ids
            ):
        ]

        for input_ids, output_ids
        in zip(
            inputs.input_ids,
            generated
        )
    ]


    raw = (
        processor
        .batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
    )


    return (
        parse_cell(
            raw,
            maximum
        ),
        raw
    )


# ============================================================
# LOAD
# ============================================================

for path in [
    MASTER,
    INVENTORY_JSON,
]:

    if not path.exists():

        raise FileNotFoundError(
            path
        )


master = Image.open(
    MASTER
).convert(
    "RGB"
)

W, H = master.size


inventory = json.loads(
    INVENTORY_JSON.read_text(
        encoding="utf-8"
    )
)["objects"]


inventory_by_id = {

    int(
        row["id"]
    ):
        row

    for row in inventory
}


# ============================================================
# MODEL
# ============================================================

dtype = (
    torch.float16
    if torch.cuda.is_available()
    else torch.float32
)


print("=" * 110)
print("PRODUCTION STAGE 06C9A")
print("HIERARCHICAL VISUAL-GRID RELOCALIZATION")
print("=" * 110)


print()
print(
    "Loading Qwen2.5-VL-7B..."
)


model = (
    Qwen2_5_VLForConditionalGeneration
    .from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto",
        cache_dir=str(
            MODEL_CACHE
        )
    )
)

model.eval()


processor = (
    AutoProcessor
    .from_pretrained(
        MODEL_ID,
        cache_dir=str(
            MODEL_CACHE
        )
    )
)


print(
    "✅ QWEN READY"
)


# ============================================================
# PROCESS
# ============================================================

results = []


FULL_REGION = [
    0.0,
    0.0,
    float(W),
    float(H),
]


for iid in TARGET_IDS:

    obj = inventory_by_id[
        iid
    ]

    name = obj.get(
        "name",
        ""
    )

    phrase = obj.get(
        "grounding_phrase",
        ""
    )


    print()
    print("=" * 100)

    print(
        f"{iid:03d}. {name}"
    )

    print("=" * 100)


    # ========================================================
    # COARSE 4x4
    # ========================================================

    coarse_image, coarse_boxes = (
        draw_grid(
            master,
            FULL_REGION,
            COARSE_ROWS,
            COARSE_COLS
        )
    )


    coarse_path = (
        GRID_DIR
        /
        f"{iid:03d}_{safe_name(name)}_coarse_4x4.png"
    )


    coarse_image.save(
        coarse_path
    )


    coarse_choice, coarse_raw = (
        run_qwen_choice(

            coarse_path,

            COARSE_PROMPT.format(
                name=name,
                phrase=phrase
            ),

            processor,
            model,

            COARSE_ROWS
            *
            COARSE_COLS,
        )
    )


    print(
        "COARSE CHOICE:",
        coarse_choice
    )


    if coarse_choice is None:

        results.append({

            "inventory_id":
                iid,

            "inventory_name":
                name,

            "grounding_phrase":
                phrase,

            "status":
                "COARSE_UNRESOLVED",

            "coarse_raw":
                coarse_raw,
        })

        continue


    coarse_bbox = (
        coarse_boxes[
            coarse_choice - 1
        ][
            "bbox"
        ]
    )


    # ========================================================
    # EXPAND COARSE REGION
    # ========================================================

    fine_region = expand_bbox(
        coarse_bbox,
        COARSE_CONTEXT_EXPANSION,
        W,
        H
    )


    fx1, fy1, fx2, fy2 = [
        int(round(v))
        for v in fine_region
    ]


    crop = master.crop(
        (
            fx1,
            fy1,
            fx2,
            fy2
        )
    )


    crop_W, crop_H = (
        crop.size
    )


    local_region = [
        0.0,
        0.0,
        float(crop_W),
        float(crop_H),
    ]


    fine_image, fine_boxes_local = (
        draw_grid(
            crop,
            local_region,
            FINE_ROWS,
            FINE_COLS
        )
    )


    fine_path = (
        GRID_DIR
        /
        f"{iid:03d}_{safe_name(name)}_fine_3x3.png"
    )


    fine_image.save(
        fine_path
    )


    fine_choice, fine_raw = (
        run_qwen_choice(

            fine_path,

            FINE_PROMPT.format(
                name=name,
                phrase=phrase
            ),

            processor,
            model,

            FINE_ROWS
            *
            FINE_COLS,
        )
    )


    print(
        "FINE CHOICE:",
        fine_choice
    )


    if fine_choice is None:

        results.append({

            "inventory_id":
                iid,

            "inventory_name":
                name,

            "grounding_phrase":
                phrase,

            "coarse_choice":
                coarse_choice,

            "coarse_bbox":
                coarse_bbox,

            "status":
                "FINE_UNRESOLVED",

            "coarse_raw":
                coarse_raw,

            "fine_raw":
                fine_raw,
        })

        continue


    fine_local_bbox = (
        fine_boxes_local[
            fine_choice - 1
        ][
            "bbox"
        ]
    )


    lx1, ly1, lx2, ly2 = (
        fine_local_bbox
    )


    fine_global_bbox = [

        fx1 + lx1,
        fy1 + ly1,
        fx1 + lx2,
        fy1 + ly2,
    ]


    print(
        "FINE GLOBAL REGION:",
        [
            round(v, 1)
            for v in fine_global_bbox
        ]
    )


    results.append({

        "inventory_id":
            iid,

        "inventory_name":
            name,

        "grounding_phrase":
            phrase,

        "coarse_choice":
            coarse_choice,

        "coarse_bbox":
            coarse_bbox,

        "fine_choice":
            fine_choice,

        "fine_region_parent":
            fine_region,

        "fine_global_bbox":
            fine_global_bbox,

        "status":
            "GRID_REGION_LOCALIZED",

        "coarse_raw":
            coarse_raw,

        "fine_raw":
            fine_raw,

        "coarse_image":
            str(
                coarse_path
            ),

        "fine_image":
            str(
                fine_path
            ),
    })


# ============================================================
# SAVE
# ============================================================

RESULT_PATH = (
    OUT
    / "00_hierarchical_grid_regions.json"
)


RESULT_PATH.write_text(
    json.dumps(
        results,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


print()
print("=" * 110)
print("PRODUCTION STAGE 06C9A RESULT")
print("=" * 110)


for row in results:

    print()

    print(
        f'{row["inventory_id"]:03d}. '
        f'{row["inventory_name"]}'
    )

    print(
        "STATUS:",
        row[
            "status"
        ]
    )

    print(
        "COARSE:",
        row.get(
            "coarse_choice"
        )
    )

    print(
        "FINE:",
        row.get(
            "fine_choice"
        )
    )

    print(
        "FINAL REGION:",
        (
            [
                round(v, 1)
                for v in row[
                    "fine_global_bbox"
                ]
            ]
            if row.get(
                "fine_global_bbox"
            )
            else None
        )
    )


print()
print(
    "RESULT JSON:",
    RESULT_PATH
)

print(
    "GRID IMAGES:",
    GRID_DIR
)

print()
print(
    "NO DINO OR SAM2 HAS BEEN RUN."
)
