
from pathlib import Path
import json

import torch
from PIL import Image, ImageDraw

from transformers import (
    AutoProcessor,
    AutoModelForMultimodalLM,
)


# ============================================================
# CONFIG
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
    / "06c10a_florence_phrase_grounding"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

PREVIEW_DIR = (
    OUT
    / "previews"
)

PREVIEW_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CACHE = (
    "/workspace/data/huggingface-cache"
)

MODEL_ID = (
    "florence-community/Florence-2-large"
)

TARGET_IDS = [
    10,
    21,
    25,
]


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


def get_objects(data):

    if isinstance(data, dict):

        if "objects" in data:
            return data["objects"]

        return list(
            data.values()
        )

    return data


def find_inventory_row(
    inventory,
    iid
):

    for row in inventory:

        candidate_id = row.get(
            "id",
            row.get(
                "inventory_id"
            )
        )

        if candidate_id is None:
            continue

        if int(candidate_id) == iid:
            return row

    raise KeyError(
        f"Inventory ID {iid} not found"
    )


def extract_boxes_from_postprocessed(
    parsed
):

    boxes = []

    if not isinstance(
        parsed,
        dict
    ):
        return boxes

    # Florence referring-expression grounding normally
    # returns a task-keyed dictionary.
    for key, value in parsed.items():

        if not isinstance(
            value,
            dict
        ):
            continue

        bboxes = value.get(
            "bboxes",
            []
        )

        labels = value.get(
            "labels",
            []
        )

        scores = value.get(
            "scores",
            []
        )

        for idx, box in enumerate(
            bboxes
        ):

            if not isinstance(
                box,
                (list, tuple)
            ):
                continue

            if len(box) != 4:
                continue

            boxes.append({

                "bbox":
                    [
                        float(v)
                        for v in box
                    ],

                "label":
                    (
                        str(labels[idx])
                        if idx < len(labels)
                        else ""
                    ),

                "score":
                    (
                        float(scores[idx])
                        if idx < len(scores)
                        else None
                    ),

                "task_key":
                    str(key),
            })

    return boxes


# ============================================================
# VALIDATE
# ============================================================

for p in [
    MASTER,
    INVENTORY_JSON,
]:

    if not p.exists():

        raise FileNotFoundError(
            p
        )


# ============================================================
# LOAD MASTER
# ============================================================

master = Image.open(
    MASTER
).convert(
    "RGB"
)

W, H = master.size


# ============================================================
# LOAD INVENTORY
# ============================================================

inventory_data = json.loads(
    INVENTORY_JSON.read_text(
        encoding="utf-8"
    )
)

inventory = get_objects(
    inventory_data
)


# ============================================================
# LOAD FLORENCE
# ============================================================

print("=" * 110)
print("PRODUCTION STAGE 06C10A")
print("FLORENCE-2 INDEPENDENT PHRASE GROUNDING")
print("=" * 110)

print()
print(
    "Loading Florence-2..."
)


processor = (
    AutoProcessor
    .from_pretrained(
        MODEL_ID,
        cache_dir=CACHE
    )
)


model = (
    AutoModelForMultimodalLM
    .from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        device_map="auto",
        cache_dir=CACHE
    )
)


model.eval()


print(
    "✅ FLORENCE READY"
)


# ============================================================
# RUN
# ============================================================

results = []


for iid in TARGET_IDS:

    row = find_inventory_row(
        inventory,
        iid
    )

    name = str(
        row.get(
            "name",
            row.get(
                "inventory_name",
                ""
            )
        )
    )

    phrase = str(
        row.get(
            "grounding_phrase",
            ""
        )
    ).strip()


    if not phrase:

        phrase = name


    # --------------------------------------------------------
    # Florence referring-expression grounding
    # --------------------------------------------------------

    task_prompt = (
        "<REFERRING_EXPRESSION_SEGMENTATION>"
    )

    text_input = (
        task_prompt
        +
        phrase
    )


    print()
    print("=" * 100)
    print(
        f"{iid:03d}. {name}"
    )
    print("=" * 100)

    print(
        "PHRASE:",
        phrase
    )


    inputs = processor(
        text=text_input,
        images=master,
        return_tensors="pt"
    )


    inputs = {
        k:
            v.to(
                model.device
            )
            if hasattr(
                v,
                "to"
            )
            else v

        for k, v
        in inputs.items()
    }


    with torch.inference_mode():

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            num_beams=3,
            do_sample=False
        )


    generated_text = (
        processor.batch_decode(
            generated_ids,
            skip_special_tokens=False
        )[0]
    )


    parsed = (
        processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(
                W,
                H
            )
        )
    )


    boxes = (
        extract_boxes_from_postprocessed(
            parsed
        )
    )


    # --------------------------------------------------------
    # FALLBACK:
    # if referring segmentation gives no boxes,
    # retry phrase grounding task
    # --------------------------------------------------------

    fallback_used = False
    fallback_prompt = None
    fallback_generated_text = None
    fallback_parsed = None


    if not boxes:

        fallback_used = True

        fallback_prompt = (
            "<CAPTION_TO_PHRASE_GROUNDING>"
        )

        fallback_text = (
            fallback_prompt
            +
            phrase
        )


        inputs = processor(
            text=fallback_text,
            images=master,
            return_tensors="pt"
        )


        inputs = {
            k:
                v.to(
                    model.device
                )
                if hasattr(
                    v,
                    "to"
                )
                else v

            for k, v
            in inputs.items()
        }


        with torch.inference_mode():

            fallback_ids = (
                model.generate(
                    **inputs,
                    max_new_tokens=256,
                    num_beams=3,
                    do_sample=False
                )
            )


        fallback_generated_text = (
            processor.batch_decode(
                fallback_ids,
                skip_special_tokens=False
            )[0]
        )


        fallback_parsed = (
            processor.post_process_generation(
                fallback_generated_text,
                task=fallback_prompt,
                image_size=(
                    W,
                    H
                )
            )
        )


        boxes = (
            extract_boxes_from_postprocessed(
                fallback_parsed
            )
        )


    print(
        "BOX COUNT:",
        len(boxes)
    )


    for idx, item in enumerate(
        boxes,
        start=1
    ):

        print(
            f"  #{idx}",
            "| bbox:",
            [
                round(v, 1)
                for v in item[
                    "bbox"
                ]
            ],
            "| label:",
            item[
                "label"
            ],
            "| score:",
            item[
                "score"
            ]
        )


    # --------------------------------------------------------
    # PREVIEW
    # --------------------------------------------------------

    preview = master.copy()

    draw = ImageDraw.Draw(
        preview
    )


    for idx, item in enumerate(
        boxes,
        start=1
    ):

        x1, y1, x2, y2 = (
            item[
                "bbox"
            ]
        )


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


        label = (
            f"{idx}"
        )


        draw.rectangle(
            [
                x1,
                y1,
                x1 + 26,
                y1 + 20
            ],
            fill="white",
            outline="red"
        )


        draw.text(
            (
                x1 + 6,
                y1 + 2
            ),
            label,
            fill="red"
        )


    preview_path = (
        PREVIEW_DIR
        /
        f"{iid:03d}_{safe_name(name)}_florence_preview.png"
    )


    preview.save(
        preview_path
    )


    result = {

        "inventory_id":
            iid,

        "inventory_name":
            name,

        "grounding_phrase":
            phrase,

        "primary_task":
            task_prompt,

        "primary_generated_text":
            generated_text,

        "primary_postprocessed":
            parsed,

        "fallback_used":
            fallback_used,

        "fallback_task":
            fallback_prompt,

        "fallback_generated_text":
            fallback_generated_text,

        "fallback_postprocessed":
            fallback_parsed,

        "boxes":
            boxes,

        "preview_path":
            str(
                preview_path
            ),
    }


    results.append(
        result
    )


# ============================================================
# SAVE
# ============================================================

RESULT_PATH = (
    OUT
    / "00_florence_phrase_grounding_results.json"
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
print("PRODUCTION STAGE 06C10A RESULT")
print("=" * 110)


for row in results:

    print()

    print(
        f'{row["inventory_id"]:03d}. '
        f'{row["inventory_name"]}'
    )

    print(
        "BOXES:",
        len(
            row[
                "boxes"
            ]
        )
    )

    print(
        "FALLBACK USED:",
        row[
            "fallback_used"
        ]
    )

    for idx, item in enumerate(
        row[
            "boxes"
        ],
        start=1
    ):

        print(
            "  #{} bbox={} label={} score={}".format(

                idx,

                [
                    round(
                        v,
                        1
                    )
                    for v in item[
                        "bbox"
                    ]
                ],

                item[
                    "label"
                ],

                item[
                    "score"
                ],
            )
        )


print()
print(
    "RESULT JSON:",
    RESULT_PATH
)

print(
    "PREVIEWS:",
    PREVIEW_DIR
)

print()
print(
    "NO SAM2 OR MASK UNION HAS BEEN RUN."
)
