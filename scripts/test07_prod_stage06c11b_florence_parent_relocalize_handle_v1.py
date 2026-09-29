
from pathlib import Path
import json

import torch
from PIL import Image, ImageDraw

from transformers import (
    AutoProcessor,
    AutoModelForMultimodalLM,
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
    / "06c11b_florence_parent_relocalize_handle"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

CACHE = (
    "/workspace/data/huggingface-cache"
)

MODEL_ID = (
    "florence-community/Florence-2-large"
)


# ============================================================
# CONFIG
# ============================================================

TARGET_ID = 21

UPSCALE = 4

PARENT_EXPANSION = 0.12


# ============================================================
# HELPERS
# ============================================================

def extract_boxes(parsed):

    result = []

    if not isinstance(
        parsed,
        dict
    ):
        return result

    for task_key, value in parsed.items():

        if not isinstance(
            value,
            dict
        ):
            continue

        boxes = value.get(
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

        for i, box in enumerate(
            boxes
        ):

            if (
                not isinstance(
                    box,
                    (list, tuple)
                )
                or
                len(box) != 4
            ):
                continue

            result.append({

                "bbox":
                    [
                        float(v)
                        for v in box
                    ],

                "label":
                    (
                        str(labels[i])
                        if i < len(labels)
                        else ""
                    ),

                "score":
                    (
                        float(scores[i])
                        if i < len(scores)
                        else None
                    ),

                "task":
                    str(task_key),
            })

    return result


def run_grounding(
    image,
    phrase,
    processor,
    model,
):

    task = (
        "<CAPTION_TO_PHRASE_GROUNDING>"
    )

    text = (
        task
        +
        phrase
    )

    inputs = processor(
        text=text,
        images=image,
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

        ids = model.generate(
            **inputs,
            max_new_tokens=256,
            num_beams=3,
            do_sample=False
        )

    generated = (
        processor.batch_decode(
            ids,
            skip_special_tokens=False
        )[0]
    )

    parsed = (
        processor.post_process_generation(
            generated,
            task=task,
            image_size=image.size
        )
    )

    return (
        generated,
        parsed,
        extract_boxes(
            parsed
        )
    )


def expand_box(
    bbox,
    ratio,
    W,
    H
):

    x1, y1, x2, y2 = [
        float(v)
        for v in bbox
    ]

    bw = max(
        1.0,
        x2 - x1
    )

    bh = max(
        1.0,
        y2 - y1
    )

    return [

        max(
            0.0,
            x1 - bw * ratio
        ),

        max(
            0.0,
            y1 - bh * ratio
        ),

        min(
            float(W),
            x2 + bw * ratio
        ),

        min(
            float(H),
            y2 + bh * ratio
        ),
    ]


def find_inventory_row(
    inventory,
    iid
):

    for row in inventory:

        rid = row.get(
            "id",
            row.get(
                "inventory_id"
            )
        )

        if (
            rid is not None
            and
            int(rid) == iid
        ):
            return row

    raise KeyError(
        iid
    )


# ============================================================
# INPUT
# ============================================================

master = Image.open(
    MASTER
).convert(
    "RGB"
)

W, H = master.size


inventory_data = json.loads(
    INVENTORY_JSON.read_text(
        encoding="utf-8"
    )
)

inventory = (
    inventory_data["objects"]
    if isinstance(
        inventory_data,
        dict
    )
    and
    "objects" in inventory_data
    else inventory_data
)


target = find_inventory_row(
    inventory,
    TARGET_ID
)

target_name = str(
    target.get(
        "name",
        "drawer handle"
    )
)

target_phrase = str(
    target.get(
        "grounding_phrase",
        target_name
    )
)


# ============================================================
# LOAD FLORENCE
# ============================================================

print("=" * 110)
print("PRODUCTION STAGE 06C11B")
print("FLORENCE PARENT RELOCALIZATION → HANDLE SEARCH")
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
# PASS 1
# FIND WHOLE VANITY / CABINET
# ============================================================

parent_phrase = (
    "the complete dark bathroom vanity cabinet unit "
    "under the white wash basin, including all visible "
    "drawer and cabinet front panels"
)


print()
print(
    "PARENT PHRASE:",
    parent_phrase
)


parent_generated, parent_parsed, parent_boxes = (
    run_grounding(
        master,
        parent_phrase,
        processor,
        model
    )
)


print(
    "PARENT CANDIDATES:",
    len(
        parent_boxes
    )
)


for i, item in enumerate(
    parent_boxes,
    start=1
):

    print(
        f"  #{i}",
        [
            round(v, 1)
            for v in item[
                "bbox"
            ]
        ],
        item[
            "label"
        ]
    )


if not parent_boxes:

    raise RuntimeError(
        "Florence could not localize vanity parent."
    )


# ============================================================
# SELECT PARENT
#
# Prefer largest returned parent candidate because phrase asks
# explicitly for COMPLETE vanity unit.
# ============================================================

def area(item):

    x1, y1, x2, y2 = (
        item["bbox"]
    )

    return max(
        0.0,
        x2 - x1
    ) * max(
        0.0,
        y2 - y1
    )


parent_boxes.sort(
    key=area,
    reverse=True
)

parent = parent_boxes[
    0
]

parent_bbox = parent[
    "bbox"
]


search_bbox = expand_box(
    parent_bbox,
    PARENT_EXPANSION,
    W,
    H
)


print()
print(
    "SELECTED PARENT:",
    [
        round(v, 1)
        for v in parent_bbox
    ]
)

print(
    "EXPANDED SEARCH:",
    [
        round(v, 1)
        for v in search_bbox
    ]
)


# ============================================================
# SAVE PARENT PREVIEW
# ============================================================

parent_preview = master.copy()

draw = ImageDraw.Draw(
    parent_preview
)

draw.rectangle(
    parent_bbox,
    outline="blue",
    width=3
)

draw.rectangle(
    search_bbox,
    outline="orange",
    width=3
)


PARENT_PREVIEW_PATH = (
    OUT
    / "01_florence_parent_preview.png"
)

parent_preview.save(
    PARENT_PREVIEW_PATH
)


# ============================================================
# CROP FLORENCE PARENT
# ============================================================

sx1, sy1, sx2, sy2 = [
    int(
        round(v)
    )
    for v in search_bbox
]


crop = master.crop(
    (
        sx1,
        sy1,
        sx2,
        sy2
    )
)


CROP_PATH = (
    OUT
    / "02_parent_crop.png"
)

crop.save(
    CROP_PATH
)


upscaled = crop.resize(
    (
        crop.width
        *
        UPSCALE,

        crop.height
        *
        UPSCALE
    ),
    Image.Resampling.LANCZOS
)


UPSCALED_PATH = (
    OUT
    / "03_parent_crop_4x.png"
)

upscaled.save(
    UPSCALED_PATH
)


# ============================================================
# PASS 2
# HANDLE INSIDE CORRECTED PARENT
# ============================================================

handle_phrases = [

    (
        "small round or short metallic handle or knob "
        "attached directly to the front face of the dark "
        "vanity cabinet"
    ),

    (
        "visible drawer or cabinet pull on the front "
        "of the dark bathroom vanity"
    ),

    (
        "small circular cabinet knob on the dark vanity"
    ),
]


all_candidates = []


for qidx, phrase in enumerate(
    handle_phrases,
    start=1
):

    print()
    print(
        f"HANDLE QUERY #{qidx}:",
        phrase
    )


    generated, parsed, boxes = (
        run_grounding(
            upscaled,
            phrase,
            processor,
            model
        )
    )


    print(
        "LOCAL CANDIDATES:",
        len(
            boxes
        )
    )


    for item in boxes:

        ux1, uy1, ux2, uy2 = (
            item[
                "bbox"
            ]
        )


        global_bbox = [

            sx1
            +
            ux1
            /
            UPSCALE,

            sy1
            +
            uy1
            /
            UPSCALE,

            sx1
            +
            ux2
            /
            UPSCALE,

            sy1
            +
            uy2
            /
            UPSCALE,
        ]


        record = {

            "query_index":
                qidx,

            "phrase":
                phrase,

            "global_bbox":
                global_bbox,

            "local_bbox_4x":
                item[
                    "bbox"
                ],

            "label":
                item[
                    "label"
                ],

            "score":
                item[
                    "score"
                ],

            "generated":
                generated,
        }


        all_candidates.append(
            record
        )


        print(
            "   global:",
            [
                round(v, 1)
                for v in global_bbox
            ],
            "| label:",
            item[
                "label"
            ]
        )


# ============================================================
# MASTER PREVIEW — ALL NEW CANDIDATES
# ============================================================

preview = master.copy()

draw = ImageDraw.Draw(
    preview
)


draw.rectangle(
    parent_bbox,
    outline="blue",
    width=2
)

draw.rectangle(
    search_bbox,
    outline="orange",
    width=2
)


for i, item in enumerate(
    all_candidates,
    start=1
):

    x1, y1, x2, y2 = (
        item[
            "global_bbox"
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

    draw.rectangle(
        [
            x1,
            y1,
            x1 + 28,
            y1 + 20
        ],
        fill="white",
        outline="red"
    )

    draw.text(
        (
            x1 + 5,
            y1 + 2
        ),
        str(i),
        fill="red"
    )


MASTER_PREVIEW_PATH = (
    OUT
    / "04_handle_candidates_master_preview.png"
)

preview.save(
    MASTER_PREVIEW_PATH
)


# ============================================================
# 4X PREVIEW
# ============================================================

local_preview = upscaled.copy()

local_draw = ImageDraw.Draw(
    local_preview
)


for i, item in enumerate(
    all_candidates,
    start=1
):

    x1, y1, x2, y2 = (
        item[
            "local_bbox_4x"
        ]
    )

    local_draw.rectangle(
        [
            x1,
            y1,
            x2,
            y2
        ],
        outline="red",
        width=6
    )

    local_draw.text(
        (
            x1 + 8,
            y1 + 8
        ),
        str(i),
        fill="red"
    )


LOCAL_PREVIEW_PATH = (
    OUT
    / "05_handle_candidates_4x_preview.png"
)

local_preview.save(
    LOCAL_PREVIEW_PATH
)


# ============================================================
# SAVE
# ============================================================

result = {

    "stage":
        "06C11B",

    "target_inventory_id":
        TARGET_ID,

    "target_inventory_name":
        target_name,

    "old_parent_not_used":
        True,

    "florence_parent_phrase":
        parent_phrase,

    "florence_parent_candidates":
        parent_boxes,

    "selected_parent_bbox":
        parent_bbox,

    "expanded_parent_search_bbox":
        search_bbox,

    "upscale":
        UPSCALE,

    "handle_queries":
        handle_phrases,

    "handle_candidates":
        all_candidates,

    "status":
        (
            "HANDLE_CANDIDATES_FOUND"
            if all_candidates
            else
            "NO_HANDLE_CANDIDATES"
        ),

    "rules": [
        "old 023 consensus bbox not used as parent",
        "parent independently relocalized by Florence",
        "search uses original Stage01 RGB",
        "parent crop upscaled 4x",
        "multiple generic handle descriptions tested",
        "no SAM2 performed",
        "no mask union performed"
    ]
}


RESULT_PATH = (
    OUT
    / "00_stage06c11b_result.json"
)


RESULT_PATH.write_text(
    json.dumps(
        result,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


print()
print("=" * 110)
print("PRODUCTION STAGE 06C11B RESULT")
print("=" * 110)

print(
    "FLORENCE PARENT:",
    [
        round(v, 1)
        for v in parent_bbox
    ]
)

print(
    "HANDLE CANDIDATES:",
    len(
        all_candidates
    )
)


for i, item in enumerate(
    all_candidates,
    start=1
):

    print(
        "  #{} query={} bbox={} label={}".format(

            i,

            item[
                "query_index"
            ],

            [
                round(v, 1)
                for v in item[
                    "global_bbox"
                ]
            ],

            item[
                "label"
            ]
        )
    )


print()
print(
    "PARENT PREVIEW:",
    PARENT_PREVIEW_PATH
)

print(
    "MASTER CANDIDATES:",
    MASTER_PREVIEW_PATH
)

print(
    "4X CANDIDATES:",
    LOCAL_PREVIEW_PATH
)

print(
    "RESULT JSON:",
    RESULT_PATH
)

print()
print(
    "NO SAM2 OR FINAL MASK UNION HAS BEEN RUN."
)
