
from pathlib import Path
import json

import numpy as np
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

CONSENSUS_JSON = (
    STAGE06
    / "06c6_multi_route_candidate_consensus"
    / "00_multi_route_consensus_all.json"
)

INVENTORY_JSON = (
    STAGE06
    / "06c2_lost_instance_existence_audit"
    / "00a_physical_inventory_bridge.json"
)

OUT = (
    STAGE06
    / "06c11a_parent_guided_florence_handle"
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
PARENT_ID = 23

PARENT_EXPANSION = 0.18
UPSCALE = 4


# ============================================================
# HELPERS
# ============================================================

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

        if rid is not None and int(rid) == iid:
            return row

    raise KeyError(
        f"inventory id {iid} not found"
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

    bw = x2 - x1
    bh = y2 - y1

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


def extract_boxes(
    parsed
):

    results = []

    if not isinstance(
        parsed,
        dict
    ):
        return results

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

            results.append({

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
                    task_key,
            })

    return results


# ============================================================
# LOAD
# ============================================================

master = Image.open(
    MASTER
).convert(
    "RGB"
)

W, H = master.size


consensus = json.loads(
    CONSENSUS_JSON.read_text(
        encoding="utf-8"
    )
)


inventory_data = json.loads(
    INVENTORY_JSON.read_text(
        encoding="utf-8"
    )
)

inventory = (
    inventory_data["objects"]
    if isinstance(inventory_data, dict)
    and "objects" in inventory_data
    else inventory_data
)


target = find_inventory_row(
    inventory,
    TARGET_ID
)

target_name = str(
    target.get(
        "name",
        target.get(
            "inventory_name",
            "drawer handle"
        )
    )
)

target_phrase = str(
    target.get(
        "grounding_phrase",
        ""
    )
).strip()

if not target_phrase:

    target_phrase = target_name


# ============================================================
# FIND PARENT 023
# ============================================================

parent = next(

    row

    for row in consensus

    if int(
        row["inventory_id"]
    ) == PARENT_ID
)


clusters = parent.get(
    "clusters",
    []
)

if not clusters:

    raise RuntimeError(
        "023 cabinet has no consensus clusters."
    )


parent_bbox = [
    float(v)
    for v in clusters[0]["bbox"]
]


search_bbox = expand_box(
    parent_bbox,
    PARENT_EXPANSION,
    W,
    H
)


print("=" * 110)
print("PRODUCTION STAGE 06C11A")
print("PARENT-GUIDED FLORENCE HANDLE LOCALIZATION")
print("=" * 110)

print()
print(
    "TARGET:",
    TARGET_ID,
    target_name
)

print(
    "PARENT:",
    PARENT_ID,
    parent["inventory_name"]
)

print(
    "PARENT BBOX:",
    [
        round(v, 1)
        for v in parent_bbox
    ]
)

print(
    "SEARCH BBOX:",
    [
        round(v, 1)
        for v in search_bbox
    ]
)


# ============================================================
# CROP ORIGINAL MASTER
# ============================================================

sx1, sy1, sx2, sy2 = [
    int(round(v))
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
    / "00_parent_search_crop_original.png"
)

crop.save(
    CROP_PATH
)


# ============================================================
# UPSCALE 4X
# ============================================================

upscaled = crop.resize(
    (
        crop.width * UPSCALE,
        crop.height * UPSCALE
    ),
    Image.Resampling.LANCZOS
)


UPSCALED_PATH = (
    OUT
    / "01_parent_search_crop_4x.png"
)

upscaled.save(
    UPSCALED_PATH
)


# ============================================================
# LOAD FLORENCE
# ============================================================

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
# PHRASE GROUNDING
# ============================================================

task = (
    "<CAPTION_TO_PHRASE_GROUNDING>"
)


# Make target semantics explicit while remaining generic enough
# for arbitrary furniture/room types.
phrase = (
    "small physical "
    +
    target_phrase
    +
    " attached directly to the front face "
    "of the cabinet or drawer"
)


text_input = (
    task
    +
    phrase
)


print()
print(
    "GROUNDING PHRASE:",
    phrase
)


inputs = processor(
    text=text_input,
    images=upscaled,
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
        task=task,
        image_size=(
            upscaled.width,
            upscaled.height
        )
    )
)


local_boxes = extract_boxes(
    parsed
)


print()
print(
    "LOCAL BOXES:",
    len(local_boxes)
)


# ============================================================
# MAP BACK TO MASTER
# ============================================================

mapped = []


for item in local_boxes:

    ux1, uy1, ux2, uy2 = (
        item["bbox"]
    )


    # Upscaled coordinates → crop coordinates
    cx1 = ux1 / UPSCALE
    cy1 = uy1 / UPSCALE
    cx2 = ux2 / UPSCALE
    cy2 = uy2 / UPSCALE


    global_bbox = [

        sx1 + cx1,
        sy1 + cy1,
        sx1 + cx2,
        sy1 + cy2,
    ]


    mapped.append({

        "bbox":
            global_bbox,

        "local_upscaled_bbox":
            item["bbox"],

        "label":
            item["label"],

        "score":
            item["score"],
    })


for i, item in enumerate(
    mapped,
    start=1
):

    print(
        f"#{i}",
        "| bbox:",
        [
            round(v, 1)
            for v in item["bbox"]
        ],
        "| label:",
        item["label"],
        "| score:",
        item["score"]
    )


# ============================================================
# MASTER PREVIEW
# ============================================================

preview = master.copy()

draw = ImageDraw.Draw(
    preview
)


# Parent region
draw.rectangle(
    parent_bbox,
    outline="blue",
    width=2
)


# Expanded search region
draw.rectangle(
    search_bbox,
    outline="orange",
    width=2
)


# Florence candidates
for i, item in enumerate(
    mapped,
    start=1
):

    x1, y1, x2, y2 = (
        item["bbox"]
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
            x1 + 24,
            y1 + 18
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


PREVIEW_PATH = (
    OUT
    / "02_parent_guided_florence_preview.png"
)

preview.save(
    PREVIEW_PATH
)


# ============================================================
# UPSCALED CROP PREVIEW
# ============================================================

local_preview = upscaled.copy()

local_draw = ImageDraw.Draw(
    local_preview
)


for i, item in enumerate(
    local_boxes,
    start=1
):

    x1, y1, x2, y2 = (
        item["bbox"]
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
    / "03_parent_guided_florence_4x_preview.png"
)

local_preview.save(
    LOCAL_PREVIEW_PATH
)


# ============================================================
# SAVE RESULT
# ============================================================

result = {

    "stage":
        "06C11A",

    "target_inventory_id":
        TARGET_ID,

    "target_inventory_name":
        target_name,

    "parent_inventory_id":
        PARENT_ID,

    "parent_inventory_name":
        parent[
            "inventory_name"
        ],

    "parent_bbox":
        parent_bbox,

    "search_bbox":
        search_bbox,

    "upscale":
        UPSCALE,

    "grounding_phrase":
        phrase,

    "generated_text":
        generated_text,

    "parsed":
        parsed,

    "mapped_boxes":
        mapped,

    "status":
        (
            "CANDIDATES_FOUND"
            if mapped
            else
            "NO_CANDIDATES"
        ),

    "rules": [
        "parent region derived automatically",
        "original Stage01 RGB used",
        "Florence run only inside parent neighborhood",
        "no SAM2 performed",
        "no mask union performed"
    ]
}


RESULT_PATH = (
    OUT
    / "00_stage06c11a_result.json"
)


RESULT_PATH.write_text(
    json.dumps(
        result,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# ============================================================
# PRINT
# ============================================================

print()
print("=" * 110)
print("PRODUCTION STAGE 06C11A RESULT")
print("=" * 110)

print(
    "TARGET:",
    f"{TARGET_ID:03d}",
    target_name
)

print(
    "PARENT:",
    f"{PARENT_ID:03d}",
    parent[
        "inventory_name"
    ]
)

print(
    "CANDIDATES:",
    len(
        mapped
    )
)


for i, item in enumerate(
    mapped,
    start=1
):

    print(
        "  #{} bbox={} label={} score={}".format(

            i,

            [
                round(v, 1)
                for v in item["bbox"]
            ],

            item["label"],

            item["score"],
        )
    )


print()
print(
    "MASTER PREVIEW:",
    PREVIEW_PATH
)

print(
    "4X PREVIEW:",
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
