
from pathlib import Path
import json

import torch
from PIL import Image, ImageDraw

import matplotlib.pyplot as plt

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

HANDLE_RESULT = (
    STAGE06
    / "06c11c_florence_handle_sam2"
    / "00_stage06c11c_result.json"
)

OUT = (
    STAGE06
    / "06c15a_vanity_hierarchy_discovery"
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

UPSCALE = 4

PARENT_EXPANSION = 0.08


# ============================================================
# QUERY GROUPS
#
# Semantic descriptions only.
# No coordinates.
# ============================================================

PARENT_QUERIES = [

    (
        "complete dark bathroom vanity cabinet under "
        "the white wash basin including all visible "
        "cabinet and drawer front panels"
    ),

    (
        "entire dark floating bathroom vanity unit "
        "supporting the white sink"
    ),

    (
        "complete dark vanity furniture below the basin"
    ),
]


COMPONENT_QUERIES = {

    "CABINET_BODY": [

        (
            "dark cabinet body and cabinet doors "
            "of the bathroom vanity"
        ),

        (
            "large dark cabinet front panels "
            "below the wash basin"
        ),

        (
            "dark vanity cabinet doors"
        ),
    ],


    "DRAWER": [

        (
            "individual drawer front panel "
            "on the dark bathroom vanity"
        ),

        (
            "dark horizontal drawer below the sink"
        ),

        (
            "visible drawer front of the vanity cabinet"
        ),
    ],


    "HANDLE": [

        (
            "small circular knob or handle "
            "attached to the vanity front"
        ),

        (
            "drawer or cabinet handle on the dark vanity"
        ),

        (
            "small round cabinet knob on the vanity"
        ),
    ],
}


# ============================================================
# HELPERS
# ============================================================

def extract_boxes(parsed):

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


        for idx, box in enumerate(
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

                "task":
                    str(task_key),
            })


    return results


def run_grounding(
    image,
    phrase,
    processor,
    model
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
            if hasattr(v, "to")
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
        extract_boxes(parsed)
    )


def clamp_box(
    box,
    W,
    H
):

    x1, y1, x2, y2 = [
        float(v)
        for v in box
    ]

    x1 = max(
        0.0,
        min(float(W - 1), x1)
    )

    y1 = max(
        0.0,
        min(float(H - 1), y1)
    )

    x2 = max(
        x1 + 1.0,
        min(float(W), x2)
    )

    y2 = max(
        y1 + 1.0,
        min(float(H), y2)
    )

    return [
        x1,
        y1,
        x2,
        y2
    ]


def expand_box(
    box,
    ratio,
    W,
    H
):

    x1, y1, x2, y2 = box

    bw = max(
        1.0,
        x2 - x1
    )

    bh = max(
        1.0,
        y2 - y1
    )

    return clamp_box(
        [
            x1 - bw * ratio,
            y1 - bh * ratio,
            x2 + bw * ratio,
            y2 + bh * ratio,
        ],
        W,
        H
    )


def bbox_area(box):

    return max(
        0.0,
        box[2] - box[0]
    ) * max(
        0.0,
        box[3] - box[1]
    )


def bbox_iou(
    a,
    b
):

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(
        0.0,
        ix2 - ix1
    )

    ih = max(
        0.0,
        iy2 - iy1
    )

    inter = iw * ih

    union = (
        bbox_area(a)
        +
        bbox_area(b)
        -
        inter
    )

    return float(
        inter
        /
        max(
            1.0,
            union
        )
    )


def containment_fraction(
    child,
    parent
):

    cx1, cy1, cx2, cy2 = child
    px1, py1, px2, py2 = parent

    ix1 = max(cx1, px1)
    iy1 = max(cy1, py1)
    ix2 = min(cx2, px2)
    iy2 = min(cy2, py2)

    iw = max(
        0.0,
        ix2 - ix1
    )

    ih = max(
        0.0,
        iy2 - iy1
    )

    intersection = iw * ih

    return float(
        intersection
        /
        max(
            1.0,
            bbox_area(child)
        )
    )


def deduplicate(
    rows,
    threshold=0.70
):

    unique = []


    for item in rows:

        duplicate = None

        for idx, kept in enumerate(
            unique
        ):

            if bbox_iou(
                item["bbox"],
                kept["bbox"]
            ) >= threshold:

                duplicate = idx
                break


        if duplicate is None:

            new = dict(item)

            new[
                "support_queries"
            ] = [
                item[
                    "query_index"
                ]
            ]

            unique.append(new)


        else:

            q = item[
                "query_index"
            ]

            if (
                q
                not in
                unique[
                    duplicate
                ][
                    "support_queries"
                ]
            ):

                unique[
                    duplicate
                ][
                    "support_queries"
                ].append(q)


    return unique


def show_image(
    path,
    title,
    figsize=(8, 8)
):

    image = Image.open(path)

    plt.figure(
        figsize=figsize
    )

    plt.imshow(image)

    plt.title(title)

    plt.axis("off")

    plt.show()


# ============================================================
# INPUT
# ============================================================

if not MASTER.exists():

    raise FileNotFoundError(
        MASTER
    )


master = Image.open(
    MASTER
).convert("RGB")

W, H = master.size


# ============================================================
# LOAD FLORENCE
# ============================================================

print("=" * 110)
print("PRODUCTION STAGE 06C15A")
print("VANITY PHYSICAL-HIERARCHY DISCOVERY")
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
# FULL-SCENE WHOLE-VANITY LOCALIZATION
# ============================================================

parent_raw = []


for qidx, phrase in enumerate(
    PARENT_QUERIES,
    start=1
):

    print()
    print(
        f"PARENT QUERY #{qidx}:",
        phrase
    )


    generated, parsed, boxes = (
        run_grounding(
            master,
            phrase,
            processor,
            model
        )
    )


    print(
        "  returned:",
        len(boxes)
    )


    for item in boxes:

        bbox = clamp_box(
            item[
                "bbox"
            ],
            W,
            H
        )


        parent_raw.append({

            "query_index":
                qidx,

            "phrase":
                phrase,

            "bbox":
                bbox,

            "label":
                item[
                    "label"
                ],

            "score":
                item[
                    "score"
                ],
        })


        print(
            "   ",
            [
                round(v, 1)
                for v in bbox
            ],
            "|",
            item[
                "label"
            ]
        )


parent_unique = deduplicate(
    parent_raw
)


if not parent_unique:

    raise RuntimeError(
        "No Florence vanity parent candidate."
    )


# ============================================================
# SELECT WHOLE-VANITY PARENT
#
# We asked explicitly for complete vanity, therefore:
#
# 1. higher independent query support
# 2. then larger area
#
# This is only crop selection, not final prop segmentation.
# ============================================================

parent_unique.sort(
    key=lambda row:
    (
        len(
            row[
                "support_queries"
            ]
        ),
        bbox_area(
            row[
                "bbox"
            ]
        ),
    ),
    reverse=True
)


selected_parent = parent_unique[
    0
]


parent_bbox = selected_parent[
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
    "SELECTED WHOLE-VANITY BBOX:",
    [
        round(v, 1)
        for v in parent_bbox
    ]
)

print(
    "QUERY SUPPORT:",
    selected_parent[
        "support_queries"
    ]
)

print(
    "EXPANDED SEARCH BBOX:",
    [
        round(v, 1)
        for v in search_bbox
    ]
)


# ============================================================
# PARENT PREVIEW
# ============================================================

parent_preview = master.copy()

draw = ImageDraw.Draw(
    parent_preview
)


for idx, item in enumerate(
    parent_unique,
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
        width=2
    )

    draw.text(
        (
            x1 + 3,
            y1 + 3
        ),
        f"P{idx}",
        fill="red"
    )


draw.rectangle(
    parent_bbox,
    outline="lime",
    width=3
)


PARENT_PREVIEW_PATH = (
    OUT
    / "01_whole_vanity_parent_candidates.png"
)


parent_preview.save(
    PARENT_PREVIEW_PATH
)


# ============================================================
# CROP ORIGINAL STAGE01
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
    / "02_vanity_crop_original.png"
)


crop.save(
    CROP_PATH
)


upscaled = crop.resize(
    (
        crop.width * UPSCALE,
        crop.height * UPSCALE
    ),
    Image.Resampling.LANCZOS
)


UPSCALED_PATH = (
    OUT
    / "03_vanity_crop_4x.png"
)


upscaled.save(
    UPSCALED_PATH
)


# ============================================================
# COMPONENT SEARCH
# ============================================================

component_results = {}


for component_type, phrases in COMPONENT_QUERIES.items():

    print()
    print("=" * 100)
    print(
        component_type
    )
    print("=" * 100)


    raw = []


    for qidx, phrase in enumerate(
        phrases,
        start=1
    ):

        print()
        print(
            f"QUERY #{qidx}:",
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
            "  returned:",
            len(boxes)
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
                ux1 / UPSCALE,

                sy1
                +
                uy1 / UPSCALE,

                sx1
                +
                ux2 / UPSCALE,

                sy1
                +
                uy2 / UPSCALE,
            ]


            global_bbox = clamp_box(
                global_bbox,
                W,
                H
            )


            record = {

                "component_type":
                    component_type,

                "query_index":
                    qidx,

                "phrase":
                    phrase,

                "bbox":
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
            }


            raw.append(
                record
            )


            print(
                "   global:",
                [
                    round(v, 1)
                    for v in global_bbox
                ],
                "|",
                item[
                    "label"
                ]
            )


    unique = deduplicate(
        raw
    )


    for idx, item in enumerate(
        unique,
        start=1
    ):

        item[
            "candidate_index"
        ] = idx

        item[
            "contained_in_parent"
        ] = containment_fraction(
            item[
                "bbox"
            ],
            search_bbox
        )


    component_results[
        component_type
    ] = unique


    print()
    print(
        "UNIQUE:",
        len(unique)
    )


# ============================================================
# MASTER COMPONENT PREVIEW
# ============================================================

master_preview = master.copy()

draw = ImageDraw.Draw(
    master_preview
)


draw.rectangle(
    search_bbox,
    outline="orange",
    width=2
)


PREFIXES = {

    "CABINET_BODY":
        "C",

    "DRAWER":
        "D",

    "HANDLE":
        "H",
}


for component_type, rows in component_results.items():

    marker = PREFIXES[
        component_type
    ]


    for item in rows:

        x1, y1, x2, y2 = (
            item[
                "bbox"
            ]
        )


        label = (
            f'{marker}'
            f'{item["candidate_index"]}'
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
                x1 + 32,
                y1 + 22
            ],
            fill="white",
            outline="red"
        )


        draw.text(
            (
                x1 + 4,
                y1 + 3
            ),
            label,
            fill="red"
        )


MASTER_COMPONENT_PREVIEW = (
    OUT
    / "04_vanity_component_candidates_master.png"
)


master_preview.save(
    MASTER_COMPONENT_PREVIEW
)


# ============================================================
# 4X COMPONENT PREVIEW
# ============================================================

local_preview = upscaled.copy()

draw = ImageDraw.Draw(
    local_preview
)


for component_type, rows in component_results.items():

    marker = PREFIXES[
        component_type
    ]


    for item in rows:

        x1, y1, x2, y2 = (
            item[
                "local_bbox_4x"
            ]
        )


        label = (
            f'{marker}'
            f'{item["candidate_index"]}'
        )


        draw.rectangle(
            [
                x1,
                y1,
                x2,
                y2
            ],
            outline="red",
            width=6
        )


        draw.text(
            (
                x1 + 8,
                y1 + 8
            ),
            label,
            fill="red"
        )


LOCAL_COMPONENT_PREVIEW = (
    OUT
    / "05_vanity_component_candidates_4x.png"
)


local_preview.save(
    LOCAL_COMPONENT_PREVIEW
)


# ============================================================
# HANDLE RELATION TO ALREADY-RESOLVED 021
# ============================================================

resolved_handle_bbox = None


if HANDLE_RESULT.exists():

    handle_state = json.loads(
        HANDLE_RESULT.read_text(
            encoding="utf-8"
        )
    )

    resolved_handle_bbox = (
        handle_state.get(
            "florence_bbox"
        )
    )


# ============================================================
# RELATION TABLE
# ============================================================

relations = []


for component_type, rows in component_results.items():

    for item in rows:

        relation = {

            "component_type":
                component_type,

            "candidate_index":
                item[
                    "candidate_index"
                ],

            "bbox":
                item[
                    "bbox"
                ],

            "parent_containment":
                containment_fraction(
                    item[
                        "bbox"
                    ],
                    search_bbox
                ),

            "iou_with_resolved_021_handle":
                (
                    bbox_iou(
                        item[
                            "bbox"
                        ],
                        resolved_handle_bbox
                    )

                    if resolved_handle_bbox
                    else None
                ),
        }


        relations.append(
            relation
        )


# ============================================================
# SAVE
# ============================================================

result = {

    "stage":
        "06C15A",

    "purpose":
        "vanity-system physical hierarchy discovery",

    "inventory_under_review": {
        "5":
            "cabinet",
        "23":
            "cabinet",
        "29":
            "drawer",
        "21":
            "drawer handle / resolved child component",
    },

    "parent_queries":
        PARENT_QUERIES,

    "parent_candidates":
        parent_unique,

    "selected_parent":
        selected_parent,

    "selected_parent_bbox":
        parent_bbox,

    "expanded_search_bbox":
        search_bbox,

    "upscale":
        UPSCALE,

    "component_queries":
        COMPONENT_QUERIES,

    "components":
        component_results,

    "resolved_021_handle_bbox":
        resolved_handle_bbox,

    "relations":
        relations,

    "status":
        "HIERARCHY_CANDIDATES_READY_FOR_VISUAL_AUDIT",

    "rules": [
        "old DINO cabinet masks are not trusted",
        "whole vanity localization is used only to define search context",
        "no component is accepted automatically",
        "no SAM2 was run",
        "no prop-mask union was created",
        "parent and child components must not be blindly unioned",
    ],
}


RESULT_PATH = (
    OUT
    / "00_stage06c15a_result.json"
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
# PRINT RESULT
# ============================================================

print()
print("=" * 110)
print("PRODUCTION STAGE 06C15A RESULT")
print("=" * 110)


print(
    "WHOLE VANITY:",
    [
        round(v, 1)
        for v in parent_bbox
    ]
)


for component_type in [
    "CABINET_BODY",
    "DRAWER",
    "HANDLE",
]:

    rows = component_results[
        component_type
    ]


    print()
    print(
        component_type,
        "CANDIDATES:",
        len(rows)
    )


    for item in rows:

        print(
            "  #{} bbox={} label={} support={} parent_contain={}".format(

                item[
                    "candidate_index"
                ],

                [
                    round(v, 1)
                    for v in item[
                        "bbox"
                    ]
                ],

                item[
                    "label"
                ],

                item[
                    "support_queries"
                ],

                round(
                    item[
                        "contained_in_parent"
                    ],
                    4
                ),
            )
        )


print()
print(
    "RESOLVED 021 HANDLE BBOX:",
    (
        [
            round(v, 1)
            for v in resolved_handle_bbox
        ]
        if resolved_handle_bbox
        else None
    )
)


print()
print(
    "RESULT JSON:",
    RESULT_PATH
)

print()
print(
    "NO SAM2 OR FINAL PROP MASK UNION WAS CREATED."
)


# ============================================================
# INLINE VISUAL VERIFICATION
# ============================================================

print()
print("=" * 110)
print("INLINE VISUAL VERIFICATION")
print("=" * 110)


show_image(
    PARENT_PREVIEW_PATH,
    (
        "06C15A — WHOLE VANITY PARENT CANDIDATES "
        "(GREEN = SELECTED)"
    ),
    figsize=(8, 9)
)


show_image(
    MASTER_COMPONENT_PREVIEW,
    (
        "06C15A — VANITY COMPONENT CANDIDATES "
        "C=Cabinet, D=Drawer, H=Handle"
    ),
    figsize=(8, 9)
)


show_image(
    LOCAL_COMPONENT_PREVIEW,
    (
        "06C15A — 4X VANITY COMPONENT AUDIT "
        "C=Cabinet, D=Drawer, H=Handle"
    ),
    figsize=(10, 8)
)


# ============================================================
# CLEANUP
# ============================================================

del model

if torch.cuda.is_available():

    torch.cuda.empty_cache()
