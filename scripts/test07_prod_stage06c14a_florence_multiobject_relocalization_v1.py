
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
    / "06c14a_florence_multiobject_relocalization"
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


# ============================================================
# TARGETS
# ============================================================

TARGET_IDS = [
    2,
    8,
    12,
    15,
]


# ============================================================
# GENERIC QUERY BANK
#
# These are semantic variants only.
# No room-specific coordinates.
# ============================================================

QUERY_BANK = {

    2: [
        (
            "small round recessed ceiling light "
            "embedded in the ceiling"
        ),

        (
            "circular ceiling downlight or recessed spotlight"
        ),

        (
            "small bright recessed light fixture "
            "mounted flush in the ceiling"
        ),
    ],

    8: [
        (
            "rectangular electrical wall outlet "
            "mounted on the wall"
        ),

        (
            "small wall-mounted electrical socket "
            "or power outlet plate"
        ),

        (
            "electrical power receptacle on the wall"
        ),
    ],

    12: [
        (
            "wall-mounted toilet paper holder fixture"
        ),

        (
            "small metal toilet roll holder "
            "attached to the wall"
        ),

        (
            "toilet paper holder beside the toilet"
        ),
    ],

    15: [
        (
            "long horizontal metal towel bar "
            "mounted on the wall"
        ),

        (
            "wall-mounted towel rail or towel bar"
        ),

        (
            "thin horizontal chrome towel rail"
        ),
    ],
}


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
        f"Inventory ID {iid} not found"
    )


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


    generated = (
        processor.batch_decode(
            generated_ids,
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
        min(
            float(W - 1),
            x1
        )
    )


    y1 = max(
        0.0,
        min(
            float(H - 1),
            y1
        )
    )


    x2 = max(
        x1 + 1.0,
        min(
            float(W),
            x2
        )
    )


    y2 = max(
        y1 + 1.0,
        min(
            float(H),
            y2
        )
    )


    return [
        x1,
        y1,
        x2,
        y2
    ]


def box_iou(
    a,
    b
):

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b


    ix1 = max(
        ax1,
        bx1
    )

    iy1 = max(
        ay1,
        by1
    )

    ix2 = min(
        ax2,
        bx2
    )

    iy2 = min(
        ay2,
        by2
    )


    iw = max(
        0.0,
        ix2 - ix1
    )

    ih = max(
        0.0,
        iy2 - iy1
    )


    intersection = (
        iw
        *
        ih
    )


    area_a = max(
        1.0,
        (
            ax2 - ax1
        )
        *
        (
            ay2 - ay1
        )
    )


    area_b = max(
        1.0,
        (
            bx2 - bx1
        )
        *
        (
            by2 - by1
        )
    )


    union = (
        area_a
        +
        area_b
        -
        intersection
    )


    return float(
        intersection
        /
        max(
            1.0,
            union
        )
    )


def box_center_distance(
    a,
    b
):

    acx = (
        a[0] + a[2]
    ) / 2

    acy = (
        a[1] + a[3]
    ) / 2

    bcx = (
        b[0] + b[2]
    ) / 2

    bcy = (
        b[1] + b[3]
    ) / 2


    dx = acx - bcx
    dy = acy - bcy


    return (
        dx * dx
        +
        dy * dy
    ) ** 0.5


def deduplicate(
    candidates
):

    unique = []


    for item in candidates:

        box = item[
            "bbox"
        ]


        duplicate_index = None


        for idx, kept in enumerate(
            unique
        ):

            kept_box = kept[
                "bbox"
            ]


            iou = box_iou(
                box,
                kept_box
            )


            center_distance = (
                box_center_distance(
                    box,
                    kept_box
                )
            )


            bw = max(
                1.0,
                box[2] - box[0]
            )

            bh = max(
                1.0,
                box[3] - box[1]
            )


            kw = max(
                1.0,
                kept_box[2]
                -
                kept_box[0]
            )

            kh = max(
                1.0,
                kept_box[3]
                -
                kept_box[1]
            )


            scale = max(
                4.0,
                min(
                    (
                        bw ** 2
                        +
                        bh ** 2
                    ) ** 0.5,

                    (
                        kw ** 2
                        +
                        kh ** 2
                    ) ** 0.5,
                )
            )


            if (
                iou >= 0.70

                or

                (
                    center_distance
                    /
                    scale
                    <= 0.12

                    and

                    abs(
                        bw - kw
                    )
                    /
                    max(
                        bw,
                        kw
                    )
                    <= 0.30

                    and

                    abs(
                        bh - kh
                    )
                    /
                    max(
                        bh,
                        kh
                    )
                    <= 0.30
                )
            ):

                duplicate_index = idx
                break


        if duplicate_index is None:

            copy = dict(
                item
            )

            copy[
                "supporting_queries"
            ] = [
                item[
                    "query_index"
                ]
            ]

            copy[
                "support_count"
            ] = 1

            unique.append(
                copy
            )


        else:

            kept = unique[
                duplicate_index
            ]


            if (
                item[
                    "query_index"
                ]
                not in
                kept[
                    "supporting_queries"
                ]
            ):

                kept[
                    "supporting_queries"
                ].append(
                    item[
                        "query_index"
                    ]
                )


            kept[
                "support_count"
            ] = len(
                kept[
                    "supporting_queries"
                ]
            )


    return unique


# ============================================================
# VALIDATE
# ============================================================

for path in [
    MASTER,
    INVENTORY_JSON,
]:

    if not path.exists():

        raise FileNotFoundError(
            path
        )


# ============================================================
# LOAD MASTER / INVENTORY
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
    inventory_data[
        "objects"
    ]

    if (
        isinstance(
            inventory_data,
            dict
        )
        and
        "objects" in inventory_data
    )

    else inventory_data
)


# ============================================================
# LOAD FLORENCE
# ============================================================

print("=" * 110)
print("PRODUCTION STAGE 06C14A")
print("FLORENCE MULTI-OBJECT RELOCALIZATION")
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
# PROCESS
# ============================================================

results = []


for iid in TARGET_IDS:

    obj = find_inventory_row(
        inventory,
        iid
    )


    name = str(
        obj.get(
            "name",
            obj.get(
                "inventory_name",
                ""
            )
        )
    )


    grounding_phrase = str(
        obj.get(
            "grounding_phrase",
            ""
        )
    ).strip()


    queries = list(
        QUERY_BANK[
            iid
        ]
    )


    # Add original inventory phrase as an additional
    # independent semantic phrasing if it is not already there.
    if (
        grounding_phrase
        and
        grounding_phrase.lower()
        not in
        {
            q.lower()
            for q in queries
        }
    ):

        queries.append(
            grounding_phrase
        )


    print()
    print("=" * 100)

    print(
        f"{iid:03d}. {name}"
    )

    print("=" * 100)


    raw_candidates = []


    for qidx, phrase in enumerate(
        queries,
        start=1
    ):

        print()
        print(
            f"QUERY #{qidx}:",
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
            "RETURNED:",
            len(
                boxes
            )
        )


        for box_item in boxes:

            bbox = clamp_box(
                box_item[
                    "bbox"
                ],
                W,
                H
            )


            record = {

                "inventory_id":
                    iid,

                "inventory_name":
                    name,

                "query_index":
                    qidx,

                "phrase":
                    phrase,

                "bbox":
                    bbox,

                "label":
                    box_item[
                        "label"
                    ],

                "score":
                    box_item[
                        "score"
                    ],

                "generated_text":
                    generated,
            }


            raw_candidates.append(
                record
            )


            print(
                "   bbox:",
                [
                    round(v, 1)
                    for v in bbox
                ],
                "| label:",
                box_item[
                    "label"
                ]
            )


    # ========================================================
    # DEDUP
    # ========================================================

    unique = deduplicate(
        raw_candidates
    )


    print()
    print(
        "RAW:",
        len(
            raw_candidates
        )
    )

    print(
        "UNIQUE:",
        len(
            unique
        )
    )


    # ========================================================
    # NUMBER UNIQUE CANDIDATES
    # ========================================================

    for idx, item in enumerate(
        unique,
        start=1
    ):

        item[
            "candidate_index"
        ] = idx


    # ========================================================
    # PREVIEW
    # ========================================================

    preview = master.copy()

    draw = ImageDraw.Draw(
        preview
    )


    for item in unique:

        idx = item[
            "candidate_index"
        ]

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


        draw.rectangle(
            [
                x1,
                y1,
                x1 + 30,
                y1 + 22
            ],
            fill="white",
            outline="red"
        )


        draw.text(
            (
                x1 + 6,
                y1 + 3
            ),
            str(idx),
            fill="red"
        )


    preview_path = (
        PREVIEW_DIR
        /
        f"{iid:03d}_{safe_name(name)}_candidates.png"
    )


    preview.save(
        preview_path
    )


    # ========================================================
    # RESULT
    # ========================================================

    results.append({

        "inventory_id":
            iid,

        "inventory_name":
            name,

        "original_grounding_phrase":
            grounding_phrase,

        "queries":
            queries,

        "raw_candidate_count":
            len(
                raw_candidates
            ),

        "unique_candidate_count":
            len(
                unique
            ),

        "candidates":
            unique,

        "preview_path":
            str(
                preview_path
            ),

        "status":
            (
                "CANDIDATES_FOUND"
                if unique
                else
                "NO_CANDIDATES"
            ),
    })


# ============================================================
# GLOBAL OVERVIEW
# ============================================================

overview = master.copy()

draw = ImageDraw.Draw(
    overview
)


for object_row in results:

    iid = object_row[
        "inventory_id"
    ]


    for item in object_row[
        "candidates"
    ]:

        x1, y1, x2, y2 = (
            item[
                "bbox"
            ]
        )


        label = (
            f"{iid:03d}:"
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
            width=2
        )


        draw.text(
            (
                x1 + 2,
                y1 + 2
            ),
            label,
            fill="red"
        )


OVERVIEW_PATH = (
    OUT
    / "01_all_candidates_overview.png"
)


overview.save(
    OVERVIEW_PATH
)


# ============================================================
# SAVE JSON
# ============================================================

RESULT_PATH = (
    OUT
    / "00_stage06c14a_results.json"
)


RESULT_PATH.write_text(
    json.dumps(
        results,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# ============================================================
# REPORT
# ============================================================

REPORT_PATH = (
    OUT
    / "02_stage06c14a_report.txt"
)


lines = [
    "=" * 100,
    "TEST07 PRODUCTION STAGE 06C14A",
    "FLORENCE MULTI-OBJECT RELOCALIZATION",
    "=" * 100,
    "",
]


for row in results:

    lines.append(
        (
            f'{row["inventory_id"]:03d}. '
            f'{row["inventory_name"]} '
            f'| raw={row["raw_candidate_count"]} '
            f'| unique={row["unique_candidate_count"]}'
        )
    )


    for item in row[
        "candidates"
    ]:

        lines.append(

            "   #{} bbox={} label={} support_queries={}".format(

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
                    "supporting_queries"
                ],
            )
        )


    lines.append(
        ""
    )


REPORT_PATH.write_text(
    "\n".join(
        lines
    ),
    encoding="utf-8"
)


# ============================================================
# PRINT
# ============================================================

print()
print("=" * 110)
print("PRODUCTION STAGE 06C14A RESULT")
print("=" * 110)


for row in results:

    print()
    print(
        f'{row["inventory_id"]:03d}. '
        f'{row["inventory_name"]}'
    )

    print(
        "RAW:",
        row[
            "raw_candidate_count"
        ]
    )

    print(
        "UNIQUE:",
        row[
            "unique_candidate_count"
        ]
    )


    for item in row[
        "candidates"
    ]:

        print(

            "  #{} bbox={} label={} support={}".format(

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
                    "supporting_queries"
                ],
            )
        )


    print(
        "  PREVIEW:",
        row[
            "preview_path"
        ]
    )


print()
print(
    "OVERVIEW:",
    OVERVIEW_PATH
)

print(
    "RESULT JSON:",
    RESULT_PATH
)

print(
    "REPORT:",
    REPORT_PATH
)

print()
print(
    "NO SAM2 OR FINAL MASK UNION HAS BEEN RUN."
)


# ============================================================
# CLEANUP
# ============================================================

del model

if torch.cuda.is_available():

    torch.cuda.empty_cache()
