
from pathlib import Path
import json
import sys

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from transformers import (
    AutoProcessor,
    AutoModelForMultimodalLM,
)


# ============================================================
# SAM2 IMPORT
# ============================================================

for source in [
    Path("/workspace/sam2_src"),
    Path("/workspace/axolotl/sam2"),
]:

    if (
        source.exists()
        and
        str(source) not in sys.path
    ):
        sys.path.insert(
            0,
            str(source)
        )


from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


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
    / "06c13a_florence_wall_switch_sam2"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

OBJECTS = (
    OUT
    / "objects"
)

OBJECTS.mkdir(
    parents=True,
    exist_ok=True
)

CACHE = (
    "/workspace/data/huggingface-cache"
)

FLORENCE_MODEL_ID = (
    "florence-community/Florence-2-large"
)

SAM2_CHECKPOINT = Path(
    "/workspace/axolotl/test07/models/sam2/"
    "sam2.1_hiera_large.pt"
)

SAM2_CONFIG = (
    "configs/sam2.1/"
    "sam2.1_hiera_l.yaml"
)


# ============================================================
# CONFIG
# ============================================================

TARGET_ID = 9

PROMPT_EXPANSION = 0.10
GATE_EXPANSION = 0.22

MIN_SAM_SCORE = 0.55
MIN_CONTAINMENT = 0.85
MAX_BORDER_TOUCH = 0.30
MIN_MASK_PIXELS = 12


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

        if (
            rid is not None
            and
            int(rid) == iid
        ):

            return row

    raise KeyError(
        iid
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
                    str(task_key),
            })

    return results


def run_florence_grounding(
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


def make_gate(
    box,
    H,
    W
):

    x1, y1, x2, y2 = [
        int(round(v))
        for v in box
    ]

    x1 = max(
        0,
        min(
            W - 1,
            x1
        )
    )

    y1 = max(
        0,
        min(
            H - 1,
            y1
        )
    )

    x2 = max(
        x1 + 1,
        min(
            W,
            x2
        )
    )

    y2 = max(
        y1 + 1,
        min(
            H,
            y2
        )
    )

    mask = np.zeros(
        (
            H,
            W
        ),
        dtype=bool
    )

    mask[
        y1:y2,
        x1:x2
    ] = True

    return mask


def largest_component(
    mask
):

    n, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask.astype(
                np.uint8
            ),
            8
        )
    )

    if n <= 1:
        return mask

    idx = (
        1
        +
        int(
            np.argmax(
                stats[
                    1:,
                    cv2.CC_STAT_AREA
                ]
            )
        )
    )

    return (
        labels
        ==
        idx
    )


def border_touch_ratio(
    mask,
    box
):

    H, W = mask.shape

    x1, y1, x2, y2 = [
        int(round(v))
        for v in box
    ]

    x1 = max(
        0,
        min(
            W - 1,
            x1
        )
    )

    y1 = max(
        0,
        min(
            H - 1,
            y1
        )
    )

    x2 = max(
        x1 + 1,
        min(
            W,
            x2
        )
    )

    y2 = max(
        y1 + 1,
        min(
            H,
            y2
        )
    )

    border = np.zeros_like(
        mask,
        dtype=bool
    )

    t = 2

    border[
        y1:min(
            y2,
            y1 + t
        ),
        x1:x2
    ] = True

    border[
        max(
            y1,
            y2 - t
        ):y2,
        x1:x2
    ] = True

    border[
        y1:y2,
        x1:min(
            x2,
            x1 + t
        )
    ] = True

    border[
        y1:y2,
        max(
            x1,
            x2 - t
        ):x2
    ] = True


    return float(
        (
            mask
            &
            border
        ).sum()
        /
        max(
            1,
            int(
                mask.sum()
            )
        )
    )


# ============================================================
# VALIDATE
# ============================================================

for p in [
    MASTER,
    INVENTORY_JSON,
    SAM2_CHECKPOINT,
]:

    if not p.exists():

        raise FileNotFoundError(
            p
        )


# ============================================================
# LOAD MASTER / INVENTORY
# ============================================================

master_pil = Image.open(
    MASTER
).convert(
    "RGB"
)

master = np.asarray(
    master_pil
)

W, H = master_pil.size


inventory_data = json.loads(
    INVENTORY_JSON.read_text(
        encoding="utf-8"
    )
)

inventory = (
    inventory_data[
        "objects"
    ]
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


name = str(
    target.get(
        "name",
        target.get(
            "inventory_name",
            "wall switch"
        )
    )
)


grounding_phrase = str(
    target.get(
        "grounding_phrase",
        ""
    )
).strip()


# ============================================================
# FLORENCE
# ============================================================

print("=" * 110)
print("PRODUCTION STAGE 06C13A")
print("FLORENCE RELOCALIZATION + SAM2 — 009 WALL SWITCH")
print("=" * 110)


print()
print(
    "Loading Florence-2..."
)


processor = (
    AutoProcessor
    .from_pretrained(
        FLORENCE_MODEL_ID,
        cache_dir=CACHE
    )
)


florence = (
    AutoModelForMultimodalLM
    .from_pretrained(
        FLORENCE_MODEL_ID,
        dtype=torch.float16,
        device_map="auto",
        cache_dir=CACHE
    )
)


florence.eval()


print(
    "✅ FLORENCE READY"
)


# Use multiple descriptions because original inventory wording
# "silver toggle switch" may be too narrow.
phrases = [

    (
        "rectangular electrical wall switch plate "
        "mounted on the patterned wall"
    ),

    (
        "small rectangular wall-mounted light switch "
        "or electrical switch plate"
    ),

    (
        grounding_phrase
        if grounding_phrase
        else
        "wall switch"
    ),
]


all_boxes = []


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
        run_florence_grounding(
            master_pil,
            phrase,
            processor,
            florence
        )
    )


    print(
        "BOXES:",
        len(
            boxes
        )
    )


    for item in boxes:

        bbox = clamp_box(
            item[
                "bbox"
            ],
            W,
            H
        )


        record = {

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

            "generated":
                generated,
        }


        all_boxes.append(
            record
        )


        print(
            "  bbox:",
            [
                round(v, 1)
                for v in bbox
            ],
            "| label:",
            item[
                "label"
            ]
        )


if not all_boxes:

    raise RuntimeError(
        "Florence returned no wall-switch candidates."
    )


# ============================================================
# DEDUP FLORENCE BOXES
# ============================================================

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

    inter = iw * ih

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

    return inter / max(
        1.0,
        area_a
        +
        area_b
        -
        inter
    )


unique = []


for item in all_boxes:

    duplicate = False

    for kept in unique:

        if box_iou(
            item[
                "bbox"
            ],
            kept[
                "bbox"
            ]
        ) >= 0.70:

            duplicate = True
            break


    if not duplicate:

        unique.append(
            item
        )


print()
print(
    "UNIQUE FLORENCE CANDIDATES:",
    len(
        unique
    )
)


# ============================================================
# FLORENCE PREVIEW BEFORE SAM2
# ============================================================

candidate_preview = master_pil.copy()

draw = ImageDraw.Draw(
    candidate_preview
)


for idx, item in enumerate(
    unique,
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
        str(idx),
        fill="red"
    )


CANDIDATE_PREVIEW_PATH = (
    OUT
    / "00_florence_wall_switch_candidates.png"
)


candidate_preview.save(
    CANDIDATE_PREVIEW_PATH
)


# ============================================================
# LOAD SAM2
# ============================================================

# Free Florence GPU before SAM2.
del florence

if torch.cuda.is_available():

    torch.cuda.empty_cache()


device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print()
print(
    "Loading SAM2..."
)


sam2_model = build_sam2(
    SAM2_CONFIG,
    str(
        SAM2_CHECKPOINT
    ),
    device=device
)


predictor = SAM2ImagePredictor(
    sam2_model
)


predictor.set_image(
    master
)


print(
    "✅ SAM2 READY"
)


# ============================================================
# RUN SAM2 ON EVERY UNIQUE FLORENCE CANDIDATE
# ============================================================

results = []


for cidx, candidate in enumerate(
    unique,
    start=1
):

    source_box = candidate[
        "bbox"
    ]


    prompt_box = expand_box(
        source_box,
        PROMPT_EXPANSION,
        W,
        H
    )


    gate_box = expand_box(
        source_box,
        GATE_EXPANSION,
        W,
        H
    )


    gate = make_gate(
        gate_box,
        H,
        W
    )


    with torch.inference_mode():

        masks, scores, _ = (
            predictor.predict(
                box=np.asarray(
                    prompt_box,
                    dtype=np.float32
                ),
                multimask_output=True
            )
        )


    evaluated = []


    for midx in range(
        len(
            masks
        )
    ):

        raw = (
            masks[
                midx
            ]
            >
            0
        )


        if not raw.any():
            continue


        raw = largest_component(
            raw
        )


        raw_area = int(
            raw.sum()
        )


        local = (
            raw
            &
            gate
        )


        area = int(
            local.sum()
        )


        if area == 0:
            continue


        containment = float(
            area
            /
            max(
                1,
                raw_area
            )
        )


        sam_score = float(
            scores[
                midx
            ]
        )


        border_touch = (
            border_touch_ratio(
                local,
                prompt_box
            )
        )


        quality = (

            0.70
            *
            sam_score

            +

            0.20
            *
            containment

            +

            0.10
            *
            (
                1.0
                -
                min(
                    1.0,
                    border_touch
                    /
                    MAX_BORDER_TOUCH
                )
            )
        )


        evaluated.append({

            "mask_index":
                int(
                    midx
                ),

            "mask":
                local,

            "sam_score":
                sam_score,

            "quality":
                quality,

            "area":
                area,

            "containment":
                containment,

            "border_touch":
                border_touch,
        })


    if not evaluated:
        continue


    evaluated.sort(
        key=lambda r:
            r[
                "quality"
            ],
        reverse=True
    )


    best = evaluated[
        0
    ]


    geometry_pass = bool(

        best[
            "area"
        ]
        >=
        MIN_MASK_PIXELS

        and

        best[
            "sam_score"
        ]
        >=
        MIN_SAM_SCORE

        and

        best[
            "containment"
        ]
        >=
        MIN_CONTAINMENT

        and

        best[
            "border_touch"
        ]
        <=
        MAX_BORDER_TOUCH
    )


    mask = best[
        "mask"
    ]


    prefix = (
        f"candidate_{cidx:02d}"
    )


    mask_path = (
        OBJECTS
        /
        f"{prefix}_mask.png"
    )


    rgba_path = (
        OBJECTS
        /
        f"{prefix}_rgba.png"
    )


    preview_path = (
        OBJECTS
        /
        f"{prefix}_preview.png"
    )


    Image.fromarray(
        mask.astype(
            np.uint8
        )
        *
        255
    ).save(
        mask_path
    )


    rgba = np.zeros(
        (
            H,
            W,
            4
        ),
        dtype=np.uint8
    )


    rgba[
        :,
        :,
        :3
    ] = master


    rgba[
        :,
        :,
        3
    ] = (
        mask.astype(
            np.uint8
        )
        *
        255
    )


    Image.fromarray(
        rgba,
        mode="RGBA"
    ).save(
        rgba_path
    )


    preview = master_pil.copy()

    draw = ImageDraw.Draw(
        preview
    )


    draw.rectangle(
        source_box,
        outline="blue",
        width=2
    )


    draw.rectangle(
        prompt_box,
        outline="red",
        width=2
    )


    contours, _ = cv2.findContours(
        (
            mask.astype(
                np.uint8
            )
            *
            255
        ),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    for contour in contours:

        pts = [

            (
                int(
                    p[0][0]
                ),
                int(
                    p[0][1]
                )
            )

            for p in contour
        ]


        if len(
            pts
        ) >= 2:

            draw.line(
                pts
                +
                [
                    pts[0]
                ],
                fill="lime",
                width=2
            )


    preview.save(
        preview_path
    )


    results.append({

        "candidate_index":
            cidx,

        "inventory_id":
            TARGET_ID,

        "inventory_name":
            name,

        "query_index":
            candidate[
                "query_index"
            ],

        "phrase":
            candidate[
                "phrase"
            ],

        "florence_label":
            candidate[
                "label"
            ],

        "florence_bbox":
            source_box,

        "prompt_bbox":
            prompt_box,

        "sam2_score":
            best[
                "sam_score"
            ],

        "quality":
            best[
                "quality"
            ],

        "mask_area":
            best[
                "area"
            ],

        "containment":
            best[
                "containment"
            ],

        "border_touch":
            best[
                "border_touch"
            ],

        "geometry_pass":
            geometry_pass,

        "mask_path":
            str(
                mask_path
            ),

        "rgba_path":
            str(
                rgba_path
            ),

        "preview_path":
            str(
                preview_path
            ),
    })


# ============================================================
# SAVE
# ============================================================

RESULT_PATH = (
    OUT
    / "01_stage06c13a_results.json"
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
print("PRODUCTION STAGE 06C13A RESULT")
print("=" * 110)


for row in results:

    print()

    print(
        "candidate #{} | bbox={} | label={} | SAM={} | area={} | contain={} | border={} | gate={}".format(

            row[
                "candidate_index"
            ],

            [
                round(v, 1)
                for v in row[
                    "florence_bbox"
                ]
            ],

            row[
                "florence_label"
            ],

            round(
                row[
                    "sam2_score"
                ],
                4
            ),

            row[
                "mask_area"
            ],

            round(
                row[
                    "containment"
                ],
                4
            ),

            round(
                row[
                    "border_touch"
                ],
                4
            ),

            row[
                "geometry_pass"
            ],
        )
    )


print()
print(
    "FLORENCE CANDIDATES:",
    CANDIDATE_PREVIEW_PATH
)

print(
    "RESULT JSON:",
    RESULT_PATH
)

print()
print(
    "NO FINAL PROP MASK UNION WAS CREATED."
)


# ============================================================
# CLEANUP
# ============================================================

del predictor
del sam2_model

if torch.cuda.is_available():

    torch.cuda.empty_cache()
