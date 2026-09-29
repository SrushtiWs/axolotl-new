
from pathlib import Path
import json
import gc

import numpy as np
import torch

from PIL import Image, ImageDraw, ImageFont

from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
)


# ============================================================
# PATHS
# ============================================================

BASE = Path(
    "/workspace/axolotl"
)

PROD = (
    BASE
    / "test08"
    / "production_pipeline"
)

MASTER_PATH = (
    PROD
    / "stage01_master"
    / "00_master_input.png"
)

OUT = (
    PROD
    / "stage03_mirror"
    / "03d_dino_mirror_discovery"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

CACHE = Path(
    "/workspace/data/huggingface-cache"
)

CACHE.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOCKED TEST07 SETTINGS
# ============================================================

MODEL_ID = (
    "IDEA-Research/grounding-dino-base"
)

BOX_THRESHOLD = 0.18
TEXT_THRESHOLD = 0.15

TOP_K_PER_PHRASE = 5
DEDUP_IOU = 0.88


# ============================================================
# MIRROR-FAMILY DISCOVERY PHRASES
#
# Semantic variants only.
# No room-specific description.
# ============================================================

PHRASES = [
    "mirror",
    "wall mirror",
    "round mirror",
    "circular mirror",
    "framed mirror",
    "wall-mounted mirror",
]


# ============================================================
# HELPERS
# ============================================================

def sanitize_box(
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
            x1,
            W - 1
        )
    )

    y1 = max(
        0.0,
        min(
            y1,
            H - 1
        )
    )

    x2 = max(
        0.0,
        min(
            x2,
            W
        )
    )

    y2 = max(
        0.0,
        min(
            y2,
            H
        )
    )

    if (
        x2 <= x1
        or
        y2 <= y1
    ):
        return None

    return [
        x1,
        y1,
        x2,
        y2,
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

    inter = (
        iw
        *
        ih
    )

    area_a = (
        max(
            0.0,
            ax2 - ax1
        )
        *
        max(
            0.0,
            ay2 - ay1
        )
    )

    area_b = (
        max(
            0.0,
            bx2 - bx1
        )
        *
        max(
            0.0,
            by2 - by1
        )
    )

    union = (
        area_a
        +
        area_b
        -
        inter
    )

    if union <= 0:
        return 0.0

    return (
        inter
        /
        union
    )


def area_ratio(
    box,
    W,
    H
):

    x1, y1, x2, y2 = box

    area = (
        max(
            0.0,
            x2 - x1
        )
        *
        max(
            0.0,
            y2 - y1
        )
    )

    return (
        area
        /
        float(
            W * H
        )
    )


# ============================================================
# LOAD MASTER
# ============================================================

if not MASTER_PATH.exists():

    raise FileNotFoundError(
        MASTER_PATH
    )


master = Image.open(
    MASTER_PATH
).convert(
    "RGB"
)

W, H = master.size


print("=" * 110)
print("TEST08 STAGE03D — MIRROR DINO DISCOVERY")
print("=" * 110)

print()
print(
    "MASTER:",
    MASTER_PATH
)

print(
    "SIZE:",
    W,
    "x",
    H
)

print()
print(
    "MODEL:",
    MODEL_ID
)

print(
    "BOX_THRESHOLD:",
    BOX_THRESHOLD
)

print(
    "TEXT_THRESHOLD:",
    TEXT_THRESHOLD
)

print(
    "TOP_K_PER_PHRASE:",
    TOP_K_PER_PHRASE
)

print(
    "DEDUP_IOU:",
    DEDUP_IOU
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# LOAD MODEL
# ============================================================

print()
print(
    "Loading Grounding DINO..."
)


processor = (
    AutoProcessor
    .from_pretrained(
        MODEL_ID,
        cache_dir=str(
            CACHE
        )
    )
)


model = (
    AutoModelForZeroShotObjectDetection
    .from_pretrained(
        MODEL_ID,
        cache_dir=str(
            CACHE
        )
    )
    .to(
        DEVICE
    )
)


model.eval()


print(
    "✅ GROUNDING DINO READY"
)


# ============================================================
# DISCOVERY
# ============================================================

all_candidates = []


for phrase_index, phrase in enumerate(
    PHRASES,
    start=1
):

    query = phrase

    if not query.endswith(
        "."
    ):
        query += "."


    print()
    print("-" * 110)

    print(
        f"PHRASE {phrase_index}:",
        phrase
    )


    inputs = processor(
        images=master,
        text=query,
        return_tensors="pt"
    )


    inputs = {
        k:
            (
                v.to(
                    DEVICE
                )
                if torch.is_tensor(
                    v
                )
                else v
            )

        for k, v
        in inputs.items()
    }


    with torch.inference_mode():

        outputs = model(
            **inputs
        )


    try:

        processed = (
            processor
            .post_process_grounded_object_detection(
                outputs,
                inputs[
                    "input_ids"
                ],
                threshold=
                    BOX_THRESHOLD,
                text_threshold=
                    TEXT_THRESHOLD,
                target_sizes=[
                    (
                        H,
                        W
                    )
                ]
            )
        )[0]


    except TypeError:

        processed = (
            processor
            .post_process_grounded_object_detection(
                outputs,
                inputs[
                    "input_ids"
                ],
                box_threshold=
                    BOX_THRESHOLD,
                text_threshold=
                    TEXT_THRESHOLD,
                target_sizes=[
                    (
                        H,
                        W
                    )
                ]
            )
        )[0]


    boxes = processed.get(
        "boxes",
        []
    )

    scores = processed.get(
        "scores",
        []
    )

    text_labels = (
        processed.get(
            "text_labels"
        )
        or
        processed.get(
            "labels"
        )
        or []
    )


    if torch.is_tensor(
        boxes
    ):

        boxes = (
            boxes
            .detach()
            .cpu()
            .numpy()
        )


    if torch.is_tensor(
        scores
    ):

        scores = (
            scores
            .detach()
            .cpu()
            .numpy()
        )


    local = []


    for idx, box in enumerate(
        boxes
    ):

        clean_box = sanitize_box(
            box,
            W,
            H
        )

        if clean_box is None:
            continue


        score = (
            float(
                scores[
                    idx
                ]
            )
            if idx < len(
                scores
            )
            else 0.0
        )


        detected_label = (
            str(
                text_labels[
                    idx
                ]
            )
            if idx < len(
                text_labels
            )
            else phrase
        )


        row = {

            "phrase":
                phrase,

            "detected_label":
                detected_label,

            "score":
                score,

            "bbox":
                clean_box,

            "area_ratio":
                area_ratio(
                    clean_box,
                    W,
                    H
                ),
        }


        local.append(
            row
        )


    local.sort(
        key=lambda r:
            r[
                "score"
            ],
        reverse=True
    )


    local = local[
        :
        TOP_K_PER_PHRASE
    ]


    print(
        "Candidates:",
        len(
            local
        )
    )


    for rank, row in enumerate(
        local,
        start=1
    ):

        row[
            "rank_within_phrase"
        ] = rank


        print(
            f"  #{rank}",
            "score=",
            round(
                row[
                    "score"
                ],
                4
            ),
            "bbox=",
            [
                round(
                    v,
                    1
                )
                for v
                in row[
                    "bbox"
                ]
            ],
            "area=",
            round(
                row[
                    "area_ratio"
                ],
                4
            )
        )


        all_candidates.append(
            row
        )


# ============================================================
# GLOBAL NEAR-DUPLICATE SUPPRESSION
#
# Same 0.88 IoU principle as TEST07.
# We sort globally by score first.
# ============================================================

all_candidates.sort(
    key=lambda r:
        r[
            "score"
        ],
    reverse=True
)


deduped = []


for candidate in all_candidates:

    duplicate = False


    for existing in deduped:

        if (
            box_iou(
                candidate[
                    "bbox"
                ],
                existing[
                    "bbox"
                ]
            )
            >=
            DEDUP_IOU
        ):

            duplicate = True
            break


    if not duplicate:

        deduped.append(
            candidate
        )


# ============================================================
# SAVE JSON
# ============================================================

RAW_JSON = (
    OUT
    / "00_all_mirror_family_candidates.json"
)


DEDUP_JSON = (
    OUT
    / "01_deduplicated_mirror_candidates.json"
)


RAW_JSON.write_text(
    json.dumps(
        all_candidates,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


DEDUP_JSON.write_text(
    json.dumps(
        deduped,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# ============================================================
# RAW PREVIEW
# ============================================================

raw_preview = master.copy()

draw = ImageDraw.Draw(
    raw_preview
)


for index, row in enumerate(
    all_candidates,
    start=1
):

    x1, y1, x2, y2 = row[
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

    draw.text(
        (
            x1 + 2,
            max(
                0,
                y1 - 13
            )
        ),
        (
            f"{index}: "
            f"{row['phrase']} "
            f"{row['score']:.2f}"
        ),
        fill="red"
    )


RAW_PREVIEW_PATH = (
    OUT
    / "02_raw_mirror_candidates_preview.png"
)


raw_preview.save(
    RAW_PREVIEW_PATH
)


# ============================================================
# DEDUP PREVIEW
# ============================================================

dedup_preview = master.copy()

draw = ImageDraw.Draw(
    dedup_preview
)


for index, row in enumerate(
    deduped,
    start=1
):

    x1, y1, x2, y2 = row[
        "bbox"
    ]

    draw.rectangle(
        [
            x1,
            y1,
            x2,
            y2
        ],
        outline="lime",
        width=3
    )

    draw.text(
        (
            x1 + 2,
            max(
                0,
                y1 - 14
            )
        ),
        (
            f"{index}: "
            f"{row['score']:.2f} "
            f"{row['phrase']}"
        ),
        fill="lime"
    )


DEDUP_PREVIEW_PATH = (
    OUT
    / "03_deduplicated_mirror_candidates_preview.png"
)


dedup_preview.save(
    DEDUP_PREVIEW_PATH
)


# ============================================================
# CONTACT SHEET OF DEDUP CANDIDATES
# ============================================================

thumb_w = 260
thumb_h = 220

n = len(
    deduped
)


if n > 0:

    sheet = Image.new(
        "RGB",
        (
            thumb_w * n,
            thumb_h + 70
        ),
        "white"
    )


    sheet_draw = ImageDraw.Draw(
        sheet
    )


    for i, row in enumerate(
        deduped
    ):

        x1, y1, x2, y2 = [
            int(
                round(
                    v
                )
            )
            for v
            in row[
                "bbox"
            ]
        ]


        crop = master.crop(
            (
                x1,
                y1,
                x2,
                y2
            )
        )


        if crop.width <= 0 or crop.height <= 0:
            continue


        crop.thumbnail(
            (
                thumb_w - 10,
                thumb_h - 10
            )
        )


        canvas = Image.new(
            "RGB",
            (
                thumb_w,
                thumb_h
            ),
            "white"
        )


        px = (
            thumb_w
            -
            crop.width
        ) // 2

        py = (
            thumb_h
            -
            crop.height
        ) // 2


        canvas.paste(
            crop,
            (
                px,
                py
            )
        )


        sheet.paste(
            canvas,
            (
                i * thumb_w,
                0
            )
        )


        sheet_draw.text(
            (
                i * thumb_w + 5,
                thumb_h + 4
            ),
            (
                f"#{i+1} score={row['score']:.3f}\n"
                f"{row['phrase']}"
            ),
            fill="black"
        )


    CONTACT_PATH = (
        OUT
        / "04_deduplicated_mirror_contact_sheet.png"
    )


    sheet.save(
        CONTACT_PATH
    )


else:

    CONTACT_PATH = None


# ============================================================
# RESULT STATE
# ============================================================

STATE = {

    "stage":
        "TEST08_STAGE03D_MIRROR_DINO_DISCOVERY",

    "model":
        MODEL_ID,

    "settings": {

        "box_threshold":
            BOX_THRESHOLD,

        "text_threshold":
            TEXT_THRESHOLD,

        "top_k_per_phrase":
            TOP_K_PER_PHRASE,

        "dedup_iou":
            DEDUP_IOU,
    },

    "phrases":
        PHRASES,

    "counts": {

        "raw_candidates":
            len(
                all_candidates
            ),

        "deduplicated_candidates":
            len(
                deduped
            ),
    },

    "guarantees": [

        "same TEST08 Stage01 coordinate system",

        "no SAM2",

        "no manual bbox",

        "no room-specific coordinates",

        "Grounding-DINO settings unchanged from TEST07 discovery stack",
    ],

    "outputs": {

        "raw_json":
            str(
                RAW_JSON
            ),

        "dedup_json":
            str(
                DEDUP_JSON
            ),

        "raw_preview":
            str(
                RAW_PREVIEW_PATH
            ),

        "dedup_preview":
            str(
                DEDUP_PREVIEW_PATH
            ),

        "contact_sheet":
            (
                str(
                    CONTACT_PATH
                )
                if CONTACT_PATH
                is not None
                else None
            ),
    },

    "status":
        "RND_REQUIRES_VISUAL_BBOX_AUDIT",
}


STATE_PATH = (
    OUT
    / "00_stage03d_result.json"
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
# PRINT
# ============================================================

print()
print("=" * 110)
print("TEST08 STAGE03D RESULT")
print("=" * 110)

print()

print(
    "RAW CANDIDATES:",
    len(
        all_candidates
    )
)

print(
    "DEDUPLICATED CANDIDATES:",
    len(
        deduped
    )
)


print()
print(
    "DEDUPLICATED LIST:"
)


for i, row in enumerate(
    deduped,
    start=1
):

    print(
        f"#{i}",
        "| score:",
        round(
            row[
                "score"
            ],
            4
        ),
        "| phrase:",
        row[
            "phrase"
        ],
        "| bbox:",
        [
            round(
                v,
                1
            )
            for v
            in row[
                "bbox"
            ]
        ]
    )


print()
print(
    "RAW PREVIEW:",
    RAW_PREVIEW_PATH
)

print(
    "DEDUP PREVIEW:",
    DEDUP_PREVIEW_PATH
)

print(
    "CONTACT SHEET:",
    CONTACT_PATH
)

print(
    "STATE:",
    STATE_PATH
)


# ============================================================
# DISPLAY
# ============================================================

from IPython.display import display


print()
print(
    "DEDUPLICATED MIRROR CANDIDATE PREVIEW"
)

display(
    Image.open(
        DEDUP_PREVIEW_PATH
    )
)


if CONTACT_PATH is not None:

    print()
    print(
        "DEDUPLICATED MIRROR CONTACT SHEET"
    )

    display(
        Image.open(
            CONTACT_PATH
        )
    )


# ============================================================
# CLEANUP
# ============================================================

del model
del processor

gc.collect()


if torch.cuda.is_available():

    torch.cuda.empty_cache()
