
from pathlib import Path
import argparse
import gc
import json
import re

import numpy as np
import torch
from PIL import Image, ImageDraw

from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection
)


# ============================================================
# TEST07 — STAGE 03
#
# DYNAMIC INVENTORY → GROUNDING DINO TOP-K
#
# INPUT:
#   Stage01 resized master
#   Stage02 locked instance-level Qwen inventory
#
# PURPOSE:
#   Localize each discovered physical entity independently.
#
# IMPORTANT:
#   - discovery stage only
#   - permissive thresholds
#   - no SAM2
#   - no mask modification
#   - no room-specific coordinates
#   - same settings across TEST07 rooms
#
# OUTPUT:
#   00_dino_topk_all.json
#   01_dino_topk_raw_preview.png
#   02_dino_per_phrase_deduplicated.json
#   03_dino_deduplicated_preview.png
#   contact_sheets/
#   stage03_report.json
# ============================================================


MODEL_ID = (
    "IDEA-Research/grounding-dino-base"
)

BOX_THRESHOLD = 0.18
TEXT_THRESHOLD = 0.15

TOP_K_PER_PHRASE = 5

# Only suppress essentially identical boxes.
DEDUP_IOU = 0.88


# ============================================================
# HELPERS
# ============================================================

def box_iou(a, b):

    ax1, ay1, ax2, ay2 = [
        float(v) for v in a
    ]

    bx1, by1, bx2, by2 = [
        float(v) for v in b
    ]

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
        iw * ih
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
            float(W),
            x1
        )
    )

    y1 = max(
        0.0,
        min(
            float(H),
            y1
        )
    )

    x2 = max(
        0.0,
        min(
            float(W),
            x2
        )
    )

    y2 = max(
        0.0,
        min(
            float(H),
            y2
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
        y2
    ]


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


def safe_name(text):

    return re.sub(
        r"[^a-zA-Z0-9_-]+",
        "_",
        str(text)
    )[:50]


# ============================================================
# MAIN
# ============================================================

def run(
    master_path,
    inventory_json,
    output_dir,
    model_cache
):

    master_path = Path(
        master_path
    )

    inventory_json = Path(
        inventory_json
    )

    output_dir = Path(
        output_dir
    )

    model_cache = Path(
        model_cache
    )


    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    model_cache.mkdir(
        parents=True,
        exist_ok=True
    )


    if not master_path.exists():
        raise FileNotFoundError(
            master_path
        )

    if not inventory_json.exists():
        raise FileNotFoundError(
            inventory_json
        )


    # ========================================================
    # LOAD INPUTS
    # ========================================================

    master = (
        Image.open(
            master_path
        )
        .convert(
            "RGB"
        )
    )

    W, H = (
        master.size
    )


    with open(
        inventory_json,
        "r",
        encoding="utf-8"
    ) as f:

        inventory = json.load(
            f
        )


    objects = inventory.get(
        "objects",
        []
    )


    if not isinstance(
        objects,
        list
    ):

        raise RuntimeError(
            "Stage02 inventory objects must be a list."
        )


    print(
        "=" * 90
    )

    print(
        "TEST07 STAGE03"
    )

    print(
        "DYNAMIC INVENTORY → GROUNDING DINO"
    )

    print(
        "=" * 90
    )


    print()
    print(
        "MASTER:",
        master_path
    )

    print(
        "Resolution:",
        W,
        "x",
        H
    )

    print(
        "Inventory objects:",
        len(
            objects
        )
    )


    # ========================================================
    # DEVICE
    # ========================================================

    DEVICE = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    print(
        "Device:",
        DEVICE
    )

    if DEVICE == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )


    # ========================================================
    # MODEL
    # ========================================================

    print()
    print(
        "Loading Grounding DINO..."
    )


    processor = (
        AutoProcessor
        .from_pretrained(
            MODEL_ID,
            cache_dir=
                str(
                    model_cache
                )
        )
    )


    model = (
        AutoModelForZeroShotObjectDetection
        .from_pretrained(
            MODEL_ID,
            cache_dir=
                str(
                    model_cache
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


    # ========================================================
    # DETECTION
    # ========================================================

    all_candidates = []


    for object_index, obj in enumerate(
        objects,
        start=1
    ):

        inventory_id = int(
            obj.get(
                "id",
                object_index
            )
        )


        name = str(
            obj.get(
                "name",
                "object"
            )
        ).strip()


        phrase = str(
            obj.get(
                "grounding_phrase",
                name
            )
        ).strip()


        query = phrase

        if not query.endswith(
            "."
        ):

            query += "."


        print()
        print(
            "-" * 90
        )

        print(
            f"[{object_index}/{len(objects)}]"
        )

        print(
            "NAME:",
            name
        )

        print(
            "QUERY:",
            query
        )


        inputs = processor(
            images=
                master,
            text=
                query,
            return_tensors=
                "pt"
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


        # ====================================================
        # POSTPROCESS
        # ====================================================

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


        for detection_index, box in enumerate(
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
                        detection_index
                    ]
                )
                if detection_index
                <
                len(scores)
                else 0.0
            )


            detected_label = (
                str(
                    text_labels[
                        detection_index
                    ]
                )
                if detection_index
                <
                len(
                    text_labels
                )
                else phrase
            )


            row = {
                "inventory_id":
                    inventory_id,

                "inventory_name":
                    name,

                "grounding_phrase":
                    phrase,

                "qwen_confidence":
                    obj.get(
                        "confidence",
                        "unknown"
                    ),

                "dino_label":
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
                    )
            }


            local.append(
                row
            )


        # Highest confidence first.
        local = sorted(
            local,
            key=lambda r:
                r[
                    "score"
                ],
            reverse=True
        )


        local = (
            local[
                :TOP_K_PER_PHRASE
            ]
        )


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
                "rank_for_phrase"
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
                "area=",
                round(
                    row[
                        "area_ratio"
                    ],
                    4
                ),
                "box=",
                [
                    round(
                        v,
                        1
                    )
                    for v in row[
                        "bbox"
                    ]
                ]
            )


            all_candidates.append(
                row
            )


    # ========================================================
    # SAVE RAW
    # ========================================================

    raw_json_path = (
        output_dir
        / "00_dino_topk_all.json"
    )


    with open(
        raw_json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            all_candidates,
            f,
            indent=2
        )


    # ========================================================
    # RAW PREVIEW
    # ========================================================

    raw_preview = (
        master.copy()
    )

    draw = ImageDraw.Draw(
        raw_preview
    )


    for index, row in enumerate(
        all_candidates,
        start=1
    ):

        x1, y1, x2, y2 = (
            row[
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


        text = (
            f'{index}: '
            f'{row["inventory_name"]}'
            f' #{row["rank_for_phrase"]}'
            f' {row["score"]:.2f}'
        )


        tx = max(
            0,
            int(
                x1
            )
        )


        ty = max(
            0,
            int(
                y1
            )
            - 17
        )


        label_width = min(
            W - tx,
            max(
                100,
                len(
                    text
                )
                * 7
            )
        )


        draw.rectangle(
            [
                tx,
                ty,
                tx
                +
                label_width,
                min(
                    H,
                    ty + 17
                )
            ],
            fill="white"
        )


        draw.text(
            (
                tx + 2,
                ty + 1
            ),
            text,
            fill="black"
        )


    raw_preview_path = (
        output_dir
        / "01_dino_topk_raw_preview.png"
    )


    raw_preview.save(
        raw_preview_path
    )


    # ========================================================
    # PER-INVENTORY DEDUP
    # ========================================================

    deduped = []


    inventory_ids = sorted(
        set(
            row[
                "inventory_id"
            ]
            for row
            in all_candidates
        )
    )


    for inventory_id in inventory_ids:

        group = [
            row
            for row
            in all_candidates
            if row[
                "inventory_id"
            ]
            ==
            inventory_id
        ]


        group = sorted(
            group,
            key=lambda r:
                r[
                    "score"
                ],
            reverse=True
        )


        keep = []


        for candidate in group:

            duplicate = False


            for existing in keep:

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

                keep.append(
                    candidate
                )


        deduped.extend(
            keep
        )


    dedup_json_path = (
        output_dir
        / "02_dino_per_phrase_deduplicated.json"
    )


    with open(
        dedup_json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            deduped,
            f,
            indent=2
        )


    # ========================================================
    # DEDUP PREVIEW
    # ========================================================

    dedup_preview = (
        master.copy()
    )

    draw = ImageDraw.Draw(
        dedup_preview
    )


    for index, row in enumerate(
        deduped,
        start=1
    ):

        x1, y1, x2, y2 = (
            row[
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


        text = (
            f'{index}: '
            f'{row["inventory_name"]}'
            f' {row["score"]:.2f}'
        )


        tx = max(
            0,
            int(
                x1
            )
        )


        ty = max(
            0,
            int(
                y1
            )
            - 18
        )


        label_width = min(
            W - tx,
            max(
                100,
                len(
                    text
                )
                * 7
            )
        )


        draw.rectangle(
            [
                tx,
                ty,
                tx
                +
                label_width,
                min(
                    H,
                    ty + 18
                )
            ],
            fill="white"
        )


        draw.text(
            (
                tx + 2,
                ty + 2
            ),
            text,
            fill="black"
        )


    dedup_preview_path = (
        output_dir
        / "03_dino_deduplicated_preview.png"
    )


    dedup_preview.save(
        dedup_preview_path
    )


    # ========================================================
    # CONTACT SHEETS
    # ========================================================

    contact_dir = (
        output_dir
        / "contact_sheets"
    )


    contact_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    for obj in objects:

        inventory_id = int(
            obj[
                "id"
            ]
        )


        candidates = [
            row
            for row
            in deduped
            if row[
                "inventory_id"
            ]
            ==
            inventory_id
        ]


        if not candidates:
            continue


        thumb_w = 256
        thumb_h = 256


        sheet = Image.new(
            "RGB",
            (
                thumb_w
                *
                len(
                    candidates
                ),
                thumb_h
                +
                40
            ),
            "white"
        )


        sheet_draw = (
            ImageDraw.Draw(
                sheet
            )
        )


        for j, row in enumerate(
            candidates
        ):

            x1, y1, x2, y2 = [
                int(
                    round(
                        v
                    )
                )
                for v in row[
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


            crop.thumbnail(
                (
                    thumb_w,
                    thumb_h
                ),
                Image.Resampling.LANCZOS
            )


            offset_x = (
                j
                *
                thumb_w
                +
                (
                    thumb_w
                    -
                    crop.width
                )
                // 2
            )


            offset_y = (
                (
                    thumb_h
                    -
                    crop.height
                )
                // 2
            )


            sheet.paste(
                crop,
                (
                    offset_x,
                    offset_y
                )
            )


            sheet_draw.text(
                (
                    j
                    *
                    thumb_w
                    +
                    5,
                    thumb_h
                    +
                    8
                ),
                (
                    f'#{j+1} '
                    f'{row["score"]:.2f}'
                ),
                fill="black"
            )


        safe = safe_name(
            obj[
                "name"
            ]
        )


        sheet.save(
            contact_dir
            /
            f'{inventory_id:02d}_{safe}.png'
        )


    # ========================================================
    # REPORT
    # ========================================================

    counts = {}


    for obj in objects:

        oid = int(
            obj[
                "id"
            ]
        )


        matches = [
            row
            for row
            in deduped
            if row[
                "inventory_id"
            ]
            ==
            oid
        ]


        counts[
            str(
                oid
            )
        ] = {
            "name":
                obj[
                    "name"
                ],

            "grounding_phrase":
                obj[
                    "grounding_phrase"
                ],

            "candidate_count":
                len(
                    matches
                ),

            "best_score":
                (
                    max(
                        row[
                            "score"
                        ]
                        for row
                        in matches
                    )
                    if matches
                    else None
                )
        }


    inventory_with_candidates = sum(
        1
        for item
        in counts.values()
        if item[
            "candidate_count"
        ]
        > 0
    )


    inventory_without_candidates = (
        len(
            objects
        )
        -
        inventory_with_candidates
    )


    report = {
        "experiment":
            "TEST07_STAGE03",

        "method":
            "Stage02 instance inventory grounded independently using Grounding DINO",

        "master":
            str(
                master_path
            ),

        "inventory_json":
            str(
                inventory_json
            ),

        "model":
            MODEL_ID,

        "box_threshold":
            BOX_THRESHOLD,

        "text_threshold":
            TEXT_THRESHOLD,

        "top_k_per_phrase":
            TOP_K_PER_PHRASE,

        "dedup_iou":
            DEDUP_IOU,

        "inventory_count":
            len(
                objects
            ),

        "raw_candidate_count":
            len(
                all_candidates
            ),

        "deduplicated_candidate_count":
            len(
                deduped
            ),

        "inventory_with_candidates":
            inventory_with_candidates,

        "inventory_without_candidates":
            inventory_without_candidates,

        "per_inventory_item":
            counts,

        "guarantees": [
            "same Stage01 coordinate system",
            "Stage02 inventory not modified",
            "one grounding phrase processed independently per inventory instance",
            "permissive proposal discovery only",
            "no SAM2",
            "no room-specific coordinates"
        ],

        "outputs": {
            "raw_json":
                str(
                    raw_json_path
                ),

            "raw_preview":
                str(
                    raw_preview_path
                ),

            "dedup_json":
                str(
                    dedup_json_path
                ),

            "dedup_preview":
                str(
                    dedup_preview_path
                ),

            "contact_sheets":
                str(
                    contact_dir
                )
        }
    }


    report_path = (
        output_dir
        / "stage03_report.json"
    )


    report_path.write_text(
        json.dumps(
            report,
            indent=2
        )
    )


    # ========================================================
    # RESULT
    # ========================================================

    print()
    print(
        "=" * 90
    )

    print(
        "TEST07 STAGE03 RESULT"
    )

    print(
        "=" * 90
    )


    print(
        "INVENTORY ITEMS:",
        len(
            objects
        )
    )


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


    print(
        "INVENTORY WITH CANDIDATES:",
        inventory_with_candidates
    )


    print(
        "INVENTORY WITHOUT CANDIDATES:",
        inventory_without_candidates
    )


    print()
    print(
        "PER INVENTORY ITEM:"
    )


    for obj in objects:

        oid = str(
            obj[
                "id"
            ]
        )

        info = counts[
            oid
        ]


        print(
            f'{int(oid):02d}.',
            info[
                "name"
            ],
            "→",
            info[
                "candidate_count"
            ],
            "candidate(s)",
            "| best:",
            (
                round(
                    info[
                        "best_score"
                    ],
                    3
                )
                if info[
                    "best_score"
                ]
                is not None
                else "NONE"
            )
        )


    print()
    print(
        "IMPORTANT OUTPUT:"
    )

    print(
        dedup_preview_path
    )


    print()
    print(
        "CONTACT SHEETS:"
    )

    print(
        contact_dir
    )


    # ========================================================
    # CLEANUP
    # ========================================================

    del model
    del processor

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--master",
        required=True
    )


    parser.add_argument(
        "--inventory-json",
        required=True
    )


    parser.add_argument(
        "--output-dir",
        required=True
    )


    parser.add_argument(
        "--model-cache",
        default=
            "/workspace/data/huggingface-cache"
    )


    args = parser.parse_args()


    run(
        master_path=
            args.master,

        inventory_json=
            args.inventory_json,

        output_dir=
            args.output_dir,

        model_cache=
            args.model_cache
    )
