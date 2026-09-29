
from pathlib import Path
import argparse
import gc
import json
import re

import torch
from PIL import Image, ImageDraw

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor
)

from qwen_vl_utils import process_vision_info


# ============================================================
# TEST07 — STAGE 04
#
# QWEN CROP-LEVEL DINO CANDIDATE VERIFICATION
#
# INPUT:
#   Stage01 resized master
#   Stage03 deduplicated Grounding DINO candidates
#
# PURPOSE:
#   Verify whether each proposed DINO bbox genuinely
#   corresponds to its requested inventory object.
#
# DIFFERENCE FROM TEST06:
#   TEST06 BB3 consumed a pre-generated crop manifest.
#   TEST07 Stage04 generates those contextual crops
#   automatically, removing the hidden/manual dependency.
#
# OUTPUT:
#   candidate_crops/
#   00_candidate_manifest.json
#   01_qwen_candidate_verification.json
#   02_verified_candidates.json
#   03_verified_candidates_preview.png
#   stage04_report.json
#
# NO SAM2.
# NO AQ MODIFICATION.
# NO ROOM-SPECIFIC COORDINATES.
# ============================================================


MODEL_ID = (
    "Qwen/Qwen2.5-VL-3B-Instruct"
)

# Context around candidate bbox.
#
# 35% of bbox dimension on every side,
# with normalized minimum/maximum padding.
CONTEXT_SCALE = 0.35

MIN_CONTEXT_RATIO = 0.025
MAX_CONTEXT_RATIO = 0.14


BASE_PROMPT = """
You are validating an object-detection candidate.

The image is a contextual crop from a larger real scene.

A RED RECTANGLE marks the candidate bounding box proposed by
an object-localization model.

TARGET OBJECT:
{target_name}

TARGET DESCRIPTION:
{target_phrase}

Your job is NOT to discover new objects.

Judge ONLY whether the RED RECTANGLE is a useful localization
of the requested target.

Return VALID only when:
- the requested physical object is visibly present inside the
  red rectangle, and
- the rectangle meaningfully corresponds to that object.

Return INVALID when:
- the rectangle mainly covers wall, floor, ceiling, empty
  background, reflection, shadow, glare, or unrelated scene;
- the requested object is absent;
- the rectangle clearly corresponds to a different object;
- the box is extremely broad and the target is only a tiny or
  incidental part of it;
- the semantic label appears hallucinated.

A little surrounding background is acceptable.
The box does NOT need to be perfectly tight.

Return ONLY one of these:

DECISION: VALID

or

DECISION: INVALID
""".strip()


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


def safe_name(
    text
):

    return re.sub(
        r"[^a-zA-Z0-9_-]+",
        "_",
        str(text)
    )[:50]


def create_context_crop(
    master,
    bbox
):

    W, H = master.size

    x1, y1, x2, y2 = bbox

    bw = (
        x2 - x1
    )

    bh = (
        y2 - y1
    )


    min_pad_x = (
        W
        *
        MIN_CONTEXT_RATIO
    )

    min_pad_y = (
        H
        *
        MIN_CONTEXT_RATIO
    )


    max_pad_x = (
        W
        *
        MAX_CONTEXT_RATIO
    )

    max_pad_y = (
        H
        *
        MAX_CONTEXT_RATIO
    )


    pad_x = min(
        max_pad_x,
        max(
            min_pad_x,
            bw
            *
            CONTEXT_SCALE
        )
    )


    pad_y = min(
        max_pad_y,
        max(
            min_pad_y,
            bh
            *
            CONTEXT_SCALE
        )
    )


    crop_x1 = max(
        0,
        int(
            round(
                x1
                -
                pad_x
            )
        )
    )

    crop_y1 = max(
        0,
        int(
            round(
                y1
                -
                pad_y
            )
        )
    )

    crop_x2 = min(
        W,
        int(
            round(
                x2
                +
                pad_x
            )
        )
    )

    crop_y2 = min(
        H,
        int(
            round(
                y2
                +
                pad_y
            )
        )
    )


    crop = master.crop(
        (
            crop_x1,
            crop_y1,
            crop_x2,
            crop_y2
        )
    )


    # Candidate bbox in crop-local coordinates.
    local_box = [
        x1 - crop_x1,
        y1 - crop_y1,
        x2 - crop_x1,
        y2 - crop_y1
    ]


    return (
        crop,
        [
            crop_x1,
            crop_y1,
            crop_x2,
            crop_y2
        ],
        local_box
    )


# ============================================================
# MAIN
# ============================================================

def run(
    master_path,
    candidate_json,
    output_dir,
    model_cache
):

    master_path = Path(
        master_path
    )

    candidate_json = Path(
        candidate_json
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


    crop_dir = (
        output_dir
        /
        "candidate_crops"
    )

    crop_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    if not master_path.exists():

        raise FileNotFoundError(
            master_path
        )


    if not candidate_json.exists():

        raise FileNotFoundError(
            candidate_json
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
        candidate_json,
        "r",
        encoding="utf-8"
    ) as f:

        candidates = json.load(
            f
        )


    if not isinstance(
        candidates,
        list
    ):

        raise RuntimeError(
            "Stage03 candidate JSON must be a list."
        )


    print(
        "=" * 90
    )

    print(
        "TEST07 STAGE04"
    )

    print(
        "QWEN CANDIDATE VERIFICATION"
    )

    print(
        "=" * 90
    )


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
        "Candidates:",
        len(
            candidates
        )
    )


    # ========================================================
    # BUILD CONTEXTUAL CROPS
    # ========================================================

    manifest = []


    for index, row in enumerate(
        candidates,
        start=1
    ):

        bbox = sanitize_box(
            row.get(
                "bbox",
                []
            ),
            W,
            H
        )


        if bbox is None:
            continue


        crop, crop_bbox, local_box = (
            create_context_crop(
                master,
                bbox
            )
        )


        draw = ImageDraw.Draw(
            crop
        )


        lx1, ly1, lx2, ly2 = (
            local_box
        )


        # Draw a visible but not overly thick red rectangle.
        line_width = max(
            2,
            int(
                round(
                    min(
                        crop.size
                    )
                    *
                    0.008
                )
            )
        )


        draw.rectangle(
            [
                lx1,
                ly1,
                lx2,
                ly2
            ],
            outline="red",
            width=line_width
        )


        safe = safe_name(
            row.get(
                "inventory_name",
                "object"
            )
        )


        crop_path = (
            crop_dir
            /
            f"{index:03d}_{safe}.png"
        )


        crop.save(
            crop_path
        )


        out = dict(
            row
        )


        out[
            "crop_path"
        ] = str(
            crop_path
        )


        out[
            "context_crop_bbox"
        ] = [
            int(v)
            for v
            in crop_bbox
        ]


        out[
            "candidate_bbox_local"
        ] = [
            float(v)
            for v
            in local_box
        ]


        manifest.append(
            out
        )


    manifest_path = (
        output_dir
        /
        "00_candidate_manifest.json"
    )


    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2
        )
    )


    print(
        "Context crops created:",
        len(
            manifest
        )
    )


    # ========================================================
    # LOAD QWEN 3B
    # ========================================================

    dtype = (
        torch.float16
        if torch.cuda.is_available()
        else torch.float32
    )


    print()
    print(
        "Loading Qwen2.5-VL-3B..."
    )


    model = (
        Qwen2_5_VLForConditionalGeneration
        .from_pretrained(
            MODEL_ID,
            torch_dtype=
                dtype,
            device_map=
                "auto",
            cache_dir=
                str(
                    model_cache
                )
        )
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


    print(
        "✅ QWEN VERIFIER READY"
    )


    # ========================================================
    # VERIFY
    # ========================================================

    results = []


    for index, row in enumerate(
        manifest,
        start=1
    ):

        target_name = str(
            row.get(
                "inventory_name",
                "object"
            )
        )


        target_phrase = str(
            row.get(
                "grounding_phrase",
                target_name
            )
        )


        crop_path = Path(
            row[
                "crop_path"
            ]
        )


        print()
        print(
            "-" * 90
        )

        print(
            f"[{index}/{len(manifest)}]",
            target_name
        )


        print(
            "DINO score:",
            round(
                float(
                    row.get(
                        "score",
                        0
                    )
                ),
                4
            )
        )


        prompt = (
            BASE_PROMPT.format(
                target_name=
                    target_name,
                target_phrase=
                    target_phrase
            )
        )


        messages = [
            {
                "role":
                    "user",

                "content": [
                    {
                        "type":
                            "image",

                        "image":
                            str(
                                crop_path
                            )
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


        text = (
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
                text
            ],
            images=
                image_inputs,
            videos=
                video_inputs,
            padding=
                True,
            return_tensors=
                "pt"
        )


        inputs = inputs.to(
            model.device
        )


        with torch.inference_mode():

            generated_ids = (
                model.generate(
                    **inputs,
                    max_new_tokens=
                        16,
                    do_sample=
                        False
                )
            )


        trimmed = [
            output_ids[
                len(input_ids):
            ]
            for input_ids, output_ids
            in zip(
                inputs.input_ids,
                generated_ids
            )
        ]


        answer = (
            processor
            .batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            .strip()
        )


        upper = (
            answer.upper()
        )


        # IMPORTANT:
        # INVALID contains VALID,
        # so check INVALID first.
        if (
            "INVALID"
            in upper
        ):

            decision = (
                "INVALID"
            )

        elif (
            "VALID"
            in upper
        ):

            decision = (
                "VALID"
            )

        else:

            decision = (
                "UNKNOWN"
            )


        print(
            "Qwen:",
            answer
        )

        print(
            "Decision:",
            decision
        )


        out = dict(
            row
        )


        out[
            "verification_raw"
        ] = (
            answer
        )


        out[
            "verification_decision"
        ] = (
            decision
        )


        results.append(
            out
        )


    # ========================================================
    # SAVE ALL DECISIONS
    # ========================================================

    result_json = (
        output_dir
        /
        "01_qwen_candidate_verification.json"
    )


    result_json.write_text(
        json.dumps(
            results,
            indent=2
        )
    )


    # ========================================================
    # ACCEPTED ONLY
    # ========================================================

    valid = [
        row
        for row
        in results
        if row[
            "verification_decision"
        ]
        ==
        "VALID"
    ]


    valid_json = (
        output_dir
        /
        "02_verified_candidates.json"
    )


    valid_json.write_text(
        json.dumps(
            valid,
            indent=2
        )
    )


    # ========================================================
    # PREVIEW
    # ========================================================

    preview = (
        master.copy()
    )


    draw = ImageDraw.Draw(
        preview
    )


    for index, row in enumerate(
        valid,
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


        label = (
            f'{index}: '
            f'{row["inventory_name"]} '
            f'{float(row["score"]):.2f}'
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
            -
            18
        )


        label_width = min(
            W - tx,
            max(
                120,
                len(
                    label
                )
                *
                7
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
                    ty
                    +
                    18
                )
            ],
            fill="white"
        )


        draw.text(
            (
                tx + 2,
                ty + 2
            ),
            label,
            fill="black"
        )


    preview_path = (
        output_dir
        /
        "03_verified_candidates_preview.png"
    )


    preview.save(
        preview_path
    )


    # ========================================================
    # PER INVENTORY INSTANCE
    # ========================================================

    per_inventory = {}


    for row in results:

        inventory_id = str(
            row.get(
                "inventory_id",
                "unknown"
            )
        )


        if (
            inventory_id
            not in per_inventory
        ):

            per_inventory[
                inventory_id
            ] = {
                "name":
                    row.get(
                        "inventory_name",
                        "object"
                    ),

                "total":
                    0,

                "valid":
                    0,

                "invalid":
                    0,

                "unknown":
                    0
            }


        item = (
            per_inventory[
                inventory_id
            ]
        )


        item[
            "total"
        ] += 1


        decision = (
            row[
                "verification_decision"
            ]
            .lower()
        )


        if (
            decision
            in item
        ):

            item[
                decision
            ] += 1


    valid_inventory_ids = {
        int(
            row[
                "inventory_id"
            ]
        )
        for row
        in valid
    }


    all_inventory_ids = {
        int(
            row[
                "inventory_id"
            ]
        )
        for row
        in manifest
    }


    inventory_with_valid = (
        len(
            valid_inventory_ids
        )
    )


    inventory_without_valid = (
        len(
            all_inventory_ids
            -
            valid_inventory_ids
        )
    )


    # ========================================================
    # REPORT
    # ========================================================

    report = {
        "experiment":
            "TEST07_STAGE04",

        "method":
            (
                "Qwen2.5-VL-3B contextual crop "
                "verification of Grounding DINO candidates"
            ),

        "model":
            MODEL_ID,

        "master":
            str(
                master_path
            ),

        "candidate_json":
            str(
                candidate_json
            ),

        "candidate_count":
            len(
                results
            ),

        "valid_count":
            len(
                valid
            ),

        "invalid_count":
            sum(
                row[
                    "verification_decision"
                ]
                ==
                "INVALID"
                for row
                in results
            ),

        "unknown_count":
            sum(
                row[
                    "verification_decision"
                ]
                ==
                "UNKNOWN"
                for row
                in results
            ),

        "inventory_with_valid_candidate":
            inventory_with_valid,

        "inventory_without_valid_candidate":
            inventory_without_valid,

        "per_inventory":
            per_inventory,

        "crop_generation": {
            "context_scale":
                CONTEXT_SCALE,

            "min_context_ratio":
                MIN_CONTEXT_RATIO,

            "max_context_ratio":
                MAX_CONTEXT_RATIO
        },

        "guarantees": [
            "context crops generated dynamically",
            "no hidden candidate manifest dependency",
            "same Stage01 coordinate system",
            "no SAM2",
            "no AQ modification",
            "no room-specific crop coordinates"
        ],

        "outputs": {
            "candidate_manifest":
                str(
                    manifest_path
                ),

            "all_decisions":
                str(
                    result_json
                ),

            "verified_json":
                str(
                    valid_json
                ),

            "verified_preview":
                str(
                    preview_path
                ),

            "candidate_crops":
                str(
                    crop_dir
                )
        }
    }


    report_path = (
        output_dir
        /
        "stage04_report.json"
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
        "TEST07 STAGE04 RESULT"
    )

    print(
        "=" * 90
    )


    print(
        "TOTAL:",
        len(
            results
        )
    )


    print(
        "VALID:",
        len(
            valid
        )
    )


    print(
        "INVALID:",
        report[
            "invalid_count"
        ]
    )


    print(
        "UNKNOWN:",
        report[
            "unknown_count"
        ]
    )


    print(
        "INVENTORY WITH VALID:",
        inventory_with_valid
    )


    print(
        "INVENTORY WITHOUT VALID:",
        inventory_without_valid
    )


    print()
    print(
        "PER INVENTORY INSTANCE:"
    )


    for inventory_id in sorted(
        per_inventory,
        key=lambda x:
            int(
                x
            )
    ):

        item = (
            per_inventory[
                inventory_id
            ]
        )


        print(
            f'{int(inventory_id):02d}.',
            item[
                "name"
            ],
            "→",
            f'{item["valid"]}/{item["total"]}',
            "VALID"
        )


    print()
    print(
        "IMPORTANT OUTPUT:"
    )

    print(
        preview_path
    )


    print()
    print(
        "CROPS:"
    )

    print(
        crop_dir
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
        "--candidate-json",
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

        candidate_json=
            args.candidate_json,

        output_dir=
            args.output_dir,

        model_cache=
            args.model_cache
    )
