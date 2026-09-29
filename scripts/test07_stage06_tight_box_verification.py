
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
# TEST07 — STAGE 06
#
# QWEN TIGHT-BOX VERIFICATION
#
# INPUT:
#   Stage01 resized master
#   Stage05 filtered/rescued candidates
#
# PURPOSE:
#   Decide whether each candidate is geometrically safe to
#   send directly to segmentation.
#
# CLASSIFICATIONS:
#   TIGHT
#   LOOSE
#   PARTIAL
#   WRONG
#   AMBIGUOUS
#
# IMPORTANT:
#   - only TIGHT = segmentation safe
#   - Stage05 rescue state is preserved
#   - no Qwen-generated coordinates
#   - no SAM2
#   - no AQ modification
#   - reporting is INSTANCE-LEVEL, not generic-name level
# ============================================================


MODEL_ID = (
    "Qwen/Qwen2.5-VL-3B-Instruct"
)

CONTEXT_SCALE = 0.30

MIN_CONTEXT_PIXELS = 20


PROMPT = """
You are validating the QUALITY of an object-detection box.

The image is a contextual crop from a larger real scene.

A RED RECTANGLE marks the candidate bounding box.

TARGET OBJECT NAME:
{target_name}

TARGET DESCRIPTION:
{target_phrase}

Your job is NOT to discover objects and NOT to generate
coordinates.

Judge how suitable this RED RECTANGLE is for sending directly
to an image segmentation model.

Use exactly ONE classification:

TIGHT
The red rectangle corresponds to the intended physical object
and contains nearly all of that object while avoiding large
amounts of unrelated wall, floor, ceiling, background, or
other objects.

LOOSE
The intended object is inside the rectangle, but the box is
too broad and contains substantial unrelated scene/background
or multiple unrelated objects.

PARTIAL
The rectangle corresponds to the intended object, but a
significant part of the object lies outside the rectangle.

WRONG
The rectangle mainly corresponds to a different object,
continuous architectural surface, background, or the target
object is not actually present.

AMBIGUOUS
The image does not provide enough visual evidence to make a
reliable decision.

IMPORTANT:
- Merely containing the target somewhere inside the box is NOT
  enough for TIGHT.
- For TIGHT, the box should be segmentation-safe.
- A small amount of surrounding background is acceptable.
- If a giant rectangle contains a small target plus lots of
  wall/ceiling/floor/scene, classify LOOSE or WRONG.
- If two separate physical objects are both inside the box and
  the target is only one of them, classify LOOSE unless they
  form one clearly unified physical object.
- Do not assume the detector label is correct.
- Evaluate the visible red rectangle, not the textual label.

Return ONLY one line exactly in one of these formats:

DECISION: TIGHT
DECISION: LOOSE
DECISION: PARTIAL
DECISION: WRONG
DECISION: AMBIGUOUS
""".strip()


# ============================================================
# HELPERS
# ============================================================

def safe_name(text):

    return re.sub(
        r"[^a-zA-Z0-9_-]+",
        "_",
        str(text)
    )[:50]


def sanitize_box(
    box,
    W,
    H
):

    if (
        not isinstance(
            box,
            (list, tuple)
        )
        or
        len(box) != 4
    ):
        return None


    x1, y1, x2, y2 = [
        int(
            round(
                float(v)
            )
        )
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


    return [
        x1,
        y1,
        x2,
        y2
    ]


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
        "crops"
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
    # LOAD
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
            "Stage05 candidates must be a list."
        )


    print(
        "=" * 90
    )

    print(
        "TEST07 STAGE06"
    )

    print(
        "QWEN TIGHT-BOX VERIFICATION"
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
        "Stage05 candidates:",
        len(
            candidates
        )
    )


    # ========================================================
    # PREPARE CROPS
    # ========================================================

    prepared = []


    for idx, row in enumerate(
        candidates,
        start=1
    ):

        bbox = sanitize_box(
            row.get(
                "bbox"
            ),
            W,
            H
        )


        if bbox is None:
            continue


        x1, y1, x2, y2 = (
            bbox
        )


        bw = (
            x2 - x1
        )

        bh = (
            y2 - y1
        )


        pad_x = max(
            MIN_CONTEXT_PIXELS,
            int(
                round(
                    bw
                    *
                    CONTEXT_SCALE
                )
            )
        )


        pad_y = max(
            MIN_CONTEXT_PIXELS,
            int(
                round(
                    bh
                    *
                    CONTEXT_SCALE
                )
            )
        )


        cx1 = max(
            0,
            x1 - pad_x
        )

        cy1 = max(
            0,
            y1 - pad_y
        )

        cx2 = min(
            W,
            x2 + pad_x
        )

        cy2 = min(
            H,
            y2 + pad_y
        )


        crop = master.crop(
            (
                cx1,
                cy1,
                cx2,
                cy2
            )
        )


        draw = ImageDraw.Draw(
            crop
        )


        rx1 = (
            x1 - cx1
        )

        ry1 = (
            y1 - cy1
        )

        rx2 = (
            x2 - cx1
        )

        ry2 = (
            y2 - cy1
        )


        line_width = max(
            3,
            int(
                round(
                    min(
                        crop.size
                    )
                    *
                    0.009
                )
            )
        )


        draw.rectangle(
            [
                rx1,
                ry1,
                rx2,
                ry2
            ],
            outline="red",
            width=line_width
        )


        name = str(
            row.get(
                "inventory_name",
                "object"
            )
        )


        iid = int(
            row.get(
                "inventory_id",
                idx
            )
        )


        crop_path = (
            crop_dir
            /
            f"{idx:03d}_"
            f"{iid:02d}_"
            f"{safe_name(name)}.png"
        )


        crop.save(
            crop_path
        )


        item = dict(
            row
        )


        item[
            "stage06_index"
        ] = idx


        item[
            "stage06_context_bbox"
        ] = [
            int(cx1),
            int(cy1),
            int(cx2),
            int(cy2)
        ]


        item[
            "stage06_crop_path"
        ] = str(
            crop_path
        )


        prepared.append(
            item
        )


    manifest_path = (
        output_dir
        /
        "00_stage06_candidate_manifest.json"
    )


    manifest_path.write_text(
        json.dumps(
            prepared,
            indent=2
        )
    )


    # ========================================================
    # LOAD QWEN
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
        "✅ QWEN READY"
    )


    # ========================================================
    # VERIFY
    # ========================================================

    results = []


    for idx, row in enumerate(
        prepared,
        start=1
    ):

        name = str(
            row.get(
                "inventory_name",
                "object"
            )
        )


        phrase = str(
            row.get(
                "grounding_phrase",
                name
            )
        )


        crop_path = row[
            "stage06_crop_path"
        ]


        rescue = bool(
            row.get(
                "stage05_rescue",
                False
            )
        )


        print()
        print(
            "-" * 90
        )


        print(
            f"[{idx}/{len(prepared)}]",
            name,
            "|",
            (
                "RESCUE"
                if rescue
                else "STRICT"
            )
        )


        print(
            "DINO score:",
            round(
                float(
                    row.get(
                        "score",
                        0.0
                    )
                ),
                4
            )
        )


        user_prompt = (
            PROMPT.format(
                target_name=
                    name,
                target_phrase=
                    phrase
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
                            crop_path
                    },
                    {
                        "type":
                            "text",

                        "text":
                            user_prompt
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


        if (
            "AMBIGUOUS"
            in upper
        ):

            decision = (
                "AMBIGUOUS"
            )

        elif (
            "PARTIAL"
            in upper
        ):

            decision = (
                "PARTIAL"
            )

        elif (
            "LOOSE"
            in upper
        ):

            decision = (
                "LOOSE"
            )

        elif (
            "WRONG"
            in upper
        ):

            decision = (
                "WRONG"
            )

        elif (
            "TIGHT"
            in upper
        ):

            decision = (
                "TIGHT"
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
            "stage06_verification_raw"
        ] = (
            answer
        )


        out[
            "stage06_decision"
        ] = (
            decision
        )


        # ====================================================
        # FINAL SEGMENTATION-SAFETY GATE
        #
        # Qwen TIGHT is necessary but NOT sufficient.
        #
        # Stage05 RESCUE candidates are explicitly low-trust
        # and cannot become segmentation-safe merely because
        # Qwen says TIGHT.
        #
        # Extremely broad geometry is also blocked.
        # ====================================================

        stage05_rescue = bool(
            row.get(
                "stage05_rescue",
                False
            )
        )

        area_ratio = float(
            row.get(
                "area_ratio_stage05",
                row.get(
                    "area_ratio",
                    0.0
                )
            )
        )

        width_ratio = float(
            row.get(
                "width_ratio_stage05",
                0.0
            )
        )

        height_ratio = float(
            row.get(
                "height_ratio_stage05",
                0.0
            )
        )

        aq_coverage = float(
            row.get(
                "aq_coverage_stage05",
                0.0
            )
        )


        hard_geometry_fail = (
            area_ratio > 0.34
            or
            width_ratio > 0.95
            or
            (
                area_ratio > 0.22
                and
                aq_coverage < 0.20
            )
        )


        segmentation_safe = (
            decision == "TIGHT"
            and
            not stage05_rescue
            and
            not hard_geometry_fail
        )


        out[
            "stage06_segmentation_safe"
        ] = bool(
            segmentation_safe
        )


        out[
            "stage06_geometry_safety"
        ] = {
            "area_ratio":
                area_ratio,

            "width_ratio":
                width_ratio,

            "height_ratio":
                height_ratio,

            "aq_coverage":
                aq_coverage,

            "stage05_rescue":
                stage05_rescue,

            "hard_geometry_fail":
                bool(
                    hard_geometry_fail
                )
        }


        if decision != "TIGHT":

            out[
                "stage06_safety_reason"
            ] = "qwen_not_tight"

        elif stage05_rescue:

            out[
                "stage06_safety_reason"
            ] = "stage05_low_trust_rescue"

        elif hard_geometry_fail:

            out[
                "stage06_safety_reason"
            ] = "deterministic_geometry_fail"

        else:

            out[
                "stage06_safety_reason"
            ] = "tight_and_geometry_safe"


        results.append(
            out
        )


    # ========================================================
    # SAVE RESULTS
    # ========================================================

    all_json = (
        output_dir
        /
        "01_stage06_all_results.json"
    )


    all_json.write_text(
        json.dumps(
            results,
            indent=2
        )
    )


    safe = [
        row
        for row
        in results
        if row[
            "stage06_segmentation_safe"
        ]
    ]


    unsafe = [
        row
        for row
        in results
        if not row[
            "stage06_segmentation_safe"
        ]
    ]


    safe_json = (
        output_dir
        /
        "02_segmentation_safe_candidates.json"
    )


    unsafe_json = (
        output_dir
        /
        "03_non_safe_candidates.json"
    )


    safe_json.write_text(
        json.dumps(
            safe,
            indent=2
        )
    )


    unsafe_json.write_text(
        json.dumps(
            unsafe,
            indent=2
        )
    )


    # ========================================================
    # SAFE PREVIEW
    # ========================================================

    safe_preview = (
        master.copy()
    )


    draw = ImageDraw.Draw(
        safe_preview
    )


    for i, row in enumerate(
        safe,
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
            f'{i}: '
            f'{row["inventory_name"]}'
            f' | TIGHT'
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
                    text
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


    safe_preview_path = (
        output_dir
        /
        "04_segmentation_safe_preview.png"
    )


    safe_preview.save(
        safe_preview_path
    )


    # ========================================================
    # ALL DECISIONS PREVIEW
    # ========================================================

    all_preview = (
        master.copy()
    )


    draw = ImageDraw.Draw(
        all_preview
    )


    for i, row in enumerate(
        results,
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


        rescue_suffix = (
            " RESCUE"
            if row.get(
                "stage05_rescue",
                False
            )
            else ""
        )


        text = (
            f'{i}: '
            f'{row["inventory_name"]}'
            f' | {row["stage06_decision"]}'
            f'{rescue_suffix}'
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
                    text
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


    all_preview_path = (
        output_dir
        /
        "05_all_decisions_preview.png"
    )


    all_preview.save(
        all_preview_path
    )


    # ========================================================
    # COUNTS
    # ========================================================

    decision_counts = {
        "TIGHT": 0,
        "LOOSE": 0,
        "PARTIAL": 0,
        "WRONG": 0,
        "AMBIGUOUS": 0,
        "UNKNOWN": 0
    }


    for row in results:

        decision = (
            row[
                "stage06_decision"
            ]
        )


        decision_counts[
            decision
        ] = (
            decision_counts.get(
                decision,
                0
            )
            +
            1
        )


    # ========================================================
    # INSTANCE-LEVEL REPORT
    # ========================================================

    per_instance = {}


    for row in results:

        iid = str(
            row.get(
                "inventory_id",
                "unknown"
            )
        )


        if iid not in per_instance:

            per_instance[
                iid
            ] = {
                "name":
                    row.get(
                        "inventory_name",
                        "object"
                    ),

                "total":
                    0,

                "tight":
                    0,

                "loose":
                    0,

                "partial":
                    0,

                "wrong":
                    0,

                "ambiguous":
                    0,

                "unknown":
                    0,

                "had_stage05_rescue":
                    False
            }


        info = (
            per_instance[
                iid
            ]
        )


        info[
            "total"
        ] += 1


        key = (
            row[
                "stage06_decision"
            ]
            .lower()
        )


        if key in info:

            info[
                key
            ] += 1


        if row.get(
            "stage05_rescue",
            False
        ):

            info[
                "had_stage05_rescue"
            ] = True


    instances_with_tight = sum(
        1
        for info
        in per_instance.values()
        if info[
            "tight"
        ]
        > 0
    )


    instances_without_tight = (
        len(
            per_instance
        )
        -
        instances_with_tight
    )


    rescue_instance_results = {}


    for iid, info in (
        per_instance.items()
    ):

        if info[
            "had_stage05_rescue"
        ]:

            rescue_instance_results[
                iid
            ] = info


    # ========================================================
    # REPORT
    # ========================================================

    report = {
        "experiment":
            "TEST07_STAGE06",

        "method":
            (
                "Qwen2.5-VL-3B tight-box quality "
                "verification before segmentation"
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

        "input_count":
            len(
                results
            ),

        "safe_count":
            len(
                safe
            ),

        "unsafe_count":
            len(
                unsafe
            ),

        "decision_counts":
            decision_counts,

        "inventory_instances":
            len(
                per_instance
            ),

        "instances_with_tight_candidate":
            instances_with_tight,

        "instances_without_tight_candidate":
            instances_without_tight,

        "per_inventory_instance":
            per_instance,

        "stage05_rescue_instance_results":
            rescue_instance_results,

        "guarantees": [
            "Stage05 rescue status preserved",
            "no Qwen-generated coordinates",
            "no SAM2",
            "no AQ modification",
            "same Stage01 coordinate system",
            "TIGHT only is segmentation-safe"
        ],

        "outputs": {
            "manifest":
                str(
                    manifest_path
                ),

            "all_results":
                str(
                    all_json
                ),

            "safe_json":
                str(
                    safe_json
                ),

            "unsafe_json":
                str(
                    unsafe_json
                ),

            "safe_preview":
                str(
                    safe_preview_path
                ),

            "all_preview":
                str(
                    all_preview_path
                ),

            "crops":
                str(
                    crop_dir
                )
        }
    }


    report_path = (
        output_dir
        /
        "stage06_report.json"
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
        "TEST07 STAGE06 RESULT"
    )

    print(
        "=" * 90
    )


    print(
        "INPUT:",
        len(
            results
        )
    )


    print(
        "SEGMENTATION SAFE:",
        len(
            safe
        )
    )


    print(
        "NON-SAFE:",
        len(
            unsafe
        )
    )


    print()
    print(
        "DECISIONS:"
    )


    for key in [
        "TIGHT",
        "LOOSE",
        "PARTIAL",
        "WRONG",
        "AMBIGUOUS",
        "UNKNOWN"
    ]:

        print(
            f"{key:12s}",
            decision_counts.get(
                key,
                0
            )
        )


    print()
    print(
        "INSTANCE COVERAGE:"
    )


    print(
        "WITH TIGHT:",
        instances_with_tight
    )


    print(
        "WITHOUT TIGHT:",
        instances_without_tight
    )


    print()
    print(
        "PER INSTANCE:"
    )


    for iid in sorted(
        per_instance,
        key=lambda x:
            int(
                x
            )
    ):

        info = (
            per_instance[
                iid
            ]
        )


        rescue_text = (
            " | STAGE05 RESCUE"
            if info[
                "had_stage05_rescue"
            ]
            else ""
        )


        print(
            f'{int(iid):02d}.',
            info[
                "name"
            ],
            "→",
            f'TIGHT={info["tight"]}',
            f'/ TOTAL={info["total"]}',
            rescue_text
        )


    print()
    print(
        "IMPORTANT OUTPUT:"
    )

    print(
        safe_preview_path
    )


    print()
    print(
        "ALL DECISIONS:"
    )

    print(
        all_preview_path
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
