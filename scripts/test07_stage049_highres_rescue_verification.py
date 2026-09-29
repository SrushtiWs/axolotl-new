
from pathlib import Path
import argparse
import json

import torch
from PIL import Image, ImageDraw

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)

from qwen_vl_utils import process_vision_info


MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

MAX_NEW_TOKENS = 80


PROMPT = """
You are validating a localization candidate for one physical object.

The supplied image is a contextual crop from a larger scene.
A RED RECTANGLE marks the proposed localization box.

TARGET OBJECT:
{name}

TARGET DESCRIPTION:
{phrase}

Decide whether the RED RECTANGLE is a usable localization box for
that exact physical object.

VALID means:
- the intended object is visibly present inside the rectangle;
- the rectangle corresponds primarily to that object;
- enough of the object is included for later segmentation.

INVALID means:
- the rectangle targets another object;
- it targets only a general region or container;
- the intended object is merely somewhere nearby;
- the rectangle contains only a tiny ambiguous part of the target;
- most of the intended object lies outside the rectangle.

IMPORTANT:
Judge the rectangle itself.

Examples:
- A vegetable basket is NOT automatically a valid localization
  box for one red pepper inside it.
- A countertop region is NOT automatically a valid box for a sink.
- A wall region containing a frame is NOT automatically a valid
  frame box.

Return exactly one line:

DECISION: VALID

or

DECISION: INVALID
""".strip()


def parse_decision(raw):

    upper = raw.upper()

    if "DECISION: INVALID" in upper:
        return "INVALID"

    if "DECISION: VALID" in upper:
        return "VALID"

    return "UNKNOWN"


def make_context_crop(
    master,
    bbox,
    margin_ratio=0.60
):

    W, H = master.size

    x1, y1, x2, y2 = [
        float(v)
        for v in bbox
    ]

    bw = max(
        2.0,
        x2 - x1
    )

    bh = max(
        2.0,
        y2 - y1
    )

    mx = max(
        18,
        bw * margin_ratio
    )

    my = max(
        18,
        bh * margin_ratio
    )

    cx1 = max(
        0,
        int(x1 - mx)
    )

    cy1 = max(
        0,
        int(y1 - my)
    )

    cx2 = min(
        W,
        int(x2 + mx)
    )

    cy2 = min(
        H,
        int(y2 + my)
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

    draw.rectangle(
        [
            x1 - cx1,
            y1 - cy1,
            x2 - cx1,
            y2 - cy1,
        ],
        outline="red",
        width=3
    )

    return crop


def run(
    master_path,
    candidate_json,
    output_dir,
    model_cache,
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

    crop_dir = (
        output_dir
        / "candidate_crops"
    )

    crop_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    master = (
        Image.open(
            master_path
        )
        .convert("RGB")
    )

    candidates = json.loads(
        candidate_json.read_text()
    )

    print("=" * 90)
    print("TEST07 STAGE04.9")
    print("HIGH-RES RESCUE CANDIDATE VERIFICATION")
    print("=" * 90)

    print(
        "INPUT CANDIDATES:",
        len(candidates)
    )

    dtype = (
        torch.float16
        if torch.cuda.is_available()
        else torch.float32
    )

    print()
    print("Loading Qwen 3B...")

    model = (
        Qwen2_5_VLForConditionalGeneration
        .from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            device_map="auto",
            cache_dir=str(
                model_cache
            )
        )
    )

    model.eval()

    processor = (
        AutoProcessor
        .from_pretrained(
            MODEL_ID,
            cache_dir=str(
                model_cache
            )
        )
    )

    print("✅ QWEN READY")

    results = []

    for index, row in enumerate(
        candidates,
        start=1
    ):

        iid = int(
            row["inventory_id"]
        )

        name = row[
            "inventory_name"
        ]

        phrase = row[
            "grounding_phrase"
        ]

        crop = make_context_crop(
            master,
            row["bbox"]
        )

        crop_path = (
            crop_dir
            /
            f"{index:03d}_{iid:02d}_{name.replace(' ', '_')}.png"
        )

        crop.save(
            crop_path
        )

        user_prompt = PROMPT.format(
            name=name,
            phrase=phrase
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": str(
                            crop_path
                        )
                    },
                    {
                        "type": "text",
                        "text": user_prompt
                    }
                ]
            }
        ]

        chat_text = (
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
            text=[chat_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )

        inputs = inputs.to(
            model.device
        )

        with torch.inference_mode():

            generated = model.generate(
                **inputs,
                max_new_tokens=
                    MAX_NEW_TOKENS,
                do_sample=False
            )

        trimmed = [
            output_ids[
                len(input_ids):
            ]
            for input_ids, output_ids
            in zip(
                inputs.input_ids,
                generated
            )
        ]

        raw = (
            processor
            .batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
        )

        decision = (
            parse_decision(
                raw
            )
        )

        out = dict(row)

        out[
            "stage049_decision"
        ] = decision

        out[
            "stage049_raw"
        ] = raw

        out[
            "stage049_crop_path"
        ] = str(
            crop_path
        )

        results.append(
            out
        )

        print(
            f'[{index:02d}/{len(candidates)}] '
            f'{iid:02d}. {name} | '
            f'DINO={float(row["score"]):.3f} | '
            f'{decision}'
        )

    # ========================================================
    # SAVE ALL
    # ========================================================

    all_path = (
        output_dir
        / "00_all_verification_results.json"
    )

    all_path.write_text(
        json.dumps(
            results,
            indent=2
        )
    )

    valid = [
        r
        for r in results
        if r[
            "stage049_decision"
        ] == "VALID"
    ]

    valid_path = (
        output_dir
        / "01_valid_rescue_candidates.json"
    )

    valid_path.write_text(
        json.dumps(
            valid,
            indent=2
        )
    )

    # ========================================================
    # BEST VALID PER INSTANCE
    # ========================================================

    ids = sorted(
        {
            int(
                r["inventory_id"]
            )
            for r in results
        }
    )

    best_valid = []
    per_instance = {}

    for iid in ids:

        rows = [
            r
            for r in results
            if int(
                r["inventory_id"]
            ) == iid
        ]

        valid_rows = [
            r
            for r in rows
            if r[
                "stage049_decision"
            ] == "VALID"
        ]

        valid_rows.sort(
            key=lambda r:
                float(
                    r["score"]
                ),
            reverse=True
        )

        best = (
            valid_rows[0]
            if valid_rows
            else None
        )

        if best:
            best_valid.append(
                best
            )

        per_instance[
            str(iid)
        ] = {
            "name":
                rows[0][
                    "inventory_name"
                ],

            "total":
                len(rows),

            "valid":
                len(valid_rows),

            "best_score":
                (
                    float(
                        best["score"]
                    )
                    if best
                    else None
                ),

            "best_bbox":
                (
                    best["bbox"]
                    if best
                    else None
                ),
        }

    best_path = (
        output_dir
        / "02_best_valid_per_instance.json"
    )

    best_path.write_text(
        json.dumps(
            best_valid,
            indent=2
        )
    )

    # ========================================================
    # PREVIEW
    # ========================================================

    preview = master.copy()

    draw = ImageDraw.Draw(
        preview
    )

    for row in best_valid:

        x1, y1, x2, y2 = (
            row["bbox"]
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

        draw.text(
            (
                x1,
                max(
                    0,
                    y1 - 14
                )
            ),
            (
                f'{row["inventory_id"]}:'
                f'{row["inventory_name"]} '
                f'{float(row["score"]):.2f}'
            ),
            fill="red"
        )

    preview_path = (
        output_dir
        / "03_best_valid_preview.png"
    )

    preview.save(
        preview_path
    )

    report = {
        "stage":
            "TEST07_STAGE049",

        "input_candidates":
            len(results),

        "valid_candidates":
            len(valid),

        "rescued_instances":
            len(best_valid),

        "per_instance":
            per_instance,

        "outputs": {
            "all":
                str(all_path),

            "valid":
                str(valid_path),

            "best":
                str(best_path),

            "preview":
                str(preview_path),
        }
    }

    (
        output_dir
        / "stage049_report.json"
    ).write_text(
        json.dumps(
            report,
            indent=2
        )
    )

    print()
    print("=" * 90)
    print("TEST07 STAGE04.9 RESULT")
    print("=" * 90)

    print(
        "TOTAL:",
        len(results)
    )

    print(
        "VALID:",
        len(valid)
    )

    print(
        "RESCUED INSTANCES:",
        len(best_valid)
    )

    print()
    print("PER INSTANCE:")

    for iid in ids:

        info = per_instance[
            str(iid)
        ]

        print(
            f'{iid:02d}. '
            f'{info["name"]} | '
            f'VALID={info["valid"]}/{info["total"]} | '
            f'best={info["best_score"]}'
        )

    print()
    print("PREVIEW:")
    print(preview_path)


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
            args.model_cache,
    )
