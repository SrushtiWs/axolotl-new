
from pathlib import Path
import argparse
import json
import re

import torch

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)

from qwen_vl_utils import process_vision_info


MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
MAX_NEW_TOKENS = 220


PROMPT = """
You are auditing possible object-instance collisions in one real image.

Each PAIR contains two inventory descriptions whose segmentation
masks are nearly identical.

For every pair, determine whether the descriptions refer to:

SAME_INSTANCE
Both descriptions correspond to the same single physical object
visible in the image.

DISTINCT_INSTANCES
They correspond to two different physical objects visible in the image.

UNCERTAIN
The image does not provide enough evidence.

IMPORTANT RULES:

1. Judge physical identity, not semantic similarity.
2. Two objects of the same class can be DISTINCT_INSTANCES.
3. Different names can still be SAME_INSTANCE if they clearly
   describe the same physical object.
4. Do not infer hidden objects.
5. Use the complete image as primary evidence.
6. Be conservative. If two separately visible objects fit the
   descriptions, return DISTINCT_INSTANCES.
7. Do not provide coordinates or new object descriptions.

Return ONLY:

pair_id || SAME_INSTANCE_or_DISTINCT_INSTANCES_or_UNCERTAIN || confidence

confidence must be:
high
medium
low

No JSON.
No explanation.
No markdown.
""".strip()


def parse(raw):

    out = {}

    for line in raw.splitlines():

        if "||" not in line:
            continue

        parts = [
            x.strip()
            for x in line.split("||")
        ]

        if len(parts) != 3:
            continue

        m = re.search(
            r"\d+",
            parts[0]
        )

        if not m:
            continue

        pid = int(
            m.group()
        )

        decision = (
            parts[1]
            .upper()
            .strip()
        )

        if decision not in {
            "SAME_INSTANCE",
            "DISTINCT_INSTANCES",
            "UNCERTAIN",
        }:
            continue

        confidence = (
            parts[2]
            .lower()
            .strip()
        )

        if confidence not in {
            "high",
            "medium",
            "low"
        }:
            confidence = "medium"

        out[pid] = {
            "decision": decision,
            "confidence": confidence,
        }

    return out


def run(
    master_path,
    inventory_json,
    overlap_json,
    output_dir,
    model_cache,
):

    master_path = Path(master_path)
    inventory_json = Path(inventory_json)
    overlap_json = Path(overlap_json)
    output_dir = Path(output_dir)
    model_cache = Path(model_cache)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    inventory = json.loads(
        inventory_json.read_text()
    )["objects"]

    inventory_by_id = {
        int(x["id"]): x
        for x in inventory
    }

    audit = json.loads(
        overlap_json.read_text()
    )

    risky = audit.get(
        "near_identical_pairs",
        []
    )

    pairs = []

    for index, row in enumerate(
        risky,
        start=1
    ):

        ia = int(
            row["id_a"]
        )

        ib = int(
            row["id_b"]
        )

        a = inventory_by_id[ia]
        b = inventory_by_id[ib]

        pairs.append({
            "pair_id": index,
            "id_a": ia,
            "id_b": ib,
            "name_a": a.get("name", ""),
            "phrase_a": a.get("grounding_phrase", ""),
            "name_b": b.get("name", ""),
            "phrase_b": b.get("grounding_phrase", ""),
            "mask_iou": row.get("iou"),
            "mask_containment": row.get("containment"),
        })

    print("=" * 90)
    print("TEST07 STAGE09.5")
    print("PHYSICAL INSTANCE COLLISION AUDIT")
    print("=" * 90)

    print(
        "HIGH-RISK PAIRS:",
        len(pairs)
    )

    lines = []

    for p in pairs:

        lines.append(
            f'PAIR {p["pair_id"]}\n'
            f'A [{p["id_a"]}]: '
            f'{p["name_a"]} | {p["phrase_a"]}\n'
            f'B [{p["id_b"]}]: '
            f'{p["name_b"]} | {p["phrase_b"]}\n'
        )

    user_text = (
        PROMPT
        +
        "\n\nPAIRS:\n"
        +
        "\n".join(lines)
    )

    dtype = (
        torch.float16
        if torch.cuda.is_available()
        else torch.float32
    )

    print()
    print("Loading Qwen 7B...")

    model = (
        Qwen2_5_VLForConditionalGeneration
        .from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            device_map="auto",
            cache_dir=str(model_cache)
        )
    )

    model.eval()

    processor = (
        AutoProcessor
        .from_pretrained(
            MODEL_ID,
            cache_dir=str(model_cache)
        )
    )

    print("✅ QWEN READY")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": str(master_path)
                },
                {
                    "type": "text",
                    "text": user_text
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
        output_ids[len(input_ids):]
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

    (
        output_dir
        / "00_raw.txt"
    ).write_text(
        raw,
        encoding="utf-8"
    )

    decisions = parse(
        raw
    )

    results = []

    collision_ids = set()
    duplicate_pairs = []

    for p in pairs:

        info = decisions.get(
            p["pair_id"],
            {
                "decision": "UNCERTAIN",
                "confidence": "low",
            }
        )

        result = dict(p)

        result.update(
            info
        )

        results.append(
            result
        )

        # Same/nearly same mask but descriptions represent
        # distinct objects => segmentation collision.
        #
        # UNCERTAIN is also treated conservatively as collision.
        if info["decision"] in {
            "DISTINCT_INSTANCES",
            "UNCERTAIN",
        }:

            collision_ids.add(
                p["id_a"]
            )

            collision_ids.add(
                p["id_b"]
            )

        elif (
            info["decision"]
            ==
            "SAME_INSTANCE"
        ):

            duplicate_pairs.append(
                [
                    p["id_a"],
                    p["id_b"]
                ]
            )

    result_path = (
        output_dir
        / "01_instance_collision_audit.json"
    )

    result_path.write_text(
        json.dumps(
            results,
            indent=2
        )
    )

    summary = {
        "stage":
            "TEST07_STAGE095",

        "high_risk_pairs":
            len(pairs),

        "collision_instance_ids":
            sorted(
                collision_ids
            ),

        "same_instance_duplicate_pairs":
            duplicate_pairs,

        "rule":
            (
                "near-identical mask + DISTINCT/UNCERTAIN "
                "means proposal cannot be trusted for those instances"
            ),
    }

    (
        output_dir
        / "stage095_report.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2
        )
    )

    print()
    print("=" * 90)
    print("TEST07 STAGE09.5 RESULT")
    print("=" * 90)

    for row in results:

        print(
            f'{row["pair_id"]:02d}. '
            f'{row["id_a"]:02d} {row["name_a"]}'
            f' ↔ '
            f'{row["id_b"]:02d} {row["name_b"]}'
            f' | {row["decision"]}'
            f' | {row["confidence"]}'
            f' | mask IoU={row["mask_iou"]}'
        )

    print()
    print(
        "COLLISION INSTANCE IDS:",
        sorted(
            collision_ids
        )
    )

    print(
        "SAME-INSTANCE DUPLICATE PAIRS:",
        duplicate_pairs
    )


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
        "--overlap-json",
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
        master_path=args.master,
        inventory_json=args.inventory_json,
        overlap_json=args.overlap_json,
        output_dir=args.output_dir,
        model_cache=args.model_cache,
    )
