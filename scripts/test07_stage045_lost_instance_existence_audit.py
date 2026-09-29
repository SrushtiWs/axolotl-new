
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

BATCH_SIZE = 8
MAX_NEW_TOKENS = 300


PROMPT = """
You are auditing unresolved object-inventory entries for ONE real scene.

You are shown the FULL ORIGINAL IMAGE.

For each numbered inventory item, decide whether that physical
object is visibly supported by the image.

Decisions:

PRESENT
The described physical object is visibly present.

ABSENT
The described object is not visibly supported by this image.

UNCERTAIN
There is insufficient visual evidence to confidently decide.

IMPORTANT RULES:

1. Judge existence only.
2. Do NOT provide coordinates.
3. Do NOT invent missing objects.
4. Do NOT assume objects hidden inside cabinets or behind objects.
5. Similar nearby objects do NOT prove that the requested object exists.
6. Example:
   a brass vessel does not prove there is a trash can.
7. A candidate can be PRESENT even if its previous detector box was wrong.
8. Use the actual full scene image as primary evidence.
9. Be conservative. Use UNCERTAIN when needed.

Return ONLY:

item_id || PRESENT_or_ABSENT_or_UNCERTAIN || confidence

confidence:
high
medium
low

Example:

30 || PRESENT || high
61 || ABSENT || high

No JSON.
No markdown.
No explanation.
""".strip()


def parse_decisions(raw):

    result = {}

    for line in raw.splitlines():

        line = line.strip()

        if not line or "||" not in line:
            continue

        parts = [
            x.strip()
            for x in line.split("||")
        ]

        if len(parts) != 3:
            continue

        raw_id, decision, confidence = parts

        match = re.search(
            r"\d+",
            raw_id
        )

        if not match:
            continue

        iid = int(match.group())

        decision = decision.upper()

        if decision not in {
            "PRESENT",
            "ABSENT",
            "UNCERTAIN"
        }:
            continue

        confidence = confidence.lower()

        if confidence not in {
            "high",
            "medium",
            "low"
        }:
            confidence = "medium"

        result[iid] = {
            "decision": decision,
            "confidence": confidence
        }

    return result


def run_batch(
    model,
    processor,
    master_path,
    batch
):

    lines = []

    for obj in batch:

        lines.append(
            f'{int(obj["id"])} || '
            f'{obj.get("name","")} || '
            f'{obj.get("grounding_phrase","")}'
        )

    user_text = (
        PROMPT
        +
        "\n\nINVENTORY ITEMS:\n"
        +
        "\n".join(lines)
    )

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
        process_vision_info(messages)
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
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=1.05
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

    return raw


def run(
    master_path,
    inventory_json,
    verified_json,
    output_dir,
    model_cache
):

    master_path = Path(master_path)
    inventory_json = Path(inventory_json)
    verified_json = Path(verified_json)
    output_dir = Path(output_dir)
    model_cache = Path(model_cache)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    inventory = json.loads(
        inventory_json.read_text()
    )["objects"]

    verified = json.loads(
        verified_json.read_text()
    )

    valid_ids = {
        int(row["inventory_id"])
        for row in verified
    }

    lost = [
        obj
        for obj in inventory
        if int(obj["id"]) not in valid_ids
    ]

    print("=" * 90)
    print("TEST07 STAGE04.5")
    print("LOST INSTANCE EXISTENCE AUDIT")
    print("=" * 90)

    print(
        "INVENTORY:",
        len(inventory)
    )

    print(
        "ALREADY VALID:",
        len(valid_ids)
    )

    print(
        "LOST TO AUDIT:",
        len(lost)
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

    all_decisions = {}

    total_batches = (
        len(lost)
        +
        BATCH_SIZE
        -
        1
    ) // BATCH_SIZE

    for batch_index in range(total_batches):

        start = batch_index * BATCH_SIZE

        end = min(
            len(lost),
            start + BATCH_SIZE
        )

        batch = lost[start:end]

        print()
        print(
            f"BATCH {batch_index+1}/{total_batches}"
        )

        raw = run_batch(
            model,
            processor,
            master_path,
            batch
        )

        (
            output_dir
            / f"batch_{batch_index+1:02d}_raw.txt"
        ).write_text(
            raw,
            encoding="utf-8"
        )

        decisions = parse_decisions(raw)

        print(
            "Parsed:",
            len(decisions),
            "/",
            len(batch)
        )

        all_decisions.update(
            decisions
        )

    results = []

    for obj in lost:

        iid = int(obj["id"])

        info = all_decisions.get(
            iid,
            {
                "decision": "UNCERTAIN",
                "confidence": "low"
            }
        )

        results.append({
            "inventory_id": iid,
            "name": obj.get("name", ""),
            "grounding_phrase":
                obj.get(
                    "grounding_phrase",
                    ""
                ),
            "decision":
                info["decision"],
            "confidence":
                info["confidence"]
        })

    result_path = (
        output_dir
        / "00_existence_audit.json"
    )

    result_path.write_text(
        json.dumps(
            results,
            indent=2
        )
    )

    counts = {
        "PRESENT": 0,
        "ABSENT": 0,
        "UNCERTAIN": 0,
    }

    for row in results:
        counts[
            row["decision"]
        ] += 1

    print()
    print("=" * 90)
    print("TEST07 STAGE04.5 RESULT")
    print("=" * 90)

    print("PRESENT:", counts["PRESENT"])
    print("ABSENT:", counts["ABSENT"])
    print("UNCERTAIN:", counts["UNCERTAIN"])

    print()
    print("INSTANCE LIST:")

    for row in results:

        print(
            f'{row["inventory_id"]:02d}. '
            f'{row["name"]} | '
            f'{row["decision"]} | '
            f'{row["confidence"]} | '
            f'{row["grounding_phrase"]}'
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
        "--verified-json",
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
        verified_json=args.verified_json,
        output_dir=args.output_dir,
        model_cache=args.model_cache
    )
