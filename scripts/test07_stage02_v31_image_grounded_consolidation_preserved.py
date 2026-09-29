
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

MAX_NEW_TOKENS = 850


PROMPT = """
You are performing the FINAL consolidation of an object inventory
for one real scene.

You are given:

1. The FULL ORIGINAL IMAGE.
2. Candidate object records discovered independently from
   overlapping image crops.

Your job is NOT to discover a new inventory from scratch.

Your job is ONLY to determine which candidate records correspond
to the SAME physical instance and consolidate those duplicates.

CRITICAL RULES:

1. Every final record must correspond to one visibly supported
   physical instance in the supplied image.

2. NEVER invent a new object.

3. NEVER delete a real visible object merely because another
   object has the same generic class name.

4. Two records with the same class can represent different
   physical objects.

5. Merge records ONLY when the full image supports that they
   describe the SAME physical instance observed in overlapping
   crops.

6. Source regions are evidence:
   nearby overlapping regions may see the same object.

7. Example:
   "brass hanging pendant light" from top_center and
   "brass hanging pendant light" from top_right may be the same
   physical lamp if the full image shows only one such lamp.

8. Conversely, several separately visible pots must remain
   separate records.

9. Prefer the clearest grounding phrase among merged candidates.

10. Do not inventory continuous architectural/background
    surfaces such as wall, floor, ceiling, paint, shadow,
    reflection or glare.

11. Remove format examples, headings, placeholders and malformed
    candidate records.

12. Keep confidence as high, medium or low.

Return ONLY:

id || name || grounding_phrase || confidence || source_regions

One line per final physical instance.

No JSON.
No markdown.
No explanation.
""".strip()


def normalize_text(text):

    text = str(text).lower().strip()

    text = re.sub(
        r"[^a-z0-9 ]+",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def parse_output(raw):

    rows = []

    seen = set()

    for line in raw.splitlines():

        line = line.strip()

        if not line or "||" not in line:
            continue

        parts = [
            x.strip()
            for x in line.split("||")
        ]

        if len(parts) != 5:
            continue

        _, name, phrase, confidence, regions = parts

        name_norm = normalize_text(name)
        phrase_norm = normalize_text(phrase)

        # Remove prompt/header artifacts.
        if name_norm in {
            "name",
            "object",
            "object name"
        }:
            continue

        if phrase_norm in {
            "grounding phrase",
            "grounding_phrase"
        }:
            continue

        if not name:
            continue

        if not phrase:
            phrase = name

        confidence = confidence.lower().strip()

        if confidence not in {
            "high",
            "medium",
            "low"
        }:
            confidence = "medium"

        key = (
            name_norm,
            phrase_norm,
            normalize_text(regions)
        )

        if key in seen:
            continue

        seen.add(key)

        rows.append({
            "name": name,
            "grounding_phrase": phrase,
            "visual_description": "",
            "confidence": confidence,
            "source_regions": regions
        })

    for i, row in enumerate(
        rows,
        start=1
    ):
        row["id"] = i

    return rows


def run(
    master_path,
    candidate_json,
    output_dir,
    model_cache
):

    master_path = Path(master_path)
    candidate_json = Path(candidate_json)
    output_dir = Path(output_dir)
    model_cache = Path(model_cache)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    model_cache.mkdir(
        parents=True,
        exist_ok=True
    )

    candidates = json.loads(
        candidate_json.read_text()
    )

    print("=" * 90)
    print("TEST07 STAGE02-V3.1")
    print("IMAGE-GROUNDED INVENTORY CONSOLIDATION")
    print("=" * 90)

    print("INPUT CANDIDATES:", len(candidates))
    print("MASTER:", master_path)

    candidate_lines = []

    for i, row in enumerate(
        candidates,
        start=1
    ):

        candidate_lines.append(
            f'{i} || '
            f'{row.get("name", "")} || '
            f'{row.get("grounding_phrase", "")} || '
            f'{row.get("confidence", "medium")} || '
            f'{row.get("source_region", "")}'
        )

    candidate_text = "\n".join(
        candidate_lines
    )

    full_prompt = (
        PROMPT
        + "\n\n"
        + "CANDIDATE RECORDS:\n"
        + candidate_text
    )

    (
        output_dir
        / "00_consolidation_prompt.txt"
    ).write_text(
        full_prompt,
        encoding="utf-8"
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
                    "text": full_prompt
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

    print()
    print("Running image-grounded consolidation...")

    with torch.inference_mode():

        generated = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=1.08
        )

    trimmed = [
        out[len(inp):]
        for inp, out
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
        / "01_consolidation_raw.txt"
    ).write_text(
        raw,
        encoding="utf-8"
    )

    final_objects = parse_output(
        raw
    )

    inventory = {
        "method":
            "Qwen 7B image-grounded consolidation of tiled candidates",

        "input_candidate_count":
            len(candidates),

        "final_inventory_count":
            len(final_objects),

        "objects":
            final_objects
    }

    inventory_path = (
        output_dir
        / "02_verified_inventory.json"
    )

    inventory_path.write_text(
        json.dumps(
            inventory,
            indent=2
        )
    )

    report_lines = [
        "=" * 90,
        "TEST07 STAGE02-V3.1 RESULT",
        "=" * 90,
        f"INPUT CANDIDATES: {len(candidates)}",
        f"FINAL INVENTORY: {len(final_objects)}",
        "",
        "OBJECT LIST:"
    ]

    for row in final_objects:

        report_lines.append(
            f'{row["id"]:03d}. '
            f'{row["name"]} | '
            f'{row["confidence"]} | '
            f'{row["grounding_phrase"]} | '
            f'sources={row["source_regions"]}'
        )

    report_path = (
        output_dir
        / "03_inventory_report.txt"
    )

    report_path.write_text(
        "\n".join(report_lines)
    )

    print()
    print("=" * 90)
    print("TEST07 STAGE02-V3.1 RESULT")
    print("=" * 90)

    print(
        "INPUT CANDIDATES:",
        len(candidates)
    )

    print(
        "FINAL INVENTORY:",
        len(final_objects)
    )

    print()
    print("OBJECT LIST:")

    for row in final_objects:

        print(
            f'{row["id"]:03d}. '
            f'{row["name"]} | '
            f'{row["confidence"]} | '
            f'{row["grounding_phrase"]} | '
            f'sources={row["source_regions"]}'
        )

    print()
    print("INVENTORY:")
    print(inventory_path)

    print()
    print("REPORT:")
    print(report_path)


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
        default="/workspace/data/huggingface-cache"
    )

    args = parser.parse_args()

    run(
        master_path=args.master,
        candidate_json=args.candidate_json,
        output_dir=args.output_dir,
        model_cache=args.model_cache
    )
