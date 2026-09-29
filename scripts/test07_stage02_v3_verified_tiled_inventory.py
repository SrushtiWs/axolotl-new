
from pathlib import Path
import argparse
import json
import re

import torch
from PIL import Image, ImageDraw

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)

from qwen_vl_utils import process_vision_info


# ============================================================
# TEST07 — STAGE02-V3
#
# VERIFIED TILED INSTANCE INVENTORY
#
# METHOD
# ------
# 1. 3x3 overlapping visual crops
# 2. Qwen2.5-VL-7B short inventory per crop
# 3. Preserve source-region provenance
# 4. Consolidation pass over textual candidate inventory
# 5. Merge duplicate descriptions only when same physical
#    instance is clearly implied
#
# NO:
# - room-specific vocabulary
# - Qwen coordinates
# - hardcoded object names
# ============================================================


MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

CROP_TOKENS = 260
MAX_CROP_OBJECTS = 12

CONSOLIDATION_TOKENS = 900

OVERLAP = 0.18


CROP_PROMPT = """
Analyze ONLY this crop.

Create an INSTANCE-LEVEL inventory of every clearly visible
DISCRETE PHYSICAL OBJECT or fixture.

Rules:

- Discover objects yourself.
- Do NOT assume a room type.
- Do NOT use a predefined object vocabulary.
- Do NOT include continuous wall, floor, ceiling, ground, sky,
  paint, tile finish, shadow, reflection or glare.
- Include small, thin, mounted, hanging, partially visible and
  distant physical objects when visually supported.
- ONE line = ONE distinct visible physical instance.
- If three separate similar pots are visible, return three lines.
- Never create hypothetical variants by changing adjectives.
- Never infer hidden objects.
- Never repeat the same visible instance.
- Do not provide coordinates.
- Maximum 12 lines.
- Stop immediately after listing the real visible instances.

Return ONLY:

id || name || grounding_phrase || confidence

Example:
1 || pendant light || brass hanging pendant light || high
2 || plant || hanging leafy plant || high

No JSON.
No markdown.
No explanation.
""".strip()


CONSOLIDATION_PROMPT = """
You are consolidating object-inventory candidates collected from
overlapping image crops of ONE scene.

Each candidate describes a physical object discovered in one crop.

Your task:

1. Remove duplicate descriptions ONLY when they clearly refer to
   the SAME physical instance seen in overlapping crops.

2. Preserve separate objects even when they have the same class.
   Example:
   three different pots must remain three records.

3. Do NOT invent any new object.

4. Do NOT delete an object merely because its confidence is low.

5. Do NOT merge objects solely because their generic names match.

6. Use source-region information as supporting evidence:
   nearby overlapping regions may describe the same instance;
   widely separated regions are more likely different instances.

7. Prefer the clearest grounding phrase among duplicate records.

8. Return every surviving physical instance exactly once.

Return ONLY compact records:

id || name || grounding_phrase || confidence || source_regions

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


def parse_compact(
    raw,
    expected_parts,
    max_objects=None
):

    text = raw.strip()

    text = re.sub(
        r"^\s*```(?:text)?\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\s*```\s*$",
        "",
        text
    )

    rows = []
    seen = set()

    for line in text.splitlines():

        line = line.strip()

        if not line or "||" not in line:
            continue

        parts = [
            x.strip()
            for x in line.split("||")
        ]

        if len(parts) != expected_parts:
            continue

        if expected_parts == 4:

            _, name, phrase, confidence = parts

            source_regions = None

        else:

            _, name, phrase, confidence, source_regions = parts

        if not name:
            continue

        # Reject accidental format/header examples.
        name_norm = normalize_text(name)
        phrase_norm = normalize_text(phrase)

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
            normalize_text(name),
            normalize_text(phrase),
            normalize_text(source_regions or "")
        )

        if key in seen:
            continue

        seen.add(key)

        row = {
            "name":
                name,

            "grounding_phrase":
                phrase,

            "visual_description":
                "",

            "confidence":
                confidence,
        }

        if source_regions is not None:
            row[
                "source_regions"
            ] = source_regions

        rows.append(row)

        if (
            max_objects is not None
            and
            len(rows) >= max_objects
        ):
            break

    return rows


def run_qwen_image(
    model,
    processor,
    image_path,
    prompt,
    max_tokens
):

    messages = [
        {
            "role":
                "user",

            "content": [
                {
                    "type":
                        "image",

                    "image":
                        str(image_path)
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
        text=[text],
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
            max_new_tokens=max_tokens,
            do_sample=False,
            repetition_penalty=1.08
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

    del inputs
    del generated
    del trimmed

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return raw


def run_qwen_text(
    model,
    processor,
    prompt,
    max_tokens
):

    messages = [
        {
            "role":
                "user",

            "content": [
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

    inputs = processor(
        text=[text],
        padding=True,
        return_tensors="pt"
    )

    inputs = inputs.to(
        model.device
    )

    with torch.inference_mode():

        generated = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            repetition_penalty=1.08
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

    del inputs
    del generated
    del trimmed

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return raw


def create_tiles(
    image,
    output_dir
):

    W, H = image.size

    tile_dir = (
        output_dir
        / "tiles"
    )

    tile_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    labels = [
        ("top_left",       0, 0),
        ("top_center",     1, 0),
        ("top_right",      2, 0),

        ("middle_left",    0, 1),
        ("middle_center",  1, 1),
        ("middle_right",   2, 1),

        ("bottom_left",    0, 2),
        ("bottom_center",  1, 2),
        ("bottom_right",   2, 2),
    ]

    base_w = W / 3.0
    base_h = H / 3.0

    mx = int(
        base_w * OVERLAP
    )

    my = int(
        base_h * OVERLAP
    )

    tiles = []

    for label, col, row in labels:

        x1 = max(
            0,
            int(col * base_w) - mx
        )

        y1 = max(
            0,
            int(row * base_h) - my
        )

        x2 = min(
            W,
            int((col + 1) * base_w) + mx
        )

        y2 = min(
            H,
            int((row + 1) * base_h) + my
        )

        crop = image.crop(
            (
                x1,
                y1,
                x2,
                y2
            )
        )

        path = (
            tile_dir
            / f"{label}.png"
        )

        crop.save(path)

        tiles.append({
            "region":
                label,

            "bbox":
                [x1, y1, x2, y2],

            "path":
                str(path)
        })

    diagnostic = image.copy()

    draw = ImageDraw.Draw(
        diagnostic
    )

    for tile in tiles:

        x1, y1, x2, y2 = (
            tile["bbox"]
        )

        draw.rectangle(
            [x1, y1, x2, y2],
            outline="red",
            width=2
        )

        draw.text(
            (
                x1 + 3,
                y1 + 3
            ),
            tile["region"],
            fill="red"
        )

    diagnostic.save(
        output_dir
        / "00_tile_layout.png"
    )

    return tiles


def build_consolidation_input(
    candidates
):

    lines = []

    for i, row in enumerate(
        candidates,
        start=1
    ):

        lines.append(
            f'{i} || '
            f'{row["name"]} || '
            f'{row["grounding_phrase"]} || '
            f'{row["confidence"]} || '
            f'{row["source_region"]}'
        )

    return "\n".join(
        lines
    )


def run(
    master_path,
    output_dir,
    model_cache
):

    master_path = Path(
        master_path
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

    print("=" * 90)
    print("TEST07 STAGE02-V3")
    print("VERIFIED TILED INSTANCE INVENTORY")
    print("=" * 90)

    print()
    print("MASTER:")
    print(master_path)

    master = (
        Image.open(
            master_path
        )
        .convert("RGB")
    )

    print(
        "SIZE:",
        master.size
    )

    print()
    print("MODEL:", MODEL_ID)
    print("Hardcoded vocabulary: NO")
    print("Qwen coordinates: NO")
    print("Tiled discovery: YES")
    print("Consolidation audit: YES")

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

    print("✅ QWEN 7B READY")

    tiles = create_tiles(
        master,
        output_dir
    )

    raw_dir = (
        output_dir
        / "tile_raw"
    )

    raw_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    all_candidates = []

    for index, tile in enumerate(
        tiles,
        start=1
    ):

        region = tile[
            "region"
        ]

        print()
        print(
            f"PASS {index:02d} —",
            region
        )

        raw = run_qwen_image(
            model,
            processor,
            Path(
                tile["path"]
            ),
            CROP_PROMPT,
            CROP_TOKENS
        )

        (
            raw_dir
            / f"{region}.txt"
        ).write_text(
            raw,
            encoding="utf-8"
        )

        records = parse_compact(
            raw,
            expected_parts=4,
            max_objects=
                MAX_CROP_OBJECTS
        )

        print(
            "Objects:",
            len(records)
        )

        for row in records:

            row[
                "source_region"
            ] = region

            row[
                "source_bbox"
            ] = tile[
                "bbox"
            ]

            all_candidates.append(
                row
            )

    candidate_path = (
        output_dir
        / "01_raw_tiled_candidates.json"
    )

    candidate_path.write_text(
        json.dumps(
            all_candidates,
            indent=2
        )
    )

    print()
    print(
        "RAW TILED CANDIDATES:",
        len(
            all_candidates
        )
    )

    # ========================================================
    # CONSOLIDATION
    # ========================================================

    candidate_text = (
        build_consolidation_input(
            all_candidates
        )
    )

    consolidation_input = (
        CONSOLIDATION_PROMPT
        +
        "\n\nCANDIDATES:\n"
        +
        candidate_text
    )

    (
        output_dir
        / "02_consolidation_input.txt"
    ).write_text(
        consolidation_input,
        encoding="utf-8"
    )

    print()
    print(
        "Running consolidation audit..."
    )

    consolidated_raw = run_qwen_text(
        model,
        processor,
        consolidation_input,
        CONSOLIDATION_TOKENS
    )

    (
        output_dir
        / "03_consolidation_raw.txt"
    ).write_text(
        consolidated_raw,
        encoding="utf-8"
    )

    final_objects = parse_compact(
        consolidated_raw,
        expected_parts=5,
        max_objects=None
    )

    for i, row in enumerate(
        final_objects,
        start=1
    ):

        row["id"] = i

    inventory = {
        "method":
            "3x3 overlapping Qwen-7B tiled discovery + Qwen consolidation",

        "model":
            MODEL_ID,

        "scene_summary":
            "",

        "objects":
            final_objects,

        "raw_candidate_count":
            len(
                all_candidates
            ),

        "final_inventory_count":
            len(
                final_objects
            ),
    }

    inventory_path = (
        output_dir
        / "04_verified_dynamic_inventory.json"
    )

    inventory_path.write_text(
        json.dumps(
            inventory,
            indent=2
        )
    )

    report_lines = [
        "=" * 90,
        "TEST07 STAGE02-V3 RESULT",
        "=" * 90,
        f"RAW TILED CANDIDATES: {len(all_candidates)}",
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
            f'sources={row.get("source_regions", "")}'
        )

    report_path = (
        output_dir
        / "05_inventory_report.txt"
    )

    report_path.write_text(
        "\n".join(
            report_lines
        )
    )

    report = {
        "stage":
            "TEST07_STAGE02_V3",

        "master":
            str(
                master_path
            ),

        "model":
            MODEL_ID,

        "raw_tiled_candidates":
            len(
                all_candidates
            ),

        "final_inventory":
            len(
                final_objects
            ),

        "inventory":
            str(
                inventory_path
            ),

        "report":
            str(
                report_path
            ),
    }

    (
        output_dir
        / "stage02_v3_report.json"
    ).write_text(
        json.dumps(
            report,
            indent=2
        )
    )

    print()
    print("=" * 90)
    print("TEST07 STAGE02-V3 RESULT")
    print("=" * 90)

    print(
        "RAW TILED CANDIDATES:",
        len(
            all_candidates
        )
    )

    print(
        "FINAL INVENTORY:",
        len(
            final_objects
        )
    )

    print()
    print("INVENTORY:")
    print(inventory_path)

    print()
    print("REPORT:")
    print(report_path)

    print()
    print("OBJECT LIST:")

    for row in final_objects:

        print(
            f'{row["id"]:03d}. '
            f'{row["name"]} | '
            f'{row["confidence"]} | '
            f'{row["grounding_phrase"]} | '
            f'sources={row.get("source_regions", "")}'
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--master",
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

        output_dir=
            args.output_dir,

        model_cache=
            args.model_cache
    )
