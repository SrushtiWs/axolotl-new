
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
# TEST07 — STAGE02-V2
#
# TILED DYNAMIC INSTANCE INVENTORY
#
# PURPOSE
# -------
# Prevent long single-pass Qwen inventory generations from:
#
#   - truncating JSON
#   - entering repetitive loops
#   - generating combinatorial object variants
#
# GENERALIZATION RULES
# --------------------
# - no hardcoded room type
# - no fixed object vocabulary
# - no user-supplied object names
# - no Qwen coordinates
# - works on indoor / outdoor scenes
# - every visible distinct object may become an inventory item
#
# METHOD
# ------
# 1. Full-image overview pass
# 2. 3 x 3 overlapping crop passes
# 3. Compact line output
# 4. Deterministic textual duplicate suppression
# 5. Preserve crop provenance
# ============================================================


MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

MAX_OVERVIEW_OBJECTS = 30
MAX_CROP_OBJECTS = 14

OVERVIEW_TOKENS = 650
CROP_TOKENS = 380

OVERLAP = 0.12


# ============================================================
# PROMPTS
# ============================================================

OVERVIEW_PROMPT = """
Analyze the entire image.

Create an INSTANCE-LEVEL inventory of visible discrete physical
objects and fixtures.

IMPORTANT:

- Discover objects yourself.
- Do NOT assume a room type.
- Do NOT use a predefined object vocabulary.
- Do NOT inventory continuous wall, floor, ceiling, ground, sky,
  shadows, reflections, glare, paint or surface finish.
- Include furniture, fixtures, doors, windows, appliances,
  decorations, plants, equipment, loose objects and other
  separately visible physical entities.
- Every output line must correspond to ONE visually distinct
  physical instance.
- Never create hypothetical variants of the same object.
- Never repeat an object merely with different adjectives.
- Do not infer hidden objects.
- If multiple similar objects are clearly separate, list each
  separately.
- Maximum 30 object lines.
- Stop after the real visible objects are listed.

Return ONLY:

id || name || grounding_phrase || confidence

Example:

1 || pendant light || brass hanging pendant light || high
2 || window || narrow center window || high

No JSON.
No markdown.
No explanation.
No coordinates.
""".strip()


CROP_PROMPT = """
Analyze ONLY the visible content inside this image crop.

Find discrete physical objects and fixtures visible in this crop.

This crop is part of a larger arbitrary indoor or outdoor scene.

IMPORTANT:

- Discover objects yourself.
- Do NOT assume a room type.
- Do NOT use a predefined object vocabulary.
- Do NOT list wall, floor, ceiling, ground, sky, shadow,
  reflection, glare, paint or surface finish.
- Include small, thin, distant, mounted and partially visible
  physical objects.
- Every line must correspond to ONE separately visible physical
  instance.
- Never manufacture extra instances by changing:
  left/right/center,
  small/large,
  decorative/plain,
  mounted/shelf/counter,
  or other adjectives.
- Never repeat the same object line.
- If a physical object is only partly visible at the crop edge,
  include it only when its identity is visually clear.
- Do not infer hidden objects.
- Maximum 14 object lines.
- Stop when the real visible objects in this crop are listed.

Return ONLY:

id || name || grounding_phrase || confidence

No JSON.
No markdown.
No explanation.
No coordinates.
""".strip()


# ============================================================
# TEXT HELPERS
# ============================================================

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


def parse_compact(raw, max_objects):

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

    records = []

    seen_exact = set()

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        if "||" not in line:
            continue

        parts = [
            x.strip()
            for x in line.split("||")
        ]

        if len(parts) != 4:
            continue

        _, name, phrase, confidence = parts

        if not name:
            continue

        if not phrase:
            phrase = name

        confidence = confidence.lower()

        if confidence not in {
            "high",
            "medium",
            "low"
        }:
            confidence = "medium"

        exact_key = (
            normalize_text(name),
            normalize_text(phrase)
        )

        # Stop exact repetition loops immediately.
        if exact_key in seen_exact:
            continue

        seen_exact.add(
            exact_key
        )

        records.append({
            "name":
                name,

            "grounding_phrase":
                phrase,

            "visual_description":
                "",

            "confidence":
                confidence,
        })

        if len(records) >= max_objects:
            break

    return records


# ============================================================
# QWEN
# ============================================================

def run_qwen(
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

    del inputs
    del generated
    del trimmed

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return raw


# ============================================================
# TILING
# ============================================================

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

    margin_x = int(
        base_w * OVERLAP
    )

    margin_y = int(
        base_h * OVERLAP
    )

    tiles = []

    for label, col, row in labels:

        x1 = int(
            col * base_w
        )

        y1 = int(
            row * base_h
        )

        x2 = int(
            (col + 1) * base_w
        )

        y2 = int(
            (row + 1) * base_h
        )

        x1 = max(
            0,
            x1 - margin_x
        )

        y1 = max(
            0,
            y1 - margin_y
        )

        x2 = min(
            W,
            x2 + margin_x
        )

        y2 = min(
            H,
            y2 + margin_y
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

    # Diagnostic tile layout
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
                x1 + 4,
                y1 + 4
            ),
            tile["region"],
            fill="red"
        )

    diagnostic.save(
        output_dir
        / "00_tile_layout.png"
    )

    return tiles


# ============================================================
# DUPLICATE CONTROL
# ============================================================

def near_duplicate(
    a,
    b
):

    name_a = normalize_text(
        a["name"]
    )

    name_b = normalize_text(
        b["name"]
    )

    phrase_a = normalize_text(
        a["grounding_phrase"]
    )

    phrase_b = normalize_text(
        b["grounding_phrase"]
    )

    # Exact same name + phrase.
    if (
        name_a == name_b
        and
        phrase_a == phrase_b
    ):
        return True

    # Exact phrase match is strong duplicate evidence.
    if (
        phrase_a
        and
        phrase_a == phrase_b
    ):
        return True

    return False


def merge_records(
    overview_records,
    tiled_records
):

    final = []

    # Full overview first.
    for row in overview_records:

        item = dict(row)

        item[
            "source_region"
        ] = "overview"

        item[
            "source_bbox"
        ] = None

        final.append(
            item
        )

    # Add tiled discoveries unless exact textual duplicate.
    for row in tiled_records:

        duplicate = False

        for existing in final:

            if near_duplicate(
                row,
                existing
            ):
                duplicate = True
                break

        if duplicate:
            continue

        final.append(
            dict(row)
        )

    # Stable IDs.
    for i, row in enumerate(
        final,
        start=1
    ):

        row["id"] = i

    return final


# ============================================================
# MAIN
# ============================================================

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
    print("TEST07 STAGE02-V2")
    print("TILED DYNAMIC INSTANCE INVENTORY")
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
    print(
        "MODEL:",
        MODEL_ID
    )

    print(
        "Hardcoded object vocabulary: NO"
    )

    print(
        "Qwen coordinates: NO"
    )

    print(
        "Full overview + tiled discovery: YES"
    )

    # ========================================================
    # MODEL
    # ========================================================

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

    print("✅ QWEN 3B READY")

    # ========================================================
    # OVERVIEW
    # ========================================================

    print()
    print("PASS 00 — FULL IMAGE")

    overview_raw = run_qwen(
        model,
        processor,
        master_path,
        OVERVIEW_PROMPT,
        OVERVIEW_TOKENS
    )

    (
        output_dir
        / "01_overview_raw.txt"
    ).write_text(
        overview_raw,
        encoding="utf-8"
    )

    overview_records = (
        parse_compact(
            overview_raw,
            MAX_OVERVIEW_OBJECTS
        )
    )

    print(
        "Overview objects:",
        len(
            overview_records
        )
    )

    # ========================================================
    # TILES
    # ========================================================

    tiles = create_tiles(
        master,
        output_dir
    )

    tiled_records = []

    raw_dir = (
        output_dir
        / "tile_raw"
    )

    raw_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for index, tile in enumerate(
        tiles,
        start=1
    ):

        region = tile[
            "region"
        ]

        path = Path(
            tile["path"]
        )

        print()
        print(
            f"PASS {index:02d} —",
            region
        )

        raw = run_qwen(
            model,
            processor,
            path,
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

        records = (
            parse_compact(
                raw,
                MAX_CROP_OBJECTS
            )
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

            tiled_records.append(
                row
            )

    # ========================================================
    # MERGE
    # ========================================================

    final_objects = merge_records(
        overview_records,
        tiled_records
    )

    # ========================================================
    # SAVE
    # ========================================================

    inventory = {
        "method":
            "full-image + 3x3 overlapping tiled Qwen inventory",

        "model":
            MODEL_ID,

        "scene_summary":
            "",

        "objects":
            final_objects,

        "rules": [
            "no hardcoded object vocabulary",
            "no Qwen coordinates",
            "one physical instance per inventory record",
            "full image overview plus local tile discovery",
            "exact textual duplicate suppression only",
            "crop provenance retained"
        ]
    }

    inventory_path = (
        output_dir
        / "02_tiled_dynamic_inventory.json"
    )

    inventory_path.write_text(
        json.dumps(
            inventory,
            indent=2
        )
    )

    report_lines = []

    report_lines.append(
        "=" * 90
    )

    report_lines.append(
        "TEST07 STAGE02-V2 RESULT"
    )

    report_lines.append(
        "=" * 90
    )

    report_lines.append(
        f"OVERVIEW OBJECTS: {len(overview_records)}"
    )

    report_lines.append(
        f"TILED RAW OBJECTS: {len(tiled_records)}"
    )

    report_lines.append(
        f"FINAL INVENTORY: {len(final_objects)}"
    )

    report_lines.append("")

    report_lines.append(
        "OBJECT LIST:"
    )

    for row in final_objects:

        report_lines.append(
            f'{row["id"]:03d}. '
            f'{row["name"]} | '
            f'{row["confidence"]} | '
            f'{row["grounding_phrase"]} | '
            f'source={row["source_region"]}'
        )

    report_path = (
        output_dir
        / "03_inventory_report.txt"
    )

    report_path.write_text(
        "\n".join(
            report_lines
        )
    )

    report = {
        "stage":
            "TEST07_STAGE02_V2",

        "model":
            MODEL_ID,

        "master":
            str(
                master_path
            ),

        "overview_objects":
            len(
                overview_records
            ),

        "tiled_raw_objects":
            len(
                tiled_records
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
        / "stage02_v2_report.json"
    ).write_text(
        json.dumps(
            report,
            indent=2
        )
    )

    print()
    print("=" * 90)
    print("TEST07 STAGE02-V2 RESULT")
    print("=" * 90)

    print(
        "OVERVIEW OBJECTS:",
        len(
            overview_records
        )
    )

    print(
        "TILED RAW OBJECTS:",
        len(
            tiled_records
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
            f'source={row["source_region"]}'
        )


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
