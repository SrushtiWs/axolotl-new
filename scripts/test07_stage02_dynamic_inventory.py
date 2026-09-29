
from pathlib import Path
import argparse
import gc
import json
import re

import torch

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)

from qwen_vl_utils import process_vision_info


# ============================================================
# TEST07  STAGE 02
#
# GENERALIZED QWEN DYNAMIC PROP INVENTORY
#
# PURPOSE:
#   Discover all visible discrete physical entities from the
#   Stage01 resized master.
#
# IMPORTANT:
#   - no room type assumption
#   - no predefined object list
#   - no coordinates
#   - no bounding boxes
#   - no segmentation masks
#   - no Grounding DINO
#   - no SAM2
#   - no modification of Stage01 masks
#
# OUTPUT:
#   00_qwen_inventory_raw.txt
#   01_qwen_dynamic_inventory.json
#   02_inventory_report.txt
#   stage02_report.json
# ============================================================


MODEL_ID = (
    "Qwen/Qwen2.5-VL-7B-Instruct"
)


PROMPT = """
Analyze this scene carefully.

Your task is to create a comprehensive INSTANCE-LEVEL inventory
of every VISIBLE DISCRETE PHYSICAL ENTITY in the image.

The future application will replace architectural surface
finishes while preserving physical objects and fixtures.

Discover the entities yourself from the image.

IMPORTANT RULES:

1. Do NOT assume a room type.

2. Do NOT use a predefined object list.

3. The image could be any indoor or outdoor scene.

4. Include visible discrete physical entities regardless of size.

5. Pay special attention to SMALL, THIN, DISTANT,
   WALL-MOUNTED, CEILING-MOUNTED, partially occluded,
   and visually subtle entities.

6. Include permanently mounted physical fixtures when they are
   visually distinct physical objects.

7. Include visible doors, windows, frames, fixtures, furniture,
   equipment, decorations, appliances and loose objects, but
   discover what actually exists from the image.

8. Do NOT inventory continuous architectural/background
   surfaces themselves.

NOT entities:
continuous wall,
continuous floor,
continuous ceiling,
ground,
sky,
shadow,
reflection,
glare,
paint finish,
tile finish,
empty background.

9. Do NOT provide coordinates, bounding boxes, pixel locations,
   normalized locations, segmentation masks or measurements.

10. CRITICAL INSTANCE RULE:
    EVERY PHYSICALLY SEPARATE OBJECT MUST BE A SEPARATE
    INVENTORY ENTRY.

    Never create plural/group entries such as:
    "two lamps",
    "bedside tables",
    "wall pictures",
    "chairs",
    "plants",
    "shelves".

    Instead create separate entries such as:
    "left bedside lamp"
    "right bedside lamp"
    "left bedside table"
    "right bedside table"
    "upper-left framed artwork"
    "upper-center framed artwork"

11. If several similar objects are visible, count them visually
    and create one output line for EACH visible instance.

12. Use spatial or visual wording only in the grounding phrase
    to distinguish similar instances, such as:
    left, right, upper, lower, near-center, dark, white, tall,
    small, hanging, wall-mounted.

13. Do NOT merge multiple framed artworks into one "wall art"
    entry. Each separate frame is its own physical entity.

14. Do NOT merge matching furniture or fixtures simply because
    they form a pair or set.

15. Describe each object precisely enough that a later
    object-localization model can identify that ONE instance.

16. Do not invent objects that are not visibly supported.

17. CRITICAL ANTI-HALLUCINATION RULE:
    Create an inventory record ONLY when you can visually identify
    one distinct physical instance in the image.

18. NEVER generate hypothetical variants of an object by combining
    attributes such as:
    left / right / center,
    small / large,
    decorative / plain,
    shelf / counter / cabinet / wall-mounted.

    Spatial and visual adjectives may ONLY describe an object that
    is actually separately visible.

19. NEVER expand one visible object category into every possible
    location, size, style, or placement combination.

20. For repeated similar objects:
    first visually count the separately visible instances,
    then output exactly that many records.
    Do not exceed the visually supported count.

21. If you cannot distinguish two supposed instances as separate
    visible physical objects, DO NOT create two entries.

22. Do not infer hidden objects inside closed cabinets, behind other
    objects, or outside the visible image.

Before returning output, internally check:
- Did I combine any two physically separate visible objects?
  If yes, split them.
- Did I create any object merely by varying adjectives?
  If yes, delete it.
- Does every output line correspond to one visually distinguishable
  physical instance?
  If no, delete that line.

RETURN ONLY COMPACT PLAIN-TEXT RECORDS.

Use exactly ONE LINE per physical instance:

id || name || grounding_phrase || confidence

Example:

1 || ceiling fan || white ceiling fan || high
2 || bedside table || left bedside table || high
3 || bedside table || right bedside table || high

OUTPUT RULES:

- No JSON.
- No markdown.
- No explanation.
- No scene summary.
- No visual_description field.
- No blank commentary before or after the records.
- name should be short and singular.
- grounding_phrase should be concise but specific enough for
  a later object-localization model.
- confidence must be exactly high, medium, or low.

Be exhaustive but evidence-based.
""".strip()


# ============================================================
# HELPERS
# ============================================================

def extract_json_from_text(raw):

    import json
    import re

    if not isinstance(raw, str):
        raise TypeError(
            "Qwen response must be text."
        )

    clean = raw.strip()

    # --------------------------------------------------------
    # Remove markdown fences
    # --------------------------------------------------------

    clean = re.sub(
        r"^\s*```(?:json)?\s*",
        "",
        clean,
        flags=re.IGNORECASE
    )

    clean = re.sub(
        r"\s*```\s*$",
        "",
        clean
    )

    # --------------------------------------------------------
    # Keep only outer JSON object
    # --------------------------------------------------------

    start = clean.find("{")
    end = clean.rfind("}")

    if start < 0 or end < 0 or end <= start:
        raise RuntimeError(
            "Qwen response did not contain a JSON object."
        )

    clean = clean[
        start:end + 1
    ]

    # --------------------------------------------------------
    # ATTEMPT 1  exact JSON
    # --------------------------------------------------------

    try:
        return json.loads(clean)

    except json.JSONDecodeError:
        pass

    # --------------------------------------------------------
    # GENERALIZED SAFE REPAIRS
    #
    # These repairs alter JSON punctuation only.
    # They DO NOT invent inventory objects or semantic content.
    # --------------------------------------------------------

    repaired = clean

    # trailing comma before ] or }
    repaired = re.sub(
        r",\s*([}\]])",
        r"\1",
        repaired
    )

    # missing comma between adjacent object records:
    # }
    # {
    repaired = re.sub(
        r"}\s*{",
        "},{",
        repaired
    )

    # missing comma between completed object and next quoted
    # array/object field
    repaired = re.sub(
        r"}\s*(?=\")",
        "},",
        repaired
    )

    # missing comma after string before next JSON key
    repaired = re.sub(
        r'"\s*\n\s*"(?=[^"]+"\s*:)',
        '",\n"',
        repaired
    )

    # missing comma after numeric / boolean / null value before
    # next key on a new line
    repaired = re.sub(
        r'([0-9]|true|false|null)\s*\n\s*"(?=[^"]+"\s*:)',
        r'\1,\n"',
        repaired,
        flags=re.IGNORECASE
    )

    # --------------------------------------------------------
    # ATTEMPT 2  repaired JSON
    # --------------------------------------------------------

    try:
        return json.loads(
            repaired
        )

    except json.JSONDecodeError as e:

        # Save useful local context in the raised message.
        pos = int(
            getattr(
                e,
                "pos",
                0
            )
        )

        context = repaired[
            max(0, pos - 300):
            min(len(repaired), pos + 300)
        ]

        raise RuntimeError(
            "Qwen inventory JSON remained invalid after "
            "generalized punctuation repair.\n"
            f"JSON error: {e}\n\n"
            "Context around failure:\n"
            f"{context}"
        ) from e


def extract_compact_inventory(raw):

    import re

    if not isinstance(raw, str):
        raise TypeError(
            "Qwen response must be text."
        )

    text = raw.strip()

    # Remove accidental markdown fences.
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

    objects = []

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        # Expected:
        # id || name || grounding_phrase || confidence
        if "||" not in line:
            continue

        parts = [
            p.strip()
            for p in line.split("||")
        ]

        if len(parts) != 4:
            continue

        raw_id, name, grounding_phrase, confidence = parts

        match = re.search(
            r"\d+",
            raw_id
        )

        if match:
            obj_id = int(match.group())
        else:
            obj_id = len(objects) + 1

        name = name.strip()

        grounding_phrase = (
            grounding_phrase.strip()
            or name
        )

        confidence = (
            confidence
            .lower()
            .strip()
        )

        if confidence not in {
            "high",
            "medium",
            "low"
        }:
            confidence = "medium"

        if not name:
            continue

        objects.append({
            "id":
                obj_id,

            "name":
                name,

            "grounding_phrase":
                grounding_phrase,

            "visual_description":
                "",

            "confidence":
                confidence
        })

    if not objects:
        raise RuntimeError(
            "Qwen compact inventory response contained "
            "no parseable object records."
        )

    # Clean sequential IDs for downstream stages.
    normalized_objects = []

    for i, obj in enumerate(
        objects,
        start=1
    ):

        obj = dict(obj)

        obj["id"] = i

        normalized_objects.append(
            obj
        )

    return {
        "scene_summary":
            "",

        "objects":
            normalized_objects
    }


def normalize_inventory(
    inventory
):

    objects = inventory.get(
        "objects",
        []
    )

    if not isinstance(
        objects,
        list
    ):

        objects = []


    normalized = []


    for i, obj in enumerate(
        objects,
        start=1
    ):

        if not isinstance(
            obj,
            dict
        ):
            continue


        name = str(
            obj.get(
                "name",
                "unknown object"
            )
        ).strip()


        grounding_phrase = str(
            obj.get(
                "grounding_phrase",
                name
            )
        ).strip()


        description = str(
            obj.get(
                "visual_description",
                ""
            )
        ).strip()


        confidence = str(
            obj.get(
                "confidence",
                "medium"
            )
        ).lower().strip()


        if confidence not in {
            "high",
            "medium",
            "low"
        }:

            confidence = (
                "medium"
            )


        # ====================================================
        # GENERALIZED STRUCTURAL-SURFACE SAFETY FILTER
        #
        # Qwen may occasionally inventory a continuous surface
        # despite the prompt. Those must never enter Stage03.
        # ====================================================

        name_lower = (
            name.lower()
            .strip()
            .replace("-", " ")
            .replace("_", " ")
        )

        # ====================================================
        # STRUCTURAL-SURFACE FILTER
        #
        # Reject the continuous surface itself, but NEVER reject
        # a discrete object simply because its name contains
        # "wall", "floor", or "ceiling".
        #
        # Examples:
        #   yellow wall       -> reject
        #   tiled floor       -> reject
        #   white ceiling     -> reject
        #
        #   ceiling fan       -> keep
        #   wall mirror       -> keep
        #   wall shelf        -> keep
        #   floor lamp        -> keep
        # ====================================================

        tokens = name_lower.split()

        surface_heads = {
            "wall",
            "floor",
            "ceiling",
            "ground",
            "sky",
            "background"
        }

        surface_descriptors = {
            "white",
            "black",
            "gray",
            "grey",
            "yellow",
            "blue",
            "green",
            "red",
            "brown",
            "beige",
            "painted",
            "textured",
            "tiled",
            "tile",
            "wooden",
            "concrete",
            "plain",
            "continuous",
            "empty",
            "interior",
            "exterior"
        }

        surface_only = False

        # Exact structural surface.
        if name_lower in surface_heads:
            surface_only = True

        # "<descriptor> wall/floor/ceiling"
        elif (
            len(tokens) >= 2
            and
            tokens[-1] in surface_heads
            and
            all(
                token in surface_descriptors
                for token in tokens[:-1]
            )
        ):
            surface_only = True

        # Explicit "... surface"
        elif (
            tokens
            and
            tokens[-1] == "surface"
            and
            any(
                token in surface_heads
                for token in tokens[:-1]
            )
        ):
            surface_only = True

        if surface_only:
            continue


        normalized.append({
            "id":
                len(normalized) + 1,

            "name":
                name,

            "grounding_phrase":
                grounding_phrase,

            "visual_description":
                description,

            "confidence":
                confidence
        })


    inventory[
        "objects"
    ] = normalized


    if (
        "scene_summary"
        not in inventory
    ):

        inventory[
            "scene_summary"
        ] = ""


    return (
        inventory,
        normalized
    )


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


    print(
        "=" * 90
    )

    print(
        "TEST07 STAGE02"
    )

    print(
        "QWEN DYNAMIC PROP INVENTORY"
    )

    print(
        "=" * 90
    )


    print()
    print(
        "MASTER:"
    )

    print(
        master_path
    )


    print()
    print(
        "Hardcoded object vocabulary: NO"
    )

    print(
        "Coordinates requested: NO"
    )

    print(
        "Grounding DINO: NO"
    )

    print(
        "SAM2: NO"
    )


    # ========================================================
    # LOAD MODEL
    # ========================================================

    dtype = (
        torch.float16
        if torch.cuda.is_available()
        else torch.float32
    )


    print()
    print(
        "Loading Qwen2.5-VL-7B..."
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
        " QWEN READY"
    )


    # ========================================================
    # INPUT
    # ========================================================

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
                            master_path
                        )
                },
                {
                    "type":
                        "text",

                    "text":
                        PROMPT
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


    # ========================================================
    # GENERATE
    # ========================================================

    print()
    print(
        "Running scene inventory..."
    )


    with torch.inference_mode():

        generated_ids = (
            model.generate(
                **inputs,
                max_new_tokens=
                    1200,
                do_sample=
                    False,
                repetition_penalty=
                    1.08
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


    raw = (
        processor
        .batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        .strip()
    )


    # ========================================================
    # SAVE RAW
    # ========================================================

    raw_path = (
        output_dir
        / "00_qwen_inventory_raw.txt"
    )


    raw_path.write_text(
        raw,
        encoding=
            "utf-8"
    )


    # ========================================================
    # PARSE + NORMALIZE
    # ========================================================

    inventory = (
        extract_compact_inventory(
            raw
        )
    )


    inventory, objects = (
        normalize_inventory(
            inventory
        )
    )


    # ========================================================
    # SAVE INVENTORY
    # ========================================================

    json_path = (
        output_dir
        / "01_qwen_dynamic_inventory.json"
    )


    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            inventory,
            f,
            indent=2,
            ensure_ascii=False
        )


    # ========================================================
    # TEXT REPORT
    # ========================================================

    report_lines = []

    report_lines.append(
        "TEST07 STAGE02"
    )

    report_lines.append(
        "QWEN DYNAMIC PROP INVENTORY"
    )

    report_lines.append(
        "=" * 80
    )

    report_lines.append(
        f"MASTER: {master_path}"
    )

    report_lines.append(
        f"OBJECT COUNT: {len(objects)}"
    )

    report_lines.append(
        ""
    )


    for obj in objects:

        report_lines.append(
            f'{obj["id"]:02d}. '
            f'{obj["name"]}'
        )

        report_lines.append(
            "    Grounding: "
            + obj[
                "grounding_phrase"
            ]
        )

        report_lines.append(
            "    Confidence: "
            + obj[
                "confidence"
            ]
        )

        if obj[
            "visual_description"
        ]:

            report_lines.append(
                "    Description: "
                + obj[
                    "visual_description"
                ]
            )

        report_lines.append(
            ""
        )


    report_path = (
        output_dir
        / "02_inventory_report.txt"
    )


    report_path.write_text(
        "\n".join(
            report_lines
        ),
        encoding=
            "utf-8"
    )


    # ========================================================
    # JSON STAGE REPORT
    # ========================================================

    confidence_counts = {
        "high": 0,
        "medium": 0,
        "low": 0
    }


    for obj in objects:

        confidence_counts[
            obj[
                "confidence"
            ]
        ] += 1


    stage_report = {
        "experiment":
            "TEST07_STAGE02",

        "method":
            "Qwen2.5-VL dynamic discrete physical entity inventory",

        "model_id":
            MODEL_ID,

        "master":
            str(
                master_path
            ),

        "object_count":
            len(
                objects
            ),

        "confidence_counts":
            confidence_counts,

        "scene_summary":
            inventory.get(
                "scene_summary",
                ""
            ),

        "guarantees": [
            "no room type assumption",
            "no predefined object vocabulary",
            "no coordinates requested",
            "no bounding boxes requested",
            "no segmentation masks requested",
            "no Grounding DINO",
            "no SAM2",
            "Stage01 outputs not modified"
        ],

        "outputs": {
            "raw":
                str(
                    raw_path
                ),

            "inventory":
                str(
                    json_path
                ),

            "report":
                str(
                    report_path
                )
        }
    }


    stage_report_path = (
        output_dir
        / "stage02_report.json"
    )


    stage_report_path.write_text(
        json.dumps(
            stage_report,
            indent=2,
            ensure_ascii=False
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
        "TEST07 STAGE02 RESULT"
    )

    print(
        "=" * 90
    )


    print(
        "OBJECTS DISCOVERED:",
        len(
            objects
        )
    )


    print(
        "HIGH CONFIDENCE:",
        confidence_counts[
            "high"
        ]
    )

    print(
        "MEDIUM CONFIDENCE:",
        confidence_counts[
            "medium"
        ]
    )

    print(
        "LOW CONFIDENCE:",
        confidence_counts[
            "low"
        ]
    )


    print()
    print(
        "INVENTORY:"
    )

    print(
        json_path
    )


    print()
    print(
        "REPORT:"
    )

    print(
        report_path
    )


    print()
    print(
        "OBJECT LIST:"
    )


    for obj in objects:

        print(
            f'{obj["id"]:02d}.',
            obj[
                "name"
            ],
            "|",
            obj[
                "confidence"
            ],
            "|",
            obj[
                "grounding_phrase"
            ]
        )


    # ========================================================
    # CLEANUP
    # ========================================================

    del inputs
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
