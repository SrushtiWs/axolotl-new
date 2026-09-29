
from pathlib import Path
from PIL import Image, ImageDraw
from IPython.display import display

import gc
import json
import re
import time

import torch

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)

from qwen_vl_utils import (
    process_vision_info
)


# ============================================================
# 1. IDENTIFIERS / SETTINGS
# ============================================================

STAGE_NAME = (
    "PRODUCTION_STAGE06A_PROP_INVENTORY"
)

MODEL_ID = (
    "Qwen/Qwen2.5-VL-7B-Instruct"
)

CROP_TOKENS = 260

MAX_CROP_OBJECTS = 12

CONSOLIDATION_TOKENS = 900

OVERLAP = 0.18


# ============================================================
# 2. PATHS
# ============================================================

BASE = Path(
    "/workspace/axolotl"
)

PROJECT = (
    BASE
    / "test07"
)

PROD = (
    PROJECT
    / "production_pipeline"
)

STAGE05 = (
    PROD
    / "stage05_clean_room_with_props"
)

STAGE06 = (
    PROD
    / "stage06_prop_layer"
)

OUT = (
    STAGE06
    / "06a_inventory"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


INPUT_PATH = (
    STAGE05
    / "07_clean_room_with_exact_props.png"
)


MODEL_CACHE = Path(
    "/workspace/data/huggingface-cache"
)

MODEL_CACHE.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 3. PROMPTS
# ============================================================

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
- Do NOT include mirror or shower glass partition.
- Include small, thin, mounted, hanging, partially visible and
  distant physical objects when visually supported.
- Include sanitary fixtures, furniture, hardware, switches,
  outlets, lights, handles and other genuine physical objects.
- ONE line = ONE distinct visible physical instance.
- If multiple separate similar objects are visible, return
  separate lines.
- Never create hypothetical variants by changing adjectives.
- Never infer hidden objects.
- Never repeat the same visible instance within this crop.
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
You are consolidating object-inventory candidates collected
from overlapping image crops of ONE room scene.

Each candidate describes a physical object discovered in one
crop.

Your task:

1. Remove duplicate descriptions ONLY when they clearly refer
   to the SAME physical instance seen in overlapping crops.

2. Preserve separate objects even when they have the same
   class.

3. Do NOT invent any new object.

4. Do NOT delete an object merely because its confidence is
   low.

5. Do NOT merge objects solely because their generic names
   match.

6. Use source-region information as supporting evidence:
   nearby overlapping regions may describe the same instance;
   widely separated regions are more likely different
   instances.

7. Prefer the clearest grounding phrase among duplicate
   records.

8. Return every surviving physical instance exactly once.

9. Exclude architectural surfaces such as wall, floor and
   ceiling.

10. Exclude mirror and shower glass partition because they
    are separate special layers in this production pipeline.

Return ONLY compact records:

id || name || grounding_phrase || confidence || source_regions

No JSON.
No markdown.
No explanation.
""".strip()


# ============================================================
# 4. HELPERS
# ============================================================

def normalize_text(text):

    text = str(
        text
    ).lower().strip()

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

    rows = []

    for line in raw.splitlines():

        line = line.strip()

        if not line:
            continue

        if "||" not in line:
            continue

        parts = [
            p.strip()
            for p in line.split("||")
        ]

        if len(parts) < expected_parts:
            continue


        if expected_parts == 4:

            try:
                local_id = int(
                    re.sub(
                        r"[^0-9]",
                        "",
                        parts[0]
                    )
                )

            except Exception:
                local_id = (
                    len(rows) + 1
                )

            name = parts[1].strip()

            grounding_phrase = (
                parts[2].strip()
            )

            confidence = (
                parts[3].strip().lower()
            )

            if not name:
                continue

            if not grounding_phrase:
                grounding_phrase = name


            rows.append({
                "local_id":
                    local_id,

                "name":
                    name,

                "grounding_phrase":
                    grounding_phrase,

                "confidence":
                    confidence,
            })


        elif expected_parts == 5:

            name = (
                parts[1].strip()
            )

            grounding_phrase = (
                parts[2].strip()
            )

            confidence = (
                parts[3].strip().lower()
            )

            source_regions_raw = (
                parts[4].strip()
            )


            if not name:
                continue

            if not grounding_phrase:
                grounding_phrase = name


            source_regions = [
                x.strip()
                for x in re.split(
                    r"[,;/]+",
                    source_regions_raw
                )
                if x.strip()
            ]


            rows.append({
                "name":
                    name,

                "grounding_phrase":
                    grounding_phrase,

                "confidence":
                    confidence,

                "source_regions":
                    source_regions,
            })


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
                        str(image_path),
                },
                {
                    "type":
                        "text",

                    "text":
                        prompt,
                },
            ],
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
                        prompt,
                },
            ],
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

        (
            "top_left",
            0,
            0
        ),

        (
            "top_center",
            1,
            0
        ),

        (
            "top_right",
            2,
            0
        ),

        (
            "middle_left",
            0,
            1
        ),

        (
            "middle_center",
            1,
            1
        ),

        (
            "middle_right",
            2,
            1
        ),

        (
            "bottom_left",
            0,
            2
        ),

        (
            "bottom_center",
            1,
            2
        ),

        (
            "bottom_right",
            2,
            2
        ),
    ]


    base_w = (
        W / 3.0
    )

    base_h = (
        H / 3.0
    )


    mx = int(
        base_w
        * OVERLAP
    )

    my = int(
        base_h
        * OVERLAP
    )


    tiles = []


    for label, col, row in labels:

        x1 = max(
            0,
            int(
                col * base_w
            ) - mx
        )

        y1 = max(
            0,
            int(
                row * base_h
            ) - my
        )

        x2 = min(
            W,
            int(
                (col + 1)
                * base_w
            ) + mx
        )

        y2 = min(
            H,
            int(
                (row + 1)
                * base_h
            ) + my
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


        crop.save(
            path
        )


        tiles.append({

            "region":
                label,

            "bbox":
                [
                    x1,
                    y1,
                    x2,
                    y2
                ],

            "path":
                str(path),
        })


    diagnostic = (
        image.copy()
    )


    draw = ImageDraw.Draw(
        diagnostic
    )


    for tile in tiles:

        x1, y1, x2, y2 = (
            tile["bbox"]
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


# ============================================================
# 5. START
# ============================================================

print()
print("=" * 100)
print(STAGE_NAME)
print("=" * 100)


if not INPUT_PATH.exists():

    raise FileNotFoundError(
        INPUT_PATH
    )


master = (
    Image.open(
        INPUT_PATH
    )
    .convert("RGB")
)


W, H = master.size


print()
print(
    "DISCOVERY INPUT:",
    INPUT_PATH
)

print(
    "SIZE:",
    W,
    "x",
    H
)

print(
    "MODEL:",
    MODEL_ID
)

print(
    "HARDCODED VOCABULARY: NO"
)

print(
    "QWEN COORDINATES: NO"
)

print(
    "3x3 TILED DISCOVERY: YES"
)

print(
    "OVERLAP:",
    OVERLAP
)


# ============================================================
# 6. CLEAN MEMORY
# ============================================================

gc.collect()


if torch.cuda.is_available():

    torch.cuda.empty_cache()


# ============================================================
# 7. LOAD MODEL
# ============================================================

dtype = (
    torch.float16
    if torch.cuda.is_available()
    else torch.float32
)


print()
print("=" * 100)
print("LOADING QWEN2.5-VL-7B")
print("=" * 100)


load_start = (
    time.time()
)


model = (
    Qwen2_5_VLForConditionalGeneration
    .from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto",
        cache_dir=str(
            MODEL_CACHE
        ),
    )
)


model.eval()


processor = (
    AutoProcessor
    .from_pretrained(
        MODEL_ID,
        cache_dir=str(
            MODEL_CACHE
        ),
    )
)


load_seconds = (
    time.time()
    -
    load_start
)


print()
print(
    "✅ QWEN 7B READY"
)

print(
    "LOAD SEC:",
    round(
        load_seconds,
        2
    )
)


# ============================================================
# 8. CREATE 3x3 CROPS
# ============================================================

tiles = create_tiles(
    master,
    OUT
)


# ============================================================
# 9. CROP INVENTORY
# ============================================================

raw_dir = (
    OUT
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

    region = (
        tile["region"]
    )


    print()
    print(
        f"PASS {index:02d}/09 -",
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
        "OBJECTS:",
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


# ============================================================
# 10. SAVE RAW CANDIDATES
# ============================================================

candidate_path = (
    OUT
    / "01_raw_tiled_candidates.json"
)


candidate_path.write_text(
    json.dumps(
        all_candidates,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


print()
print(
    "RAW TILED CANDIDATES:",
    len(
        all_candidates
    )
)


# ============================================================
# 11. CONSOLIDATION
# ============================================================

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


consolidation_input_path = (
    OUT
    / "02_consolidation_input.txt"
)


consolidation_input_path.write_text(
    consolidation_input,
    encoding="utf-8"
)


print()
print("=" * 100)
print("RUNNING CONSOLIDATION")
print("=" * 100)


consolidated_raw = run_qwen_text(
    model,
    processor,
    consolidation_input,
    CONSOLIDATION_TOKENS
)


consolidated_raw_path = (
    OUT
    / "03_consolidation_raw.txt"
)


consolidated_raw_path.write_text(
    consolidated_raw,
    encoding="utf-8"
)


final_objects = parse_compact(
    consolidated_raw,
    expected_parts=5,
    max_objects=None
)


# ============================================================
# 12. RECOVER SOURCE BBOXES
# ============================================================

for i, row in enumerate(
    final_objects,
    start=1
):

    row["id"] = i


    regions = set(
        row.get(
            "source_regions",
            []
        )
    )


    matching = [
        c
        for c
        in all_candidates

        if (
            c["source_region"]
            in regions
        )
        and
        (
            normalize_text(
                c["name"]
            )
            ==
            normalize_text(
                row["name"]
            )
            or
            normalize_text(
                c["grounding_phrase"]
            )
            ==
            normalize_text(
                row["grounding_phrase"]
            )
        )
    ]


    if not matching:

        matching = [
            c
            for c
            in all_candidates

            if c["source_region"]
            in regions
        ]


    source_bboxes = []


    for c in matching:

        bbox = (
            c["source_bbox"]
        )

        if bbox not in source_bboxes:

            source_bboxes.append(
                bbox
            )


    row[
        "source_bboxes"
    ] = source_bboxes


# ============================================================
# 13. FINAL INVENTORY
# ============================================================

inventory = {

    "stage":
        STAGE_NAME,

    "status":
        "INVENTORY_ONLY_PENDING_GROUNDING",

    "method":
        (
            "3x3 overlapping Qwen2.5-VL-7B "
            "tiled discovery + conservative "
            "Qwen consolidation"
        ),

    "model":
        MODEL_ID,

    "input":
        str(
            INPUT_PATH
        ),

    "input_size": {
        "width":
            W,

        "height":
            H,
    },

    "settings": {
        "overlap":
            OVERLAP,

        "crop_tokens":
            CROP_TOKENS,

        "max_crop_objects":
            MAX_CROP_OBJECTS,

        "consolidation_tokens":
            CONSOLIDATION_TOKENS,
    },

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

    "rules": [

        "Stage05 image is used only for easier object discovery.",

        "No object RGB is extracted in Stage06A.",

        "Stage01 master remains eventual final RGB source.",

        "Mirror is excluded because it is a separate Stage03 layer.",

        "Glass partition is excluded because it is a separate Stage04 layer.",

        "Architecture surfaces are excluded.",

        "Final inventory remains subject to grounding and verification.",
    ],
}


inventory_path = (
    OUT
    / "04_verified_dynamic_inventory.json"
)


inventory_path.write_text(
    json.dumps(
        inventory,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# ============================================================
# 14. HUMAN REPORT
# ============================================================

report_lines = [

    "=" * 100,

    "PRODUCTION STAGE06A PROP INVENTORY",

    "=" * 100,

    f"RAW TILED CANDIDATES: {len(all_candidates)}",

    f"FINAL INVENTORY: {len(final_objects)}",

    "",

    "OBJECT LIST:",
]


for row in final_objects:

    report_lines.append(
        (
            f'{row["id"]:02d}. '
            f'{row["name"]} | '
            f'{row["grounding_phrase"]} | '
            f'{row["confidence"]} | '
            f'{",".join(row.get("source_regions", []))}'
        )
    )


report_text = (
    "\n".join(
        report_lines
    )
)


report_path = (
    OUT
    / "05_inventory_report.txt"
)


report_path.write_text(
    report_text,
    encoding="utf-8"
)


print()
print(report_text)


# ============================================================
# 15. RELEASE MODEL
# ============================================================

try:
    del model
except Exception:
    pass

try:
    del processor
except Exception:
    pass


gc.collect()


if torch.cuda.is_available():

    torch.cuda.empty_cache()


# ============================================================
# 16. COMPLETE
# ============================================================

print()
print("=" * 100)
print("PRODUCTION STAGE 06A COMPLETE")
print("=" * 100)

print()
print(
    "INVENTORY:",
    inventory_path
)

print(
    "REPORT:",
    report_path
)

print(
    "TILE LAYOUT:",
    OUT
    / "00_tile_layout.png"
)


print()
print(
    "TILE LAYOUT"
)

display(
    Image.open(
        OUT
        / "00_tile_layout.png"
    )
)
