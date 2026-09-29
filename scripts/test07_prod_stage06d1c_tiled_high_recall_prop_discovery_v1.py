
from pathlib import Path
import json
import re
import gc

import torch

from PIL import (
    Image,
    ImageDraw
)

import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE = Path("/workspace/axolotl")

PROD = (
    BASE
    / "test07"
    / "production_pipeline"
)

DETECTION_MASTER = (
    PROD
    / "stage05_clean_room_with_props"
    / "12_final_canonical_prop_detection_master.png"
)

OUT = (
    PROD
    / "stage06_prop_layer"
    / "06d1c_tiled_high_recall_prop_discovery"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

TILES_DIR = (
    OUT
    / "tiles"
)

TILES_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RAW_DIR = (
    OUT
    / "raw"
)

RAW_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CACHE = (
    "/workspace/data/huggingface-cache"
)

MODEL_ID = (
    "Qwen/Qwen2.5-VL-7B-Instruct"
)


# ============================================================
# CONFIG
# ============================================================

ROWS = 3
COLS = 3

OVERLAP = 0.18

MAX_OBJECTS_PER_TILE = 14

MAX_NEW_TOKENS = 360


# ============================================================
# VALIDATE
# ============================================================

if not DETECTION_MASTER.exists():

    raise FileNotFoundError(
        DETECTION_MASTER
    )


master = Image.open(
    DETECTION_MASTER
).convert(
    "RGB"
)

W, H = master.size


# ============================================================
# HELPERS
# ============================================================

def show(
    image_or_path,
    title,
    figsize=(8, 9)
):

    if isinstance(
        image_or_path,
        (str, Path)
    ):

        image = Image.open(
            image_or_path
        )

    else:

        image = image_or_path


    plt.figure(
        figsize=figsize
    )

    plt.imshow(
        image
    )

    plt.title(
        title
    )

    plt.axis(
        "off"
    )

    plt.show()


def parse_json_array(text):

    text = text.strip()

    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.I
    )

    text = re.sub(
        r"\s*```$",
        "",
        text
    )


    try:

        parsed = json.loads(
            text
        )

        if isinstance(
            parsed,
            list
        ):

            return parsed

    except Exception:
        pass


    s = text.find(
        "["
    )

    e = text.rfind(
        "]"
    )


    if (
        s >= 0
        and
        e > s
    ):

        try:

            parsed = json.loads(
                text[
                    s:e + 1
                ]
            )

            if isinstance(
                parsed,
                list
            ):

                return parsed

        except Exception:
            pass


    return []


def compute_axis_windows(
    length,
    count,
    overlap_ratio
):

    if count <= 1:

        return [
            (
                0,
                length
            )
        ]


    nominal = (
        length
        /
        count
    )


    overlap_px = (
        nominal
        *
        overlap_ratio
    )


    windows = []


    for i in range(
        count
    ):

        start = (
            i * nominal
            -
            overlap_px
        )

        end = (
            (i + 1) * nominal
            +
            overlap_px
        )


        if i == 0:
            start = 0

        if i == count - 1:
            end = length


        start = int(
            round(
                max(
                    0,
                    start
                )
            )
        )

        end = int(
            round(
                min(
                    length,
                    end
                )
            )
        )


        windows.append(
            (
                start,
                end
            )
        )


    return windows


# ============================================================
# TILE DEFINITIONS
# ============================================================

x_windows = compute_axis_windows(
    W,
    COLS,
    OVERLAP
)

y_windows = compute_axis_windows(
    H,
    ROWS,
    OVERLAP
)


region_names = [

    [
        "top_left",
        "top_center",
        "top_right",
    ],

    [
        "middle_left",
        "middle_center",
        "middle_right",
    ],

    [
        "bottom_left",
        "bottom_center",
        "bottom_right",
    ],
]


tiles = []


for row in range(
    ROWS
):

    for col in range(
        COLS
    ):

        x1, x2 = x_windows[
            col
        ]

        y1, y2 = y_windows[
            row
        ]


        name = region_names[
            row
        ][
            col
        ]


        bbox = [
            x1,
            y1,
            x2,
            y2,
        ]


        tile = master.crop(
            (
                x1,
                y1,
                x2,
                y2
            )
        )


        tile_path = (
            TILES_DIR
            / f"{name}.png"
        )


        tile.save(
            tile_path
        )


        tiles.append({

            "region":
                name,

            "row":
                row,

            "col":
                col,

            "bbox":
                bbox,

            "path":
                str(
                    tile_path
                ),
        })


# ============================================================
# TILE LAYOUT PREVIEW
# ============================================================

layout = master.copy()

draw = ImageDraw.Draw(
    layout
)


for tile in tiles:

    x1, y1, x2, y2 = (
        tile[
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


    draw.rectangle(
        [
            x1 + 2,
            y1 + 2,
            x1 + 102,
            y1 + 21
        ],
        fill="white"
    )


    draw.text(
        (
            x1 + 5,
            y1 + 4
        ),
        tile[
            "region"
        ],
        fill="red"
    )


LAYOUT_PATH = (
    OUT
    / "00_tile_layout.png"
)


layout.save(
    LAYOUT_PATH
)


# ============================================================
# LOAD QWEN
# ============================================================

print("=" * 110)
print("PRODUCTION STAGE 06D1C")
print("HIGH-RECALL TILED PHYSICAL-PROP DISCOVERY")
print("=" * 110)

print()
print(
    "INPUT:",
    DETECTION_MASTER
)

print(
    "SIZE:",
    (W, H)
)

print(
    "TILES:",
    len(
        tiles
    )
)


from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)

from qwen_vl_utils import (
    process_vision_info,
)


processor = (
    AutoProcessor
    .from_pretrained(
        MODEL_ID,
        cache_dir=CACHE
    )
)


model = (
    Qwen2_5_VLForConditionalGeneration
    .from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=CACHE,
        low_cpu_mem_usage=True,
    )
)


model.eval()


# ============================================================
# TILE PROMPT
# ============================================================

PROMPT = f"""
You are performing HIGH-RECALL physical-object discovery on
ONE CROP of a CLEAN room image.

The mirror and transparent glass enclosure have already been
removed from the image.

List every clearly visible DISCRETE PHYSICAL OBJECT OR FIXTURE
that appears at least partly inside this crop.

At this stage DO NOT combine touching objects.

List individual visible objects separately because a later
stage will determine which ones physically touch or belong to
one connected prop group.

Include:
- large furniture
- small furniture components
- fixtures
- handles and knobs
- bottles and containers
- sinks and basins
- faucets
- toilets and seats
- electrical plates
- lights
- shower fixtures
- mounted hardware
- small thin objects
- partial objects at crop edges

Do NOT list:
- wall
- floor
- ceiling
- architectural surface
- paint
- tile
- wall texture
- floor texture
- shadow
- reflection
- glare
- light patch
- seam
- corner
- removed mirror
- removed glass

VERY IMPORTANT:
A physical object mounted to a wall or ceiling is still an
object. Do not exclude a wall switch, ceiling light, shower
fixture, or holder just because it touches an architectural
surface.

Do not invent anything that is not visible.

Return at most {MAX_OBJECTS_PER_TILE} objects.

Return ONLY JSON:

[
  {{
    "name": "short generic physical object name",
    "grounding_phrase": "specific visual description of this visible object",
    "confidence": 0.0
  }}
]
"""


# ============================================================
# RUN EACH TILE
# ============================================================

all_observations = []


for tile_index, tile in enumerate(
    tiles,
    start=1
):

    print()
    print("=" * 100)

    print(
        f'TILE {tile_index}/9 — '
        f'{tile["region"]}'
    )

    print("=" * 100)


    messages = [

        {
            "role":
                "user",

            "content": [

                {
                    "type":
                        "image",

                    "image":
                        tile[
                            "path"
                        ],
                },

                {
                    "type":
                        "text",

                    "text":
                        PROMPT,
                },
            ],
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
        text=[
            chat_text
        ],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )


    inputs = inputs.to(
        model.device
    )


    with torch.inference_mode():

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=1.05,
        )


    trimmed = [

        output[
            len(
                input_ids
            ):
        ]

        for input_ids, output
        in zip(
            inputs.input_ids,
            generated_ids
        )
    ]


    raw_text = (
        processor
        .batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
    )


    raw_path = (
        RAW_DIR
        / f'{tile["region"]}.txt'
    )


    raw_path.write_text(
        raw_text,
        encoding="utf-8"
    )


    parsed = parse_json_array(
        raw_text
    )


    accepted = []


    for item in parsed:

        if not isinstance(
            item,
            dict
        ):
            continue


        name = str(
            item.get(
                "name",
                ""
            )
        ).strip()


        phrase = str(
            item.get(
                "grounding_phrase",
                ""
            )
        ).strip()


        try:

            confidence = float(
                item.get(
                    "confidence",
                    0.5
                )
            )

        except Exception:

            confidence = 0.5


        if not name:
            continue


        if not phrase:
            phrase = name


        observation = {

            "observation_id":
                len(
                    all_observations
                )
                +
                1,

            "name":
                name,

            "grounding_phrase":
                phrase,

            "confidence":
                confidence,

            "source_region":
                tile[
                    "region"
                ],

            "source_bbox":
                tile[
                    "bbox"
                ],

            "source_tile_path":
                tile[
                    "path"
                ],
        }


        all_observations.append(
            observation
        )


        accepted.append(
            observation
        )


    print(
        "OBJECTS:",
        len(
            accepted
        )
    )


    for obs in accepted:

        print(
            "  {:03d}. {:25s} | conf={:.2f} | {}".format(

                obs[
                    "observation_id"
                ],

                obs[
                    "name"
                ][:25],

                obs[
                    "confidence"
                ],

                obs[
                    "grounding_phrase"
                ],
            )
        )


# ============================================================
# SAVE RAW OBSERVATIONS
# ============================================================

OBS_PATH = (
    OUT
    / "01_raw_tiled_prop_observations.json"
)


OBS_PATH.write_text(
    json.dumps(
        all_observations,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# ============================================================
# SIMPLE NAME FREQUENCY REPORT
#
# Diagnostic only — NO deduplication.
# ============================================================

name_counts = {}


for obs in all_observations:

    key = obs[
        "name"
    ].strip().lower()


    name_counts[
        key
    ] = (
        name_counts.get(
            key,
            0
        )
        +
        1
    )


sorted_names = sorted(
    name_counts.items(),
    key=lambda item:
        (
            -item[1],
            item[0]
        )
)


# ============================================================
# REPORT
# ============================================================

REPORT_PATH = (
    OUT
    / "02_high_recall_report.txt"
)


lines = [

    "=" * 100,

    "TEST07 PRODUCTION STAGE 06D1C",

    "HIGH-RECALL TILED PHYSICAL-PROP DISCOVERY",

    "=" * 100,

    "",

    f"TOTAL RAW OBSERVATIONS: {len(all_observations)}",

    "",

    "NAME FREQUENCY:",

]


for name, count in sorted_names:

    lines.append(
        f"{count:3d} | {name}"
    )


lines.extend(
    [
        "",
        "=" * 100,
        "ALL OBSERVATIONS",
        "=" * 100,
        "",
    ]
)


for obs in all_observations:

    lines.append(

        "{:03d} | {:18s} | {:25s} | {}".format(

            obs[
                "observation_id"
            ],

            obs[
                "source_region"
            ],

            obs[
                "name"
            ],

            obs[
                "grounding_phrase"
            ],
        )
    )


REPORT_PATH.write_text(
    "\n".join(
        lines
    ),
    encoding="utf-8"
)


# ============================================================
# PRINT RESULT
# ============================================================

print()
print("=" * 110)
print("PRODUCTION STAGE 06D1C RESULT")
print("=" * 110)


print(
    "RAW OBSERVATIONS:",
    len(
        all_observations
    )
)


print()
print(
    "DISCOVERED NAME FREQUENCY:"
)


for name, count in sorted_names:

    print(
        f"  {count:2d} × {name}"
    )


print()
print(
    "OBSERVATIONS JSON:",
    OBS_PATH
)

print(
    "REPORT:",
    REPORT_PATH
)

print()
print(
    "NO DEDUPLICATION WAS PERFORMED."
)

print(
    "NO CONNECTED-GROUP MERGING WAS PERFORMED."
)

print(
    "NO FLORENCE OR SAM2 WAS RUN."
)


# ============================================================
# INLINE VERIFICATION
# ============================================================

print()
print("=" * 110)
print("INLINE 06D1C VERIFICATION")
print("=" * 110)


show(
    LAYOUT_PATH,
    (
        "06D1C — 3×3 OVERLAPPING HIGH-RECALL "
        "DISCOVERY LAYOUT"
    ),
    figsize=(8, 9)
)


# Also print a compact per-tile inventory directly in output.

print()
print("=" * 110)
print("PER-TILE INVENTORY")
print("=" * 110)


for tile in tiles:

    region_rows = [

        obs

        for obs in all_observations

        if obs[
            "source_region"
        ]
        ==
        tile[
            "region"
        ]
    ]


    print()
    print(
        tile[
            "region"
        ].upper()
    )

    print(
        "-" * 70
    )


    if not region_rows:

        print(
            "  NONE"
        )

        continue


    for obs in region_rows:

        print(
            "  {:03d}. {:25s} | {}".format(

                obs[
                    "observation_id"
                ],

                obs[
                    "name"
                ][:25],

                obs[
                    "grounding_phrase"
                ],
            )
        )


# ============================================================
# CLEANUP
# ============================================================

del model
del processor

gc.collect()

if torch.cuda.is_available():

    torch.cuda.empty_cache()
