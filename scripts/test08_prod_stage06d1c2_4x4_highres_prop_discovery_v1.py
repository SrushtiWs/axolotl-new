
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
    / "test08"
    / "production_pipeline"
)

MASTER = (
    PROD
    / "stage05_clean_room_with_props"
    / "12_final_canonical_prop_detection_master.png"
)

OUT = (
    PROD
    / "stage06_prop_layer"
    / "06d1c2_4x4_highres_prop_discovery"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

TILE_DIR = (
    OUT
    / "tiles"
)

TILE_DIR.mkdir(
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

CACHE = "/workspace/data/huggingface-cache"

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"


# ============================================================
# CONFIG
# ============================================================

ROWS = 4
COLS = 4

OVERLAP = 0.20

UPSCALE = 2

MAX_OBJECTS = 12

MAX_NEW_TOKENS = 340


# ============================================================
# LOAD
# ============================================================

if not MASTER.exists():
    raise FileNotFoundError(MASTER)

master = Image.open(
    MASTER
).convert("RGB")

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
        image = Image.open(image_or_path)
    else:
        image = image_or_path

    plt.figure(
        figsize=figsize
    )

    plt.imshow(image)

    plt.title(title)

    plt.axis("off")

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
        obj = json.loads(text)

        if isinstance(obj, list):
            return obj

    except Exception:
        pass


    s = text.find("[")
    e = text.rfind("]")

    if (
        s >= 0
        and
        e > s
    ):

        try:
            obj = json.loads(
                text[s:e + 1]
            )

            if isinstance(obj, list):
                return obj

        except Exception:
            pass


    return []


def make_windows(
    length,
    count,
    overlap
):

    nominal = (
        length
        /
        count
    )

    margin = (
        nominal
        *
        overlap
    )

    windows = []


    for i in range(count):

        start = (
            i * nominal
            -
            margin
        )

        end = (
            (i + 1) * nominal
            +
            margin
        )

        if i == 0:
            start = 0

        if i == count - 1:
            end = length

        windows.append(
            (
                int(
                    round(
                        max(
                            0,
                            start
                        )
                    )
                ),

                int(
                    round(
                        min(
                            length,
                            end
                        )
                    )
                ),
            )
        )


    return windows


# ============================================================
# CREATE 4×4 CROPS
# ============================================================

xs = make_windows(
    W,
    COLS,
    OVERLAP
)

ys = make_windows(
    H,
    ROWS,
    OVERLAP
)


tiles = []


for r in range(ROWS):

    for c in range(COLS):

        x1, x2 = xs[c]
        y1, y2 = ys[r]

        name = (
            f"r{r + 1}_c{c + 1}"
        )

        crop = master.crop(
            (
                x1,
                y1,
                x2,
                y2
            )
        )

        upscaled = crop.resize(
            (
                crop.width * UPSCALE,
                crop.height * UPSCALE
            ),
            Image.Resampling.LANCZOS
        )

        tile_path = (
            TILE_DIR
            / f"{name}.png"
        )

        upscaled.save(
            tile_path
        )

        tiles.append({

            "tile_id":
                len(tiles) + 1,

            "name":
                name,

            "row":
                r,

            "col":
                c,

            "source_bbox":
                [
                    x1,
                    y1,
                    x2,
                    y2
                ],

            "path":
                str(tile_path),
        })


# ============================================================
# LAYOUT PREVIEW
# ============================================================

layout = master.copy()

draw = ImageDraw.Draw(
    layout
)


for tile in tiles:

    x1, y1, x2, y2 = (
        tile[
            "source_bbox"
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
            x1 + 58,
            y1 + 20
        ],
        fill="white"
    )

    draw.text(
        (
            x1 + 4,
            y1 + 4
        ),
        str(
            tile[
                "tile_id"
            ]
        ),
        fill="red"
    )


LAYOUT_PATH = (
    OUT
    / "00_4x4_layout.png"
)


layout.save(
    LAYOUT_PATH
)


# ============================================================
# LOAD QWEN
# ============================================================

print("=" * 110)
print("PRODUCTION STAGE 06D1C2")
print("4×4 HIGH-RES SMALL-PROP DISCOVERY")
print("=" * 110)

print()
print("INPUT:", MASTER)
print("SIZE:", (W, H))
print("TILES:", len(tiles))
print("UPSCALE:", UPSCALE)


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
# PROMPT
# ============================================================

PROMPT = f"""
You are inspecting one HIGH-RESOLUTION crop of a clean room
image for HIGH-RECALL physical-object discovery.

Find every visible DISCRETE PHYSICAL OBJECT OR FIXTURE in this
crop, including very small, thin, partial, wall-mounted,
ceiling-mounted, edge-cut, or distant objects.

At this stage, list individual physical objects separately.
Do NOT combine objects because they touch.

A physical fixture mounted to a wall or ceiling is still an
object.

Pay special attention to objects that are easy to overlook:
small mounted fixtures, holders, knobs, handles, plates,
lights, taps, fittings, rails, bottles, controls, small
hardware and partly visible objects.

Do NOT list architectural/background surfaces:
- wall
- floor
- ceiling
- baseboard
- paint
- tile
- texture
- grout
- corner

Do NOT list:
- shadows
- reflections
- glare
- lighting patches
- seams
- removed mirror
- removed glass enclosure

Do not invent an object if you cannot see it.

Return at most {MAX_OBJECTS} objects.

Return ONLY JSON:

[
  {{
    "name": "short generic physical object name",
    "grounding_phrase": "specific visual description of the visible physical object",
    "confidence": 0.0
  }}
]
"""


# ============================================================
# DISCOVERY
# ============================================================

observations = []


for tile in tiles:

    print()
    print("=" * 90)

    print(
        f'TILE {tile["tile_id"]:02d} '
        f'{tile["name"]}'
    )

    print("=" * 90)


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

        ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=1.05,
        )


    trimmed = [

        out[
            len(inp):
        ]

        for inp, out
        in zip(
            inputs.input_ids,
            ids
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
        RAW_DIR
        / f'{tile["name"]}.txt'
    ).write_text(
        raw,
        encoding="utf-8"
    )


    rows = parse_json_array(
        raw
    )


    accepted = []


    for item in rows:

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


        if not name:
            continue

        if not phrase:
            phrase = name


        try:

            confidence = float(
                item.get(
                    "confidence",
                    0.5
                )
            )

        except Exception:

            confidence = 0.5


        obs = {

            "observation_id":
                len(
                    observations
                )
                +
                1,

            "name":
                name,

            "grounding_phrase":
                phrase,

            "confidence":
                confidence,

            "source_tile_id":
                tile[
                    "tile_id"
                ],

            "source_tile":
                tile[
                    "name"
                ],

            "source_bbox":
                tile[
                    "source_bbox"
                ],

            "upscale":
                UPSCALE,
        }


        observations.append(
            obs
        )

        accepted.append(
            obs
        )


    print(
        "OBJECTS:",
        len(accepted)
    )


    for row in accepted:

        print(
            "  {:03d}. {:24s} | {:.2f} | {}".format(

                row[
                    "observation_id"
                ],

                row[
                    "name"
                ][:24],

                row[
                    "confidence"
                ],

                row[
                    "grounding_phrase"
                ],
            )
        )


# ============================================================
# SAVE
# ============================================================

OBS_PATH = (
    OUT
    / "01_raw_4x4_observations.json"
)


OBS_PATH.write_text(
    json.dumps(
        observations,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# ============================================================
# FREQUENCY
# ============================================================

freq = {}


for row in observations:

    key = (
        row[
            "name"
        ]
        .strip()
        .lower()
    )

    freq[
        key
    ] = (
        freq.get(
            key,
            0
        )
        +
        1
    )


ordered = sorted(
    freq.items(),
    key=lambda x:
        (
            -x[1],
            x[0]
        )
)


# ============================================================
# REPORT
# ============================================================

REPORT_PATH = (
    OUT
    / "02_report.txt"
)


lines = [

    "=" * 100,

    "TEST07 PRODUCTION 06D1C2",

    "4x4 HIGH-RES SMALL-PROP DISCOVERY",

    "=" * 100,

    "",

    f"RAW OBSERVATIONS: {len(observations)}",

    "",
]


for name, count in ordered:

    lines.append(
        f"{count:3d} | {name}"
    )


REPORT_PATH.write_text(
    "\n".join(lines),
    encoding="utf-8"
)


# ============================================================
# PRINT
# ============================================================

print()
print("=" * 110)
print("PRODUCTION STAGE 06D1C2 RESULT")
print("=" * 110)

print(
    "RAW OBSERVATIONS:",
    len(observations)
)

print()
print(
    "DISCOVERED NAME FREQUENCY:"
)


for name, count in ordered:

    print(
        f"  {count:2d} × {name}"
    )


print()
print(
    "OBS JSON:",
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
    "NO GROUPING WAS PERFORMED."
)

print(
    "NO FLORENCE OR SAM2 WAS RUN."
)


# ============================================================
# INLINE
# ============================================================

print()
print("=" * 110)
print("INLINE 06D1C2 AUDIT")
print("=" * 110)


show(
    LAYOUT_PATH,
    "06D1C2 — 4×4 HIGH-RES DISCOVERY LAYOUT"
)


print()
print("=" * 110)
print("PER-TILE 4×4 INVENTORY")
print("=" * 110)


for tile in tiles:

    rows = [

        r

        for r in observations

        if int(
            r[
                "source_tile_id"
            ]
        )
        ==
        int(
            tile[
                "tile_id"
            ]
        )
    ]


    print()
    print(
        f'TILE {tile["tile_id"]:02d} '
        f'{tile["name"]}'
    )

    print("-" * 70)


    if not rows:

        print("  NONE")

        continue


    for row in rows:

        print(
            "  {:03d}. {:24s} | {}".format(

                row[
                    "observation_id"
                ],

                row[
                    "name"
                ][:24],

                row[
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
