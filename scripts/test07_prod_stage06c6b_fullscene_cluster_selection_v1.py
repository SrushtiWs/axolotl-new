
from pathlib import Path
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
# PATHS
# ============================================================

BASE = Path("/workspace/axolotl")

PROD = (
    BASE
    / "test07"
    / "production_pipeline"
)

STAGE06 = (
    PROD
    / "stage06_prop_layer"
)

MASTER = (
    PROD
    / "stage01_master"
    / "00_master_input.png"
)

CONSENSUS_JSON = (
    STAGE06
    / "06c6_multi_route_candidate_consensus"
    / "00_multi_route_consensus_all.json"
)

OUT = (
    STAGE06
    / "06c6b_fullscene_cluster_selection"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

VIS_DIR = (
    OUT
    / "candidate_overlays"
)

VIS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_CACHE = Path(
    "/workspace/data/huggingface-cache"
)


# ============================================================
# MODEL
# ============================================================

MODEL_ID = (
    "Qwen/Qwen2.5-VL-7B-Instruct"
)

MAX_NEW_TOKENS = 50


PROMPT = """
You are selecting the best localization candidate for one physical
object in a real room image.

The target object is ALREADY CONFIRMED TO EXIST in this image.

TARGET OBJECT:
{name}

TARGET DESCRIPTION:
{phrase}

The image shows the COMPLETE original room.

Candidate localization boxes are numbered.

Your task is ONLY to decide which numbered box best corresponds
to the target physical object.

Rules:

1. Judge the actual visible physical object.
2. Do not choose a box merely because it is near the object.
3. Do not choose a large container or furniture region when the
   target is only a small fixture on it.
4. Different object classes must remain distinct.
   Example: a sink is not a faucet.
5. Prefer the box that most directly corresponds to the intended
   physical object.
6. If none of the numbered boxes correctly localize the object,
   return NONE.
7. Do not invent another coordinate or location.
8. Return exactly one line.

Return:

CHOICE: 1

or

CHOICE: 2

or

CHOICE: 3

or

CHOICE: NONE
""".strip()


# ============================================================
# HELPERS
# ============================================================

def parse_choice(raw, max_choice):

    upper = raw.upper()

    if "CHOICE: NONE" in upper:
        return None

    m = re.search(
        r"CHOICE:\s*(\d+)",
        upper
    )

    if not m:
        return None

    value = int(
        m.group(1)
    )

    if (
        value < 1
        or
        value > max_choice
    ):
        return None

    return value


def safe_name(text):

    return "".join(
        c
        if c.isalnum() or c in "-_"
        else "_"
        for c in str(text)
    )[:60]


# ============================================================
# LOAD INPUT
# ============================================================

if not MASTER.exists():
    raise FileNotFoundError(MASTER)

if not CONSENSUS_JSON.exists():
    raise FileNotFoundError(
        CONSENSUS_JSON
    )


master = Image.open(
    MASTER
).convert(
    "RGB"
)

consensus = json.loads(
    CONSENSUS_JSON.read_text(
        encoding="utf-8"
    )
)


print("=" * 110)
print("PRODUCTION STAGE 06C6B")
print("FULL-SCENE COMPARATIVE CLUSTER SELECTION")
print("=" * 110)

print(
    "INSTANCES:",
    len(
        consensus
    )
)


# ============================================================
# LOAD QWEN
# ============================================================

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
        torch_dtype=dtype,
        device_map="auto",
        cache_dir=str(
            MODEL_CACHE
        )
    )
)

model.eval()


processor = (
    AutoProcessor
    .from_pretrained(
        MODEL_ID,
        cache_dir=str(
            MODEL_CACHE
        )
    )
)


print(
    "✅ QWEN READY"
)


# ============================================================
# PROCESS EACH PHYSICAL INSTANCE
# ============================================================

results = []
selected = []
unresolved = []


for index, instance in enumerate(
    consensus,
    start=1
):

    iid = int(
        instance[
            "inventory_id"
        ]
    )

    name = instance.get(
        "inventory_name",
        ""
    )

    phrase = instance.get(
        "grounding_phrase",
        ""
    )

    clusters = instance.get(
        "clusters",
        []
    )[:3]


    print()
    print("=" * 90)

    print(
        f"{iid:03d}. {name}"
    )


    if not clusters:

        result = {
            "inventory_id":
                iid,

            "inventory_name":
                name,

            "grounding_phrase":
                phrase,

            "choice":
                None,

            "status":
                "NO_CLUSTER",
        }

        results.append(
            result
        )

        unresolved.append(
            result
        )

        print(
            "❌ NO CLUSTER"
        )

        continue


    # ========================================================
    # DRAW FULL-SCENE NUMBERED BOXES
    # ========================================================

    overlay = master.copy()

    draw = ImageDraw.Draw(
        overlay
    )


    for rank, cluster in enumerate(
        clusters,
        start=1
    ):

        x1, y1, x2, y2 = [
            float(v)
            for v in cluster[
                "bbox"
            ]
        ]


        draw.rectangle(
            [
                x1,
                y1,
                x2,
                y2
            ],
            outline="red",
            width=4
        )


        label_x = max(
            0,
            int(
                x1
            )
        )

        label_y = max(
            0,
            int(
                y1
            )
            -
            18
        )


        draw.rectangle(
            [
                label_x,
                label_y,
                label_x + 20,
                label_y + 18
            ],
            fill="white",
            outline="red"
        )


        draw.text(
            (
                label_x + 5,
                label_y + 2
            ),
            str(
                rank
            ),
            fill="red"
        )


    overlay_path = (
        VIS_DIR
        /
        f"{iid:03d}_{safe_name(name)}_candidates.png"
    )


    overlay.save(
        overlay_path
    )


    # ========================================================
    # QWEN COMPARATIVE CHOICE
    # ========================================================

    prompt = PROMPT.format(
        name=name,
        phrase=phrase
    )


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
                            overlay_path
                        )
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

        generated = model.generate(
            **inputs,
            max_new_tokens=
                MAX_NEW_TOKENS,
            do_sample=False
        )


    trimmed = [
        output_ids[
            len(
                input_ids
            ):
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


    choice = parse_choice(
        raw,
        len(
            clusters
        )
    )


    # ========================================================
    # RESULT
    # ========================================================

    if choice is None:

        result = {

            "inventory_id":
                iid,

            "inventory_name":
                name,

            "grounding_phrase":
                phrase,

            "choice":
                None,

            "choice_raw":
                raw,

            "status":
                "UNRESOLVED_NONE",

            "overlay_path":
                str(
                    overlay_path
                ),
        }


        unresolved.append(
            result
        )


        print(
            "CHOICE: NONE"
        )


    else:

        cluster = clusters[
            choice - 1
        ]


        result = {

            "inventory_id":
                iid,

            "inventory_name":
                name,

            "grounding_phrase":
                phrase,

            "choice":
                choice,

            "choice_raw":
                raw,

            "bbox":
                cluster[
                    "bbox"
                ],

            "score":
                cluster[
                    "max_dino_score"
                ],

            "consensus_score":
                cluster[
                    "consensus_score"
                ],

            "consensus_routes":
                cluster[
                    "routes"
                ],

            "consensus_route_count":
                cluster[
                    "route_count"
                ],

            "consensus_proposal_count":
                cluster[
                    "proposal_count"
                ],

            "source_regions":
                cluster[
                    "source_regions"
                ],

            "status":
                "COMPARATIVE_CLUSTER_SELECTED",

            "overlay_path":
                str(
                    overlay_path
                ),

            "candidate_origin":
                "06C6B_FULLSCENE_COMPARATIVE_SELECTION",
        }


        selected.append(
            result
        )


        print(
            f"CHOICE: {choice}"
        )

        print(
            "BBOX:",
            [
                round(
                    v,
                    1
                )
                for v in result[
                    "bbox"
                ]
            ]
        )


    results.append(
        result
    )


# ============================================================
# SAVE
# ============================================================

ALL_PATH = (
    OUT
    / "00_fullscene_cluster_choices.json"
)

SELECTED_PATH = (
    OUT
    / "01_selected_sam2_seed_boxes.json"
)

UNRESOLVED_PATH = (
    OUT
    / "02_unresolved_instances.json"
)


ALL_PATH.write_text(
    json.dumps(
        results,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


SELECTED_PATH.write_text(
    json.dumps(
        selected,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


UNRESOLVED_PATH.write_text(
    json.dumps(
        unresolved,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# ============================================================
# PREVIEW SELECTED ONLY
# ============================================================

preview = master.copy()

draw = ImageDraw.Draw(
    preview
)


for row in selected:

    x1, y1, x2, y2 = [
        float(v)
        for v in row[
            "bbox"
        ]
    ]


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
                y1 - 13
            )
        ),
        (
            f'{row["inventory_id"]}:'
            f'{row["inventory_name"]}'
        ),
        fill="red"
    )


PREVIEW_PATH = (
    OUT
    / "03_selected_seed_preview.png"
)

preview.save(
    PREVIEW_PATH
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print()
print("=" * 110)
print("PRODUCTION STAGE 06C6B RESULT")
print("=" * 110)


print(
    "PRESENT INPUT:",
    len(
        consensus
    )
)

print(
    "SELECTED SEEDS:",
    len(
        selected
    )
)

print(
    "UNRESOLVED:",
    len(
        unresolved
    )
)


print()
print(
    "PER INSTANCE:"
)


for row in results:

    iid = row[
        "inventory_id"
    ]

    name = row[
        "inventory_name"
    ]


    if row.get(
        "choice"
    ) is None:

        print(
            f"{iid:03d}. {name} | NONE | UNRESOLVED"
        )

    else:

        print(
            "{:03d}. {} | cluster={} | bbox={} | routes={} | support={}".format(

                iid,

                name,

                row[
                    "choice"
                ],

                [
                    round(
                        v,
                        1
                    )
                    for v in row[
                        "bbox"
                    ]
                ],

                row[
                    "consensus_routes"
                ],

                row[
                    "consensus_proposal_count"
                ],
            )
        )


print()
print(
    "SELECTED SEEDS:",
    SELECTED_PATH
)

print(
    "UNRESOLVED:",
    UNRESOLVED_PATH
)

print(
    "PREVIEW:",
    PREVIEW_PATH
)

print()
print(
    "No SAM2 has been run."
)
