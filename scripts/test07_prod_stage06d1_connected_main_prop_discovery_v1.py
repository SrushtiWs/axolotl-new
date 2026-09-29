
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

STAGE06 = (
    PROD
    / "stage06_prop_layer"
)

DETECTION_MASTER = (
    PROD
    / "stage05_clean_room_with_props"
    / "12_final_canonical_prop_detection_master.png"
)

RGB_MASTER = (
    PROD
    / "stage01_master"
    / "00_master_input.png"
)

OUT = (
    STAGE06
    / "06d1_connected_main_prop_discovery"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

PREVIEWS = (
    OUT
    / "previews"
)

PREVIEWS.mkdir(
    parents=True,
    exist_ok=True
)

CACHE = (
    "/workspace/data/huggingface-cache"
)


QWEN_MODEL = (
    "Qwen/Qwen2.5-VL-7B-Instruct"
)

FLORENCE_MODEL = (
    "florence-community/Florence-2-large"
)


# ============================================================
# VALIDATE INPUTS
# ============================================================

for path in [
    DETECTION_MASTER,
    RGB_MASTER,
]:

    if not path.exists():

        raise FileNotFoundError(
            path
        )


clean_pil = Image.open(
    DETECTION_MASTER
).convert(
    "RGB"
)

rgb_master_pil = Image.open(
    RGB_MASTER
).convert(
    "RGB"
)


if clean_pil.size != rgb_master_pil.size:

    raise RuntimeError(
        "Stage05F and Stage01 dimensions differ."
    )


W, H = clean_pil.size


print("=" * 110)
print("PRODUCTION STAGE 06D1")
print("CONNECTED MAIN-PROP GROUP DISCOVERY")
print("=" * 110)

print()
print(
    "DETECTION INPUT:",
    DETECTION_MASTER
)

print(
    "FINAL RGB SOURCE:",
    RGB_MASTER
)

print(
    "SIZE:",
    (W, H)
)


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


def extract_json_array(
    text
):

    text = text.strip()


    # --------------------------------------------------------
    # Remove markdown fences if present.
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Direct parse.
    # --------------------------------------------------------

    try:

        obj = json.loads(
            text
        )

        if isinstance(
            obj,
            list
        ):
            return obj

        if (
            isinstance(
                obj,
                dict
            )
            and
            isinstance(
                obj.get(
                    "groups"
                ),
                list
            )
        ):

            return obj[
                "groups"
            ]

    except Exception:
        pass


    # --------------------------------------------------------
    # Extract outer JSON array.
    # --------------------------------------------------------

    start = text.find(
        "["
    )

    end = text.rfind(
        "]"
    )


    if (
        start >= 0
        and
        end > start
    ):

        candidate = text[
            start:end + 1
        ]

        try:

            obj = json.loads(
                candidate
            )

            if isinstance(
                obj,
                list
            ):

                return obj

        except Exception:
            pass


    raise RuntimeError(
        "Could not parse connected-group JSON from Qwen."
    )


def normalize_groups(
    rows
):

    result = []


    for idx, row in enumerate(
        rows,
        start=1
    ):

        if not isinstance(
            row,
            dict
        ):
            continue


        name = str(
            row.get(
                "group_name",
                row.get(
                    "name",
                    ""
                )
            )
        ).strip()


        phrase = str(
            row.get(
                "grounding_phrase",
                ""
            )
        ).strip()


        members = row.get(
            "members",
            []
        )


        if isinstance(
            members,
            str
        ):

            members = [
                members
            ]


        members = [

            str(x).strip()

            for x in members

            if str(x).strip()
        ]


        rationale = str(
            row.get(
                "connection_reason",
                row.get(
                    "reason",
                    ""
                )
            )
        ).strip()


        if not name:

            continue


        if not phrase:

            if members:

                phrase = (
                    "complete connected physical group containing "
                    +
                    ", ".join(
                        members
                    )
                )

            else:

                phrase = name


        result.append({

            "group_id":
                len(result) + 1,

            "group_name":
                name,

            "members":
                members,

            "grounding_phrase":
                phrase,

            "connection_reason":
                rationale,
        })


    return result


def extract_florence_boxes(
    parsed
):

    result = []


    if not isinstance(
        parsed,
        dict
    ):

        return result


    for task_key, value in parsed.items():

        if not isinstance(
            value,
            dict
        ):
            continue


        boxes = value.get(
            "bboxes",
            []
        )

        labels = value.get(
            "labels",
            []
        )

        scores = value.get(
            "scores",
            []
        )


        for i, box in enumerate(
            boxes
        ):

            if (
                not isinstance(
                    box,
                    (list, tuple)
                )
                or
                len(box) != 4
            ):

                continue


            result.append({

                "bbox":
                    [
                        float(v)
                        for v in box
                    ],

                "label":
                    (
                        str(labels[i])
                        if i < len(labels)
                        else ""
                    ),

                "score":
                    (
                        float(scores[i])
                        if i < len(scores)
                        else None
                    ),

                "task":
                    str(task_key),
            })


    return result


def clamp_box(
    box
):

    x1, y1, x2, y2 = [
        float(v)
        for v in box
    ]


    x1 = max(
        0.0,
        min(
            float(W - 1),
            x1
        )
    )

    y1 = max(
        0.0,
        min(
            float(H - 1),
            y1
        )
    )

    x2 = max(
        x1 + 1.0,
        min(
            float(W),
            x2
        )
    )

    y2 = max(
        y1 + 1.0,
        min(
            float(H),
            y2
        )
    )


    return [
        x1,
        y1,
        x2,
        y2
    ]


def bbox_area(
    box
):

    return max(
        0.0,
        box[2] - box[0]
    ) * max(
        0.0,
        box[3] - box[1]
    )


def bbox_iou(
    a,
    b
):

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b


    ix1 = max(
        ax1,
        bx1
    )

    iy1 = max(
        ay1,
        by1
    )

    ix2 = min(
        ax2,
        bx2
    )

    iy2 = min(
        ay2,
        by2
    )


    iw = max(
        0.0,
        ix2 - ix1
    )

    ih = max(
        0.0,
        iy2 - iy1
    )


    intersection = (
        iw
        *
        ih
    )


    union = (
        bbox_area(
            a
        )
        +
        bbox_area(
            b
        )
        -
        intersection
    )


    return float(
        intersection
        /
        max(
            1.0,
            union
        )
    )


def deduplicate_boxes(
    boxes
):

    unique = []


    for item in boxes:

        duplicate_index = None


        for idx, kept in enumerate(
            unique
        ):

            if bbox_iou(
                item[
                    "bbox"
                ],
                kept[
                    "bbox"
                ]
            ) >= 0.75:

                duplicate_index = idx
                break


        if duplicate_index is None:

            unique.append(
                dict(
                    item
                )
            )


    return unique


# ============================================================
# PHASE A
# QWEN CONNECTED-GROUP UNDERSTANDING
# ============================================================

print()
print("=" * 110)
print("PHASE A — QWEN CONNECTED-GROUP INVENTORY")
print("=" * 110)


from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)

from qwen_vl_utils import (
    process_vision_info,
)


qwen_processor = (
    AutoProcessor
    .from_pretrained(
        QWEN_MODEL,
        cache_dir=CACHE
    )
)


qwen_model = (
    Qwen2_5_VLForConditionalGeneration
    .from_pretrained(
        QWEN_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=CACHE,
        low_cpu_mem_usage=True,
    )
)


qwen_model.eval()


GROUP_PROMPT = """
You are analyzing a CLEAN room image for computer-vision
foreground extraction.

The mirror and transparent glass enclosure have already been
removed intentionally.

Your task is NOT to list every small object independently.

Your task is to identify the distinct CONNECTED PHYSICAL PROP
GROUPS visible in the image.

CONNECTED PROP GROUP RULE:

If multiple non-architectural physical objects visibly touch,
rest on, overlap, attach to, connect to, or form one physical
assembly, combine them into ONE group.

Examples of grouping logic:
- cabinet + countertop + basin + faucet + bottle resting on
  countertop + drawer + handles -> one connected group
- toilet body + seat/lid -> one connected group
- shower head + shower arm -> one connected group

IMPORTANT:

Walls, floors and ceilings are architectural surfaces.
They are NEVER included in a prop group merely because a
fixture is mounted on them.

Therefore:
- a wall switch remains a standalone prop;
- a wall-mounted toilet-paper holder remains a standalone prop;
- a recessed ceiling light remains a standalone prop;
- a wall-mounted shower fixture remains a standalone connected
  fixture group.

Do not include:
- wall
- floor
- ceiling
- shadows
- reflections
- lighting patches
- stains
- seams
- surface texture
- tile patterns
- removed mirror
- removed glass

Do not invent objects that are not visibly present.

For every connected physical group, return:
1. group_name
2. members
3. grounding_phrase
4. connection_reason

The grounding_phrase must visually describe the COMPLETE
connected group so another vision model can locate the whole
group.

Return ONLY valid JSON in exactly this format:

[
  {
    "group_name": "short generic group name",
    "members": [
      "visible physical member",
      "visible physical member"
    ],
    "grounding_phrase": "visual phrase describing the whole connected physical group",
    "connection_reason": "why these visible members form one physical group"
  }
]
"""


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
                        DETECTION_MASTER
                    ),
            },

            {
                "type":
                    "text",

                "text":
                    GROUP_PROMPT,
            },
        ],
    }
]


text = (
    qwen_processor
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


inputs = qwen_processor(
    text=[
        text
    ],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt"
)


inputs = inputs.to(
    qwen_model.device
)


with torch.inference_mode():

    generated_ids = (
        qwen_model.generate(
            **inputs,
            max_new_tokens=1200,
            do_sample=False,
            repetition_penalty=1.05,
        )
    )


generated_trimmed = [

    out_ids[
        len(in_ids):
    ]

    for in_ids, out_ids in zip(
        inputs.input_ids,
        generated_ids
    )
]


qwen_text = (
    qwen_processor
    .batch_decode(
        generated_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
)


RAW_QWEN_PATH = (
    OUT
    / "00_qwen_connected_groups_raw.txt"
)


RAW_QWEN_PATH.write_text(
    qwen_text,
    encoding="utf-8"
)


print()
print("QWEN RAW OUTPUT:")
print("-" * 110)
print(qwen_text)


raw_groups = extract_json_array(
    qwen_text
)


groups = normalize_groups(
    raw_groups
)


if not groups:

    raise RuntimeError(
        "Qwen produced no usable connected groups."
    )


print()
print("-" * 110)
print(
    "NORMALIZED CONNECTED GROUPS:",
    len(groups)
)
print("-" * 110)


for group in groups:

    print()
    print(
        f'GROUP {group["group_id"]:02d}: '
        f'{group["group_name"]}'
    )

    print(
        "  MEMBERS:",
        group[
            "members"
        ]
    )

    print(
        "  GROUNDING:",
        group[
            "grounding_phrase"
        ]
    )

    print(
        "  WHY:",
        group[
            "connection_reason"
        ]
    )


GROUPS_JSON = (
    OUT
    / "01_connected_group_inventory.json"
)


GROUPS_JSON.write_text(
    json.dumps(
        groups,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# ============================================================
# FREE QWEN BEFORE FLORENCE
# ============================================================

del generated_ids
del generated_trimmed
del inputs
del qwen_model
del qwen_processor

gc.collect()

if torch.cuda.is_available():

    torch.cuda.empty_cache()


print()
print(
    "✅ QWEN UNLOADED"
)


# ============================================================
# PHASE B
# FLORENCE LOCALIZATION
# ============================================================

print()
print("=" * 110)
print("PHASE B — FLORENCE CONNECTED-GROUP LOCALIZATION")
print("=" * 110)


from transformers import (
    AutoProcessor,
    AutoModelForMultimodalLM,
)


florence_processor = (
    AutoProcessor
    .from_pretrained(
        FLORENCE_MODEL,
        cache_dir=CACHE
    )
)


florence_model = (
    AutoModelForMultimodalLM
    .from_pretrained(
        FLORENCE_MODEL,
        dtype=torch.float16,
        device_map="auto",
        cache_dir=CACHE
    )
)


florence_model.eval()


def run_florence(
    phrase
):

    task = (
        "<CAPTION_TO_PHRASE_GROUNDING>"
    )


    text = (
        task
        +
        phrase
    )


    model_inputs = (
        florence_processor(
            text=text,
            images=clean_pil,
            return_tensors="pt"
        )
    )


    model_inputs = {

        k:
            v.to(
                florence_model.device
            )
            if hasattr(
                v,
                "to"
            )
            else v

        for k, v
        in model_inputs.items()
    }


    with torch.inference_mode():

        ids = (
            florence_model.generate(
                **model_inputs,
                max_new_tokens=256,
                num_beams=3,
                do_sample=False
            )
        )


    generated = (
        florence_processor
        .batch_decode(
            ids,
            skip_special_tokens=False
        )[0]
    )


    parsed = (
        florence_processor
        .post_process_generation(
            generated,
            task=task,
            image_size=clean_pil.size
        )
    )


    boxes = extract_florence_boxes(
        parsed
    )


    return (
        generated,
        boxes
    )


localized_groups = []


for group in groups:

    gid = group[
        "group_id"
    ]

    name = group[
        "group_name"
    ]

    phrase = group[
        "grounding_phrase"
    ]


    print()
    print("=" * 100)

    print(
        f"GROUP {gid:02d}: {name}"
    )

    print("=" * 100)

    print(
        "PHRASE:",
        phrase
    )


    generated, boxes = run_florence(
        phrase
    )


    normalized_boxes = []


    for item in boxes:

        bbox = clamp_box(
            item[
                "bbox"
            ]
        )

        normalized_boxes.append({

            "bbox":
                bbox,

            "label":
                item[
                    "label"
                ],

            "score":
                item[
                    "score"
                ],
        })


    unique_boxes = deduplicate_boxes(
        normalized_boxes
    )


    # --------------------------------------------------------
    # NUMBER CANDIDATES
    # --------------------------------------------------------

    for idx, item in enumerate(
        unique_boxes,
        start=1
    ):

        item[
            "candidate_index"
        ] = idx


    print(
        "CANDIDATES:",
        len(
            unique_boxes
        )
    )


    for item in unique_boxes:

        print(
            "  #{} bbox={} label={}".format(

                item[
                    "candidate_index"
                ],

                [
                    round(v, 1)
                    for v in item[
                        "bbox"
                    ]
                ],

                item[
                    "label"
                ],
            )
        )


    # --------------------------------------------------------
    # INDIVIDUAL GROUP PREVIEW
    # --------------------------------------------------------

    preview = clean_pil.copy()

    draw = ImageDraw.Draw(
        preview
    )


    for item in unique_boxes:

        idx = item[
            "candidate_index"
        ]

        x1, y1, x2, y2 = (
            item[
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
            width=3
        )


        draw.rectangle(
            [
                x1,
                y1,
                x1 + 32,
                y1 + 22
            ],
            fill="white",
            outline="red"
        )


        draw.text(
            (
                x1 + 5,
                y1 + 3
            ),
            str(idx),
            fill="red"
        )


    preview_path = (
        PREVIEWS
        /
        f'group_{gid:02d}_candidates.png'
    )


    preview.save(
        preview_path
    )


    localized_groups.append({

        **group,

        "florence_generated":
            generated,

        "candidate_count":
            len(
                unique_boxes
            ),

        "candidates":
            unique_boxes,

        "preview_path":
            str(
                preview_path
            ),

        "status":
            (
                "CANDIDATES_FOUND"
                if unique_boxes
                else
                "NO_LOCALIZATION"
            ),
    })


# ============================================================
# GLOBAL OVERVIEW
# ============================================================

overview = clean_pil.copy()

draw = ImageDraw.Draw(
    overview
)


for group in localized_groups:

    gid = group[
        "group_id"
    ]


    for item in group[
        "candidates"
    ]:

        x1, y1, x2, y2 = (
            item[
                "bbox"
            ]
        )


        label = (
            f'G{gid}:'
            f'{item["candidate_index"]}'
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
                x1 + 2,
                y1 + 2
            ),
            label,
            fill="red"
        )


OVERVIEW_PATH = (
    OUT
    / "02_all_connected_group_candidates.png"
)


overview.save(
    OVERVIEW_PATH
)


# ============================================================
# SAVE
# ============================================================

RESULT = {

    "stage":
        "06D1",

    "detection_input":
        str(
            DETECTION_MASTER
        ),

    "final_rgb_source":
        str(
            RGB_MASTER
        ),

    "connected_group_rule":
        (
            "visible non-architectural physical objects that "
            "touch, rest on, overlap, attach to, connect to, "
            "or form one physical assembly are represented "
            "as one connected main-prop group"
        ),

    "architectural_surface_exception":
        (
            "wall, floor and ceiling never become members "
            "of a prop group merely because a fixture is "
            "mounted to them"
        ),

    "groups":
        localized_groups,

    "status":
        "REQUIRES_VISUAL_CONNECTED_GROUP_AUDIT",

    "rules": [
        "Stage05F is the only detection/grouping image",
        "Stage01 remains final RGB source",
        "mirror is owned by Stage03",
        "glass enclosure is owned by Stage04",
        "no SAM2 was run",
        "no final mask union was created",
        "no Florence candidate is automatically accepted",
    ],
}


RESULT_PATH = (
    OUT
    / "00_stage06d1_result.json"
)


RESULT_PATH.write_text(
    json.dumps(
        RESULT,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print()
print("=" * 110)
print("PRODUCTION STAGE 06D1 RESULT")
print("=" * 110)


print(
    "CONNECTED GROUPS:",
    len(
        localized_groups
    )
)


for group in localized_groups:

    print()
    print(
        "G{:02d}. {:35s} | candidates={}".format(

            group[
                "group_id"
            ],

            group[
                "group_name"
            ][:35],

            group[
                "candidate_count"
            ],
        )
    )

    print(
        "     MEMBERS:",
        group[
            "members"
        ]
    )

    print(
        "     PHRASE:",
        group[
            "grounding_phrase"
        ]
    )


    for item in group[
        "candidates"
    ]:

        print(
            "     #{} bbox={} label={}".format(

                item[
                    "candidate_index"
                ],

                [
                    round(v, 1)
                    for v in item[
                        "bbox"
                    ]
                ],

                item[
                    "label"
                ],
            )
        )


print()
print(
    "RESULT JSON:",
    RESULT_PATH
)

print(
    "OVERVIEW:",
    OVERVIEW_PATH
)

print()
print(
    "NO SAM2 OR FINAL PROP MASK UNION WAS RUN."
)


# ============================================================
# INLINE VISUAL VERIFICATION
# ============================================================

print()
print("=" * 110)
print("INLINE CONNECTED-GROUP VERIFICATION")
print("=" * 110)


show(
    clean_pil,
    (
        "06D1 INPUT — Stage05F "
        "Canonical Clean Prop-Detection Master"
    ),
    figsize=(8, 9)
)


show(
    OVERVIEW_PATH,
    (
        "06D1 — ALL CONNECTED MAIN-PROP "
        "LOCALIZATION CANDIDATES"
    ),
    figsize=(8, 9)
)


for group in localized_groups:

    show(

        group[
            "preview_path"
        ],

        (
            f'G{group["group_id"]:02d} — '
            f'{group["group_name"]} '
            f'| Florence Candidates'
        ),

        figsize=(8, 9)
    )


# ============================================================
# CLEANUP
# ============================================================

del florence_model
del florence_processor

gc.collect()

if torch.cuda.is_available():

    torch.cuda.empty_cache()
