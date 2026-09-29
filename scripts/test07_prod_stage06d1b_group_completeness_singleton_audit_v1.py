
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

D1_RESULT = (
    STAGE06
    / "06d1_connected_main_prop_discovery"
    / "00_stage06d1_result.json"
)

OUT = (
    STAGE06
    / "06d1b_group_completeness_singleton_audit"
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
# VALIDATE
# ============================================================

for path in [
    DETECTION_MASTER,
    RGB_MASTER,
    D1_RESULT,
]:

    if not path.exists():
        raise FileNotFoundError(path)


clean_pil = Image.open(
    DETECTION_MASTER
).convert(
    "RGB"
)

W, H = clean_pil.size


d1 = json.loads(
    D1_RESULT.read_text(
        encoding="utf-8"
    )
)

existing_groups = d1[
    "groups"
]


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


def extract_json_object(text):

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

        obj = json.loads(
            text
        )

        if isinstance(
            obj,
            dict
        ):
            return obj

    except Exception:
        pass


    start = text.find(
        "{"
    )

    end = text.rfind(
        "}"
    )

    if (
        start >= 0
        and
        end > start
    ):

        candidate = text[
            start:end + 1
        ]

        obj = json.loads(
            candidate
        )

        if isinstance(
            obj,
            dict
        ):
            return obj


    raise RuntimeError(
        "Could not parse Qwen completeness-audit JSON."
    )


def extract_florence_boxes(parsed):

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
            })


    return result


def clamp_box(box):

    x1, y1, x2, y2 = [
        float(v)
        for v in box
    ]

    x1 = max(
        0.0,
        min(float(W - 1), x1)
    )

    y1 = max(
        0.0,
        min(float(H - 1), y1)
    )

    x2 = max(
        x1 + 1.0,
        min(float(W), x2)
    )

    y2 = max(
        y1 + 1.0,
        min(float(H), y2)
    )

    return [
        x1,
        y1,
        x2,
        y2
    ]


def bbox_area(box):

    return max(
        0.0,
        box[2] - box[0]
    ) * max(
        0.0,
        box[3] - box[1]
    )


def bbox_iou(a, b):

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(
        0.0,
        ix2 - ix1
    )

    ih = max(
        0.0,
        iy2 - iy1
    )

    inter = iw * ih

    union = (
        bbox_area(a)
        +
        bbox_area(b)
        -
        inter
    )

    return float(
        inter
        /
        max(
            1.0,
            union
        )
    )


def dedup_boxes(rows):

    unique = []

    for row in rows:

        duplicate = False

        for kept in unique:

            if bbox_iou(
                row[
                    "bbox"
                ],
                kept[
                    "bbox"
                ]
            ) >= 0.75:

                duplicate = True
                break

        if not duplicate:

            unique.append(
                dict(row)
            )

    return unique


# ============================================================
# BUILD CURRENT GROUP SUMMARY DYNAMICALLY
# ============================================================

group_summary = []


for group in existing_groups:

    group_summary.append({

        "group_id":
            group[
                "group_id"
            ],

        "group_name":
            group[
                "group_name"
            ],

        "members":
            group[
                "members"
            ],

        "grounding_phrase":
            group[
                "grounding_phrase"
            ],
    })


# ============================================================
# PHASE A — QWEN COMPLETENESS AUDIT
# ============================================================

print("=" * 110)
print("PRODUCTION STAGE 06D1B")
print("CONNECTED-GROUP COMPLETENESS + STANDALONE-PROP AUDIT")
print("=" * 110)


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
        QWEN_MODEL,
        cache_dir=CACHE
    )
)


model = (
    Qwen2_5_VLForConditionalGeneration
    .from_pretrained(
        QWEN_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=CACHE,
        low_cpu_mem_usage=True,
    )
)


model.eval()


AUDIT_PROMPT = f"""
You are performing a SECOND-PASS completeness audit on a
clean room image for foreground-prop extraction.

A first pass already produced these connected physical groups:

{json.dumps(group_summary, indent=2)}

Do NOT assume this first-pass inventory is complete.

You must perform TWO tasks.

============================================================
TASK A — EXISTING GROUP COMPLETENESS
============================================================

For each existing group, inspect the image carefully and
identify any VISIBLE NON-ARCHITECTURAL physical object that:

- physically touches that group,
- rests on that group,
- overlaps/contact it,
- is attached to it,
- is mechanically connected to it,
- or forms one continuous physical assembly with it,

but was omitted from the existing members list.

Important:
An object resting on another prop counts as part of the same
connected group even if it is movable.

Examples:
a bottle sitting on a countertop belongs to that countertop/
vanity connected group.

Do not add objects merely because they are nearby.

============================================================
TASK B — MISSING STANDALONE PROPS
============================================================

Find every visible discrete physical prop or fixture that is
NOT represented by any existing connected group.

A standalone prop may be mounted to an architectural surface.

Examples:
- recessed ceiling light
- wall electrical switch/outlet plate
- toilet paper holder
- wall-mounted fixture
- independent shower fixture

Architectural surfaces themselves are NEVER props:
- wall
- floor
- ceiling

Do not include:
- shadows
- reflections
- lighting patches
- seams
- texture
- removed mirror
- removed glass enclosure

Do not invent anything not visibly present.

Return ONLY valid JSON:

{{
  "existing_group_updates": [
    {{
      "group_id": 1,
      "missing_members": [
        "visible omitted object"
      ],
      "reason": "why each omitted object physically belongs to this group"
    }}
  ],
  "standalone_props": [
    {{
      "name": "short generic standalone prop name",
      "grounding_phrase": "precise visual phrase for locating this standalone physical prop",
      "reason": "why it is a standalone prop and not part of an existing group"
    }}
  ]
}}
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
                    AUDIT_PROMPT,
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

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=1200,
        do_sample=False,
        repetition_penalty=1.05,
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


raw_text = (
    processor
    .batch_decode(
        generated_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
)


RAW_PATH = (
    OUT
    / "00_qwen_completeness_audit_raw.txt"
)


RAW_PATH.write_text(
    raw_text,
    encoding="utf-8"
)


print()
print("QWEN RAW AUDIT:")
print("-" * 110)
print(raw_text)


audit = extract_json_object(
    raw_text
)


updates = audit.get(
    "existing_group_updates",
    []
)

standalone = audit.get(
    "standalone_props",
    []
)


# ============================================================
# NORMALIZE UPDATES
# ============================================================

normalized_updates = []


for row in updates:

    try:

        gid = int(
            row.get(
                "group_id"
            )
        )

    except Exception:

        continue


    missing = row.get(
        "missing_members",
        []
    )


    if isinstance(
        missing,
        str
    ):

        missing = [
            missing
        ]


    missing = [

        str(x).strip()

        for x in missing

        if str(x).strip()
    ]


    if not missing:
        continue


    normalized_updates.append({

        "group_id":
            gid,

        "missing_members":
            missing,

        "reason":
            str(
                row.get(
                    "reason",
                    ""
                )
            ).strip(),
    })


# ============================================================
# NORMALIZE STANDALONE
# ============================================================

normalized_standalone = []


for row in standalone:

    if not isinstance(
        row,
        dict
    ):
        continue


    name = str(
        row.get(
            "name",
            ""
        )
    ).strip()


    phrase = str(
        row.get(
            "grounding_phrase",
            ""
        )
    ).strip()


    reason = str(
        row.get(
            "reason",
            ""
        )
    ).strip()


    if not name:
        continue


    if not phrase:
        phrase = name


    normalized_standalone.append({

        "standalone_id":
            len(
                normalized_standalone
            )
            + 1,

        "name":
            name,

        "grounding_phrase":
            phrase,

        "reason":
            reason,
    })


print()
print("=" * 110)
print("GROUP COMPLETENESS UPDATES")
print("=" * 110)


if not normalized_updates:

    print(
        "NONE"
    )


for row in normalized_updates:

    print()
    print(
        "GROUP:",
        row[
            "group_id"
        ]
    )

    print(
        "MISSING MEMBERS:",
        row[
            "missing_members"
        ]
    )

    print(
        "REASON:",
        row[
            "reason"
        ]
    )


print()
print("=" * 110)
print("NEW STANDALONE PROP INVENTORY")
print("=" * 110)


if not normalized_standalone:

    print(
        "NONE"
    )


for row in normalized_standalone:

    print()
    print(
        f'S{row["standalone_id"]:02d}. '
        f'{row["name"]}'
    )

    print(
        "PHRASE:",
        row[
            "grounding_phrase"
        ]
    )

    print(
        "WHY:",
        row[
            "reason"
        ]
    )


# ============================================================
# APPLY MEMBER UPDATES TO EXISTING GROUPS
# ============================================================

corrected_groups = []


for group in existing_groups:

    corrected = {

        "group_id":
            group[
                "group_id"
            ],

        "group_name":
            group[
                "group_name"
            ],

        "members":
            list(
                group[
                    "members"
                ]
            ),

        "grounding_phrase":
            group[
                "grounding_phrase"
            ],

        "connection_reason":
            group[
                "connection_reason"
            ],
    }


    matching_updates = [

        u

        for u in normalized_updates

        if int(
            u[
                "group_id"
            ]
        )
        ==
        int(
            group[
                "group_id"
            ]
        )
    ]


    added = []


    for update in matching_updates:

        for member in update[
            "missing_members"
        ]:

            if (
                member.lower()
                not in
                {
                    x.lower()
                    for x in corrected[
                        "members"
                    ]
                }
            ):

                corrected[
                    "members"
                ].append(
                    member
                )

                added.append(
                    member
                )


    corrected[
        "members_added_by_06D1B"
    ] = added


    corrected_groups.append(
        corrected
    )


# ============================================================
# FREE QWEN
# ============================================================

del model
del processor
del inputs
del generated_ids
del generated_trimmed

gc.collect()

if torch.cuda.is_available():

    torch.cuda.empty_cache()


# ============================================================
# PHASE B — FLORENCE LOCALIZE ONLY NEW STANDALONE PROPS
# ============================================================

print()
print("=" * 110)
print("FLORENCE STANDALONE-PROP LOCALIZATION")
print("=" * 110)


from transformers import (
    AutoProcessor,
    AutoModelForMultimodalLM,
)


f_processor = (
    AutoProcessor
    .from_pretrained(
        FLORENCE_MODEL,
        cache_dir=CACHE
    )
)


f_model = (
    AutoModelForMultimodalLM
    .from_pretrained(
        FLORENCE_MODEL,
        dtype=torch.float16,
        device_map="auto",
        cache_dir=CACHE
    )
)


f_model.eval()


def run_florence(phrase):

    task = (
        "<CAPTION_TO_PHRASE_GROUNDING>"
    )

    inputs = f_processor(
        text=task + phrase,
        images=clean_pil,
        return_tensors="pt"
    )


    inputs = {

        k:
            v.to(
                f_model.device
            )
            if hasattr(
                v,
                "to"
            )
            else v

        for k, v in inputs.items()
    }


    with torch.inference_mode():

        ids = f_model.generate(
            **inputs,
            max_new_tokens=256,
            num_beams=3,
            do_sample=False
        )


    generated = (
        f_processor
        .batch_decode(
            ids,
            skip_special_tokens=False
        )[0]
    )


    parsed = (
        f_processor
        .post_process_generation(
            generated,
            task=task,
            image_size=clean_pil.size
        )
    )


    return extract_florence_boxes(
        parsed
    )


localized_standalone = []


for row in normalized_standalone:

    sid = row[
        "standalone_id"
    ]

    name = row[
        "name"
    ]

    phrase = row[
        "grounding_phrase"
    ]


    print()
    print("=" * 100)
    print(
        f"S{sid:02d}. {name}"
    )
    print("=" * 100)


    boxes = run_florence(
        phrase
    )


    normalized_boxes = []


    for box_row in boxes:

        normalized_boxes.append({

            "bbox":
                clamp_box(
                    box_row[
                        "bbox"
                    ]
                ),

            "label":
                box_row[
                    "label"
                ],

            "score":
                box_row[
                    "score"
                ],
        })


    unique = dedup_boxes(
        normalized_boxes
    )


    for idx, item in enumerate(
        unique,
        start=1
    ):

        item[
            "candidate_index"
        ] = idx


    print(
        "CANDIDATES:",
        len(unique)
    )


    for item in unique:

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


    preview = clean_pil.copy()

    draw = ImageDraw.Draw(
        preview
    )


    for item in unique:

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
        f'S{sid:02d}_{re.sub(r"[^A-Za-z0-9_-]+", "_", name)}.png'
    )


    preview.save(
        preview_path
    )


    localized_standalone.append({

        **row,

        "candidate_count":
            len(unique),

        "candidates":
            unique,

        "preview_path":
            str(
                preview_path
            ),
    })


# ============================================================
# GLOBAL STANDALONE OVERVIEW
# ============================================================

overview = clean_pil.copy()

draw = ImageDraw.Draw(
    overview
)


for row in localized_standalone:

    sid = row[
        "standalone_id"
    ]


    for item in row[
        "candidates"
    ]:

        x1, y1, x2, y2 = (
            item[
                "bbox"
            ]
        )


        label = (
            f'S{sid}:'
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
    / "01_standalone_prop_candidates_overview.png"
)


overview.save(
    OVERVIEW_PATH
)


# ============================================================
# SAVE FINAL AUDIT STATE
# ============================================================

result = {

    "stage":
        "06D1B",

    "detection_input":
        str(
            DETECTION_MASTER
        ),

    "group_updates":
        normalized_updates,

    "corrected_connected_groups":
        corrected_groups,

    "standalone_props":
        localized_standalone,

    "status":
        "REQUIRES_VISUAL_COMPLETENESS_AUDIT",

    "rules": [
        "existing connected groups may gain omitted touching members",
        "standalone fixtures remain independent props",
        "architectural surfaces are excluded",
        "no SAM2 was run",
        "no final prop mask union was created",
    ],
}


RESULT_PATH = (
    OUT
    / "00_stage06d1b_result.json"
)


RESULT_PATH.write_text(
    json.dumps(
        result,
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
print("PRODUCTION STAGE 06D1B RESULT")
print("=" * 110)


print()
print("CORRECTED CONNECTED GROUPS:")


for group in corrected_groups:

    print()

    print(
        "G{:02d}. {}".format(
            group[
                "group_id"
            ],
            group[
                "group_name"
            ]
        )
    )

    print(
        "  MEMBERS:",
        group[
            "members"
        ]
    )

    print(
        "  ADDED:",
        group[
            "members_added_by_06D1B"
        ]
    )


print()
print("STANDALONE PROPS:")


for row in localized_standalone:

    print()

    print(
        "S{:02d}. {} | candidates={}".format(

            row[
                "standalone_id"
            ],

            row[
                "name"
            ],

            row[
                "candidate_count"
            ],
        )
    )

    for item in row[
        "candidates"
    ]:

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


print()
print(
    "RESULT JSON:",
    RESULT_PATH
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
print("INLINE 06D1B COMPLETENESS AUDIT")
print("=" * 110)


show(
    OVERVIEW_PATH,
    (
        "06D1B — ALL NEW STANDALONE PROP "
        "LOCALIZATION CANDIDATES"
    ),
    figsize=(8, 9)
)


for row in localized_standalone:

    show(
        row[
            "preview_path"
        ],
        (
            f'S{row["standalone_id"]:02d} — '
            f'{row["name"]} | Florence Candidates'
        ),
        figsize=(8, 9)
    )


# ============================================================
# CLEANUP
# ============================================================

del f_model
del f_processor

gc.collect()

if torch.cuda.is_available():

    torch.cuda.empty_cache()
