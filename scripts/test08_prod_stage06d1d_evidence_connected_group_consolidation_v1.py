
from pathlib import Path
import json
import re
import gc
from collections import Counter

import torch
from PIL import Image
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

STAGE06 = (
    PROD
    / "stage06_prop_layer"
)

DETECTION_MASTER = (
    PROD
    / "stage05_clean_room_with_props"
    / "12_final_canonical_prop_detection_master.png"
)

OBS_3X3 = (
    STAGE06
    / "06d1c_tiled_high_recall_prop_discovery"
    / "01_raw_tiled_prop_observations.json"
)

OBS_4X4 = (
    STAGE06
    / "06d1c2b_recovered_4x4_observations"
    / "01_recovered_4x4_observations.json"
)

OUT = (
    STAGE06
    / "06d1d_evidence_connected_group_consolidation"
)

OUT.mkdir(
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
# VALIDATE
# ============================================================

for path in [
    DETECTION_MASTER,
    OBS_3X3,
    OBS_4X4,
]:

    if not path.exists():

        raise FileNotFoundError(
            path
        )


image = Image.open(
    DETECTION_MASTER
).convert(
    "RGB"
)


obs3 = json.loads(
    OBS_3X3.read_text(
        encoding="utf-8"
    )
)


obs4 = json.loads(
    OBS_4X4.read_text(
        encoding="utf-8"
    )
)


# ============================================================
# NORMALIZE OBSERVATIONS
# ============================================================

observations = []


for row in obs3:

    observations.append({

        "evidence_id":
            f'3X3_{int(row["observation_id"]):03d}',

        "discovery_pass":
            "3X3",

        "name":
            str(
                row.get(
                    "name",
                    ""
                )
            ).strip(),

        "grounding_phrase":
            str(
                row.get(
                    "grounding_phrase",
                    ""
                )
            ).strip(),

        "confidence":
            float(
                row.get(
                    "confidence",
                    0.5
                )
            ),

        "source_region":
            row.get(
                "source_region"
            ),

        "source_bbox":
            row.get(
                "source_bbox"
            ),
    })


for row in obs4:

    observations.append({

        "evidence_id":
            f'4X4_{int(row["observation_id"]):03d}',

        "discovery_pass":
            "4X4",

        "name":
            str(
                row.get(
                    "name",
                    ""
                )
            ).strip(),

        "grounding_phrase":
            str(
                row.get(
                    "grounding_phrase",
                    ""
                )
            ).strip(),

        "confidence":
            float(
                row.get(
                    "confidence",
                    0.5
                )
            ),

        "source_tile_id":
            row.get(
                "source_tile_id"
            ),

        "source_tile":
            row.get(
                "source_tile"
            ),

        "source_bbox":
            row.get(
                "source_bbox"
            ),
    })


# ============================================================
# DETERMINISTIC OBVIOUS-SURFACE FLAG
#
# IMPORTANT:
# These are NOT automatically deleted.
# They are merely flagged to help the consolidation model.
# The final output must still explicitly account for them.
# ============================================================

STRUCTURAL_TERMS = {

    "wall",
    "floor",
    "ceiling",
    "corner",
    "baseboard",
    "wall surface",
    "floor surface",
    "ceiling surface",
}


for row in observations:

    normalized_name = (
        row[
            "name"
        ]
        .strip()
        .lower()
    )

    row[
        "structural_term_flag"
    ] = (
        normalized_name
        in STRUCTURAL_TERMS
    )


# ============================================================
# COMPACT EVIDENCE TEXT
# ============================================================

evidence_lines = []


for row in observations:

    evidence_lines.append(

        "{} | pass={} | name={} | phrase={} | conf={:.2f} | structural_flag={}".format(

            row[
                "evidence_id"
            ],

            row[
                "discovery_pass"
            ],

            row[
                "name"
            ],

            row[
                "grounding_phrase"
            ],

            row[
                "confidence"
            ],

            row[
                "structural_term_flag"
            ],
        )
    )


evidence_text = "\n".join(
    evidence_lines
)


# ============================================================
# PROMPT
# ============================================================

PROMPT = f"""
You are consolidating HIGH-RECALL physical-object evidence
from two overlapping crop-based discovery passes of the SAME
clean room image.

The image has already had mirror and transparent glass
enclosure removed.

The discovery passes intentionally favor recall, so the
evidence contains:
- duplicate descriptions of the same physical object,
- partial observations,
- incorrect semantic labels,
- architectural surfaces,
- and occasional hallucinations.

Your job is to infer the ACTUAL VISIBLE PHYSICAL PROP STATE
from the image AND the evidence.

============================================================
RAW EVIDENCE
============================================================

{evidence_text}

============================================================
STEP 1 — PHYSICAL INSTANCE CONSOLIDATION
============================================================

Merge evidence records only when they refer to the same actual
visible physical object.

Do NOT merge two distinct physical objects simply because they
have the same semantic class.

A discovery label can be wrong. Trust the visible image more
than the label text.

============================================================
STEP 2 — CONNECTED MAIN-PROP GROUPING
============================================================

After determining actual visible physical instances, apply
this production rule:

If NON-ARCHITECTURAL physical objects visibly:
- touch each other,
- rest on each other,
- overlap/contact each other,
- attach to each other,
- are mechanically connected,
- or form one continuous physical assembly,

they belong to ONE connected main-prop group.

A movable object resting directly on another prop counts as
part of that connected group in THIS image.

Examples of the rule:
- furniture + countertop + basin + attached tap + bottle
  resting on countertop + drawers/handles
  -> ONE connected group
- toilet body + seat/lid
  -> ONE connected group

============================================================
ARCHITECTURAL-SURFACE EXCEPTION
============================================================

Wall, floor, ceiling and baseboard are architectural surfaces.

A physical fixture does NOT merge with wall/floor/ceiling just
because it is mounted to that surface.

Therefore:
- wall electrical plate = standalone physical prop
- ceiling light = standalone physical prop
- wall-mounted holder = standalone physical prop
- wall-mounted shower fixture = standalone physical prop

Do NOT create wall/floor/ceiling prop groups.

============================================================
SPECIAL LAYER EXCEPTION
============================================================

Mirror is owned by Stage03.
Glass enclosure and hardware physically belonging to that
glass enclosure are owned by Stage04.

They are intentionally absent from this clean detection image
and must NOT be reconstructed or added.

============================================================
ACCOUNTABILITY REQUIREMENT
============================================================

EVERY evidence_id listed above MUST appear exactly once in one
of:

1. "used_evidence_ids" of a connected group,
2. "used_evidence_ids" of a standalone prop,
3. "rejected_evidence".

No evidence may disappear silently.

If an evidence item has a wrong label but points to a real
object, assign it to the actual physical object and explain
the correction.

============================================================
OUTPUT
============================================================

Return ONLY valid JSON:

{{
  "connected_groups": [
    {{
      "group_id": "G01",
      "group_name": "short generic main prop name",
      "members": [
        "actual visible physical member",
        "actual visible physical member"
      ],
      "group_grounding_phrase": "precise visual description of the entire connected group",
      "connection_reason": "why all members physically form one connected group",
      "used_evidence_ids": [
        "3X3_001",
        "4X4_002"
      ]
    }}
  ],

  "standalone_props": [
    {{
      "prop_id": "S01",
      "name": "short generic prop name",
      "grounding_phrase": "precise visual description",
      "reason": "why it is standalone",
      "used_evidence_ids": [
        "3X3_010"
      ]
    }}
  ],

  "rejected_evidence": [
    {{
      "evidence_id": "4X4_001",
      "reason": "architectural surface / hallucination / duplicate evidence already assigned / etc"
    }}
  ]
}}
"""


# ============================================================
# LOAD QWEN
# ============================================================

print("=" * 110)
print("PRODUCTION STAGE 06D1D")
print("EVIDENCE-ASSISTED CONNECTED-GROUP CONSOLIDATION")
print("=" * 110)

print()
print(
    "3X3 OBSERVATIONS:",
    len(obs3)
)

print(
    "4X4 OBSERVATIONS:",
    len(obs4)
)

print(
    "TOTAL EVIDENCE:",
    len(observations)
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
        max_new_tokens=2400,
        do_sample=False,
        repetition_penalty=1.04,
    )


trimmed = [

    out[
        len(inp):
    ]

    for inp, out
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


RAW_PATH = (
    OUT
    / "00_qwen_raw_consolidation.txt"
)


RAW_PATH.write_text(
    raw_text,
    encoding="utf-8"
)


print()
print("=" * 110)
print("RAW QWEN OUTPUT")
print("=" * 110)

print(
    raw_text
)


# ============================================================
# PARSE JSON
# ============================================================

def parse_json_object(text):

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

        obj = json.loads(
            text[
                start:end + 1
            ]
        )

        if isinstance(
            obj,
            dict
        ):
            return obj


    raise RuntimeError(
        "Could not parse 06D1D JSON."
    )


result = parse_json_object(
    raw_text
)


connected_groups = result.get(
    "connected_groups",
    []
)

standalone_props = result.get(
    "standalone_props",
    []
)

rejected = result.get(
    "rejected_evidence",
    []
)


# ============================================================
# ACCOUNTABILITY CHECK
# ============================================================

all_ids = {

    row[
        "evidence_id"
    ]

    for row in observations
}


accounted_ids = []


for group in connected_groups:

    accounted_ids.extend(
        group.get(
            "used_evidence_ids",
            []
        )
    )


for prop in standalone_props:

    accounted_ids.extend(
        prop.get(
            "used_evidence_ids",
            []
        )
    )


for row in rejected:

    if row.get(
        "evidence_id"
    ):

        accounted_ids.append(
            row[
                "evidence_id"
            ]
        )


counts = Counter(
    accounted_ids
)


missing_ids = sorted(
    all_ids
    -
    set(
        accounted_ids
    )
)


duplicate_accounting = sorted(

    evidence_id

    for evidence_id, count
    in counts.items()

    if count > 1
)


unknown_ids = sorted(
    set(
        accounted_ids
    )
    -
    all_ids
)


accountability_pass = bool(

    not missing_ids
    and
    not duplicate_accounting
    and
    not unknown_ids
)


# ============================================================
# SAVE
# ============================================================

final_state = {

    "stage":
        "06D1D",

    "detection_input":
        str(
            DETECTION_MASTER
        ),

    "input_evidence_count":
        len(
            observations
        ),

    "connected_groups":
        connected_groups,

    "standalone_props":
        standalone_props,

    "rejected_evidence":
        rejected,

    "accountability": {

        "pass":
            accountability_pass,

        "missing_evidence_ids":
            missing_ids,

        "duplicate_accounting":
            duplicate_accounting,

        "unknown_evidence_ids":
            unknown_ids,
    },

    "status":
        (
            "READY_FOR_VISUAL_GROUP_AUDIT"
            if accountability_pass
            else
            "ACCOUNTABILITY_FAIL"
        ),

    "rules": [
        "Stage05F is canonical detection image",
        "connected physical props become one main prop",
        "architectural surfaces never join prop groups",
        "mirror remains Stage03-owned",
        "glass system remains Stage04-owned",
        "no Florence was run",
        "no SAM2 was run",
        "no final prop mask union was created",
    ],
}


RESULT_PATH = (
    OUT
    / "01_stage06d1d_result.json"
)


RESULT_PATH.write_text(
    json.dumps(
        final_state,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# ============================================================
# PRINT
# ============================================================

print()
print("=" * 110)
print("PRODUCTION STAGE 06D1D RESULT")
print("=" * 110)


print()
print(
    "ACCOUNTABILITY PASS:",
    accountability_pass
)

print(
    "MISSING IDS:",
    missing_ids
)

print(
    "DUPLICATE ACCOUNTING:",
    duplicate_accounting
)

print(
    "UNKNOWN IDS:",
    unknown_ids
)


print()
print("=" * 110)
print("CONNECTED MAIN-PROP GROUPS")
print("=" * 110)


for group in connected_groups:

    print()

    print(
        "{}. {}".format(
            group.get(
                "group_id"
            ),
            group.get(
                "group_name"
            )
        )
    )

    print(
        "  MEMBERS:",
        group.get(
            "members"
        )
    )

    print(
        "  WHY:",
        group.get(
            "connection_reason"
        )
    )

    print(
        "  GROUNDING:",
        group.get(
            "group_grounding_phrase"
        )
    )

    print(
        "  EVIDENCE:",
        group.get(
            "used_evidence_ids"
        )
    )


print()
print("=" * 110)
print("STANDALONE PROPS")
print("=" * 110)


for prop in standalone_props:

    print()

    print(
        "{}. {}".format(
            prop.get(
                "prop_id"
            ),
            prop.get(
                "name"
            )
        )
    )

    print(
        "  GROUNDING:",
        prop.get(
            "grounding_phrase"
        )
    )

    print(
        "  WHY:",
        prop.get(
            "reason"
        )
    )

    print(
        "  EVIDENCE:",
        prop.get(
            "used_evidence_ids"
        )
    )


print()
print("=" * 110)
print("REJECTED EVIDENCE")
print("=" * 110)


for row in rejected:

    print(
        "{} | {}".format(
            row.get(
                "evidence_id"
            ),
            row.get(
                "reason"
            )
        )
    )


print()
print(
    "RESULT JSON:",
    RESULT_PATH
)

print()
print(
    "NO FLORENCE OR SAM2 WAS RUN."
)


# ============================================================
# INLINE IMAGE FOR FINAL GROUP REVIEW
# ============================================================

print()
print("=" * 110)
print("INLINE 06D1D VISUAL CONTEXT")
print("=" * 110)


plt.figure(
    figsize=(8, 9)
)

plt.imshow(
    image
)

plt.title(
    "06D1D — Stage05F Clean Detection Master"
)

plt.axis(
    "off"
)

plt.show()


# ============================================================
# CLEANUP
# ============================================================

del model
del processor

gc.collect()

if torch.cuda.is_available():

    torch.cuda.empty_cache()
