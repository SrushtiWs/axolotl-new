
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
    / "test07"
    / "production_pipeline"
)

STAGE06 = (
    PROD
    / "stage06_prop_layer"
)

MASTER_PATH = (
    PROD
    / "stage05_clean_room_with_props"
    / "12_final_canonical_prop_detection_master.png"
)

EVIDENCE_PATH = (
    STAGE06
    / "06d1d3_crop_reverify_structural_cleanup"
    / "00_stage06d1d3_result.json"
)

OUT = (
    STAGE06
    / "06d2a_final_connected_main_prop_inventory"
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
    MASTER_PATH,
    EVIDENCE_PATH,
]:

    if not path.exists():

        raise FileNotFoundError(
            path
        )


master = Image.open(
    MASTER_PATH
).convert(
    "RGB"
)


evidence_state = json.loads(
    EVIDENCE_PATH.read_text(
        encoding="utf-8"
    )
)


physical_evidence = (
    evidence_state[
        "physical_object_evidence"
    ]
)


# ============================================================
# HELPERS
# ============================================================

def show(
    image,
    title,
    figsize=(8, 9)
):

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


def parse_json_object(
    text
):

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
        "Could not parse Stage06D2A JSON."
    )


def get_final_name(
    row
):

    return str(

        row.get(
            "final_name"
        )

        or

        row.get(
            "corrected_name"
        )

        or

        row.get(
            "name"
        )

        or

        ""
    ).strip()


# ============================================================
# BUILD COMPLETENESS-EVIDENCE CONCEPTS
#
# We intentionally collapse repeated observation NAMES here.
#
# We are NOT deduplicating physical instances.
# We are only building a semantic completeness checklist for
# the final main-prop grouping model.
# ============================================================

concept_counter = Counter()


for row in physical_evidence:

    name = get_final_name(
        row
    )

    if not name:
        continue

    concept_counter[
        name.lower()
    ] += 1


concepts = [

    {
        "concept":
            name,

        "observation_count":
            count,
    }

    for name, count
    in sorted(
        concept_counter.items()
    )
]


# ============================================================
# PRINT EVIDENCE CHECKLIST
# ============================================================

print("=" * 110)
print("PRODUCTION STAGE 06D2A")
print("FINAL CONNECTED MAIN-PROP INVENTORY")
print("=" * 110)

print()
print(
    "CLEAN PHYSICAL EVIDENCE RECORDS:",
    len(
        physical_evidence
    )
)

print(
    "UNIQUE COMPLETENESS CONCEPTS:",
    len(
        concepts
    )
)


print()
print("=" * 110)
print("COMPLETENESS EVIDENCE — NOT FINAL PROP TARGETS")
print("=" * 110)


for row in concepts:

    print(
        "{:2d} × {}".format(

            row[
                "observation_count"
            ],

            row[
                "concept"
            ],
        )
    )


# ============================================================
# QWEN FINAL MAIN-PROP REASONING
# ============================================================

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


PROMPT = f"""
You are defining the FINAL MAIN-PROP inventory for a room
foreground-extraction pipeline.

You are looking at a CLEAN room image.

The mirror and transparent glass enclosure have already been
removed intentionally by earlier pipeline stages.

The following terms come from multiple high-recall discovery
passes.

They are ONLY COMPLETENESS EVIDENCE.

They are NOT automatically separate final props.

COMPLETENESS EVIDENCE:

{json.dumps(concepts, indent=2)}

============================================================
CRITICAL FINAL GROUPING RULE
============================================================

A final extraction target is a CONNECTED MAIN PROP.

If multiple NON-ARCHITECTURAL physical items visibly:

- touch each other,
- rest directly on each other,
- are attached to each other,
- are integrated into one another,
- mechanically connect,
- overlap/contact physically,
- or form one continuous visible physical assembly,

they MUST become ONE final main prop.

Do NOT produce separate final prop entries for children,
components, accessories, or objects resting directly on a
larger parent prop.

For example:

cabinet
+ drawer
+ cabinet handles
+ countertop
+ basin/sink
+ sink faucet
+ bottle sitting directly on countertop

must become ONE complete connected furniture/fixture system,
not separate extraction targets.

Similarly:

toilet body
+ seat
+ lid

must become ONE complete toilet system.

============================================================
ARCHITECTURAL SURFACE EXCEPTION
============================================================

Wall, floor and ceiling are NOT props.

A fixture mounted on an architectural surface remains a prop,
but it does NOT merge with the wall/ceiling itself.

Therefore a visible:
- electrical plate
- recessed ceiling light
- wall-mounted holder
- shower fixture

may remain independent main props if they do not physically
connect to another non-architectural prop.

============================================================
SPECIAL OWNERSHIP
============================================================

Mirror:
owned by Stage03.
Do not include.

Glass enclosure and hardware belonging to the glass enclosure:
owned by Stage04.
Do not include.

============================================================
FALSE / NOISY DISCOVERY TERMS
============================================================

Some evidence labels may be inaccurate because they were
generated by high-recall crop discovery.

Examples could include:
- a wrong object name,
- a duplicate description,
- a room edge called an object,
- a child component named independently.

Use the IMAGE as the final authority.

Do not create a main prop merely because a noisy evidence
term exists.

============================================================
OUTPUT REQUIREMENT
============================================================

Return the final visible CONNECTED MAIN-PROP inventory.

For each main prop give:

- main_prop_id
- main_prop_name
- visible_members
- completeness_evidence
- grounding_phrase
- grouping_reason
- prop_type

prop_type must be one of:

CONNECTED_SYSTEM
STANDALONE_FIXTURE

Also return rejected/noisy completeness concepts separately.

Return ONLY valid JSON:

{{
  "main_props": [
    {{
      "main_prop_id": "P01",
      "main_prop_name": "generic complete prop name",
      "prop_type": "CONNECTED_SYSTEM",
      "visible_members": [
        "visible component",
        "visible component"
      ],
      "completeness_evidence": [
        "concept from evidence list"
      ],
      "grounding_phrase": "precise description of the COMPLETE visible prop system",
      "grouping_reason": "why these components must be extracted as one prop"
    }}
  ],

  "rejected_or_noise_concepts": [
    {{
      "concept": "evidence concept",
      "reason": "why it should not become a final prop"
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
                        MASTER_PATH
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
        max_new_tokens=1800,
        do_sample=False,
        repetition_penalty=1.05,
    )


generated_trimmed = [

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
        generated_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
)


RAW_PATH = (
    OUT
    / "00_qwen_final_main_prop_inventory_raw.txt"
)


RAW_PATH.write_text(
    raw_text,
    encoding="utf-8"
)


print()
print("=" * 110)
print("RAW QWEN MAIN-PROP INVENTORY")
print("=" * 110)

print(
    raw_text
)


parsed = parse_json_object(
    raw_text
)


main_props = parsed.get(
    "main_props",
    []
)


noise = parsed.get(
    "rejected_or_noise_concepts",
    []
)


# ============================================================
# NORMALIZE MAIN PROP IDS
#
# Do not trust generated numbering.
# ============================================================

normalized_props = []


for idx, prop in enumerate(
    main_props,
    start=1
):

    if not isinstance(
        prop,
        dict
    ):

        continue


    name = str(
        prop.get(
            "main_prop_name",
            ""
        )
    ).strip()


    if not name:

        continue


    prop_type = str(
        prop.get(
            "prop_type",
            ""
        )
    ).strip().upper()


    if prop_type not in {

        "CONNECTED_SYSTEM",

        "STANDALONE_FIXTURE",

    }:

        prop_type = (
            "CONNECTED_SYSTEM"
        )


    members = prop.get(
        "visible_members",
        []
    )


    if isinstance(
        members,
        str
    ):

        members = [
            members
        ]


    evidence_terms = prop.get(
        "completeness_evidence",
        []
    )


    if isinstance(
        evidence_terms,
        str
    ):

        evidence_terms = [
            evidence_terms
        ]


    normalized_props.append({

        "main_prop_id":
            f"P{len(normalized_props) + 1:02d}",

        "main_prop_name":
            name,

        "prop_type":
            prop_type,

        "visible_members":
            [
                str(x).strip()
                for x in members
                if str(x).strip()
            ],

        "completeness_evidence":
            [
                str(x).strip()
                for x in evidence_terms
                if str(x).strip()
            ],

        "grounding_phrase":
            str(
                prop.get(
                    "grounding_phrase",
                    name
                )
            ).strip(),

        "grouping_reason":
            str(
                prop.get(
                    "grouping_reason",
                    ""
                )
            ).strip(),
    })


# ============================================================
# SAVE FINAL INVENTORY STATE
# ============================================================

FINAL_STATE = {

    "stage":
        "06D2A",

    "detection_input":
        str(
            MASTER_PATH
        ),

    "main_prop_count":
        len(
            normalized_props
        ),

    "main_props":
        normalized_props,

    "rejected_or_noise_concepts":
        noise,

    "source_evidence": {

        "stage":
            "06D1D3",

        "physical_evidence_records":
            len(
                physical_evidence
            ),

        "unique_completeness_concepts":
            concepts,
    },

    "production_rule":
        (
            "All physically touching, resting, attached, "
            "integrated or mechanically connected "
            "non-architectural items are extracted once "
            "as one complete main prop."
        ),

    "status":
        "REQUIRES_FINAL_MAIN_PROP_INVENTORY_AUDIT",

    "next_stage":
        "06D2B_MAIN_PROP_LOCALIZATION",

    "rules": [

        "individual discovery observations are no longer final extraction targets",

        "child components must not become independent final layers",

        "main prop extraction occurs once per complete connected assembly",

        "wall floor ceiling remain architecture",

        "mirror remains Stage03-owned",

        "glass enclosure remains Stage04-owned",

        "Stage05F remains detection image",

        "Stage01 remains final RGB source",

        "no localization performed",

        "no SAM2 performed",

        "no mask union performed",
    ],
}


RESULT_PATH = (
    OUT
    / "00_stage06d2a_final_main_prop_inventory.json"
)


RESULT_PATH.write_text(
    json.dumps(
        FINAL_STATE,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# ============================================================
# PRINT FINAL RESULT
# ============================================================

print()
print("=" * 110)
print("PRODUCTION STAGE 06D2A RESULT")
print("=" * 110)

print()
print(
    "FINAL MAIN PROPS:",
    len(
        normalized_props
    )
)


for prop in normalized_props:

    print()
    print(
        "{}. {} [{}]".format(

            prop[
                "main_prop_id"
            ],

            prop[
                "main_prop_name"
            ],

            prop[
                "prop_type"
            ],
        )
    )

    print(
        "  MEMBERS:",
        prop[
            "visible_members"
        ]
    )

    print(
        "  EVIDENCE:",
        prop[
            "completeness_evidence"
        ]
    )

    print(
        "  GROUNDING:",
        prop[
            "grounding_phrase"
        ]
    )

    print(
        "  WHY:",
        prop[
            "grouping_reason"
        ]
    )


print()
print("=" * 110)
print("REJECTED / NOISE CONCEPTS")
print("=" * 110)


if not noise:

    print(
        "NONE"
    )


for row in noise:

    print(
        "{} | {}".format(

            row.get(
                "concept"
            ),

            row.get(
                "reason"
            ),
        )
    )


print()
print(
    "RESULT JSON:",
    RESULT_PATH
)

print()
print(
    "NO INDIVIDUAL PROP EXTRACTION WAS PERFORMED."
)

print(
    "NO FLORENCE / DINO / SAM2 WAS RUN."
)

print(
    "NO PROP MASK WAS CREATED."
)


# ============================================================
# INLINE VISUAL CONTEXT
# ============================================================

print()
print("=" * 110)
print("INLINE 06D2A FINAL MAIN-PROP INVENTORY AUDIT")
print("=" * 110)


show(
    master,
    (
        "06D2A — Stage05F Canonical Detection Master\n"
        "Review FINAL MAIN-PROP INVENTORY Against This Image"
    ),
)


# ============================================================
# CLEANUP
# ============================================================

del model
del processor

gc.collect()

if torch.cuda.is_available():

    torch.cuda.empty_cache()
