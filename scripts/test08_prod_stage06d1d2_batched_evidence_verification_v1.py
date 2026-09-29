
from pathlib import Path
import json
import re
import gc

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

MASTER = (
    PROD
    / "stage05_clean_room_with_props"
    / "12_final_canonical_prop_detection_master.png"
)

OBS3 = (
    STAGE06
    / "06d1c_tiled_high_recall_prop_discovery"
    / "01_raw_tiled_prop_observations.json"
)

OBS4 = (
    STAGE06
    / "06d1c2b_recovered_4x4_observations"
    / "01_recovered_4x4_observations.json"
)

OUT = (
    STAGE06
    / "06d1d2_batched_evidence_verification"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

RAW_DIR = (
    OUT
    / "raw_batches"
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

BATCH_SIZE = 8

MAX_NEW_TOKENS = 1000

MAX_RETRIES = 2


# ============================================================
# LOAD
# ============================================================

for path in [
    MASTER,
    OBS3,
    OBS4,
]:

    if not path.exists():

        raise FileNotFoundError(
            path
        )


image = Image.open(
    MASTER
).convert(
    "RGB"
)


obs3 = json.loads(
    OBS3.read_text(
        encoding="utf-8"
    )
)


obs4 = json.loads(
    OBS4.read_text(
        encoding="utf-8"
    )
)


# ============================================================
# NORMALIZE EVIDENCE
# ============================================================

evidence = []


for row in obs3:

    evidence.append({

        "evidence_id":
            f'3X3_{int(row["observation_id"]):03d}',

        "pass":
            "3X3",

        "name":
            str(
                row.get(
                    "name",
                    ""
                )
            ),

        "grounding_phrase":
            str(
                row.get(
                    "grounding_phrase",
                    ""
                )
            ),

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

    evidence.append({

        "evidence_id":
            f'4X4_{int(row["observation_id"]):03d}',

        "pass":
            "4X4",

        "name":
            str(
                row.get(
                    "name",
                    ""
                )
            ),

        "grounding_phrase":
            str(
                row.get(
                    "grounding_phrase",
                    ""
                )
            ),

        "confidence":
            float(
                row.get(
                    "confidence",
                    0.5
                )
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


evidence_by_id = {

    row[
        "evidence_id"
    ]:
        row

    for row in evidence
}


# ============================================================
# JSON PARSER
# ============================================================

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

        obj = json.loads(
            text
        )

        if isinstance(
            obj,
            list
        ):
            return obj

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

            obj = json.loads(
                text[
                    s:e + 1
                ]
            )

            if isinstance(
                obj,
                list
            ):
                return obj

        except Exception:
            pass


    return []


# ============================================================
# LOAD QWEN
# ============================================================

print("=" * 110)
print("PRODUCTION STAGE 06D1D2")
print("BATCHED EVIDENCE VERIFICATION")
print("=" * 110)

print()
print(
    "TOTAL EVIDENCE:",
    len(
        evidence
    )
)

print(
    "3X3:",
    len(
        obs3
    )
)

print(
    "4X4:",
    len(
        obs4
    )
)

print(
    "BATCH SIZE:",
    BATCH_SIZE
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
# VERIFY ONE BATCH
# ============================================================

def verify_batch(
    batch,
    batch_number,
):

    batch_payload = [

        {
            "evidence_id":
                row[
                    "evidence_id"
                ],

            "name":
                row[
                    "name"
                ],

            "grounding_phrase":
                row[
                    "grounding_phrase"
                ],

            "source_area":
                (
                    row.get(
                        "source_region"
                    )
                    or
                    row.get(
                        "source_tile"
                    )
                ),
        }

        for row in batch
    ]


    required_ids = {

        row[
            "evidence_id"
        ]

        for row in batch
    }


    PROMPT = f"""
You are verifying a SMALL BATCH of high-recall object
observations against this clean room image.

IMPORTANT:
The discovery labels can be wrong.
Judge what is actually visible in the image.

These exact evidence records must ALL be classified:

{json.dumps(batch_payload, indent=2)}

For EACH evidence_id choose exactly one status:

PHYSICAL_OBJECT
- There is a real discrete physical object or fixture
  corresponding to the observation.

STRUCTURAL_SURFACE
- The observation is actually wall, floor, ceiling, baseboard,
  room corner, paint, tile, or another architectural surface.

HALLUCINATION_OR_WRONG
- No corresponding discrete physical object is visibly
  supported by the image.

If the original name is wrong but it points to a REAL visible
physical object, use PHYSICAL_OBJECT and supply the corrected
generic physical-object name.

Examples:
- something called "faucet" that is visibly a shower head:
  PHYSICAL_OBJECT, corrected_name="shower head"
- something called "toilet tank" when no toilet tank is
  visible:
  HALLUCINATION_OR_WRONG

A physical object mounted on a wall or ceiling is still a
PHYSICAL_OBJECT.

Do not reject:
- wall switches
- electrical plates
- recessed ceiling lights
- holders
- shower heads
simply because they touch an architectural surface.

Mirror and glass enclosure were intentionally removed.

CRITICAL:
Return every evidence_id from this batch EXACTLY ONCE.
Do not omit any ID.
Do not invent new IDs.

Return ONLY JSON:

[
  {{
    "evidence_id": "3X3_001",
    "status": "PHYSICAL_OBJECT",
    "corrected_name": "generic physical object name",
    "reason": "brief visible-image reason"
  }}
]
"""


    for attempt in range(
        1,
        MAX_RETRIES + 2
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
                            str(
                                MASTER
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

            ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
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


        raw_path = (
            RAW_DIR
            /
            f"batch_{batch_number:02d}_attempt_{attempt}.txt"
        )


        raw_path.write_text(
            raw,
            encoding="utf-8"
        )


        parsed = parse_json_array(
            raw
        )


        returned_ids = [

            str(
                row.get(
                    "evidence_id",
                    ""
                )
            )

            for row in parsed

            if isinstance(
                row,
                dict
            )
        ]


        returned_set = set(
            returned_ids
        )


        complete = bool(

            returned_set
            ==
            required_ids

            and

            len(
                returned_ids
            )
            ==
            len(
                required_ids
            )
        )


        print(
            f"Batch {batch_number:02d} "
            f"attempt {attempt} | "
            f"returned={len(returned_ids)}/{len(required_ids)} "
            f"| complete={complete}"
        )


        if complete:

            return parsed


    raise RuntimeError(
        f"Batch {batch_number} could not achieve "
        f"complete evidence accountability."
    )


# ============================================================
# RUN BATCHES
# ============================================================

verified_rows = []


for start in range(
    0,
    len(evidence),
    BATCH_SIZE
):

    batch = evidence[
        start:
        start + BATCH_SIZE
    ]


    batch_number = (
        start
        //
        BATCH_SIZE
        +
        1
    )


    print()
    print("=" * 90)

    print(
        f"BATCH {batch_number:02d}"
    )

    print("=" * 90)


    rows = verify_batch(
        batch,
        batch_number
    )


    verified_rows.extend(
        rows
    )


# ============================================================
# GLOBAL ACCOUNTABILITY
# ============================================================

all_input_ids = {

    row[
        "evidence_id"
    ]

    for row in evidence
}


all_output_ids = [

    str(
        row.get(
            "evidence_id"
        )
    )

    for row in verified_rows
]


missing = sorted(
    all_input_ids
    -
    set(
        all_output_ids
    )
)


duplicates = sorted({

    evidence_id

    for evidence_id
    in all_output_ids

    if all_output_ids.count(
        evidence_id
    ) > 1
})


unknown = sorted(
    set(
        all_output_ids
    )
    -
    all_input_ids
)


accountability_pass = bool(

    not missing
    and
    not duplicates
    and
    not unknown
    and
    len(
        all_output_ids
    )
    ==
    len(
        all_input_ids
    )
)


# ============================================================
# ENRICH WITH ORIGINAL EVIDENCE
# ============================================================

enriched = []


VALID_STATUSES = {

    "PHYSICAL_OBJECT",

    "STRUCTURAL_SURFACE",

    "HALLUCINATION_OR_WRONG",
}


for row in verified_rows:

    evidence_id = str(
        row[
            "evidence_id"
        ]
    )


    original = evidence_by_id[
        evidence_id
    ]


    status = str(
        row.get(
            "status",
            ""
        )
    ).strip().upper()


    if status not in VALID_STATUSES:

        status = (
            "HALLUCINATION_OR_WRONG"
        )


    corrected_name = str(
        row.get(
            "corrected_name",
            original[
                "name"
            ]
        )
    ).strip()


    if not corrected_name:

        corrected_name = original[
            "name"
        ]


    enriched.append({

        **original,

        "verification_status":
            status,

        "corrected_name":
            corrected_name,

        "verification_reason":
            str(
                row.get(
                    "reason",
                    ""
                )
            ).strip(),
    })


# ============================================================
# SPLIT
# ============================================================

physical = [

    row

    for row in enriched

    if row[
        "verification_status"
    ]
    ==
    "PHYSICAL_OBJECT"
]


structural = [

    row

    for row in enriched

    if row[
        "verification_status"
    ]
    ==
    "STRUCTURAL_SURFACE"
]


hallucinated = [

    row

    for row in enriched

    if row[
        "verification_status"
    ]
    ==
    "HALLUCINATION_OR_WRONG"
]


# ============================================================
# SAVE
# ============================================================

RESULT = {

    "stage":
        "06D1D2",

    "input_evidence_count":
        len(
            evidence
        ),

    "accountability": {

        "pass":
            accountability_pass,

        "missing_ids":
            missing,

        "duplicate_ids":
            duplicates,

        "unknown_ids":
            unknown,
    },

    "physical_object_evidence":
        physical,

    "structural_surface_evidence":
        structural,

    "hallucination_or_wrong_evidence":
        hallucinated,

    "counts": {

        "physical":
            len(
                physical
            ),

        "structural":
            len(
                structural
            ),

        "hallucinated_or_wrong":
            len(
                hallucinated
            ),
    },

    "status":
        (
            "READY_FOR_PHYSICAL_INSTANCE_DEDUP"
            if accountability_pass
            else
            "ACCOUNTABILITY_FAIL"
        ),

    "rules": [
        "no evidence is silently discarded",
        "wrong semantic labels may be corrected",
        "no touching/connected grouping performed yet",
        "no Florence",
        "no SAM2",
        "no final mask union",
    ],
}


RESULT_PATH = (
    OUT
    / "00_stage06d1d2_result.json"
)


RESULT_PATH.write_text(
    json.dumps(
        RESULT,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


PHYSICAL_PATH = (
    OUT
    / "01_verified_physical_evidence.json"
)


PHYSICAL_PATH.write_text(
    json.dumps(
        physical,
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
print("PRODUCTION STAGE 06D1D2 RESULT")
print("=" * 110)


print()
print(
    "ACCOUNTABILITY PASS:",
    accountability_pass
)

print(
    "MISSING:",
    missing
)

print(
    "DUPLICATES:",
    duplicates
)

print(
    "UNKNOWN:",
    unknown
)


print()
print(
    "PHYSICAL OBJECT EVIDENCE:",
    len(
        physical
    )
)

print(
    "STRUCTURAL SURFACE:",
    len(
        structural
    )
)

print(
    "HALLUCINATION / WRONG:",
    len(
        hallucinated
    )
)


print()
print("=" * 110)
print("VERIFIED PHYSICAL OBJECT EVIDENCE")
print("=" * 110)


for row in physical:

    print(
        "{:9s} | original={:22s} | corrected={:24s} | {}".format(

            row[
                "evidence_id"
            ],

            row[
                "name"
            ][:22],

            row[
                "corrected_name"
            ][:24],

            row[
                "verification_reason"
            ],
        )
    )


print()
print("=" * 110)
print("STRUCTURAL REJECTS")
print("=" * 110)


for row in structural:

    print(
        "{:9s} | {:22s} | {}".format(

            row[
                "evidence_id"
            ],

            row[
                "name"
            ][:22],

            row[
                "verification_reason"
            ],
        )
    )


print()
print("=" * 110)
print("HALLUCINATION / WRONG REJECTS")
print("=" * 110)


for row in hallucinated:

    print(
        "{:9s} | {:22s} | {}".format(

            row[
                "evidence_id"
            ],

            row[
                "name"
            ][:22],

            row[
                "verification_reason"
            ],
        )
    )


print()
print(
    "RESULT JSON:",
    RESULT_PATH
)

print(
    "PHYSICAL EVIDENCE JSON:",
    PHYSICAL_PATH
)

print()
print(
    "NO DEDUPLICATION OR CONNECTED GROUPING WAS RUN."
)

print(
    "NO FLORENCE OR SAM2 WAS RUN."
)


# ============================================================
# INLINE VISUAL CONTEXT
# ============================================================

print()
print("=" * 110)
print("INLINE 06D1D2 VISUAL CONTEXT")
print("=" * 110)


plt.figure(
    figsize=(8, 9)
)

plt.imshow(
    image
)

plt.title(
    "06D1D2 — Stage05F Detection Master"
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
