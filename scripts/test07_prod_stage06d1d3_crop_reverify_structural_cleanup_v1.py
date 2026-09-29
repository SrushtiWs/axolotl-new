
from pathlib import Path
import json
import re
import gc
from collections import Counter

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

MASTER = (
    PROD
    / "stage05_clean_room_with_props"
    / "12_final_canonical_prop_detection_master.png"
)

D1D2 = (
    STAGE06
    / "06d1d2_batched_evidence_verification"
    / "00_stage06d1d2_result.json"
)

OUT = (
    STAGE06
    / "06d1d3_crop_reverify_structural_cleanup"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

CROP_DIR = (
    OUT
    / "crops"
)

CROP_DIR.mkdir(
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

PREVIEW_DIR = (
    OUT
    / "previews"
)

PREVIEW_DIR.mkdir(
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

MAX_NEW_TOKENS = 420

CROP_EXPANSION = 0.05


# ============================================================
# DEFINITE ARCHITECTURAL TERMS
#
# These are deterministically rejected.
# Do NOT put ambiguous fixtures here.
# ============================================================

STRUCTURAL_TERMS = {

    "wall",
    "floor",
    "ceiling",
    "baseboard",
    "room corner",
    "corner",
    "wall surface",
    "floor surface",
    "ceiling surface",
}


# ============================================================
# ARCHITECTURAL OBJECTS THAT ARE NOT NORMAL PROPS
# ============================================================

ARCHITECTURAL_OBJECT_TERMS = {

    "door",
    "door frame",
}


# ============================================================
# SUSPICIOUS SEMANTIC TERMS
#
# These require source-crop verification.
# ============================================================

AMBIGUOUS_TERMS = {

    "faucet",
    "handle",
    "door handle",
    "wall-mounted object",
    "ceiling-mounted object",
    "toilet tank",
    "toilet base",
    "small object",
}


# ============================================================
# VALIDATE
# ============================================================

for path in [
    MASTER,
    D1D2,
]:
    if not path.exists():
        raise FileNotFoundError(path)


master = Image.open(
    MASTER
).convert("RGB")

W, H = master.size


d1d2 = json.loads(
    D1D2.read_text(
        encoding="utf-8"
    )
)


physical = d1d2[
    "physical_object_evidence"
]

previous_structural = d1d2[
    "structural_surface_evidence"
]

previous_wrong = d1d2[
    "hallucination_or_wrong_evidence"
]


# ============================================================
# HELPERS
# ============================================================

def show(
    image_or_path,
    title,
    figsize=(7, 7)
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


def normalize_name(name):

    return (
        str(name)
        .strip()
        .lower()
    )


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


def expand_box(
    box,
    ratio
):

    x1, y1, x2, y2 = box

    bw = max(
        1.0,
        x2 - x1
    )

    bh = max(
        1.0,
        y2 - y1
    )

    return clamp_box(
        [
            x1 - bw * ratio,
            y1 - bh * ratio,
            x2 + bw * ratio,
            y2 + bh * ratio,
        ]
    )


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
        obj = json.loads(text)

        if isinstance(obj, dict):
            return obj

    except Exception:
        pass


    start = text.find("{")
    end = text.rfind("}")

    if (
        start >= 0
        and
        end > start
    ):

        try:

            obj = json.loads(
                text[
                    start:end + 1
                ]
            )

            if isinstance(obj, dict):
                return obj

        except Exception:
            pass


    return None


# ============================================================
# PHASE A
# DETERMINISTIC CLASSIFICATION
# ============================================================

clean_physical = []

deterministic_structural = []

architectural_review = []

ambiguous = []


for row in physical:

    original_name = normalize_name(
        row.get(
            "name",
            ""
        )
    )

    corrected_name = normalize_name(
        row.get(
            "corrected_name",
            original_name
        )
    )


    # --------------------------------------------------------
    # DEFINITE STRUCTURAL
    # --------------------------------------------------------

    if (
        original_name in STRUCTURAL_TERMS
        or
        corrected_name in STRUCTURAL_TERMS
    ):

        new_row = dict(row)

        new_row[
            "final_status"
        ] = "STRUCTURAL_SURFACE"

        new_row[
            "final_reason"
        ] = (
            "Deterministic architectural-surface exclusion."
        )

        deterministic_structural.append(
            new_row
        )

        continue


    # --------------------------------------------------------
    # DOOR / ROOM ARCHITECTURE
    # --------------------------------------------------------

    if (
        original_name in ARCHITECTURAL_OBJECT_TERMS
        or
        corrected_name in ARCHITECTURAL_OBJECT_TERMS
    ):

        new_row = dict(row)

        new_row[
            "final_status"
        ] = "ARCHITECTURAL_OBJECT"

        new_row[
            "final_reason"
        ] = (
            "Door/door-frame belongs to room architecture, "
            "not the normal Stage06 prop layer."
        )

        architectural_review.append(
            new_row
        )

        continue


    # --------------------------------------------------------
    # AMBIGUOUS / CONTRADICTORY
    # --------------------------------------------------------

    phrase = str(
        row.get(
            "grounding_phrase",
            ""
        )
    ).lower()


    suspicious_phrase = bool(

        "on toilet" in phrase

        or

        "toilet tank" in phrase

        or

        "circular opening" in phrase

        or

        "small protrusion" in phrase
    )


    if (
        original_name in AMBIGUOUS_TERMS
        or
        corrected_name in AMBIGUOUS_TERMS
        or
        suspicious_phrase
    ):

        ambiguous.append(
            dict(row)
        )

        continue


    # --------------------------------------------------------
    # CLEAN PHYSICAL EVIDENCE
    # --------------------------------------------------------

    new_row = dict(row)

    new_row[
        "final_status"
    ] = "PHYSICAL_OBJECT"

    new_row[
        "final_name"
    ] = (
        row.get(
            "corrected_name"
        )
        or
        row.get(
            "name"
        )
    )

    new_row[
        "final_reason"
    ] = (
        "Passed 06D1D2 and no structural or ambiguity rule triggered."
    )

    clean_physical.append(
        new_row
    )


print("=" * 110)
print("PRODUCTION STAGE 06D1D3")
print("STRUCTURAL CLEANUP + SOURCE-CROP REVERIFICATION")
print("=" * 110)

print()
print(
    "CLEAN PHYSICAL WITHOUT REVERIFY:",
    len(clean_physical)
)

print(
    "DETERMINISTIC STRUCTURAL:",
    len(deterministic_structural)
)

print(
    "ARCHITECTURAL OBJECTS:",
    len(architectural_review)
)

print(
    "AMBIGUOUS TO REVERIFY:",
    len(ambiguous)
)


# ============================================================
# LOAD QWEN ONLY IF NEEDED
# ============================================================

reverified = []


if ambiguous:

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


    # ========================================================
    # VERIFY ONE AMBIGUOUS OBSERVATION USING SOURCE CROP
    # ========================================================

    for idx, row in enumerate(
        ambiguous,
        start=1
    ):

        evidence_id = row[
            "evidence_id"
        ]


        source_bbox = row.get(
            "source_bbox"
        )


        if (
            not source_bbox
            or
            len(source_bbox) != 4
        ):

            result = dict(row)

            result[
                "final_status"
            ] = "UNRESOLVED"

            result[
                "final_reason"
            ] = "Missing usable source crop bbox."

            reverified.append(
                result
            )

            continue


        crop_box = expand_box(
            source_bbox,
            CROP_EXPANSION
        )


        x1, y1, x2, y2 = [
            int(round(v))
            for v in crop_box
        ]


        crop = master.crop(
            (
                x1,
                y1,
                x2,
                y2
            )
        )


        crop_path = (
            CROP_DIR
            /
            f"{evidence_id}.png"
        )


        crop.save(
            crop_path
        )


        PROMPT = f"""
Inspect this LOCAL SOURCE CROP from a clean room image.

We are re-verifying ONE high-recall observation.

Evidence ID:
{evidence_id}

Original discovery name:
{row.get("name")}

Previously corrected name:
{row.get("corrected_name")}

Discovery phrase:
{row.get("grounding_phrase")}

Your job is to determine what REAL discrete physical object,
if any, this evidence corresponds to.

Do NOT trust the original label.

Choose exactly one status:

PHYSICAL_OBJECT
- this evidence corresponds to a visible discrete physical
  object or fixture in this crop.

STRUCTURAL_SURFACE
- it actually refers to wall, floor, ceiling, baseboard,
  corner, or other architectural surface.

ARCHITECTURAL_OBJECT
- it is a room architecture element such as door/door frame.

HALLUCINATION_OR_WRONG
- the claimed observation does not correspond to a supported
  discrete object.

If PHYSICAL_OBJECT:
return a corrected generic name describing the real object.

Examples:
- a mislabeled faucet that is actually shower hardware:
  corrected_name = "shower head"
- a mislabeled faucet that is actually basin hardware:
  corrected_name = "sink faucet"

Do not invent an object.

Return ONLY JSON:

{{
  "evidence_id": "{evidence_id}",
  "status": "PHYSICAL_OBJECT",
  "corrected_name": "generic name",
  "reason": "brief local-image reason"
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
                            str(crop_path),
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
            text=[chat_text],
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


        (
            RAW_DIR
            /
            f"{evidence_id}.txt"
        ).write_text(
            raw,
            encoding="utf-8"
        )


        parsed = parse_json_object(
            raw
        )


        result = dict(row)


        if (
            parsed is None
            or
            str(
                parsed.get(
                    "evidence_id",
                    ""
                )
            )
            !=
            evidence_id
        ):

            result[
                "final_status"
            ] = "UNRESOLVED"

            result[
                "final_reason"
            ] = (
                "Crop reverification returned invalid result."
            )

            result[
                "crop_path"
            ] = str(crop_path)


        else:

            status = str(
                parsed.get(
                    "status",
                    ""
                )
            ).strip().upper()


            allowed = {

                "PHYSICAL_OBJECT",
                "STRUCTURAL_SURFACE",
                "ARCHITECTURAL_OBJECT",
                "HALLUCINATION_OR_WRONG",
            }


            if status not in allowed:

                status = "UNRESOLVED"


            result[
                "final_status"
            ] = status

            result[
                "final_name"
            ] = str(
                parsed.get(
                    "corrected_name",
                    row.get(
                        "corrected_name",
                        row.get(
                            "name"
                        )
                    )
                )
            ).strip()

            result[
                "final_reason"
            ] = str(
                parsed.get(
                    "reason",
                    ""
                )
            ).strip()

            result[
                "crop_path"
            ] = str(crop_path)


        reverified.append(
            result
        )


        print(
            "{:9s} | {:22s} -> {:22s} | {}".format(

                evidence_id,

                str(
                    row.get(
                        "name"
                    )
                )[:22],

                str(
                    result.get(
                        "final_name",
                        "-"
                    )
                )[:22],

                result[
                    "final_status"
                ],
            )
        )


    del model
    del processor

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# MERGE FINAL STATES
# ============================================================

final_physical = list(
    clean_physical
)

final_structural = list(
    previous_structural
) + list(
    deterministic_structural
)

final_wrong = list(
    previous_wrong
)

final_architectural = list(
    architectural_review
)

final_unresolved = []


for row in reverified:

    status = row[
        "final_status"
    ]


    if status == "PHYSICAL_OBJECT":

        final_physical.append(
            row
        )


    elif status == "STRUCTURAL_SURFACE":

        final_structural.append(
            row
        )


    elif status == "ARCHITECTURAL_OBJECT":

        final_architectural.append(
            row
        )


    elif status == "HALLUCINATION_OR_WRONG":

        final_wrong.append(
            row
        )


    else:

        final_unresolved.append(
            row
        )


# ============================================================
# ACCOUNTABILITY
# ============================================================

original_total = (

    len(
        physical
    )

    +
    len(
        previous_structural
    )

    +
    len(
        previous_wrong
    )
)


final_total = (

    len(
        final_physical
    )

    +
    len(
        final_structural
    )

    +
    len(
        final_wrong
    )

    +
    len(
        final_architectural
    )

    +
    len(
        final_unresolved
    )
)


accountability_pass = bool(
    original_total
    ==
    final_total
)


# ============================================================
# SAVE
# ============================================================

result = {

    "stage":
        "06D1D3",

    "input_06d1d2_count":
        original_total,

    "accountability": {

        "pass":
            accountability_pass,

        "input_count":
            original_total,

        "output_count":
            final_total,
    },

    "physical_object_evidence":
        final_physical,

    "structural_surface_evidence":
        final_structural,

    "architectural_object_evidence":
        final_architectural,

    "hallucination_or_wrong_evidence":
        final_wrong,

    "unresolved_evidence":
        final_unresolved,

    "counts": {

        "physical":
            len(
                final_physical
            ),

        "structural":
            len(
                final_structural
            ),

        "architectural":
            len(
                final_architectural
            ),

        "wrong":
            len(
                final_wrong
            ),

        "unresolved":
            len(
                final_unresolved
            ),
    },

    "status":
        (
            "READY_FOR_INSTANCE_DEDUP"
            if
            accountability_pass
            and
            not final_unresolved
            else
            "REQUIRES_REVIEW"
        ),

    "rules": [

        "obvious architectural surfaces rejected deterministically",

        "ambiguous evidence reverified from its source crop",

        "doors remain architecture rather than normal props",

        "no instance deduplication performed",

        "no connected grouping performed",

        "no Florence or SAM2 performed",
    ],
}


RESULT_PATH = (
    OUT
    / "00_stage06d1d3_result.json"
)


RESULT_PATH.write_text(
    json.dumps(
        result,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


PHYSICAL_PATH = (
    OUT
    / "01_clean_physical_evidence.json"
)


PHYSICAL_PATH.write_text(
    json.dumps(
        final_physical,
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
print("PRODUCTION STAGE 06D1D3 RESULT")
print("=" * 110)

print()
print(
    "ACCOUNTABILITY PASS:",
    accountability_pass
)

print(
    "PHYSICAL:",
    len(final_physical)
)

print(
    "STRUCTURAL:",
    len(final_structural)
)

print(
    "ARCHITECTURAL:",
    len(final_architectural)
)

print(
    "WRONG / HALLUCINATION:",
    len(final_wrong)
)

print(
    "UNRESOLVED:",
    len(final_unresolved)
)


print()
print("=" * 110)
print("FINAL PHYSICAL EVIDENCE")
print("=" * 110)


for row in final_physical:

    name = (
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
    )

    print(
        "{:9s} | {:26s} | {}".format(

            row[
                "evidence_id"
            ],

            str(name)[:26],

            row.get(
                "final_reason",
                row.get(
                    "verification_reason",
                    ""
                )
            ),
        )
    )


print()
print("=" * 110)
print("STRUCTURAL")
print("=" * 110)

for row in final_structural:

    print(
        "{} | {}".format(
            row.get(
                "evidence_id"
            ),
            row.get(
                "name"
            )
        )
    )


print()
print("=" * 110)
print("ARCHITECTURAL OBJECTS")
print("=" * 110)

for row in final_architectural:

    print(
        "{} | {}".format(
            row.get(
                "evidence_id"
            ),
            row.get(
                "name"
            )
        )
    )


print()
print("=" * 110)
print("WRONG / HALLUCINATION")
print("=" * 110)

for row in final_wrong:

    print(
        "{} | {} | {}".format(

            row.get(
                "evidence_id"
            ),

            row.get(
                "name"
            ),

            row.get(
                "final_reason",
                row.get(
                    "verification_reason",
                    ""
                )
            ),
        )
    )


if final_unresolved:

    print()
    print("=" * 110)
    print("UNRESOLVED")
    print("=" * 110)

    for row in final_unresolved:

        print(
            row[
                "evidence_id"
            ]
        )


print()
print(
    "RESULT JSON:",
    RESULT_PATH
)

print(
    "CLEAN PHYSICAL JSON:",
    PHYSICAL_PATH
)

print()
print(
    "NO INSTANCE DEDUPLICATION WAS PERFORMED."
)

print(
    "NO CONNECTED GROUPING WAS PERFORMED."
)

print(
    "NO FLORENCE OR SAM2 WAS RUN."
)


# ============================================================
# INLINE VISUAL AUDIT OF EVERY REVERIFIED CROP
# ============================================================

if reverified:

    print()
    print("=" * 110)
    print("INLINE AMBIGUOUS-EVIDENCE CROP AUDIT")
    print("=" * 110)


    for row in reverified:

        path = row.get(
            "crop_path"
        )

        if not path:
            continue


        title = (
            f'{row["evidence_id"]} | '
            f'{row.get("name")} → '
            f'{row.get("final_name", "-")} | '
            f'{row["final_status"]}'
        )


        show(
            path,
            title,
            figsize=(6, 6)
        )
