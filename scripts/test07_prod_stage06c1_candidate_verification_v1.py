
from pathlib import Path
import importlib.util
import json


# ============================================================
# PATHS
# ============================================================

BASE = Path(
    "/workspace/axolotl"
)

SCRIPTS = (
    BASE
    / "scripts"
)

PROD = (
    BASE
    / "test07"
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


DETECTION_IMAGE = (
    STAGE05
    / "07_clean_room_with_exact_props.png"
)


CANDIDATE_JSON = (
    STAGE06
    / "06b_source_crop_grounding"
    / "05_parent_mapped_dino_candidates_flat.json"
)


PARENT_GROUNDING_SUMMARY = (
    STAGE06
    / "06b_source_crop_grounding"
    / "06_parent_grounding_summary.json"
)


OUT = (
    STAGE06
    / "06c1_candidate_verification"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


LEGACY_SCRIPT = (
    SCRIPTS
    / "test07_stage04_candidate_verification.py"
)


MODEL_CACHE = Path(
    "/workspace/data/huggingface-cache"
)


# ============================================================
# VALIDATE
# ============================================================

print("=" * 110)
print("PRODUCTION STAGE 06C1")
print("QWEN DINO CANDIDATE VERIFICATION")
print("=" * 110)


required = {

    "STAGE05 DETECTION IMAGE":
        DETECTION_IMAGE,

    "06B CANDIDATES":
        CANDIDATE_JSON,

    "06B PARENT SUMMARY":
        PARENT_GROUNDING_SUMMARY,

    "VERIFIED STAGE04 SCRIPT":
        LEGACY_SCRIPT,

    "MODEL CACHE":
        MODEL_CACHE,
}


for label, path in required.items():

    ok = path.exists()

    print(
        "✅" if ok else "❌",
        label,
        path
    )

    if not ok:

        raise FileNotFoundError(
            path
        )


# ============================================================
# INPUT CHECK
# ============================================================

input_candidates = json.loads(
    CANDIDATE_JSON.read_text(
        encoding="utf-8"
    )
)


if not isinstance(
    input_candidates,
    list
):

    raise RuntimeError(
        "Candidate JSON must contain a list."
    )


print()
print(
    "INPUT DINO CANDIDATES:",
    len(
        input_candidates
    )
)


with_parent = sum(
    1
    for row in input_candidates
    if row.get(
        "parent_inventory_id"
    ) is not None
)


print(
    "CANDIDATES WITH PARENT ID:",
    with_parent
)


if (
    with_parent
    != len(
        input_candidates
    )
):

    raise RuntimeError(
        "Some 06B candidates lost parent_inventory_id."
    )


# ============================================================
# LOAD VERIFIED STAGE04
# ============================================================

spec = (
    importlib.util
    .spec_from_file_location(
        "test07_stage04_verifier",
        LEGACY_SCRIPT
    )
)


module = (
    importlib.util
    .module_from_spec(
        spec
    )
)


spec.loader.exec_module(
    module
)


print()
print(
    "✅ VERIFIED STAGE04 IMPLEMENTATION LOADED"
)


# ============================================================
# RUN FROZEN VERIFIER
# ============================================================

module.run(
    master_path=
        DETECTION_IMAGE,

    candidate_json=
        CANDIDATE_JSON,

    output_dir=
        OUT,

    model_cache=
        MODEL_CACHE,
)


# ============================================================
# OUTPUT PATHS FROM FROZEN STAGE04
# ============================================================

ALL_DECISIONS_PATH = (
    OUT
    / "01_qwen_candidate_verification.json"
)


VALID_PATH = (
    OUT
    / "02_verified_candidates.json"
)


PREVIEW_PATH = (
    OUT
    / "03_verified_candidates_preview.png"
)


LEGACY_REPORT_PATH = (
    OUT
    / "stage04_report.json"
)


for p in [
    ALL_DECISIONS_PATH,
    VALID_PATH,
    PREVIEW_PATH,
    LEGACY_REPORT_PATH,
]:

    if not p.exists():

        raise RuntimeError(
            "Expected Stage04 output missing: {}".format(
                p
            )
        )


# ============================================================
# LOAD RESULTS
# ============================================================

all_decisions = json.loads(
    ALL_DECISIONS_PATH.read_text(
        encoding="utf-8"
    )
)


valid_candidates = json.loads(
    VALID_PATH.read_text(
        encoding="utf-8"
    )
)


grounding_parent_rows = json.loads(
    PARENT_GROUNDING_SUMMARY.read_text(
        encoding="utf-8"
    )
)


print()
print(
    "VERIFICATION DECISIONS:",
    len(
        all_decisions
    )
)

print(
    "VALID CANDIDATES:",
    len(
        valid_candidates
    )
)


# ============================================================
# PARENT-LEVEL SUMMARY
# ============================================================

parent_summary = {}


# Start from ALL 28 physical parent inventory items.
# This preserves the cabinet that received no DINO proposal.
for row in grounding_parent_rows:

    pid = int(
        row[
            "id"
        ]
    )

    parent_summary[
        pid
    ] = {

        "parent_inventory_id":
            pid,

        "name":
            row.get(
                "name",
                ""
            ),

        "grounding_phrase":
            row.get(
                "grounding_phrase",
                ""
            ),

        "grounding_observation_count":
            int(
                row.get(
                    "observation_count",
                    0
                )
            ),

        "dino_candidate_count":
            int(
                row.get(
                    "dino_candidate_count",
                    0
                )
            ),

        "best_dino_score":
            row.get(
                "best_score"
            ),

        "verified_valid":
            0,

        "verified_invalid":
            0,

        "verified_unknown":
            0,

        "valid_candidates":
            [],

        "status":
            None,
    }


# ============================================================
# ADD VERIFICATION DECISIONS
# ============================================================

for row in all_decisions:

    pid_raw = row.get(
        "parent_inventory_id"
    )


    if pid_raw is None:
        continue


    pid = int(
        pid_raw
    )


    if pid not in parent_summary:
        continue


    decision = str(
        row.get(
            "verification_decision",
            "UNKNOWN"
        )
    ).upper()


    if decision == "VALID":

        parent_summary[
            pid
        ][
            "verified_valid"
        ] += 1


    elif decision == "INVALID":

        parent_summary[
            pid
        ][
            "verified_invalid"
        ] += 1


    else:

        parent_summary[
            pid
        ][
            "verified_unknown"
        ] += 1


# ============================================================
# ATTACH VALID CANDIDATES
# ============================================================

for row in valid_candidates:

    pid_raw = row.get(
        "parent_inventory_id"
    )


    if pid_raw is None:
        continue


    pid = int(
        pid_raw
    )


    if pid not in parent_summary:
        continue


    candidate_record = {

        "observation_id":
            row.get(
                "observation_id"
            ),

        "source_region":
            row.get(
                "source_region"
            ),

        "score":
            float(
                row.get(
                    "score",
                    0.0
                )
            ),

        "bbox":
            row.get(
                "bbox"
            ),

        "dino_label":
            row.get(
                "dino_label"
            ),

        "verification_raw":
            row.get(
                "verification_raw"
            ),
    }


    parent_summary[
        pid
    ][
        "valid_candidates"
    ].append(
        candidate_record
    )


# ============================================================
# SORT VALID CANDIDATES BY DINO SCORE
# ============================================================

for pid in parent_summary:

    parent_summary[
        pid
    ][
        "valid_candidates"
    ] = sorted(

        parent_summary[
            pid
        ][
            "valid_candidates"
        ],

        key=lambda x:
            x[
                "score"
            ],

        reverse=True
    )


# ============================================================
# ASSIGN PARENT STATUS
# ============================================================

for pid, row in parent_summary.items():

    dino_count = (
        row[
            "dino_candidate_count"
        ]
    )

    valid_count = (
        row[
            "verified_valid"
        ]
    )

    unknown_count = (
        row[
            "verified_unknown"
        ]
    )


    if dino_count == 0:

        status = (
            "NO_DINO_CANDIDATE"
        )


    elif valid_count > 0:

        status = (
            "HAS_VERIFIED_LOCALIZATION"
        )


    elif unknown_count > 0:

        status = (
            "UNRESOLVED_VERIFICATION"
        )


    else:

        status = (
            "NO_VALID_CANDIDATE"
        )


    row[
        "status"
    ] = status


# ============================================================
# SAVE PARENT SUMMARY
# ============================================================

parent_rows = [

    parent_summary[
        pid
    ]

    for pid in sorted(
        parent_summary
    )
]


PARENT_SUMMARY_PATH = (
    OUT
    / "04_parent_verification_summary.json"
)


PARENT_SUMMARY_PATH.write_text(
    json.dumps(
        parent_rows,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# ============================================================
# SAVE ONLY VERIFIED PARENT CANDIDATES
# ============================================================

verified_parent_candidates = []


for parent in parent_rows:

    for candidate in parent[
        "valid_candidates"
    ]:

        row = dict(
            candidate
        )

        row[
            "parent_inventory_id"
        ] = parent[
            "parent_inventory_id"
        ]

        row[
            "parent_inventory_name"
        ] = parent[
            "name"
        ]

        row[
            "grounding_phrase"
        ] = parent[
            "grounding_phrase"
        ]


        verified_parent_candidates.append(
            row
        )


VERIFIED_PARENT_CANDIDATES_PATH = (
    OUT
    / "05_verified_parent_candidates_flat.json"
)


VERIFIED_PARENT_CANDIDATES_PATH.write_text(
    json.dumps(
        verified_parent_candidates,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# ============================================================
# COUNTS
# ============================================================

valid_total = sum(

    1
    for row in all_decisions

    if str(
        row.get(
            "verification_decision",
            ""
        )
    ).upper()
    ==
    "VALID"
)


invalid_total = sum(

    1
    for row in all_decisions

    if str(
        row.get(
            "verification_decision",
            ""
        )
    ).upper()
    ==
    "INVALID"
)


unknown_total = sum(

    1
    for row in all_decisions

    if str(
        row.get(
            "verification_decision",
            ""
        )
    ).upper()
    ==
    "UNKNOWN"
)


parents_verified = sum(

    1
    for row in parent_rows

    if row[
        "status"
    ]
    ==
    "HAS_VERIFIED_LOCALIZATION"
)


parents_no_valid = sum(

    1
    for row in parent_rows

    if row[
        "status"
    ]
    ==
    "NO_VALID_CANDIDATE"
)


parents_unresolved = sum(

    1
    for row in parent_rows

    if row[
        "status"
    ]
    ==
    "UNRESOLVED_VERIFICATION"
)


parents_no_dino = sum(

    1
    for row in parent_rows

    if row[
        "status"
    ]
    ==
    "NO_DINO_CANDIDATE"
)


# ============================================================
# HUMAN REPORT
# ============================================================

REPORT_PATH = (
    OUT
    / "06_parent_verification_report.txt"
)


report_lines = [

    "=" * 110,

    "PRODUCTION STAGE 06C1 - CANDIDATE VERIFICATION",

    "=" * 110,

    "",

    "CANDIDATE LEVEL",

    "INPUT DINO CANDIDATES : {}".format(
        len(
            all_decisions
        )
    ),

    "VALID                 : {}".format(
        valid_total
    ),

    "INVALID               : {}".format(
        invalid_total
    ),

    "UNKNOWN               : {}".format(
        unknown_total
    ),

    "",

    "PARENT PHYSICAL INVENTORY LEVEL",

    "TOTAL PARENTS         : {}".format(
        len(
            parent_rows
        )
    ),

    "WITH VERIFIED BOX     : {}".format(
        parents_verified
    ),

    "NO VALID BOX          : {}".format(
        parents_no_valid
    ),

    "UNRESOLVED            : {}".format(
        parents_unresolved
    ),

    "NO DINO CANDIDATE     : {}".format(
        parents_no_dino
    ),

    "",

    "PER PHYSICAL INVENTORY ITEM:",
]


for row in parent_rows:

    valid_boxes = (
        row[
            "valid_candidates"
        ]
    )


    best_valid = (
        valid_boxes[0][
            "score"
        ]
        if valid_boxes
        else None
    )


    best_text = (

        "{:.4f}".format(
            best_valid
        )

        if best_valid is not None

        else "NONE"
    )


    report_lines.append(

        "{:03d}. {} | dino={} | valid={} | invalid={} | unknown={} | best_valid={} | {}".format(

            row[
                "parent_inventory_id"
            ],

            row[
                "name"
            ],

            row[
                "dino_candidate_count"
            ],

            row[
                "verified_valid"
            ],

            row[
                "verified_invalid"
            ],

            row[
                "verified_unknown"
            ],

            best_text,

            row[
                "status"
            ],
        )
    )


REPORT_PATH.write_text(
    "\n".join(
        report_lines
    ),
    encoding="utf-8"
)


# ============================================================
# PRINT
# ============================================================

print()
print(
    "\n".join(
        report_lines
    )
)


print()
print(
    "VERIFIED PREVIEW:",
    PREVIEW_PATH
)


print(
    "PARENT SUMMARY:",
    PARENT_SUMMARY_PATH
)


print(
    "VERIFIED PARENT CANDIDATES:",
    VERIFIED_PARENT_CANDIDATES_PATH
)


print(
    "REPORT:",
    REPORT_PATH
)
