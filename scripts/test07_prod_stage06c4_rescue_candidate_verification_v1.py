
from pathlib import Path
import importlib.util
import json


BASE = Path("/workspace/axolotl")
SCRIPTS = BASE / "scripts"

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


RESCUE_JSON = (
    STAGE06
    / "06c3_local_tile_rescue"
    / "00_local_tile_rescue_candidates.json"
)


LEGACY_VERIFIER = (
    SCRIPTS
    / "test07_stage047_rescue_candidate_verification.py"
)


MODEL_CACHE = Path(
    "/workspace/data/huggingface-cache"
)


OUT = (
    STAGE06
    / "06c4_rescue_candidate_verification"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# VALIDATE
# ============================================================

print("=" * 110)
print("PRODUCTION STAGE 06C4")
print("RESCUE CANDIDATE VERIFICATION")
print("=" * 110)


required = {

    "STAGE01 MASTER":
        MASTER,

    "06C3 RESCUE CANDIDATES":
        RESCUE_JSON,

    "FROZEN STAGE04.7":
        LEGACY_VERIFIER,

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

rescue_candidates = json.loads(
    RESCUE_JSON.read_text(
        encoding="utf-8"
    )
)


if not isinstance(
    rescue_candidates,
    list
):

    raise RuntimeError(
        "06C3 rescue candidate JSON must be a list."
    )


print()
print(
    "RESCUE PROPOSALS:",
    len(
        rescue_candidates
    )
)


input_ids = sorted(
    {
        int(
            row[
                "inventory_id"
            ]
        )
        for row in rescue_candidates
    }
)


print(
    "PHYSICAL INSTANCES:",
    len(
        input_ids
    )
)

print(
    "INSTANCE IDS:",
    input_ids
)


# ============================================================
# LOAD FROZEN STAGE04.7
# ============================================================

spec = (
    importlib.util
    .spec_from_file_location(
        "test07_stage047_rescue_verifier",
        LEGACY_VERIFIER
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
    "✅ FROZEN STAGE04.7 LOADED"
)


# ============================================================
# RUN AGAINST ORIGINAL MASTER
# ============================================================

module.run(

    master_path=
        MASTER,

    rescue_json=
        RESCUE_JSON,

    output_dir=
        OUT,

    model_cache=
        MODEL_CACHE,
)


# ============================================================
# EXPECTED OUTPUTS
# ============================================================

ALL_PATH = (
    OUT
    / "00_rescue_verification_all.json"
)


VALID_PATH = (
    OUT
    / "01_verified_rescue_candidates.json"
)


BEST_PATH = (
    OUT
    / "02_best_verified_rescue_per_instance.json"
)


PREVIEW_PATH = (
    OUT
    / "03_best_verified_rescue_preview.png"
)


LEGACY_REPORT = (
    OUT
    / "stage047_report.json"
)


for p in [
    ALL_PATH,
    VALID_PATH,
    BEST_PATH,
    PREVIEW_PATH,
    LEGACY_REPORT,
]:

    if not p.exists():

        raise RuntimeError(
            f"Expected Stage04.7 output missing: {p}"
        )


# ============================================================
# LOAD
# ============================================================

all_rows = json.loads(
    ALL_PATH.read_text(
        encoding="utf-8"
    )
)


valid_rows = json.loads(
    VALID_PATH.read_text(
        encoding="utf-8"
    )
)


best_rows = json.loads(
    BEST_PATH.read_text(
        encoding="utf-8"
    )
)


# ============================================================
# BUILD PRODUCTION SUMMARY
# ============================================================

per_instance = {}


for iid in input_ids:

    rows = [
        r
        for r in all_rows
        if int(
            r[
                "inventory_id"
            ]
        ) == iid
    ]


    valid = [
        r
        for r in rows
        if r.get(
            "stage047_decision"
        ) == "VALID"
    ]


    invalid = [
        r
        for r in rows
        if r.get(
            "stage047_decision"
        ) == "INVALID"
    ]


    unknown = [
        r
        for r in rows
        if r.get(
            "stage047_decision"
        ) == "UNKNOWN"
    ]


    valid = sorted(
        valid,
        key=lambda r:
            float(
                r.get(
                    "score",
                    0.0
                )
            ),
        reverse=True
    )


    name = (
        rows[0].get(
            "inventory_name",
            ""
        )
        if rows
        else ""
    )


    per_instance[
        iid
    ] = {

        "inventory_id":
            iid,

        "name":
            name,

        "total":
            len(
                rows
            ),

        "valid":
            len(
                valid
            ),

        "invalid":
            len(
                invalid
            ),

        "unknown":
            len(
                unknown
            ),

        "best_valid_score":
            (
                float(
                    valid[0][
                        "score"
                    ]
                )
                if valid
                else None
            ),

        "best_valid_bbox":
            (
                valid[0].get(
                    "bbox"
                )
                if valid
                else None
            ),

        "status":
            (
                "RESCUE_VERIFIED"
                if valid
                else
                "STILL_UNRESOLVED"
            ),
    }


summary_rows = [

    per_instance[
        iid
    ]

    for iid in input_ids
]


SUMMARY_PATH = (
    OUT
    / "04_production_rescue_verification_summary.json"
)


SUMMARY_PATH.write_text(
    json.dumps(
        summary_rows,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# ============================================================
# COUNTS
# ============================================================

verified_instances = [

    r
    for r in summary_rows
    if r[
        "status"
    ] == "RESCUE_VERIFIED"
]


unresolved_instances = [

    r
    for r in summary_rows
    if r[
        "status"
    ] == "STILL_UNRESOLVED"
]


valid_total = sum(
    r[
        "valid"
    ]
    for r in summary_rows
)


invalid_total = sum(
    r[
        "invalid"
    ]
    for r in summary_rows
)


unknown_total = sum(
    r[
        "unknown"
    ]
    for r in summary_rows
)


# ============================================================
# PRINT PRODUCTION RESULT
# ============================================================

print()
print("=" * 110)
print("PRODUCTION STAGE 06C4 RESULT")
print("=" * 110)


print(
    "INPUT RESCUE CANDIDATES:",
    len(
        all_rows
    )
)


print(
    "VALID CANDIDATES:",
    valid_total
)


print(
    "INVALID CANDIDATES:",
    invalid_total
)


print(
    "UNKNOWN CANDIDATES:",
    unknown_total
)


print()
print(
    "PRESENT INSTANCES:",
    len(
        summary_rows
    )
)


print(
    "RESCUE VERIFIED INSTANCES:",
    len(
        verified_instances
    )
)


print(
    "STILL UNRESOLVED INSTANCES:",
    len(
        unresolved_instances
    )
)


print()
print(
    "PER PRESENT PHYSICAL INSTANCE:"
)


for row in summary_rows:

    best = row[
        "best_valid_score"
    ]


    best_text = (
        "{:.4f}".format(
            best
        )
        if best is not None
        else "NONE"
    )


    print(

        "{:03d}. {} | VALID={}/{} | INVALID={} | UNKNOWN={} | best={} | {}".format(

            row[
                "inventory_id"
            ],

            row[
                "name"
            ],

            row[
                "valid"
            ],

            row[
                "total"
            ],

            row[
                "invalid"
            ],

            row[
                "unknown"
            ],

            best_text,

            row[
                "status"
            ],
        )
    )


print()
print(
    "BEST VERIFIED RESCUE JSON:",
    BEST_PATH
)


print(
    "PREVIEW:",
    PREVIEW_PATH
)


print(
    "PRODUCTION SUMMARY:",
    SUMMARY_PATH
)


print()
print(
    "IMPORTANT:"
)

print(
    "No SAM2 has been run."
)

print(
    "Any STILL_UNRESOLVED instances remain protected "
    "for the next rescue stage."
)
