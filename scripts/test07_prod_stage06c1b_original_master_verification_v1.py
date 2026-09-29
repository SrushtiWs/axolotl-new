
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


ORIGINAL_MASTER = (
    PROD
    / "stage01_master"
    / "00_master_input.png"
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


LEGACY_VERIFIER = (
    SCRIPTS
    / "test07_stage04_candidate_verification.py"
)


MODEL_CACHE = Path(
    "/workspace/data/huggingface-cache"
)


OUT = (
    STAGE06
    / "06c1b_original_master_verification"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CHECK
# ============================================================

print("=" * 110)
print("PRODUCTION STAGE 06C1B")
print("ORIGINAL-MASTER CANDIDATE VERIFICATION")
print("=" * 110)


for label, path in {

    "STAGE01 ORIGINAL MASTER":
        ORIGINAL_MASTER,

    "06B DINO CANDIDATES":
        CANDIDATE_JSON,

    "PARENT SUMMARY":
        PARENT_GROUNDING_SUMMARY,

    "VERIFIED STAGE04":
        LEGACY_VERIFIER,

    "MODEL CACHE":
        MODEL_CACHE,

}.items():

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


candidates = json.loads(
    CANDIDATE_JSON.read_text(
        encoding="utf-8"
    )
)


print()
print(
    "DINO CANDIDATES TO REVERIFY:",
    len(candidates)
)


# ============================================================
# LOAD FROZEN VERIFIER
# ============================================================

spec = (
    importlib.util
    .spec_from_file_location(
        "test07_stage04_original_verifier",
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
    "✅ VERIFIED STAGE04 LOADED"
)


# ============================================================
# CRITICAL DIFFERENCE:
#
# SAME DINO BOXES
# BUT VERIFY AGAINST STAGE01 ORIGINAL MASTER
# ============================================================

module.run(

    master_path=
        ORIGINAL_MASTER,

    candidate_json=
        CANDIDATE_JSON,

    output_dir=
        OUT,

    model_cache=
        MODEL_CACHE,
)


# ============================================================
# LOAD OUTPUT
# ============================================================

ALL_PATH = (
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


grounding_parents = json.loads(
    PARENT_GROUNDING_SUMMARY.read_text(
        encoding="utf-8"
    )
)


# ============================================================
# PARENT SUMMARY
# ============================================================

parents = {}


for row in grounding_parents:

    pid = int(
        row["id"]
    )

    parents[pid] = {

        "id":
            pid,

        "name":
            row.get(
                "name",
                ""
            ),

        "dino":
            int(
                row.get(
                    "dino_candidate_count",
                    0
                )
            ),

        "valid":
            0,

        "invalid":
            0,

        "unknown":
            0,

        "best_valid":
            None,
    }


for row in all_rows:

    pid_raw = row.get(
        "parent_inventory_id"
    )

    if pid_raw is None:
        continue


    pid = int(
        pid_raw
    )


    if pid not in parents:
        continue


    decision = str(
        row.get(
            "verification_decision",
            "UNKNOWN"
        )
    ).upper()


    if decision == "VALID":

        parents[pid][
            "valid"
        ] += 1


        score = float(
            row.get(
                "score",
                0.0
            )
        )


        old = parents[pid][
            "best_valid"
        ]


        if (
            old is None
            or
            score > old
        ):

            parents[pid][
                "best_valid"
            ] = score


    elif decision == "INVALID":

        parents[pid][
            "invalid"
        ] += 1


    else:

        parents[pid][
            "unknown"
        ] += 1


# ============================================================
# COUNTS
# ============================================================

valid_total = sum(
    1
    for r in all_rows
    if str(
        r.get(
            "verification_decision",
            ""
        )
    ).upper() == "VALID"
)


invalid_total = sum(
    1
    for r in all_rows
    if str(
        r.get(
            "verification_decision",
            ""
        )
    ).upper() == "INVALID"
)


unknown_total = (
    len(all_rows)
    -
    valid_total
    -
    invalid_total
)


parents_with_valid = sum(
    1
    for r in parents.values()
    if r[
        "valid"
    ] > 0
)


parents_without_valid = sum(
    1
    for r in parents.values()
    if (
        r[
            "dino"
        ] > 0
        and
        r[
            "valid"
        ] == 0
    )
)


parents_no_dino = sum(
    1
    for r in parents.values()
    if r[
        "dino"
    ] == 0
)


# ============================================================
# REPORT
# ============================================================

print()
print("=" * 110)
print("PRODUCTION STAGE 06C1B RESULT")
print("=" * 110)

print()
print("CANDIDATE LEVEL")

print(
    "INPUT :",
    len(
        all_rows
    )
)

print(
    "VALID :",
    valid_total
)

print(
    "INVALID:",
    invalid_total
)

print(
    "UNKNOWN:",
    unknown_total
)


print()
print("PARENT LEVEL")

print(
    "TOTAL PARENTS:",
    len(
        parents
    )
)

print(
    "WITH VALID:",
    parents_with_valid
)

print(
    "NO VALID:",
    parents_without_valid
)

print(
    "NO DINO:",
    parents_no_dino
)


print()
print(
    "PER PHYSICAL INVENTORY ITEM:"
)


for pid in sorted(
    parents
):

    row = parents[
        pid
    ]


    best = row[
        "best_valid"
    ]


    best_text = (
        "{:.4f}".format(
            best
        )
        if best is not None
        else "NONE"
    )


    print(

        "{:03d}. {} | dino={} | valid={} | invalid={} | unknown={} | best={}".format(

            row["id"],
            row["name"],
            row["dino"],
            row["valid"],
            row["invalid"],
            row["unknown"],
            best_text
        )
    )


# ============================================================
# SAVE PRODUCTION SUMMARY
# ============================================================

SUMMARY_PATH = (
    OUT
    / "04_parent_original_master_verification_summary.json"
)


SUMMARY_PATH.write_text(

    json.dumps(
        [
            parents[k]
            for k in sorted(
                parents
            )
        ],
        indent=2,
        ensure_ascii=False
    ),

    encoding="utf-8"
)


print()
print(
    "VERIFIED PREVIEW:",
    PREVIEW_PATH
)

print(
    "PARENT SUMMARY:",
    SUMMARY_PATH
)
