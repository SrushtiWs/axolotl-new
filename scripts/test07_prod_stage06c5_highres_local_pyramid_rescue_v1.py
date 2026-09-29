
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


INVENTORY_JSON = (
    STAGE06
    / "06c2_lost_instance_existence_audit"
    / "00a_physical_inventory_bridge.json"
)


EXISTENCE_JSON = (
    STAGE06
    / "06c2_lost_instance_existence_audit"
    / "00_existence_audit.json"
)


TILED_CANDIDATES_JSON = (
    STAGE06
    / "06a_inventory"
    / "01_raw_tiled_candidates.json"
)


TILE_DIR = (
    STAGE06
    / "06a_inventory"
    / "tiles"
)


LEGACY_STAGE048 = (
    SCRIPTS
    / "test07_stage048_highres_local_pyramid_rescue.py"
)


MODEL_CACHE = Path(
    "/workspace/data/huggingface-cache"
)


OUT = (
    STAGE06
    / "06c5_highres_local_pyramid_rescue"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# VALIDATE INPUTS
# ============================================================

print("=" * 110)
print("PRODUCTION STAGE 06C5")
print("HIGH-RES LOCAL PYRAMID DINO RESCUE")
print("=" * 110)


required = {

    "STAGE01 MASTER":
        MASTER,

    "PHYSICAL INVENTORY":
        INVENTORY_JSON,

    "EXISTENCE AUDIT":
        EXISTENCE_JSON,

    "06A1 TILED METADATA":
        TILED_CANDIDATES_JSON,

    "06A1 TILE DIRECTORY":
        TILE_DIR,

    "FROZEN STAGE04.8":
        LEGACY_STAGE048,

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
# CONFIRM PRESENT INPUT SET
# ============================================================

existence = json.loads(
    EXISTENCE_JSON.read_text(
        encoding="utf-8"
    )
)


present_ids = sorted(
    int(
        row["inventory_id"]
    )
    for row in existence
    if row.get(
        "decision"
    ) == "PRESENT"
)


absent_ids = sorted(
    int(
        row["inventory_id"]
    )
    for row in existence
    if row.get(
        "decision"
    ) == "ABSENT"
)


print()
print(
    "PRESENT TO HIGHRES RESCUE:",
    len(
        present_ids
    )
)

print(
    "PRESENT IDS:",
    present_ids
)

print(
    "ABSENT EXCLUDED:",
    len(
        absent_ids
    )
)


if len(
    present_ids
) != 10:

    print(
        "⚠️ PRESENT count differs from previous Stage06C2 result."
    )


# ============================================================
# LOAD FROZEN STAGE04.8
# ============================================================

spec = (
    importlib.util
    .spec_from_file_location(
        "test07_stage048_highres_rescue",
        LEGACY_STAGE048
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
    "✅ FROZEN STAGE04.8 LOADED"
)


print(
    "UPSCALE:",
    module.UPSCALE
)

print(
    "BOX_THRESHOLD:",
    module.BOX_THRESHOLD
)

print(
    "TEXT_THRESHOLD:",
    module.TEXT_THRESHOLD
)

print(
    "TOP_K_PER_VIEW:",
    module.TOP_K_PER_VIEW
)


# ============================================================
# RUN
# ============================================================

module.run(

    master_path=
        MASTER,

    inventory_json=
        INVENTORY_JSON,

    existence_json=
        EXISTENCE_JSON,

    tiled_candidates_json=
        TILED_CANDIDATES_JSON,

    tile_dir=
        TILE_DIR,

    output_dir=
        OUT,

    model_cache=
        MODEL_CACHE,
)


# ============================================================
# EXPECTED OUTPUTS
# ============================================================

CANDIDATE_PATH = (
    OUT
    / "00_highres_pyramid_candidates.json"
)


PREVIEW_PATH = (
    OUT
    / "01_highres_pyramid_preview.png"
)


REPORT_PATH = (
    OUT
    / "stage048_report.json"
)


for p in [
    CANDIDATE_PATH,
    PREVIEW_PATH,
    REPORT_PATH,
]:

    if not p.exists():

        raise RuntimeError(
            f"Missing expected Stage04.8 output: {p}"
        )


# ============================================================
# LOAD CANDIDATES
# ============================================================

candidates = json.loads(
    CANDIDATE_PATH.read_text(
        encoding="utf-8"
    )
)


# ============================================================
# BUILD PER-INSTANCE PRODUCTION SUMMARY
# ============================================================

inventory = json.loads(
    INVENTORY_JSON.read_text(
        encoding="utf-8"
    )
)["objects"]


inventory_by_id = {

    int(
        row["id"]
    ):
        row

    for row in inventory
}


summary = []


for iid in present_ids:

    rows = [

        r
        for r in candidates

        if int(
            r[
                "inventory_id"
            ]
        ) == iid
    ]


    rows = sorted(
        rows,
        key=lambda r:
            float(
                r.get(
                    "score",
                    0.0
                )
            ),
        reverse=True
    )


    obj = inventory_by_id[
        iid
    ]


    summary.append({

        "inventory_id":
            iid,

        "name":
            obj.get(
                "name",
                ""
            ),

        "grounding_phrase":
            obj.get(
                "grounding_phrase",
                ""
            ),

        "candidate_count":
            len(
                rows
            ),

        "best_score":
            (
                float(
                    rows[0][
                        "score"
                    ]
                )
                if rows
                else None
            ),

        "best_bbox":
            (
                rows[0].get(
                    "bbox"
                )
                if rows
                else None
            ),

        "best_source_region":
            (
                rows[0].get(
                    "source_region"
                )
                if rows
                else None
            ),

        "best_pyramid_view":
            (
                rows[0].get(
                    "pyramid_view"
                )
                if rows
                else None
            ),

        "best_dino_query":
            (
                rows[0].get(
                    "dino_query"
                )
                if rows
                else None
            ),

        "status":
            (
                "HIGHRES_PROPOSALS_AVAILABLE"
                if rows
                else
                "NO_HIGHRES_PROPOSAL"
            ),
    })


SUMMARY_PATH = (
    OUT
    / "02_production_highres_summary.json"
)


SUMMARY_PATH.write_text(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# ============================================================
# COUNTS
# ============================================================

with_candidates = [

    row
    for row in summary

    if row[
        "candidate_count"
    ] > 0
]


without_candidates = [

    row
    for row in summary

    if row[
        "candidate_count"
    ] == 0
]


# ============================================================
# PRINT
# ============================================================

print()
print("=" * 110)
print("PRODUCTION STAGE 06C5 RESULT")
print("=" * 110)


print(
    "PRESENT INPUT:",
    len(
        present_ids
    )
)


print(
    "REDUCED HIGHRES CANDIDATES:",
    len(
        candidates
    )
)


print(
    "PRESENT WITH HIGHRES PROPOSALS:",
    len(
        with_candidates
    )
)


print(
    "PRESENT WITHOUT HIGHRES PROPOSAL:",
    len(
        without_candidates
    )
)


print()
print(
    "PER PRESENT PHYSICAL INSTANCE:"
)


for row in summary:

    best = row[
        "best_score"
    ]


    best_text = (
        "{:.4f}".format(
            best
        )
        if best is not None
        else "NONE"
    )


    print(

        "{:03d}. {} | candidates={} | best={} | region={} | view={} | query={} | {}".format(

            row[
                "inventory_id"
            ],

            row[
                "name"
            ],

            row[
                "candidate_count"
            ],

            best_text,

            row[
                "best_source_region"
            ],

            row[
                "best_pyramid_view"
            ],

            row[
                "best_dino_query"
            ],

            row[
                "status"
            ],
        )
    )


print()
print(
    "HIGHRES CANDIDATES:",
    CANDIDATE_PATH
)


print(
    "PREVIEW:",
    PREVIEW_PATH
)


print(
    "SUMMARY:",
    SUMMARY_PATH
)


print()
print(
    "IMPORTANT:"
)

print(
    "These are localization proposals only."
)

print(
    "Failed Qwen red-box verifier is NOT used."
)

print(
    "No SAM2 has been run."
)
