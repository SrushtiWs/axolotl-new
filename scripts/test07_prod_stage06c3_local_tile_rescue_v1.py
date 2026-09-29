
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

STAGE01 = (
    PROD
    / "stage01_master"
)

STAGE06 = (
    PROD
    / "stage06_prop_layer"
)


MASTER = (
    STAGE01
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


LEGACY_RESCUE = (
    SCRIPTS
    / "test07_stage046_local_tile_rescue.py"
)


MODEL_CACHE = Path(
    "/workspace/data/huggingface-cache"
)


OUT = (
    STAGE06
    / "06c3_local_tile_rescue"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# VALIDATE
# ============================================================

print("=" * 110)
print("PRODUCTION STAGE 06C3")
print("LOCAL TILE DINO RESCUE")
print("=" * 110)


required = {

    "STAGE01 MASTER":
        MASTER,

    "PHYSICAL INVENTORY":
        INVENTORY_JSON,

    "EXISTENCE AUDIT":
        EXISTENCE_JSON,

    "06A1 RAW TILED METADATA":
        TILED_CANDIDATES_JSON,

    "06A1 TILE DIRECTORY":
        TILE_DIR,

    "FROZEN STAGE04.6":
        LEGACY_RESCUE,

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
# PRE-RUN SAFETY CHECK
# ============================================================

inventory = json.loads(
    INVENTORY_JSON.read_text(
        encoding="utf-8"
    )
)["objects"]


existence = json.loads(
    EXISTENCE_JSON.read_text(
        encoding="utf-8"
    )
)


tiled = json.loads(
    TILED_CANDIDATES_JSON.read_text(
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


uncertain_ids = sorted(
    int(
        row["inventory_id"]
    )
    for row in existence
    if row.get(
        "decision"
    ) == "UNCERTAIN"
)


print()
print(
    "PHYSICAL INVENTORY:",
    len(
        inventory
    )
)

print(
    "PRESENT TO RESCUE:",
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

print(
    "UNCERTAIN:",
    len(
        uncertain_ids
    )
)


# ============================================================
# VERIFY SOURCE REGION METADATA
# ============================================================

region_bbox = {}

for row in tiled:

    region = row.get(
        "source_region"
    )

    bbox = row.get(
        "source_bbox"
    )

    if region and bbox:

        region_bbox[
            str(region)
        ] = bbox


expected_regions = [
    "top_left",
    "top_center",
    "top_right",
    "middle_left",
    "middle_center",
    "middle_right",
    "bottom_left",
    "bottom_center",
    "bottom_right",
]


missing_region_metadata = [

    region
    for region in expected_regions
    if region not in region_bbox
]


missing_tiles = [

    region
    for region in expected_regions
    if not (
        TILE_DIR
        / f"{region}.png"
    ).exists()
]


print()
print(
    "REGION METADATA:",
    len(
        region_bbox
    ),
    "/ 9"
)

print(
    "MISSING REGION METADATA:",
    missing_region_metadata
)

print(
    "MISSING TILE PNGS:",
    missing_tiles
)


if missing_region_metadata:

    raise RuntimeError(
        "Missing source-region bbox metadata."
    )


if missing_tiles:

    raise RuntimeError(
        "Missing source-region tile PNGs."
    )


# ============================================================
# CHECK PRESENT OBJECT SOURCE REGIONS
# ============================================================

inventory_by_id = {

    int(
        obj["id"]
    ):
        obj

    for obj in inventory
}


print()
print(
    "PRESENT INSTANCE SOURCE REGIONS:"
)


for iid in present_ids:

    if iid not in inventory_by_id:

        raise RuntimeError(
            f"PRESENT inventory ID missing: {iid}"
        )


    obj = inventory_by_id[
        iid
    ]


    regions = obj.get(
        "source_regions",
        []
    )


    if isinstance(
        regions,
        str
    ):

        regions = [
            regions
        ]


    if not regions:

        sr = obj.get(
            "source_region",
            ""
        )

        if sr:

            regions = [
                sr
            ]


    print(
        f"{iid:03d}. "
        f'{obj.get("name","")} '
        f"→ {regions}"
    )


    if not regions:

        raise RuntimeError(
            f"No source regions for PRESENT ID {iid}"
        )


    for region in regions:

        if region not in region_bbox:

            raise RuntimeError(
                f"Unknown source region {region} "
                f"for PRESENT ID {iid}"
            )


# ============================================================
# LOAD FROZEN STAGE04.6
# ============================================================

spec = (
    importlib.util
    .spec_from_file_location(
        "test07_stage046_local_rescue",
        LEGACY_RESCUE
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
    "✅ FROZEN STAGE04.6 LOADED"
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
# LOAD RESULTS
# ============================================================

CANDIDATE_PATH = (
    OUT
    / "00_local_tile_rescue_candidates.json"
)


PREVIEW_PATH = (
    OUT
    / "01_local_tile_rescue_preview.png"
)


LEGACY_REPORT = (
    OUT
    / "stage046_report.json"
)


if not CANDIDATE_PATH.exists():

    raise RuntimeError(
        "Rescue candidate JSON missing."
    )


candidates = json.loads(
    CANDIDATE_PATH.read_text(
        encoding="utf-8"
    )
)


# ============================================================
# PRODUCTION PARENT SUMMARY
# ============================================================

summary = {}


for iid in present_ids:

    obj = inventory_by_id[
        iid
    ]


    rows = [

        row
        for row in candidates

        if int(
            row[
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


    summary[
        iid
    ] = {

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

        "source_regions":
            obj.get(
                "source_regions",
                []
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

        "candidate_source_regions":
            sorted(
                {
                    str(
                        row.get(
                            "source_region",
                            ""
                        )
                    )

                    for row in rows

                    if row.get(
                        "source_region"
                    )
                }
            ),

        "status":
            (
                "RESCUE_PROPOSALS_AVAILABLE"
                if rows
                else
                "NO_RESCUE_PROPOSAL"
            ),
    }


summary_rows = [

    summary[
        iid
    ]

    for iid in present_ids
]


SUMMARY_PATH = (
    OUT
    / "02_production_rescue_summary.json"
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

parents_with_proposals = sum(

    1
    for row in summary_rows

    if row[
        "candidate_count"
    ] > 0
)


parents_without_proposals = (
    len(
        summary_rows
    )
    -
    parents_with_proposals
)


# ============================================================
# HUMAN REPORT
# ============================================================

print()
print("=" * 110)
print("PRODUCTION STAGE 06C3 RESULT")
print("=" * 110)


print(
    "PRESENT INPUT:",
    len(
        present_ids
    )
)


print(
    "TOTAL RESCUE CANDIDATES:",
    len(
        candidates
    )
)


print(
    "PRESENT WITH PROPOSALS:",
    parents_with_proposals
)


print(
    "PRESENT WITHOUT PROPOSAL:",
    parents_without_proposals
)


print()
print(
    "PER PRESENT PHYSICAL INSTANCE:"
)


for row in summary_rows:

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

        "{:03d}. {} | candidates={} | best={} | regions={} | {}".format(

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
                "candidate_source_regions"
            ],

            row[
                "status"
            ]
        )
    )


print()
print(
    "RESCUE CANDIDATES:",
    CANDIDATE_PATH
)

print(
    "PREVIEW:",
    PREVIEW_PATH
)

print(
    "PRODUCTION SUMMARY:",
    SUMMARY_PATH
)

print(
    "LEGACY REPORT:",
    LEGACY_REPORT
)


print()
print(
    "IMPORTANT:"
)

print(
    "These are rescue PROPOSALS only."
)

print(
    "No rescued bbox is accepted yet."
)
