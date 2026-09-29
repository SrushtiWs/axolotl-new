
from pathlib import Path
import importlib.util
import json


# ============================================================
# PATHS
# ============================================================

BASE = Path("/workspace/axolotl")
SCRIPTS = BASE / "scripts"

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

INPUT_INVENTORY = (
    STAGE06
    / "06a3_surface_filter"
    / "00_conservative_inventory.json"
)

OUT = (
    STAGE06
    / "06b_source_crop_grounding"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


LEGACY_SCRIPT = (
    SCRIPTS
    / "test07_stage03_v45_full_source_crop_grounding.py"
)

MODEL_CACHE = Path(
    "/workspace/data/huggingface-cache"
)


OBSERVATION_INVENTORY = (
    OUT
    / "00_grounding_observation_inventory.json"
)

OBSERVATION_MAP = (
    OUT
    / "00b_observation_parent_map.json"
)

DUMMY_4X4_MANIFEST = (
    OUT
    / "00c_unused_4x4_manifest.json"
)


# ============================================================
# VALIDATE
# ============================================================

print("=" * 110)
print("PRODUCTION STAGE 06B")
print("SOURCE-CROP CONSTRAINED GROUNDING")
print("=" * 110)


required = {
    "STAGE05 DETECTION IMAGE":
        DETECTION_IMAGE,

    "06A3 INVENTORY":
        INPUT_INVENTORY,

    "VERIFIED V4.5 SCRIPT":
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
        raise FileNotFoundError(path)


# ============================================================
# LOAD 06A3 INVENTORY
# ============================================================

inventory = json.loads(
    INPUT_INVENTORY.read_text(
        encoding="utf-8"
    )
)

objects = inventory.get(
    "objects",
    []
)

if not isinstance(
    objects,
    list
):
    raise RuntimeError(
        'inventory["objects"] must be a list'
    )


print()
print(
    "06A3 INPUT OBJECTS:",
    len(objects)
)


# ============================================================
# REMOVE ONLY DETERMINISTIC NON-PHYSICAL "CORNER"
# ============================================================

kept_objects = []

removed_pre_grounding = []


for obj in objects:

    name = str(
        obj.get(
            "name",
            ""
        )
    ).strip().lower()

    phrase = str(
        obj.get(
            "grounding_phrase",
            ""
        )
    ).strip().lower()


    if (
        name == "corner"
        or
        phrase == "room corner"
    ):

        removed_pre_grounding.append(
            obj
        )

    else:

        kept_objects.append(
            obj
        )


print(
    "REMOVED BEFORE GROUNDING:",
    len(
        removed_pre_grounding
    )
)

for obj in removed_pre_grounding:

    print(
        " -",
        obj.get("name"),
        "|",
        obj.get("grounding_phrase")
    )


print(
    "PHYSICAL INVENTORY RECORDS:",
    len(
        kept_objects
    )
)


# ============================================================
# EXPAND PHYSICAL INVENTORY
# INTO ONE OBSERVATION PER SOURCE REGION
# ============================================================

observations = []

observation_map = {}

observation_id = 1


for parent in kept_objects:

    parent_id = int(
        parent.get(
            "id"
        )
    )


    regions = parent.get(
        "source_regions"
    )


    if not regions:

        single_region = parent.get(
            "source_region"
        )

        regions = (
            [single_region]
            if single_region
            else []
        )


    # Remove duplicates without changing order.
    clean_regions = []

    for region in regions:

        region = str(
            region
        ).strip()

        if (
            region
            and
            region not in clean_regions
        ):
            clean_regions.append(
                region
            )


    # Important:
    # An object with no provenance is preserved in the report,
    # but cannot be crop-grounded by V4.5.
    if not clean_regions:

        observation_map[
            "unlocalized_parent_{}".format(
                parent_id
            )
        ] = {
            "parent_inventory_id":
                parent_id,

            "name":
                parent.get(
                    "name",
                    ""
                ),

            "status":
                "NO_SOURCE_REGION"
        }

        continue


    for region in clean_regions:

        obs = {
            "id":
                observation_id,

            "name":
                parent.get(
                    "name",
                    "object"
                ),

            "grounding_phrase":
                parent.get(
                    "grounding_phrase",
                    parent.get(
                        "name",
                        "object"
                    )
                ),

            "confidence":
                parent.get(
                    "confidence",
                    "low"
                ),

            # Required by V4.5.
            "discovery_pass":
                "3x3",

            "source_region":
                region,
        }


        observations.append(
            obs
        )


        observation_map[
            str(
                observation_id
            )
        ] = {
            "observation_id":
                observation_id,

            "parent_inventory_id":
                parent_id,

            "parent_name":
                parent.get(
                    "name",
                    ""
                ),

            "grounding_phrase":
                parent.get(
                    "grounding_phrase",
                    ""
                ),

            "source_region":
                region,
        }


        observation_id += 1


bridge_inventory = {
    "stage":
        "PRODUCTION_STAGE06B_GROUNDING_OBSERVATIONS",

    "method":
        (
            "Physical inventory expanded into one "
            "grounding observation per preserved 3x3 source region"
        ),

    "source_inventory":
        str(
            INPUT_INVENTORY
        ),

    "parent_inventory_count":
        len(
            kept_objects
        ),

    "observation_count":
        len(
            observations
        ),

    "objects":
        observations,
}


OBSERVATION_INVENTORY.write_text(
    json.dumps(
        bridge_inventory,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


OBSERVATION_MAP.write_text(
    json.dumps(
        observation_map,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# V4.5 requires this argument even though our production
# observations are all 3x3.
DUMMY_4X4_MANIFEST.write_text(
    "[]",
    encoding="utf-8"
)


print()
print(
    "GROUNDING OBSERVATIONS:",
    len(
        observations
    )
)


print(
    "OBSERVATION INVENTORY:",
    OBSERVATION_INVENTORY
)


# ============================================================
# LOAD FROZEN V4.5
# ============================================================

spec = importlib.util.spec_from_file_location(
    "test07_stage03_v45",
    LEGACY_SCRIPT
)

module = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    module
)


print()
print(
    "✅ VERIFIED V4.5 GROUNDING IMPLEMENTATION LOADED"
)


# ============================================================
# RUN V4.5
# ============================================================

module.run(
    master_path=
        DETECTION_IMAGE,

    inventory_json=
        OBSERVATION_INVENTORY,

    manifest_4x4=
        DUMMY_4X4_MANIFEST,

    output_dir=
        OUT,

    model_cache=
        MODEL_CACHE,
)


# ============================================================
# MAP OBSERVATION RESULTS BACK TO PHYSICAL PARENT IDS
# ============================================================

V45_RESULTS = (
    OUT
    / "01_source_crop_grounding_by_observation.json"
)

V45_FLAT = (
    OUT
    / "02_source_crop_dino_candidates_flat.json"
)


if not V45_RESULTS.exists():
    raise RuntimeError(
        "V4.5 observation result missing."
    )

if not V45_FLAT.exists():
    raise RuntimeError(
        "V4.5 flat candidate result missing."
    )


grounding_results = json.loads(
    V45_RESULTS.read_text(
        encoding="utf-8"
    )
)

flat_candidates = json.loads(
    V45_FLAT.read_text(
        encoding="utf-8"
    )
)


def parent_info(
    observation_id
):

    return observation_map.get(
        str(
            observation_id
        ),
        {}
    )


mapped_results = []


for row in grounding_results:

    new_row = dict(
        row
    )

    info = parent_info(
        row.get(
            "inventory_id"
        )
    )

    new_row[
        "observation_id"
    ] = row.get(
        "inventory_id"
    )

    new_row[
        "parent_inventory_id"
    ] = info.get(
        "parent_inventory_id"
    )

    new_row[
        "parent_inventory_name"
    ] = info.get(
        "parent_name"
    )

    mapped_results.append(
        new_row
    )


mapped_flat = []


for row in flat_candidates:

    new_row = dict(
        row
    )

    info = parent_info(
        row.get(
            "inventory_id"
        )
    )

    new_row[
        "observation_id"
    ] = row.get(
        "inventory_id"
    )

    new_row[
        "parent_inventory_id"
    ] = info.get(
        "parent_inventory_id"
    )

    new_row[
        "parent_inventory_name"
    ] = info.get(
        "parent_name"
    )

    mapped_flat.append(
        new_row
    )


MAPPED_RESULTS_PATH = (
    OUT
    / "04_parent_mapped_grounding.json"
)

MAPPED_FLAT_PATH = (
    OUT
    / "05_parent_mapped_dino_candidates_flat.json"
)


MAPPED_RESULTS_PATH.write_text(
    json.dumps(
        mapped_results,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


MAPPED_FLAT_PATH.write_text(
    json.dumps(
        mapped_flat,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# ============================================================
# PARENT-LEVEL SUMMARY
# ============================================================

parent_summary = {}


for parent in kept_objects:

    pid = int(
        parent["id"]
    )

    parent_summary[
        pid
    ] = {
        "id":
            pid,

        "name":
            parent.get(
                "name",
                ""
            ),

        "grounding_phrase":
            parent.get(
                "grounding_phrase",
                ""
            ),

        "observation_count":
            0,

        "grounded_observations":
            0,

        "dino_candidate_count":
            0,

        "best_score":
            None,
    }


for row in mapped_results:

    pid = row.get(
        "parent_inventory_id"
    )

    if pid not in parent_summary:
        continue


    parent_summary[
        pid
    ][
        "observation_count"
    ] += 1


    candidate_count = int(
        row.get(
            "candidate_count",
            0
        )
    )


    if candidate_count > 0:

        parent_summary[
            pid
        ][
            "grounded_observations"
        ] += 1


for row in mapped_flat:

    pid = row.get(
        "parent_inventory_id"
    )

    if pid not in parent_summary:
        continue


    parent_summary[
        pid
    ][
        "dino_candidate_count"
    ] += 1


    score = float(
        row.get(
            "score",
            0.0
        )
    )


    old_best = parent_summary[
        pid
    ][
        "best_score"
    ]


    if (
        old_best is None
        or
        score > old_best
    ):

        parent_summary[
            pid
        ][
            "best_score"
        ] = score


parent_rows = [
    parent_summary[k]
    for k in sorted(
        parent_summary
    )
]


PARENT_SUMMARY_PATH = (
    OUT
    / "06_parent_grounding_summary.json"
)


PARENT_SUMMARY_PATH.write_text(
    json.dumps(
        parent_rows,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


parents_with_candidates = sum(
    1
    for row in parent_rows
    if row[
        "dino_candidate_count"
    ] > 0
)


parents_without_candidates = (
    len(parent_rows)
    -
    parents_with_candidates
)


# ============================================================
# REPORT
# ============================================================

print()
print("=" * 110)
print("PRODUCTION STAGE 06B SUMMARY")
print("=" * 110)

print(
    "PHYSICAL INVENTORY ITEMS     :",
    len(
        kept_objects
    )
)

print(
    "GROUNDING OBSERVATIONS       :",
    len(
        observations
    )
)

print(
    "PARENTS WITH DINO CANDIDATES :",
    parents_with_candidates
)

print(
    "PARENTS WITHOUT CANDIDATES   :",
    parents_without_candidates
)

print(
    "TOTAL DINO CANDIDATES        :",
    len(
        mapped_flat
    )
)


print()
print("PER PHYSICAL INVENTORY ITEM:")


for row in parent_rows:

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
        "{:03d}. {} | observations={} | grounded={} | candidates={} | best={}".format(
            row["id"],
            row["name"],
            row["observation_count"],
            row["grounded_observations"],
            row["dino_candidate_count"],
            best_text
        )
    )


print()
print(
    "PREVIEW:",
    OUT
    / "03_source_crop_grounding_preview.png"
)

print(
    "PARENT SUMMARY:",
    PARENT_SUMMARY_PATH
)

print(
    "MAPPED RESULTS:",
    MAPPED_RESULTS_PATH
)

print(
    "MAPPED FLAT:",
    MAPPED_FLAT_PATH
)
