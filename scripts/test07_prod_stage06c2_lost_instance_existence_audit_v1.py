
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


ORIGINAL_MASTER = (
    STAGE01
    / "00_master_input.png"
)


SOURCE_INVENTORY = (
    STAGE06
    / "06a3_surface_filter"
    / "00_conservative_inventory.json"
)


SOURCE_VERIFIED = (
    STAGE06
    / "06c1b_original_master_verification"
    / "02_verified_candidates.json"
)


LEGACY_AUDIT = (
    SCRIPTS
    / "test07_stage045_lost_instance_existence_audit.py"
)


MODEL_CACHE = Path(
    "/workspace/data/huggingface-cache"
)


OUT = (
    STAGE06
    / "06c2_lost_instance_existence_audit"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# Production bridge files.
PHYSICAL_INVENTORY_PATH = (
    OUT
    / "00a_physical_inventory_bridge.json"
)


PARENT_VERIFIED_PATH = (
    OUT
    / "00b_parent_verified_bridge.json"
)


# ============================================================
# VALIDATE
# ============================================================

print("=" * 110)
print("PRODUCTION STAGE 06C2")
print("LOST PHYSICAL INSTANCE EXISTENCE AUDIT")
print("=" * 110)


required = {

    "STAGE01 ORIGINAL MASTER":
        ORIGINAL_MASTER,

    "06A3 INVENTORY":
        SOURCE_INVENTORY,

    "06C1B VERIFIED CANDIDATES":
        SOURCE_VERIFIED,

    "FROZEN STAGE04.5 AUDIT":
        LEGACY_AUDIT,

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
# BUILD 28-ITEM PHYSICAL INVENTORY
#
# Remove only deterministic non-physical room corner.
# Everything else remains eligible for existence audit.
# ============================================================

inventory_data = json.loads(
    SOURCE_INVENTORY.read_text(
        encoding="utf-8"
    )
)


objects = inventory_data.get(
    "objects",
    []
)


physical_objects = []

removed_nonphysical = []


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

        removed_nonphysical.append(
            obj
        )

        continue


    physical_objects.append(
        obj
    )


physical_inventory = {

    "stage":
        "PRODUCTION_STAGE06C2_PHYSICAL_INVENTORY_BRIDGE",

    "source":
        str(
            SOURCE_INVENTORY
        ),

    "objects":
        physical_objects,
}


PHYSICAL_INVENTORY_PATH.write_text(
    json.dumps(
        physical_inventory,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


print()
print(
    "06A3 INVENTORY:",
    len(objects)
)

print(
    "NON-PHYSICAL REMOVED:",
    len(
        removed_nonphysical
    )
)

print(
    "PHYSICAL INVENTORY:",
    len(
        physical_objects
    )
)


# ============================================================
# CONVERT VERIFIED OBSERVATION IDS
# INTO VERIFIED PHYSICAL PARENT IDS
#
# Stage04.5 expects:
# row["inventory_id"] == physical inventory id
# ============================================================

verified_candidates = json.loads(
    SOURCE_VERIFIED.read_text(
        encoding="utf-8"
    )
)


parent_best = {}


for row in verified_candidates:

    pid_raw = row.get(
        "parent_inventory_id"
    )


    if pid_raw is None:
        continue


    pid = int(
        pid_raw
    )


    score = float(
        row.get(
            "score",
            0.0
        )
    )


    previous = parent_best.get(
        pid
    )


    if (
        previous is None
        or
        score
        >
        float(
            previous.get(
                "score",
                0.0
            )
        )
    ):

        bridge = dict(
            row
        )


        # Critical compatibility conversion:
        bridge[
            "source_observation_inventory_id"
        ] = row.get(
            "inventory_id"
        )


        bridge[
            "inventory_id"
        ] = pid


        parent_best[
            pid
        ] = bridge


parent_verified = [

    parent_best[
        pid
    ]

    for pid in sorted(
        parent_best
    )
]


PARENT_VERIFIED_PATH.write_text(
    json.dumps(
        parent_verified,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


print(
    "RAW VERIFIED BOXES:",
    len(
        verified_candidates
    )
)

print(
    "UNIQUE VERIFIED PHYSICAL PARENTS:",
    len(
        parent_verified
    )
)


print()
print(
    "ALREADY VERIFIED PHYSICAL IDS:",
    [
        row[
            "inventory_id"
        ]
        for row in parent_verified
    ]
)


# ============================================================
# PREDICT AUDIT COUNT
# ============================================================

physical_ids = {

    int(
        obj[
            "id"
        ]
    )

    for obj in physical_objects
}


already_valid_ids = {

    int(
        row[
            "inventory_id"
        ]
    )

    for row in parent_verified
}


lost_ids = sorted(
    physical_ids
    -
    already_valid_ids
)


print()
print(
    "EXPECTED LOST TO AUDIT:",
    len(
        lost_ids
    )
)

print(
    "LOST IDS:",
    lost_ids
)


# ============================================================
# LOAD FROZEN STAGE04.5
# ============================================================

spec = (
    importlib.util
    .spec_from_file_location(
        "test07_stage045_existence_audit",
        LEGACY_AUDIT
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
    "✅ FROZEN STAGE04.5 IMPLEMENTATION LOADED"
)


# ============================================================
# RUN EXISTENCE AUDIT
#
# IMPORTANT:
# FULL ORIGINAL STAGE01 IMAGE
# ============================================================

module.run(

    master_path=
        ORIGINAL_MASTER,

    inventory_json=
        PHYSICAL_INVENTORY_PATH,

    verified_json=
        PARENT_VERIFIED_PATH,

    output_dir=
        OUT,

    model_cache=
        MODEL_CACHE,
)


# ============================================================
# READ RESULT
# ============================================================

AUDIT_PATH = (
    OUT
    / "00_existence_audit.json"
)


if not AUDIT_PATH.exists():

    raise RuntimeError(
        "Stage04.5 existence audit output missing."
    )


audit = json.loads(
    AUDIT_PATH.read_text(
        encoding="utf-8"
    )
)


# ============================================================
# COUNTS
# ============================================================

present = [
    r
    for r in audit
    if r.get(
        "decision"
    ) == "PRESENT"
]


absent = [
    r
    for r in audit
    if r.get(
        "decision"
    ) == "ABSENT"
]


uncertain = [
    r
    for r in audit
    if r.get(
        "decision"
    ) == "UNCERTAIN"
]


# ============================================================
# CREATE PRODUCTION STATE
# ============================================================

state = {

    "stage":
        "PRODUCTION_STAGE06C2",

    "method":
        (
            "Qwen2.5-VL-7B full-original-image "
            "existence audit for physical inventory "
            "instances lacking verified localization"
        ),

    "physical_inventory_count":
        len(
            physical_objects
        ),

    "already_verified_parent_count":
        len(
            parent_verified
        ),

    "lost_audited_count":
        len(
            audit
        ),

    "present_count":
        len(
            present
        ),

    "absent_count":
        len(
            absent
        ),

    "uncertain_count":
        len(
            uncertain
        ),

    "already_verified_ids":
        sorted(
            already_valid_ids
        ),

    "present_ids":
        [
            int(
                r[
                    "inventory_id"
                ]
            )
            for r in present
        ],

    "absent_ids":
        [
            int(
                r[
                    "inventory_id"
                ]
            )
            for r in absent
        ],

    "uncertain_ids":
        [
            int(
                r[
                    "inventory_id"
                ]
            )
            for r in uncertain
        ],

    "rules": [

        "Stage01 original master is existence truth.",

        "Previous invalid DINO boxes do not imply object absence.",

        "PRESENT objects require localization rescue.",

        "ABSENT objects are candidate hallucinations but remain recorded.",

        "UNCERTAIN objects remain unresolved and are not deleted.",

        "No coordinates are generated in this stage.",

        "No SAM2 is performed in this stage.",
    ],
}


STATE_PATH = (
    OUT
    / "01_production_existence_state.json"
)


STATE_PATH.write_text(
    json.dumps(
        state,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# ============================================================
# HUMAN REPORT
# ============================================================

print()
print("=" * 110)
print("PRODUCTION STAGE 06C2 RESULT")
print("=" * 110)

print(
    "PHYSICAL INVENTORY:",
    len(
        physical_objects
    )
)

print(
    "ALREADY VERIFIED:",
    len(
        parent_verified
    )
)

print(
    "LOST AUDITED:",
    len(
        audit
    )
)

print()

print(
    "PRESENT:",
    len(
        present
    )
)

print(
    "ABSENT:",
    len(
        absent
    )
)

print(
    "UNCERTAIN:",
    len(
        uncertain
    )
)


print()
print(
    "LOST INSTANCE LIST:"
)


for row in audit:

    print(
        "{:03d}. {} | {} | {} | {}".format(

            int(
                row[
                    "inventory_id"
                ]
            ),

            row.get(
                "name",
                ""
            ),

            row.get(
                "decision",
                ""
            ),

            row.get(
                "confidence",
                ""
            ),

            row.get(
                "grounding_phrase",
                ""
            )
        )
    )


print()
print(
    "AUDIT JSON:",
    AUDIT_PATH
)

print(
    "PRODUCTION STATE:",
    STATE_PATH
)

print(
    "PHYSICAL INVENTORY BRIDGE:",
    PHYSICAL_INVENTORY_PATH
)

print(
    "PARENT VERIFIED BRIDGE:",
    PARENT_VERIFIED_PATH
)
