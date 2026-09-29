
from pathlib import Path
import json


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


SOURCE = (
    STAGE06
    / "06c14b_semantic_audit_sam2_inline_preview"
    / "00_stage06c14b_state.json"
)


OUT = (
    STAGE06
    / "06c14c_visual_state_freeze"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


if not SOURCE.exists():

    raise FileNotFoundError(
        SOURCE
    )


source = json.loads(
    SOURCE.read_text(
        encoding="utf-8"
    )
)


resolved_by_id = {

    int(row["inventory_id"]):
        row

    for row in source[
        "resolved_candidates"
    ]
}


# ============================================================
# FREEZE VISUAL DECISIONS
# ============================================================

objects = []


# ------------------------------------------------------------
# 002 RECESSED LIGHT
# ------------------------------------------------------------

row = resolved_by_id[2]

objects.append({

    "inventory_id":
        2,

    "inventory_name":
        "recessed light",

    "state":
        "RESOLVED",

    "visual_audit":
        "ACCEPT",

    "mask_ready":
        True,

    "mask_path":
        row[
            "mask_path"
        ],

    "rgba_path":
        row[
            "rgba_path"
        ],

    "source_stage":
        "06C14B",

    "reason":
        (
            "Florence candidate corresponds to actual "
            "recessed ceiling light and SAM2 contour "
            "visually follows the physical fixture."
        ),
})


# ------------------------------------------------------------
# 008 WALL OUTLET
# ------------------------------------------------------------

objects.append({

    "inventory_id":
        8,

    "inventory_name":
        "wall outlet",

    "state":
        "DUPLICATE",

    "duplicate_of":
        9,

    "mask_ready":
        False,

    "source_stage":
        "06C14B",

    "reason":
        (
            "Florence localization is identical to physical "
            "instance 009; 008↔009 bbox IoU is 1.0. "
            "Do not create a second mask."
        ),
})


# ------------------------------------------------------------
# 012 TOILET PAPER HOLDER
# ------------------------------------------------------------

row = resolved_by_id[12]

objects.append({

    "inventory_id":
        12,

    "inventory_name":
        "toilet paper holder",

    "state":
        "RESOLVED",

    "visual_audit":
        "ACCEPT",

    "mask_ready":
        True,

    "mask_path":
        row[
            "mask_path"
        ],

    "rgba_path":
        row[
            "rgba_path"
        ],

    "source_stage":
        "06C14B",

    "reason":
        (
            "Florence candidate corresponds to actual "
            "wall-mounted toilet-paper-holder fixture and "
            "SAM2 contour visually follows the fixture."
        ),
})


# ------------------------------------------------------------
# 015 TOWEL BAR
# ------------------------------------------------------------

objects.append({

    "inventory_id":
        15,

    "inventory_name":
        "towel bar",

    "state":
        "SPECIAL_LAYER_COMPONENT",

    "layer_owner":
        "STAGE04_GLASS_SYSTEM",

    "mask_ready_for_prop_layer":
        False,

    "source_stage":
        "06C14B",

    "reason":
        (
            "Detected horizontal bar is physically attached "
            "to the glass enclosure. It must be handled with "
            "the glass layer and not independently composited "
            "as a normal prop."
        ),
})


# ============================================================
# GLOBAL CURRENT STATE
# ============================================================

current_state = {

    "resolved_normal_props": [
        2,
        9,
        10,
        12,
        13,
        14,
        21,
    ],

    "duplicate_instances": {
        "8": 9,
        "25": 10,
    },

    "special_layer_components": {
        "15":
            "STAGE04_GLASS_SYSTEM",
    },

    "provisional": [
        24,
    ],

    "architectural_review": [
        1,
    ],

    "vanity_hierarchy_review": [
        5,
        23,
        29,
    ],
}


result = {

    "stage":
        "06C14C",

    "objects":
        objects,

    "current_state":
        current_state,

    "production_rules": [

        (
            "002 and 012 are visually accepted production "
            "masks."
        ),

        (
            "008 must reuse physical instance 009 rather "
            "than create a duplicate prop."
        ),

        (
            "015 belongs to Stage04 glass-system ownership."
        ),

        (
            "No duplicate or special-layer object may enter "
            "the normal prop-mask union independently."
        ),

        (
            "Exact RGB remains sourced from Stage01 master."
        ),
    ],
}


RESULT_PATH = (
    OUT
    / "00_stage06c14c_state.json"
)


RESULT_PATH.write_text(
    json.dumps(
        result,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


REPORT_PATH = (
    OUT
    / "01_stage06c14c_report.txt"
)


lines = [

    "=" * 100,

    "TEST07 PRODUCTION STAGE 06C14C",

    "VISUAL STATE FREEZE",

    "=" * 100,

    "",

]


for row in objects:

    lines.append(

        "{:03d}. {:24s} | STATE={} | MASK_READY={}".format(

            row[
                "inventory_id"
            ],

            row[
                "inventory_name"
            ][:24],

            row[
                "state"
            ],

            row.get(
                "mask_ready",
                row.get(
                    "mask_ready_for_prop_layer",
                    False
                )
            ),
        )
    )


REPORT_PATH.write_text(
    "\n".join(
        lines
    ),
    encoding="utf-8"
)


# ============================================================
# PRINT
# ============================================================

print("=" * 110)
print("PRODUCTION STAGE 06C14C RESULT")
print("=" * 110)


for row in objects:

    print(

        "{:03d}. {:24s} | STATE={} | MASK_READY={}".format(

            row[
                "inventory_id"
            ],

            row[
                "inventory_name"
            ][:24],

            row[
                "state"
            ],

            row.get(
                "mask_ready",
                row.get(
                    "mask_ready_for_prop_layer",
                    False
                )
            ),
        )
    )


print()
print("-" * 110)

print(
    "RESOLVED NORMAL PROPS:",
    current_state[
        "resolved_normal_props"
    ]
)

print(
    "DUPLICATES:",
    current_state[
        "duplicate_instances"
    ]
)

print(
    "SPECIAL LAYERS:",
    current_state[
        "special_layer_components"
    ]
)

print(
    "PROVISIONAL:",
    current_state[
        "provisional"
    ]
)

print(
    "ARCHITECTURAL REVIEW:",
    current_state[
        "architectural_review"
    ]
)

print(
    "VANITY HIERARCHY REVIEW:",
    current_state[
        "vanity_hierarchy_review"
    ]
)


print()
print(
    "RESULT JSON:",
    RESULT_PATH
)

print(
    "REPORT:",
    REPORT_PATH
)

print()
print(
    "NO FINAL PROP MASK UNION WAS CREATED."
)
