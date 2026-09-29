
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

SOURCE_JSON = (
    STAGE06
    / "06c12b_strict_localization_sam2"
    / "00_stage06c12b_results.json"
)

OUT = (
    STAGE06
    / "06c12c_visual_audit_state"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


if not SOURCE_JSON.exists():

    raise FileNotFoundError(
        SOURCE_JSON
    )


rows = json.loads(
    SOURCE_JSON.read_text(
        encoding="utf-8"
    )
)


# ============================================================
# VISUAL DECISIONS
# ============================================================

DECISIONS = {

    5: {
        "decision":
            "REJECT",

        "next_state":
            "INSTANCE_HIERARCHY_REVIEW",

        "reason":
            (
                "SAM2 mask covers a broad lower-left vanity/"
                "cabinet region and is not a trustworthy "
                "single physical cabinet instance."
            ),
    },

    9: {
        "decision":
            "REJECT",

        "next_state":
            "RELOCALIZE",

        "reason":
            (
                "high SAM2 geometry score but localization is "
                "on the wrong left-side fixture; it is not the "
                "actual wall switch visible on the tiled wall."
            ),
    },

    13: {
        "decision":
            "ACCEPT",

        "next_state":
            "RESOLVED",

        "reason":
            (
                "SAM2 contour visually matches the physical "
                "toilet seat and passed geometry checks."
            ),
    },

    14: {
        "decision":
            "ACCEPT",

        "next_state":
            "RESOLVED",

        "reason":
            (
                "SAM2 contour visually matches the physical "
                "toilet and passed geometry checks."
            ),
    },

    29: {
        "decision":
            "REJECT",

        "next_state":
            "INSTANCE_HIERARCHY_REVIEW",

        "reason":
            (
                "SAM2 mask represents a large vanity/cabinet "
                "region rather than a trustworthy individual "
                "drawer instance."
            ),
    },
}


# ============================================================
# BUILD AUDIT
# ============================================================

audited = []


for row in rows:

    iid = int(
        row[
            "inventory_id"
        ]
    )

    decision = DECISIONS[
        iid
    ]


    item = dict(
        row
    )


    item[
        "visual_audit_decision"
    ] = decision[
        "decision"
    ]

    item[
        "production_resolution_state"
    ] = decision[
        "next_state"
    ]

    item[
        "visual_audit_reason"
    ] = decision[
        "reason"
    ]


    # --------------------------------------------------------
    # Mask is production-ready only after semantic visual pass
    # --------------------------------------------------------

    item[
        "production_mask_ready"
    ] = bool(
        decision[
            "decision"
        ]
        ==
        "ACCEPT"
    )


    audited.append(
        item
    )


# ============================================================
# SUMMARY
# ============================================================

summary = {

    "ACCEPTED_RESOLVED": [
        row[
            "inventory_id"
        ]
        for row in audited
        if row[
            "production_resolution_state"
        ]
        ==
        "RESOLVED"
    ],

    "RELOCALIZE": [
        row[
            "inventory_id"
        ]
        for row in audited
        if row[
            "production_resolution_state"
        ]
        ==
        "RELOCALIZE"
    ],

    "INSTANCE_HIERARCHY_REVIEW": [
        row[
            "inventory_id"
        ]
        for row in audited
        if row[
            "production_resolution_state"
        ]
        ==
        "INSTANCE_HIERARCHY_REVIEW"
    ],
}


# ============================================================
# VANITY COMPONENT GROUP HYPOTHESIS
#
# This is evidence bookkeeping only.
# It does NOT merge any objects automatically.
# ============================================================

hierarchy_hypothesis = {

    "group_name":
        "vanity_system_candidate_group",

    "member_inventory_ids": [
        5,
        21,
        23,
        29,
    ],

    "member_roles": {

        "5":
            "cabinet candidate",

        "21":
            "drawer/cabinet handle component",

        "23":
            "cabinet candidate",

        "29":
            "drawer component",
    },

    "status":
        "REQUIRES_PHYSICAL_INSTANCE_HIERARCHY_AUDIT",

    "rule":
        (
            "do not independently union overlapping parent/"
            "child masks until physical hierarchy is resolved"
        )
}


# ============================================================
# SAVE
# ============================================================

result = {

    "stage":
        "06C12C",

    "purpose":
        (
            "semantic visual audit of geometry-safe SAM2 masks"
        ),

    "summary":
        summary,

    "objects":
        audited,

    "hierarchy_hypotheses": [
        hierarchy_hypothesis
    ],

    "production_rules": [
        (
            "SAM2 geometry pass alone is not semantic proof"
        ),
        (
            "rejected masks are preserved but must not enter "
            "the production prop union"
        ),
        (
            "parent and child components must not be blindly "
            "unioned as independent physical instances"
        ),
        (
            "exact RGB remains sourced from Stage01 master"
        ),
    ],
}


RESULT_PATH = (
    OUT
    / "00_visual_audit_state.json"
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
    / "01_visual_audit_report.txt"
)


lines = [

    "=" * 100,

    "TEST07 PRODUCTION STAGE 06C12C",

    "STRICT-SEED SAM2 VISUAL AUDIT",

    "=" * 100,

    "",

]


for row in audited:

    lines.append(

        "{:03d}. {:20s} | visual={} | next={} | mask_ready={}".format(

            row[
                "inventory_id"
            ],

            row[
                "inventory_name"
            ][:20],

            row[
                "visual_audit_decision"
            ],

            row[
                "production_resolution_state"
            ],

            row[
                "production_mask_ready"
            ],
        )
    )


lines.extend(
    [
        "",
        "-" * 100,
        "HIERARCHY HYPOTHESIS",
        "-" * 100,
        json.dumps(
            hierarchy_hypothesis,
            indent=2
        ),
    ]
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
print("PRODUCTION STAGE 06C12C RESULT")
print("=" * 110)


for row in audited:

    print(

        "{:03d}. {:20s} | VISUAL={} | STATE={} | MASK_READY={}".format(

            row[
                "inventory_id"
            ],

            row[
                "inventory_name"
            ][:20],

            row[
                "visual_audit_decision"
            ],

            row[
                "production_resolution_state"
            ],

            row[
                "production_mask_ready"
            ],
        )
    )


print()
print(
    "ACCEPTED:",
    summary[
        "ACCEPTED_RESOLVED"
    ]
)

print(
    "RELOCALIZE:",
    summary[
        "RELOCALIZE"
    ]
)

print(
    "HIERARCHY REVIEW:",
    summary[
        "INSTANCE_HIERARCHY_REVIEW"
    ]
)


print()
print(
    "VANITY GROUP:",
    hierarchy_hypothesis[
        "member_inventory_ids"
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
