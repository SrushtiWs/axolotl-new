
from pathlib import Path
import json


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


STRICT_VERIFIED_JSON = (
    STAGE06
    / "06c1_candidate_verification"
    / "00_verified_candidates.json"
)


FLORENCE_FAUCET_RESULT = (
    STAGE06
    / "06c10b_florence_faucet_sam2_duplicate_state"
    / "01_stage06c10b_result.json"
)


HANDLE_RESULT = (
    STAGE06
    / "06c11c_florence_handle_sam2"
    / "00_stage06c11c_result.json"
)


SINK_RESULT = (
    STAGE06
    / "06c7a_sam2_alternative_candidate_evaluation"
    / "00_sam2_candidate_evaluation.json"
)


CONSENSUS_JSON = (
    STAGE06
    / "06c6_multi_route_candidate_consensus"
    / "00_multi_route_consensus_all.json"
)


OUT = (
    STAGE06
    / "06c12a_resolution_state_consolidation"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# HELPERS
# ============================================================

def load_json(
    path,
    default=None
):

    if path.exists():

        return json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

    return default


def normalize_inventory(
    data
):

    if isinstance(
        data,
        dict
    ):

        if "objects" in data:

            return data[
                "objects"
            ]

        return list(
            data.values()
        )

    return data


def get_id(
    row
):

    value = row.get(
        "id",
        row.get(
            "inventory_id"
        )
    )

    return (
        int(value)
        if value is not None
        else None
    )


# ============================================================
# LOAD CORE INVENTORY
# ============================================================

if not INVENTORY_JSON.exists():

    raise FileNotFoundError(
        INVENTORY_JSON
    )


inventory_data = load_json(
    INVENTORY_JSON
)

inventory = normalize_inventory(
    inventory_data
)


# ============================================================
# EXISTENCE STATE
# ============================================================

existence_data = load_json(
    EXISTENCE_JSON,
    []
)


existence_by_id = {}


if isinstance(
    existence_data,
    list
):

    for row in existence_data:

        iid = get_id(
            row
        )

        if iid is None:
            continue

        decision = str(
            row.get(
                "decision",
                row.get(
                    "existence",
                    row.get(
                        "status",
                        ""
                    )
                )
            )
        ).upper()

        existence_by_id[
            iid
        ] = decision


# ============================================================
# KNOWN STRICT LOCALIZED IDS
#
# From successful old strict verifier:
# 005,009,013,014,029
#
# We retain these as resolved localization seeds.
# ============================================================

STRICT_RESOLVED_IDS = {
    5,
    9,
    13,
    14,
    29,
}


# ============================================================
# LOAD NEW RECOVERY RESULTS
# ============================================================

faucet_result = load_json(
    FLORENCE_FAUCET_RESULT,
    {}
)

handle_result = load_json(
    HANDLE_RESULT,
    {}
)

sink_rows = load_json(
    SINK_RESULT,
    []
)

consensus_rows = load_json(
    CONSENSUS_JSON,
    []
)


# ============================================================
# SINK 024 PROVISIONAL
# ============================================================

sink024 = None

for row in sink_rows:

    if (
        int(
            row.get(
                "inventory_id",
                -1
            )
        )
        ==
        24
        and
        int(
            row.get(
                "cluster_rank",
                -1
            )
        )
        ==
        1
    ):

        sink024 = row
        break


# ============================================================
# DUPLICATE STATE
# ============================================================

duplicate_map = {

    25: {
        "duplicate_of":
            10,

        "status":
            "POSSIBLE_DUPLICATE",

        "evidence":
            {
                "florence_box_iou":
                    (
                        faucet_result
                        .get(
                            "duplicate",
                            {}
                        )
                        .get(
                            "florence_iou_with_010"
                        )
                    ),

                "same_qwen_grid_region":
                    True,

                "segmented_separately":
                    False,
            }
    }
}


# ============================================================
# ARCHITECTURAL REVIEW IDS
#
# Door-like instances are retained but not merged into props
# until architectural classification.
# ============================================================

ARCHITECTURAL_REVIEW_IDS = {
    1,
}


# ============================================================
# BUILD CANONICAL STATE
# ============================================================

states = []


for row in inventory:

    iid = get_id(
        row
    )

    if iid is None:
        continue


    name = str(
        row.get(
            "name",
            row.get(
                "inventory_name",
                ""
            )
        )
    )


    existence = existence_by_id.get(
        iid,
        ""
    )


    state = {

        "inventory_id":
            iid,

        "inventory_name":
            name,

        "existence_state":
            existence,

        "resolution_state":
            None,

        "mask_ready":
            False,

        "mask_path":
            None,

        "rgba_path":
            None,

        "source_stage":
            None,

        "duplicate_of":
            None,

        "notes":
            [],
    }


    # --------------------------------------------------------
    # ABSENT
    # --------------------------------------------------------

    if existence == "ABSENT":

        state[
            "resolution_state"
        ] = "ABSENT"

        state[
            "notes"
        ].append(
            "excluded by full-scene existence audit"
        )


    # --------------------------------------------------------
    # DUPLICATE
    # --------------------------------------------------------

    elif iid in duplicate_map:

        dup = duplicate_map[
            iid
        ]

        state[
            "resolution_state"
        ] = dup[
            "status"
        ]

        state[
            "duplicate_of"
        ] = dup[
            "duplicate_of"
        ]

        state[
            "source_stage"
        ] = "06C10B"

        state[
            "notes"
        ].append(
            "do not segment separately before physical-instance audit"
        )


    # --------------------------------------------------------
    # ARCHITECTURAL REVIEW
    # --------------------------------------------------------

    elif iid in ARCHITECTURAL_REVIEW_IDS:

        state[
            "resolution_state"
        ] = "ARCHITECTURAL_REVIEW"

        state[
            "notes"
        ].append(
            "door-like object retained but may belong to room architecture rather than prop layer"
        )


    # --------------------------------------------------------
    # STRICT RESOLVED OLD SEEDS
    # --------------------------------------------------------

    elif iid in STRICT_RESOLVED_IDS:

        state[
            "resolution_state"
        ] = "RESOLVED_LOCALIZATION"

        state[
            "source_stage"
        ] = "06C1_STRICT_VERIFIED"

        state[
            "notes"
        ].append(
            "trusted localization seed exists; final production SAM2 mask still needs consolidation"
        )


    # --------------------------------------------------------
    # 010 FAUCET
    # --------------------------------------------------------

    elif iid == 10:

        primary = faucet_result.get(
            "primary",
            {}
        )

        geometry_pass = bool(
            primary.get(
                "geometry_pass",
                False
            )
        )

        if geometry_pass:

            state[
                "resolution_state"
            ] = "RESOLVED"

            state[
                "mask_ready"
            ] = True

            state[
                "mask_path"
            ] = primary.get(
                "mask_path"
            )

            state[
                "rgba_path"
            ] = primary.get(
                "rgba_path"
            )

            state[
                "source_stage"
            ] = "06C10B"

            state[
                "notes"
            ].append(
                "Florence localization + SAM2 mask passed geometry and visual audit"
            )

        else:

            state[
                "resolution_state"
            ] = "UNRESOLVED"


    # --------------------------------------------------------
    # 021 DRAWER HANDLE
    # --------------------------------------------------------

    elif iid == 21:

        geometry_pass = bool(
            handle_result.get(
                "geometry_pass",
                False
            )
        )

        if geometry_pass:

            state[
                "resolution_state"
            ] = "RESOLVED"

            state[
                "mask_ready"
            ] = True

            state[
                "mask_path"
            ] = handle_result.get(
                "mask_path"
            )

            state[
                "rgba_path"
            ] = handle_result.get(
                "rgba_path"
            )

            state[
                "source_stage"
            ] = "06C11C"

            state[
                "notes"
            ].append(
                "parent-guided Florence localization + SAM2 passed visual and geometry audit"
            )

        else:

            state[
                "resolution_state"
            ] = "UNRESOLVED"


    # --------------------------------------------------------
    # 024 SINK
    # --------------------------------------------------------

    elif iid == 24:

        if (
            sink024 is not None
            and
            bool(
                sink024.get(
                    "passes_geometry_gate",
                    False
                )
            )
        ):

            state[
                "resolution_state"
            ] = "PROVISIONAL"

            state[
                "mask_ready"
            ] = True

            state[
                "mask_path"
            ] = sink024.get(
                "mask_path"
            )

            state[
                "rgba_path"
            ] = sink024.get(
                "rgba_path"
            )

            state[
                "source_stage"
            ] = "06C7A"

            state[
                "notes"
            ].append(
                "sink cluster #1 visually plausible and SAM2 geometry-safe; awaiting final physical-instance audit"
            )

        else:

            state[
                "resolution_state"
            ] = "UNRESOLVED"


    # --------------------------------------------------------
    # ALL OTHER PRESENT OBJECTS
    # --------------------------------------------------------

    else:

        if existence == "PRESENT":

            state[
                "resolution_state"
            ] = "UNRESOLVED"

            state[
                "notes"
            ].append(
                "confirmed present but no final trustworthy production mask yet"
            )

        elif existence == "UNCERTAIN":

            state[
                "resolution_state"
            ] = "UNRESOLVED_EXISTENCE"

        else:

            state[
                "resolution_state"
            ] = "UNCLASSIFIED"


    states.append(
        state
    )


# ============================================================
# SUMMARY
# ============================================================

summary = {

    "RESOLVED": [],
    "RESOLVED_LOCALIZATION": [],
    "PROVISIONAL": [],
    "POSSIBLE_DUPLICATE": [],
    "ARCHITECTURAL_REVIEW": [],
    "UNRESOLVED": [],
    "UNRESOLVED_EXISTENCE": [],
    "ABSENT": [],
    "UNCLASSIFIED": [],
}


for row in states:

    key = row[
        "resolution_state"
    ]

    if key not in summary:

        summary[
            key
        ] = []

    summary[
        key
    ].append(
        row[
            "inventory_id"
        ]
    )


# ============================================================
# SAVE
# ============================================================

RESULT = {

    "stage":
        "06C12A",

    "purpose":
        "canonical physical-instance resolution state before final prop mask consolidation",

    "summary":
        summary,

    "objects":
        states,

    "rules": [
        "no masks were generated in this stage",
        "no masks were unioned",
        "duplicate instances are not segmented twice",
        "architectural-review objects are not merged into prop layer",
        "exact RGB remains sourced from Stage01 master"
    ]
}


RESULT_PATH = (
    OUT
    / "00_resolution_state.json"
)


RESULT_PATH.write_text(
    json.dumps(
        RESULT,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


REPORT_PATH = (
    OUT
    / "01_resolution_report.txt"
)


lines = []

lines.append(
    "=" * 100
)

lines.append(
    "TEST07 PRODUCTION STAGE 06C12A"
)

lines.append(
    "PHYSICAL INSTANCE RESOLUTION STATE"
)

lines.append(
    "=" * 100
)


for key, ids in summary.items():

    lines.append(
        f"{key:24s}: {ids}"
    )


lines.append(
    ""
)

lines.append(
    "-" * 100
)

lines.append(
    "OBJECT STATES"
)

lines.append(
    "-" * 100
)


for row in states:

    lines.append(
        "{:03d}. {:28s} | {:24s} | mask_ready={} | source={}".format(

            row[
                "inventory_id"
            ],

            row[
                "inventory_name"
            ][:28],

            row[
                "resolution_state"
            ],

            row[
                "mask_ready"
            ],

            row[
                "source_stage"
            ],
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
print("PRODUCTION STAGE 06C12A RESULT")
print("=" * 110)


for key, ids in summary.items():

    print(
        f"{key:24s}:",
        ids
    )


print()
print("-" * 110)


for row in states:

    print(
        "{:03d}. {:26s} | {:22s} | mask={} | source={}".format(

            row[
                "inventory_id"
            ],

            row[
                "inventory_name"
            ][:26],

            row[
                "resolution_state"
            ],

            row[
                "mask_ready"
            ],

            row[
                "source_stage"
            ],
        )
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
