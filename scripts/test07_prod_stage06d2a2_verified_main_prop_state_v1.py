
from pathlib import Path
import json


BASE = Path(
    "/workspace/axolotl"
)

PROD = (
    BASE
    / "test07"
    / "production_pipeline"
)

STAGE06 = (
    PROD
    / "stage06_prop_layer"
)


MASTER_PATH = (
    PROD
    / "stage05_clean_room_with_props"
    / "12_final_canonical_prop_detection_master.png"
)


FAILED_D2A_PATH = (
    STAGE06
    / "06d2a_final_connected_main_prop_inventory"
    / "00_stage06d2a_final_main_prop_inventory.json"
)


OUT = (
    STAGE06
    / "06d2a2_verified_main_prop_state"
)


OUT.mkdir(
    parents=True,
    exist_ok=True
)


for path in [
    MASTER_PATH,
    FAILED_D2A_PATH,
]:

    if not path.exists():

        raise FileNotFoundError(
            path
        )


# ============================================================
# HUMAN-VISUALLY-VERIFIED MAIN-PROP STATE
# ============================================================

MAIN_PROPS = [

    {
        "main_prop_id":
            "P01",

        "main_prop_name":
            "complete vanity system",

        "prop_type":
            "CONNECTED_SYSTEM",

        "visible_members": [

            "vanity cabinet/body",

            "cabinet doors/drawers",

            "attached cabinet handles/knobs",

            "countertop",

            "sink/basin",

            "sink faucet",

            "hand-wash bottle",

        ],

        "grouping_rule":
            (
                "All visible non-architectural items are "
                "physically attached to, integrated into, "
                "or resting directly on the vanity system."
            ),

        "final_extraction":
            "ONE_PROP_LAYER",
    },


    {
        "main_prop_id":
            "P02",

        "main_prop_name":
            "complete toilet system",

        "prop_type":
            "CONNECTED_SYSTEM",

        "visible_members": [

            "toilet body",

            "toilet seat",

            "toilet lid/seat cover",

            "physically attached toilet components",

        ],

        "explicit_exclusions": [

            "toilet paper holder",

            "toilet tank",

        ],

        "grouping_rule":
            (
                "Only the physically continuous visible "
                "toilet assembly belongs to this prop."
            ),

        "final_extraction":
            "ONE_PROP_LAYER",
    },


    {
        "main_prop_id":
            "P03",

        "main_prop_name":
            "shower fixture",

        "prop_type":
            "CONNECTED_SYSTEM",

        "visible_members": [

            "shower arm",

            "shower head",

        ],

        "explicit_exclusions": [

            "glass enclosure",

            "glass hardware",

            "electrical plate",

            "ceiling cable",

        ],

        "grouping_rule":
            (
                "The visible shower arm and shower head "
                "form one physically connected fixture."
            ),

        "final_extraction":
            "ONE_PROP_LAYER",
    },


    {
        "main_prop_id":
            "P04",

        "main_prop_name":
            "wall electrical plate",

        "prop_type":
            "STANDALONE_FIXTURE",

        "visible_members": [

            "wall electrical plate",

        ],

        "grouping_rule":
            (
                "Mounted to the architectural wall but "
                "not physically connected to another prop."
            ),

        "final_extraction":
            "ONE_PROP_LAYER",
    },


    {
        "main_prop_id":
            "P05",

        "main_prop_name":
            "toilet paper holder",

        "prop_type":
            "STANDALONE_FIXTURE",

        "visible_members": [

            "wall-mounted toilet paper holder",

        ],

        "grouping_rule":
            (
                "Physically separate from the toilet; "
                "wall mounting does not merge it with "
                "either wall or toilet."
            ),

        "final_extraction":
            "ONE_PROP_LAYER",
    },


    {
        "main_prop_id":
            "P06",

        "main_prop_name":
            "ceiling light",

        "prop_type":
            "STANDALONE_FIXTURE",

        "visible_members": [

            "recessed ceiling light",

        ],

        "grouping_rule":
            (
                "Mounted to ceiling but physically separate "
                "from every other non-architectural prop."
            ),

        "final_extraction":
            "ONE_PROP_LAYER",
    },

]


# ============================================================
# REJECTED / NON-PROP CONCEPTS
# ============================================================

REJECTED_CONCEPTS = [

    {
        "concept":
            "toilet tank",

        "reason":
            "No visible toilet tank exists in this image.",
    },

    {
        "concept":
            "ceiling-mounted cable",

        "reason":
            (
                "False/noisy discovery associated with "
                "architectural edge/structure."
            ),
    },

    {
        "concept":
            "wire",

        "reason":
            (
                "False/noisy discovery associated with "
                "architectural edge/structure."
            ),
    },

    {
        "concept":
            "wall",

        "reason":
            "Architectural surface.",
    },

    {
        "concept":
            "floor",

        "reason":
            "Architectural surface.",
    },

    {
        "concept":
            "ceiling",

        "reason":
            "Architectural surface.",
    },

    {
        "concept":
            "baseboard",

        "reason":
            "Architectural room component.",
    },

    {
        "concept":
            "room corner",

        "reason":
            "Architectural geometry.",
    },

]


# ============================================================
# FAILED 06D2A AUDIT RECORD
# ============================================================

failed_state = json.loads(
    FAILED_D2A_PATH.read_text(
        encoding="utf-8"
    )
)


FAILED_D2A_AUDIT = {

    "status":
        "REJECTED_AS_FINAL_INVENTORY",

    "problems": [

        "toilet paper holder incorrectly merged into toilet",

        "nonexistent toilet tank included",

        "electrical plate omitted as standalone fixture",

        "ceiling light omitted",

        "shower fixture incorrectly grouped with switch/cable",

        "semantic relationship was incorrectly treated as physical connectivity",

    ],

    "preserved_source":
        str(
            FAILED_D2A_PATH
        ),
}


# ============================================================
# FINAL STATE
# ============================================================

FINAL_STATE = {

    "stage":
        "06D2A2",

    "status":
        "REFERENCE_VERIFIED_MAIN_PROP_INVENTORY",

    "detection_master":
        str(
            MASTER_PATH
        ),

    "main_prop_count":
        len(
            MAIN_PROPS
        ),

    "main_props":
        MAIN_PROPS,

    "rejected_non_prop_concepts":
        REJECTED_CONCEPTS,

    "failed_06D2A_audit":
        FAILED_D2A_AUDIT,

    "production_grouping_rule":
        (
            "Semantic relationship never creates a connected "
            "prop group. Non-architectural objects are merged "
            "only when visible physical contact, attachment, "
            "integration, or direct resting contact exists."
        ),

    "special_layer_ownership": {

        "mirror":
            "STAGE03",

        "glass_enclosure_and_owned_hardware":
            "STAGE04",
    },

    "rgb_rule":
        (
            "Stage05F is used for detection/localization. "
            "Stage01 master remains final exact RGB source."
        ),

    "next_stage":
        "06D2B_COMPLETE_MAIN_PROP_LOCALIZATION",
}


RESULT_PATH = (
    OUT
    / "00_stage06d2a2_verified_main_prop_state.json"
)


RESULT_PATH.write_text(
    json.dumps(
        FINAL_STATE,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# ============================================================
# PRINT
# ============================================================

print("=" * 110)
print("PRODUCTION STAGE 06D2A2 RESULT")
print("=" * 110)

print()
print(
    "STATUS:",
    FINAL_STATE[
        "status"
    ]
)

print(
    "FINAL MAIN PROPS:",
    len(
        MAIN_PROPS
    )
)


for prop in MAIN_PROPS:

    print()
    print(
        "{}. {} [{}]".format(

            prop[
                "main_prop_id"
            ],

            prop[
                "main_prop_name"
            ],

            prop[
                "prop_type"
            ],
        )
    )

    print(
        "   MEMBERS:",
        prop[
            "visible_members"
        ]
    )

    if prop.get(
        "explicit_exclusions"
    ):

        print(
            "   EXCLUDE:",
            prop[
                "explicit_exclusions"
            ]
        )


print()
print("=" * 110)
print("FAILED 06D2A")
print("=" * 110)

for problem in FAILED_D2A_AUDIT[
    "problems"
]:

    print(
        "❌",
        problem
    )


print()
print(
    "RESULT JSON:",
    RESULT_PATH
)

print()
print(
    "NO DETECTION OR SEGMENTATION WAS RUN."
)

print(
    "READY FOR COMPLETE MAIN-PROP LOCALIZATION."
)
