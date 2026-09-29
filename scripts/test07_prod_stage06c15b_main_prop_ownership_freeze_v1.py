
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

OUT = (
    STAGE06
    / "06c15b_main_prop_ownership_freeze"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# GLOBAL OWNERSHIP RULE
# ============================================================

production_rule = {

    "rule_id":
        "MAIN_PROP_OWNS_ATTACHED_COMPONENTS",

    "description":
        (
            "If a detected object is physically attached to, "
            "integrated into, or forms part of a larger main "
            "physical prop, the child object must not become "
            "an independent final prop layer. The complete "
            "main prop is extracted once."
        ),

    "examples": [

        "drawer + drawer handle -> vanity",

        "sink + attached faucet -> vanity system",

        "toilet seat -> toilet",

        "glass-mounted towel bar -> glass enclosure",

        "duplicate descriptions of same wall plate -> one wall plate",
    ],
}


# ============================================================
# PHYSICAL MAIN-PROP GROUPS
# ============================================================

groups = {


    # --------------------------------------------------------
    # VANITY
    # --------------------------------------------------------

    "VANITY_SYSTEM": {

        "main_prop":
            True,

        "canonical_inventory_id":
            23,

        "canonical_name":
            "complete bathroom vanity system",

        "component_inventory_ids": [
            5,
            10,
            21,
            23,
            24,
            25,
            29,
        ],

        "components": {

            "5":
                "cabinet/body description",

            "10":
                "attached faucet",

            "21":
                "drawer/cabinet knob or handle",

            "23":
                "cabinet / vanity parent",

            "24":
                "attached wash basin / sink",

            "25":
                "duplicate faucet description",

            "29":
                "drawer/front component",
        },

        "final_layer_policy":
            "EXTRACT_AS_ONE_MAIN_PROP",

        "independent_final_layers_for_components":
            False,

        "existing_component_masks":
            {
                "10":
                    "retain as evidence only",

                "21":
                    "retain as evidence only",

                "24":
                    "retain provisional mask as evidence only",
            },

        "next_action":
            (
                "segment the complete physical vanity system "
                "including its attached/integrated components"
            ),
    },


    # --------------------------------------------------------
    # TOILET
    # --------------------------------------------------------

    "TOILET_SYSTEM": {

        "main_prop":
            True,

        "canonical_inventory_id":
            14,

        "canonical_name":
            "complete toilet",

        "component_inventory_ids": [
            13,
            14,
        ],

        "components": {

            "13":
                "toilet seat / lid component",

            "14":
                "toilet main body",
        },

        "final_layer_policy":
            "USE_COMPLETE_TOILET_MAIN_PROP",

        "independent_final_layers_for_components":
            False,

        "existing_component_masks":
            {
                "13":
                    "component evidence only",

                "14":
                    "main-prop mask candidate",
            },

        "next_action":
            (
                "ensure final toilet mask represents the "
                "complete toilet including seat/lid"
            ),
    },


    # --------------------------------------------------------
    # GLASS ENCLOSURE
    # --------------------------------------------------------

    "GLASS_ENCLOSURE_SYSTEM": {

        "main_prop":
            True,

        "canonical_owner":
            "STAGE04_GLASS_SYSTEM",

        "component_inventory_ids": [
            15,
        ],

        "components": {

            "15":
                "glass-mounted towel bar / rail",
        },

        "final_layer_policy":
            "OWNED_BY_STAGE04_GLASS_LAYER",

        "independent_final_layers_for_components":
            False,

        "next_action":
            (
                "reconstruct/render towel bar together with "
                "the final glass enclosure"
            ),
    },


    # --------------------------------------------------------
    # ELECTRICAL WALL PLATE
    # --------------------------------------------------------

    "WALL_ELECTRICAL_PLATE": {

        "main_prop":
            True,

        "canonical_inventory_id":
            9,

        "canonical_name":
            "wall electrical plate",

        "component_inventory_ids": [
            8,
            9,
        ],

        "components": {

            "8":
                "wall outlet description",

            "9":
                "wall switch description",
        },

        "final_layer_policy":
            "ONE_PHYSICAL_PROP",

        "independent_final_layers_for_components":
            False,

        "evidence":
            "008 and 009 Florence bbox IoU = 1.0",
    },
}


# ============================================================
# STANDALONE MAIN PROPS
# ============================================================

standalone_main_props = {


    "2": {

        "name":
            "recessed light",

        "state":
            "RESOLVED",

        "final_prop_layer":
            True,
    },


    "12": {

        "name":
            "toilet paper holder",

        "state":
            "RESOLVED",

        "final_prop_layer":
            True,
    },


    "9": {

        "name":
            "wall electrical plate",

        "state":
            "RESOLVED",

        "final_prop_layer":
            True,
    },


    "14": {

        "name":
            "complete toilet",

        "state":
            "MAIN_PROP_REQUIRES_COMPLETE_MASK_AUDIT",

        "final_prop_layer":
            True,
    },


    "23": {

        "name":
            "complete vanity system",

        "state":
            "MAIN_PROP_REQUIRES_COMPLETE_MASK",

        "final_prop_layer":
            True,
    },
}


# ============================================================
# CHILD COMPONENTS — DO NOT FINAL-UNION
# ============================================================

child_components = {


    "5": {
        "parent":
            "VANITY_SYSTEM",
        "final_prop_layer":
            False,
    },


    "10": {
        "parent":
            "VANITY_SYSTEM",
        "final_prop_layer":
            False,
        "existing_mask":
            "EVIDENCE_ONLY",
    },


    "13": {
        "parent":
            "TOILET_SYSTEM",
        "final_prop_layer":
            False,
        "existing_mask":
            "EVIDENCE_ONLY",
    },


    "15": {
        "parent":
            "GLASS_ENCLOSURE_SYSTEM",
        "final_prop_layer":
            False,
    },


    "21": {
        "parent":
            "VANITY_SYSTEM",
        "final_prop_layer":
            False,
        "existing_mask":
            "EVIDENCE_ONLY",
    },


    "24": {
        "parent":
            "VANITY_SYSTEM",
        "final_prop_layer":
            False,
        "existing_mask":
            "EVIDENCE_ONLY",
    },


    "25": {
        "parent":
            "VANITY_SYSTEM",
        "final_prop_layer":
            False,
        "duplicate_of":
            10,
    },


    "29": {
        "parent":
            "VANITY_SYSTEM",
        "final_prop_layer":
            False,
    },


    "8": {
        "parent":
            "WALL_ELECTRICAL_PLATE",
        "final_prop_layer":
            False,
        "duplicate_of":
            9,
    },
}


# ============================================================
# CURRENT FINAL-PROP TARGETS
# ============================================================

final_prop_targets = [

    {
        "key":
            "002",

        "name":
            "recessed light",

        "state":
            "RESOLVED",
    },

    {
        "key":
            "009",

        "name":
            "wall electrical plate",

        "state":
            "RESOLVED",
    },

    {
        "key":
            "012",

        "name":
            "toilet paper holder",

        "state":
            "RESOLVED",
    },

    {
        "key":
            "014",

        "name":
            "complete toilet",

        "state":
            "COMPLETE_MASK_AUDIT_REQUIRED",
    },

    {
        "key":
            "023",

        "name":
            "complete vanity system",

        "state":
            "COMPLETE_MASK_REQUIRED",
    },
]


# ============================================================
# SPECIAL NON-PROP OWNERS
# ============================================================

special_owners = [

    {
        "owner":
            "STAGE04_GLASS_SYSTEM",

        "components": [
            15,
        ],

        "state":
            "SPECIAL_LAYER",
    },
]


# ============================================================
# SAVE
# ============================================================

result = {

    "stage":
        "06C15B",

    "production_rule":
        production_rule,

    "physical_main_prop_groups":
        groups,

    "standalone_main_props":
        standalone_main_props,

    "child_components":
        child_components,

    "current_final_prop_targets":
        final_prop_targets,

    "special_layer_owners":
        special_owners,

    "rules": [

        (
            "attached/integrated components must not be "
            "separate final prop layers"
        ),

        (
            "existing child-component masks remain useful "
            "for validation and training evidence"
        ),

        (
            "final prop union contains main physical props "
            "only"
        ),

        (
            "Stage04 owns glass and attached glass hardware"
        ),

        (
            "exact final RGB remains sourced from Stage01 "
            "master"
        ),
    ],
}


RESULT_PATH = (
    OUT
    / "00_main_prop_ownership_state.json"
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
    / "01_main_prop_ownership_report.txt"
)


lines = [

    "=" * 110,

    "TEST07 PRODUCTION STAGE 06C15B",

    "MAIN-PROP / ATTACHED-COMPONENT OWNERSHIP",

    "=" * 110,

    "",

    "PRODUCTION RULE:",

    production_rule[
        "description"
    ],

    "",

    "-" * 110,

    "CURRENT FINAL PROP TARGETS",

    "-" * 110,
]


for row in final_prop_targets:

    lines.append(

        "{} | {:30s} | {}".format(

            row[
                "key"
            ],

            row[
                "name"
            ],

            row[
                "state"
            ],
        )
    )


lines.extend(
    [
        "",
        "-" * 110,
        "CHILD COMPONENTS — EXCLUDED FROM INDEPENDENT FINAL UNION",
        "-" * 110,
    ]
)


for iid, row in child_components.items():

    lines.append(

        "{} -> {} | final_layer={}".format(

            iid,

            row[
                "parent"
            ],

            row[
                "final_prop_layer"
            ],
        )
    )


REPORT_PATH.write_text(
    "\n".join(lines),
    encoding="utf-8"
)


# ============================================================
# PRINT
# ============================================================

print("=" * 110)
print("PRODUCTION STAGE 06C15B RESULT")
print("=" * 110)

print()
print(
    "RULE:"
)

print(
    production_rule[
        "description"
    ]
)


print()
print("-" * 110)
print("FINAL MAIN-PROP TARGETS")
print("-" * 110)


for row in final_prop_targets:

    print(

        "{}. {:30s} | {}".format(

            row[
                "key"
            ],

            row[
                "name"
            ],

            row[
                "state"
            ],
        )
    )


print()
print("-" * 110)
print("CHILD / ATTACHED COMPONENTS")
print("-" * 110)


for iid, row in child_components.items():

    print(

        "{} -> {:28s} | INDEPENDENT_FINAL_LAYER={}".format(

            iid,

            row[
                "parent"
            ],

            row[
                "final_prop_layer"
            ],
        )
    )


print()
print("-" * 110)

print(
    "GLASS OWNER:",
    "015 -> STAGE04_GLASS_SYSTEM"
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
    "NO MODEL WAS RUN."
)

print(
    "NO FINAL PROP MASK UNION WAS CREATED."
)
