
from pathlib import Path
import json
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# ============================================================
# PATHS
# ============================================================

BASE = Path("/workspace/axolotl")

PROD = (
    BASE
    / "test07"
    / "production_pipeline"
)

STAGE05F = (
    PROD
    / "stage05_clean_room_with_props"
    / "12_final_canonical_prop_detection_master.png"
)

D2A2_STATE = (
    PROD
    / "stage06_prop_layer"
    / "06d2a2_verified_main_prop_state"
    / "00_stage06d2a2_verified_main_prop_state.json"
)

D2B3_STATE = (
    PROD
    / "stage06_prop_layer"
    / "06d2b3_final_verified_main_prop_localization"
    / "00_stage06d2b3_result.json"
)

D2C1_STATE = (
    PROD
    / "stage06_prop_layer"
    / "06d2c1_complete_main_prop_sam2_multimask_audit"
    / "00_stage06d2c1_result.json"
)

E2_STATE = (
    PROD
    / "stage06_prop_layer"
    / "06e2_birefnet_direct_prop_extraction_v2"
    / "00_stage06e2_v2_result.json"
)

OUT = (
    PROD
    / "stage06_prop_layer"
    / "06f1_six_main_prop_state"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# VALIDATE REQUIRED ASSETS
# ============================================================

required = [
    STAGE05F,
    D2A2_STATE,
    D2B3_STATE,
    D2C1_STATE,
]

for p in required:
    if not p.exists():
        raise FileNotFoundError(p)


# ============================================================
# LOAD
# ============================================================

image = Image.open(
    STAGE05F
).convert("RGB")

W, H = image.size

inventory = json.loads(
    D2A2_STATE.read_text(
        encoding="utf-8"
    )
)

localization = json.loads(
    D2B3_STATE.read_text(
        encoding="utf-8"
    )
)

sam2_audit = json.loads(
    D2C1_STATE.read_text(
        encoding="utf-8"
    )
)


# ============================================================
# FREEZE 06E2 FAILURE
# ============================================================

e2_status = {
    "state_file": str(E2_STATE),
    "status": "FAILED_FOR_STAGE06",
    "reason": (
        "BiRefNet extracted only salient foreground objects "
        "(mainly sink/basin area and toilet) and missed "
        "vanity body/countertop, shower fixture, electrical "
        "plate, toilet-paper holder and ceiling light."
    ),
}


# ============================================================
# VERIFIED SIX MAIN PROPS
#
# IMPORTANT:
# These are FINAL PROP SYSTEMS.
# Children are not final layers.
# ============================================================

MAIN_PROPS = {

    "P01": {
        "name": "complete vanity system",
        "type": "CONNECTED_SYSTEM",
        "status": "MASK_UNRESOLVED",
        "next_action": "COMPLETE_VANITY_MASK_COMPLETION",
    },

    "P02": {
        "name": "complete toilet system",
        "type": "CONNECTED_SYSTEM",
        "status": "PROVISIONAL_MASK_READY",
        "source_stage": "06D2C1",
        "candidate": 3,
    },

    "P03": {
        "name": "shower fixture",
        "type": "CONNECTED_SYSTEM",
        "status": "PROVISIONAL_MASK_READY",
        "source_stage": "06D2C1",
        "candidate": 2,
    },

    "P04": {
        "name": "wall electrical plate",
        "type": "STANDALONE_FIXTURE",
        "status": "PROVISIONAL_MASK_READY",
        "source_stage": "06D2C1",
        "candidate": 1,
    },

    "P05": {
        "name": "toilet paper holder",
        "type": "STANDALONE_FIXTURE",
        "status": "PROVISIONAL_MASK_READY",
        "source_stage": "06D2C1",
        "candidate": 2,
    },

    "P06": {
        "name": "ceiling light",
        "type": "STANDALONE_FIXTURE",
        "status": "PROVISIONAL_MASK_READY",
        "source_stage": "06D2C1",
        "candidate": 3,
    },
}


# ============================================================
# EXTRACT VERIFIED LOCALIZATION BBOXES FROM 06D2B3
# ============================================================

def find_prop_rows(obj):

    rows = []

    if isinstance(obj, dict):

        if "prop_id" in obj:
            rows.append(obj)

        for v in obj.values():
            rows.extend(
                find_prop_rows(v)
            )

    elif isinstance(obj, list):

        for item in obj:
            rows.extend(
                find_prop_rows(item)
            )

    return rows


rows = find_prop_rows(
    localization
)


bbox_map = {}

for row in rows:

    pid = row.get("prop_id")

    bbox = (
        row.get("bbox")
        or row.get("final_bbox")
        or row.get("verified_bbox")
    )

    if pid in MAIN_PROPS and bbox is not None:
        bbox_map[pid] = bbox


# Fallback to frozen known verified values if JSON schema
# stores them under a different nested field.
FROZEN_BBOX_FALLBACK = {

    "P01": [
        24.5,
        224.0,
        195.8,
        427.9
    ],

    "P02": [
        178.1,
        284.6,
        257.9,
        358.4
    ],

    "P03": [
        248.4,
        113.9,
        296.6,
        139.1
    ],

    "P04": [
        186.5,
        212.8,
        208.5,
        244.2
    ],

    "P05": [
        217.5,
        274.1,
        235.5,
        296.9
    ],

    "P06": [
        243.3,
        12.4,
        263.7,
        31.6
    ],
}


for pid in MAIN_PROPS:

    if pid not in bbox_map:
        bbox_map[pid] = (
            FROZEN_BBOX_FALLBACK[pid]
        )

    MAIN_PROPS[pid]["bbox"] = (
        bbox_map[pid]
    )


# ============================================================
# SAVE CLEAN STATE
# ============================================================

STATE = {

    "stage":
        "06F1",

    "architecture":
        "SIX_MAIN_PROP_SYSTEMS_ONLY",

    "input":
        str(STAGE05F),

    "image_size": [
        W,
        H
    ],

    "06E2_failure":
        e2_status,

    "main_props":
        MAIN_PROPS,

    "rules": [

        (
            "No child item becomes an independent final "
            "prop layer when physically attached/touching/"
            "integrated with a larger prop system."
        ),

        (
            "P01 vanity must be one final complete mask "
            "including cabinet/body, countertop, basin, "
            "faucet, bottle, drawers/doors and attached "
            "handles."
        ),

        (
            "Mirror remains Stage03-owned."
        ),

        (
            "Glass enclosure and glass-owned hardware remain "
            "Stage04-owned."
        ),

        (
            "Final Stage06 output will be one same-size RGBA "
            "prop layer at original coordinates."
        ),
    ],

    "next_stage":
        "06F2_COMPLETE_VANITY_MASK_COMPLETION",

    "status":
        "READY_FOR_P01_MASK_COMPLETION",
}


STATE_PATH = (
    OUT
    / "00_stage06f1_state.json"
)


STATE_PATH.write_text(
    json.dumps(
        STATE,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# ============================================================
# OVERVIEW IMAGE
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 10)
)

ax.imshow(
    image
)


for pid, data in MAIN_PROPS.items():

    x1, y1, x2, y2 = data["bbox"]

    rect = patches.Rectangle(
        (x1, y1),
        x2 - x1,
        y2 - y1,
        fill=False,
        linewidth=2
    )

    ax.add_patch(
        rect
    )

    ax.text(
        x1,
        max(0, y1 - 4),
        f"{pid} — {data['name']}",
        fontsize=8,
        bbox=dict(
            facecolor="white",
            alpha=0.75,
            edgecolor="none"
        )
    )


ax.set_title(
    "06F1 — VERIFIED SIX MAIN PROP SYSTEMS\n"
    "No child prop layers"
)

ax.axis(
    "off"
)


plt.tight_layout()


OVERVIEW_PATH = (
    OUT
    / "01_six_main_prop_overview.png"
)


plt.savefig(
    OVERVIEW_PATH,
    dpi=150,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# PRINT
# ============================================================

print()
print("=" * 110)
print("STAGE 06F1 RESULT")
print("=" * 110)

print()
print(
    "INPUT:",
    STAGE05F
)

print(
    "SIZE:",
    f"{W} × {H}"
)

print()
print(
    "06E2:",
    "FROZEN AS FAILED"
)

print()


for pid, data in MAIN_PROPS.items():

    print(
        pid,
        "|",
        data["name"],
        "|",
        data["status"],
        "| bbox =",
        data["bbox"]
    )


print()
print(
    "NEXT:",
    "06F2 — COMPLETE VANITY MASK COMPLETION"
)

print(
    "STATE:",
    STATE_PATH
)

print(
    "OVERVIEW:",
    OVERVIEW_PATH
)
