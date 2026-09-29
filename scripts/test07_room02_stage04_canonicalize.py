
from pathlib import Path
import json


ROOT = Path(
    "/workspace/axolotl/test07/runs/"
    "room02_kitchen/stages"
)

INVENTORY_JSON = (
    ROOT
    / "02_v34_conservative_inventory"
    / "00_conservative_inventory.json"
)

VERIFIED_JSON = (
    ROOT
    / "04_candidate_verification"
    / "02_verified_candidates.json"
)

EXISTENCE_JSON = (
    ROOT
    / "04_5_lost_instance_existence_audit"
    / "00_existence_audit.json"
)

OUT = (
    ROOT
    / "04_10_canonical_stage04_state"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


inventory = json.loads(
    INVENTORY_JSON.read_text()
)["objects"]

verified = json.loads(
    VERIFIED_JSON.read_text()
)

existence = json.loads(
    EXISTENCE_JSON.read_text()
)


inventory_by_id = {
    int(row["id"]): row
    for row in inventory
}


verified_ids = {
    int(row["inventory_id"])
    for row in verified
}


existence_by_id = {
    int(row["inventory_id"]): row
    for row in existence
}


canonical = []

localized = []
absent = []
present_unlocalized = []


for iid in sorted(
    inventory_by_id.keys()
):

    obj = inventory_by_id[iid]

    if iid in verified_ids:

        status = "LOCALIZED_VERIFIED"

        localized.append(iid)

    else:

        audit = existence_by_id.get(
            iid
        )

        if audit is None:

            status = "UNRESOLVED"

        elif (
            audit["decision"]
            ==
            "ABSENT"
        ):

            status = "ABSENT"

            absent.append(iid)

        elif (
            audit["decision"]
            ==
            "PRESENT"
        ):

            status = (
                "PRESENT_UNLOCALIZED"
            )

            present_unlocalized.append(
                iid
            )

        else:

            status = "UNRESOLVED"

    canonical.append({
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

        "inventory_confidence":
            obj.get(
                "confidence",
                ""
            ),

        "status":
            status,
    })


# ============================================================
# SAVE CANONICAL STATE
# ============================================================

state = {
    "stage":
        "TEST07_ROOM02_STAGE04_CANONICAL",

    "inventory_total":
        len(
            inventory
        ),

    "localized_verified_instances":
        len(
            localized
        ),

    "localized_verified_ids":
        localized,

    "absent_instances":
        len(
            absent
        ),

    "absent_ids":
        absent,

    "present_unlocalized_instances":
        len(
            present_unlocalized
        ),

    "present_unlocalized_ids":
        present_unlocalized,

    "verified_candidate_count":
        len(
            verified
        ),

    "instances":
        canonical,

    "rules": [
        "ABSENT entries do not enter later segmentation stages",
        "PRESENT_UNLOCALIZED entries are preserved but receive no SAM2 box",
        "only Stage04 verified localization candidates proceed to Stage05",
        "Stage04.9 rescue failure does not delete confirmed-present objects",
    ],
}


STATE_PATH = (
    OUT
    / "00_canonical_stage04_state.json"
)

STATE_PATH.write_text(
    json.dumps(
        state,
        indent=2
    )
)


# Preserve exact Stage04 verified candidates
VERIFIED_OUT = (
    OUT
    / "01_localized_verified_candidates.json"
)

VERIFIED_OUT.write_text(
    json.dumps(
        verified,
        indent=2
    )
)


# Present but unlocalized
UNLOCALIZED_ROWS = [
    row
    for row in canonical
    if row[
        "status"
    ] == "PRESENT_UNLOCALIZED"
]

UNLOCALIZED_PATH = (
    OUT
    / "02_present_unlocalized.json"
)

UNLOCALIZED_PATH.write_text(
    json.dumps(
        UNLOCALIZED_ROWS,
        indent=2
    )
)


# Absent
ABSENT_ROWS = [
    row
    for row in canonical
    if row[
        "status"
    ] == "ABSENT"
]

ABSENT_PATH = (
    OUT
    / "03_absent_inventory_entries.json"
)

ABSENT_PATH.write_text(
    json.dumps(
        ABSENT_ROWS,
        indent=2
    )
)


print("=" * 90)
print("TEST07 ROOM02 CANONICAL STAGE04 STATE")
print("=" * 90)

print(
    "INVENTORY TOTAL:",
    len(
        inventory
    )
)

print(
    "LOCALIZED VERIFIED:",
    len(
        localized
    )
)

print(
    "VERIFIED CANDIDATES:",
    len(
        verified
    )
)

print(
    "ABSENT:",
    len(
        absent
    )
)

print(
    "PRESENT UNLOCALIZED:",
    len(
        present_unlocalized
    )
)

print()
print(
    "PRESENT UNLOCALIZED IDS:",
    present_unlocalized
)

print()
print(
    "ABSENT IDS:",
    absent
)

print()
print("OUTPUT:")
print(STATE_PATH)
