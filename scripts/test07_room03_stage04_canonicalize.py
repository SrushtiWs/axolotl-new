
from pathlib import Path
import json


ROOT = Path(
    "/workspace/axolotl/test07/runs/"
    "room03_bathroom/stages"
)

INVENTORY_PATH = (
    ROOT
    / "02_v35_applied_surface_filter"
    / "00_applied_surface_filtered_inventory.json"
)

VERIFIED_PATH = (
    ROOT
    / "04_candidate_verification"
    / "02_verified_candidates.json"
)

AUDIT_PATH = (
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


def load(path):
    return json.loads(
        path.read_text()
    )


def get_objects(data):

    if isinstance(data, list):
        return data

    if isinstance(data, dict):

        for key in [
            "objects",
            "inventory",
            "items"
        ]:

            if isinstance(
                data.get(key),
                list
            ):
                return data[key]

    raise RuntimeError(
        "Could not locate object list."
    )


def get_id(row):

    for key in [
        "inventory_id",
        "id",
        "object_id"
    ]:

        if key in row:

            try:
                return int(row[key])
            except Exception:
                pass

    return None


inventory_data = load(
    INVENTORY_PATH
)

verified_data = load(
    VERIFIED_PATH
)

audit_data = load(
    AUDIT_PATH
)

inventory = get_objects(
    inventory_data
)

verified = get_objects(
    verified_data
)


# ------------------------------------------------------------
# Verified/localized IDs
# ------------------------------------------------------------

verified_ids = sorted({
    get_id(row)
    for row in verified
    if get_id(row) is not None
})


# ------------------------------------------------------------
# Parse existence audit
# ------------------------------------------------------------

audit_objects = get_objects(
    audit_data
)

present_ids = []
absent_ids = []
uncertain_ids = []

for row in audit_objects:

    iid = get_id(row)

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

    if decision == "PRESENT":
        present_ids.append(iid)

    elif decision == "ABSENT":
        absent_ids.append(iid)

    else:
        uncertain_ids.append(iid)


present_ids = sorted(set(present_ids))
absent_ids = sorted(set(absent_ids))
uncertain_ids = sorted(set(uncertain_ids))


# ------------------------------------------------------------
# Inventory bookkeeping
# ------------------------------------------------------------

inventory_ids = sorted({
    get_id(row)
    for row in inventory
    if get_id(row) is not None
})

localized_ids = verified_ids

present_unlocalized_ids = sorted(
    set(present_ids)
    -
    set(localized_ids)
)

absent_ids = sorted(
    set(absent_ids)
    -
    set(localized_ids)
)

uncertain_ids = sorted(
    set(uncertain_ids)
    -
    set(localized_ids)
)


# ------------------------------------------------------------
# Save localized verified candidate list
# ------------------------------------------------------------

localized_candidates = [
    row
    for row in verified
    if get_id(row) in localized_ids
]

localized_path = (
    OUT
    / "01_localized_verified_candidates.json"
)

localized_path.write_text(
    json.dumps(
        localized_candidates,
        indent=2
    )
)


# ------------------------------------------------------------
# Save present-unlocalized inventory records
# ------------------------------------------------------------

present_unlocalized = [
    row
    for row in inventory
    if get_id(row)
    in present_unlocalized_ids
]

present_path = (
    OUT
    / "02_present_unlocalized.json"
)

present_path.write_text(
    json.dumps(
        present_unlocalized,
        indent=2
    )
)


# ------------------------------------------------------------
# Save absent inventory records
# ------------------------------------------------------------

absent = [
    row
    for row in inventory
    if get_id(row) in absent_ids
]

absent_path = (
    OUT
    / "03_absent_inventory.json"
)

absent_path.write_text(
    json.dumps(
        absent,
        indent=2
    )
)


# ------------------------------------------------------------
# Canonical report
# ------------------------------------------------------------

report = {

    "experiment":
        "TEST07_ROOM03_CANONICAL_STAGE04",

    "inventory_total":
        len(inventory_ids),

    "localized_verified_instances":
        len(localized_ids),

    "localized_verified_candidate_records":
        len(localized_candidates),

    "present_unlocalized_instances":
        len(present_unlocalized_ids),

    "absent_instances":
        len(absent_ids),

    "uncertain_instances":
        len(uncertain_ids),

    "localized_verified_ids":
        localized_ids,

    "present_unlocalized_ids":
        present_unlocalized_ids,

    "absent_ids":
        absent_ids,

    "uncertain_ids":
        uncertain_ids,

    "rules": [
        "Only verified localizations enter Stage05.",
        "Confirmed-present but unlocalized objects remain preserved for later recovery.",
        "Confirmed-absent inventory entries do not enter segmentation.",
        "Applied wall/floor/ceiling tiles are architectural structure, not props."
    ],
}

report_path = (
    OUT
    / "00_canonical_stage04_state.json"
)

report_path.write_text(
    json.dumps(
        report,
        indent=2
    )
)


print("=" * 90)
print("TEST07 ROOM03 CANONICAL STAGE04")
print("=" * 90)

print(
    "INVENTORY TOTAL:",
    len(inventory_ids)
)

print(
    "LOCALIZED VERIFIED:",
    len(localized_ids)
)

print(
    "VERIFIED CANDIDATES:",
    len(localized_candidates)
)

print(
    "PRESENT UNLOCALIZED:",
    len(present_unlocalized_ids)
)

print(
    "ABSENT:",
    len(absent_ids)
)

print(
    "UNCERTAIN:",
    len(uncertain_ids)
)

print()

print(
    "PRESENT UNLOCALIZED IDS:",
    present_unlocalized_ids
)

print(
    "ABSENT IDS:",
    absent_ids
)

print()

print(
    "LOCALIZED CANDIDATES:",
    localized_path
)

print(
    "CANONICAL REPORT:",
    report_path
)
