
from pathlib import Path
import argparse
import json


def get_iid(row):

    for key in [
        "inventory_id",
        "instance_id",
        "id",
        "object_id",
    ]:

        if key in row:

            try:
                return int(row[key])
            except Exception:
                pass

    return None


def get_decision(row):

    for key in [
        "decision",
        "proposal_decision",
        "status",
        "result",
    ]:

        if key in row:
            return str(row[key]).upper()

    return ""


def run(
    stage09_report,
    collision_report,
    output_dir,
):

    stage09_report = Path(stage09_report)
    collision_report = Path(collision_report)
    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    s09 = json.loads(
        stage09_report.read_text()
    )

    s095 = json.loads(
        collision_report.read_text()
    )

    objects = s09.get(
        "objects",
        []
    )

    collision_ids = set(
        int(x)
        for x in s095.get(
            "collision_instance_ids",
            []
        )
    )

    duplicate_pairs = [
        [int(a), int(b)]
        for a, b in s095.get(
            "same_instance_duplicate_pairs",
            []
        )
    ]

    # Representative → aliases.
    alias_map = {}

    alias_ids = set()

    for pair in duplicate_pairs:

        representative = min(pair)

        aliases = sorted(
            x
            for x in pair
            if x != representative
        )

        alias_map[
            representative
        ] = aliases

        alias_ids.update(
            aliases
        )

    safe = []
    rejected_or_unresolved = []
    aliases = []

    for row in objects:

        iid = get_iid(row)

        if iid is None:
            continue

        decision = get_decision(row)

        accepted = (
            "ACCEPT" in decision
            and
            "REJECT" not in decision
        )

        # -----------------------------------------------
        # True duplicate alias
        # -----------------------------------------------

        if iid in alias_ids:

            representative = None

            for rep, vals in alias_map.items():

                if iid in vals:
                    representative = rep
                    break

            aliases.append({
                "inventory_id":
                    iid,

                "status":
                    "SAME_PHYSICAL_INSTANCE_ALIAS",

                "representative_id":
                    representative,

                "original_stage09":
                    row,
            })

            continue

        # -----------------------------------------------
        # Distinct objects whose SAM2 masks collided
        # -----------------------------------------------

        if iid in collision_ids:

            rejected_or_unresolved.append({
                "inventory_id":
                    iid,

                "status":
                    "SAM2_MASK_COLLISION",

                "original_stage09":
                    row,
            })

            continue

        # -----------------------------------------------
        # Stage09 normal rejection
        # -----------------------------------------------

        if not accepted:

            rejected_or_unresolved.append({
                "inventory_id":
                    iid,

                "status":
                    "STAGE09_REJECTED",

                "original_stage09":
                    row,
            })

            continue

        # -----------------------------------------------
        # Safe unique proposal
        # -----------------------------------------------

        safe.append(
            row
        )

    SAFE_PATH = (
        output_dir
        / "00_stage10_safe_proposals.json"
    )

    SAFE_PATH.write_text(
        json.dumps(
            safe,
            indent=2
        )
    )

    unresolved_path = (
        output_dir
        / "01_remaining_unresolved.json"
    )

    unresolved_path.write_text(
        json.dumps(
            rejected_or_unresolved,
            indent=2
        )
    )

    alias_path = (
        output_dir
        / "02_same_instance_aliases.json"
    )

    alias_path.write_text(
        json.dumps(
            aliases,
            indent=2
        )
    )

    safe_ids = [
        get_iid(row)
        for row in safe
    ]

    unresolved_ids = [
        int(row["inventory_id"])
        for row in rejected_or_unresolved
    ]

    alias_ids_out = [
        int(row["inventory_id"])
        for row in aliases
    ]

    summary = {
        "stage":
            "TEST07_STAGE096",

        "safe_unique_proposals":
            len(safe),

        "safe_ids":
            safe_ids,

        "remaining_unresolved":
            len(
                rejected_or_unresolved
            ),

        "remaining_unresolved_ids":
            unresolved_ids,

        "same_instance_aliases":
            len(aliases),

        "alias_ids":
            alias_ids_out,

        "alias_map":
            {
                str(k): v
                for k, v in alias_map.items()
            },

        "rules": [
            "distinct-instance mask collisions never enter Stage10",
            "same-physical-instance duplicate proposals are represented once",
            "Stage09 rejected proposals remain unresolved",
            "original Stage09 files are not modified",
        ],
    }

    report_path = (
        output_dir
        / "stage096_report.json"
    )

    report_path.write_text(
        json.dumps(
            summary,
            indent=2
        )
    )

    print("=" * 90)
    print("TEST07 STAGE09.6 RESULT")
    print("=" * 90)

    print(
        "SAFE UNIQUE PROPOSALS:",
        len(safe)
    )

    print(
        "SAFE IDS:",
        safe_ids
    )

    print()

    print(
        "REMAINING UNRESOLVED:",
        len(
            rejected_or_unresolved
        )
    )

    print(
        "UNRESOLVED IDS:",
        unresolved_ids
    )

    print()

    print(
        "SAME-INSTANCE ALIASES:",
        len(
            aliases
        )
    )

    print(
        "ALIAS IDS:",
        alias_ids_out
    )

    print(
        "ALIAS MAP:",
        alias_map
    )

    print()
    print("STAGE10 INPUT:")
    print(SAFE_PATH)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stage09-report",
        required=True
    )

    parser.add_argument(
        "--collision-report",
        required=True
    )

    parser.add_argument(
        "--output-dir",
        required=True
    )

    args = parser.parse_args()

    run(
        stage09_report=
            args.stage09_report,

        collision_report=
            args.collision_report,

        output_dir=
            args.output_dir,
    )
