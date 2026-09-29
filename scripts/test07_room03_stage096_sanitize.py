
from pathlib import Path
import argparse
import json


def get_id(row):
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
    return str(
        row.get(
            "decision",
            row.get(
                "proposal_decision",
                ""
            )
        )
    ).upper()


# ============================================================
# UNION FIND
# ============================================================

class DSU:

    def __init__(self):
        self.parent = {}

    def add(self, x):
        if x not in self.parent:
            self.parent[x] = x

    def find(self, x):

        self.add(x)

        if self.parent[x] != x:
            self.parent[x] = self.find(
                self.parent[x]
            )

        return self.parent[x]

    def union(self, a, b):

        ra = self.find(a)
        rb = self.find(b)

        if ra == rb:
            return

        # Stable representative = lower inventory ID
        root = min(ra, rb)
        other = max(ra, rb)

        self.parent[other] = root


def run(
    stage09_report,
    collision_report,
    output_dir,
):

    stage09_report = Path(
        stage09_report
    )

    collision_report = Path(
        collision_report
    )

    output_dir = Path(
        output_dir
    )

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

    collision_ids = {
        int(x)
        for x in s095.get(
            "collision_instance_ids",
            []
        )
    }

    duplicate_pairs = [
        [
            int(pair[0]),
            int(pair[1])
        ]
        for pair in s095.get(
            "same_instance_duplicate_pairs",
            []
        )
    ]


    # ========================================================
    # BUILD TRANSITIVE SAME-INSTANCE GROUPS
    # ========================================================

    dsu = DSU()

    for a, b in duplicate_pairs:
        dsu.union(a, b)

    groups = {}

    duplicate_members = set()

    for a, b in duplicate_pairs:

        for x in [a, b]:

            duplicate_members.add(x)

            root = dsu.find(x)

            groups.setdefault(
                root,
                set()
            ).add(x)

    groups = {
        min(values): sorted(values)
        for values in groups.values()
    }


    # ========================================================
    # ALIAS MAP
    # ========================================================

    alias_to_rep = {}

    representative_to_aliases = {}

    for representative, members in groups.items():

        aliases = [
            x
            for x in members
            if x != representative
        ]

        representative_to_aliases[
            representative
        ] = aliases

        for alias in aliases:
            alias_to_rep[alias] = representative


    # ========================================================
    # SANITIZE STAGE09
    # ========================================================

    safe = []

    unresolved = []

    aliases = []

    safe_ids = set()

    unresolved_ids = set()


    for row in objects:

        iid = get_id(row)

        if iid is None:
            continue

        decision = get_decision(row)

        accepted = (
            decision == "PROPOSAL_ACCEPT"
        )


        # ----------------------------------------------------
        # SAME-INSTANCE ALIAS
        # ----------------------------------------------------

        if iid in alias_to_rep:

            representative = (
                alias_to_rep[iid]
            )

            aliases.append({
                "inventory_id":
                    iid,

                "representative_id":
                    representative,

                "status":
                    "SAME_PHYSICAL_INSTANCE_ALIAS",

                "original_stage09":
                    row,
            })

            continue


        # ----------------------------------------------------
        # DISTINCT OBJECT MASK COLLISION
        # ----------------------------------------------------

        if iid in collision_ids:

            unresolved.append({
                "inventory_id":
                    iid,

                "status":
                    "SAM2_MASK_COLLISION",

                "original_stage09":
                    row,
            })

            unresolved_ids.add(iid)

            continue


        # ----------------------------------------------------
        # Representative whose aliases participate in a
        # distinct-instance collision.
        #
        # Example Room03:
        # 17/44 same instance, but their mask also collides
        # with 14 and 36. Therefore representative 17 must
        # remain unresolved too.
        # ----------------------------------------------------

        members = groups.get(
            iid,
            [iid]
        )

        if any(
            member in collision_ids
            for member in members
        ):

            unresolved.append({
                "inventory_id":
                    iid,

                "status":
                    "SAME_INSTANCE_CLUSTER_WITH_MASK_COLLISION",

                "same_instance_members":
                    members,

                "original_stage09":
                    row,
            })

            unresolved_ids.add(iid)

            continue


        # ----------------------------------------------------
        # NORMAL STAGE09 REJECTION
        # ----------------------------------------------------

        if not accepted:

            unresolved.append({
                "inventory_id":
                    iid,

                "status":
                    "STAGE09_REJECTED",

                "original_stage09":
                    row,
            })

            unresolved_ids.add(iid)

            continue


        # ----------------------------------------------------
        # SAFE UNIQUE / SAFE REPRESENTATIVE
        # ----------------------------------------------------

        safe.append(row)
        safe_ids.add(iid)


    # ========================================================
    # IMPORTANT:
    # Collision aliases remain represented through their
    # physical representative and are not separately counted
    # as independent unresolved physical objects.
    # ========================================================

    safe = sorted(
        safe,
        key=lambda r: get_id(r)
    )

    unresolved = sorted(
        unresolved,
        key=lambda r: int(
            r["inventory_id"]
        )
    )

    aliases = sorted(
        aliases,
        key=lambda r: int(
            r["inventory_id"]
        )
    )


    # ========================================================
    # SAVE
    # ========================================================

    safe_path = (
        output_dir
        /
        "00_stage10_safe_proposals.json"
    )

    safe_path.write_text(
        json.dumps(
            safe,
            indent=2
        )
    )

    unresolved_path = (
        output_dir
        /
        "01_remaining_unresolved.json"
    )

    unresolved_path.write_text(
        json.dumps(
            unresolved,
            indent=2
        )
    )

    alias_path = (
        output_dir
        /
        "02_same_instance_aliases.json"
    )

    alias_path.write_text(
        json.dumps(
            aliases,
            indent=2
        )
    )


    report = {

        "stage":
            "TEST07_ROOM03_STAGE096",

        "safe_ids":
            sorted(safe_ids),

        "remaining_unresolved_ids":
            sorted(unresolved_ids),

        "collision_ids":
            sorted(collision_ids),

        "same_instance_groups":
            {
                str(k): v
                for k, v
                in representative_to_aliases.items()
            },

        "alias_to_representative":
            {
                str(k): v
                for k, v
                in alias_to_rep.items()
            },

        "rules": [
            "Distinct-instance SAM2 collisions never enter Stage10.",
            "Transitive same-instance duplicates use one physical representative.",
            "If any member of a same-instance cluster participates in a distinct-instance mask collision, the representative remains unresolved.",
            "Stage09 rejected objects remain unresolved.",
            "Original Stage09 output is never modified."
        ]
    }

    report_path = (
        output_dir
        /
        "stage096_report.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2
        )
    )


    print("=" * 90)
    print("TEST07 ROOM03 STAGE09.6")
    print("=" * 90)

    print(
        "SAFE STAGE10 IDS:",
        sorted(safe_ids)
    )

    print(
        "REMAINING UNRESOLVED PHYSICAL IDS:",
        sorted(unresolved_ids)
    )

    print()

    print(
        "SAME-INSTANCE GROUPS:",
        groups
    )

    print(
        "ALIASES:",
        alias_to_rep
    )

    print()

    print(
        "COLLISION IDS:",
        sorted(collision_ids)
    )

    print()

    print(
        "STAGE10 INPUT:",
        safe_path
    )


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
