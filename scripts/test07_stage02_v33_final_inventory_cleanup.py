
from pathlib import Path
import argparse
import json
import re


# ============================================================
# TEST07 — STAGE02-V3.3
#
# FINAL CONSERVATIVE INVENTORY CLEANUP
#
# INPUT:
#   V3.2 pairwise-verified inventory
#
# PURPOSE:
#   1. Remove prohibited continuous structural surfaces
#   2. Merge only extremely strong residual textual duplicates
#   3. Preserve ambiguous instances
#
# NO MODEL CALLS.
# NO ROOM-SPECIFIC RULES.
# ============================================================


REGION_POSITIONS = {
    "top_left":       (0, 0),
    "top_center":     (1, 0),
    "top_right":      (2, 0),

    "middle_left":    (0, 1),
    "middle_center":  (1, 1),
    "middle_right":   (2, 1),

    "bottom_left":    (0, 2),
    "bottom_center":  (1, 2),
    "bottom_right":   (2, 2),
}


STRUCTURAL_SURFACE_NAMES = {
    "wall",
    "floor",
    "ceiling",
    "ground",
    "sky",
    "background",
}


def normalize_text(text):

    text = str(text).lower().strip()

    text = re.sub(
        r"[^a-z0-9 ]+",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def tokens(text):

    stop = {
        "a",
        "an",
        "the",
        "on",
        "in",
        "with",
        "of",
        "at",
    }

    return {
        x
        for x in normalize_text(text).split()
        if x not in stop
    }


def similarity(a, b):

    a = tokens(a)
    b = tokens(b)

    if not a or not b:
        return 0.0

    return len(a & b) / len(a | b)


def get_regions(row):

    regions = row.get(
        "source_regions",
        []
    )

    if isinstance(regions, str):
        regions = [
            x.strip()
            for x in regions.split(",")
            if x.strip()
        ]

    if not regions:

        region = row.get(
            "source_region",
            ""
        )

        if region:
            regions = [region]

    return regions


def regions_touch(
    regions_a,
    regions_b
):

    for ra in regions_a:

        if ra not in REGION_POSITIONS:
            continue

        ax, ay = REGION_POSITIONS[ra]

        for rb in regions_b:

            if rb not in REGION_POSITIONS:
                continue

            bx, by = REGION_POSITIONS[rb]

            if (
                abs(ax - bx) <= 1
                and
                abs(ay - by) <= 1
            ):
                return True

    return False


def prohibited_surface(row):

    name = normalize_text(
        row.get(
            "name",
            ""
        )
    )

    if name in STRUCTURAL_SURFACE_NAMES:
        return True

    phrase = normalize_text(
        row.get(
            "grounding_phrase",
            ""
        )
    )

    surface_phrases = {
        "floor surface",
        "tiled floor surface",
        "wall surface",
        "ceiling surface",
        "ground surface",
    }

    return phrase in surface_phrases


def strong_duplicate(a, b):

    name_a = normalize_text(
        a.get(
            "name",
            ""
        )
    )

    name_b = normalize_text(
        b.get(
            "name",
            ""
        )
    )

    if not name_a or name_a != name_b:
        return False

    if not regions_touch(
        get_regions(a),
        get_regions(b)
    ):
        return False

    phrase_a = a.get(
        "grounding_phrase",
        ""
    )

    phrase_b = b.get(
        "grounding_phrase",
        ""
    )

    score = similarity(
        phrase_a,
        phrase_b
    )

    # Very conservative.
    return score >= 0.72


def choose_best(group):

    confidence_rank = {
        "high": 3,
        "medium": 2,
        "low": 1,
    }

    best = max(
        group,
        key=lambda r: (
            confidence_rank.get(
                str(
                    r.get(
                        "confidence",
                        ""
                    )
                ).lower(),
                0
            ),
            len(
                str(
                    r.get(
                        "grounding_phrase",
                        ""
                    )
                )
            )
        )
    )

    result = dict(best)

    regions = []

    for row in group:

        for region in get_regions(row):

            if region not in regions:
                regions.append(region)

    result["source_regions"] = sorted(
        regions
    )

    result[
        "merged_candidate_count"
    ] = sum(
        int(
            row.get(
                "merged_candidate_count",
                1
            )
        )
        for row in group
    )

    result.pop(
        "source_region",
        None
    )

    return result


def run(
    input_json,
    output_dir
):

    input_json = Path(input_json)
    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    data = json.loads(
        input_json.read_text()
    )

    objects = data.get(
        "objects",
        []
    )

    print("=" * 90)
    print("TEST07 STAGE02-V3.3")
    print("FINAL CONSERVATIVE INVENTORY CLEANUP")
    print("=" * 90)

    print(
        "INPUT OBJECTS:",
        len(objects)
    )

    kept = []
    removed_surfaces = []

    for row in objects:

        if prohibited_surface(row):

            removed_surfaces.append(
                row
            )

        else:

            kept.append(
                row
            )

    # --------------------------------------------------------
    # Conservative duplicate grouping
    # --------------------------------------------------------

    used = set()
    final = []
    merge_groups = []

    for i in range(
        len(kept)
    ):

        if i in used:
            continue

        group = [
            kept[i]
        ]

        used.add(i)

        for j in range(
            i + 1,
            len(kept)
        ):

            if j in used:
                continue

            if strong_duplicate(
                kept[i],
                kept[j]
            ):

                group.append(
                    kept[j]
                )

                used.add(j)

        merged = choose_best(
            group
        )

        final.append(
            merged
        )

        if len(group) > 1:

            merge_groups.append({
                "representative":
                    merged,

                "members":
                    group,
            })

    for index, row in enumerate(
        final,
        start=1
    ):

        row["id"] = index

    result = {
        "stage":
            "TEST07_STAGE02_V33",

        "input_inventory":
            str(input_json),

        "input_objects":
            len(objects),

        "removed_structural_surfaces":
            len(
                removed_surfaces
            ),

        "residual_duplicate_groups_merged":
            len(
                merge_groups
            ),

        "final_inventory":
            len(final),

        "objects":
            final,
    }

    inventory_path = (
        output_dir
        / "00_final_inventory.json"
    )

    inventory_path.write_text(
        json.dumps(
            result,
            indent=2
        )
    )

    (
        output_dir
        / "01_removed_structural_surfaces.json"
    ).write_text(
        json.dumps(
            removed_surfaces,
            indent=2
        )
    )

    (
        output_dir
        / "02_residual_duplicate_merges.json"
    ).write_text(
        json.dumps(
            merge_groups,
            indent=2
        )
    )

    report = [
        "=" * 90,
        "TEST07 STAGE02-V3.3 RESULT",
        "=" * 90,
        f"INPUT OBJECTS: {len(objects)}",
        (
            "REMOVED STRUCTURAL SURFACES: "
            f"{len(removed_surfaces)}"
        ),
        (
            "RESIDUAL DUPLICATE GROUPS MERGED: "
            f"{len(merge_groups)}"
        ),
        f"FINAL INVENTORY: {len(final)}",
        "",
        "OBJECT LIST:",
    ]

    for row in final:

        report.append(
            f'{row["id"]:03d}. '
            f'{row.get("name", "")} | '
            f'{row.get("confidence", "")} | '
            f'{row.get("grounding_phrase", "")} | '
            f'sources={row.get("source_regions", [])} | '
            f'merged={row.get("merged_candidate_count", 1)}'
        )

    report_path = (
        output_dir
        / "03_inventory_report.txt"
    )

    report_path.write_text(
        "\n".join(report)
    )

    print(
        "REMOVED STRUCTURAL SURFACES:",
        len(
            removed_surfaces
        )
    )

    print(
        "RESIDUAL DUPLICATE GROUPS MERGED:",
        len(
            merge_groups
        )
    )

    print(
        "FINAL INVENTORY:",
        len(final)
    )

    print()
    print("OBJECT LIST:")

    for row in final:

        print(
            f'{row["id"]:03d}. '
            f'{row.get("name", "")} | '
            f'{row.get("confidence", "")} | '
            f'{row.get("grounding_phrase", "")} | '
            f'sources={row.get("source_regions", [])} | '
            f'merged={row.get("merged_candidate_count", 1)}'
        )

    print()
    print("INVENTORY:")
    print(inventory_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-json",
        required=True
    )

    parser.add_argument(
        "--output-dir",
        required=True
    )

    args = parser.parse_args()

    run(
        input_json=
            args.input_json,

        output_dir=
            args.output_dir
    )
