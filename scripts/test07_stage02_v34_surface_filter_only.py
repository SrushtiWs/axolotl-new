
from pathlib import Path
import argparse
import json
import re


# ============================================================
# TEST07 STAGE02-V3.4
#
# CONSERVATIVE FINAL INVENTORY
#
# INPUT:
#   Stage02-V3.2 pairwise verified inventory
#
# PURPOSE:
#   Remove ONLY prohibited continuous structural surfaces.
#
# IMPORTANT:
#   NO textual duplicate merging.
#
# Duplicate physical instances will be resolved AFTER
# localization, when actual spatial bounding boxes exist.
# ============================================================


STRUCTURAL_NAMES = {
    "wall",
    "floor",
    "ceiling",
    "ground",
    "sky",
    "background",
}


def normalize(text):

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


def is_structural_surface(row):

    name = normalize(
        row.get(
            "name",
            ""
        )
    )

    phrase = normalize(
        row.get(
            "grounding_phrase",
            ""
        )
    )

    if name in STRUCTURAL_NAMES:
        return True

    explicit_surface_phrases = {
        "floor surface",
        "tiled floor surface",
        "wall surface",
        "ceiling surface",
        "ground surface",
        "continuous wall",
        "continuous floor",
        "continuous ceiling",
    }

    if phrase in explicit_surface_phrases:
        return True

    return False


def run(
    input_json,
    output_dir
):

    input_json = Path(
        input_json
    )

    output_dir = Path(
        output_dir
    )

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
    print("TEST07 STAGE02-V3.4")
    print("CONSERVATIVE SURFACE FILTER ONLY")
    print("=" * 90)

    print(
        "INPUT OBJECTS:",
        len(objects)
    )

    final = []
    removed = []

    for row in objects:

        if is_structural_surface(
            row
        ):

            removed.append(
                row
            )

        else:

            final.append(
                dict(row)
            )

    for i, row in enumerate(
        final,
        start=1
    ):

        row["id"] = i

    result = {
        "stage":
            "TEST07_STAGE02_V34",

        "method":
            "V3.2 pairwise inventory + structural surface removal only",

        "input_inventory":
            str(
                input_json
            ),

        "input_objects":
            len(
                objects
            ),

        "removed_structural_surfaces":
            len(
                removed
            ),

        "final_inventory":
            len(
                final
            ),

        "objects":
            final,

        "rules": [
            "no textual duplicate merge",
            "no room-specific vocabulary",
            "continuous structural surfaces removed",
            "possible duplicate instances preserved until spatial localization",
        ]
    }

    inventory_path = (
        output_dir
        / "00_conservative_inventory.json"
    )

    inventory_path.write_text(
        json.dumps(
            result,
            indent=2
        )
    )

    removed_path = (
        output_dir
        / "01_removed_surfaces.json"
    )

    removed_path.write_text(
        json.dumps(
            removed,
            indent=2
        )
    )

    report = [
        "=" * 90,
        "TEST07 STAGE02-V3.4 RESULT",
        "=" * 90,
        f"INPUT OBJECTS: {len(objects)}",
        f"REMOVED STRUCTURAL SURFACES: {len(removed)}",
        f"FINAL INVENTORY: {len(final)}",
        "",
        "OBJECT LIST:",
    ]

    for row in final:

        regions = row.get(
            "source_regions",
            [
                row.get(
                    "source_region",
                    ""
                )
            ]
        )

        report.append(
            f'{row["id"]:03d}. '
            f'{row.get("name", "")} | '
            f'{row.get("confidence", "")} | '
            f'{row.get("grounding_phrase", "")} | '
            f'sources={regions}'
        )

    report_path = (
        output_dir
        / "02_inventory_report.txt"
    )

    report_path.write_text(
        "\n".join(
            report
        )
    )

    print(
        "REMOVED STRUCTURAL SURFACES:",
        len(
            removed
        )
    )

    print(
        "FINAL INVENTORY:",
        len(
            final
        )
    )

    print()
    print("REMOVED:")

    for row in removed:

        print(
            "-",
            row.get(
                "name",
                ""
            ),
            "|",
            row.get(
                "grounding_phrase",
                ""
            )
        )

    print()
    print("INVENTORY:")
    print(inventory_path)

    print()
    print("REPORT:")
    print(report_path)


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
