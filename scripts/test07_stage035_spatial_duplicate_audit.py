
from pathlib import Path
import argparse
import json
import math
import re


# ============================================================
# TEST07 — STAGE03.5
#
# SPATIAL DUPLICATE INVENTORY AUDIT
#
# INPUT:
#   Stage02-V3.4 conservative inventory
#   Stage03 Grounding DINO deduplicated candidates
#
# PURPOSE:
#   Find inventory records that may describe the SAME physical
#   object using actual localization geometry.
#
# IMPORTANT:
#   AUDIT ONLY.
#   NOTHING IS MERGED IN THIS STAGE.
# ============================================================


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


def bbox_area(box):

    x1, y1, x2, y2 = box

    return max(
        0.0,
        x2 - x1
    ) * max(
        0.0,
        y2 - y1
    )


def bbox_iou(a, b):

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(
        0.0,
        ix2 - ix1
    )

    ih = max(
        0.0,
        iy2 - iy1
    )

    inter = iw * ih

    if inter <= 0:
        return 0.0

    union = (
        bbox_area(a)
        +
        bbox_area(b)
        -
        inter
    )

    if union <= 0:
        return 0.0

    return inter / union


def containment(a, b):

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(
        0.0,
        ix2 - ix1
    )

    ih = max(
        0.0,
        iy2 - iy1
    )

    inter = iw * ih

    smaller = min(
        bbox_area(a),
        bbox_area(b)
    )

    if smaller <= 0:
        return 0.0

    return inter / smaller


def center_distance_normalized(a, b):

    acx = (
        a[0] + a[2]
    ) / 2.0

    acy = (
        a[1] + a[3]
    ) / 2.0

    bcx = (
        b[0] + b[2]
    ) / 2.0

    bcy = (
        b[1] + b[3]
    ) / 2.0

    aw = max(
        1.0,
        a[2] - a[0]
    )

    ah = max(
        1.0,
        a[3] - a[1]
    )

    bw = max(
        1.0,
        b[2] - b[0]
    )

    bh = max(
        1.0,
        b[3] - b[1]
    )

    scale = max(
        1.0,
        (
            math.sqrt(
                aw * aw
                +
                ah * ah
            )
            +
            math.sqrt(
                bw * bw
                +
                bh * bh
            )
        )
        /
        2.0
    )

    dist = math.sqrt(
        (acx - bcx) ** 2
        +
        (acy - bcy) ** 2
    )

    return dist / scale


def token_similarity(a, b):

    ta = set(
        normalize(a).split()
    )

    tb = set(
        normalize(b).split()
    )

    if not ta or not tb:
        return 0.0

    return (
        len(ta & tb)
        /
        len(ta | tb)
    )


def run(
    inventory_json,
    dino_json,
    output_dir
):

    inventory_json = Path(
        inventory_json
    )

    dino_json = Path(
        dino_json
    )

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    inventory_data = json.loads(
        inventory_json.read_text()
    )

    inventory = inventory_data.get(
        "objects",
        []
    )

    dino = json.loads(
        dino_json.read_text()
    )

    if not isinstance(
        dino,
        list
    ):
        raise RuntimeError(
            "Stage03 DINO JSON must be a list."
        )

    print("=" * 90)
    print("TEST07 STAGE03.5")
    print("SPATIAL DUPLICATE AUDIT")
    print("=" * 90)

    print(
        "INVENTORY ITEMS:",
        len(inventory)
    )

    print(
        "DINO CANDIDATES:",
        len(dino)
    )

    # ========================================================
    # BEST DINO BOX PER INVENTORY INSTANCE
    # ========================================================

    grouped = {}

    for row in dino:

        iid = int(
            row.get(
                "inventory_id"
            )
        )

        grouped.setdefault(
            iid,
            []
        ).append(
            row
        )

    best = {}

    for item in inventory:

        iid = int(
            item["id"]
        )

        rows = grouped.get(
            iid,
            []
        )

        if not rows:
            continue

        rows = sorted(
            rows,
            key=lambda r:
                float(
                    r.get(
                        "score",
                        0
                    )
                ),
            reverse=True
        )

        top = rows[0]

        best[iid] = {
            "inventory_id":
                iid,

            "name":
                item.get(
                    "name",
                    ""
                ),

            "grounding_phrase":
                item.get(
                    "grounding_phrase",
                    ""
                ),

            "confidence":
                item.get(
                    "confidence",
                    ""
                ),

            "bbox":
                [
                    float(v)
                    for v in top[
                        "bbox"
                    ]
                ],

            "dino_score":
                float(
                    top.get(
                        "score",
                        0
                    )
                ),

            "dino_label":
                top.get(
                    "dino_label",
                    ""
                ),
        }

    # ========================================================
    # PAIRWISE AUDIT
    # ========================================================

    ids = sorted(
        best.keys()
    )

    suspicious = []

    for ai in range(
        len(ids)
    ):

        for bi in range(
            ai + 1,
            len(ids)
        ):

            a = best[
                ids[ai]
            ]

            b = best[
                ids[bi]
            ]

            iou = bbox_iou(
                a["bbox"],
                b["bbox"]
            )

            contain = containment(
                a["bbox"],
                b["bbox"]
            )

            center_dist = (
                center_distance_normalized(
                    a["bbox"],
                    b["bbox"]
                )
            )

            same_name = (
                normalize(
                    a["name"]
                )
                ==
                normalize(
                    b["name"]
                )
            )

            phrase_similarity = (
                token_similarity(
                    a[
                        "grounding_phrase"
                    ],
                    b[
                        "grounding_phrase"
                    ]
                )
            )

            # -----------------------------------------------
            # AUDIT GATES
            #
            # Very conservative:
            #
            # A pair is only surfaced if localization strongly
            # suggests the same physical region.
            # -----------------------------------------------

            reason = None

            if (
                iou >= 0.80
                and
                same_name
            ):

                reason = (
                    "same_name_high_iou"
                )

            elif (
                iou >= 0.90
                and
                phrase_similarity
                >= 0.35
            ):

                reason = (
                    "very_high_iou_phrase_support"
                )

            elif (
                contain >= 0.93
                and
                same_name
                and
                center_dist <= 0.20
            ):

                reason = (
                    "same_name_high_containment"
                )

            if reason is None:
                continue

            suspicious.append({
                "id_a":
                    a[
                        "inventory_id"
                    ],

                "name_a":
                    a[
                        "name"
                    ],

                "phrase_a":
                    a[
                        "grounding_phrase"
                    ],

                "score_a":
                    round(
                        a[
                            "dino_score"
                        ],
                        4
                    ),

                "bbox_a":
                    a[
                        "bbox"
                    ],

                "id_b":
                    b[
                        "inventory_id"
                    ],

                "name_b":
                    b[
                        "name"
                    ],

                "phrase_b":
                    b[
                        "grounding_phrase"
                    ],

                "score_b":
                    round(
                        b[
                            "dino_score"
                        ],
                        4
                    ),

                "bbox_b":
                    b[
                        "bbox"
                    ],

                "iou":
                    round(
                        iou,
                        4
                    ),

                "containment":
                    round(
                        contain,
                        4
                    ),

                "center_distance":
                    round(
                        center_dist,
                        4
                    ),

                "phrase_similarity":
                    round(
                        phrase_similarity,
                        4
                    ),

                "reason":
                    reason,
            })

    suspicious.sort(
        key=lambda r: (
            r["iou"],
            r["containment"]
        ),
        reverse=True
    )

    # ========================================================
    # SAVE
    # ========================================================

    (
        output_dir
        / "00_best_box_per_inventory.json"
    ).write_text(
        json.dumps(
            list(
                best.values()
            ),
            indent=2
        )
    )

    audit_path = (
        output_dir
        / "01_spatial_duplicate_candidates.json"
    )

    audit_path.write_text(
        json.dumps(
            suspicious,
            indent=2
        )
    )

    report = [
        "=" * 90,
        "TEST07 STAGE03.5 RESULT",
        "=" * 90,
        f"INVENTORY ITEMS: {len(inventory)}",
        f"LOCALIZED ITEMS: {len(best)}",
        (
            "SPATIAL DUPLICATE PAIRS: "
            f"{len(suspicious)}"
        ),
        "",
        "PAIR LIST:",
    ]

    for index, row in enumerate(
        suspicious,
        start=1
    ):

        report.append(
            (
                f'{index:03d}. '
                f'{row["id_a"]:02d} '
                f'{row["name_a"]}'
                f' ↔ '
                f'{row["id_b"]:02d} '
                f'{row["name_b"]}'
                f' | IoU={row["iou"]}'
                f' | contain={row["containment"]}'
                f' | center={row["center_distance"]}'
                f' | {row["reason"]}'
            )
        )

        report.append(
            f'     A: {row["phrase_a"]}'
        )

        report.append(
            f'     B: {row["phrase_b"]}'
        )

    report_path = (
        output_dir
        / "02_spatial_duplicate_report.txt"
    )

    report_path.write_text(
        "\n".join(
            report
        )
    )

    summary = {
        "stage":
            "TEST07_STAGE035",

        "inventory_items":
            len(
                inventory
            ),

        "localized_items":
            len(
                best
            ),

        "spatial_duplicate_pairs":
            len(
                suspicious
            ),

        "audit_json":
            str(
                audit_path
            ),

        "report":
            str(
                report_path
            ),

        "important":
            "AUDIT ONLY - no inventory records were merged"
    }

    (
        output_dir
        / "stage035_report.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2
        )
    )

    print()
    print("=" * 90)
    print("TEST07 STAGE03.5 RESULT")
    print("=" * 90)

    print(
        "LOCALIZED ITEMS:",
        len(
            best
        )
    )

    print(
        "SPATIAL DUPLICATE PAIRS:",
        len(
            suspicious
        )
    )

    print()
    print("PAIR LIST:")

    for index, row in enumerate(
        suspicious,
        start=1
    ):

        print(
            f'{index:03d}. '
            f'{row["id_a"]:02d} '
            f'{row["name_a"]}'
            f' ↔ '
            f'{row["id_b"]:02d} '
            f'{row["name_b"]}'
            f' | IoU={row["iou"]}'
            f' | contain={row["containment"]}'
            f' | center={row["center_distance"]}'
            f' | {row["reason"]}'
        )

        print(
            "     A:",
            row[
                "phrase_a"
            ]
        )

        print(
            "     B:",
            row[
                "phrase_b"
            ]
        )

    print()
    print("AUDIT ONLY — NOTHING MERGED.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--inventory-json",
        required=True
    )

    parser.add_argument(
        "--dino-json",
        required=True
    )

    parser.add_argument(
        "--output-dir",
        required=True
    )

    args = parser.parse_args()

    run(
        inventory_json=
            args.inventory_json,

        dino_json=
            args.dino_json,

        output_dir=
            args.output_dir
    )
