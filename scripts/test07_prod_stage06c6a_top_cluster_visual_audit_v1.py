
from pathlib import Path
from PIL import Image, ImageDraw
import json
import math


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

MASTER = (
    PROD
    / "stage01_master"
    / "00_master_input.png"
)

CONSENSUS_JSON = (
    STAGE06
    / "06c6_multi_route_candidate_consensus"
    / "00_multi_route_consensus_all.json"
)

CANONICAL_JSON = (
    STAGE06
    / "06c6_multi_route_candidate_consensus"
    / "01_canonical_sam2_seed_boxes.json"
)

OUT = (
    STAGE06
    / "06c6a_top_cluster_visual_audit"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

SHEETS = (
    OUT
    / "instance_sheets"
)

SHEETS.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD
# ============================================================

for p in [
    MASTER,
    CONSENSUS_JSON,
    CANONICAL_JSON,
]:

    if not p.exists():
        raise FileNotFoundError(p)


master = Image.open(
    MASTER
).convert(
    "RGB"
)

W, H = master.size


consensus = json.loads(
    CONSENSUS_JSON.read_text(
        encoding="utf-8"
    )
)

canonical = json.loads(
    CANONICAL_JSON.read_text(
        encoding="utf-8"
    )
)


# ============================================================
# HELPERS
# ============================================================

def clamp_box(box):

    x1, y1, x2, y2 = [
        float(v)
        for v in box
    ]

    return [
        max(0, min(W, x1)),
        max(0, min(H, y1)),
        max(0, min(W, x2)),
        max(0, min(H, y2)),
    ]


def crop_context(
    box,
    expansion=0.80,
):

    x1, y1, x2, y2 = clamp_box(
        box
    )

    bw = max(
        4.0,
        x2 - x1
    )

    bh = max(
        4.0,
        y2 - y1
    )

    mx = max(
        25.0,
        bw * expansion
    )

    my = max(
        25.0,
        bh * expansion
    )

    cx1 = int(
        max(
            0,
            math.floor(
                x1 - mx
            )
        )
    )

    cy1 = int(
        max(
            0,
            math.floor(
                y1 - my
            )
        )
    )

    cx2 = int(
        min(
            W,
            math.ceil(
                x2 + mx
            )
        )
    )

    cy2 = int(
        min(
            H,
            math.ceil(
                y2 + my
            )
        )
    )

    crop = master.crop(
        (
            cx1,
            cy1,
            cx2,
            cy2
        )
    )

    draw = ImageDraw.Draw(
        crop
    )

    draw.rectangle(
        [
            x1 - cx1,
            y1 - cy1,
            x2 - cx1,
            y2 - cy1,
        ],
        outline="red",
        width=3
    )

    return crop


def fit_panel(
    image,
    width=340,
    height=300,
):

    image = image.copy()

    image.thumbnail(
        (
            width - 20,
            height - 65
        )
    )

    panel = Image.new(
        "RGB",
        (
            width,
            height
        ),
        "white"
    )

    x = (
        width
        -
        image.width
    ) // 2

    y = 50

    panel.paste(
        image,
        (
            x,
            y
        )
    )

    return panel


# ============================================================
# CURRENT CANONICAL OVERVIEW
# ============================================================

overview = master.copy()

draw = ImageDraw.Draw(
    overview
)


for row in canonical:

    iid = int(
        row[
            "inventory_id"
        ]
    )

    name = row.get(
        "inventory_name",
        ""
    )

    x1, y1, x2, y2 = clamp_box(
        row[
            "bbox"
        ]
    )

    draw.rectangle(
        [
            x1,
            y1,
            x2,
            y2
        ],
        outline="red",
        width=2
    )

    draw.text(
        (
            x1,
            max(
                0,
                y1 - 12
            )
        ),
        f"{iid}:{name}",
        fill="red"
    )


OVERVIEW_PATH = (
    OUT
    / "00_current_canonical_seed_overview.png"
)

overview.save(
    OVERVIEW_PATH
)


# ============================================================
# PER-INSTANCE TOP-3 CLUSTER SHEETS
# ============================================================

diagnostic_rows = []


for instance in consensus:

    iid = int(
        instance[
            "inventory_id"
        ]
    )

    name = instance.get(
        "inventory_name",
        ""
    )

    phrase = instance.get(
        "grounding_phrase",
        ""
    )

    clusters = instance.get(
        "clusters",
        []
    )[:3]


    panels = []


    for rank, cluster in enumerate(
        clusters,
        start=1
    ):

        box = cluster[
            "bbox"
        ]

        crop = crop_context(
            box
        )

        panel = fit_panel(
            crop
        )

        pdraw = ImageDraw.Draw(
            panel
        )


        title1 = (
            f"#{rank}  "
            f"routes={cluster['route_count']}  "
            f"support={cluster['proposal_count']}"
        )

        title2 = (
            f"max={cluster['max_dino_score']:.3f}  "
            f"cons={cluster['consensus_score']:.3f}"
        )


        pdraw.text(
            (
                8,
                6
            ),
            title1,
            fill="black"
        )

        pdraw.text(
            (
                8,
                24
            ),
            title2,
            fill="black"
        )


        panels.append(
            panel
        )


        x1, y1, x2, y2 = [
            float(v)
            for v in box
        ]


        diagnostic_rows.append({

            "inventory_id":
                iid,

            "inventory_name":
                name,

            "cluster_rank":
                rank,

            "bbox":
                box,

            "box_width":
                x2 - x1,

            "box_height":
                y2 - y1,

            "box_area":
                max(
                    0.0,
                    x2 - x1
                )
                *
                max(
                    0.0,
                    y2 - y1
                ),

            "proposal_count":
                cluster[
                    "proposal_count"
                ],

            "routes":
                cluster[
                    "routes"
                ],

            "route_count":
                cluster[
                    "route_count"
                ],

            "source_regions":
                cluster[
                    "source_regions"
                ],

            "max_dino_score":
                cluster[
                    "max_dino_score"
                ],

            "mean_dino_score":
                cluster[
                    "mean_dino_score"
                ],

            "consensus_score":
                cluster[
                    "consensus_score"
                ],
        })


    if not panels:

        continue


    sheet_width = (
        340
        *
        len(
            panels
        )
    )

    sheet_height = 350


    sheet = Image.new(
        "RGB",
        (
            sheet_width,
            sheet_height
        ),
        "white"
    )


    for i, panel in enumerate(
        panels
    ):

        sheet.paste(
            panel,
            (
                i * 340,
                50
            )
        )


    header = ImageDraw.Draw(
        sheet
    )

    header.text(
        (
            8,
            8
        ),
        f"{iid:03d}. {name}",
        fill="black"
    )

    header.text(
        (
            8,
            26
        ),
        phrase,
        fill="black"
    )


    safe_name = "".join(
        c
        if c.isalnum() or c in "-_"
        else "_"
        for c in name
    )


    SHEET_PATH = (
        SHEETS
        /
        f"{iid:03d}_{safe_name}_top3.png"
    )


    sheet.save(
        SHEET_PATH
    )


# ============================================================
# SAVE DIAGNOSTIC JSON
# ============================================================

DIAGNOSTIC_PATH = (
    OUT
    / "01_top3_cluster_geometry.json"
)


DIAGNOSTIC_PATH.write_text(
    json.dumps(
        diagnostic_rows,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# ============================================================
# PRINT
# ============================================================

print("=" * 110)
print("PRODUCTION STAGE 06C6A RESULT")
print("=" * 110)

print(
    "PRESENT INSTANCES:",
    len(
        consensus
    )
)

print(
    "TOP-3 CLUSTERS EXPORTED:",
    len(
        diagnostic_rows
    )
)

print()

for instance in consensus:

    iid = int(
        instance[
            "inventory_id"
        ]
    )

    name = instance[
        "inventory_name"
    ]

    print()
    print(
        f"{iid:03d}. {name}"
    )

    for rank, cluster in enumerate(
        instance.get(
            "clusters",
            []
        )[:3],
        start=1
    ):

        box = cluster[
            "bbox"
        ]

        width = (
            box[2]
            -
            box[0]
        )

        height = (
            box[3]
            -
            box[1]
        )

        print(
            "   #{} | routes={} | support={} | max={:.4f} | consensus={:.4f} | size={:.1f}x{:.1f} | bbox={}".format(

                rank,

                cluster[
                    "route_count"
                ],

                cluster[
                    "proposal_count"
                ],

                cluster[
                    "max_dino_score"
                ],

                cluster[
                    "consensus_score"
                ],

                width,

                height,

                [
                    round(
                        v,
                        1
                    )
                    for v in box
                ],
            )
        )


print()
print(
    "CANONICAL OVERVIEW:",
    OVERVIEW_PATH
)

print(
    "INSTANCE SHEETS:",
    SHEETS
)

print(
    "GEOMETRY JSON:",
    DIAGNOSTIC_PATH
)

print()
print(
    "DIAGNOSTIC ONLY — NO SEEDS MODIFIED."
)
