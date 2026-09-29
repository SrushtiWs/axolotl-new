
from pathlib import Path
import json
import re
import numpy as np

from PIL import (
    Image,
    ImageDraw
)

import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

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

MASTER_PATH = (
    PROD
    / "stage05_clean_room_with_props"
    / "12_final_canonical_prop_detection_master.png"
)

INVENTORY_PATH = (
    STAGE06
    / "06d2a2_verified_main_prop_state"
    / "00_stage06d2a2_verified_main_prop_state.json"
)

D2B_PATH = (
    STAGE06
    / "06d2b_complete_main_prop_localization"
    / "00_stage06d2b_result.json"
)

D1D4_PATH = (
    STAGE06
    / "06d1d4_crossmodel_localization_support"
    / "00_stage06d1d4_result.json"
)

OUT = (
    STAGE06
    / "06d2b2_evidence_completed_main_prop_localization"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

PREVIEW_DIR = (
    OUT
    / "previews"
)

PREVIEW_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# VALIDATE
# ============================================================

for path in [
    MASTER_PATH,
    INVENTORY_PATH,
    D2B_PATH,
    D1D4_PATH,
]:

    if not path.exists():

        raise FileNotFoundError(
            path
        )


master = Image.open(
    MASTER_PATH
).convert("RGB")

W, H = master.size


inventory = json.loads(
    INVENTORY_PATH.read_text(
        encoding="utf-8"
    )
)

d2b = json.loads(
    D2B_PATH.read_text(
        encoding="utf-8"
    )
)

d1d4 = json.loads(
    D1D4_PATH.read_text(
        encoding="utf-8"
    )
)


# ============================================================
# INDEXES
# ============================================================

prop_by_id = {

    p["main_prop_id"]:
        p

    for p in inventory["main_props"]
}


d2b_by_id = {

    p["main_prop_id"]:
        p

    for p in d2b["results"]
}


evidence = d1d4["results"]


# ============================================================
# HELPERS
# ============================================================

def show(
    image_or_path,
    title,
    figsize=(8, 9)
):

    if isinstance(
        image_or_path,
        (str, Path)
    ):

        image = Image.open(
            image_or_path
        )

    else:

        image = image_or_path


    plt.figure(
        figsize=figsize
    )

    plt.imshow(image)

    plt.title(title)

    plt.axis("off")

    plt.show()


def clamp_box(box):

    x1, y1, x2, y2 = [
        float(v)
        for v in box
    ]

    return [

        max(
            0.0,
            min(
                float(W - 1),
                x1
            )
        ),

        max(
            0.0,
            min(
                float(H - 1),
                y1
            )
        ),

        max(
            1.0,
            min(
                float(W),
                x2
            )
        ),

        max(
            1.0,
            min(
                float(H),
                y2
            )
        ),
    ]


def union_boxes(boxes):

    boxes = [
        clamp_box(b)
        for b in boxes
        if b
    ]

    if not boxes:
        return None

    return clamp_box(
        [
            min(b[0] for b in boxes),
            min(b[1] for b in boxes),
            max(b[2] for b in boxes),
            max(b[3] for b in boxes),
        ]
    )


def expand_box(
    box,
    x_ratio,
    y_ratio,
):

    x1, y1, x2, y2 = box

    bw = max(
        1.0,
        x2 - x1
    )

    bh = max(
        1.0,
        y2 - y1
    )

    return clamp_box(
        [
            x1 - bw * x_ratio,
            y1 - bh * y_ratio,
            x2 + bw * x_ratio,
            y2 + bh * y_ratio,
        ]
    )


def evidence_name(row):

    return str(
        row.get(
            "final_name"
        )
        or
        row.get(
            "input_name"
        )
        or
        ""
    ).strip().lower()


def get_evidence_boxes(
    allowed_names,
):

    allowed_names = {
        x.lower()
        for x in allowed_names
    }

    rows = []


    for row in evidence:

        name = evidence_name(
            row
        )

        if name not in allowed_names:
            continue


        bbox = row.get(
            "crossmodel_canonical_bbox"
        )

        if not bbox:
            continue


        rows.append({

            "evidence_id":
                row["evidence_id"],

            "name":
                name,

            "bbox":
                clamp_box(
                    bbox
                ),
        })


    return rows


def get_d2b_rank1(
    prop_id
):

    row = d2b_by_id[
        prop_id
    ]

    clusters = row.get(
        "candidate_clusters",
        []
    )

    for cluster in clusters:

        if int(
            cluster.get(
                "rank",
                0
            )
        ) == 1:

            return clamp_box(
                cluster["bbox"]
            )

    return None


def box_area(box):

    return max(
        0.0,
        box[2] - box[0]
    ) * max(
        0.0,
        box[3] - box[1]
    )


def choose_smallest_evidence_box(
    rows
):

    if not rows:
        return None

    return min(
        rows,
        key=lambda r:
            box_area(
                r["bbox"]
            )
    )


# ============================================================
# FINAL LOCALIZATION BUILD
# ============================================================

results = []


# ============================================================
# P01 — COMPLETE VANITY SYSTEM
#
# Whole Florence rank1 provides main body.
# Existing evidence is used only to make sure all connected
# visible members are inside the final region.
# ============================================================

p01_base = get_d2b_rank1(
    "P01"
)

p01_evidence = get_evidence_boxes(
    {
        "cabinet",
        "countertop",
        "sink",
        "sink faucet",
        "faucet",
        "bottle",
        "drawer handle",
        "door handle",
    }
)


p01_boxes = [
    p01_base
]

p01_boxes.extend(
    r["bbox"]
    for r in p01_evidence
)


p01_final = union_boxes(
    p01_boxes
)


# Small safety margin around complete connected assembly.
p01_final = expand_box(
    p01_final,
    x_ratio=0.025,
    y_ratio=0.035,
)


results.append({

    "main_prop_id":
        "P01",

    "main_prop_name":
        prop_by_id["P01"][
            "main_prop_name"
        ],

    "source":
        "D2B_WHOLE_PROP_PLUS_MEMBER_COMPLETENESS",

    "base_whole_prop_bbox":
        p01_base,

    "supporting_evidence":
        p01_evidence,

    "final_bbox":
        p01_final,

    "status":
        "FINAL_LOCALIZATION_READY",
})


# ============================================================
# P02 — COMPLETE TOILET SYSTEM
#
# D2B rank1 is already visually correct.
#
# We allow verified toilet/seat/base evidence to complete it,
# but explicitly exclude:
#   toilet tank
#   toilet paper holder
# ============================================================

p02_base = get_d2b_rank1(
    "P02"
)

p02_evidence = get_evidence_boxes(
    {
        "toilet",
        "toilet seat",
        "toilet seat cover",
        "toilet base",
    }
)


p02_boxes = [
    p02_base
]

p02_boxes.extend(
    r["bbox"]
    for r in p02_evidence
)


p02_final = union_boxes(
    p02_boxes
)


p02_final = expand_box(
    p02_final,
    x_ratio=0.02,
    y_ratio=0.025,
)


results.append({

    "main_prop_id":
        "P02",

    "main_prop_name":
        prop_by_id["P02"][
            "main_prop_name"
        ],

    "source":
        "D2B_WHOLE_PROP_PLUS_MEMBER_COMPLETENESS",

    "base_whole_prop_bbox":
        p02_base,

    "supporting_evidence":
        p02_evidence,

    "explicitly_excluded_evidence_names": [
        "toilet tank",
        "toilet paper holder",
    ],

    "final_bbox":
        p02_final,

    "status":
        "FINAL_LOCALIZATION_READY",
})


# ============================================================
# P03 — SHOWER FIXTURE
#
# D2B rank1 is visually correct.
# Shower evidence only.
# ============================================================

p03_base = get_d2b_rank1(
    "P03"
)

p03_evidence = get_evidence_boxes(
    {
        "shower head",
        "shower fixture",
    }
)


p03_boxes = [
    p03_base
]

p03_boxes.extend(
    r["bbox"]
    for r in p03_evidence
)


p03_final = union_boxes(
    p03_boxes
)


p03_final = expand_box(
    p03_final,
    x_ratio=0.05,
    y_ratio=0.08,
)


results.append({

    "main_prop_id":
        "P03",

    "main_prop_name":
        prop_by_id["P03"][
            "main_prop_name"
        ],

    "source":
        "D2B_WHOLE_PROP_PLUS_SHOWER_EVIDENCE",

    "base_whole_prop_bbox":
        p03_base,

    "supporting_evidence":
        p03_evidence,

    "final_bbox":
        p03_final,

    "status":
        "FINAL_LOCALIZATION_READY",
})


# ============================================================
# P04 — WALL ELECTRICAL PLATE
#
# D2B failed.
# Reuse already-established electrical-plate localization.
# ============================================================

p04_evidence = get_evidence_boxes(
    {
        "electrical plate",
        "light switch",
        "wall switch",
    }
)


p04_selected = choose_smallest_evidence_box(
    p04_evidence
)


p04_final = (
    expand_box(
        p04_selected["bbox"],
        x_ratio=0.08,
        y_ratio=0.08,
    )
    if p04_selected
    else None
)


results.append({

    "main_prop_id":
        "P04",

    "main_prop_name":
        prop_by_id["P04"][
            "main_prop_name"
        ],

    "source":
        "REUSED_EXISTING_VERIFIED_ELECTRICAL_PLATE_LOCALIZATION",

    "supporting_evidence":
        p04_evidence,

    "selected_evidence":
        p04_selected,

    "final_bbox":
        p04_final,

    "status":
        (
            "FINAL_LOCALIZATION_READY"
            if p04_final
            else
            "UNRESOLVED"
        ),
})


# ============================================================
# P05 — TOILET PAPER HOLDER
#
# D2B failed.
# Reuse existing verified toilet-paper-holder localization.
# ============================================================

p05_evidence = get_evidence_boxes(
    {
        "toilet paper holder",
    }
)


p05_selected = choose_smallest_evidence_box(
    p05_evidence
)


p05_final = (
    expand_box(
        p05_selected["bbox"],
        x_ratio=0.10,
        y_ratio=0.10,
    )
    if p05_selected
    else None
)


results.append({

    "main_prop_id":
        "P05",

    "main_prop_name":
        prop_by_id["P05"][
            "main_prop_name"
        ],

    "source":
        "REUSED_EXISTING_TOILET_PAPER_HOLDER_LOCALIZATION",

    "supporting_evidence":
        p05_evidence,

    "selected_evidence":
        p05_selected,

    "final_bbox":
        p05_final,

    "status":
        (
            "FINAL_LOCALIZATION_READY"
            if p05_final
            else
            "UNRESOLVED"
        ),
})


# ============================================================
# P06 — CEILING LIGHT
#
# D2B rank1 visually correct.
# Existing light evidence used only as support.
# ============================================================

p06_base = get_d2b_rank1(
    "P06"
)

p06_evidence = get_evidence_boxes(
    {
        "ceiling light",
        "light fixture",
    }
)


# Use the D2B rank1 because it is already a tight correct box.
p06_final = expand_box(
    p06_base,
    x_ratio=0.08,
    y_ratio=0.08,
)


results.append({

    "main_prop_id":
        "P06",

    "main_prop_name":
        prop_by_id["P06"][
            "main_prop_name"
        ],

    "source":
        "D2B_WHOLE_PROP_RANK1",

    "base_whole_prop_bbox":
        p06_base,

    "supporting_evidence":
        p06_evidence,

    "final_bbox":
        p06_final,

    "status":
        "FINAL_LOCALIZATION_READY",
})


# ============================================================
# SAVE INDIVIDUAL + OVERVIEW PREVIEWS
# ============================================================

for result in results:

    preview = master.copy()

    draw = ImageDraw.Draw(
        preview
    )


    box = result.get(
        "final_bbox"
    )


    if box:

        draw.rectangle(
            box,
            outline="lime",
            width=4
        )


        x1, y1, _, _ = box


        draw.rectangle(
            [
                x1,
                y1,
                x1 + 55,
                y1 + 25
            ],
            fill="white",
            outline="lime"
        )


        draw.text(
            (
                x1 + 5,
                y1 + 4
            ),
            result[
                "main_prop_id"
            ],
            fill="green"
        )


    path = (
        PREVIEW_DIR
        /
        f'{result["main_prop_id"]}_final_localization.png'
    )


    preview.save(
        path
    )


    result[
        "preview_path"
    ] = str(
        path
    )


# ============================================================
# GLOBAL OVERVIEW
# ============================================================

overview = master.copy()

draw = ImageDraw.Draw(
    overview
)


for result in results:

    box = result.get(
        "final_bbox"
    )


    if not box:
        continue


    draw.rectangle(
        box,
        outline="lime",
        width=3
    )


    x1, y1, _, _ = box


    draw.text(
        (
            x1 + 3,
            y1 + 3
        ),
        result[
            "main_prop_id"
        ],
        fill="green"
    )


OVERVIEW_PATH = (
    OUT
    / "01_all_final_main_prop_localizations.png"
)


overview.save(
    OVERVIEW_PATH
)


# ============================================================
# FINAL STATE
# ============================================================

all_ready = all(
    row[
        "status"
    ]
    ==
    "FINAL_LOCALIZATION_READY"

    for row in results
)


FINAL_STATE = {

    "stage":
        "06D2B2",

    "input_inventory":
        str(
            INVENTORY_PATH
        ),

    "detection_master":
        str(
            MASTER_PATH
        ),

    "final_main_prop_count":
        len(
            results
        ),

    "all_localizations_ready":
        all_ready,

    "results":
        results,

    "overview_preview":
        str(
            OVERVIEW_PATH
        ),

    "status":
        (
            "READY_FOR_COMPLETE_MAIN_PROP_SEGMENTATION_AUDIT"
            if all_ready
            else
            "LOCALIZATION_REVIEW_REQUIRED"
        ),

    "rules": [

        "exactly one localization per final main prop",

        "child components do not become extraction targets",

        "component boxes may only complete a connected-system bounding region",

        "false toilet tank excluded from toilet",

        "toilet paper holder remains separate from toilet",

        "Stage05F used for localization",

        "Stage01 remains final RGB source",

        "no SAM2 performed",

        "no final mask union created",
    ],
}


RESULT_PATH = (
    OUT
    / "00_stage06d2b2_result.json"
)


RESULT_PATH.write_text(
    json.dumps(
        FINAL_STATE,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# ============================================================
# PRINT
# ============================================================

print("=" * 110)
print("PRODUCTION STAGE 06D2B2 RESULT")
print("=" * 110)

print()
print(
    "ALL LOCALIZATIONS READY:",
    all_ready
)


for result in results:

    print()

    print(
        "{}. {} | {}".format(

            result[
                "main_prop_id"
            ],

            result[
                "main_prop_name"
            ],

            result[
                "status"
            ],
        )
    )

    print(
        "   SOURCE:",
        result[
            "source"
        ]
    )

    print(
        "   FINAL BBOX:",
        (
            [
                round(v, 1)
                for v in result[
                    "final_bbox"
                ]
            ]
            if result.get(
                "final_bbox"
            )
            else None
        )
    )


print()
print(
    "RESULT JSON:",
    RESULT_PATH
)

print(
    "OVERVIEW:",
    OVERVIEW_PATH
)

print()
print(
    "NO NEW PROP DETECTION WAS RUN."
)

print(
    "NO CHILD-PROP EXTRACTION WAS RUN."
)

print(
    "NO SAM2 OR FINAL MASK UNION WAS RUN."
)


# ============================================================
# INLINE REVIEW
# ============================================================

print()
print("=" * 110)
print("INLINE 06D2B2 FINAL LOCALIZATION AUDIT")
print("=" * 110)


show(
    OVERVIEW_PATH,
    (
        "06D2B2 — ALL SIX FINAL MAIN-PROP LOCALIZATIONS\n"
        "GREEN = Final Main-Prop Bounding Region"
    ),
)


for result in results:

    show(
        result[
            "preview_path"
        ],
        (
            f'{result["main_prop_id"]} — '
            f'{result["main_prop_name"]}\n'
            'GREEN = Final Localization'
        ),
        figsize=(7, 8)
    )
