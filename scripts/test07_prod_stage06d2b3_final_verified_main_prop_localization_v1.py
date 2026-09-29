
from pathlib import Path
import json

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

P04_VERIFIED_PATH = (
    STAGE06
    / "06c13a_florence_wall_switch_sam2"
    / "01_stage06c13a_results.json"
)

P05_P06_VERIFIED_PATH = (
    STAGE06
    / "06c14b_semantic_audit_sam2_inline_preview"
    / "00_stage06c14b_state.json"
)

OUT = (
    STAGE06
    / "06d2b3_final_verified_main_prop_localization"
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
    P04_VERIFIED_PATH,
    P05_P06_VERIFIED_PATH,
]:

    if not path.exists():
        raise FileNotFoundError(path)


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

p04_verified = json.loads(
    P04_VERIFIED_PATH.read_text(
        encoding="utf-8"
    )
)

p05p06_verified = json.loads(
    P05_P06_VERIFIED_PATH.read_text(
        encoding="utf-8"
    )
)


# ============================================================
# INDEXES
# ============================================================

inventory_by_id = {
    row["main_prop_id"]: row
    for row in inventory["main_props"]
}

d2b_by_id = {
    row["main_prop_id"]: row
    for row in d2b["results"]
}


# ============================================================
# HELPERS
# ============================================================

def show(
    image_or_path,
    title,
    figsize=(7, 8)
):

    if isinstance(image_or_path, (str, Path)):
        image = Image.open(image_or_path)
    else:
        image = image_or_path

    plt.figure(figsize=figsize)
    plt.imshow(image)
    plt.title(title)
    plt.axis("off")
    plt.show()


def clamp_box(box):

    x1, y1, x2, y2 = [
        float(v)
        for v in box
    ]

    x1 = max(0.0, min(float(W - 1), x1))
    y1 = max(0.0, min(float(H - 1), y1))

    x2 = max(
        x1 + 1.0,
        min(float(W), x2)
    )

    y2 = max(
        y1 + 1.0,
        min(float(H), y2)
    )

    return [
        x1,
        y1,
        x2,
        y2
    ]


def get_d2b_rank(
    prop_id,
    rank=1
):

    clusters = d2b_by_id[
        prop_id
    ].get(
        "candidate_clusters",
        []
    )

    for cluster in clusters:

        if int(
            cluster.get("rank", 0)
        ) == rank:

            return clamp_box(
                cluster["bbox"]
            )

    return None


def union_boxes(boxes):

    valid = [
        clamp_box(box)
        for box in boxes
        if box
    ]

    if not valid:
        return None

    return clamp_box(
        [
            min(x[0] for x in valid),
            min(x[1] for x in valid),
            max(x[2] for x in valid),
            max(x[3] for x in valid),
        ]
    )


def expand_box(
    box,
    x_ratio,
    y_ratio
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


def horizontal_overlap_fraction(
    a,
    b
):

    overlap = max(
        0.0,
        min(a[2], b[2])
        -
        max(a[0], b[0])
    )

    smaller_width = max(
        1.0,
        min(
            a[2] - a[0],
            b[2] - b[0]
        )
    )

    return float(
        overlap
        /
        smaller_width
    )


def vertical_gap(
    a,
    b
):

    if a[3] < b[1]:
        return b[1] - a[3]

    if b[3] < a[1]:
        return a[1] - b[3]

    return 0.0


def area(box):

    return max(
        0.0,
        box[2] - box[0]
    ) * max(
        0.0,
        box[3] - box[1]
    )


def evidence_name(row):

    return str(
        row.get("final_name")
        or
        row.get("input_name")
        or
        ""
    ).strip().lower()


# ============================================================
# P01 VANITY
#
# Start from correct whole-prop D2B candidate #1:
#   [27,259,193,423]
#
# Only accept existing member-localization evidence when it is
# spatially connected to that base.
#
# This avoids the previous giant union.
# ============================================================

P01_BASE = get_d2b_rank(
    "P01",
    1
)


P01_ALLOWED_NAMES = {
    "cabinet",
    "countertop",
    "sink",
    "sink faucet",
    "faucet",
    "bottle",
    "drawer handle",
    "door handle",
}


p01_candidate_evidence = []


for row in d1d4["results"]:

    name = evidence_name(row)

    if name not in P01_ALLOWED_NAMES:
        continue

    bbox = row.get(
        "crossmodel_canonical_bbox"
    )

    if not bbox:
        continue

    bbox = clamp_box(
        bbox
    )


    # --------------------------------------------------------
    # Spatial attachment gate.
    #
    # Child/member evidence may extend above the vanity base
    # (basin/faucet/bottle), but it must remain horizontally
    # connected to the vanity.
    # --------------------------------------------------------

    h_overlap = horizontal_overlap_fraction(
        P01_BASE,
        bbox
    )

    v_gap = vertical_gap(
        P01_BASE,
        bbox
    )


    # Reject very large contaminated evidence.
    size_ok = bool(
        area(bbox)
        <=
        area(P01_BASE)
        *
        0.85
    )


    connected = bool(

        h_overlap >= 0.40

        and

        v_gap <= 35.0

        and

        size_ok
    )


    p01_candidate_evidence.append({

        "evidence_id":
            row["evidence_id"],

        "name":
            name,

        "bbox":
            bbox,

        "horizontal_overlap":
            h_overlap,

        "vertical_gap":
            v_gap,

        "area":
            area(bbox),

        "accepted_for_completion":
            connected,
    })


p01_accepted = [
    row
    for row in p01_candidate_evidence
    if row["accepted_for_completion"]
]


p01_boxes = [
    P01_BASE
]

p01_boxes.extend(
    row["bbox"]
    for row in p01_accepted
)


P01_FINAL = union_boxes(
    p01_boxes
)


# Minimal final safety expansion.
P01_FINAL = expand_box(
    P01_FINAL,
    x_ratio=0.015,
    y_ratio=0.025
)


# ============================================================
# P02 TOILET — FREEZE CORRECT D2B #1
# ============================================================

P02_BASE = get_d2b_rank(
    "P02",
    1
)

P02_FINAL = expand_box(
    P02_BASE,
    x_ratio=0.025,
    y_ratio=0.035
)


# ============================================================
# P03 SHOWER — FREEZE CORRECT TIGHT D2B #1
# ============================================================

P03_BASE = get_d2b_rank(
    "P03",
    1
)

P03_FINAL = expand_box(
    P03_BASE,
    x_ratio=0.06,
    y_ratio=0.10
)


# ============================================================
# P04 ELECTRICAL PLATE
# EXACT VERIFIED 06C13A CANDIDATE #1
# ============================================================

p04_row = None


for row in p04_verified:

    if (
        int(
            row.get(
                "candidate_index",
                -1
            )
        )
        ==
        1

        and

        bool(
            row.get(
                "geometry_pass",
                False
            )
        )
    ):

        p04_row = row
        break


if p04_row is None:

    raise RuntimeError(
        "Verified P04 electrical plate candidate not found."
    )


P04_BASE = clamp_box(
    p04_row[
        "florence_bbox"
    ]
)


P04_FINAL = expand_box(
    P04_BASE,
    x_ratio=0.08,
    y_ratio=0.08
)


# ============================================================
# P05 TOILET PAPER HOLDER
# EXACT VERIFIED 06C14B INVENTORY 12
# ============================================================

p05_row = None

p06_row = None


for row in p05p06_verified[
    "resolved_candidates"
]:

    iid = int(
        row.get(
            "inventory_id",
            -1
        )
    )

    if iid == 12:
        p05_row = row

    elif iid == 2:
        p06_row = row


if p05_row is None:

    raise RuntimeError(
        "Verified P05 toilet-paper-holder record not found."
    )


P05_BASE = clamp_box(
    p05_row[
        "florence_bbox"
    ]
)


P05_FINAL = expand_box(
    P05_BASE,
    x_ratio=0.10,
    y_ratio=0.10
)


# ============================================================
# P06 CEILING LIGHT
# EXACT VERIFIED 06C14B INVENTORY 2
# ============================================================

if p06_row is None:

    raise RuntimeError(
        "Verified P06 ceiling-light record not found."
    )


P06_BASE = clamp_box(
    p06_row[
        "florence_bbox"
    ]
)


P06_FINAL = expand_box(
    P06_BASE,
    x_ratio=0.10,
    y_ratio=0.10
)


# ============================================================
# BUILD FINAL STATE
# ============================================================

results = [

    {
        "main_prop_id":
            "P01",

        "main_prop_name":
            inventory_by_id[
                "P01"
            ][
                "main_prop_name"
            ],

        "source":
            "D2B_RANK1_PLUS_SPATIALLY_CONNECTED_MEMBER_COMPLETION",

        "base_bbox":
            P01_BASE,

        "candidate_member_evidence":
            p01_candidate_evidence,

        "accepted_member_evidence_ids":
            [
                row["evidence_id"]
                for row in p01_accepted
            ],

        "final_bbox":
            P01_FINAL,

        "status":
            "FINAL_LOCALIZATION_READY",
    },


    {
        "main_prop_id":
            "P02",

        "main_prop_name":
            inventory_by_id[
                "P02"
            ][
                "main_prop_name"
            ],

        "source":
            "FROZEN_D2B_RANK1",

        "base_bbox":
            P02_BASE,

        "final_bbox":
            P02_FINAL,

        "status":
            "FINAL_LOCALIZATION_READY",
    },


    {
        "main_prop_id":
            "P03",

        "main_prop_name":
            inventory_by_id[
                "P03"
            ][
                "main_prop_name"
            ],

        "source":
            "FROZEN_D2B_RANK1",

        "base_bbox":
            P03_BASE,

        "final_bbox":
            P03_FINAL,

        "status":
            "FINAL_LOCALIZATION_READY",
    },


    {
        "main_prop_id":
            "P04",

        "main_prop_name":
            inventory_by_id[
                "P04"
            ][
                "main_prop_name"
            ],

        "source":
            "VERIFIED_06C13A_CANDIDATE_01",

        "base_bbox":
            P04_BASE,

        "verification": {

            "sam2_score":
                p04_row[
                    "sam2_score"
                ],

            "quality":
                p04_row[
                    "quality"
                ],

            "geometry_pass":
                p04_row[
                    "geometry_pass"
                ],
        },

        "final_bbox":
            P04_FINAL,

        "status":
            "FINAL_LOCALIZATION_READY",
    },


    {
        "main_prop_id":
            "P05",

        "main_prop_name":
            inventory_by_id[
                "P05"
            ][
                "main_prop_name"
            ],

        "source":
            "VERIFIED_06C14B_INVENTORY_12",

        "base_bbox":
            P05_BASE,

        "verification": {

            "sam2_score":
                p05_row[
                    "sam2_score"
                ],

            "quality":
                p05_row[
                    "quality"
                ],

            "geometry_pass":
                p05_row[
                    "geometry_pass"
                ],
        },

        "final_bbox":
            P05_FINAL,

        "status":
            "FINAL_LOCALIZATION_READY",
    },


    {
        "main_prop_id":
            "P06",

        "main_prop_name":
            inventory_by_id[
                "P06"
            ][
                "main_prop_name"
            ],

        "source":
            "VERIFIED_06C14B_INVENTORY_02",

        "base_bbox":
            P06_BASE,

        "verification": {

            "sam2_score":
                p06_row[
                    "sam2_score"
                ],

            "quality":
                p06_row[
                    "quality"
                ],

            "geometry_pass":
                p06_row[
                    "geometry_pass"
                ],
        },

        "final_bbox":
            P06_FINAL,

        "status":
            "FINAL_LOCALIZATION_READY",
    },
]


# ============================================================
# PREVIEWS
# ============================================================

for row in results:

    preview = master.copy()

    draw = ImageDraw.Draw(
        preview
    )

    bbox = row[
        "final_bbox"
    ]


    draw.rectangle(
        bbox,
        outline="lime",
        width=4
    )


    x1, y1, _, _ = bbox


    draw.rectangle(
        [
            x1,
            y1,
            x1 + 60,
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
        row[
            "main_prop_id"
        ],
        fill="green"
    )


    preview_path = (
        PREVIEW_DIR
        /
        f'{row["main_prop_id"]}_verified_localization.png'
    )


    preview.save(
        preview_path
    )


    row[
        "preview_path"
    ] = str(
        preview_path
    )


# ============================================================
# OVERVIEW
# ============================================================

overview = master.copy()

draw = ImageDraw.Draw(
    overview
)


for row in results:

    bbox = row[
        "final_bbox"
    ]


    draw.rectangle(
        bbox,
        outline="lime",
        width=3
    )


    draw.text(
        (
            bbox[0] + 3,
            bbox[1] + 3
        ),
        row[
            "main_prop_id"
        ],
        fill="green"
    )


OVERVIEW_PATH = (
    OUT
    / "01_final_six_main_prop_localizations.png"
)


overview.save(
    OVERVIEW_PATH
)


# ============================================================
# SAVE
# ============================================================

FINAL_STATE = {

    "stage":
        "06D2B3",

    "status":
        "REQUIRES_FINAL_LOCALIZATION_VISUAL_AUDIT",

    "main_prop_count":
        6,

    "all_localizations_ready":
        True,

    "results":
        results,

    "overview_preview":
        str(
            OVERVIEW_PATH
        ),

    "production_rules": [

        "exactly six main-prop targets",

        "no child component is an independent extraction target",

        "P01 completion evidence must be spatially connected to the whole-vanity base",

        "P02 and P03 use visually verified D2B whole-prop localizations",

        "P04 reuses verified 06C13A localization",

        "P05 and P06 reuse verified 06C14B localizations",

        "no new detection performed",

        "no SAM2 segmentation performed in this stage",

        "Stage05F is localization input",

        "Stage01 remains final exact RGB source",
    ],

    "next_stage_if_visual_pass":
        "06D2C_COMPLETE_MAIN_PROP_SEGMENTATION",
}


RESULT_PATH = (
    OUT
    / "00_stage06d2b3_result.json"
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
print("PRODUCTION STAGE 06D2B3 RESULT")
print("=" * 110)

print()


for row in results:

    print(
        "{}. {:28s} | bbox={} | source={}".format(

            row[
                "main_prop_id"
            ],

            row[
                "main_prop_name"
            ],

            [
                round(v, 1)
                for v in row[
                    "final_bbox"
                ]
            ],

            row[
                "source"
            ],
        )
    )


print()
print(
    "P01 ACCEPTED COMPLETION EVIDENCE:",
    [
        row["evidence_id"]
        for row in p01_accepted
    ]
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
    "NO NEW DETECTION WAS RUN."
)

print(
    "NO CHILD-PROP EXTRACTION WAS RUN."
)

print(
    "NO SAM2 OR FINAL MASK UNION WAS RUN."
)


# ============================================================
# INLINE AUDIT
# ============================================================

print()
print("=" * 110)
print("INLINE 06D2B3 FINAL LOCALIZATION AUDIT")
print("=" * 110)


show(
    OVERVIEW_PATH,
    (
        "06D2B3 — FINAL SIX MAIN-PROP LOCALIZATIONS\n"
        "GREEN = Final Localization"
    ),
    figsize=(8, 9)
)


for row in results:

    show(
        row[
            "preview_path"
        ],
        (
            f'{row["main_prop_id"]} — '
            f'{row["main_prop_name"]}\n'
            'GREEN = Final Main-Prop Localization'
        ),
        figsize=(7, 8)
    )
