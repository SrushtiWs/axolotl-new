
from pathlib import Path
import json
import gc

import cv2
import numpy as np
import torch

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

LOCALIZATION_PATH = (
    STAGE06
    / "06d2b3_final_verified_main_prop_localization"
    / "00_stage06d2b3_result.json"
)

C1_PATH = (
    STAGE06
    / "06d2c1_complete_main_prop_sam2_multimask_audit"
    / "00_stage06d2c1_result.json"
)

OUT = (
    STAGE06
    / "06d2c2_vanity_multipoint_refinement"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

OBJECT_DIR = (
    OUT
    / "objects"
)

OBJECT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SAM2
# ============================================================

SAM2_CHECKPOINT = (
    BASE
    / "test07"
    / "models"
    / "sam2"
    / "sam2.1_hiera_large.pt"
)

SAM2_CONFIG = (
    "configs/sam2.1/sam2.1_hiera_l.yaml"
)


# ============================================================
# VISUALLY VERIFIED C1 SELECTIONS
# ============================================================

FROZEN_SELECTIONS = {

    "P02": 3,  # complete toilet

    "P03": 2,  # shower arm + head

    "P04": 1,  # electrical plate

    "P05": 2,  # toilet paper holder

    "P06": 3,  # ceiling light
}


# ============================================================
# VALIDATE
# ============================================================

for path in [
    MASTER_PATH,
    LOCALIZATION_PATH,
    C1_PATH,
    SAM2_CHECKPOINT,
]:

    if not path.exists():
        raise FileNotFoundError(path)


master_pil = Image.open(
    MASTER_PATH
).convert("RGB")

master_np = np.asarray(
    master_pil
)

H, W = master_np.shape[:2]


localization = json.loads(
    LOCALIZATION_PATH.read_text(
        encoding="utf-8"
    )
)

c1 = json.loads(
    C1_PATH.read_text(
        encoding="utf-8"
    )
)


localization_by_id = {
    row["main_prop_id"]: row
    for row in localization["results"]
}

c1_by_id = {
    row["main_prop_id"]: row
    for row in c1["results"]
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


def load_mask(path):

    return (
        np.asarray(
            Image.open(path).convert("L")
        )
        >
        0
    )


def save_mask(mask, path):

    Image.fromarray(
        mask.astype(np.uint8) * 255
    ).save(path)


def mask_bbox(mask):

    ys, xs = np.where(mask)

    if len(xs) == 0:
        return None

    return [
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    ]


def draw_outline(
    image,
    mask,
    color="lime",
    width=3
):

    result = image.copy()

    contours, _ = cv2.findContours(
        mask.astype(np.uint8) * 255,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    draw = ImageDraw.Draw(result)

    for contour in contours:

        points = [
            (
                int(p[0][0]),
                int(p[0][1])
            )
            for p in contour
        ]

        if len(points) >= 2:

            draw.line(
                points + [points[0]],
                fill=color,
                width=width
            )

    return result


def point_inside_image(x, y):

    return (
        0 <= x < W
        and
        0 <= y < H
    )


# ============================================================
# BUILD P01 POSITIVE POINTS
#
# IMPORTANT:
#
# We use connected-member localization evidence from the
# already frozen 06D2B3 state.
#
# These are prompt points ONLY.
# They do NOT become independent extraction targets.
# ============================================================

p01 = localization_by_id["P01"]

p01_bbox = [
    float(v)
    for v in p01["final_bbox"]
]


member_rows = p01.get(
    "candidate_member_evidence",
    []
)


# ------------------------------------------------------------
# Prefer one point from each meaningful connected-member type.
#
# We intentionally ignore duplicate repeated observations.
# ------------------------------------------------------------

PREFERRED_NAMES = [
    "cabinet",
    "countertop",
    "sink",
    "sink faucet",
    "faucet",
    "bottle",
    "drawer handle",
    "door handle",
]


selected_member_points = []

seen_names = set()


for target_name in PREFERRED_NAMES:

    candidates = [
        row
        for row in member_rows
        if (
            row.get(
                "accepted_for_completion",
                False
            )
            and
            str(
                row.get(
                    "name",
                    ""
                )
            ).lower()
            ==
            target_name
        )
    ]

    if not candidates:
        continue


    # --------------------------------------------------------
    # Prefer smaller localization for a component because
    # gigantic evidence boxes produce poor point locations.
    # --------------------------------------------------------

    candidates.sort(
        key=lambda row:
            float(
                row.get(
                    "area",
                    1e18
                )
            )
    )


    row = candidates[0]

    x1, y1, x2, y2 = row[
        "bbox"
    ]


    cx = (
        float(x1)
        +
        float(x2)
    ) / 2.0

    cy = (
        float(y1)
        +
        float(y2)
    ) / 2.0


    if point_inside_image(
        cx,
        cy
    ):

        selected_member_points.append({

            "name":
                target_name,

            "evidence_id":
                row[
                    "evidence_id"
                ],

            "point":
                [
                    cx,
                    cy
                ],

            "source_bbox":
                row[
                    "bbox"
                ],
        })


# ============================================================
# ADD ROBUST SYSTEM-LEVEL POINTS
#
# These help ensure the large cabinet body is represented even
# if some small component evidence is noisy.
# ============================================================

x1, y1, x2, y2 = p01_bbox

bw = x2 - x1
bh = y2 - y1


system_points = [

    {
        "name":
            "vanity_upper_body",

        "point":
            [
                x1 + 0.50 * bw,
                y1 + 0.42 * bh
            ],
    },

    {
        "name":
            "vanity_lower_body",

        "point":
            [
                x1 + 0.50 * bw,
                y1 + 0.72 * bh
            ],
    },
]


all_points = (
    selected_member_points
    +
    system_points
)


point_coords = np.asarray(
    [
        row[
            "point"
        ]
        for row in all_points
    ],
    dtype=np.float32
)


point_labels = np.ones(
    len(
        point_coords
    ),
    dtype=np.int32
)


# ============================================================
# PRINT POINT AUDIT
# ============================================================

print("=" * 110)
print("PRODUCTION STAGE 06D2C2")
print("P01 COMPLETE VANITY MULTI-POINT REFINEMENT")
print("=" * 110)

print()
print(
    "P01 BBOX:",
    [
        round(v, 1)
        for v in p01_bbox
    ]
)

print()
print(
    "POSITIVE POINTS:",
    len(
        all_points
    )
)


for index, row in enumerate(
    all_points,
    start=1
):

    print(
        "#{:02d} {:20s} point={}".format(

            index,

            row[
                "name"
            ],

            [
                round(v, 1)
                for v in row[
                    "point"
                ]
            ],
        )
    )


# ============================================================
# POINT PREVIEW
# ============================================================

point_preview = master_pil.copy()

draw = ImageDraw.Draw(
    point_preview
)


draw.rectangle(
    p01_bbox,
    outline="yellow",
    width=3
)


for index, row in enumerate(
    all_points,
    start=1
):

    px, py = row[
        "point"
    ]


    r = 4


    draw.ellipse(
        [
            px - r,
            py - r,
            px + r,
            py + r
        ],
        fill="lime",
        outline="black"
    )


    draw.text(
        (
            px + 5,
            py - 5
        ),
        str(index),
        fill="red"
    )


POINT_PREVIEW_PATH = (
    OUT
    / "01_p01_positive_point_preview.png"
)


point_preview.save(
    POINT_PREVIEW_PATH
)


show(
    POINT_PREVIEW_PATH,
    (
        "06D2C2 — P01 VANITY POSITIVE PROMPT POINTS\n"
        "GREEN = positive point | YELLOW = whole vanity bbox"
    )
)


# ============================================================
# LOAD SAM2
# ============================================================

print()
print(
    "Loading SAM2..."
)


from sam2.build_sam import (
    build_sam2,
)

from sam2.sam2_image_predictor import (
    SAM2ImagePredictor,
)


sam2_model = build_sam2(
    SAM2_CONFIG,
    str(
        SAM2_CHECKPOINT
    ),
    device=(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    ),
)


predictor = SAM2ImagePredictor(
    sam2_model
)


predictor.set_image(
    master_np
)


print(
    "✅ SAM2 READY"
)


# ============================================================
# P01 — ONE COMPLETE-SYSTEM SAM2 CALL
# ============================================================

masks, scores, logits = predictor.predict(

    point_coords=point_coords,

    point_labels=point_labels,

    box=np.asarray(
        p01_bbox,
        dtype=np.float32
    ),

    multimask_output=True,
)


p01_candidates = []


for idx in range(
    len(
        masks
    )
):

    mask = (
        masks[idx]
        >
        0
    )


    area = int(
        mask.sum()
    )


    score = float(
        scores[idx]
    )


    bbox = mask_bbox(
        mask
    )


    mask_path = (
        OBJECT_DIR
        /
        f"P01_refined_candidate_{idx + 1}_mask.png"
    )


    save_mask(
        mask,
        mask_path
    )


    preview = draw_outline(
        master_pil,
        mask,
        "lime",
        3
    )


    draw = ImageDraw.Draw(
        preview
    )


    draw.rectangle(
        p01_bbox,
        outline="yellow",
        width=2
    )


    for point_index, row in enumerate(
        all_points,
        start=1
    ):

        px, py = row[
            "point"
        ]


        r = 3


        draw.ellipse(
            [
                px - r,
                py - r,
                px + r,
                py + r
            ],
            fill="red"
        )


    preview_path = (
        OBJECT_DIR
        /
        f"P01_refined_candidate_{idx + 1}_preview.png"
    )


    preview.save(
        preview_path
    )


    p01_candidates.append({

        "candidate_index":
            idx + 1,

        "sam_score":
            score,

        "area":
            area,

        "mask_bbox":
            bbox,

        "mask_path":
            str(
                mask_path
            ),

        "preview_path":
            str(
                preview_path
            ),
    })


    print(
        "P01 candidate #{} | SAM={:.4f} | area={} | bbox={}".format(

            idx + 1,

            score,

            area,

            bbox,
        )
    )


# ============================================================
# FREEZE P02-P06 FROM C1
# ============================================================

frozen = []


for prop_id, candidate_index in FROZEN_SELECTIONS.items():

    row = c1_by_id[
        prop_id
    ]


    candidate = next(

        c

        for c in row[
            "candidates"
        ]

        if int(
            c[
                "candidate_index"
            ]
        )
        ==
        candidate_index
    )


    frozen.append({

        "main_prop_id":
            prop_id,

        "main_prop_name":
            row[
                "main_prop_name"
            ],

        "selected_candidate_index":
            candidate_index,

        "sam_score":
            candidate[
                "sam_score"
            ],

        "area":
            candidate[
                "area"
            ],

        "mask_path":
            candidate[
                "mask_path"
            ],

        "preview_path":
            candidate[
                "preview_path"
            ],

        "status":
            "VISUALLY_FROZEN",
    })


# ============================================================
# SAVE STATE
# ============================================================

FINAL_STATE = {

    "stage":
        "06D2C2",

    "status":
        "P01_REQUIRES_REFINED_CANDIDATE_VISUAL_SELECTION",

    "p01": {

        "main_prop_id":
            "P01",

        "main_prop_name":
            "complete vanity system",

        "whole_system_bbox":
            p01_bbox,

        "positive_prompt_points":
            all_points,

        "candidate_count":
            len(
                p01_candidates
            ),

        "candidates":
            p01_candidates,

        "status":
            "REQUIRES_VISUAL_SELECTION",
    },

    "frozen_other_main_props":
        frozen,

    "rules": [

        "child component locations are prompt evidence only",

        "no child component mask or layer is created",

        "P01 remains exactly one final prop",

        "P02-P06 remain exactly five final props",

        "no final prop union created",

        "Stage05F used for segmentation",

        "Stage01 remains final RGB source",
    ],
}


RESULT_PATH = (
    OUT
    / "00_stage06d2c2_result.json"
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
# PRINT FROZEN STATE
# ============================================================

print()
print("=" * 110)
print("FROZEN P02-P06")
print("=" * 110)


for row in frozen:

    print(
        "{}. {:25s} → CANDIDATE #{} | SAM={:.4f} | area={}".format(

            row[
                "main_prop_id"
            ],

            row[
                "main_prop_name"
            ],

            row[
                "selected_candidate_index"
            ],

            row[
                "sam_score"
            ],

            row[
                "area"
            ],
        )
    )


print()
print(
    "RESULT JSON:",
    RESULT_PATH
)

print()
print(
    "NO FINAL PROP MASK UNION WAS CREATED."
)


# ============================================================
# INLINE P01 REFINED CANDIDATES
# ============================================================

print()
print("=" * 110)
print("INLINE P01 REFINED COMPLETE-SYSTEM AUDIT")
print("=" * 110)


for row in p01_candidates:

    show(
        row[
            "preview_path"
        ],
        (
            "P01 — COMPLETE VANITY SYSTEM\n"
            f'REFINED CANDIDATE #{row["candidate_index"]} | '
            f'SAM={row["sam_score"]:.3f} | '
            f'AREA={row["area"]}'
        ),
        figsize=(7, 8)
    )


# ============================================================
# CLEANUP
# ============================================================

del predictor
del sam2_model

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()
