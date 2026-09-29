
from pathlib import Path
import json
import sys

import cv2
import numpy as np
import torch

from PIL import (
    Image,
    ImageDraw
)

import matplotlib.pyplot as plt


# ============================================================
# SAM2 IMPORT
# ============================================================

for source in [
    Path("/workspace/sam2_src"),
    Path("/workspace/axolotl/sam2"),
]:

    if (
        source.exists()
        and
        str(source) not in sys.path
    ):
        sys.path.insert(
            0,
            str(source)
        )


from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


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

MASTER = (
    PROD
    / "stage01_master"
    / "00_master_input.png"
)

VANITY_DISCOVERY = (
    STAGE06
    / "06c15a_vanity_hierarchy_discovery"
    / "00_stage06c15a_result.json"
)

HANDLE_MASK = (
    STAGE06
    / "06c11c_florence_handle_sam2"
    / "objects"
    / "021_drawer_handle_florence_sam2_mask.png"
)

FAUCET_MASK = (
    STAGE06
    / "06c10b_florence_faucet_sam2_duplicate_state"
    / "objects"
    / "010_faucet_florence_sam2_mask.png"
)

OUT = (
    STAGE06
    / "06c15c_complete_vanity_sam2"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

OBJECTS = (
    OUT
    / "objects"
)

OBJECTS.mkdir(
    parents=True,
    exist_ok=True
)

CHECKPOINT = Path(
    "/workspace/axolotl/test07/models/sam2/"
    "sam2.1_hiera_large.pt"
)

SAM2_CONFIG = (
    "configs/sam2.1/"
    "sam2.1_hiera_l.yaml"
)


# ============================================================
# CONFIG
# ============================================================

TARGET_ID = 23
TARGET_NAME = "complete vanity system"

PROMPT_EXPANSION = 0.04
GATE_EXPANSION = 0.08

MIN_SAM_SCORE = 0.50
MIN_CONTAINMENT = 0.90
MAX_BORDER_TOUCH = 0.25
MIN_MASK_PIXELS = 500


# ============================================================
# HELPERS
# ============================================================

def clamp_box(
    box,
    W,
    H
):

    x1, y1, x2, y2 = [
        float(v)
        for v in box
    ]

    x1 = max(
        0.0,
        min(float(W - 1), x1)
    )

    y1 = max(
        0.0,
        min(float(H - 1), y1)
    )

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


def expand_box(
    box,
    ratio,
    W,
    H
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
            x1 - bw * ratio,
            y1 - bh * ratio,
            x2 + bw * ratio,
            y2 + bh * ratio,
        ],
        W,
        H
    )


def make_gate(
    box,
    H,
    W
):

    x1, y1, x2, y2 = [
        int(round(v))
        for v in box
    ]

    x1 = max(
        0,
        min(W - 1, x1)
    )

    y1 = max(
        0,
        min(H - 1, y1)
    )

    x2 = max(
        x1 + 1,
        min(W, x2)
    )

    y2 = max(
        y1 + 1,
        min(H, y2)
    )

    gate = np.zeros(
        (H, W),
        dtype=bool
    )

    gate[
        y1:y2,
        x1:x2
    ] = True

    return gate


def largest_component(
    mask
):

    n, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask.astype(np.uint8),
            8
        )
    )

    if n <= 1:
        return mask

    idx = (
        1
        +
        int(
            np.argmax(
                stats[
                    1:,
                    cv2.CC_STAT_AREA
                ]
            )
        )
    )

    return labels == idx


def border_touch_ratio(
    mask,
    box
):

    H, W = mask.shape

    x1, y1, x2, y2 = [
        int(round(v))
        for v in box
    ]

    x1 = max(
        0,
        min(W - 1, x1)
    )

    y1 = max(
        0,
        min(H - 1, y1)
    )

    x2 = max(
        x1 + 1,
        min(W, x2)
    )

    y2 = max(
        y1 + 1,
        min(H, y2)
    )

    border = np.zeros_like(
        mask,
        dtype=bool
    )

    t = 2

    border[
        y1:min(y2, y1 + t),
        x1:x2
    ] = True

    border[
        max(y1, y2 - t):y2,
        x1:x2
    ] = True

    border[
        y1:y2,
        x1:min(x2, x1 + t)
    ] = True

    border[
        y1:y2,
        max(x1, x2 - t):x2
    ] = True

    return float(
        (
            mask
            &
            border
        ).sum()
        /
        max(
            1,
            int(mask.sum())
        )
    )


def mask_overlap_fraction(
    child_mask,
    parent_mask
):

    child_area = int(
        child_mask.sum()
    )

    if child_area == 0:
        return 0.0

    intersection = int(
        (
            child_mask
            &
            parent_mask
        ).sum()
    )

    return float(
        intersection
        /
        child_area
    )


def show_image(
    path,
    title,
    figsize=(8, 9)
):

    image = Image.open(
        path
    )

    plt.figure(
        figsize=figsize
    )

    plt.imshow(
        image
    )

    plt.title(
        title
    )

    plt.axis(
        "off"
    )

    plt.show()


# ============================================================
# VALIDATE
# ============================================================

for path in [
    MASTER,
    VANITY_DISCOVERY,
    CHECKPOINT,
]:

    if not path.exists():

        raise FileNotFoundError(
            path
        )


# ============================================================
# LOAD MASTER
# ============================================================

master_pil = Image.open(
    MASTER
).convert(
    "RGB"
)

master = np.asarray(
    master_pil
)

W, H = master_pil.size


# ============================================================
# LOAD VANITY BBOX
# ============================================================

discovery = json.loads(
    VANITY_DISCOVERY.read_text(
        encoding="utf-8"
    )
)

vanity_box = clamp_box(
    discovery[
        "selected_parent_bbox"
    ],
    W,
    H
)


prompt_box = expand_box(
    vanity_box,
    PROMPT_EXPANSION,
    W,
    H
)

gate_box = expand_box(
    vanity_box,
    GATE_EXPANSION,
    W,
    H
)

gate = make_gate(
    gate_box,
    H,
    W
)


print("=" * 110)
print("PRODUCTION STAGE 06C15C")
print("COMPLETE VANITY SYSTEM → SAM2")
print("=" * 110)

print()
print(
    "VANITY BBOX:",
    [
        round(v, 1)
        for v in vanity_box
    ]
)

print(
    "PROMPT BBOX:",
    [
        round(v, 1)
        for v in prompt_box
    ]
)

print(
    "GATE BBOX:",
    [
        round(v, 1)
        for v in gate_box
    ]
)


# ============================================================
# LOAD SAM2
# ============================================================

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print()
print(
    "Loading SAM2..."
)


sam2_model = build_sam2(
    SAM2_CONFIG,
    str(
        CHECKPOINT
    ),
    device=device
)


predictor = SAM2ImagePredictor(
    sam2_model
)


predictor.set_image(
    master
)


print(
    "✅ SAM2 READY"
)


# ============================================================
# MULTIMASK
# ============================================================

with torch.inference_mode():

    masks, scores, _ = (
        predictor.predict(
            box=np.asarray(
                prompt_box,
                dtype=np.float32
            ),
            multimask_output=True
        )
    )


evaluated = []


for midx in range(
    len(masks)
):

    raw = (
        masks[
            midx
        ]
        >
        0
    )

    if not raw.any():
        continue

    raw = largest_component(
        raw
    )

    raw_area = int(
        raw.sum()
    )

    local = (
        raw
        &
        gate
    )

    area = int(
        local.sum()
    )

    if area == 0:
        continue

    containment = float(
        area
        /
        max(
            1,
            raw_area
        )
    )

    sam_score = float(
        scores[
            midx
        ]
    )

    border_touch = border_touch_ratio(
        local,
        prompt_box
    )

    quality = (

        0.70
        *
        sam_score

        +

        0.20
        *
        containment

        +

        0.10
        *
        (
            1.0
            -
            min(
                1.0,
                border_touch
                /
                MAX_BORDER_TOUCH
            )
        )
    )

    evaluated.append({

        "mask_index":
            int(midx),

        "mask":
            local,

        "sam2_score":
            sam_score,

        "quality":
            quality,

        "mask_area":
            area,

        "containment":
            containment,

        "border_touch":
            border_touch,
    })


if not evaluated:

    raise RuntimeError(
        "SAM2 produced no vanity mask."
    )


evaluated.sort(
    key=lambda x:
        x[
            "quality"
        ],
    reverse=True
)


# ============================================================
# SAVE ALL MULTIMASK OPTIONS FOR VISUAL AUDIT
# ============================================================

candidate_previews = []


for rank, item in enumerate(
    evaluated,
    start=1
):

    mask = item[
        "mask"
    ]


    candidate_mask_path = (
        OBJECTS
        /
        f"candidate_{rank:02d}_mask.png"
    )


    Image.fromarray(
        mask.astype(
            np.uint8
        )
        *
        255
    ).save(
        candidate_mask_path
    )


    preview = master_pil.copy()

    draw = ImageDraw.Draw(
        preview
    )


    draw.rectangle(
        vanity_box,
        outline="blue",
        width=2
    )


    draw.rectangle(
        prompt_box,
        outline="red",
        width=2
    )


    contours, _ = cv2.findContours(
        (
            mask.astype(
                np.uint8
            )
            *
            255
        ),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    for contour in contours:

        pts = [

            (
                int(p[0][0]),
                int(p[0][1])
            )

            for p in contour
        ]

        if len(pts) >= 2:

            draw.line(
                pts
                +
                [
                    pts[0]
                ],
                fill="lime",
                width=2
            )


    preview_path = (
        OBJECTS
        /
        f"candidate_{rank:02d}_preview.png"
    )


    preview.save(
        preview_path
    )


    candidate_previews.append({

        "rank":
            rank,

        "mask_path":
            str(
                candidate_mask_path
            ),

        "preview_path":
            str(
                preview_path
            ),

        "metrics":
            {
                k:
                    v
                for k, v
                in item.items()
                if k != "mask"
            },
    })


# ============================================================
# BEST NUMERICAL MASK
# ============================================================

best = evaluated[
    0
]

best_mask = best[
    "mask"
]


geometry_pass = bool(

    best[
        "mask_area"
    ]
    >=
    MIN_MASK_PIXELS

    and

    best[
        "sam2_score"
    ]
    >=
    MIN_SAM_SCORE

    and

    best[
        "containment"
    ]
    >=
    MIN_CONTAINMENT

    and

    best[
        "border_touch"
    ]
    <=
    MAX_BORDER_TOUCH
)


# ============================================================
# CHILD-EVIDENCE COVERAGE
#
# A complete vanity mask should contain already resolved
# attached child evidence such as faucet and handle.
# ============================================================

child_coverage = {}


for label, path in [

    (
        "010_faucet",
        FAUCET_MASK
    ),

    (
        "021_handle",
        HANDLE_MASK
    ),
]:

    if path.exists():

        child = (
            np.asarray(
                Image.open(
                    path
                ).convert(
                    "L"
                )
            )
            >
            0
        )


        child_coverage[
            label
        ] = mask_overlap_fraction(
            child,
            best_mask
        )


# ============================================================
# SAVE BEST
# ============================================================

MASK_PATH = (
    OBJECTS
    / "023_complete_vanity_mask.png"
)


Image.fromarray(
    best_mask.astype(
        np.uint8
    )
    *
    255
).save(
    MASK_PATH
)


rgba = np.zeros(
    (
        H,
        W,
        4
    ),
    dtype=np.uint8
)


rgba[
    :,
    :,
    :3
] = master


rgba[
    :,
    :,
    3
] = (
    best_mask.astype(
        np.uint8
    )
    *
    255
)


RGBA_PATH = (
    OBJECTS
    / "023_complete_vanity_rgba.png"
)


Image.fromarray(
    rgba,
    mode="RGBA"
).save(
    RGBA_PATH
)


# ============================================================
# RESULT
# ============================================================

result = {

    "stage":
        "06C15C",

    "inventory_id":
        TARGET_ID,

    "inventory_name":
        TARGET_NAME,

    "source_stage":
        "06C15A",

    "vanity_bbox":
        vanity_box,

    "prompt_bbox":
        prompt_box,

    "gate_bbox":
        gate_box,

    "best_sam2_mask_index":
        best[
            "mask_index"
        ],

    "sam2_score":
        best[
            "sam2_score"
        ],

    "quality":
        best[
            "quality"
        ],

    "mask_area":
        best[
            "mask_area"
        ],

    "containment":
        best[
            "containment"
        ],

    "border_touch":
        best[
            "border_touch"
        ],

    "geometry_pass":
        geometry_pass,

    "child_component_coverage":
        child_coverage,

    "mask_path":
        str(
            MASK_PATH
        ),

    "rgba_path":
        str(
            RGBA_PATH
        ),

    "candidate_previews":
        candidate_previews,

    "status":
        "REQUIRES_VISUAL_MAIN_PROP_AUDIT",

    "rules": [

        (
            "vanity must be evaluated as one complete main "
            "physical prop"
        ),

        (
            "child component masks are evidence only"
        ),

        (
            "numerical SAM2 ranking is not semantic proof"
        ),

        (
            "no final prop union performed"
        ),
    ],
}


RESULT_PATH = (
    OUT
    / "00_stage06c15c_result.json"
)


RESULT_PATH.write_text(
    json.dumps(
        result,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# ============================================================
# PRINT
# ============================================================

print()
print("=" * 110)
print("PRODUCTION STAGE 06C15C RESULT")
print("=" * 110)

print(
    "SAM2:",
    round(
        best[
            "sam2_score"
        ],
        4
    )
)

print(
    "QUALITY:",
    round(
        best[
            "quality"
        ],
        4
    )
)

print(
    "AREA:",
    best[
        "mask_area"
    ]
)

print(
    "CONTAINMENT:",
    round(
        best[
            "containment"
        ],
        4
    )
)

print(
    "BORDER TOUCH:",
    round(
        best[
            "border_touch"
        ],
        4
    )
)

print(
    "GEOMETRY PASS:",
    geometry_pass
)


print()
print(
    "CHILD COMPONENT COVERAGE:"
)

for name, value in child_coverage.items():

    print(
        " ",
        name,
        "=",
        round(
            value,
            4
        )
    )


print()
print(
    "RESULT JSON:",
    RESULT_PATH
)

print(
    "NO FINAL PROP MASK UNION WAS CREATED."
)


# ============================================================
# INLINE VISUAL VERIFICATION
# ============================================================

print()
print("=" * 110)
print("INLINE VANITY MULTIMASK AUDIT")
print("=" * 110)


for item in candidate_previews:

    metrics = item[
        "metrics"
    ]

    show_image(

        item[
            "preview_path"
        ],

        (
            f'VANITY MASK CANDIDATE #{item["rank"]} '
            f'| SAM={metrics["sam2_score"]:.4f} '
            f'| area={metrics["mask_area"]}'
        ),

        figsize=(8, 9)
    )


# ============================================================
# CLEANUP
# ============================================================

del predictor
del sam2_model

if torch.cuda.is_available():

    torch.cuda.empty_cache()
