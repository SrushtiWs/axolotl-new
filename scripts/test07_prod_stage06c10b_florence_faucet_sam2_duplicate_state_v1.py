
from pathlib import Path
import json
import sys

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw


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

FLORENCE_RESULTS = (
    STAGE06
    / "06c10a_florence_phrase_grounding"
    / "00_florence_phrase_grounding_results.json"
)

OUT = (
    STAGE06
    / "06c10b_florence_faucet_sam2_duplicate_state"
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

SAM2_CHECKPOINT = Path(
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

PRIMARY_ID = 10
DUPLICATE_ID = 25
UNRESOLVED_ID = 21

BOX_EXPANSION = 0.12

MIN_SAM_SCORE = 0.55
MIN_CONTAINMENT = 0.85
MAX_BORDER_TOUCH = 0.30
MIN_MASK_PIXELS = 20


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
        min(
            float(W - 1),
            x1
        )
    )

    y1 = max(
        0.0,
        min(
            float(H - 1),
            y1
        )
    )

    x2 = max(
        x1 + 1.0,
        min(
            float(W),
            x2
        )
    )

    y2 = max(
        y1 + 1.0,
        min(
            float(H),
            y2
        )
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


def box_mask(
    box,
    H,
    W
):

    x1, y1, x2, y2 = [
        int(
            round(v)
        )
        for v in box
    ]

    x1 = max(
        0,
        min(
            W - 1,
            x1
        )
    )

    y1 = max(
        0,
        min(
            H - 1,
            y1
        )
    )

    x2 = max(
        x1 + 1,
        min(
            W,
            x2
        )
    )

    y2 = max(
        y1 + 1,
        min(
            H,
            y2
        )
    )

    mask = np.zeros(
        (
            H,
            W
        ),
        dtype=bool
    )

    mask[
        y1:y2,
        x1:x2
    ] = True

    return mask


def border_touch_ratio(
    mask,
    gate_box
):

    x1, y1, x2, y2 = [
        int(
            round(v)
        )
        for v in gate_box
    ]

    H, W = mask.shape

    x1 = max(
        0,
        min(
            W - 1,
            x1
        )
    )

    y1 = max(
        0,
        min(
            H - 1,
            y1
        )
    )

    x2 = max(
        x1 + 1,
        min(
            W,
            x2
        )
    )

    y2 = max(
        y1 + 1,
        min(
            H,
            y2
        )
    )

    border = np.zeros_like(
        mask,
        dtype=bool
    )

    thickness = 2

    border[
        y1:min(
            y2,
            y1 + thickness
        ),
        x1:x2
    ] = True

    border[
        max(
            y1,
            y2 - thickness
        ):y2,
        x1:x2
    ] = True

    border[
        y1:y2,
        x1:min(
            x2,
            x1 + thickness
        )
    ] = True

    border[
        y1:y2,
        max(
            x1,
            x2 - thickness
        ):x2
    ] = True

    touched = int(
        (
            mask
            &
            border
        ).sum()
    )

    return float(
        touched
        /
        max(
            1,
            int(
                mask.sum()
            )
        )
    )


def largest_component(
    mask
):

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask.astype(
                np.uint8
            ),
            8
        )
    )

    if num_labels <= 1:
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

    return (
        labels
        ==
        idx
    )


def get_row(
    rows,
    iid
):

    return next(
        row
        for row in rows
        if int(
            row[
                "inventory_id"
            ]
        ) == iid
    )


# ============================================================
# VALIDATE
# ============================================================

for p in [
    MASTER,
    FLORENCE_RESULTS,
    SAM2_CHECKPOINT,
]:

    if not p.exists():

        raise FileNotFoundError(
            p
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
# LOAD FLORENCE RESULTS
# ============================================================

rows = json.loads(
    FLORENCE_RESULTS.read_text(
        encoding="utf-8"
    )
)

primary_row = get_row(
    rows,
    PRIMARY_ID
)

duplicate_row = get_row(
    rows,
    DUPLICATE_ID
)

unresolved_row = get_row(
    rows,
    UNRESOLVED_ID
)


if not primary_row.get(
    "boxes"
):

    raise RuntimeError(
        "010 Florence result has no box."
    )


if not duplicate_row.get(
    "boxes"
):

    raise RuntimeError(
        "025 Florence result has no box."
    )


primary_box = clamp_box(
    primary_row[
        "boxes"
    ][0][
        "bbox"
    ],
    W,
    H
)

duplicate_box = clamp_box(
    duplicate_row[
        "boxes"
    ][0][
        "bbox"
    ],
    W,
    H
)


# ============================================================
# DUPLICATE GEOMETRY
# ============================================================

def bbox_iou(
    a,
    b
):

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(
        ax1,
        bx1
    )

    iy1 = max(
        ay1,
        by1
    )

    ix2 = min(
        ax2,
        bx2
    )

    iy2 = min(
        ay2,
        by2
    )

    iw = max(
        0.0,
        ix2 - ix1
    )

    ih = max(
        0.0,
        iy2 - iy1
    )

    inter = (
        iw
        *
        ih
    )

    area_a = max(
        1.0,
        (
            ax2 - ax1
        )
        *
        (
            ay2 - ay1
        )
    )

    area_b = max(
        1.0,
        (
            bx2 - bx1
        )
        *
        (
            by2 - by1
        )
    )

    return float(
        inter
        /
        max(
            1.0,
            area_a
            +
            area_b
            -
            inter
        )
    )


florence_iou = bbox_iou(
    primary_box,
    duplicate_box
)


# ============================================================
# LOAD SAM2
# ============================================================

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 110)
print("PRODUCTION STAGE 06C10B")
print("FLORENCE FAUCET SAM2 + DUPLICATE STATE")
print("=" * 110)

print()
print(
    "010 FLORENCE BOX:",
    [
        round(v, 2)
        for v in primary_box
    ]
)

print(
    "025 FLORENCE BOX:",
    [
        round(v, 2)
        for v in duplicate_box
    ]
)

print(
    "010↔025 FLORENCE IOU:",
    round(
        florence_iou,
        4
    )
)


print()
print(
    "Loading SAM2..."
)


sam2_model = build_sam2(
    SAM2_CONFIG,
    str(
        SAM2_CHECKPOINT
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
# SAM2 ON PRIMARY 010 ONLY
# ============================================================

prompt_box = expand_box(
    primary_box,
    BOX_EXPANSION,
    W,
    H
)


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


gate = box_mask(
    expand_box(
        primary_box,
        0.20,
        W,
        H
    ),
    H,
    W
)


evaluated = []


for idx in range(
    len(
        masks
    )
):

    raw_mask = (
        masks[
            idx
        ]
        >
        0
    )

    if not raw_mask.any():
        continue

    raw_mask = largest_component(
        raw_mask
    )

    raw_area = int(
        raw_mask.sum()
    )

    local_mask = (
        raw_mask
        &
        gate
    )

    local_area = int(
        local_mask.sum()
    )

    if local_area == 0:
        continue

    containment = float(
        local_area
        /
        max(
            1,
            raw_area
        )
    )

    border_touch = border_touch_ratio(
        local_mask,
        prompt_box
    )

    sam_score = float(
        scores[
            idx
        ]
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
                0.30
            )
        )
    )

    evaluated.append({

        "mask_index":
            int(
                idx
            ),

        "mask":
            local_mask,

        "sam_score":
            sam_score,

        "quality":
            quality,

        "area":
            local_area,

        "containment":
            containment,

        "border_touch":
            border_touch,
    })


if not evaluated:

    raise RuntimeError(
        "SAM2 produced no usable faucet masks."
    )


evaluated.sort(
    key=lambda x:
        x[
            "quality"
        ],
    reverse=True
)

best = evaluated[
    0
]

best_mask = best[
    "mask"
]


geometry_pass = bool(

    best[
        "area"
    ]
    >=
    MIN_MASK_PIXELS

    and

    best[
        "sam_score"
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
# SAVE MASK
# ============================================================

MASK_PATH = (
    OBJECTS
    / "010_faucet_florence_sam2_mask.png"
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


# ============================================================
# SAVE EXACT-RGB RGBA
# ============================================================

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
    / "010_faucet_florence_sam2_rgba.png"
)

Image.fromarray(
    rgba,
    mode="RGBA"
).save(
    RGBA_PATH
)


# ============================================================
# PREVIEW
# ============================================================

preview = master_pil.copy()

draw = ImageDraw.Draw(
    preview
)


# Florence original box
draw.rectangle(
    primary_box,
    outline="blue",
    width=2
)


# SAM2 prompt box
draw.rectangle(
    prompt_box,
    outline="red",
    width=2
)


# Mask outline
mask_u8 = (
    best_mask.astype(
        np.uint8
    )
    *
    255
)

contours, _ = cv2.findContours(
    mask_u8,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

for cnt in contours:

    pts = [
        (
            int(
                p[
                    0
                ][0]
            ),
            int(
                p[
                    0
                ][1]
            )
        )
        for p in cnt
    ]

    if len(
        pts
    ) >= 2:

        draw.line(
            pts
            +
            [
                pts[
                    0
                ]
            ],
            fill="lime",
            width=2
        )


PREVIEW_PATH = (
    OBJECTS
    / "010_faucet_florence_sam2_preview.png"
)

preview.save(
    PREVIEW_PATH
)


# ============================================================
# DUPLICATE STATE
# ============================================================

duplicate_state = {

    "primary_inventory_id":
        PRIMARY_ID,

    "primary_inventory_name":
        primary_row[
            "inventory_name"
        ],

    "duplicate_inventory_id":
        DUPLICATE_ID,

    "duplicate_inventory_name":
        duplicate_row[
            "inventory_name"
        ],

    "relationship":
        "POSSIBLE_SAME_PHYSICAL_INSTANCE",

    "evidence": {

        "florence_primary_bbox":
            primary_box,

        "florence_duplicate_bbox":
            duplicate_box,

        "florence_bbox_iou":
            florence_iou,

        "qwen_grid_same_final_region":
            True,

        "do_not_segment_duplicate_separately":
            True,
    },

    "production_action":
        (
            "retain 025 as duplicate hypothesis of 010; "
            "do not create a second faucet mask until "
            "physical-instance collision audit confirms"
        )
}


DUPLICATE_STATE_PATH = (
    OUT
    / "00_duplicate_instance_state.json"
)


DUPLICATE_STATE_PATH.write_text(
    json.dumps(
        duplicate_state,
        indent=2
    ),
    encoding="utf-8"
)


# ============================================================
# RESULT
# ============================================================

result = {

    "stage":
        "06C10B",

    "primary": {

        "inventory_id":
            PRIMARY_ID,

        "inventory_name":
            primary_row[
                "inventory_name"
            ],

        "florence_bbox":
            primary_box,

        "sam2_prompt_bbox":
            prompt_box,

        "sam2_mask_index":
            best[
                "mask_index"
            ],

        "sam2_score":
            best[
                "sam_score"
            ],

        "sam2_quality":
            best[
                "quality"
            ],

        "mask_area":
            best[
                "area"
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

        "mask_path":
            str(
                MASK_PATH
            ),

        "rgba_path":
            str(
                RGBA_PATH
            ),

        "preview_path":
            str(
                PREVIEW_PATH
            ),
    },

    "duplicate": {

        "inventory_id":
            DUPLICATE_ID,

        "inventory_name":
            duplicate_row[
                "inventory_name"
            ],

        "status":
            "POSSIBLE_DUPLICATE_OF_010",

        "segmented_separately":
            False,

        "florence_bbox":
            duplicate_box,

        "florence_iou_with_010":
            florence_iou,
    },

    "unresolved": {

        "inventory_id":
            UNRESOLVED_ID,

        "inventory_name":
            unresolved_row[
                "inventory_name"
            ],

        "status":
            "UNRESOLVED",

        "reason":
            (
                "full-scene DINO, Qwen grid and Florence "
                "did not produce trustworthy drawer-handle localization"
            ),
    },

    "rules": [
        "no final prop union performed",
        "025 is not segmented separately",
        "021 remains unresolved",
        "exact RGB comes from Stage01 master",
        "SAM2 geometry pass is not semantic proof"
    ]
}


RESULT_PATH = (
    OUT
    / "01_stage06c10b_result.json"
)


RESULT_PATH.write_text(
    json.dumps(
        result,
        indent=2
    ),
    encoding="utf-8"
)


# ============================================================
# PRINT
# ============================================================

print()
print("=" * 110)
print("PRODUCTION STAGE 06C10B RESULT")
print("=" * 110)

print()
print("010 faucet")
print(
    "  Florence bbox:",
    [
        round(v, 1)
        for v in primary_box
    ]
)

print(
    "  SAM2 prompt:",
    [
        round(v, 1)
        for v in prompt_box
    ]
)

print(
    "  SAM score:",
    round(
        best[
            "sam_score"
        ],
        4
    )
)

print(
    "  quality:",
    round(
        best[
            "quality"
        ],
        4
    )
)

print(
    "  area:",
    best[
        "area"
    ]
)

print(
    "  containment:",
    round(
        best[
            "containment"
        ],
        4
    )
)

print(
    "  border touch:",
    round(
        best[
            "border_touch"
        ],
        4
    )
)

print(
    "  GEOMETRY PASS:",
    geometry_pass
)


print()
print("025 faucet")
print(
    "  Florence bbox:",
    [
        round(v, 1)
        for v in duplicate_box
    ]
)

print(
    "  IoU with 010:",
    round(
        florence_iou,
        4
    )
)

print(
    "  STATUS: POSSIBLE_DUPLICATE_OF_010"
)

print(
    "  SEGMENTED SEPARATELY: False"
)


print()
print("021 drawer handle")
print(
    "  STATUS: UNRESOLVED"
)


print()
print(
    "MASK:",
    MASK_PATH
)

print(
    "RGBA:",
    RGBA_PATH
)

print(
    "PREVIEW:",
    PREVIEW_PATH
)

print(
    "DUPLICATE STATE:",
    DUPLICATE_STATE_PATH
)

print(
    "RESULT JSON:",
    RESULT_PATH
)

print()
print(
    "NO FINAL PROP MASK UNION WAS CREATED."
)


# ============================================================
# CLEANUP
# ============================================================

del predictor
del sam2_model

if torch.cuda.is_available():

    torch.cuda.empty_cache()
