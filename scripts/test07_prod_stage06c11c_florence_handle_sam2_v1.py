
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

SOURCE_JSON = (
    STAGE06
    / "06c11b_florence_parent_relocalize_handle"
    / "00_stage06c11b_result.json"
)

OUT = (
    STAGE06
    / "06c11c_florence_handle_sam2"
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

TARGET_ID = 21

# 06C11B candidate #4 was visually confirmed.
SELECTED_CANDIDATE_INDEX = 4

PROMPT_EXPANSION = 0.15
GATE_EXPANSION = 0.30

MIN_SAM_SCORE = 0.55
MIN_CONTAINMENT = 0.85
MAX_BORDER_TOUCH = 0.35
MIN_MASK_PIXELS = 8


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


def make_gate(
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

    gate = np.zeros(
        (
            H,
            W
        ),
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
            mask.astype(
                np.uint8
            ),
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

    return (
        labels
        ==
        idx
    )


def border_touch_ratio(
    mask,
    box
):

    H, W = mask.shape

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

    border = np.zeros_like(
        mask,
        dtype=bool
    )

    t = 2

    border[
        y1:min(
            y2,
            y1 + t
        ),
        x1:x2
    ] = True

    border[
        max(
            y1,
            y2 - t
        ):y2,
        x1:x2
    ] = True

    border[
        y1:y2,
        x1:min(
            x2,
            x1 + t
        )
    ] = True

    border[
        y1:y2,
        max(
            x1,
            x2 - t
        ):x2
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
            mask.sum()
        )
    )


# ============================================================
# VALIDATE
# ============================================================

for p in [
    MASTER,
    SOURCE_JSON,
    CHECKPOINT,
]:

    if not p.exists():

        raise FileNotFoundError(
            p
        )


# ============================================================
# LOAD
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


source = json.loads(
    SOURCE_JSON.read_text(
        encoding="utf-8"
    )
)


candidates = source[
    "handle_candidates"
]


if len(candidates) < SELECTED_CANDIDATE_INDEX:

    raise RuntimeError(
        "06C11B candidate #4 not found."
    )


selected = candidates[
    SELECTED_CANDIDATE_INDEX - 1
]


candidate_box = clamp_box(
    selected[
        "global_bbox"
    ],
    W,
    H
)


print("=" * 110)
print("PRODUCTION STAGE 06C11C")
print("FLORENCE HANDLE → SAM2")
print("=" * 110)

print()
print(
    "TARGET:",
    TARGET_ID,
    "drawer handle"
)

print(
    "SOURCE CANDIDATE:",
    SELECTED_CANDIDATE_INDEX
)

print(
    "FLORENCE LABEL:",
    selected[
        "label"
    ]
)

print(
    "CANDIDATE BOX:",
    [
        round(v, 2)
        for v in candidate_box
    ]
)


# ============================================================
# PROMPT / GATE
# ============================================================

prompt_box = expand_box(
    candidate_box,
    PROMPT_EXPANSION,
    W,
    H
)

gate_box = expand_box(
    candidate_box,
    GATE_EXPANSION,
    W,
    H
)

gate = make_gate(
    gate_box,
    H,
    W
)


print(
    "PROMPT BOX:",
    [
        round(v, 2)
        for v in prompt_box
    ]
)

print(
    "GATE BOX:",
    [
        round(v, 2)
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


model = build_sam2(
    SAM2_CONFIG,
    str(
        CHECKPOINT
    ),
    device=device
)

predictor = SAM2ImagePredictor(
    model
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


for mi in range(
    len(
        masks
    )
):

    raw = (
        masks[
            mi
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
            mi
        ]
    )


    border_touch = (
        border_touch_ratio(
            local,
            prompt_box
        )
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
                0.35
            )
        )
    )


    evaluated.append({

        "mask_index":
            mi,

        "mask":
            local,

        "sam_score":
            sam_score,

        "quality":
            quality,

        "area":
            area,

        "containment":
            containment,

        "border_touch":
            border_touch,
    })


if not evaluated:

    raise RuntimeError(
        "No SAM2 mask produced."
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

mask = best[
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
    / "021_drawer_handle_florence_sam2_mask.png"
)


Image.fromarray(
    mask.astype(
        np.uint8
    )
    *
    255
).save(
    MASK_PATH
)


# ============================================================
# EXACT RGB RGBA
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
    mask.astype(
        np.uint8
    )
    *
    255
)


RGBA_PATH = (
    OBJECTS
    / "021_drawer_handle_florence_sam2_rgba.png"
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


# Florence candidate
draw.rectangle(
    candidate_box,
    outline="blue",
    width=2
)


# SAM2 prompt
draw.rectangle(
    prompt_box,
    outline="red",
    width=2
)


mask_u8 = (
    mask.astype(
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
                p[0][0]
            ),
            int(
                p[0][1]
            )
        )

        for p in cnt
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


PREVIEW_PATH = (
    OBJECTS
    / "021_drawer_handle_florence_sam2_preview.png"
)


preview.save(
    PREVIEW_PATH
)


# ============================================================
# RESULT
# ============================================================

result = {

    "stage":
        "06C11C",

    "inventory_id":
        TARGET_ID,

    "inventory_name":
        "drawer handle",

    "source_stage":
        "06C11B",

    "source_candidate_index":
        SELECTED_CANDIDATE_INDEX,

    "source_label":
        selected[
            "label"
        ],

    "florence_bbox":
        candidate_box,

    "sam2_prompt_bbox":
        prompt_box,

    "sam2_gate_bbox":
        gate_box,

    "sam2_mask_index":
        best[
            "mask_index"
        ],

    "sam2_score":
        best[
            "sam_score"
        ],

    "quality":
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

    "status":
        (
            "PROVISIONAL_PASS"
            if geometry_pass
            else
            "GEOMETRY_FAIL"
        ),

    "rules": [
        "no final prop union performed",
        "exact RGB from Stage01 master",
        "SAM2 geometry pass is not semantic proof"
    ]
}


RESULT_PATH = (
    OUT
    / "00_stage06c11c_result.json"
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
print("PRODUCTION STAGE 06C11C RESULT")
print("=" * 110)

print(
    "Florence bbox:",
    [
        round(v, 1)
        for v in candidate_box
    ]
)

print(
    "SAM2 score:",
    round(
        best[
            "sam_score"
        ],
        4
    )
)

print(
    "Quality:",
    round(
        best[
            "quality"
        ],
        4
    )
)

print(
    "Area:",
    best[
        "area"
    ]
)

print(
    "Containment:",
    round(
        best[
            "containment"
        ],
        4
    )
)

print(
    "Border touch:",
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
del model

if torch.cuda.is_available():

    torch.cuda.empty_cache()
