
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

SOURCE_JSON = (
    STAGE06
    / "06c14a_florence_multiobject_relocalization"
    / "00_stage06c14a_results.json"
)

WALL_SWITCH_RESULT = (
    STAGE06
    / "06c13a_florence_wall_switch_sam2"
    / "01_stage06c13a_results.json"
)

OUT = (
    STAGE06
    / "06c14b_semantic_audit_sam2_inline_preview"
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
# DECISIONS
# ============================================================

ACCEPTED = {
    2: 1,
    12: 1,
}

DUPLICATE_HYPOTHESES = {
    8: 9,
}

LAYER_REVIEW = {
    15: "GLASS_LAYER_COMPONENT_REVIEW",
}


# ============================================================
# SAM2 CONFIG
# ============================================================

PROMPT_EXPANSION = 0.12
GATE_EXPANSION = 0.25

MIN_SAM_SCORE = 0.55
MIN_CONTAINMENT = 0.85
MAX_BORDER_TOUCH = 0.35
MIN_MASK_PIXELS = 10


# ============================================================
# HELPERS
# ============================================================

def safe_name(text):

    return "".join(
        c
        if c.isalnum() or c in "-_"
        else "_"
        for c in str(text)
    )[:60]


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

    result = np.zeros(
        (H, W),
        dtype=bool
    )

    result[
        y1:y2,
        x1:x2
    ] = True

    return result


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


def bbox_iou(
    a,
    b
):

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

    area_a = max(
        1.0,
        (ax2 - ax1) * (ay2 - ay1)
    )

    area_b = max(
        1.0,
        (bx2 - bx1) * (by2 - by1)
    )

    return float(
        inter
        /
        max(
            1.0,
            area_a + area_b - inter
        )
    )


def show_image(
    image_path,
    title,
    figsize=(8, 8)
):

    image = Image.open(
        image_path
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
    SOURCE_JSON,
    CHECKPOINT,
]:

    if not path.exists():

        raise FileNotFoundError(
            path
        )


# ============================================================
# LOAD INPUTS
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


source_rows = json.loads(
    SOURCE_JSON.read_text(
        encoding="utf-8"
    )
)


source_by_id = {

    int(row["inventory_id"]):
        row

    for row in source_rows
}


# ============================================================
# ACCEPTED CANDIDATES
# ============================================================

accepted_candidates = {}


for iid, candidate_index in ACCEPTED.items():

    candidates = source_by_id[
        iid
    ][
        "candidates"
    ]

    match = next(

        c

        for c in candidates

        if int(
            c[
                "candidate_index"
            ]
        ) == candidate_index
    )

    accepted_candidates[
        iid
    ] = match


# ============================================================
# 008 DUPLICATE EVIDENCE
# ============================================================

duplicate_evidence = {}


outlet_candidate = (
    source_by_id[
        8
    ][
        "candidates"
    ][0]
)


if WALL_SWITCH_RESULT.exists():

    switch_results = json.loads(
        WALL_SWITCH_RESULT.read_text(
            encoding="utf-8"
        )
    )

    switch_good = next(
        (
            r
            for r in switch_results
            if int(
                r.get(
                    "candidate_index",
                    -1
                )
            ) == 1
        ),
        None
    )


    if switch_good is not None:

        duplicate_evidence[
            "008_bbox"
        ] = outlet_candidate[
            "bbox"
        ]

        duplicate_evidence[
            "009_bbox"
        ] = switch_good[
            "florence_bbox"
        ]

        duplicate_evidence[
            "008_vs_009_bbox_iou"
        ] = bbox_iou(
            outlet_candidate[
                "bbox"
            ],
            switch_good[
                "florence_bbox"
            ]
        )


# ============================================================
# CREATE 008 VS 009 COMPARISON PREVIEW
# ============================================================

duplicate_preview = master_pil.copy()

draw = ImageDraw.Draw(
    duplicate_preview
)


outlet_box = outlet_candidate[
    "bbox"
]


draw.rectangle(
    outlet_box,
    outline="red",
    width=3
)

draw.text(
    (
        outlet_box[0],
        outlet_box[1] - 12
    ),
    "008 outlet",
    fill="red"
)


if (
    duplicate_evidence.get(
        "009_bbox"
    )
    is not None
):

    switch_box = duplicate_evidence[
        "009_bbox"
    ]

    draw.rectangle(
        switch_box,
        outline="lime",
        width=2
    )

    draw.text(
        (
            switch_box[0],
            switch_box[3] + 2
        ),
        "009 switch",
        fill="lime"
    )


DUPLICATE_PREVIEW_PATH = (
    OUT
    / "01_008_vs_009_duplicate_preview.png"
)


duplicate_preview.save(
    DUPLICATE_PREVIEW_PATH
)


# ============================================================
# CREATE 015 GLASS REVIEW PREVIEW
# ============================================================

towel_candidate = (
    source_by_id[
        15
    ][
        "candidates"
    ][0]
)


glass_preview = master_pil.copy()

draw = ImageDraw.Draw(
    glass_preview
)


tb = towel_candidate[
    "bbox"
]


draw.rectangle(
    tb,
    outline="red",
    width=3
)

draw.text(
    (
        tb[0],
        tb[1] - 12
    ),
    "015 towel bar",
    fill="red"
)


GLASS_PREVIEW_PATH = (
    OUT
    / "02_015_glass_component_preview.png"
)


glass_preview.save(
    GLASS_PREVIEW_PATH
)


# ============================================================
# LOAD SAM2
# ============================================================

print("=" * 110)
print("PRODUCTION STAGE 06C14B V2")
print("SEMANTIC AUDIT + SAM2 + INLINE PREVIEWS")
print("=" * 110)

print()
print("002 → candidate #1 ACCEPTED")
print("008 → POSSIBLE_DUPLICATE_OF_009")
print("012 → candidate #1 ACCEPTED")
print("015 → GLASS_LAYER_COMPONENT_REVIEW")


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
# RUN SAM2 ON 002 + 012
# ============================================================

sam_results = []


for iid in [
    2,
    12,
]:

    object_row = source_by_id[
        iid
    ]

    candidate = accepted_candidates[
        iid
    ]

    name = object_row[
        "inventory_name"
    ]

    source_box = clamp_box(
        candidate[
            "bbox"
        ],
        W,
        H
    )

    prompt_box = expand_box(
        source_box,
        PROMPT_EXPANSION,
        W,
        H
    )

    gate_box = expand_box(
        source_box,
        GATE_EXPANSION,
        W,
        H
    )

    gate = make_gate(
        gate_box,
        H,
        W
    )


    print()
    print("=" * 100)
    print(
        f"{iid:03d}. {name}"
    )
    print("=" * 100)

    print(
        "FLORENCE:",
        [
            round(v, 1)
            for v in source_box
        ]
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
            f"No SAM2 result for {iid}"
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


    prefix = (
        f"{iid:03d}_"
        f"{safe_name(name)}"
    )


    MASK_PATH = (
        OBJECTS
        /
        f"{prefix}_mask.png"
    )

    RGBA_PATH = (
        OBJECTS
        /
        f"{prefix}_rgba.png"
    )

    PREVIEW_PATH = (
        OBJECTS
        /
        f"{prefix}_preview.png"
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


    Image.fromarray(
        rgba,
        mode="RGBA"
    ).save(
        RGBA_PATH
    )


    preview = master_pil.copy()

    draw = ImageDraw.Draw(
        preview
    )


    draw.rectangle(
        source_box,
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


    preview.save(
        PREVIEW_PATH
    )


    result = {

        "inventory_id":
            iid,

        "inventory_name":
            name,

        "source_stage":
            "06C14A",

        "selected_candidate_index":
            1,

        "florence_bbox":
            source_box,

        "florence_label":
            candidate[
                "label"
            ],

        "supporting_queries":
            candidate[
                "supporting_queries"
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

        "status":
            (
                "PROVISIONAL_PASS"
                if geometry_pass
                else
                "GEOMETRY_FAIL"
            ),

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
    }


    sam_results.append(
        result
    )


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
            "mask_area"
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


# ============================================================
# SAVE STATE
# ============================================================

state = {

    "stage":
        "06C14B_V2",

    "resolved_candidates":
        sam_results,

    "duplicate_hypotheses": [

        {
            "inventory_id":
                8,

            "inventory_name":
                source_by_id[
                    8
                ][
                    "inventory_name"
                ],

            "state":
                "POSSIBLE_DUPLICATE_OF_009",

            "duplicate_of":
                9,

            "evidence":
                duplicate_evidence,

            "segmented_separately":
                False,
        }
    ],

    "layer_reviews": [

        {
            "inventory_id":
                15,

            "inventory_name":
                source_by_id[
                    15
                ][
                    "inventory_name"
                ],

            "state":
                "GLASS_LAYER_COMPONENT_REVIEW",

            "candidate_bbox":
                towel_candidate[
                    "bbox"
                ],

            "segmented_separately":
                False,
        }
    ],
}


RESULT_PATH = (
    OUT
    / "00_stage06c14b_state.json"
)


RESULT_PATH.write_text(
    json.dumps(
        state,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# ============================================================
# PRINT RESULT
# ============================================================

print()
print("=" * 110)
print("PRODUCTION STAGE 06C14B RESULT")
print("=" * 110)


for row in sam_results:

    print()

    print(
        "{:03d}. {:24s} | status={} | SAM={} | area={} | contain={} | border={}".format(

            row[
                "inventory_id"
            ],

            row[
                "inventory_name"
            ][:24],

            row[
                "status"
            ],

            round(
                row[
                    "sam2_score"
                ],
                4
            ),

            row[
                "mask_area"
            ],

            round(
                row[
                    "containment"
                ],
                4
            ),

            round(
                row[
                    "border_touch"
                ],
                4
            ),
        )
    )


print()
print(
    "008:",
    "POSSIBLE_DUPLICATE_OF_009"
)

print(
    "008↔009 IOU:",
    duplicate_evidence.get(
        "008_vs_009_bbox_iou"
    )
)


print()
print(
    "015:",
    "GLASS_LAYER_COMPONENT_REVIEW"
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
# INLINE VISUAL VERIFICATION
# ============================================================

print()
print("=" * 110)
print("INLINE VISUAL VERIFICATION")
print("=" * 110)


for row in sam_results:

    show_image(
        row[
            "preview_path"
        ],
        (
            f'{row["inventory_id"]:03d} '
            f'{row["inventory_name"]} '
            f'— SAM2 PREVIEW'
        ),
        figsize=(7, 9)
    )


show_image(
    DUPLICATE_PREVIEW_PATH,
    (
        "008 WALL OUTLET vs 009 WALL SWITCH "
        "— DUPLICATE CHECK"
    ),
    figsize=(7, 9)
)


show_image(
    GLASS_PREVIEW_PATH,
    (
        "015 TOWEL BAR "
        "— GLASS LAYER COMPONENT REVIEW"
    ),
    figsize=(7, 9)
)


# ============================================================
# CLEANUP
# ============================================================

del predictor
del sam2_model

if torch.cuda.is_available():

    torch.cuda.empty_cache()
