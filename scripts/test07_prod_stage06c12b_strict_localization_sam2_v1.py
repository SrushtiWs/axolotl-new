
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

VERIFIED_JSON = (
    STAGE06
    / "06c1b_original_master_verification"
    / "02_verified_candidates.json"
)

OUT = (
    STAGE06
    / "06c12b_strict_localization_sam2"
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
# TARGETS
# ============================================================

TARGET_IDS = [
    5,
    9,
    13,
    14,
    29,
]


# ============================================================
# SAM2 SAFETY SETTINGS
# ============================================================

PROMPT_EXPANSION = 0.08
GATE_EXPANSION = 0.16

MIN_SAM_SCORE = 0.55
MIN_CONTAINMENT = 0.85
MAX_BORDER_TOUCH = 0.30
MIN_MASK_PIXELS = 20


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
        int(round(v))
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

    result = np.zeros(
        (
            H,
            W
        ),
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
        int(round(v))
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

    return float(
        (
            mask
            &
            border
        ).sum()
        /
        max(
            1,
            int(
                mask.sum()
            )
        )
    )


# ============================================================
# VALIDATE
# ============================================================

for path in [
    MASTER,
    VERIFIED_JSON,
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
# LOAD STRICT VERIFICATION
# ============================================================

verified = json.loads(
    VERIFIED_JSON.read_text(
        encoding="utf-8"
    )
)


# ============================================================
# GROUP VALID CANDIDATES BY PHYSICAL PARENT
# ============================================================

valid_by_parent = {
    iid: []
    for iid in TARGET_IDS
}


for row in verified:

    if str(
        row.get(
            "verification_decision",
            ""
        )
    ).upper() != "VALID":

        continue


    parent_id = row.get(
        "parent_inventory_id"
    )

    if parent_id is None:
        continue


    parent_id = int(
        parent_id
    )


    if parent_id not in valid_by_parent:
        continue


    valid_by_parent[
        parent_id
    ].append(
        row
    )


# ============================================================
# SELECT HIGHEST-SCORE VALID CANDIDATE
# ============================================================

selected = {}


for iid in TARGET_IDS:

    candidates = valid_by_parent[
        iid
    ]

    if not candidates:

        raise RuntimeError(
            f"No VALID strict candidate for parent {iid}"
        )


    candidates.sort(
        key=lambda r:
            float(
                r.get(
                    "score",
                    0.0
                )
            ),
        reverse=True
    )


    selected[
        iid
    ] = candidates[
        0
    ]


print("=" * 110)
print("PRODUCTION STAGE 06C12B")
print("STRICT ORIGINAL-MASTER LOCALIZATION → SAM2")
print("=" * 110)


for iid in TARGET_IDS:

    row = selected[
        iid
    ]

    print()

    print(
        f"{iid:03d}. "
        f'{row["parent_inventory_name"]}'
    )

    print(
        "  VALID candidates:",
        len(
            valid_by_parent[
                iid
            ]
        )
    )

    print(
        "  selected DINO:",
        round(
            float(
                row[
                    "score"
                ]
            ),
            4
        )
    )

    print(
        "  bbox:",
        [
            round(v, 1)
            for v in row[
                "bbox"
            ]
        ]
    )


# ============================================================
# LOAD SAM2 ONCE
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
# PROCESS EACH PHYSICAL INSTANCE
# ============================================================

results = []


for iid in TARGET_IDS:

    source = selected[
        iid
    ]


    name = str(
        source[
            "parent_inventory_name"
        ]
    )


    source_box = clamp_box(
        source[
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


    gate = box_mask(
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


    # ========================================================
    # SAM2 MULTIMASK
    # ========================================================

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


    for mask_index in range(
        len(
            masks
        )
    ):

        raw = (
            masks[
                mask_index
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
                mask_index
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
                    MAX_BORDER_TOUCH
                )
            )
        )


        evaluated.append({

            "mask_index":
                int(
                    mask_index
                ),

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

        results.append({

            "inventory_id":
                iid,

            "inventory_name":
                name,

            "status":
                "NO_SAM2_MASK",

            "geometry_pass":
                False,

            "source_bbox":
                source_box,

            "source_score":
                float(
                    source[
                        "score"
                    ]
                ),
        })

        print(
            "❌ NO SAM2 MASK"
        )

        continue


    evaluated.sort(
        key=lambda r:
            r[
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


    # ========================================================
    # OUTPUT NAMES
    # ========================================================

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


    # ========================================================
    # MASK
    # ========================================================

    Image.fromarray(
        mask.astype(
            np.uint8
        )
        *
        255
    ).save(
        MASK_PATH
    )


    # ========================================================
    # EXACT STAGE01 RGB RGBA
    # ========================================================

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


    # ========================================================
    # PREVIEW
    #
    # blue  = original strict localization
    # red   = SAM2 prompt box
    # green = selected SAM2 contour
    # ========================================================

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


    for contour in contours:

        points = [

            (
                int(
                    p[0][0]
                ),
                int(
                    p[0][1]
                )
            )

            for p in contour
        ]


        if len(
            points
        ) >= 2:

            draw.line(
                points
                +
                [
                    points[
                        0
                    ]
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

        "status":
            (
                "PROVISIONAL_PASS"
                if geometry_pass
                else
                "GEOMETRY_FAIL"
            ),

        "geometry_pass":
            geometry_pass,

        "source_stage":
            "06C1B_ORIGINAL_MASTER_VERIFICATION",

        "source_observation_id":
            source.get(
                "observation_id"
            ),

        "source_region":
            source.get(
                "source_region"
            ),

        "source_dino_score":
            float(
                source[
                    "score"
                ]
            ),

        "source_bbox":
            source_box,

        "prompt_bbox":
            prompt_box,

        "gate_bbox":
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


    results.append(
        result
    )


    print(
        "SAM2:",
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


# ============================================================
# OVERVIEW PREVIEW
# ============================================================

overview = master_pil.copy()

overview_draw = ImageDraw.Draw(
    overview
)


for row in results:

    if not row.get(
        "mask_path"
    ):
        continue


    mask_arr = (
        np.asarray(
            Image.open(
                row[
                    "mask_path"
                ]
            ).convert(
                "L"
            )
        )
        >
        0
    )


    contours, _ = cv2.findContours(
        (
            mask_arr.astype(
                np.uint8
            )
            *
            255
        ),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    for contour in contours:

        points = [

            (
                int(
                    p[0][0]
                ),
                int(
                    p[0][1]
                )
            )

            for p in contour
        ]


        if len(points) >= 2:

            overview_draw.line(
                points
                +
                [
                    points[0]
                ],
                fill="lime",
                width=2
            )


OVERVIEW_PATH = (
    OUT
    / "01_all_strict_masks_preview.png"
)


overview.save(
    OVERVIEW_PATH
)


# ============================================================
# SAVE RESULTS
# ============================================================

RESULT_PATH = (
    OUT
    / "00_stage06c12b_results.json"
)


RESULT_PATH.write_text(
    json.dumps(
        results,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print()
print("=" * 110)
print("PRODUCTION STAGE 06C12B RESULT")
print("=" * 110)


for row in results:

    print()

    print(
        "{:03d}. {:20s} | status={} | SAM={} | area={} | contain={} | border={}".format(

            row[
                "inventory_id"
            ],

            row[
                "inventory_name"
            ][:20],

            row[
                "status"
            ],

            (
                round(
                    row[
                        "sam2_score"
                    ],
                    4
                )
                if row.get(
                    "sam2_score"
                ) is not None
                else None
            ),

            row.get(
                "mask_area"
            ),

            (
                round(
                    row[
                        "containment"
                    ],
                    4
                )
                if row.get(
                    "containment"
                ) is not None
                else None
            ),

            (
                round(
                    row[
                        "border_touch"
                    ],
                    4
                )
                if row.get(
                    "border_touch"
                ) is not None
                else None
            ),
        )
    )


print()
print(
    "OVERVIEW:",
    OVERVIEW_PATH
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
