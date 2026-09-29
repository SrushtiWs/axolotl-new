
from pathlib import Path
import json
import math

import cv2
import numpy as np

from PIL import Image
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

ROOM_PATH = (
    PROD
    / "stage07_empty_room"
    / "07a_qwen_empty_room"
    / "00_stage07_empty_room_candidate.png"
)

FLOOR_MASK_PATH = (
    PROD
    / "stage08_tile_application"
    / "08b2_floor_plus_screeding_target"
    / "01_main_floor_mask.png"
)

OUT = (
    PROD
    / "stage08_tile_application"
    / "08e1b_strict_floor_joint_geometry"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

# Remove boundary zone very aggressively.
INTERIOR_DISTANCE_PX = 11

# Ignore image borders.
IMAGE_BORDER_PX = 7

# Local contrast
CLAHE_CLIP = 2.8
CLAHE_GRID = (6, 6)

# Minimum accepted segment length
MIN_LINE_LENGTH = 18.0

# Strong floor ownership
MIN_FLOOR_OVERLAP = 0.90

# Do not accept almost horizontal lines.
MIN_ABS_ANGLE = 8.0

# Do not accept almost vertical lines.
MAX_ABS_ANGLE = 82.0


# ============================================================
# LOAD
# ============================================================

for p in [
    ROOM_PATH,
    FLOOR_MASK_PATH,
]:

    if not p.exists():
        raise FileNotFoundError(p)


room = Image.open(
    ROOM_PATH
).convert("RGB")

room_np = np.asarray(room)

H, W = room_np.shape[:2]


floor = (
    np.asarray(
        Image.open(
            FLOOR_MASK_PATH
        ).convert("L")
    )
    > 127
)


if floor.shape != (H, W):
    raise RuntimeError(
        "Floor mask size mismatch."
    )


print("=" * 110)
print("STAGE 08E1B — STRICT INTERIOR FLOOR JOINTS")
print("=" * 110)

print()
print("ROOM:", ROOM_PATH)
print("SIZE:", f"{W} × {H}")


# ============================================================
# DISTANCE FROM FLOOR BOUNDARY
#
# This is the main difference from 08E1.
# We only use floor pixels at least N pixels away from the edge.
# ============================================================

distance = cv2.distanceTransform(
    floor.astype(np.uint8),
    cv2.DIST_L2,
    5
)


interior = (
    floor
    &
    (
        distance
        >=
        INTERIOR_DISTANCE_PX
    )
)


# image-border exclusion
interior[
    :IMAGE_BORDER_PX,
    :
] = False

interior[
    H - IMAGE_BORDER_PX:,
    :
] = False

interior[
    :,
    :IMAGE_BORDER_PX
] = False

interior[
    :,
    W - IMAGE_BORDER_PX:
] = False


# ============================================================
# ENHANCE FLOOR TEXTURE
# ============================================================

gray = cv2.cvtColor(
    room_np,
    cv2.COLOR_RGB2GRAY
)


clahe = cv2.createCLAHE(
    clipLimit=CLAHE_CLIP,
    tileGridSize=CLAHE_GRID
)


enhanced = clahe.apply(
    gray
)


# Preserve only floor interior.
masked = np.zeros_like(
    enhanced
)


masked[
    interior
] = enhanced[
    interior
]


# mild denoise
masked_blur = cv2.GaussianBlur(
    masked,
    (3, 3),
    0.7
)


# ============================================================
# LSD LINE DETECTOR
# ============================================================

lsd = cv2.createLineSegmentDetector(
    cv2.LSD_REFINE_STD
)


detected = lsd.detect(
    masked_blur
)


raw_lines = (
    detected[0]
    if detected is not None
    else None
)


segments = []


if raw_lines is not None:

    raw_lines = np.asarray(
        raw_lines
    ).reshape(
        -1,
        4
    )


    print(
        "RAW LSD SEGMENTS:",
        len(raw_lines)
    )


    for row in raw_lines:

        x1, y1, x2, y2 = map(
            float,
            row
        )


        dx = (
            x2 - x1
        )

        dy = (
            y2 - y1
        )


        length = math.hypot(
            dx,
            dy
        )


        if length < MIN_LINE_LENGTH:
            continue


        angle = math.degrees(
            math.atan2(
                dy,
                dx
            )
        )


        while angle >= 90:
            angle -= 180

        while angle < -90:
            angle += 180


        if abs(angle) < MIN_ABS_ANGLE:
            continue


        if abs(angle) > MAX_ABS_ANGLE:
            continue


        # ----------------------------------------------------
        # Sample full segment for strict interior ownership.
        # ----------------------------------------------------

        steps = max(
            10,
            int(
                math.ceil(
                    length
                )
            )
        )


        xx = np.linspace(
            x1,
            x2,
            steps
        ).round().astype(int)


        yy = np.linspace(
            y1,
            y2,
            steps
        ).round().astype(int)


        valid = (
            (xx >= 0)
            &
            (xx < W)
            &
            (yy >= 0)
            &
            (yy < H)
        )


        xx = xx[
            valid
        ]

        yy = yy[
            valid
        ]


        if len(xx) < 5:
            continue


        overlap = float(
            interior[
                yy,
                xx
            ].mean()
        )


        if overlap < MIN_FLOOR_OVERLAP:
            continue


        # median distance from floor boundary
        median_boundary_distance = float(
            np.median(
                distance[
                    yy,
                    xx
                ]
            )
        )


        if (
            median_boundary_distance
            <
            INTERIOR_DISTANCE_PX
        ):
            continue


        segments.append(
            {
                "p1": [
                    float(x1),
                    float(y1)
                ],

                "p2": [
                    float(x2),
                    float(y2)
                ],

                "length":
                    float(length),

                "angle":
                    float(angle),

                "floor_overlap":
                    overlap,

                "median_boundary_distance":
                    median_boundary_distance
            }
        )

else:

    print(
        "RAW LSD SEGMENTS: 0"
    )


print(
    "STRICT INTERIOR SEGMENTS:",
    len(segments)
)


# ============================================================
# CLUSTER ORIENTATIONS
# ============================================================

cluster_summary = {}


if len(segments) >= 4:

    features = []


    for row in segments:

        theta = math.radians(
            row["angle"]
        )


        features.append(
            [
                math.cos(
                    2.0 * theta
                ),
                math.sin(
                    2.0 * theta
                )
            ]
        )


    features = np.asarray(
        features,
        dtype=np.float32
    )


    criteria = (
        cv2.TERM_CRITERIA_EPS
        +
        cv2.TERM_CRITERIA_MAX_ITER,
        100,
        1e-4
    )


    _, labels_k, centers = cv2.kmeans(
        features,
        2,
        None,
        criteria,
        30,
        cv2.KMEANS_PP_CENTERS
    )


    labels_k = labels_k.reshape(-1)


    for i, row in enumerate(
        segments
    ):

        row["cluster"] = int(
            labels_k[i]
        )


    # --------------------------------------------------------
    # summary
    # --------------------------------------------------------

    for cid in [0, 1]:

        rows = [
            r
            for r in segments
            if r.get("cluster") == cid
        ]


        if not rows:
            continue


        weights = np.asarray(
            [
                r["length"]
                for r in rows
            ],
            dtype=np.float64
        )


        angles = np.radians(
            [
                r["angle"]
                for r in rows
            ]
        )


        c = np.sum(
            weights
            *
            np.cos(
                2 * angles
            )
        )


        s = np.sum(
            weights
            *
            np.sin(
                2 * angles
            )
        )


        mean_angle = (
            0.5
            *
            math.degrees(
                math.atan2(
                    s,
                    c
                )
            )
        )


        cluster_summary[
            str(cid)
        ] = {

            "count":
                len(rows),

            "total_length":
                float(
                    weights.sum()
                ),

            "mean_angle_deg":
                float(
                    mean_angle
                )
        }


# ============================================================
# VISUALIZATION
# ============================================================

preview = room_np.copy()


colors = {
    0: (
        255,
        0,
        0
    ),

    1: (
        0,
        255,
        255
    ),
}


for row in segments:

    cid = int(
        row.get(
            "cluster",
            0
        )
    )


    p1 = tuple(
        np.round(
            row["p1"]
        ).astype(int)
    )


    p2 = tuple(
        np.round(
            row["p2"]
        ).astype(int)
    )


    cv2.line(
        preview,
        p1,
        p2,
        colors[
            cid
        ],
        2,
        cv2.LINE_AA
    )


# ============================================================
# SAVE
# ============================================================

INTERIOR_PATH = (
    OUT
    / "01_strict_floor_interior.png"
)


Image.fromarray(
    interior.astype(np.uint8)
    *
    255
).save(
    INTERIOR_PATH
)


ENHANCED_PATH = (
    OUT
    / "02_strict_floor_enhanced.png"
)


Image.fromarray(
    masked
).save(
    ENHANCED_PATH
)


LINES_PATH = (
    OUT
    / "03_strict_floor_joint_lines.png"
)


Image.fromarray(
    preview
).save(
    LINES_PATH
)


# ============================================================
# 4-PANEL AUDIT
# ============================================================

fig, axes = plt.subplots(
    1,
    4,
    figsize=(
        18,
        7
    )
)


axes[0].imshow(
    room
)

axes[0].set_title(
    "1. Stage07A"
)

axes[0].axis(
    "off"
)


axes[1].imshow(
    interior,
    cmap="gray"
)

axes[1].set_title(
    "2. Strict Interior\nBoundary excluded"
)

axes[1].axis(
    "off"
)


axes[2].imshow(
    masked,
    cmap="gray"
)

axes[2].set_title(
    "3. Enhanced Interior Floor"
)

axes[2].axis(
    "off"
)


axes[3].imshow(
    preview
)

axes[3].set_title(
    "4. Strict Floor-Joint Lines\n"
    "RED/YELLOW = direction clusters"
)

axes[3].axis(
    "off"
)


plt.tight_layout()


AUDIT_PATH = (
    OUT
    / "04_stage08e1b_strict_floor_joint_audit.png"
)


plt.savefig(
    AUDIT_PATH,
    dpi=150,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# STATE
# ============================================================

STATE = {

    "stage":
        "08E1B",

    "purpose":
        "STRICT_INTERIOR_FLOOR_JOINT_GEOMETRY",

    "settings": {

        "interior_distance_px":
            INTERIOR_DISTANCE_PX,

        "image_border_px":
            IMAGE_BORDER_PX,

        "min_line_length":
            MIN_LINE_LENGTH,

        "min_floor_overlap":
            MIN_FLOOR_OVERLAP,

        "min_abs_angle":
            MIN_ABS_ANGLE,

        "max_abs_angle":
            MAX_ABS_ANGLE
    },

    "segments":
        segments,

    "cluster_summary":
        cluster_summary,

    "outputs": {

        "strict_interior":
            str(
                INTERIOR_PATH
            ),

        "enhanced":
            str(
                ENHANCED_PATH
            ),

        "lines":
            str(
                LINES_PATH
            ),

        "audit":
            str(
                AUDIT_PATH
            )
    },

    "status":
        "REQUIRES_STRICT_FLOOR_LINE_AUDIT"
}


STATE_PATH = (
    OUT
    / "00_stage08e1b_result.json"
)


STATE_PATH.write_text(
    json.dumps(
        STATE,
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
print("STAGE 08E1B RESULT")
print("=" * 110)

print()
print(
    "STRICT INTERIOR SEGMENTS:",
    len(
        segments
    )
)


for cid, summary in cluster_summary.items():

    print(
        "CLUSTER",
        cid,
        "| lines =",
        summary["count"],
        "| angle =",
        round(
            summary["mean_angle_deg"],
            2
        ),
        "deg",
        "| total length =",
        round(
            summary["total_length"],
            1
        )
    )


print()
print(
    "AUDIT:",
    AUDIT_PATH
)

print(
    "STATE:",
    STATE_PATH
)

print()
print(
    "NO TILE APPLICATION WAS PERFORMED."
)
