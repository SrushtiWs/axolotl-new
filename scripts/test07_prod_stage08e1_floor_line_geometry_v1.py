
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
    / "08e1_floor_line_geometry"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

# Increase local contrast enough to reveal faint floor joints.
CLAHE_CLIP = 2.2
CLAHE_GRID = (8, 8)

# Ignore strong wall-floor boundary using erosion.
FLOOR_ERODE_PX = 7

# Line detection.
CANNY_LOW = 18
CANNY_HIGH = 55

HOUGH_THRESHOLD = 15
MIN_LINE_LENGTH = 14
MAX_LINE_GAP = 8

# Remove almost-horizontal noise caused by image boundaries.
MIN_ABS_ANGLE_DEG = 7

# Limit lines to lower/floor portion.
MIN_FLOOR_OVERLAP = 0.72


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
print("STAGE 08E1 — EXISTING FLOOR LINE GEOMETRY")
print("=" * 110)

print()
print("ROOM:", ROOM_PATH)
print("FLOOR MASK:", FLOOR_MASK_PATH)
print("SIZE:", f"{W} × {H}")


# ============================================================
# SAFE FLOOR INTERIOR
# ============================================================

kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (
        FLOOR_ERODE_PX * 2 + 1,
        FLOOR_ERODE_PX * 2 + 1
    )
)


safe_floor = cv2.erode(
    floor.astype(np.uint8),
    kernel,
    iterations=1
).astype(bool)


# ============================================================
# GRAYSCALE + LOCAL CONTRAST
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


# ============================================================
# SUPPRESS NON-FLOOR AREA
# ============================================================

floor_enhanced = np.zeros_like(
    enhanced
)


floor_enhanced[
    safe_floor
] = enhanced[
    safe_floor
]


# ============================================================
# EDGE DETECTION
# ============================================================

edges = cv2.Canny(
    floor_enhanced,
    CANNY_LOW,
    CANNY_HIGH
)


edges[
    ~safe_floor
] = 0


# ============================================================
# HOUGH LINES
# ============================================================

lines = cv2.HoughLinesP(
    edges,
    rho=1,
    theta=np.pi / 180.0,
    threshold=HOUGH_THRESHOLD,
    minLineLength=MIN_LINE_LENGTH,
    maxLineGap=MAX_LINE_GAP
)


segments = []


if lines is not None:

    for row in lines[:, 0, :]:

        x1, y1, x2, y2 = map(
            int,
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


        # Normalize orientation to [-90, 90)
        while angle >= 90:
            angle -= 180

        while angle < -90:
            angle += 180


        if abs(angle) < MIN_ABS_ANGLE_DEG:
            continue


        # Sample line pixels and require strong floor ownership.
        steps = max(
            2,
            int(length)
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


        inside = (
            (xx >= 0)
            &
            (xx < W)
            &
            (yy >= 0)
            &
            (yy < H)
        )


        xx = xx[
            inside
        ]

        yy = yy[
            inside
        ]


        if len(xx) == 0:
            continue


        overlap = float(
            safe_floor[
                yy,
                xx
            ].mean()
        )


        if overlap < MIN_FLOOR_OVERLAP:
            continue


        segments.append(
            {
                "p1":
                    [x1, y1],

                "p2":
                    [x2, y2],

                "length":
                    float(length),

                "angle":
                    float(angle),

                "floor_overlap":
                    overlap
            }
        )


# ============================================================
# ORIENTATION CLUSTERING
#
# Use doubled-angle representation because line orientation
# is periodic by 180 degrees.
# ============================================================

if len(segments) < 4:

    print()
    print(
        "⚠️ Only",
        len(segments),
        "usable floor lines detected."
    )

    print(
        "We will visually inspect before deciding the next detector."
    )


else:

    features = []


    for row in segments:

        theta = math.radians(
            row["angle"]
        )


        features.append(
            [
                math.cos(
                    2 * theta
                ),
                math.sin(
                    2 * theta
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
        20,
        cv2.KMEANS_PP_CENTERS
    )


    labels_k = labels_k.reshape(-1)


    for i, row in enumerate(segments):

        row["cluster"] = int(
            labels_k[i]
        )


# ============================================================
# COMPUTE CLUSTER SUMMARY
# ============================================================

cluster_summary = {}


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
# LINE VISUALIZATION
# ============================================================

line_preview = room_np.copy()


cluster_colors = {
    0: (255, 0, 0),
    1: (0, 255, 255),
}


for row in segments:

    cid = row.get(
        "cluster",
        0
    )


    color = cluster_colors[
        cid
    ]


    cv2.line(
        line_preview,
        tuple(
            row["p1"]
        ),
        tuple(
            row["p2"]
        ),
        color,
        2,
        cv2.LINE_AA
    )


# ============================================================
# SAVE INTERMEDIATE
# ============================================================

SAFE_FLOOR_PATH = (
    OUT
    / "01_safe_floor_mask.png"
)


Image.fromarray(
    safe_floor.astype(np.uint8)
    *
    255
).save(
    SAFE_FLOOR_PATH
)


ENHANCED_PATH = (
    OUT
    / "02_floor_contrast_enhanced.png"
)


Image.fromarray(
    floor_enhanced
).save(
    ENHANCED_PATH
)


EDGES_PATH = (
    OUT
    / "03_floor_edges.png"
)


Image.fromarray(
    edges
).save(
    EDGES_PATH
)


LINES_PATH = (
    OUT
    / "04_detected_floor_lines.png"
)


Image.fromarray(
    line_preview
).save(
    LINES_PATH
)


# ============================================================
# 5-PANEL AUDIT
# ============================================================

fig, axes = plt.subplots(
    1,
    5,
    figsize=(22, 7)
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
    safe_floor,
    cmap="gray"
)

axes[1].set_title(
    "2. Safe Floor Interior"
)

axes[1].axis(
    "off"
)


axes[2].imshow(
    floor_enhanced,
    cmap="gray"
)

axes[2].set_title(
    "3. Enhanced Floor"
)

axes[2].axis(
    "off"
)


axes[3].imshow(
    edges,
    cmap="gray"
)

axes[3].set_title(
    "4. Floor Edges"
)

axes[3].axis(
    "off"
)


axes[4].imshow(
    line_preview
)

axes[4].set_title(
    "5. Floor Line Clusters\nRED/YELLOW = two directions"
)

axes[4].axis(
    "off"
)


plt.tight_layout()


AUDIT_PATH = (
    OUT
    / "05_stage08e1_floor_line_audit.png"
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
        "08E1",

    "purpose":
        "RECOVER_FLOOR_PERSPECTIVE_DIRECTIONS_FROM_EXISTING_LINES",

    "input": {

        "room":
            str(
                ROOM_PATH
            ),

        "floor_mask":
            str(
                FLOOR_MASK_PATH
            )
    },

    "settings": {

        "floor_erode_px":
            FLOOR_ERODE_PX,

        "canny_low":
            CANNY_LOW,

        "canny_high":
            CANNY_HIGH,

        "hough_threshold":
            HOUGH_THRESHOLD,

        "min_line_length":
            MIN_LINE_LENGTH,

        "max_line_gap":
            MAX_LINE_GAP
    },

    "segments":
        segments,

    "cluster_summary":
        cluster_summary,

    "outputs": {

        "safe_floor":
            str(
                SAFE_FLOOR_PATH
            ),

        "enhanced_floor":
            str(
                ENHANCED_PATH
            ),

        "edges":
            str(
                EDGES_PATH
            ),

        "detected_lines":
            str(
                LINES_PATH
            ),

        "audit":
            str(
                AUDIT_PATH
            )
    },

    "status":
        "REQUIRES_VISUAL_LINE_GEOMETRY_AUDIT"
}


STATE_PATH = (
    OUT
    / "00_stage08e1_result.json"
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
print("STAGE 08E1 RESULT")
print("=" * 110)

print()
print(
    "USABLE LINE SEGMENTS:",
    len(
        segments
    )
)

print()

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
