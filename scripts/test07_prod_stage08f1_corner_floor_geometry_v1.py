
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
    / "08f1_corner_floor_geometry"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOCKED METRIC ROOM
# ============================================================

LEFT_AXIS_MM = 2400.0
RIGHT_AXIS_MM = 1800.0


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
print("STAGE 08F1 — CORNER-BASED FLOOR GEOMETRY")
print("=" * 110)

print()
print(
    "LEFT AXIS:",
    LEFT_AXIS_MM,
    "mm"
)

print(
    "RIGHT AXIS:",
    RIGHT_AXIS_MM,
    "mm"
)


# ============================================================
# MAIN FLOOR COMPONENT
# ============================================================

n, labels, stats, _ = cv2.connectedComponentsWithStats(
    floor.astype(np.uint8),
    connectivity=8
)


if n <= 1:
    raise RuntimeError(
        "No floor component."
    )


largest = (
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


floor_main = (
    labels == largest
)


# ============================================================
# UPPER FLOOR BOUNDARY
#
# topmost floor pixel per x
# ============================================================

upper = np.full(
    W,
    np.nan,
    dtype=np.float32
)


for x in range(W):

    ys = np.where(
        floor_main[:, x]
    )[0]

    if len(ys):

        upper[x] = float(
            ys.min()
        )


valid_x = np.where(
    np.isfinite(
        upper
    )
)[0]


if len(valid_x) < 30:

    raise RuntimeError(
        "Too little floor boundary."
    )


xmin = int(
    valid_x.min()
)

xmax = int(
    valid_x.max()
)


# ============================================================
# SMOOTH BOUNDARY
# ============================================================

xs = valid_x.astype(
    np.float32
)

ys = upper[
    valid_x
].astype(
    np.float32
)


ys_smooth = cv2.GaussianBlur(
    ys.reshape(
        1,
        -1
    ),
    (0, 0),
    sigmaX=2.0
).reshape(-1)


# ============================================================
# BACK CORNER = V APEX
#
# Minimum boundary Y.
# ============================================================

apex_i = int(
    np.argmin(
        ys_smooth
    )
)


BACK = np.array(
    [
        float(
            xs[
                apex_i
            ]
        ),
        float(
            ys_smooth[
                apex_i
            ]
        )
    ],
    dtype=np.float64
)


# ============================================================
# FIT LEFT / RIGHT ROOM EDGES
# ============================================================

left_sel = (
    xs
    <
    BACK[0] - 5
)


right_sel = (
    xs
    >
    BACK[0] + 5
)


left_x = xs[
    left_sel
]

left_y = ys_smooth[
    left_sel
]


right_x = xs[
    right_sel
]

right_y = ys_smooth[
    right_sel
]


if (
    len(left_x) < 15
    or
    len(right_x) < 15
):

    raise RuntimeError(
        "Not enough left/right boundary points."
    )


# ============================================================
# ROBUST LINE FIT
# ============================================================

def robust_fit(
    x,
    y,
    iterations=5
):

    keep = np.ones(
        len(x),
        dtype=bool
    )

    coef = np.polyfit(
        x,
        y,
        1
    )


    for _ in range(
        iterations
    ):

        pred = (
            coef[0] * x
            +
            coef[1]
        )

        residual = (
            y - pred
        )


        med = np.median(
            residual[
                keep
            ]
        )


        mad = np.median(
            np.abs(
                residual[
                    keep
                ]
                -
                med
            )
        )


        sigma = max(
            0.8,
            1.4826 * mad
        )


        keep = (
            np.abs(
                residual - med
            )
            <=
            2.3 * sigma
        )


        if keep.sum() < 8:
            break


        coef = np.polyfit(
            x[
                keep
            ],
            y[
                keep
            ],
            1
        )


    return coef, keep


left_coef, left_keep = robust_fit(
    left_x,
    left_y
)


right_coef, right_keep = robust_fit(
    right_x,
    right_y
)


# ============================================================
# USE ACTUAL MASK EXTREMES FOR ADJACENT CORNERS
#
# LEFT_CORNER:
# intersection of fitted left room edge with the lower visible
# floor boundary / image side.
#
# RIGHT_CORNER:
# same on right.
#
# These are the two adjacent rectangle corners visible along
# the wall-floor edges.
# ============================================================

def floor_bottom_at_x(
    x
):

    x = int(
        np.clip(
            round(x),
            0,
            W - 1
        )
    )

    ys = np.where(
        floor_main[
            :,
            x
        ]
    )[0]


    if not len(ys):
        return None


    return float(
        ys.max()
    )


# ------------------------------------------------------------
# LEFT adjacent corner
#
# Prefer floor-mask contact with left image border/near border.
# ------------------------------------------------------------

left_candidates = []


for x in range(
    xmin,
    min(
        xmin + 20,
        W
    )
):

    yb = floor_bottom_at_x(
        x
    )

    if yb is not None:

        left_candidates.append(
            (
                x,
                yb
            )
        )


if not left_candidates:

    raise RuntimeError(
        "Could not locate left adjacent floor corner."
    )


LEFT = np.array(
    max(
        left_candidates,
        key=lambda p: p[1]
    ),
    dtype=np.float64
)


# ------------------------------------------------------------
# RIGHT adjacent corner
# ------------------------------------------------------------

right_candidates = []


for x in range(
    max(
        0,
        xmax - 19
    ),
    xmax + 1
):

    yb = floor_bottom_at_x(
        x
    )

    if yb is not None:

        right_candidates.append(
            (
                x,
                yb
            )
        )


if not right_candidates:

    raise RuntimeError(
        "Could not locate right adjacent floor corner."
    )


RIGHT = np.array(
    max(
        right_candidates,
        key=lambda p: p[1]
    ),
    dtype=np.float64
)


# ============================================================
# DIRECTION VECTORS
#
# BACK -> LEFT  = 2400 mm axis
# BACK -> RIGHT = 1800 mm axis
# ============================================================

v_left = (
    LEFT - BACK
)


v_right = (
    RIGHT - BACK
)


# ============================================================
# FIRST FOURTH-CORNER ESTIMATE
#
# In an affine plane:
#
# FRONT = LEFT + RIGHT - BACK
#
# This is NOT yet our final perspective mapping.
#
# It gives us a geometrically interpretable initial fourth
# corner to audit before adding projective correction.
# ============================================================

FRONT_AFFINE = (
    LEFT
    +
    RIGHT
    -
    BACK
)


# ============================================================
# PROJECTIVE-AWARE FRONT CORNER REGULARIZATION
#
# The true fourth corner may lie below the visible image.
#
# We constrain:
# - its x position close to affine estimate
# - it must be farther from BACK than both adjacent corners
# - if affine estimate falls above image bottom, push it toward
#   the lower image edge.
# ============================================================

front_x = float(
    FRONT_AFFINE[0]
)


front_y = float(
    FRONT_AFFINE[1]
)


# If too high, push downward.
minimum_front_y = max(
    LEFT[1],
    RIGHT[1]
) + 30.0


front_y = max(
    front_y,
    minimum_front_y
)


# Allow front corner below image.
front_y = min(
    front_y,
    H + 250.0
)


FRONT = np.array(
    [
        front_x,
        front_y
    ],
    dtype=np.float64
)


# ============================================================
# METRIC CORNER CONVENTION
#
# BACK  = (0,0)
# LEFT  = (2400,0)
# RIGHT = (0,1800)
# FRONT = (2400,1800)
# ============================================================

METRIC_CORNERS = np.array(
    [
        [
            0.0,
            0.0
        ],
        [
            LEFT_AXIS_MM,
            0.0
        ],
        [
            0.0,
            RIGHT_AXIS_MM
        ],
        [
            LEFT_AXIS_MM,
            RIGHT_AXIS_MM
        ]
    ],
    dtype=np.float32
)


IMAGE_CORNERS = np.array(
    [
        BACK,
        LEFT,
        RIGHT,
        FRONT
    ],
    dtype=np.float32
)


# ============================================================
# HOMOGRAPHY
#
# This is a candidate only.
# ============================================================

H_METRIC_TO_IMAGE = cv2.getPerspectiveTransform(
    METRIC_CORNERS,
    IMAGE_CORNERS
)


# ============================================================
# GENERATE COARSE METRIC TEST GRID
#
# We are NOT applying tile.
#
# Grid:
# - 600 mm along 2400 axis
# - 600 mm increments along 1800 axis
#
# purely geometry audit.
# ============================================================

preview = room_np.copy()


def project_metric_points(
    points
):

    arr = np.asarray(
        points,
        dtype=np.float32
    ).reshape(
        1,
        -1,
        2
    )


    projected = cv2.perspectiveTransform(
        arr,
        H_METRIC_TO_IMAGE
    )[0]


    return projected


# ------------------------------------------------------------
# Grid along 2400-mm direction
# ------------------------------------------------------------

for x_mm in [
    0,
    600,
    1200,
    1800,
    2400
]:

    pts = project_metric_points(
        [
            [
                x_mm,
                0
            ],
            [
                x_mm,
                RIGHT_AXIS_MM
            ]
        ]
    )


    p1 = tuple(
        np.round(
            pts[0]
        ).astype(int)
    )

    p2 = tuple(
        np.round(
            pts[1]
        ).astype(int)
    )


    cv2.line(
        preview,
        p1,
        p2,
        (
            255,
            0,
            0
        ),
        1,
        cv2.LINE_AA
    )


# ------------------------------------------------------------
# Grid along 1800-mm direction
# ------------------------------------------------------------

for y_mm in [
    0,
    600,
    1200,
    1800
]:

    pts = project_metric_points(
        [
            [
                0,
                y_mm
            ],
            [
                LEFT_AXIS_MM,
                y_mm
            ]
        ]
    )


    p1 = tuple(
        np.round(
            pts[0]
        ).astype(int)
    )

    p2 = tuple(
        np.round(
            pts[1]
        ).astype(int)
    )


    cv2.line(
        preview,
        p1,
        p2,
        (
            0,
            255,
            255
        ),
        1,
        cv2.LINE_AA
    )


# ============================================================
# CORNER AUDIT PREVIEW
# ============================================================

corner_preview = room_np.copy()


names = [
    ("B", BACK),
    ("L", LEFT),
    ("R", RIGHT),
    ("F", FRONT),
]


colors = {
    "B": (
        255,
        0,
        255
    ),
    "L": (
        255,
        0,
        0
    ),
    "R": (
        0,
        255,
        255
    ),
    "F": (
        0,
        255,
        0
    )
}


for name, p in names:

    px = int(
        round(
            p[0]
        )
    )

    py = int(
        round(
            p[1]
        )
    )


    # Draw only if visible.
    if (
        0 <= px < W
        and
        0 <= py < H
    ):

        cv2.circle(
            corner_preview,
            (
                px,
                py
            ),
            5,
            colors[
                name
            ],
            -1
        )


        cv2.putText(
            corner_preview,
            name,
            (
                px + 6,
                py - 6
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            colors[
                name
            ],
            1,
            cv2.LINE_AA
        )


# visible room edges
cv2.line(
    corner_preview,
    tuple(
        np.round(
            BACK
        ).astype(int)
    ),
    tuple(
        np.round(
            LEFT
        ).astype(int)
    ),
    (
        255,
        0,
        0
    ),
    2,
    cv2.LINE_AA
)


cv2.line(
    corner_preview,
    tuple(
        np.round(
            BACK
        ).astype(int)
    ),
    tuple(
        np.round(
            RIGHT
        ).astype(int)
    ),
    (
        0,
        255,
        255
    ),
    2,
    cv2.LINE_AA
)


# ============================================================
# SAVE
# ============================================================

CORNER_PATH = (
    OUT
    / "01_floor_corner_model.png"
)


GRID_PATH = (
    OUT
    / "02_corner_based_metric_grid_audit.png"
)


Image.fromarray(
    corner_preview
).save(
    CORNER_PATH
)


Image.fromarray(
    preview
).save(
    GRID_PATH
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
    floor_main,
    cmap="gray"
)

axes[1].set_title(
    "2. Verified Floor Mask"
)

axes[1].axis(
    "off"
)


axes[2].imshow(
    corner_preview
)

axes[2].set_title(
    "3. Rectangle Corner Model\n"
    "B=back | L=2400 axis | R=1800 axis"
)

axes[2].axis(
    "off"
)


axes[3].imshow(
    preview
)

axes[3].set_title(
    "4. Candidate Metric Grid\n"
    "600-mm diagnostic spacing"
)

axes[3].axis(
    "off"
)


plt.tight_layout()


AUDIT_PATH = (
    OUT
    / "03_stage08f1_corner_geometry_audit.png"
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
        "08F1",

    "purpose":
        "CORNER_BASED_RECTANGULAR_FLOOR_MODEL",

    "metric_mapping": {

        "back_corner":
            [
                0.0,
                0.0
            ],

        "left_direction_mm":
            LEFT_AXIS_MM,

        "right_direction_mm":
            RIGHT_AXIS_MM
    },

    "image_corners": {

        "back":
            BACK.tolist(),

        "left":
            LEFT.tolist(),

        "right":
            RIGHT.tolist(),

        "front_affine_initial":
            FRONT_AFFINE.tolist(),

        "front_candidate":
            FRONT.tolist()
    },

    "homography_metric_to_image":
        H_METRIC_TO_IMAGE.tolist(),

    "important_note":
        (
            "08F1 is a geometry audit only. "
            "The fourth/front floor corner is estimated and "
            "must be visually accepted before tile projection."
        ),

    "outputs": {

        "corner_model":
            str(
                CORNER_PATH
            ),

        "grid_audit":
            str(
                GRID_PATH
            ),

        "audit":
            str(
                AUDIT_PATH
            )
    },

    "status":
        "REQUIRES_FRONT_CORNER_AND_GRID_VISUAL_AUDIT"
}


STATE_PATH = (
    OUT
    / "00_stage08f1_result.json"
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
print("STAGE 08F1 RESULT")
print("=" * 110)

print()

print(
    "BACK:",
    [
        round(
            float(x),
            2
        )
        for x in BACK
    ]
)

print(
    "LEFT (2400-mm axis):",
    [
        round(
            float(x),
            2
        )
        for x in LEFT
    ]
)

print(
    "RIGHT (1800-mm axis):",
    [
        round(
            float(x),
            2
        )
        for x in RIGHT
    ]
)

print(
    "FRONT AFFINE INITIAL:",
    [
        round(
            float(x),
            2
        )
        for x in FRONT_AFFINE
    ]
)

print(
    "FRONT CANDIDATE:",
    [
        round(
            float(x),
            2
        )
        for x in FRONT
    ]
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
    "NO TILE WAS APPLIED."
)
