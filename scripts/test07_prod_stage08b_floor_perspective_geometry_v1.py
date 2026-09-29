
from pathlib import Path
import json

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
    / "08a2_stage07_surface_consensus"
    / "17_floor_majority_2of3.png"
)

OUT = (
    PROD
    / "stage08_tile_application"
    / "08b_floor_perspective_geometry"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


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

W, H = room.size


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
        "Floor mask dimensions do not match room."
    )


print("=" * 110)
print("STAGE 08B — FLOOR PERSPECTIVE GEOMETRY")
print("=" * 110)

print()
print("ROOM:", ROOM_PATH)
print("FLOOR MASK:", FLOOR_MASK_PATH)
print("SIZE:", f"{W} × {H}")


# ============================================================
# KEEP MAIN FLOOR COMPONENT
# ============================================================

num_labels, labels, stats, _ = (
    cv2.connectedComponentsWithStats(
        floor.astype(np.uint8),
        connectivity=8
    )
)


if num_labels <= 1:
    raise RuntimeError(
        "No floor connected component."
    )


areas = stats[
    1:,
    cv2.CC_STAT_AREA
]


largest_label = (
    1
    +
    int(
        np.argmax(areas)
    )
)


floor_main = (
    labels
    ==
    largest_label
)


# ============================================================
# EXTRACT MAIN CONTOUR
# ============================================================

contours, _ = cv2.findContours(
    (
        floor_main.astype(np.uint8)
        *
        255
    ),
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_NONE
)


contour = max(
    contours,
    key=cv2.contourArea
)


# ============================================================
# FIND UPPER FLOOR ENVELOPE
#
# For each x column:
# first floor pixel from top = visible wall-floor boundary.
# ============================================================

upper_y = np.full(
    W,
    np.nan,
    dtype=np.float32
)


for x in range(W):

    ys = np.where(
        floor_main[:, x]
    )[0]

    if len(ys):
        upper_y[x] = float(
            ys.min()
        )


valid = np.isfinite(
    upper_y
)


# ============================================================
# REMOVE EXTREME SIDE COLUMNS
#
# Side edges can contain door jamb / border artifacts.
# ============================================================

valid_x = np.where(
    valid
)[0]


if len(valid_x) < 30:
    raise RuntimeError(
        "Too few valid floor-boundary columns."
    )


xmin = int(
    valid_x.min()
)

xmax = int(
    valid_x.max()
)


trim = max(
    3,
    int(
        0.04
        *
        (
            xmax - xmin
        )
    )
)


usable = (
    valid
    &
    (
        np.arange(W)
        >=
        xmin + trim
    )
    &
    (
        np.arange(W)
        <=
        xmax - trim
    )
)


xs = np.where(
    usable
)[0].astype(
    np.float32
)


ys = upper_y[
    usable
].astype(
    np.float32
)


# ============================================================
# SMOOTH UPPER ENVELOPE
# ============================================================

if len(ys) >= 9:

    smooth_y = cv2.GaussianBlur(
        ys.reshape(1, -1),
        (0, 0),
        sigmaX=2.0
    ).reshape(-1)

else:
    smooth_y = ys.copy()


# ============================================================
# FIND BACK-CORNER / APEX
#
# In this bathroom, wall-floor boundary consists of two
# dominant segments meeting near the room back corner.
#
# We estimate the highest-confidence change in slope.
# ============================================================

dy = np.gradient(
    smooth_y
)


# left half should generally slope upward toward corner;
# right half should slope away from corner.
#
# Find the minimum Y of the smoothed upper boundary first.
corner_local = int(
    np.argmin(
        smooth_y
    )
)


corner_x = float(
    xs[
        corner_local
    ]
)

corner_y = float(
    smooth_y[
        corner_local
    ]
)


# ============================================================
# FIT LEFT AND RIGHT BACK-BOUNDARY SEGMENTS
# ============================================================

MIN_SIDE_POINTS = 20


left_sel = (
    xs
    <
    corner_x - 4
)


right_sel = (
    xs
    >
    corner_x + 4
)


if (
    left_sel.sum()
    <
    MIN_SIDE_POINTS
    or
    right_sel.sum()
    <
    MIN_SIDE_POINTS
):

    raise RuntimeError(
        "Not enough points on both sides of floor corner."
    )


left_x = xs[
    left_sel
]

left_y = smooth_y[
    left_sel
]


right_x = xs[
    right_sel
]

right_y = smooth_y[
    right_sel
]


# Robust-ish initial linear fits.
left_coef = np.polyfit(
    left_x,
    left_y,
    1
)

right_coef = np.polyfit(
    right_x,
    right_y,
    1
)


# ============================================================
# ITERATIVE RESIDUAL TRIM
# ============================================================

def robust_line_fit(
    x,
    y,
    iterations=4,
    keep_sigma=2.0
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
            coef[0]
            *
            x
            +
            coef[1]
        )

        residual = (
            y
            -
            pred
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
            1.0,
            1.4826 * mad
        )

        keep = (
            np.abs(
                residual - med
            )
            <=
            keep_sigma * sigma
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


left_coef, left_keep = robust_line_fit(
    left_x,
    left_y
)

right_coef, right_keep = robust_line_fit(
    right_x,
    right_y
)


# ============================================================
# INTERSECTION OF BACK-BOUNDARY LINES
# ============================================================

m1, b1 = left_coef
m2, b2 = right_coef


if abs(
    m1 - m2
) < 1e-6:

    line_corner = [
        corner_x,
        corner_y
    ]

else:

    ix = (
        b2 - b1
    ) / (
        m1 - m2
    )

    iy = (
        m1 * ix
        +
        b1
    )

    line_corner = [
        float(ix),
        float(iy)
    ]


# ============================================================
# SAVE CLEAN MAIN FLOOR MASK
# ============================================================

MAIN_MASK_PATH = (
    OUT
    / "01_stage08b_main_floor_mask.png"
)


Image.fromarray(
    floor_main.astype(
        np.uint8
    )
    *
    255
).save(
    MAIN_MASK_PATH
)


# ============================================================
# SAVE UPPER BOUNDARY POINTS
# ============================================================

boundary_vis = room_np.copy()


for x, y in zip(
    xs.astype(int),
    smooth_y.astype(int)
):

    if (
        0 <= x < W
        and
        0 <= y < H
    ):

        boundary_vis[
            y,
            x
        ] = [
            255,
            0,
            0
        ]


BOUNDARY_PATH = (
    OUT
    / "02_upper_floor_boundary.png"
)


Image.fromarray(
    boundary_vis
).save(
    BOUNDARY_PATH
)


# ============================================================
# GEOMETRY AUDIT
# ============================================================

fig, axes = plt.subplots(
    1,
    4,
    figsize=(
        18,
        7
    )
)


# -------------------------
# 1 room
# -------------------------

axes[0].imshow(
    room
)

axes[0].set_title(
    "1. Stage07A"
)

axes[0].axis(
    "off"
)


# -------------------------
# 2 clean mask
# -------------------------

axes[1].imshow(
    floor_main,
    cmap="gray"
)

axes[1].set_title(
    "2. 08A2 Main Floor Mask"
)

axes[1].axis(
    "off"
)


# -------------------------
# 3 upper envelope
# -------------------------

axes[2].imshow(
    room
)

axes[2].plot(
    xs,
    smooth_y,
    linewidth=2
)

axes[2].scatter(
    [
        corner_x
    ],
    [
        corner_y
    ],
    s=50
)

axes[2].set_title(
    "3. Extracted Wall-Floor Boundary"
)

axes[2].axis(
    "off"
)


# -------------------------
# 4 fitted perspective lines
# -------------------------

axes[3].imshow(
    room
)


lx = np.array(
    [
        left_x.min(),
        corner_x
    ]
)


ly = (
    left_coef[0]
    *
    lx
    +
    left_coef[1]
)


rx = np.array(
    [
        corner_x,
        right_x.max()
    ]
)


ry = (
    right_coef[0]
    *
    rx
    +
    right_coef[1]
)


axes[3].plot(
    lx,
    ly,
    linewidth=2
)

axes[3].plot(
    rx,
    ry,
    linewidth=2
)


axes[3].scatter(
    [
        line_corner[0]
    ],
    [
        line_corner[1]
    ],
    s=60
)


axes[3].set_title(
    "4. Back Floor Perspective Segments"
)

axes[3].axis(
    "off"
)


plt.tight_layout()


AUDIT_PATH = (
    OUT
    / "03_stage08b_geometry_audit.png"
)


plt.savefig(
    AUDIT_PATH,
    dpi=150,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# SAVE STATE
# ============================================================

STATE = {

    "stage":
        "08B",

    "purpose":
        "FLOOR_PERSPECTIVE_GEOMETRY_ONLY",

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

    "image_size": [
        W,
        H
    ],

    "geometry": {

        "visible_floor_x_range": [
            xmin,
            xmax
        ],

        "boundary_corner_from_envelope": [
            corner_x,
            corner_y
        ],

        "left_line": {

            "slope":
                float(
                    left_coef[0]
                ),

            "intercept":
                float(
                    left_coef[1]
                )
        },

        "right_line": {

            "slope":
                float(
                    right_coef[0]
                ),

            "intercept":
                float(
                    right_coef[1]
                )
        },

        "line_intersection_corner": 
            line_corner
    },

    "important_note":
        (
            "08B establishes projective floor geometry only. "
            "No millimetre scale has been assigned yet."
        ),

    "outputs": {

        "main_floor_mask":
            str(
                MAIN_MASK_PATH
            ),

        "upper_boundary":
            str(
                BOUNDARY_PATH
            ),

        "audit":
            str(
                AUDIT_PATH
            )
    },

    "status":
        "REQUIRES_VISUAL_GEOMETRY_AUDIT",

    "next_if_pass":
        "08C_METRIC_SCALE_CALIBRATION_AND_600x1200_GRID"
}


STATE_PATH = (
    OUT
    / "00_stage08b_result.json"
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
print("STAGE 08B RESULT")
print("=" * 110)

print()
print(
    "BOUNDARY CORNER:",
    [
        round(
            corner_x,
            2
        ),
        round(
            corner_y,
            2
        )
    ]
)

print(
    "LINE INTERSECTION:",
    [
        round(
            line_corner[0],
            2
        ),
        round(
            line_corner[1],
            2
        )
    ]
)

print()
print(
    "LEFT SLOPE:",
    round(
        float(
            left_coef[0]
        ),
        5
    )
)

print(
    "RIGHT SLOPE:",
    round(
        float(
            right_coef[0]
        ),
        5
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
    "NO TILE WAS APPLIED."
)

print(
    "NO METRIC SCALE WAS ASSUMED."
)
