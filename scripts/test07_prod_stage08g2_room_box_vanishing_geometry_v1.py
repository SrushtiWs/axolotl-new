
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

G1_STATE_PATH = (
    PROD
    / "stage08_tile_application"
    / "08g1_room_box_geometry"
    / "00_stage08g1_result.json"
)

OUT = (
    PROD
    / "stage08_tile_application"
    / "08g2_room_box_vanishing_geometry"
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
    G1_STATE_PATH,
]:

    if not p.exists():
        raise FileNotFoundError(p)


room = Image.open(
    ROOM_PATH
).convert("RGB")

room_np = np.asarray(room)

H, W = room_np.shape[:2]


state = json.loads(
    G1_STATE_PATH.read_text(
        encoding="utf-8"
    )
)


lines = state[
    "fitted_lines"
]


floor_left = np.asarray(
    lines[
        "floor_left"
    ],
    dtype=np.float64
)

floor_right = np.asarray(
    lines[
        "floor_right"
    ],
    dtype=np.float64
)

ceiling_left = np.asarray(
    lines[
        "ceiling_left"
    ],
    dtype=np.float64
)

ceiling_right = np.asarray(
    lines[
        "ceiling_right"
    ],
    dtype=np.float64
)


back = state[
    "back_vertical"
]

floor_corner = np.asarray(
    back[
        "floor_corner"
    ],
    dtype=np.float64
)

ceiling_corner = np.asarray(
    back[
        "ceiling_corner"
    ],
    dtype=np.float64
)


# ============================================================
# y = m*x + b
# ->
# homogeneous image line:
#
# m*x - y + b = 0
# ============================================================

def coef_to_line(coef):

    m, b = map(
        float,
        coef
    )

    line = np.array(
        [
            m,
            -1.0,
            b
        ],
        dtype=np.float64
    )

    norm = math.hypot(
        line[0],
        line[1]
    )

    line /= norm

    return line


def intersect_lines(
    l1,
    l2
):

    p = np.cross(
        l1,
        l2
    )

    if abs(
        p[2]
    ) < 1e-10:

        raise RuntimeError(
            "Lines are numerically parallel."
        )

    return (
        p[:2]
        /
        p[2]
    )


LF = coef_to_line(
    floor_left
)

LC = coef_to_line(
    ceiling_left
)

RF = coef_to_line(
    floor_right
)

RC = coef_to_line(
    ceiling_right
)


# ============================================================
# TRUE ROOM-DIRECTION VANISHING POINTS
#
# 2400-mm physical axis:
# left wall/floor + left wall/ceiling
#
# 1800-mm physical axis:
# right wall/floor + right wall/ceiling
# ============================================================

VP_2400 = intersect_lines(
    LF,
    LC
)

VP_1800 = intersect_lines(
    RF,
    RC
)


# ============================================================
# FLOOR HORIZON
# ============================================================

v1_h = np.array(
    [
        VP_2400[0],
        VP_2400[1],
        1.0
    ],
    dtype=np.float64
)

v2_h = np.array(
    [
        VP_1800[0],
        VP_1800[1],
        1.0
    ],
    dtype=np.float64
)


horizon = np.cross(
    v1_h,
    v2_h
)


norm = math.hypot(
    horizon[0],
    horizon[1]
)


if norm > 1e-9:

    horizon /= norm


# ============================================================
# DIAGNOSTIC:
# check floor/ceiling rays against their calculated VP
# ============================================================

def point_line_distance(
    point,
    line
):

    x, y = point

    return abs(
        line[0] * x
        +
        line[1] * y
        +
        line[2]
    )


residuals = {

    "vp2400_to_floor_left":
        point_line_distance(
            VP_2400,
            LF
        ),

    "vp2400_to_ceiling_left":
        point_line_distance(
            VP_2400,
            LC
        ),

    "vp1800_to_floor_right":
        point_line_distance(
            VP_1800,
            RF
        ),

    "vp1800_to_ceiling_right":
        point_line_distance(
            VP_1800,
            RC
        ),
}


# ============================================================
# DRAW INFINITE LINES THROUGH BACK FLOOR CORNER + VPs
# ============================================================

preview = room_np.copy()


def draw_infinite_line(
    image,
    point_a,
    point_b,
    color,
    thickness=2
):

    p = np.asarray(
        point_a,
        dtype=np.float64
    )

    q = np.asarray(
        point_b,
        dtype=np.float64
    )


    direction = (
        q - p
    )


    length = np.linalg.norm(
        direction
    )


    if length < 1e-9:
        return


    direction /= length


    p1 = (
        p
        -
        direction
        *
        10000.0
    )

    p2 = (
        p
        +
        direction
        *
        10000.0
    )


    pt1 = tuple(
        np.round(
            p1
        ).astype(int)
    )

    pt2 = tuple(
        np.round(
            p2
        ).astype(int)
    )


    ok, c1, c2 = cv2.clipLine(
        (
            0,
            0,
            W,
            H
        ),
        pt1,
        pt2
    )


    if ok:

        cv2.line(
            image,
            c1,
            c2,
            color,
            thickness,
            cv2.LINE_AA
        )


# floor axes
draw_infinite_line(
    preview,
    floor_corner,
    VP_2400,
    (
        255,
        0,
        0
    ),
    2
)


draw_infinite_line(
    preview,
    floor_corner,
    VP_1800,
    (
        0,
        255,
        255
    ),
    2
)


# ceiling counterparts
draw_infinite_line(
    preview,
    ceiling_corner,
    VP_2400,
    (
        255,
        128,
        0
    ),
    1
)


draw_infinite_line(
    preview,
    ceiling_corner,
    VP_1800,
    (
        0,
        255,
        0
    ),
    1
)


# back vertical
cv2.line(
    preview,
    tuple(
        np.round(
            ceiling_corner
        ).astype(int)
    ),
    tuple(
        np.round(
            floor_corner
        ).astype(int)
    ),
    (
        255,
        0,
        255
    ),
    2,
    cv2.LINE_AA
)


# ============================================================
# HORIZON PREVIEW
# ============================================================

horizon_preview = preview.copy()


a, b, c = horizon


if abs(b) > 1e-9:

    y0 = (
        -c
        -
        a * 0
    ) / b


    y1 = (
        -c
        -
        a * (W - 1)
    ) / b


    pt1 = (
        0,
        int(
            round(
                y0
            )
        )
    )

    pt2 = (
        W - 1,
        int(
            round(
                y1
            )
        )
    )


    ok, c1, c2 = cv2.clipLine(
        (
            0,
            0,
            W,
            H
        ),
        pt1,
        pt2
    )


    if ok:

        cv2.line(
            horizon_preview,
            c1,
            c2,
            (
                255,
                0,
                255
            ),
            2,
            cv2.LINE_AA
        )


# ============================================================
# EXPANDED VP PLOT
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(
        17,
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
    horizon_preview
)

axes[1].set_title(
    "2. Room Directions\n"
    "Blue=2400 | Cyan=1800"
)

axes[1].axis(
    "off"
)


axes[2].imshow(
    room,
    alpha=0.25
)


axes[2].scatter(
    [
        VP_2400[0],
        VP_1800[0]
    ],
    [
        VP_2400[1],
        VP_1800[1]
    ],
    s=80
)


axes[2].text(
    VP_2400[0],
    VP_2400[1],
    " VP_2400"
)


axes[2].text(
    VP_1800[0],
    VP_1800[1],
    " VP_1800"
)


all_x = [
    0,
    W,
    VP_2400[0],
    VP_1800[0]
]

all_y = [
    0,
    H,
    VP_2400[1],
    VP_1800[1]
]


xmin = max(
    -5000,
    min(all_x) - 100
)

xmax = min(
    5000,
    max(all_x) + 100
)

ymin = max(
    -5000,
    min(all_y) - 100
)

ymax = min(
    5000,
    max(all_y) + 100
)


axes[2].set_xlim(
    xmin,
    xmax
)

axes[2].set_ylim(
    ymax,
    ymin
)

axes[2].set_title(
    "3. True Architectural VPs"
)


plt.tight_layout()


AUDIT_PATH = (
    OUT
    / "01_stage08g2_vanishing_geometry_audit.png"
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
        "08G2",

    "purpose":
        "ROOM_BOX_HORIZONTAL_VANISHING_GEOMETRY",

    "metric_axis_mapping": {

        "vp_2400_axis":
            "left-wall floor direction",

        "vp_1800_axis":
            "right-wall floor direction"
    },

    "vanishing_points": {

        "vp_2400":
            VP_2400.tolist(),

        "vp_1800":
            VP_1800.tolist()
    },

    "floor_horizon_abc":
        horizon.tolist(),

    "back_floor_corner":
        floor_corner.tolist(),

    "back_ceiling_corner":
        ceiling_corner.tolist(),

    "residuals":
        residuals,

    "important_note":
        (
            "Vanishing geometry establishes perspective "
            "directions but not absolute millimetre scale "
            "along a cropped floor axis."
        ),

    "outputs": {

        "audit":
            str(
                AUDIT_PATH
            )
    },

    "status":
        "REQUIRES_ARCHITECTURAL_VP_AUDIT",

    "next_if_pass":
        "METRIC_ANCHOR_REQUIRED_BEFORE_TRUE_600x1200_GRID"
}


STATE_PATH = (
    OUT
    / "00_stage08g2_result.json"
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
print("STAGE 08G2 RESULT")
print("=" * 110)

print()

print(
    "VP_2400:",
    [
        round(
            float(x),
            2
        )
        for x in VP_2400
    ]
)

print(
    "VP_1800:",
    [
        round(
            float(x),
            2
        )
        for x in VP_1800
    ]
)

print()

print(
    "HORIZON:",
    [
        round(
            float(x),
            6
        )
        for x in horizon
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

print(
    "NO CROPPED IMAGE EDGE WAS ASSIGNED A PHYSICAL LENGTH."
)
