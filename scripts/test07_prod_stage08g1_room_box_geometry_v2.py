
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

WALL_PATH = (
    PROD
    / "stage08_tile_application"
    / "08a2_stage07_surface_consensus"
    / "13_wall_consensus_3of3.png"
)

FLOOR_PATH = (
    PROD
    / "stage08_tile_application"
    / "08a2_stage07_surface_consensus"
    / "14_floor_consensus_3of3.png"
)

CEILING_PATH = (
    PROD
    / "stage08_tile_application"
    / "08a2_stage07_surface_consensus"
    / "15_ceiling_consensus_3of3.png"
)

OUT = (
    PROD
    / "stage08_tile_application"
    / "08g1_room_box_geometry"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOCKED METRIC ROOM
# ============================================================

LEFT_FLOOR_AXIS_MM = 2400.0
RIGHT_FLOOR_AXIS_MM = 1800.0


# ============================================================
# LOAD
# ============================================================

for p in [
    ROOM_PATH,
    WALL_PATH,
    FLOOR_PATH,
    CEILING_PATH,
]:

    if not p.exists():
        raise FileNotFoundError(p)


room = Image.open(
    ROOM_PATH
).convert("RGB")

room_np = np.asarray(room)

H, W = room_np.shape[:2]


def load_mask(path):

    return (
        np.asarray(
            Image.open(
                path
            ).convert("L")
        )
        > 127
    )


wall = load_mask(
    WALL_PATH
)

floor = load_mask(
    FLOOR_PATH
)

ceiling = load_mask(
    CEILING_PATH
)


print("=" * 110)
print("STAGE 08G1 — MANHATTAN ROOM-BOX GEOMETRY")
print("=" * 110)

print()
print(
    "ROOM:",
    ROOM_PATH
)

print(
    "SIZE:",
    f"{W} × {H}"
)


# ============================================================
# FLOOR UPPER BOUNDARY
# ============================================================

floor_y = np.full(
    W,
    np.nan,
    dtype=np.float64
)


for x in range(W):

    ys = np.where(
        floor[:, x]
    )[0]

    if len(ys):

        floor_y[x] = float(
            ys.min()
        )


floor_valid = np.where(
    np.isfinite(
        floor_y
    )
)[0]


# ============================================================
# CEILING LOWER BOUNDARY
#
# last ceiling pixel from top in each x-column
# ============================================================

ceiling_y = np.full(
    W,
    np.nan,
    dtype=np.float64
)


for x in range(W):

    ys = np.where(
        ceiling[:, x]
    )[0]

    if len(ys):

        ceiling_y[x] = float(
            ys.max()
        )


ceiling_valid = np.where(
    np.isfinite(
        ceiling_y
    )
)[0]


# ============================================================
# ROBUST POLYLINE FIT HELPER
# ============================================================

def robust_line(
    x,
    y,
    iterations=6
):

    x = np.asarray(
        x,
        dtype=np.float64
    )

    y = np.asarray(
        y,
        dtype=np.float64
    )


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
            0.7,
            1.4826 * mad
        )


        new_keep = (
            np.abs(
                residual - med
            )
            <=
            2.2 * sigma
        )


        if new_keep.sum() < 8:

            break


        keep = new_keep


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


# ============================================================
# BACK FLOOR CORNER
#
# highest point of V-shaped floor boundary
# ============================================================

fx = floor_valid.astype(
    np.float64
)

fy = floor_y[
    floor_valid
]


fy_smooth = cv2.GaussianBlur(
    fy.reshape(1, -1),
    (0, 0),
    sigmaX=2.0
).reshape(-1)


floor_apex_idx = int(
    np.argmin(
        fy_smooth
    )
)


floor_corner = np.array(
    [
        fx[
            floor_apex_idx
        ],
        fy_smooth[
            floor_apex_idx
        ]
    ],
    dtype=np.float64
)


# ============================================================
# BACK CEILING CORNER
#
# ceiling boundary has inverted V:
# the central room corner is its LOWEST point.
# ============================================================

cx = ceiling_valid.astype(
    np.float64
)

cy = ceiling_y[
    ceiling_valid
]


cy_smooth = cv2.GaussianBlur(
    cy.reshape(1, -1),
    (0, 0),
    sigmaX=2.0
).reshape(-1)


ceiling_apex_idx = int(
    np.argmax(
        cy_smooth
    )
)


ceiling_corner = np.array(
    [
        cx[
            ceiling_apex_idx
        ],
        cy_smooth[
            ceiling_apex_idx
        ]
    ],
    dtype=np.float64
)


# ============================================================
# FORCE SHARED BACK VERTICAL X ESTIMATE
#
# Average the floor and ceiling corner x positions.
# ============================================================

back_x = float(
    (
        floor_corner[0]
        +
        ceiling_corner[0]
    )
    /
    2.0
)


floor_corner[0] = back_x
ceiling_corner[0] = back_x


# ============================================================
# FIT FLOOR LEFT/RIGHT RAYS
# ============================================================

FLOOR_GAP = 5


left_floor_sel = (
    fx
    <
    back_x - FLOOR_GAP
)


right_floor_sel = (
    fx
    >
    back_x + FLOOR_GAP
)


left_floor_coef, lf_keep = robust_line(
    fx[
        left_floor_sel
    ],
    fy_smooth[
        left_floor_sel
    ]
)


right_floor_coef, rf_keep = robust_line(
    fx[
        right_floor_sel
    ],
    fy_smooth[
        right_floor_sel
    ]
)


# ============================================================
# FIT CEILING LEFT/RIGHT RAYS
# ============================================================

CEIL_GAP = 5


left_ceil_sel = (
    cx
    <
    back_x - CEIL_GAP
)


right_ceil_sel = (
    cx
    >
    back_x + CEIL_GAP
)


left_ceil_coef, lc_keep = robust_line(
    cx[
        left_ceil_sel
    ],
    cy_smooth[
        left_ceil_sel
    ]
)


right_ceil_coef, rc_keep = robust_line(
    cx[
        right_ceil_sel
    ],
    cy_smooth[
        right_ceil_sel
    ]
)


# ============================================================
# LINE INTERSECTION HELPER
# ============================================================

def line_intersection(
    c1,
    c2
):

    m1, b1 = c1
    m2, b2 = c2


    denom = (
        m1 - m2
    )


    if abs(
        denom
    ) < 1e-9:

        return None


    x = (
        b2 - b1
    ) / denom


    y = (
        m1 * x
        +
        b1
    )


    return np.array(
        [
            x,
            y
        ],
        dtype=np.float64
    )


floor_line_corner = line_intersection(
    left_floor_coef,
    right_floor_coef
)


ceiling_line_corner = line_intersection(
    left_ceil_coef,
    right_ceil_coef
)


# ============================================================
# ARCHITECTURAL LINE ANGLES
# ============================================================

def line_angle(coef):

    return math.degrees(
        math.atan(
            float(
                coef[0]
            )
        )
    )


ANGLES = {

    "floor_left":
        line_angle(
            left_floor_coef
        ),

    "floor_right":
        line_angle(
            right_floor_coef
        ),

    "ceiling_left":
        line_angle(
            left_ceil_coef
        ),

    "ceiling_right":
        line_angle(
            right_ceil_coef
        )
}


# ============================================================
# VISUAL EVIDENCE
# ============================================================

preview = room_np.copy()


def draw_fitted_line(
    image,
    coef,
    x1,
    x2,
    color,
    thickness=2
):

    m, b = coef


    p1 = (
        int(
            round(
                x1
            )
        ),
        int(
            round(
                m * x1
                +
                b
            )
        )
    )


    p2 = (
        int(
            round(
                x2
            )
        ),
        int(
            round(
                m * x2
                +
                b
            )
        )
    )


    cv2.line(
        image,
        p1,
        p2,
        color,
        thickness,
        cv2.LINE_AA
    )


# floor rays
draw_fitted_line(
    preview,
    left_floor_coef,
    0,
    back_x,
    (
        255,
        0,
        0
    )
)


draw_fitted_line(
    preview,
    right_floor_coef,
    back_x,
    W - 1,
    (
        0,
        255,
        255
    )
)


# ceiling rays
draw_fitted_line(
    preview,
    left_ceil_coef,
    0,
    back_x,
    (
        255,
        128,
        0
    )
)


draw_fitted_line(
    preview,
    right_ceil_coef,
    back_x,
    W - 1,
    (
        0,
        255,
        0
    )
)


# vertical back corner
cv2.line(
    preview,
    (
        int(
            round(
                back_x
            )
        ),
        int(
            round(
                ceiling_corner[1]
            )
        )
    ),
    (
        int(
            round(
                back_x
            )
        ),
        int(
            round(
                floor_corner[1]
            )
        )
    ),
    (
        255,
        0,
        255
    ),
    2,
    cv2.LINE_AA
)


# Corner markers
for label, p, color in [

    (
        "FC",
        floor_corner,
        (
            255,
            0,
            255
        )
    ),

    (
        "CC",
        ceiling_corner,
        (
            255,
            0,
            255
        )
    ),

]:

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


    if (
        0 <= px < W
        and
        0 <= py < H
    ):

        cv2.circle(
            preview,
            (
                px,
                py
            ),
            4,
            color,
            -1
        )


        cv2.putText(
            preview,
            label,
            (
                px + 5,
                py - 5
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA
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
    floor,
    cmap="gray"
)

axes[1].set_title(
    "2. Floor Consensus"
)

axes[1].axis(
    "off"
)


axes[2].imshow(
    ceiling,
    cmap="gray"
)

axes[2].set_title(
    "3. Ceiling Consensus"
)

axes[2].axis(
    "off"
)


axes[3].imshow(
    preview
)

axes[3].set_title(
    "4. Room-Box Evidence\n"
    "floor rays + ceiling rays + back vertical"
)

axes[3].axis(
    "off"
)


plt.tight_layout()


AUDIT_PATH = (
    OUT
    / "01_stage08g1_room_box_audit.png"
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
        "08G1",

    "purpose":
        "MANHATTAN_ROOM_BOX_GEOMETRY_EVIDENCE",

    "metric_room": {

        "left_floor_axis_mm":
            LEFT_FLOOR_AXIS_MM,

        "right_floor_axis_mm":
            RIGHT_FLOOR_AXIS_MM
    },

    "back_vertical": {

        "x":
            back_x,

        "ceiling_corner":
            ceiling_corner.tolist(),

        "floor_corner":
            floor_corner.tolist()
    },

    "fitted_lines": {

        "floor_left":
            left_floor_coef.tolist(),

        "floor_right":
            right_floor_coef.tolist(),

        "ceiling_left":
            left_ceil_coef.tolist(),

        "ceiling_right":
            right_ceil_coef.tolist()
    },

    "angles_deg":
        ANGLES,

    "line_intersections": {

        "floor":
            (
                None
                if floor_line_corner is None
                else floor_line_corner.tolist()
            ),

        "ceiling":
            (
                None
                if ceiling_line_corner is None
                else ceiling_line_corner.tolist()
            )
    },

    "important_rule":
        (
            "Image-bottom intersections are NOT treated as "
            "physical room corners."
        ),

    "outputs": {

        "audit":
            str(
                AUDIT_PATH
            )
    },

    "status":
        "REQUIRES_ROOM_BOX_EVIDENCE_AUDIT",

    "next_if_pass":
        "08G2_CAMERA_AND_RECTANGULAR_ROOM_FIT"
}


STATE_PATH = (
    OUT
    / "00_stage08g1_result.json"
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
print("STAGE 08G1 RESULT")
print("=" * 110)

print()

print(
    "BACK FLOOR CORNER:",
    [
        round(
            float(x),
            2
        )
        for x in floor_corner
    ]
)

print(
    "BACK CEILING CORNER:",
    [
        round(
            float(x),
            2
        )
        for x in ceiling_corner
    ]
)

print()

for key, value in ANGLES.items():

    print(
        key.upper(),
        "=",
        round(
            value,
            3
        ),
        "deg"
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
    "NO IMAGE-BOTTOM POINT WAS ASSIGNED A METRIC DIMENSION."
)

print(
    "NO TILE WAS APPLIED."
)
