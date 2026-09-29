
from pathlib import Path
import json

import cv2
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE = Path(
    "/workspace/axolotl"
)

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


SCREEDING_MASK_PATH = (
    PROD
    / "stage08_tile_application"
    / "08b2_floor_plus_screeding_target"
    / "02_wall_screeding_mask.png"
)


TILE_PATH = (
    BASE
    / "test07"
    / "inputs"
    / "tile 01.png"
)


OUT = (
    PROD
    / "stage08_tile_application"
    / "08d_metric_floor_projection_v1"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOCKED METRIC SPEC
# ============================================================

ROOM_W_MM = 2400.0
ROOM_L_MM = 1800.0

TILE_W_MM = 600.0
TILE_L_MM = 1200.0

GROUT_MM = 5.0

ORIENTATION = "VERTICAL"
SURFACE = "GLOSSY"

# provisional default screeding height
SCREED_H_MM = 100.0


# ============================================================
# TILE LAYOUT
#
# Exact fit including grout:
#
# WIDTH:
# 600
# 5 grout
# 600
# 5 grout
# 600
# 5 grout
# 585 cut
#
# LENGTH:
# 1200
# 5 grout
# 595 cut
# ============================================================

X_TILE_INTERVALS = [
    (0.0,    600.0, 0),
    (605.0, 1205.0, 1),
    (1210.0, 1810.0, 2),
    (1815.0, 2400.0, 3),
]

X_GROUT_INTERVALS = [
    (600.0, 605.0),
    (1205.0, 1210.0),
    (1810.0, 1815.0),
]

Y_TILE_INTERVALS = [
    (0.0,    1200.0, 0),
    (1205.0, 1800.0, 1),
]

Y_GROUT_INTERVALS = [
    (1200.0, 1205.0),
]


# ============================================================
# VALIDATE INPUTS
# ============================================================

for p in [
    ROOM_PATH,
    FLOOR_MASK_PATH,
    SCREEDING_MASK_PATH,
    TILE_PATH,
]:

    if not p.exists():

        raise FileNotFoundError(
            p
        )


# ============================================================
# LOAD
# ============================================================

room = Image.open(
    ROOM_PATH
).convert(
    "RGB"
)


tile = Image.open(
    TILE_PATH
).convert(
    "RGB"
)


room_np = np.asarray(
    room
).astype(
    np.uint8
)


tile_np = np.asarray(
    tile
).astype(
    np.uint8
)


H, W = room_np.shape[:2]


floor = (
    np.asarray(
        Image.open(
            FLOOR_MASK_PATH
        ).convert(
            "L"
        )
    )
    >
    127
)


screeding = (
    np.asarray(
        Image.open(
            SCREEDING_MASK_PATH
        ).convert(
            "L"
        )
    )
    >
    127
)


if floor.shape != (
    H,
    W
):

    raise RuntimeError(
        "Floor mask size mismatch."
    )


if screeding.shape != (
    H,
    W
):

    raise RuntimeError(
        "Screeding mask size mismatch."
    )


print("=" * 110)
print("STAGE 08D V1 — METRIC FLOOR TILE PROJECTION")
print("=" * 110)

print()
print(
    "ROOM:",
    ROOM_PATH
)

print(
    "ROOM SIZE:",
    f"{W} × {H}"
)

print()
print(
    "ROOM METRIC:",
    "2400 × 1800 mm"
)

print(
    "TILE:",
    "600 × 1200 mm"
)

print(
    "ORIENTATION:",
    ORIENTATION
)

print(
    "GROUT:",
    "5 mm"
)

print(
    "SURFACE:",
    SURFACE
)


# ============================================================
# MAIN FLOOR COMPONENT
# ============================================================

num_labels, labels, stats, _ = (
    cv2.connectedComponentsWithStats(
        floor.astype(
            np.uint8
        ),
        connectivity=8
    )
)


if num_labels <= 1:

    raise RuntimeError(
        "No main floor component."
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
    labels
    ==
    largest
)


# ============================================================
# FIND VISIBLE FLOOR QUADRILATERAL
#
# BACK:
# derived from upper floor envelope
#
# FRONT:
# bottom visible floor intersections
#
# This is the V1 metric projection assumption.
# ============================================================

upper = np.full(
    W,
    np.nan,
    dtype=np.float32
)


for x in range(
    W
):

    ys = np.where(
        floor_main[
            :,
            x
        ]
    )[0]


    if len(
        ys
    ):

        upper[
            x
        ] = float(
            ys.min()
        )


valid_x = np.where(
    np.isfinite(
        upper
    )
)[0]


if len(
    valid_x
) < 20:

    raise RuntimeError(
        "Insufficient floor boundary."
    )


x_left = int(
    valid_x.min()
)

x_right = int(
    valid_x.max()
)


# slight side trim protects border artifacts
trim = max(
    2,
    int(
        0.015
        *
        (
            x_right
            -
            x_left
        )
    )
)


back_left_x = (
    x_left
    +
    trim
)

back_right_x = (
    x_right
    -
    trim
)


back_left_y = float(
    upper[
        back_left_x
    ]
)


back_right_y = float(
    upper[
        back_right_x
    ]
)


# ============================================================
# FRONT VISIBLE FLOOR EDGE
#
# Choose bottom-most visible floor pixels near left/right sides.
# ============================================================

def bottom_floor_y(
    x
):

    ys = np.where(
        floor_main[
            :,
            x
        ]
    )[0]


    if not len(
        ys
    ):

        return np.nan


    return float(
        ys.max()
    )


front_left_x = back_left_x
front_right_x = back_right_x


front_left_y = bottom_floor_y(
    front_left_x
)

front_right_y = bottom_floor_y(
    front_right_x
)


# Often floor touches the image bottom.
# Search a small side neighborhood for more robust endpoint.

def robust_bottom(
    center_x,
    radius=8
):

    values = []


    for xx in range(
        max(
            0,
            center_x - radius
        ),
        min(
            W,
            center_x + radius + 1
        )
    ):

        value = bottom_floor_y(
            xx
        )


        if np.isfinite(
            value
        ):

            values.append(
                value
            )


    if not values:

        return float(
            H - 1
        )


    return float(
        np.max(
            values
        )
    )


front_left_y = robust_bottom(
    front_left_x
)

front_right_y = robust_bottom(
    front_right_x
)


# ============================================================
# QUADRILATERAL
#
# Metric coordinates:
#
# back-left  = (0,0)
# back-right = (2400,0)
# front-right= (2400,1800)
# front-left = (0,1800)
# ============================================================

IMG_QUAD = np.array(
    [
        [
            back_left_x,
            back_left_y
        ],
        [
            back_right_x,
            back_right_y
        ],
        [
            front_right_x,
            front_right_y
        ],
        [
            front_left_x,
            front_left_y
        ],
    ],
    dtype=np.float32
)


METRIC_QUAD = np.array(
    [
        [
            0.0,
            0.0
        ],
        [
            ROOM_W_MM,
            0.0
        ],
        [
            ROOM_W_MM,
            ROOM_L_MM
        ],
        [
            0.0,
            ROOM_L_MM
        ],
    ],
    dtype=np.float32
)


# ============================================================
# HOMOGRAPHIES
# ============================================================

H_METRIC_TO_IMG = cv2.getPerspectiveTransform(
    METRIC_QUAD,
    IMG_QUAD
)


H_IMG_TO_METRIC = cv2.getPerspectiveTransform(
    IMG_QUAD,
    METRIC_QUAD
)


# ============================================================
# METRIC GRID DEBUG CANVAS
# ============================================================

DEBUG_SCALE = 0.25

debug_w = int(
    ROOM_W_MM
    *
    DEBUG_SCALE
)

debug_h = int(
    ROOM_L_MM
    *
    DEBUG_SCALE
)


metric_debug = np.full(
    (
        debug_h,
        debug_w,
        3
    ),
    240,
    dtype=np.uint8
)


def mx(
    value
):

    return int(
        round(
            value
            *
            DEBUG_SCALE
        )
    )


# tile cells
for x1, x2, ci in X_TILE_INTERVALS:

    for y1, y2, ri in Y_TILE_INTERVALS:

        cv2.rectangle(
            metric_debug,
            (
                mx(
                    x1
                ),
                mx(
                    y1
                )
            ),
            (
                max(
                    mx(
                        x2
                    )
                    -
                    1,
                    0
                ),
                max(
                    mx(
                        y2
                    )
                    -
                    1,
                    0
                )
            ),
            (
                210,
                210,
                210
            ),
            -1
        )


# grout lines
for gx1, gx2 in X_GROUT_INTERVALS:

    cv2.rectangle(
        metric_debug,
        (
            mx(
                gx1
            ),
            0
        ),
        (
            max(
                mx(
                    gx2
                ),
                mx(
                    gx1
                )
                +
                1
            ),
            debug_h - 1
        ),
        (
            80,
            80,
            80
        ),
        -1
    )


for gy1, gy2 in Y_GROUT_INTERVALS:

    cv2.rectangle(
        metric_debug,
        (
            0,
            mx(
                gy1
            )
        ),
        (
            debug_w - 1,
            max(
                mx(
                    gy2
                ),
                mx(
                    gy1
                )
                +
                1
            )
        ),
        (
            80,
            80,
            80
        ),
        -1
    )


METRIC_DEBUG_PATH = (
    OUT
    / "01_metric_2400x1800_layout.png"
)


Image.fromarray(
    metric_debug
).save(
    METRIC_DEBUG_PATH
)


# ============================================================
# HELPERS
# ============================================================

def in_interval(
    v,
    a,
    b
):

    return (
        v >= a
        and
        v < b
    )


def find_tile_interval(
    value,
    intervals
):

    for start, end, index in intervals:

        if (
            value >= start
            and
            value < end
        ):

            return (
                start,
                end,
                index
            )


    return None


def in_grout(
    value,
    grout_intervals
):

    for start, end in grout_intervals:

        if (
            value >= start
            and
            value < end
        ):

            return True


    return False


# ============================================================
# TILE SAMPLER
# ============================================================

tile_h, tile_w = tile_np.shape[:2]


def sample_tile(
    local_x_mm,
    local_y_mm
):

    # Exact 600×1200 source tile coordinate system.
    #
    # Vertical orientation:
    # source width  -> 600 mm
    # source height -> 1200 mm

    u = (
        local_x_mm
        /
        TILE_W_MM
    )

    v = (
        local_y_mm
        /
        TILE_L_MM
    )


    u = np.clip(
        u,
        0.0,
        0.999999
    )

    v = np.clip(
        v,
        0.0,
        0.999999
    )


    tx = int(
        u
        *
        tile_w
    )

    ty = int(
        v
        *
        tile_h
    )


    return tile_np[
        ty,
        tx
    ]


# ============================================================
# PROJECT FLOOR TILE
# ============================================================

raw_floor_tile = room_np.copy()


floor_tile_only = np.zeros_like(
    room_np
)


grout_mask = np.zeros(
    (
        H,
        W
    ),
    dtype=bool
)


tile_mask = np.zeros(
    (
        H,
        W
    ),
    dtype=bool
)


# neutral warm-light grout for first test
GROUT_RGB = np.array(
    [
        225,
        225,
        220
    ],
    dtype=np.uint8
)


ys_floor, xs_floor = np.where(
    floor_main
)


points_img = np.stack(
    [
        xs_floor.astype(
            np.float32
        ),
        ys_floor.astype(
            np.float32
        )
    ],
    axis=1
).reshape(
    -1,
    1,
    2
)


points_metric = cv2.perspectiveTransform(
    points_img,
    H_IMG_TO_METRIC
).reshape(
    -1,
    2
)


for (
    px,
    py,
    metric
) in zip(
    xs_floor,
    ys_floor,
    points_metric
):

    x_mm = float(
        metric[
            0
        ]
    )

    y_mm = float(
        metric[
            1
        ]
    )


    if not (
        0.0
        <=
        x_mm
        <=
        ROOM_W_MM
        and
        0.0
        <=
        y_mm
        <=
        ROOM_L_MM
    ):

        continue


    grout_x = in_grout(
        x_mm,
        X_GROUT_INTERVALS
    )

    grout_y = in_grout(
        y_mm,
        Y_GROUT_INTERVALS
    )


    if (
        grout_x
        or
        grout_y
    ):

        raw_floor_tile[
            py,
            px
        ] = GROUT_RGB

        floor_tile_only[
            py,
            px
        ] = GROUT_RGB

        grout_mask[
            py,
            px
        ] = True

        continue


    x_cell = find_tile_interval(
        x_mm,
        X_TILE_INTERVALS
    )

    y_cell = find_tile_interval(
        y_mm,
        Y_TILE_INTERVALS
    )


    if (
        x_cell is None
        or
        y_cell is None
    ):

        continue


    x_start = x_cell[
        0
    ]

    y_start = y_cell[
        0
    ]


    local_x = (
        x_mm
        -
        x_start
    )


    local_y = (
        y_mm
        -
        y_start
    )


    color = sample_tile(
        local_x,
        local_y
    )


    raw_floor_tile[
        py,
        px
    ] = color

    floor_tile_only[
        py,
        px
    ] = color

    tile_mask[
        py,
        px
    ] = True


# ============================================================
# PROJECT SAME MATERIAL ON SCREEDING
#
# We align screeding horizontally to the nearest floor-junction
# metric X coordinate.
#
# Vertical texture is treated as a 100-mm cut from same tile.
# ============================================================

combined_raw = raw_floor_tile.copy()


# get floor top per x for junction mapping
floor_top = np.full(
    W,
    np.nan,
    dtype=np.float32
)


for x in range(
    W
):

    ys = np.where(
        floor_main[
            :,
            x
        ]
    )[0]


    if len(
        ys
    ):

        floor_top[
            x
        ] = float(
            ys.min()
        )


screed_ys, screed_xs = np.where(
    screeding
)


# derive top/bottom screed limits column-wise
for px, py in zip(
    screed_xs,
    screed_ys
):

    if not np.isfinite(
        floor_top[
            px
        ]
    ):

        continue


    junction_y = float(
        floor_top[
            px
        ]
    )


    # Map the wall-floor junction pixel into metric X.
    pt = np.array(
        [
            [
                [
                    float(
                        px
                    ),
                    junction_y
                ]
            ]
        ],
        dtype=np.float32
    )


    metric_pt = cv2.perspectiveTransform(
        pt,
        H_IMG_TO_METRIC
    )[0, 0]


    x_mm = float(
        metric_pt[
            0
        ]
    )


    if not (
        0.0
        <=
        x_mm
        <=
        ROOM_W_MM
    ):

        continue


    # column's screeding vertical bounds
    col_ys = np.where(
        screeding[
            :,
            px
        ]
    )[0]


    if not len(
        col_ys
    ):

        continue


    sy_top = float(
        col_ys.min()
    )

    sy_bottom = float(
        col_ys.max()
        +
        1
    )


    denom = max(
        1.0,
        sy_bottom
        -
        sy_top
    )


    vertical_fraction = (
        float(
            py
        )
        -
        sy_top
    ) / denom


    vertical_fraction = np.clip(
        vertical_fraction,
        0.0,
        1.0
    )


    z_mm = (
        vertical_fraction
        *
        SCREED_H_MM
    )


    grout_x = in_grout(
        x_mm,
        X_GROUT_INTERVALS
    )


    if grout_x:

        combined_raw[
            py,
            px
        ] = GROUT_RGB

        continue


    x_cell = find_tile_interval(
        x_mm,
        X_TILE_INTERVALS
    )


    if x_cell is None:

        continue


    local_x = (
        x_mm
        -
        x_cell[
            0
        ]
    )


    # use lower 100-mm cut of same vertical tile face
    local_y = (
        TILE_L_MM
        -
        SCREED_H_MM
        +
        z_mm
    )


    color = sample_tile(
        local_x,
        local_y
    )


    combined_raw[
        py,
        px
    ] = color


# ============================================================
# PROJECTED GRID DEBUG
# ============================================================

grid_debug = room_np.copy()


# draw metric X boundaries
metric_x_lines = [
    0.0,
    600.0,
    605.0,
    1205.0,
    1210.0,
    1810.0,
    1815.0,
    2400.0,
]


for x_mm in metric_x_lines:

    pts_metric = np.array(
        [
            [
                [
                    x_mm,
                    0.0
                ],
                [
                    x_mm,
                    ROOM_L_MM
                ]
            ]
        ],
        dtype=np.float32
    )


    pts_img = cv2.perspectiveTransform(
        pts_metric,
        H_METRIC_TO_IMG
    )[0]


    p1 = tuple(
        np.round(
            pts_img[
                0
            ]
        ).astype(
            int
        )
    )

    p2 = tuple(
        np.round(
            pts_img[
                1
            ]
        ).astype(
            int
        )
    )


    cv2.line(
        grid_debug,
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


# draw metric Y boundaries
metric_y_lines = [
    0.0,
    1200.0,
    1205.0,
    1800.0,
]


for y_mm in metric_y_lines:

    pts_metric = np.array(
        [
            [
                [
                    0.0,
                    y_mm
                ],
                [
                    ROOM_W_MM,
                    y_mm
                ]
            ]
        ],
        dtype=np.float32
    )


    pts_img = cv2.perspectiveTransform(
        pts_metric,
        H_METRIC_TO_IMG
    )[0]


    p1 = tuple(
        np.round(
            pts_img[
                0
            ]
        ).astype(
            int
        )
    )

    p2 = tuple(
        np.round(
            pts_img[
                1
            ]
        ).astype(
            int
        )
    )


    cv2.line(
        grid_debug,
        p1,
        p2,
        (
            0,
            0,
            255
        ),
        1,
        cv2.LINE_AA
    )


GRID_PATH = (
    OUT
    / "02_projected_metric_grid.png"
)


Image.fromarray(
    grid_debug
).save(
    GRID_PATH
)


# ============================================================
# LIGHTING TRANSFER
#
# We preserve low-frequency Stage07 floor illumination.
#
# The goal is:
# exact tile texture
# ×
# room lighting field
#
# not AI-generated gloss.
# ============================================================

room_float = room_np.astype(
    np.float32
)


# grayscale luminance
room_luma = cv2.cvtColor(
    room_np,
    cv2.COLOR_RGB2GRAY
).astype(
    np.float32
)


# Strong blur isolates low-frequency illumination.
illum = cv2.GaussianBlur(
    room_luma,
    (0, 0),
    sigmaX=24.0,
    sigmaY=24.0
)


floor_values = illum[
    floor_main
]


if len(
    floor_values
):

    reference = float(
        np.median(
            floor_values
        )
    )

else:

    reference = 180.0


reference = max(
    reference,
    1.0
)


illum_ratio = (
    illum
    /
    reference
)


# Restrained range.
illum_ratio = np.clip(
    illum_ratio,
    0.78,
    1.18
)


lit_result = combined_raw.astype(
    np.float32
)


tile_target = (
    floor_main
    |
    screeding
)


lit_result[
    tile_target
] *= (
    illum_ratio[
        tile_target,
        None
    ]
)


# ============================================================
# SIMPLE GLOSS RESPONSE
#
# Not a fake mirror reflection.
# Just subtle highlight response from room luminance.
# ============================================================

highlight = np.clip(
    (
        illum_ratio
        -
        1.02
    )
    /
    0.16,
    0.0,
    1.0
)


highlight = cv2.GaussianBlur(
    highlight,
    (0, 0),
    sigmaX=10.0,
    sigmaY=10.0
)


GLOSS_STRENGTH = 0.07


lit_result[
    tile_target
] = (

    lit_result[
        tile_target
    ]
    *
    (
        1.0
        -
        GLOSS_STRENGTH
        *
        highlight[
            tile_target,
            None
        ]
    )

    +

    255.0
    *
    (
        GLOSS_STRENGTH
        *
        highlight[
            tile_target,
            None
        ]
    )
)


lit_result = np.clip(
    lit_result,
    0,
    255
).astype(
    np.uint8
)


# ============================================================
# SAVE OUTPUTS
# ============================================================

RAW_FLOOR_PATH = (
    OUT
    / "03_raw_exact_tile_floor_only.png"
)


Image.fromarray(
    raw_floor_tile
).save(
    RAW_FLOOR_PATH
)


RAW_COMBINED_PATH = (
    OUT
    / "04_raw_floor_plus_screeding.png"
)


Image.fromarray(
    combined_raw
).save(
    RAW_COMBINED_PATH
)


FINAL_PATH = (
    OUT
    / "05_lit_glossy_floor_plus_screeding.png"
)


Image.fromarray(
    lit_result
).save(
    FINAL_PATH
)


# ============================================================
# QUAD PREVIEW
# ============================================================

quad_preview = room_np.copy()


quad_int = np.round(
    IMG_QUAD
).astype(
    np.int32
)


cv2.polylines(
    quad_preview,
    [
        quad_int.reshape(
            -1,
            1,
            2
        )
    ],
    True,
    (
        255,
        0,
        255
    ),
    2,
    cv2.LINE_AA
)


for i, p in enumerate(
    quad_int,
    start=1
):

    cv2.circle(
        quad_preview,
        tuple(
            p
        ),
        4,
        (
            0,
            255,
            255
        ),
        -1
    )


    cv2.putText(
        quad_preview,
        str(
            i
        ),
        (
            int(
                p[
                    0
                ]
            )
            +
            4,
            int(
                p[
                    1
                ]
            )
            -
            4
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (
            255,
            0,
            0
        ),
        1,
        cv2.LINE_AA
    )


QUAD_PATH = (
    OUT
    / "06_floor_metric_quad_assumption.png"
)


Image.fromarray(
    quad_preview
).save(
    QUAD_PATH
)


# ============================================================
# 6-PANEL AUDIT
# ============================================================

fig, axes = plt.subplots(
    1,
    6,
    figsize=(
        26,
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
    quad_preview
)

axes[1].set_title(
    "2. Metric Floor Quad\nV1 Assumption"
)

axes[1].axis(
    "off"
)


axes[2].imshow(
    grid_debug
)

axes[2].set_title(
    "3. 600×1200 Metric Grid\n5 mm grout"
)

axes[2].axis(
    "off"
)


axes[3].imshow(
    raw_floor_tile
)

axes[3].set_title(
    "4. Exact Tile\nFloor Only"
)

axes[3].axis(
    "off"
)


axes[4].imshow(
    combined_raw
)

axes[4].set_title(
    "5. Floor + Screeding"
)

axes[4].axis(
    "off"
)


axes[5].imshow(
    lit_result
)

axes[5].set_title(
    "6. Lighting + Gloss"
)

axes[5].axis(
    "off"
)


plt.tight_layout()


AUDIT_PATH = (
    OUT
    / "07_stage08d_metric_tile_audit.png"
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
        "08D_V1",

    "purpose":
        "METRIC_600x1200_FLOOR_TILE_PROJECTION",

    "room_metric_mm": {

        "width":
            ROOM_W_MM,

        "length":
            ROOM_L_MM
    },

    "tile": {

        "source":
            str(
                TILE_PATH
            ),

        "width_mm":
            TILE_W_MM,

        "length_mm":
            TILE_L_MM,

        "orientation":
            ORIENTATION,

        "surface":
            SURFACE,

        "grout_mm":
            GROUT_MM
    },

    "layout": {

        "columns": [
            600,
            600,
            600,
            585
        ],

        "column_grouts_mm": [
            5,
            5,
            5
        ],

        "rows": [
            1200,
            595
        ],

        "row_grouts_mm": [
            5
        ]
    },

    "projection": {

        "metric_quad":
            METRIC_QUAD.tolist(),

        "image_quad":
            IMG_QUAD.tolist(),

        "assumption":
            (
                "For V1, visible bottom floor edge is treated "
                "as the near 1800-mm room boundary. "
                "Must be visually audited before freezing."
            ),

        "homography_metric_to_image":
            H_METRIC_TO_IMG.tolist(),

        "homography_image_to_metric":
            H_IMG_TO_METRIC.tolist()
    },

    "screeding": {

        "enabled":
            True,

        "height_mm_provisional":
            SCREED_H_MM,

        "same_floor_tile":
            True
    },

    "outputs": {

        "metric_layout":
            str(
                METRIC_DEBUG_PATH
            ),

        "projected_grid":
            str(
                GRID_PATH
            ),

        "raw_floor":
            str(
                RAW_FLOOR_PATH
            ),

        "raw_floor_plus_screeding":
            str(
                RAW_COMBINED_PATH
            ),

        "lit_glossy":
            str(
                FINAL_PATH
            ),

        "quad_assumption":
            str(
                QUAD_PATH
            ),

        "audit":
            str(
                AUDIT_PATH
            )
    },

    "status":
        "REQUIRES_METRIC_TILE_VISUAL_AUDIT"
}


STATE_PATH = (
    OUT
    / "00_stage08d_v1_result.json"
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
print("STAGE 08D V1 RESULT")
print("=" * 110)

print()

print(
    "IMAGE QUAD:"
)

for i, p in enumerate(
    IMG_QUAD,
    start=1
):

    print(
        f"P{i}:",
        [
            round(
                float(
                    p[
                        0
                    ]
                ),
                2
            ),
            round(
                float(
                    p[
                        1
                    ]
                ),
                2
            )
        ]
    )


print()
print(
    "WIDTH LAYOUT:",
    "600 + 5 + 600 + 5 + 600 + 5 + 585 = 2400 mm"
)

print(
    "LENGTH LAYOUT:",
    "1200 + 5 + 595 = 1800 mm"
)

print()

print(
    "GRID:",
    GRID_PATH
)

print(
    "RAW FLOOR:",
    RAW_FLOOR_PATH
)

print(
    "FLOOR + SCREEDING:",
    RAW_COMBINED_PATH
)

print(
    "LIT/GLOSSY:",
    FINAL_PATH
)

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
    "NO GENERATIVE AI WAS USED."
)

print(
    "EXACT SOURCE TILE PIXELS WERE SAMPLED."
)
