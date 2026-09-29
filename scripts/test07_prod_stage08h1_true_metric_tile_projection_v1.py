
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

SCREEDING_MASK_PATH = (
    PROD
    / "stage08_tile_application"
    / "08b2_floor_plus_screeding_target"
    / "02_wall_screeding_mask.png"
)

G3_STATE_PATH = (
    PROD
    / "stage08_tile_application"
    / "08g3_metric_camera_fit"
    / "00_stage08g3_room_box_state.json"
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
    / "08h1_true_metric_tile_projection"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOCKED ROOM / TILE SPEC
# ============================================================

ROOM_U_MM = 2400.0
ROOM_V_MM = 1800.0

TILE_U_MM = 600.0
TILE_V_MM = 1200.0

GROUT_MM = 5.0

SCREEDING_MAX_HEIGHT_MM = 180.0

# Visual finish
GLOSS_STRENGTH = 0.055

# Neutral grout
GROUT_RGB = np.array(
    [224, 224, 220],
    dtype=np.float32
)


# ============================================================
# VALIDATE
# ============================================================

for p in [
    ROOM_PATH,
    FLOOR_MASK_PATH,
    SCREEDING_MASK_PATH,
    G3_STATE_PATH,
    TILE_PATH,
]:

    if not p.exists():
        raise FileNotFoundError(p)


# ============================================================
# LOAD
# ============================================================

room = Image.open(
    ROOM_PATH
).convert("RGB")

tile = Image.open(
    TILE_PATH
).convert("RGB")


room_np = np.asarray(
    room
).astype(np.uint8)

tile_np = np.asarray(
    tile
).astype(np.uint8)


H, W = room_np.shape[:2]

tile_h, tile_w = tile_np.shape[:2]


floor = (
    np.asarray(
        Image.open(
            FLOOR_MASK_PATH
        ).convert("L")
    )
    > 127
)


screeding = (
    np.asarray(
        Image.open(
            SCREEDING_MASK_PATH
        ).convert("L")
    )
    > 127
)


state = json.loads(
    G3_STATE_PATH.read_text(
        encoding="utf-8"
    )
)


K = np.asarray(
    state["camera"]["K"],
    dtype=np.float64
)

R = np.asarray(
    state["camera"]["R"],
    dtype=np.float64
)

t = np.asarray(
    state["camera"]["t"],
    dtype=np.float64
)


Kinv = np.linalg.inv(K)


CAMERA_CENTER = (
    -R.T
    @
    t
)


BACK_FLOOR = np.asarray(
    state["evidence"]["back_floor_corner"],
    dtype=np.float64
)

BACK_X_IMAGE = float(
    BACK_FLOOR[0]
)


print("=" * 110)
print("STAGE 08H1 — TRUE METRIC TILE PROJECTION")
print("=" * 110)

print()
print("ROOM:", ROOM_PATH)
print("TILE:", TILE_PATH)

print()
print("ROOM:", "2400 × 1800 mm")
print("TILE:", "600 × 1200 mm")
print("GROUT:", "5 mm")
print("ORIENTATION:", "600 along U | 1200 along V")

print()
print(
    "CAMERA CENTER:",
    [
        round(float(v), 2)
        for v in CAMERA_CENTER
    ]
)


# ============================================================
# CAMERA RAY
# ============================================================

def pixel_ray_world(px, py):

    p = np.array(
        [
            float(px),
            float(py),
            1.0
        ],
        dtype=np.float64
    )

    d_cam = (
        Kinv
        @
        p
    )

    d_world = (
        R.T
        @
        d_cam
    )

    norm = np.linalg.norm(
        d_world
    )

    if norm < 1e-12:
        return None

    return (
        d_world
        /
        norm
    )


# ============================================================
# RAY -> FLOOR Y=0
#
# Returns metric:
# U = -world X
# V = +world Z
# ============================================================

def intersect_floor(px, py):

    d = pixel_ray_world(
        px,
        py
    )

    if d is None:
        return None

    if abs(d[1]) < 1e-10:
        return None

    lam = (
        -CAMERA_CENTER[1]
        /
        d[1]
    )

    if lam <= 0:
        return None

    P = (
        CAMERA_CENTER
        +
        lam * d
    )

    u = -float(
        P[0]
    )

    v = float(
        P[2]
    )

    return (
        u,
        v,
        P
    )


# ============================================================
# RAY -> LEFT WALL
#
# Left wall plane:
# Z = 0
#
# Horizontal wall coordinate = U=-X
# Vertical coordinate = world Y above floor
# ============================================================

def intersect_left_wall(px, py):

    d = pixel_ray_world(
        px,
        py
    )

    if d is None:
        return None

    if abs(d[2]) < 1e-10:
        return None

    lam = (
        -CAMERA_CENTER[2]
        /
        d[2]
    )

    if lam <= 0:
        return None

    P = (
        CAMERA_CENTER
        +
        lam * d
    )

    u = -float(
        P[0]
    )

    h = float(
        P[1]
    )

    return (
        u,
        h,
        P
    )


# ============================================================
# RAY -> RIGHT WALL
#
# Right wall plane:
# X = 0
#
# Horizontal wall coordinate = V=Z
# Vertical coordinate = world Y
# ============================================================

def intersect_right_wall(px, py):

    d = pixel_ray_world(
        px,
        py
    )

    if d is None:
        return None

    if abs(d[0]) < 1e-10:
        return None

    lam = (
        -CAMERA_CENTER[0]
        /
        d[0]
    )

    if lam <= 0:
        return None

    P = (
        CAMERA_CENTER
        +
        lam * d
    )

    v = float(
        P[2]
    )

    h = float(
        P[1]
    )

    return (
        v,
        h,
        P
    )


# ============================================================
# METRIC TILE LAYOUT
#
# We treat the nominal tile as:
#
# 600 mm artwork
# + 5 mm grout between installed pieces.
#
# Pitch:
# 605 along U
# 1205 along V
#
# This is more general than manually hardcoding 4 columns.
# ============================================================

PITCH_U = (
    TILE_U_MM
    +
    GROUT_MM
)

PITCH_V = (
    TILE_V_MM
    +
    GROUT_MM
)


def tile_coordinate(
    coord,
    tile_size,
    pitch
):

    if coord < 0:
        return None

    local = (
        coord
        %
        pitch
    )

    # grout occupies final 5 mm of each pitch
    if local >= tile_size:
        return (
            True,
            local,
            0.0
        )

    uv = (
        local
        /
        tile_size
    )

    return (
        False,
        local,
        uv
    )


# ============================================================
# EXACT TILE RGB SAMPLE
# ============================================================

def sample_tile(u_frac, v_frac):

    u_frac = float(
        np.clip(
            u_frac,
            0.0,
            0.999999
        )
    )

    v_frac = float(
        np.clip(
            v_frac,
            0.0,
            0.999999
        )
    )


    tx = int(
        u_frac
        *
        tile_w
    )

    ty = int(
        v_frac
        *
        tile_h
    )


    return tile_np[
        ty,
        tx
    ].astype(
        np.float32
    )


# ============================================================
# FLOOR RAW PROJECTION
# ============================================================

raw = room_np.astype(
    np.float32
).copy()


metric_valid_floor = np.zeros(
    (
        H,
        W
    ),
    dtype=bool
)


grout_floor = np.zeros(
    (
        H,
        W
    ),
    dtype=bool
)


tile_floor = np.zeros(
    (
        H,
        W
    ),
    dtype=bool
)


floor_u_map = np.full(
    (
        H,
        W
    ),
    np.nan,
    dtype=np.float32
)

floor_v_map = np.full(
    (
        H,
        W
    ),
    np.nan,
    dtype=np.float32
)


ys, xs = np.where(
    floor
)


for py, px in zip(
    ys,
    xs
):

    hit = intersect_floor(
        px,
        py
    )

    if hit is None:
        continue

    u, v, P = hit


    if not (
        0.0 <= u <= ROOM_U_MM
        and
        0.0 <= v <= ROOM_V_MM
    ):
        continue


    floor_u_map[
        py,
        px
    ] = u

    floor_v_map[
        py,
        px
    ] = v


    metric_valid_floor[
        py,
        px
    ] = True


    cu = tile_coordinate(
        u,
        TILE_U_MM,
        PITCH_U
    )

    cv = tile_coordinate(
        v,
        TILE_V_MM,
        PITCH_V
    )


    if (
        cu is None
        or
        cv is None
    ):
        continue


    grout_u = cu[0]
    grout_v = cv[0]


    if (
        grout_u
        or
        grout_v
    ):

        raw[
            py,
            px
        ] = GROUT_RGB

        grout_floor[
            py,
            px
        ] = True

        continue


    color = sample_tile(
        cu[2],
        cv[2]
    )


    raw[
        py,
        px
    ] = color


    tile_floor[
        py,
        px
    ] = True


# ============================================================
# SCREEDING
#
# Split by the visible back corner:
#
# x < back_x  -> left wall (Z=0)
# x >= back_x -> right wall (X=0)
#
# Use same tile material.
#
# The narrow skirting is a cut piece from the SAME tile.
# ============================================================

metric_valid_screed = np.zeros(
    (
        H,
        W
    ),
    dtype=bool
)


screed_grout = np.zeros(
    (
        H,
        W
    ),
    dtype=bool
)


sy, sx = np.where(
    screeding
)


for py, px in zip(
    sy,
    sx
):

    # --------------------------------------------------------
    # LEFT WALL
    # --------------------------------------------------------

    if px < BACK_X_IMAGE:

        hit = intersect_left_wall(
            px,
            py
        )


        if hit is None:
            continue


        horizontal_mm, height_mm, P = hit


        if not (
            0.0
            <=
            horizontal_mm
            <=
            ROOM_U_MM
        ):
            continue


    # --------------------------------------------------------
    # RIGHT WALL
    # --------------------------------------------------------

    else:

        hit = intersect_right_wall(
            px,
            py
        )


        if hit is None:
            continue


        horizontal_mm, height_mm, P = hit


        if not (
            0.0
            <=
            horizontal_mm
            <=
            ROOM_V_MM
        ):
            continue


    if not (
        0.0
        <=
        height_mm
        <=
        SCREEDING_MAX_HEIGHT_MM
    ):

        continue


    metric_valid_screed[
        py,
        px
    ] = True


    # --------------------------------------------------------
    # Horizontal joints follow the installed floor rhythm.
    #
    # For left wall:
    # horizontal coordinate naturally follows 600-mm U tile.
    #
    # For right wall, the floor V direction is the 1200-mm
    # dimension; preserve that rhythm.
    # --------------------------------------------------------

    if px < BACK_X_IMAGE:

        c_horizontal = tile_coordinate(
            horizontal_mm,
            TILE_U_MM,
            PITCH_U
        )

        if c_horizontal is None:
            continue

        is_grout = c_horizontal[0]

        u_frac = c_horizontal[2]


    else:

        c_horizontal = tile_coordinate(
            horizontal_mm,
            TILE_V_MM,
            PITCH_V
        )

        if c_horizontal is None:
            continue

        is_grout = c_horizontal[0]

        # map horizontal cut of long tile dimension into V
        # while using a stable lower strip of the source face.
        u_frac = 0.5


    if is_grout:

        raw[
            py,
            px
        ] = GROUT_RGB

        screed_grout[
            py,
            px
        ] = True

        continue


    # --------------------------------------------------------
    # Screeding uses lower part of same tile face.
    #
    # Its vertical physical height is only a narrow cut.
    # --------------------------------------------------------

    vertical_fraction_of_tile = np.clip(
        height_mm
        /
        TILE_V_MM,
        0.0,
        1.0
    )


    v_frac = (
        1.0
        -
        vertical_fraction_of_tile
    )


    if px < BACK_X_IMAGE:

        color = sample_tile(
            u_frac,
            v_frac
        )

    else:

        # Right wall runs along tile's long installed direction.
        h_frac = (
            c_horizontal[2]
        )

        color = sample_tile(
            0.5,
            h_frac
        )


    raw[
        py,
        px
    ] = color


# ============================================================
# SAVE RAW EXACT TEXTURE
# ============================================================

RAW_PATH = (
    OUT
    / "01_raw_true_metric_tile_projection.png"
)


Image.fromarray(
    np.clip(
        raw,
        0,
        255
    ).astype(
        np.uint8
    )
).save(
    RAW_PATH
)


# ============================================================
# LIGHTING FIELD
#
# Preserve only low-frequency illumination from Stage07A.
# ============================================================

room_luma = cv2.cvtColor(
    room_np,
    cv2.COLOR_RGB2GRAY
).astype(
    np.float32
)


illum = cv2.GaussianBlur(
    room_luma,
    (0, 0),
    sigmaX=24.0,
    sigmaY=24.0
)


target = (
    metric_valid_floor
    |
    metric_valid_screed
)


if target.any():

    reference = float(
        np.median(
            illum[
                target
            ]
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


illum_ratio = np.clip(
    illum_ratio,
    0.82,
    1.16
)


lit = raw.copy()


lit[
    target
] *= (
    illum_ratio[
        target,
        None
    ]
)


# ============================================================
# RESTRAINED GLOSS RESPONSE
#
# No hallucinated reflection.
# ============================================================

highlight = np.clip(
    (
        illum_ratio
        -
        1.015
    )
    /
    0.145,
    0.0,
    1.0
)


highlight = cv2.GaussianBlur(
    highlight,
    (0, 0),
    sigmaX=8.0,
    sigmaY=8.0
)


lit[
    target
] = (

    lit[
        target
    ]
    *
    (
        1.0
        -
        GLOSS_STRENGTH
        *
        highlight[
            target,
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
            target,
            None
        ]
    )

)


lit = np.clip(
    lit,
    0,
    255
).astype(
    np.uint8
)


FINAL_PATH = (
    OUT
    / "02_lit_glossy_true_metric_projection.png"
)


Image.fromarray(
    lit
).save(
    FINAL_PATH
)


# ============================================================
# TRUE GROUT / GRID DEBUG
# ============================================================

grid_preview = room_np.copy()


grid_preview[
    grout_floor
] = [
    255,
    0,
    0
]


grid_preview[
    screed_grout
] = [
    255,
    0,
    255
]


GRID_PATH = (
    OUT
    / "03_true_5mm_grout_debug.png"
)


Image.fromarray(
    grid_preview
).save(
    GRID_PATH
)


# ============================================================
# VALID TARGET AUDIT
# ============================================================

valid_preview = room_np.astype(
    np.float32
).copy()


valid_preview[
    metric_valid_floor
] = (

    valid_preview[
        metric_valid_floor
    ]
    *
    0.55

    +

    np.array(
        [
            0,
            255,
            0
        ],
        dtype=np.float32
    )
    *
    0.45
)


valid_preview[
    metric_valid_screed
] = (

    valid_preview[
        metric_valid_screed
    ]
    *
    0.45

    +

    np.array(
        [
            0,
            255,
            255
        ],
        dtype=np.float32
    )
    *
    0.55
)


valid_preview = np.clip(
    valid_preview,
    0,
    255
).astype(
    np.uint8
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
    tile
)

axes[1].set_title(
    "2. Exact Tile Source\n"
    "600×1200"
)

axes[1].axis(
    "off"
)


axes[2].imshow(
    valid_preview
)

axes[2].set_title(
    "3. True Metric Target\n"
    "GREEN=floor | CYAN=screeding"
)

axes[2].axis(
    "off"
)


axes[3].imshow(
    grid_preview
)

axes[3].set_title(
    "4. True 5-mm Grout\n"
    "RED=floor | MAGENTA=screeding"
)

axes[3].axis(
    "off"
)


axes[4].imshow(
    np.clip(
        raw,
        0,
        255
    ).astype(
        np.uint8
    )
)

axes[4].set_title(
    "5. Exact RGB\n"
    "True Metric Projection"
)

axes[4].axis(
    "off"
)


axes[5].imshow(
    lit
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
    / "04_stage08h1_true_metric_tile_audit.png"
)


plt.savefig(
    AUDIT_PATH,
    dpi=150,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# STATISTICS
# ============================================================

floor_pixels = int(
    floor.sum()
)

valid_floor_pixels = int(
    metric_valid_floor.sum()
)

screed_pixels = int(
    screeding.sum()
)

valid_screed_pixels = int(
    metric_valid_screed.sum()
)


# ============================================================
# STATE
# ============================================================

STATE = {

    "stage":
        "08H1",

    "purpose":
        "TRUE_METRIC_600x1200_TILE_PROJECTION",

    "room_mm": {

        "u":
            ROOM_U_MM,

        "v":
            ROOM_V_MM
    },

    "tile": {

        "source":
            str(
                TILE_PATH
            ),

        "u_mm":
            TILE_U_MM,

        "v_mm":
            TILE_V_MM,

        "grout_mm":
            GROUT_MM,

        "orientation":
            "600_mm_along_U__1200_mm_along_V",

        "finish":
            "GLOSSY"
    },

    "projection": {

        "floor":
            "raycast_to_world_Y_0",

        "left_screeding":
            "raycast_to_world_Z_0",

        "right_screeding":
            "raycast_to_world_X_0"
    },

    "statistics": {

        "floor_mask_pixels":
            floor_pixels,

        "valid_metric_floor_pixels":
            valid_floor_pixels,

        "floor_metric_coverage":
            (
                valid_floor_pixels
                /
                max(
                    floor_pixels,
                    1
                )
            ),

        "screeding_mask_pixels":
            screed_pixels,

        "valid_metric_screeding_pixels":
            valid_screed_pixels,

        "screeding_metric_coverage":
            (
                valid_screed_pixels
                /
                max(
                    screed_pixels,
                    1
                )
            )
    },

    "outputs": {

        "raw":
            str(
                RAW_PATH
            ),

        "lit_glossy":
            str(
                FINAL_PATH
            ),

        "grout_debug":
            str(
                GRID_PATH
            ),

        "audit":
            str(
                AUDIT_PATH
            )
    },

    "status":
        "REQUIRES_TRUE_METRIC_TILE_VISUAL_AUDIT"
}


STATE_PATH = (
    OUT
    / "00_stage08h1_result.json"
)


STATE_PATH.write_text(
    json.dumps(
        STATE,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


print()
print("=" * 110)
print("STAGE 08H1 RESULT")
print("=" * 110)

print()

print(
    "FLOOR METRIC COVERAGE:",
    round(
        valid_floor_pixels
        /
        max(
            floor_pixels,
            1
        ),
        4
    )
)

print(
    "SCREEDING METRIC COVERAGE:",
    round(
        valid_screed_pixels
        /
        max(
            screed_pixels,
            1
        ),
        4
    )
)

print()

print(
    "RAW:",
    RAW_PATH
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
    "TILE SOURCE ARTWORK WAS SAMPLED DIRECTLY."
)
