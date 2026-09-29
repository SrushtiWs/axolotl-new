
from pathlib import Path
import json
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

G3_STATE_PATH = (
    PROD
    / "stage08_tile_application"
    / "08g3_metric_camera_fit"
    / "00_stage08g3_room_box_state.json"
)

OUT = (
    PROD
    / "stage08_tile_application"
    / "08g4_visible_floor_metric_audit"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOCKED METRIC ROOM
# ============================================================

ROOM_WIDTH_MM = 2400.0
ROOM_LENGTH_MM = 1800.0

GRID_STEP_MM = 600.0


# ============================================================
# LOAD
# ============================================================

for p in [
    ROOM_PATH,
    FLOOR_MASK_PATH,
    G3_STATE_PATH,
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


Kinv = np.linalg.inv(
    K
)


# ============================================================
# CAMERA CENTER IN WORLD
#
# Pc = R Pw + t
#
# therefore:
# Cworld = -R^T t
# ============================================================

CAMERA_CENTER = (
    -R.T
    @
    t
)


print("=" * 110)
print("STAGE 08G4 — VISIBLE FLOOR METRIC AUDIT")
print("=" * 110)

print()

print(
    "CAMERA CENTER WORLD [mm]:",
    [
        round(
            float(x),
            2
        )
        for x in CAMERA_CENTER
    ]
)

print()

print(
    "EXPECTED ROOM DOMAIN:"
)

print(
    "WORLD X:",
    "0 -> -2400 mm"
)

print(
    "WORLD Z:",
    "0 -> +1800 mm"
)


# ============================================================
# IMAGE PIXEL -> WORLD FLOOR PLANE
#
# Camera ray:
#
# d_cam = K^-1 p
#
# d_world = R^T d_cam
#
# Pworld = C + lambda*d
#
# Floor:
# Y = 0
#
# lambda = -C_y / d_y
# ============================================================

U_MM = np.full(
    (
        H,
        W
    ),
    np.nan,
    dtype=np.float32
)

V_MM = np.full(
    (
        H,
        W
    ),
    np.nan,
    dtype=np.float32
)


valid_metric = np.zeros(
    (
        H,
        W
    ),
    dtype=bool
)


ys, xs = np.where(
    floor
)


for py, px in zip(
    ys,
    xs
):

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


    if abs(
        d_world[1]
    ) < 1e-10:

        continue


    lam = (
        -CAMERA_CENTER[1]
        /
        d_world[1]
    )


    # Only forward ray intersection.
    if lam <= 0:

        continue


    P = (
        CAMERA_CENTER
        +
        lam
        *
        d_world
    )


    # ----------------------------------------------
    # Correct physical room coordinate convention
    #
    # u = left-wall direction
    #     = -world X
    #
    # v = right-wall direction
    #     = +world Z
    # ----------------------------------------------

    u = float(
        -P[0]
    )

    v = float(
        P[2]
    )


    U_MM[
        py,
        px
    ] = u


    V_MM[
        py,
        px
    ] = v


    if (
        0.0 <= u <= ROOM_WIDTH_MM
        and
        0.0 <= v <= ROOM_LENGTH_MM
    ):

        valid_metric[
            py,
            px
        ] = True


# ============================================================
# STATISTICS
# ============================================================

finite = (
    np.isfinite(
        U_MM
    )
    &
    np.isfinite(
        V_MM
    )
)


print(
    "FLOOR MASK PIXELS:",
    int(
        floor.sum()
    )
)

print(
    "RAY-PLANE INTERSECTIONS:",
    int(
        finite.sum()
    )
)

print(
    "INSIDE 2400×1800 DOMAIN:",
    int(
        valid_metric.sum()
    )
)


if finite.any():

    print()

    print(
        "VISIBLE U RANGE:",
        [
            round(
                float(
                    np.nanmin(
                        U_MM[
                            finite
                        ]
                    )
                ),
                2
            ),
            round(
                float(
                    np.nanmax(
                        U_MM[
                            finite
                        ]
                    )
                ),
                2
            )
        ],
        "mm"
    )


    print(
        "VISIBLE V RANGE:",
        [
            round(
                float(
                    np.nanmin(
                        V_MM[
                            finite
                        ]
                    )
                ),
                2
            ),
            round(
                float(
                    np.nanmax(
                        V_MM[
                            finite
                        ]
                    )
                ),
                2
            )
        ],
        "mm"
    )


# ============================================================
# BUILD TRUE METRIC GRID MASK
#
# Instead of projecting a fake quadrilateral, identify image
# pixels whose recovered metric coordinate lies near a 600-mm
# grid line.
# ============================================================

GRID_HALF_WIDTH_MM = 16.0


grid_u = np.zeros(
    (
        H,
        W
    ),
    dtype=bool
)


grid_v = np.zeros(
    (
        H,
        W
    ),
    dtype=bool
)


# ------------------------------------------------------------
# U grid:
# 0, 600, 1200, 1800, 2400
# ------------------------------------------------------------

for value in np.arange(
    0.0,
    ROOM_WIDTH_MM + 0.1,
    GRID_STEP_MM
):

    hit = (
        valid_metric
        &
        (
            np.abs(
                U_MM - value
            )
            <=
            GRID_HALF_WIDTH_MM
        )
    )


    grid_u |= hit


# ------------------------------------------------------------
# V grid:
# 0, 600, 1200, 1800
# ------------------------------------------------------------

for value in np.arange(
    0.0,
    ROOM_LENGTH_MM + 0.1,
    GRID_STEP_MM
):

    hit = (
        valid_metric
        &
        (
            np.abs(
                V_MM - value
            )
            <=
            GRID_HALF_WIDTH_MM
        )
    )


    grid_v |= hit


# ============================================================
# RGB GRID OVERLAY
# ============================================================

grid_preview = room_np.copy()


# U = RED
grid_preview[
    grid_u
] = [
    255,
    0,
    0
]


# V = CYAN
grid_preview[
    grid_v
] = [
    0,
    255,
    255
]


# intersections = yellow
grid_preview[
    grid_u
    &
    grid_v
] = [
    255,
    255,
    0
]


# ============================================================
# VALID METRIC REGION OVERLAY
# ============================================================

region_preview = room_np.astype(
    np.float32
).copy()


region_preview[
    valid_metric
] = (

    region_preview[
        valid_metric
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


region_preview = np.clip(
    region_preview,
    0,
    255
).astype(
    np.uint8
)


# ============================================================
# METRIC HEATMAPS
# ============================================================

U_VIS = np.zeros(
    (
        H,
        W
    ),
    dtype=np.float32
)


V_VIS = np.zeros(
    (
        H,
        W
    ),
    dtype=np.float32
)


U_VIS[
    valid_metric
] = (
    U_MM[
        valid_metric
    ]
    /
    ROOM_WIDTH_MM
)


V_VIS[
    valid_metric
] = (
    V_MM[
        valid_metric
    ]
    /
    ROOM_LENGTH_MM
)


# ============================================================
# SAVE
# ============================================================

VALID_PATH = (
    OUT
    / "01_valid_metric_floor_region.png"
)


GRID_PATH = (
    OUT
    / "02_true_600mm_metric_grid.png"
)


Image.fromarray(
    valid_metric.astype(
        np.uint8
    )
    *
    255
).save(
    VALID_PATH
)


Image.fromarray(
    grid_preview
).save(
    GRID_PATH
)


# ============================================================
# 5-PANEL AUDIT
# ============================================================

fig, axes = plt.subplots(
    1,
    5,
    figsize=(
        22,
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
    region_preview
)

axes[1].set_title(
    "2. Metric-valid Floor\n"
    "GREEN=inside 2400×1800"
)

axes[1].axis(
    "off"
)


axes[2].imshow(
    U_VIS,
    cmap="gray",
    vmin=0,
    vmax=1
)

axes[2].set_title(
    "3. U Coordinate\n"
    "0 → 2400 mm"
)

axes[2].axis(
    "off"
)


axes[3].imshow(
    V_VIS,
    cmap="gray",
    vmin=0,
    vmax=1
)

axes[3].set_title(
    "4. V Coordinate\n"
    "0 → 1800 mm"
)

axes[3].axis(
    "off"
)


axes[4].imshow(
    grid_preview
)

axes[4].set_title(
    "5. TRUE 600-mm Grid\n"
    "RED=U | CYAN=V"
)

axes[4].axis(
    "off"
)


plt.tight_layout()


AUDIT_PATH = (
    OUT
    / "03_stage08g4_metric_floor_audit.png"
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
        "08G4",

    "purpose":
        "RAYCAST_VISIBLE_FLOOR_TO_METRIC_PLANE",

    "room_domain_mm": {

        "u_width":
            [
                0.0,
                ROOM_WIDTH_MM
            ],

        "v_length":
            [
                0.0,
                ROOM_LENGTH_MM
            ]
    },

    "world_convention": {

        "back_corner":
            [
                0.0,
                0.0,
                0.0
            ],

        "u":
            "-world_X",

        "v":
            "+world_Z",

        "floor":
            "world_Y=0"
    },

    "camera_center_world_mm":
        CAMERA_CENTER.tolist(),

    "statistics": {

        "floor_pixels":
            int(
                floor.sum()
            ),

        "ray_plane_pixels":
            int(
                finite.sum()
            ),

        "inside_metric_domain_pixels":
            int(
                valid_metric.sum()
            )
    },

    "outputs": {

        "valid_region":
            str(
                VALID_PATH
            ),

        "grid":
            str(
                GRID_PATH
            ),

        "audit":
            str(
                AUDIT_PATH
            )
    },

    "status":
        "REQUIRES_TRUE_METRIC_GRID_VISUAL_AUDIT",

    "next_if_pass":
        "08H1_PROJECT_EXACT_600x1200_TILE_FROM_METRIC_COORDINATES"
}


STATE_PATH = (
    OUT
    / "00_stage08g4_result.json"
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
print("STAGE 08G4 RESULT")
print("=" * 110)

print()

print(
    "CAMERA CENTER:",
    [
        round(
            float(x),
            2
        )
        for x in CAMERA_CENTER
    ]
)

print(
    "VALID METRIC PIXELS:",
    int(
        valid_metric.sum()
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
