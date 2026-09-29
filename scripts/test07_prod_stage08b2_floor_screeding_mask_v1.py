
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
    / "08b2_floor_plus_screeding_target"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
#
# Search upward from wall-floor edge for the visible upper edge
# of the small screeding/skirting strip.
# ============================================================

MIN_HEIGHT_PX = 3
MAX_HEIGHT_PX = 22

# Smooth image slightly before gradient measurement.
BLUR_SIGMA = 1.0

# Smooth the detected upper edge across neighboring columns.
EDGE_SMOOTH_SIGMA = 3.0

# Ignore tiny isolated columns.
MIN_RUN_WIDTH = 5


# ============================================================
# VALIDATE
# ============================================================

for p in [
    ROOM_PATH,
    FLOOR_MASK_PATH,
]:
    if not p.exists():
        raise FileNotFoundError(p)


# ============================================================
# LOAD
# ============================================================

room = Image.open(
    ROOM_PATH
).convert("RGB")

room_np = np.asarray(
    room
)

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
        "Floor mask and room dimensions do not match."
    )


print("=" * 110)
print("STAGE 08B2 — FLOOR + WALL SCREEDING TARGET")
print("=" * 110)

print()
print("ROOM:", ROOM_PATH)
print("FLOOR MASK:", FLOOR_MASK_PATH)
print("SIZE:", f"{W} × {H}")


# ============================================================
# KEEP ONLY MAIN FLOOR COMPONENT
# ============================================================

num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
    floor.astype(np.uint8),
    connectivity=8
)


if num_labels <= 1:
    raise RuntimeError(
        "No connected floor component found."
    )


largest_label = (
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
    largest_label
)


# ============================================================
# WALL-FLOOR BOUNDARY
#
# topmost floor pixel in every valid column
# ============================================================

floor_top = np.full(
    W,
    np.nan,
    dtype=np.float32
)


for x in range(W):

    ys = np.where(
        floor_main[:, x]
    )[0]

    if len(ys):
        floor_top[x] = float(
            ys.min()
        )


valid = np.isfinite(
    floor_top
)


# ============================================================
# PREPARE GRAYSCALE / VERTICAL GRADIENT
# ============================================================

gray = cv2.cvtColor(
    room_np,
    cv2.COLOR_RGB2GRAY
).astype(
    np.float32
)


gray = cv2.GaussianBlur(
    gray,
    (0, 0),
    sigmaX=BLUR_SIGMA,
    sigmaY=BLUR_SIGMA
)


# Vertical image derivative:
# useful for detecting the horizontal-ish upper skirting edge.
grad_y = cv2.Sobel(
    gray,
    cv2.CV_32F,
    0,
    1,
    ksize=3
)


grad_mag = np.abs(
    grad_y
)


# ============================================================
# SEARCH FOR UPPER SCREEDING EDGE
#
# Search only ABOVE the floor boundary.
# ============================================================

upper_edge = np.full(
    W,
    np.nan,
    dtype=np.float32
)


edge_confidence = np.zeros(
    W,
    dtype=np.float32
)


for x in range(W):

    if not valid[x]:
        continue


    fy = int(
        round(
            floor_top[x]
        )
    )


    y_bottom = max(
        0,
        fy - MIN_HEIGHT_PX
    )

    y_top = max(
        0,
        fy - MAX_HEIGHT_PX
    )


    if y_bottom <= y_top:
        continue


    profile = grad_mag[
        y_top:y_bottom + 1,
        x
    ]


    if len(profile) == 0:
        continue


    local_idx = int(
        np.argmax(
            profile
        )
    )


    candidate_y = (
        y_top
        +
        local_idx
    )


    upper_edge[x] = float(
        candidate_y
    )


    edge_confidence[x] = float(
        profile[
            local_idx
        ]
    )


# ============================================================
# REMOVE LOW-CONFIDENCE OUTLIERS
#
# Use robust confidence threshold from detected columns.
# ============================================================

detected = np.isfinite(
    upper_edge
)


if detected.sum() < 20:
    raise RuntimeError(
        "Too few screeding-edge detections."
    )


confidence_values = edge_confidence[
    detected
]


conf_threshold = float(
    np.percentile(
        confidence_values,
        35
    )
)


strong = (
    detected
    &
    (
        edge_confidence
        >=
        conf_threshold
    )
)


# ============================================================
# INTERPOLATE MISSING COLUMNS
# ============================================================

strong_x = np.where(
    strong
)[0]


strong_y = upper_edge[
    strong
]


if len(strong_x) < 10:
    raise RuntimeError(
        "Too few strong screeding edge points."
    )


interp_edge = np.full(
    W,
    np.nan,
    dtype=np.float32
)


valid_floor_x = np.where(
    valid
)[0]


interp_values = np.interp(
    valid_floor_x,
    strong_x,
    strong_y
)


interp_edge[
    valid_floor_x
] = interp_values


# Smooth the resulting edge.
smooth_edge = interp_edge.copy()


finite_idx = np.where(
    np.isfinite(
        smooth_edge
    )
)[0]


vals = smooth_edge[
    finite_idx
].reshape(
    1,
    -1
)


vals = cv2.GaussianBlur(
    vals,
    (0, 0),
    sigmaX=EDGE_SMOOTH_SIGMA
).reshape(
    -1
)


smooth_edge[
    finite_idx
] = vals


# ============================================================
# BUILD SCREEDING MASK
#
# Region between:
#   detected top screeding edge
# and
#   floor top boundary
# ============================================================

screeding = np.zeros(
    (
        H,
        W
    ),
    dtype=bool
)


for x in valid_floor_x:

    bottom = int(
        round(
            floor_top[x]
        )
    )

    top = int(
        round(
            smooth_edge[x]
        )
    )


    # Enforce plausible screeding height.
    height = bottom - top


    if height < MIN_HEIGHT_PX:
        top = bottom - MIN_HEIGHT_PX

    elif height > MAX_HEIGHT_PX:
        top = bottom - MAX_HEIGHT_PX


    top = max(
        0,
        top
    )

    bottom = min(
        H,
        bottom
    )


    if bottom > top:

        screeding[
            top:bottom,
            x
        ] = True


# ============================================================
# DO NOT INCLUDE FLOOR ITSELF IN SCREEDING
# ============================================================

screeding &= (
    ~floor_main
)


# ============================================================
# CLEAN SMALL NOISE
# ============================================================

kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT,
    (
        MIN_RUN_WIDTH,
        3
    )
)


screeding_clean = cv2.morphologyEx(
    screeding.astype(np.uint8),
    cv2.MORPH_CLOSE,
    kernel
).astype(bool)


# ============================================================
# FINAL FUTURE FLOOR-TILE TARGET
#
# SAME TILE will apply to BOTH:
# floor + bottom wall screeding
# ============================================================

tile_target = (
    floor_main
    |
    screeding_clean
)


# ============================================================
# SAVE MASKS
# ============================================================

FLOOR_PATH = (
    OUT
    / "01_main_floor_mask.png"
)


SCREEDING_PATH = (
    OUT
    / "02_wall_screeding_mask.png"
)


TARGET_PATH = (
    OUT
    / "03_floor_plus_screeding_tile_target.png"
)


Image.fromarray(
    floor_main.astype(np.uint8)
    *
    255
).save(
    FLOOR_PATH
)


Image.fromarray(
    screeding_clean.astype(np.uint8)
    *
    255
).save(
    SCREEDING_PATH
)


Image.fromarray(
    tile_target.astype(np.uint8)
    *
    255
).save(
    TARGET_PATH
)


# ============================================================
# RGB OVERLAY
# ============================================================

overlay = room_np.astype(
    np.float32
).copy()


# FLOOR = GREEN
overlay[
    floor_main
] = (
    overlay[
        floor_main
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


# SCREEDING = CYAN
overlay[
    screeding_clean
] = (
    overlay[
        screeding_clean
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


overlay = np.clip(
    overlay,
    0,
    255
).astype(
    np.uint8
)


# ============================================================
# EDGE PREVIEW
# ============================================================

edge_preview = room_np.copy()


for x in valid_floor_x:

    fy = int(
        round(
            floor_top[x]
        )
    )

    sy = int(
        round(
            smooth_edge[x]
        )
    )


    if 0 <= fy < H:
        edge_preview[
            fy,
            x
        ] = [
            0,
            255,
            0
        ]


    if 0 <= sy < H:
        edge_preview[
            sy,
            x
        ] = [
            255,
            0,
            0
        ]


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
    floor_main,
    cmap="gray"
)

axes[1].set_title(
    "2. Main Floor"
)

axes[1].axis(
    "off"
)


axes[2].imshow(
    edge_preview
)

axes[2].set_title(
    "3. Detected Screeding Edges\n"
    "RED=top | GREEN=floor junction"
)

axes[2].axis(
    "off"
)


axes[3].imshow(
    screeding_clean,
    cmap="gray"
)

axes[3].set_title(
    "4. Wall Screeding Mask"
)

axes[3].axis(
    "off"
)


axes[4].imshow(
    overlay
)

axes[4].set_title(
    "5. Final Tile Target\n"
    "GREEN=floor | CYAN=screeding"
)

axes[4].axis(
    "off"
)


plt.tight_layout()


AUDIT_PATH = (
    OUT
    / "04_stage08b2_screeding_audit.png"
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
        "08B2",

    "purpose":
        "DETECT_BOTTOM_WALL_SCREEEDING_FOR_FLOOR_TILE",

    "rule":
        (
            "Floor tile material applies to the main floor "
            "and the small bottom wall screeding/skirting band."
        ),

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

        "min_height_px":
            MIN_HEIGHT_PX,

        "max_height_px":
            MAX_HEIGHT_PX,

        "blur_sigma":
            BLUR_SIGMA,

        "edge_smooth_sigma":
            EDGE_SMOOTH_SIGMA
    },

    "statistics": {

        "floor_pixels":
            int(
                floor_main.sum()
            ),

        "screeding_pixels":
            int(
                screeding_clean.sum()
            ),

        "combined_tile_target_pixels":
            int(
                tile_target.sum()
            ),

        "edge_confidence_threshold":
            conf_threshold
    },

    "outputs": {

        "floor":
            str(
                FLOOR_PATH
            ),

        "screeding":
            str(
                SCREEDING_PATH
            ),

        "floor_plus_screeding":
            str(
                TARGET_PATH
            ),

        "audit":
            str(
                AUDIT_PATH
            )
    },

    "status":
        "REQUIRES_SCREEEDING_VISUAL_AUDIT",

    "next_if_pass":
        "08C_METRIC_GRID_CALIBRATION"
}


STATE_PATH = (
    OUT
    / "00_stage08b2_result.json"
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
print("STAGE 08B2 RESULT")
print("=" * 110)

print()
print(
    "FLOOR PIXELS:",
    int(
        floor_main.sum()
    )
)

print(
    "SCREEDING PIXELS:",
    int(
        screeding_clean.sum()
    )
)

print(
    "COMBINED TILE TARGET PIXELS:",
    int(
        tile_target.sum()
    )
)

print()
print(
    "SCREEDING MASK:",
    SCREEDING_PATH
)

print(
    "COMBINED TARGET:",
    TARGET_PATH
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
    "NO TILE WAS APPLIED."
)
