
from pathlib import Path
import json
import hashlib

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


EMPTY_ROOM_PATH = (
    PROD
    / "stage07_empty_room"
    / "07a_qwen_empty_room"
    / "00_stage07_empty_room_candidate.png"
)


FLOOR_MASK_PATH = (
    PROD
    / "stage02_structure"
    / "17_floor_majority_2of3.png"
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
    / "08a_floor_input_geometry_audit"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# TILE SPECIFICATION — FROZEN FOR THIS TEST
# ============================================================

TILE_SPEC = {

    "width_mm":
        600,

    "length_mm":
        1200,

    "surface":
        "GLOSSY",

    "target_surface":
        "FLOOR_ONLY",

    "orientation":
        "VERTICAL",

    "grout":
        True,

    "grout_mm":
        5,
}


# ============================================================
# VALIDATE INPUTS
# ============================================================

for path in [
    EMPTY_ROOM_PATH,
    FLOOR_MASK_PATH,
    TILE_PATH,
]:

    if not path.exists():

        raise FileNotFoundError(
            path
        )


# ============================================================
# HASH HELPER
# ============================================================

def sha256_file(path):

    h = hashlib.sha256()

    with open(
        path,
        "rb"
    ) as f:

        while True:

            chunk = f.read(
                1024 * 1024
            )

            if not chunk:
                break

            h.update(
                chunk
            )

    return h.hexdigest()


# ============================================================
# LOAD INPUTS
# ============================================================

room = Image.open(
    EMPTY_ROOM_PATH
).convert(
    "RGB"
)


tile = Image.open(
    TILE_PATH
).convert(
    "RGB"
)


floor_mask_img = Image.open(
    FLOOR_MASK_PATH
).convert(
    "L"
)


W, H = room.size


if floor_mask_img.size != (
    W,
    H
):

    raise RuntimeError(
        (
            "Floor mask size does not match room: "
            f"{floor_mask_img.size} vs {(W, H)}"
        )
    )


room_np = np.asarray(
    room
)


floor_mask = (
    np.asarray(
        floor_mask_img
    )
    >
    127
)


# ============================================================
# BASIC INPUT REPORT
# ============================================================

print("=" * 110)
print("STAGE 08A — FLOOR INPUT + GEOMETRY AUDIT")
print("=" * 110)

print()
print(
    "EMPTY ROOM:",
    EMPTY_ROOM_PATH
)

print(
    "ROOM SIZE:",
    f"{W} × {H}"
)

print(
    "ROOM SHA256:",
    sha256_file(
        EMPTY_ROOM_PATH
    )
)


print()
print(
    "FLOOR MASK:",
    FLOOR_MASK_PATH
)

print(
    "FLOOR MASK PIXELS:",
    int(
        floor_mask.sum()
    )
)

print(
    "FLOOR FRACTION:",
    round(
        float(
            floor_mask.mean()
        ),
        4
    )
)


print()
print(
    "TILE:",
    TILE_PATH
)

print(
    "TILE IMAGE SIZE:",
    tile.size
)

print(
    "TILE SHA256:",
    sha256_file(
        TILE_PATH
    )
)


print()
print(
    "TILE SPEC:",
    TILE_SPEC
)


# ============================================================
# FIND FLOOR CONTOURS
# ============================================================

mask_u8 = (
    floor_mask.astype(
        np.uint8
    )
    *
    255
)


contours, hierarchy = cv2.findContours(
    mask_u8,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)


if not contours:

    raise RuntimeError(
        "No floor contour found."
    )


# Main floor region = largest external contour.
main_contour = max(
    contours,
    key=cv2.contourArea
)


main_area = float(
    cv2.contourArea(
        main_contour
    )
)


x, y, bw, bh = cv2.boundingRect(
    main_contour
)


# ============================================================
# POLYGON APPROXIMATION
#
# Diagnostic only.
# We are NOT yet using this as final metric geometry.
# ============================================================

perimeter = cv2.arcLength(
    main_contour,
    True
)


approx = cv2.approxPolyDP(
    main_contour,
    0.015 * perimeter,
    True
)


polygon_points = [
    [
        int(
            pt[0][0]
        ),
        int(
            pt[0][1]
        )
    ]
    for pt in approx
]


print()
print("=" * 110)
print("FLOOR GEOMETRY")
print("=" * 110)

print(
    "MAIN CONTOUR AREA:",
    round(
        main_area,
        2
    )
)

print(
    "FLOOR BBOX:",
    [
        x,
        y,
        x + bw,
        y + bh
    ]
)

print(
    "APPROX POLYGON POINTS:",
    len(
        polygon_points
    )
)

for i, p in enumerate(
    polygon_points,
    start=1
):

    print(
        f"  P{i}:",
        p
    )


# ============================================================
# FLOOR BOUNDARY IMAGE
# ============================================================

boundary = np.zeros(
    (
        H,
        W
    ),
    dtype=np.uint8
)


cv2.drawContours(
    boundary,
    [
        main_contour
    ],
    -1,
    255,
    1
)


BOUNDARY_PATH = (
    OUT
    / "01_floor_boundary.png"
)


Image.fromarray(
    boundary
).save(
    BOUNDARY_PATH
)


# ============================================================
# FLOOR MASK OVERLAY
# ============================================================

overlay = room_np.copy()


green = np.zeros_like(
    overlay
)

green[
    ...,
    1
] = 255


alpha = 0.42


overlay[
    floor_mask
] = (

    overlay[
        floor_mask
    ].astype(
        np.float32
    )
    *
    (
        1.0
        -
        alpha
    )

    +

    green[
        floor_mask
    ].astype(
        np.float32
    )
    *
    alpha

).astype(
    np.uint8
)


# Draw boundary in red.
overlay_boundary = overlay.copy()


overlay_boundary[
    boundary > 0
] = [
    255,
    0,
    0
]


# ============================================================
# POLYGON PREVIEW
# ============================================================

polygon_preview = room_np.copy()


for i, (
    px,
    py
) in enumerate(
    polygon_points,
    start=1
):

    cv2.circle(
        polygon_preview,
        (
            px,
            py
        ),
        3,
        (
            255,
            0,
            0
        ),
        -1
    )


    cv2.putText(
        polygon_preview,
        str(
            i
        ),
        (
            px + 4,
            py - 4
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (
            255,
            0,
            0
        ),
        1,
        cv2.LINE_AA
    )


# ============================================================
# TILE ORIENTATION PREVIEW
#
# No projection yet.
#
# This simply confirms which side is treated as:
#
# horizontal floor-axis = 600 mm
# depth/vertical-axis    = 1200 mm
# ============================================================

tile_preview = np.asarray(
    tile
).copy()


tile_h, tile_w = tile_preview.shape[:2]


fig, ax = plt.subplots(
    figsize=(
        6,
        8
    )
)


ax.imshow(
    tile
)


ax.set_title(
    "Tile 01 — Locked Source\n"
    "600 mm × 1200 mm | Vertical | Glossy | 5 mm grout"
)


# width annotation
ax.annotate(
    "600 mm",
    xy=(
        tile_w * 0.5,
        tile_h * 0.04
    ),
    ha="center",
    va="top",
    fontsize=10,
    bbox=dict(
        facecolor="white",
        alpha=0.8
    )
)


# long/depth orientation annotation
ax.annotate(
    "1200 mm\nVERTICAL",
    xy=(
        tile_w * 0.05,
        tile_h * 0.5
    ),
    ha="left",
    va="center",
    rotation=90,
    fontsize=10,
    bbox=dict(
        facecolor="white",
        alpha=0.8
    )
)


ax.axis(
    "off"
)


TILE_PREVIEW_PATH = (
    OUT
    / "02_tile_spec_orientation_preview.png"
)


plt.tight_layout()

plt.savefig(
    TILE_PREVIEW_PATH,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


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


# ------------------------------------------------------------
# 1 — Empty room
# ------------------------------------------------------------

axes[0].imshow(
    room
)

axes[0].set_title(
    "1. Stage07A Empty Room"
)

axes[0].axis(
    "off"
)


# ------------------------------------------------------------
# 2 — Floor mask
# ------------------------------------------------------------

axes[1].imshow(
    floor_mask,
    cmap="gray",
    vmin=0,
    vmax=1
)

axes[1].set_title(
    "2. Stage02 Floor Majority Mask"
)

axes[1].axis(
    "off"
)


# ------------------------------------------------------------
# 3 — Mask + boundary
# ------------------------------------------------------------

axes[2].imshow(
    overlay_boundary
)

axes[2].set_title(
    "3. Floor Mask + Boundary\nGREEN=Floor | RED=Boundary"
)

axes[2].axis(
    "off"
)


# ------------------------------------------------------------
# 4 — Tile source
# ------------------------------------------------------------

axes[3].imshow(
    tile
)

axes[3].set_title(
    "4. Tile 01\n600×1200 mm | Vertical"
)

axes[3].axis(
    "off"
)


plt.tight_layout()


AUDIT_PATH = (
    OUT
    / "03_stage08a_floor_geometry_audit.png"
)


plt.savefig(
    AUDIT_PATH,
    dpi=150,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# SEPARATE POLYGON AUDIT
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        8,
        10
    )
)


ax.imshow(
    polygon_preview
)


# contour
contour_xy = main_contour[
    :,
    0,
    :
]


ax.plot(
    contour_xy[
        :,
        0
    ],
    contour_xy[
        :,
        1
    ],
    linewidth=1.5
)


ax.set_title(
    "08A — Floor Contour / Polygon Geometry Audit\n"
    "This is NOT yet the final metric projection"
)

ax.axis(
    "off"
)


POLYGON_PATH = (
    OUT
    / "04_floor_polygon_geometry_audit.png"
)


plt.tight_layout()

plt.savefig(
    POLYGON_PATH,
    dpi=150,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# SAVE STATE
# ============================================================

STATE = {

    "stage":
        "08A",

    "purpose":
        "FLOOR_INPUT_AND_GEOMETRY_AUDIT",

    "inputs": {

        "empty_room":
            str(
                EMPTY_ROOM_PATH
            ),

        "floor_mask":
            str(
                FLOOR_MASK_PATH
            ),

        "tile":
            str(
                TILE_PATH
            ),
    },

    "hashes": {

        "empty_room_sha256":
            sha256_file(
                EMPTY_ROOM_PATH
            ),

        "tile_sha256":
            sha256_file(
                TILE_PATH
            ),
    },

    "image_size": [
        W,
        H
    ],

    "tile_spec":
        TILE_SPEC,

    "floor_geometry": {

        "mask_pixels":
            int(
                floor_mask.sum()
            ),

        "mask_fraction":
            float(
                floor_mask.mean()
            ),

        "main_contour_area":
            main_area,

        "bounding_box": [
            x,
            y,
            x + bw,
            y + bh
        ],

        "approx_polygon":
            polygon_points,
    },

    "orientation_rule": {

        "floor_u_axis_mm":
            600,

        "floor_v_depth_axis_mm":
            1200,

        "meaning":
            (
                "Vertical orientation: long 1200 mm tile "
                "dimension follows the floor depth direction. "
                "This must be visually verified during grid projection."
            )
    },

    "outputs": {

        "floor_boundary":
            str(
                BOUNDARY_PATH
            ),

        "tile_orientation_preview":
            str(
                TILE_PREVIEW_PATH
            ),

        "main_audit":
            str(
                AUDIT_PATH
            ),

        "polygon_audit":
            str(
                POLYGON_PATH
            ),
    },

    "status":
        "REQUIRES_FLOOR_GEOMETRY_VISUAL_AUDIT",

    "next_if_pass":
        "08B_FLOOR_PERSPECTIVE_AND_METRIC_GRID"
}


STATE_PATH = (
    OUT
    / "00_stage08a_result.json"
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
print("STAGE 08A RESULT")
print("=" * 110)

print()
print(
    "ROOM:",
    EMPTY_ROOM_PATH
)

print(
    "FLOOR MASK:",
    FLOOR_MASK_PATH
)

print(
    "TILE:",
    TILE_PATH
)

print()

print(
    "TILE SIZE:",
    "600 × 1200 mm"
)

print(
    "ORIENTATION:",
    "VERTICAL"
)

print(
    "SURFACE:",
    "GLOSSY"
)

print(
    "GROUT:",
    "5 mm"
)

print()

print(
    "FLOOR MASK PIXELS:",
    int(
        floor_mask.sum()
    )
)

print(
    "MAIN CONTOUR AREA:",
    round(
        main_area,
        2
    )
)

print(
    "APPROX POLYGON POINT COUNT:",
    len(
        polygon_points
    )
)

print()

print(
    "MAIN AUDIT:",
    AUDIT_PATH
)

print(
    "POLYGON AUDIT:",
    POLYGON_PATH
)

print(
    "STATE:",
    STATE_PATH
)

print()
print(
    "NO TILE APPLICATION WAS PERFORMED IN 08A."
)
