
from pathlib import Path
import json

import cv2
import numpy as np

from PIL import (
    Image,
    ImageDraw
)

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


MASTER_PATH = (
    PROD
    / "stage01_master"
    / "00_master_input.png"
)


FLOOR_MASK_PATH = (
    PROD
    / "stage08_tile_application"
    / "08b2_floor_plus_screeding_target"
    / "01_main_floor_mask.png"
)


P01_MASK_PATH = (
    PROD
    / "stage06_prop_layer"
    / "06f3_vanity_internal_completion"
    / "01_p01_completed_vanity_mask.png"
)


P02_MASK_PATH = (
    PROD
    / "stage06_prop_layer"
    / "06d2c1_complete_main_prop_sam2_multimask_audit"
    / "objects"
    / "P02_candidate_3_mask.png"
)


E1_DIR = (
    PROD
    / "stage07_empty_room"
    / "07e1_qwen_semantic_shadow_localization"
)


PARSED_PATH = (
    E1_DIR
    / "02_qwen_shadow_parsed.json"
)


OUT = (
    PROD
    / "stage07_empty_room"
    / "07e1b_qwen_coordinate_fix"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# VALIDATE
# ============================================================

for p in [
    MASTER_PATH,
    FLOOR_MASK_PATH,
    P01_MASK_PATH,
    P02_MASK_PATH,
    PARSED_PATH,
]:

    if not p.exists():
        raise FileNotFoundError(p)


# ============================================================
# LOAD
# ============================================================

master = Image.open(
    MASTER_PATH
).convert("RGB")

master_np = np.asarray(
    master
)

W, H = master.size


floor = (
    np.asarray(
        Image.open(
            FLOOR_MASK_PATH
        ).convert("L")
    ) > 127
)


p01 = (
    np.asarray(
        Image.open(
            P01_MASK_PATH
        ).convert("L")
    ) > 127
)


p02 = (
    np.asarray(
        Image.open(
            P02_MASK_PATH
        ).convert("L")
    ) > 127
)


parsed = json.loads(
    PARSED_PATH.read_text(
        encoding="utf-8"
    )
)


print("=" * 110)
print("STAGE 07E1B — QWEN COORDINATE FIX")
print("=" * 110)

print()
print(
    "IMAGE SIZE:",
    W,
    "x",
    H
)


# ============================================================
# COORDINATE MODE DETECTION
#
# Possible modes:
#
# 1. native_pixels
#    x roughly 0..W
#    y roughly 0..H
#
# 2. normalized_1000
#    coordinates potentially much larger than image size,
#    intended as 0..1000.
#
# We inspect ALL returned polygon points.
# ============================================================

def collect_points(
    parsed_data
):

    points = []

    for key in [
        "P01",
        "P02"
    ]:

        record = parsed_data.get(
            key,
            {}
        )

        for poly in record.get(
            "polygons",
            []
        ):

            for pt in poly:

                if (
                    isinstance(
                        pt,
                        list
                    )
                    and
                    len(pt) >= 2
                ):

                    try:

                        x = float(
                            pt[0]
                        )

                        y = float(
                            pt[1]
                        )

                        points.append(
                            (
                                x,
                                y
                            )
                        )

                    except Exception:
                        pass

    return points


all_points = collect_points(
    parsed
)


if not all_points:

    raise RuntimeError(
        "No polygon points found in Qwen JSON."
    )


max_x = max(
    p[0]
    for p in all_points
)

max_y = max(
    p[1]
    for p in all_points
)


print(
    "RAW MAX X:",
    max_x
)

print(
    "RAW MAX Y:",
    max_y
)


# ------------------------------------------------------------
# Native-pixel test
#
# Allow modest tolerance because model might output coordinate
# exactly on image boundary or slightly outside.
# ------------------------------------------------------------

native_like = (
    max_x <= W * 1.15
    and
    max_y <= H * 1.15
)


if native_like:

    COORD_MODE = (
        "native_pixels"
    )

else:

    COORD_MODE = (
        "normalized_1000"
    )


print()
print(
    "DETECTED COORDINATE MODE:",
    COORD_MODE
)


# ============================================================
# POINT CONVERSION
# ============================================================

def convert_point(
    x,
    y
):

    x = float(x)
    y = float(y)


    if COORD_MODE == "native_pixels":

        px = int(
            round(
                np.clip(
                    x,
                    0,
                    W - 1
                )
            )
        )

        py = int(
            round(
                np.clip(
                    y,
                    0,
                    H - 1
                )
            )
        )


    else:

        px = int(
            round(
                np.clip(
                    x,
                    0,
                    1000
                )
                /
                1000.0
                *
                (W - 1)
            )
        )

        py = int(
            round(
                np.clip(
                    y,
                    0,
                    1000
                )
                /
                1000.0
                *
                (H - 1)
            )
        )


    return (
        px,
        py
    )


# ============================================================
# RASTERIZE ONE RECORD
# ============================================================

def record_to_raw_mask(
    name,
    record
):

    canvas = Image.new(
        "L",
        (
            W,
            H
        ),
        0
    )


    draw = ImageDraw.Draw(
        canvas
    )


    print()
    print("-" * 110)
    print(name)
    print("-" * 110)


    print(
        "VISIBLE:",
        record.get(
            "shadow_visible"
        )
    )


    print(
        "CONFIDENCE:",
        record.get(
            "confidence"
        )
    )


    print(
        "REASON:",
        record.get(
            "reason"
        )
    )


    if not record.get(
        "shadow_visible",
        False
    ):

        return np.zeros(
            (
                H,
                W
            ),
            dtype=bool
        )


    polygons = record.get(
        "polygons",
        []
    )


    print(
        "POLYGONS:",
        len(polygons)
    )


    for i, poly in enumerate(
        polygons,
        start=1
    ):

        pts = []


        for point in poly:

            if (
                not isinstance(
                    point,
                    list
                )
                or
                len(point) < 2
            ):

                continue


            try:

                px, py = convert_point(
                    point[0],
                    point[1]
                )

            except Exception:

                continue


            pts.append(
                (
                    px,
                    py
                )
            )


        print(
            f"POLYGON {i} PIXELS:",
            pts
        )


        if len(
            pts
        ) >= 3:

            draw.polygon(
                pts,
                fill=255
            )


    return (
        np.asarray(
            canvas
        ) > 127
    )


# ============================================================
# RAW CORRECTLY INTERPRETED MASKS
# ============================================================

p01_raw = record_to_raw_mask(
    "P01 VANITY",
    parsed.get(
        "P01",
        {}
    )
)


p02_raw = record_to_raw_mask(
    "P02 TOILET",
    parsed.get(
        "P02",
        {}
    )
)


# ============================================================
# FLOOR CLIPPING
# ============================================================

p01_floor = (
    p01_raw
    &
    floor
)


p02_floor = (
    p02_raw
    &
    floor
)


# ============================================================
# REMOVE PHYSICAL PROP PIXELS
# ============================================================

all_props = (
    p01
    |
    p02
)


p01_floor &= (
    ~all_props
)


p02_floor &= (
    ~all_props
)


# ============================================================
# OBJECT PROXIMITY SAFETY REGION
#
# Semantic AI decides WHERE.
# Geometry only prevents implausibly distant detections.
# ============================================================

def proximity_region(
    prop,
    radius
):

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            radius * 2 + 1,
            radius * 2 + 1
        )
    )


    dilated = (
        cv2.dilate(
            prop.astype(
                np.uint8
            ) * 255,
            kernel
        ) > 0
    )


    return (
        dilated
        &
        floor
    )


p01_allowed = proximity_region(
    p01,
    85
)


p02_allowed = proximity_region(
    p02,
    65
)


p01_final = (
    p01_floor
    &
    p01_allowed
)


p02_final = (
    p02_floor
    &
    p02_allowed
)


# ============================================================
# VERY LIGHT CLEANUP
# ============================================================

def cleanup(
    mask
):

    u8 = (
        mask.astype(
            np.uint8
        ) * 255
    )


    u8 = cv2.morphologyEx(
        u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                3,
                3
            )
        )
    )


    return (
        u8 > 0
    )


p01_final = cleanup(
    p01_final
)


p02_final = cleanup(
    p02_final
)


combined = (
    p01_final
    |
    p02_final
)


# ============================================================
# PRINT PIXEL COUNTS
# ============================================================

print()
print("=" * 110)
print("PIXEL FLOW")
print("=" * 110)

print()

print(
    "P01 RAW:",
    int(
        p01_raw.sum()
    )
)

print(
    "P01 AFTER FLOOR:",
    int(
        p01_floor.sum()
    )
)

print(
    "P01 FINAL:",
    int(
        p01_final.sum()
    )
)


print()

print(
    "P02 RAW:",
    int(
        p02_raw.sum()
    )
)

print(
    "P02 AFTER FLOOR:",
    int(
        p02_floor.sum()
    )
)

print(
    "P02 FINAL:",
    int(
        p02_final.sum()
    )
)


# ============================================================
# SAVE
# ============================================================

P01_PATH = (
    OUT
    / "01_p01_shadow_mask.png"
)


P02_PATH = (
    OUT
    / "02_p02_shadow_mask.png"
)


COMBINED_PATH = (
    OUT
    / "03_combined_shadow_mask.png"
)


Image.fromarray(
    p01_final.astype(
        np.uint8
    ) * 255
).save(
    P01_PATH
)


Image.fromarray(
    p02_final.astype(
        np.uint8
    ) * 255
).save(
    P02_PATH
)


Image.fromarray(
    combined.astype(
        np.uint8
    ) * 255
).save(
    COMBINED_PATH
)


# ============================================================
# LOCATION OVERLAY
# ============================================================

overlay = master_np.astype(
    np.float32
).copy()


overlay[
    p01_final
] = (
    overlay[
        p01_final
    ] * 0.35
    +
    np.array(
        [
            255,
            0,
            0
        ],
        dtype=np.float32
    ) * 0.65
)


overlay[
    p02_final
] = (
    overlay[
        p02_final
    ] * 0.35
    +
    np.array(
        [
            0,
            255,
            255
        ],
        dtype=np.float32
    ) * 0.65
)


overlay = np.clip(
    overlay,
    0,
    255
).astype(
    np.uint8
)


OVERLAY_PATH = (
    OUT
    / "04_shadow_location_overlay.png"
)


Image.fromarray(
    overlay
).save(
    OVERLAY_PATH
)


# ============================================================
# TEMPORARY NEUTRAL SURFACE TEST
#
# Still testing LOCATION, not final softness.
# ============================================================

mult = np.ones(
    (
        H,
        W
    ),
    dtype=np.float32
)


mult[
    p01_final
] = 0.80


mult[
    p02_final
] = 0.74


neutral = np.full(
    (
        H,
        W,
        3
    ),
    210.0,
    dtype=np.float32
)


neutral[
    floor
] *= mult[
        floor,
        None
    ]


neutral = np.clip(
    neutral,
    0,
    255
).astype(
    np.uint8
)


# ============================================================
# 8-PANEL AUDIT
# ============================================================

fig, axes = plt.subplots(
    1,
    8,
    figsize=(
        34,
        7
    )
)


axes[0].imshow(
    master
)

axes[0].set_title(
    "1. Original"
)

axes[0].axis(
    "off"
)


axes[1].imshow(
    p01_raw,
    cmap="gray"
)

axes[1].set_title(
    "2. P01 Raw\nCorrect Coordinate Mode"
)

axes[1].axis(
    "off"
)


axes[2].imshow(
    p01_final,
    cmap="gray"
)

axes[2].set_title(
    "3. P01 Final"
)

axes[2].axis(
    "off"
)


axes[3].imshow(
    p02_raw,
    cmap="gray"
)

axes[3].set_title(
    "4. P02 Raw\nCorrect Coordinate Mode"
)

axes[3].axis(
    "off"
)


axes[4].imshow(
    p02_final,
    cmap="gray"
)

axes[4].set_title(
    "5. P02 Final"
)

axes[4].axis(
    "off"
)


axes[5].imshow(
    combined,
    cmap="gray"
)

axes[5].set_title(
    "6. Combined"
)

axes[5].axis(
    "off"
)


axes[6].imshow(
    overlay
)

axes[6].set_title(
    "7. Location Audit\n"
    "RED=P01 | CYAN=P02"
)

axes[6].axis(
    "off"
)


axes[7].imshow(
    neutral
)

axes[7].set_title(
    "8. Neutral Position Test"
)

axes[7].axis(
    "off"
)


plt.tight_layout()


AUDIT_PATH = (
    OUT
    / "05_stage07e1b_coordinate_fix_audit.png"
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
        "07E1B",

    "purpose":
        "CORRECT_QWEN_SHADOW_COORDINATE_INTERPRETATION",

    "model_rerun":
        False,

    "source":
        str(
            PARSED_PATH
        ),

    "coordinate_mode":
        COORD_MODE,

    "image_size": [
        W,
        H
    ],

    "statistics": {

        "p01_raw":
            int(
                p01_raw.sum()
            ),

        "p01_floor":
            int(
                p01_floor.sum()
            ),

        "p01_final":
            int(
                p01_final.sum()
            ),

        "p02_raw":
            int(
                p02_raw.sum()
            ),

        "p02_floor":
            int(
                p02_floor.sum()
            ),

        "p02_final":
            int(
                p02_final.sum()
            ),

        "combined":
            int(
                combined.sum()
            )
    },

    "outputs": {

        "p01":
            str(
                P01_PATH
            ),

        "p02":
            str(
                P02_PATH
            ),

        "combined":
            str(
                COMBINED_PATH
            ),

        "overlay":
            str(
                OVERLAY_PATH
            ),

        "audit":
            str(
                AUDIT_PATH
            )
    },

    "status":
        "REQUIRES_VISUAL_LOCATION_AUDIT"
}


STATE_PATH = (
    OUT
    / "00_stage07e1b_result.json"
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
print("STAGE 07E1B RESULT")
print("=" * 110)

print()

print(
    "COORDINATE MODE:",
    COORD_MODE
)

print()

print(
    "P01 FINAL PIXELS:",
    int(
        p01_final.sum()
    )
)

print(
    "P02 FINAL PIXELS:",
    int(
        p02_final.sum()
    )
)

print(
    "COMBINED:",
    int(
        combined.sum()
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
    "NO MODEL WAS RERUN."
)
