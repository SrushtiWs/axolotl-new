
from pathlib import Path
import json
import hashlib

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


# Exact original RGB truth
MASTER = (
    PROD
    / "stage01_master"
    / "00_master_input.png"
)


# Best neutral clean-room base
V1 = (
    BASE
    / "test07"
    / "runs"
    / "room03_bathroom"
    / "stages"
    / "01z1_qwen_coherent_cleanroom_v1"
    / "02_safe_props_restored.png"
)


# Stage05B successfully erased glass / towel bar,
# although its floor was not ideal.
# We use ONLY its pixels inside Stage04-owned mask.
STAGE05B = (
    PROD
    / "stage05_clean_room_with_props"
    / "08_neutral_prop_detection_master.png"
)


# Independent shower fixture mask
RESTORE_MASK = (
    PROD
    / "stage05_clean_room_with_props"
    / "09_independent_behind_glass_restore_mask.png"
)


# Glass-owned horizontal bar mask
GLASS_OWNED_MASK = (
    PROD
    / "stage05_clean_room_with_props"
    / "09_glass_owned_excluded_mask.png"
)


OUT = (
    PROD
    / "stage05_clean_room_with_props"
)


FINAL_PATH = (
    OUT
    / "11_final_canonical_prop_detection_master.png"
)


STATE_PATH = (
    OUT
    / "11_final_canonical_prop_detection_master_state.json"
)


REMOVAL_PREVIEW_PATH = (
    OUT
    / "11_glass_owned_removal_preview.png"
)


OWNERSHIP_PREVIEW_PATH = (
    OUT
    / "11_final_ownership_preview.png"
)


COMPARISON_PATH = (
    OUT
    / "11_v1_vs_final_comparison.png"
)


# ============================================================
# HELPERS
# ============================================================

def show(
    image,
    title,
    figsize=(8, 9)
):

    plt.figure(
        figsize=figsize
    )

    plt.imshow(
        image
    )

    plt.title(
        title
    )

    plt.axis(
        "off"
    )

    plt.show()


def draw_contours(
    image,
    mask,
    color,
    width=2
):

    result = image.copy()

    draw = ImageDraw.Draw(
        result
    )


    contours, _ = cv2.findContours(
        (
            mask.astype(
                np.uint8
            )
            *
            255
        ),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    for contour in contours:

        pts = [

            (
                int(p[0][0]),
                int(p[0][1])
            )

            for p in contour
        ]


        if len(pts) >= 2:

            draw.line(
                pts + [pts[0]],
                fill=color,
                width=width
            )


    return result


def sha256_pixels(
    array
):

    return hashlib.sha256(
        array.tobytes()
    ).hexdigest()


# ============================================================
# VALIDATE
# ============================================================

for path in [
    MASTER,
    V1,
    STAGE05B,
    RESTORE_MASK,
    GLASS_OWNED_MASK,
]:

    if not path.exists():

        raise FileNotFoundError(
            path
        )


# ============================================================
# LOAD
# ============================================================

master_pil = Image.open(
    MASTER
).convert(
    "RGB"
)

v1_pil = Image.open(
    V1
).convert(
    "RGB"
)

stage05b_pil = Image.open(
    STAGE05B
).convert(
    "RGB"
)


master = np.asarray(
    master_pil
)

v1 = np.asarray(
    v1_pil
)

stage05b = np.asarray(
    stage05b_pil
)


H, W = master.shape[:2]


for name, image in [
    ("V1", v1),
    ("Stage05B", stage05b),
]:

    if image.shape != master.shape:

        raise RuntimeError(
            f"{name} size mismatch: "
            f"{image.shape} != {master.shape}"
        )


restore_mask = (
    np.asarray(
        Image.open(
            RESTORE_MASK
        ).convert(
            "L"
        )
    )
    >
    0
)


glass_owned_mask = (
    np.asarray(
        Image.open(
            GLASS_OWNED_MASK
        ).convert(
            "L"
        )
    )
    >
    0
)


if restore_mask.shape != (H, W):

    raise RuntimeError(
        "Restore mask size mismatch."
    )


if glass_owned_mask.shape != (H, W):

    raise RuntimeError(
        "Glass-owned mask size mismatch."
    )


# ============================================================
# SAFETY CHECK
#
# The two ownership masks should not meaningfully overlap.
# ============================================================

ownership_overlap = int(
    (
        restore_mask
        &
        glass_owned_mask
    ).sum()
)


print("=" * 110)
print("PRODUCTION STAGE 05E")
print("CORRECTED FINAL CANONICAL PROP-DETECTION MASTER")
print("=" * 110)

print()
print(
    "INDEPENDENT RESTORE PIXELS:",
    int(
        restore_mask.sum()
    )
)

print(
    "GLASS-OWNED REMOVE PIXELS:",
    int(
        glass_owned_mask.sum()
    )
)

print(
    "OWNERSHIP MASK OVERLAP:",
    ownership_overlap
)


# ============================================================
# BUILD FINAL IMAGE
# ============================================================

final = v1.copy()


# ------------------------------------------------------------
# STEP 1
# REMOVE GLASS-OWNED HARDWARE
#
# Use the already-clean Stage05B pixels only in the exact
# Stage04-owned component mask.
# ------------------------------------------------------------

final[
    glass_owned_mask
] = stage05b[
    glass_owned_mask
]


# ------------------------------------------------------------
# STEP 2
# RESTORE INDEPENDENT SHOWER-AREA PROPS
#
# Exact original Stage01 RGB.
# ------------------------------------------------------------

final[
    restore_mask
] = master[
    restore_mask
]


final_pil = Image.fromarray(
    final
)


final_pil.save(
    FINAL_PATH
)


# ============================================================
# VALIDATE EXACT RESTORE
# ============================================================

if restore_mask.any():

    restore_diff = np.abs(

        final[
            restore_mask
        ].astype(
            np.int16
        )

        -

        master[
            restore_mask
        ].astype(
            np.int16
        )
    )

    restore_max_error = int(
        restore_diff.max()
    )

else:

    restore_max_error = 0


# ============================================================
# VALIDATE GLASS-OWNED REPLACEMENT
# ============================================================

if glass_owned_mask.any():

    glass_diff = np.abs(

        final[
            glass_owned_mask
        ].astype(
            np.int16
        )

        -

        stage05b[
            glass_owned_mask
        ].astype(
            np.int16
        )
    )

    glass_replace_max_error = int(
        glass_diff.max()
    )

else:

    glass_replace_max_error = 0


# ============================================================
# PREVIEW 1
# SHOW WHAT WAS REMOVED
# ============================================================

removal_preview = v1_pil.copy()


removal_preview = draw_contours(
    removal_preview,
    glass_owned_mask,
    "red",
    4
)


Image.fromarray(
    np.asarray(
        removal_preview
    )
).save(
    REMOVAL_PREVIEW_PATH
)


# ============================================================
# PREVIEW 2
# FINAL OWNERSHIP
#
# GREEN = independent shower prop present
# RED   = Stage04-owned region that must remain clean
# ============================================================

ownership_preview = final_pil.copy()


ownership_preview = draw_contours(
    ownership_preview,
    restore_mask,
    "lime",
    3
)


ownership_preview = draw_contours(
    ownership_preview,
    glass_owned_mask,
    "red",
    3
)


ownership_preview.save(
    OWNERSHIP_PREVIEW_PATH
)


# ============================================================
# COMPARISON
# ============================================================

comparison = Image.new(
    "RGB",
    (
        W * 2,
        H
    ),
    "white"
)


comparison.paste(
    v1_pil,
    (0, 0)
)


comparison.paste(
    final_pil,
    (W, 0)
)


comparison.save(
    COMPARISON_PATH
)


# ============================================================
# STATE
# ============================================================

state = {

    "stage":
        "05E",

    "purpose":
        (
            "final corrected neutral Stage06 "
            "prop-detection master"
        ),

    "status":
        "REQUIRES_FINAL_VISUAL_AUDIT",

    "base":
        str(
            V1
        ),

    "glass_owned_background_source":
        str(
            STAGE05B
        ),

    "original_rgb_source":
        str(
            MASTER
        ),

    "independent_restore_mask":
        str(
            RESTORE_MASK
        ),

    "glass_owned_remove_mask":
        str(
            GLASS_OWNED_MASK
        ),

    "output":
        str(
            FINAL_PATH
        ),

    "dimensions":
        [
            W,
            H
        ],

    "independent_restore_pixels":
        int(
            restore_mask.sum()
        ),

    "glass_owned_removed_pixels":
        int(
            glass_owned_mask.sum()
        ),

    "ownership_overlap_pixels":
        ownership_overlap,

    "restore_max_rgb_error":
        restore_max_error,

    "glass_replacement_max_rgb_error":
        glass_replace_max_error,

    "sha256_pixels":
        sha256_pixels(
            final
        ),

    "production_contract": {

        "canonical_stage06_detection_input":
            str(
                FINAL_PATH
            ),

        "final_prop_rgb_source":
            str(
                MASTER
            ),

        "mirror_owner":
            "STAGE03",

        "glass_owner":
            "STAGE04",

        "stage06_connected_group_rule":
            (
                "visible props that touch, rest on, overlap, "
                "mount to, attach to, or physically connect "
                "to each other are extracted as one connected "
                "main-prop group"
            ),
    },

    "rules": [

        (
            "glass-owned components must not appear in "
            "Stage06 detection master"
        ),

        (
            "independent props visible behind removed glass "
            "must remain in Stage06 detection master"
        ),

        (
            "Stage05 is understanding/detection source only"
        ),

        (
            "Stage01 remains final exact RGB source"
        ),
    ],
}


STATE_PATH.write_text(
    json.dumps(
        state,
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
print("PRODUCTION STAGE 05E RESULT")
print("=" * 110)

print()
print(
    "FINAL:",
    FINAL_PATH
)

print(
    "RESTORE RGB ERROR:",
    restore_max_error
)

print(
    "GLASS REPLACEMENT RGB ERROR:",
    glass_replace_max_error
)

print(
    "OWNERSHIP OVERLAP:",
    ownership_overlap
)

print(
    "STATE:",
    STATE_PATH
)

print()
print(
    "NO AI MODEL WAS RUN."
)

print(
    "NO STAGE06 PROP MASK WAS CREATED."
)


# ============================================================
# INLINE VISUAL VERIFICATION
# ============================================================

print()
print("=" * 110)
print("INLINE FINAL STAGE05E AUDIT")
print("=" * 110)


show(
    v1_pil,
    (
        "BEFORE — Frozen V1 "
        "(Glass-Owned Bar Still Present)"
    )
)


show(
    final_pil,
    (
        "AFTER — Stage05E FINAL CANONICAL "
        "PROP-DETECTION MASTER"
    )
)


show(
    ownership_preview,
    (
        "Stage05E Ownership Audit "
        "GREEN=Keep for Stage06 | "
        "RED=Removed / Owned by Stage04"
    )
)


show(
    comparison,
    (
        "Stage05E Comparison "
        "LEFT=V1 | RIGHT=Corrected Final"
    ),
    figsize=(14, 8)
)
