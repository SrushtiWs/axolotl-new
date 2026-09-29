
from pathlib import Path
import json
import hashlib

import cv2
import numpy as np
from PIL import Image, ImageDraw
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

MASTER = (
    PROD
    / "stage01_master"
    / "00_master_input.png"
)


# ------------------------------------------------------------
# Frozen successful V1 clean-room reconstruction
# ------------------------------------------------------------

V1 = (
    BASE
    / "test07"
    / "runs"
    / "room03_bathroom"
    / "stages"
    / "01z1_qwen_coherent_cleanroom_v1"
    / "02_safe_props_restored.png"
)


# ------------------------------------------------------------
# Verified Stage05C independent behind-glass restore mask
# ------------------------------------------------------------

RESTORE_MASK = (
    PROD
    / "stage05_clean_room_with_props"
    / "09_independent_behind_glass_restore_mask.png"
)


# ------------------------------------------------------------
# Glass-owned mask only for diagnostic verification
# ------------------------------------------------------------

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
    / "10_final_canonical_prop_detection_master.png"
)


STATE_PATH = (
    OUT
    / "10_final_canonical_prop_detection_master_state.json"
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

    plt.imshow(image)

    plt.title(title)

    plt.axis("off")

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
            mask.astype(np.uint8)
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


def sha256_image(
    image
):

    return hashlib.sha256(
        np.asarray(
            image
        ).tobytes()
    ).hexdigest()


# ============================================================
# VALIDATE
# ============================================================

for path in [
    MASTER,
    V1,
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


master = np.asarray(
    master_pil
)

v1 = np.asarray(
    v1_pil
)


H, W = master.shape[:2]


if v1.shape != master.shape:

    raise RuntimeError(
        f"V1 size mismatch: {v1.shape} vs {master.shape}"
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
# CREATE FINAL DETECTION MASTER
# ============================================================

final = v1.copy()


# ------------------------------------------------------------
# Restore only verified independent prop pixels.
#
# DO NOT restore Stage04-owned towel bar.
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
# EXACT RESTORE CHECK
# ============================================================

if restore_mask.any():

    diff = np.abs(
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

    max_restore_rgb_error = int(
        diff.max()
    )

else:

    max_restore_rgb_error = 0


# ============================================================
# DIAGNOSTIC PREVIEW
#
# GREEN = independent prop restored
# RED   = glass-owned component intentionally not restored
# ============================================================

ownership = final_pil.copy()


ownership = draw_contours(
    ownership,
    restore_mask,
    "lime",
    3
)


ownership = draw_contours(
    ownership,
    glass_owned_mask,
    "red",
    3
)


OWNERSHIP_PATH = (
    OUT
    / "10_final_detection_master_ownership_preview.png"
)


ownership.save(
    OWNERSHIP_PATH
)


# ============================================================
# SIDE-BY-SIDE COMPARISON IMAGE
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


COMPARISON_PATH = (
    OUT
    / "10_v1_vs_final_detection_master.png"
)


comparison.save(
    COMPARISON_PATH
)


# ============================================================
# STATE
# ============================================================

state = {

    "stage":
        "05D",

    "status":
        "REQUIRES_FINAL_VISUAL_AUDIT",

    "purpose":
        "canonical Stage06 neutral prop-detection master",

    "base_image":
        str(
            V1
        ),

    "stage01_rgb_source":
        str(
            MASTER
        ),

    "independent_behind_glass_restore_mask":
        str(
            RESTORE_MASK
        ),

    "glass_owned_excluded_mask":
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

    "restore_pixels":
        int(
            restore_mask.sum()
        ),

    "glass_owned_pixels_excluded":
        int(
            glass_owned_mask.sum()
        ),

    "max_rgb_error_on_restored_pixels":
        max_restore_rgb_error,

    "output_sha256_pixels":
        sha256_image(
            final_pil
        ),

    "production_contract": {

        "stage06_detection_input":
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

        "connected_prop_rule":
            (
                "all visible physical props that touch, "
                "rest on, overlap, attach to, mount to, or "
                "physically connect with each other belong "
                "to one connected main-prop group"
            ),
    },
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

print("=" * 110)
print("PRODUCTION STAGE 05D RESULT")
print("=" * 110)

print()
print(
    "BASE V1:",
    V1
)

print(
    "FINAL:",
    FINAL_PATH
)

print(
    "SIZE:",
    (W, H)
)

print(
    "RESTORED INDEPENDENT PIXELS:",
    int(
        restore_mask.sum()
    )
)

print(
    "EXCLUDED GLASS-OWNED PIXELS:",
    int(
        glass_owned_mask.sum()
    )
)

print(
    "RESTORED PIXEL MAX RGB ERROR:",
    max_restore_rgb_error
)

print()
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
print("INLINE FINAL STAGE05 AUDIT")
print("=" * 110)


show(
    v1_pil,
    "Stage05D BASE — Frozen V1 Clean Room"
)


show(
    final_pil,
    "Stage05D — FINAL CANONICAL PROP-DETECTION MASTER"
)


show(
    ownership,
    (
        "Stage05D Ownership Audit "
        "GREEN=Independent Shower Prop Restored | "
        "RED=Glass-Owned Hardware Excluded"
    )
)


show(
    comparison,
    (
        "Stage05D Comparison "
        "LEFT=V1 Base | RIGHT=Final Detection Master"
    ),
    figsize=(14, 8)
)
