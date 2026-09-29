
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

STAGE05 = (
    PROD
    / "stage05_clean_room_with_props"
)

INPUT = (
    STAGE05
    / "11_final_canonical_prop_detection_master.png"
)

GLASS_OWNED_MASK = (
    STAGE05
    / "09_glass_owned_excluded_mask.png"
)

RESTORE_MASK = (
    STAGE05
    / "09_independent_behind_glass_restore_mask.png"
)

OUTPUT = (
    STAGE05
    / "12_final_canonical_prop_detection_master.png"
)

INPAINT_MASK_PATH = (
    STAGE05
    / "12_glass_hardware_inpaint_mask.png"
)

PREVIEW_PATH = (
    STAGE05
    / "12_final_detection_master_preview.png"
)

STATE_PATH = (
    STAGE05
    / "12_final_detection_master_state.json"
)


# ============================================================
# CONFIG
# ============================================================

DILATE_KERNEL = 7
DILATE_ITERATIONS = 2
INPAINT_RADIUS = 5


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


def draw_contour(
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


# ============================================================
# VALIDATE
# ============================================================

for path in [
    INPUT,
    GLASS_OWNED_MASK,
    RESTORE_MASK,
]:

    if not path.exists():

        raise FileNotFoundError(
            path
        )


# ============================================================
# LOAD
# ============================================================

input_pil = Image.open(
    INPUT
).convert(
    "RGB"
)

image = np.asarray(
    input_pil
).copy()


H, W = image.shape[:2]


glass_mask = (
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


if glass_mask.shape != (H, W):

    raise RuntimeError(
        "Glass-owned mask size mismatch."
    )


if restore_mask.shape != (H, W):

    raise RuntimeError(
        "Restore mask size mismatch."
    )


# ============================================================
# EXPAND GLASS-HARDWARE REGION
#
# Need enough margin to remove the bright white halo too.
# ============================================================

kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (
        DILATE_KERNEL,
        DILATE_KERNEL
    )
)


expanded = cv2.dilate(
    glass_mask.astype(
        np.uint8
    ),
    kernel,
    iterations=DILATE_ITERATIONS
) > 0


# ============================================================
# SAFETY:
# NEVER overwrite independently restored shower props.
# ============================================================

expanded[
    restore_mask
] = False


overlap_after_protection = int(
    (
        expanded
        &
        restore_mask
    ).sum()
)


# ============================================================
# SAVE INPAINT MASK
# ============================================================

inpaint_mask_u8 = (
    expanded.astype(
        np.uint8
    )
    *
    255
)


Image.fromarray(
    inpaint_mask_u8
).save(
    INPAINT_MASK_PATH
)


# ============================================================
# INPAINT
#
# OpenCV expects BGR.
# ============================================================

bgr = cv2.cvtColor(
    image,
    cv2.COLOR_RGB2BGR
)


clean_bgr = cv2.inpaint(
    bgr,
    inpaint_mask_u8,
    INPAINT_RADIUS,
    cv2.INPAINT_TELEA
)


clean = cv2.cvtColor(
    clean_bgr,
    cv2.COLOR_BGR2RGB
)


# ============================================================
# SAFETY RESTORE
#
# Re-copy independent shower prop pixels exactly from INPUT
# so the inpainting operation can never modify them.
# ============================================================

clean[
    restore_mask
] = image[
    restore_mask
]


final_pil = Image.fromarray(
    clean
)


final_pil.save(
    OUTPUT
)


# ============================================================
# VERIFICATION PREVIEW
# ============================================================

preview = final_pil.copy()


preview = draw_contour(
    preview,
    expanded,
    "red",
    2
)


preview = draw_contour(
    preview,
    restore_mask,
    "lime",
    2
)


preview.save(
    PREVIEW_PATH
)


# ============================================================
# STATE
# ============================================================

state = {

    "stage":
        "05F",

    "purpose":
        (
            "remove residual glass-mounted hardware artifact "
            "from final Stage06 detection master"
        ),

    "source":
        str(INPUT),

    "output":
        str(OUTPUT),

    "method":
        "OpenCV Telea inpainting",

    "dilate_kernel":
        DILATE_KERNEL,

    "dilate_iterations":
        DILATE_ITERATIONS,

    "inpaint_radius":
        INPAINT_RADIUS,

    "original_glass_owned_pixels":
        int(
            glass_mask.sum()
        ),

    "expanded_inpaint_pixels":
        int(
            expanded.sum()
        ),

    "protected_restore_pixels":
        int(
            restore_mask.sum()
        ),

    "protected_overlap_after_exclusion":
        overlap_after_protection,

    "production_contract": {

        "stage06_detection_input":
            str(OUTPUT),

        "final_rgb_source":
            str(
                PROD
                / "stage01_master"
                / "00_master_input.png"
            ),

        "mirror_owner":
            "STAGE03",

        "glass_owner":
            "STAGE04",
    },

    "status":
        "REQUIRES_FINAL_VISUAL_AUDIT",
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
print("PRODUCTION STAGE 05F RESULT")
print("=" * 110)

print()
print(
    "ORIGINAL GLASS MASK PIXELS:",
    int(
        glass_mask.sum()
    )
)

print(
    "EXPANDED INPAINT PIXELS:",
    int(
        expanded.sum()
    )
)

print(
    "SHOWER-PROP PROTECTED PIXELS:",
    int(
        restore_mask.sum()
    )
)

print(
    "OVERLAP AFTER PROTECTION:",
    overlap_after_protection
)

print()
print(
    "FINAL:",
    OUTPUT
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
print("INLINE STAGE05F FINAL AUDIT")
print("=" * 110)


show(
    input_pil,
    (
        "BEFORE — Stage05E "
        "(White Glass-Hardware Artifact Visible)"
    )
)


show(
    final_pil,
    (
        "AFTER — Stage05F "
        "FINAL CANONICAL PROP-DETECTION MASTER"
    )
)


show(
    preview,
    (
        "Stage05F Safety Audit "
        "RED=Inpainted Region | "
        "GREEN=Protected Independent Shower Prop"
    )
)
