
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


# ------------------------------------------------------------
# Canonical Stage06 extraction/reference image
# ------------------------------------------------------------

STAGE05F = (
    PROD
    / "stage05_clean_room_with_props"
    / "12_final_canonical_prop_detection_master.png"
)


# ------------------------------------------------------------
# Original master
# Secondary RGB audit only at this stage.
# ------------------------------------------------------------

STAGE01 = (
    PROD
    / "stage01_master"
    / "00_master_input.png"
)


# ------------------------------------------------------------
# P01 — completed vanity from 06F3
# ------------------------------------------------------------

P01_MASK = (
    PROD
    / "stage06_prop_layer"
    / "06f3_vanity_internal_completion"
    / "01_p01_completed_vanity_mask.png"
)


# ------------------------------------------------------------
# P02-P06 — visually selected frozen 06D2C1 masks
# ------------------------------------------------------------

D2C1_OBJECTS = (
    PROD
    / "stage06_prop_layer"
    / "06d2c1_complete_main_prop_sam2_multimask_audit"
    / "objects"
)


MASK_PATHS = {

    "P01": P01_MASK,

    "P02": (
        D2C1_OBJECTS
        / "P02_candidate_3_mask.png"
    ),

    "P03": (
        D2C1_OBJECTS
        / "P03_candidate_2_mask.png"
    ),

    "P04": (
        D2C1_OBJECTS
        / "P04_candidate_1_mask.png"
    ),

    "P05": (
        D2C1_OBJECTS
        / "P05_candidate_2_mask.png"
    ),

    "P06": (
        D2C1_OBJECTS
        / "P06_candidate_3_mask.png"
    ),
}


PROP_NAMES = {

    "P01":
        "complete vanity system",

    "P02":
        "complete toilet system",

    "P03":
        "shower fixture",

    "P04":
        "wall electrical plate",

    "P05":
        "toilet paper holder",

    "P06":
        "ceiling light",
}


OUT = (
    PROD
    / "stage06_prop_layer"
    / "06f4_final_six_prop_layer"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# VALIDATE
# ============================================================

for p in [
    STAGE05F,
    STAGE01,
    *MASK_PATHS.values(),
]:

    if not p.exists():
        raise FileNotFoundError(p)


# ============================================================
# LOAD CANONICAL IMAGES
# ============================================================

stage05f = Image.open(
    STAGE05F
).convert("RGB")


stage01 = Image.open(
    STAGE01
).convert("RGB")


if stage05f.size != stage01.size:

    raise RuntimeError(
        (
            "Stage05F / Stage01 size mismatch: "
            f"{stage05f.size} vs {stage01.size}"
        )
    )


W, H = stage05f.size


print("=" * 110)
print("STAGE 06F4 — FINAL SIX-MAIN-PROP LAYER")
print("=" * 110)

print()
print(
    "CANVAS:",
    f"{W} × {H}"
)


# ============================================================
# LOAD SIX MASKS
# ============================================================

prop_masks = {}

for pid, path in MASK_PATHS.items():

    img = Image.open(
        path
    ).convert("L")


    if img.size != (
        W,
        H
    ):

        raise RuntimeError(
            (
                f"{pid} mask size mismatch: "
                f"{img.size} != {(W, H)}"
            )
        )


    mask = (
        np.asarray(
            img
        )
        >
        127
    )


    prop_masks[
        pid
    ] = mask


    print(
        pid,
        "|",
        PROP_NAMES[pid],
        "| pixels =",
        int(
            mask.sum()
        ),
        "|",
        path
    )


# ============================================================
# UNION SIX MAIN PROPS
# ============================================================

combined_mask = np.zeros(
    (
        H,
        W
    ),
    dtype=bool
)


for pid in [
    "P01",
    "P02",
    "P03",
    "P04",
    "P05",
    "P06",
]:

    combined_mask |= (
        prop_masks[
            pid
        ]
    )


combined_pixels = int(
    combined_mask.sum()
)


# ============================================================
# OVERLAP DIAGNOSTIC
#
# We do not expect meaningful overlap between unrelated props.
# ============================================================

stack = np.stack(
    [
        prop_masks[
            pid
        ].astype(
            np.uint8
        )
        for pid in [
            "P01",
            "P02",
            "P03",
            "P04",
            "P05",
            "P06",
        ]
    ],
    axis=0
)


ownership_count = stack.sum(
    axis=0
)


multi_owned = (
    ownership_count
    >
    1
)


multi_owned_pixels = int(
    multi_owned.sum()
)


# ============================================================
# SAVE COMBINED MASK
# ============================================================

COMBINED_MASK_PATH = (
    OUT
    / "01_final_six_prop_union_mask.png"
)


Image.fromarray(
    combined_mask.astype(
        np.uint8
    )
    *
    255
).save(
    COMBINED_MASK_PATH
)


# ============================================================
# PRIMARY TRANSPARENT PNG — STAGE05F RGB
# ============================================================

stage05f_np = np.asarray(
    stage05f
)


rgba05 = np.zeros(
    (
        H,
        W,
        4
    ),
    dtype=np.uint8
)


rgba05[
    ...,
    :3
] = stage05f_np


rgba05[
    ...,
    3
] = (
    combined_mask.astype(
        np.uint8
    )
    *
    255
)


RGBA05_PATH = (
    OUT
    / "02_final_props_stage05f_rgb_transparent.png"
)


Image.fromarray(
    rgba05,
    mode="RGBA"
).save(
    RGBA05_PATH
)


# ============================================================
# SECONDARY AUDIT — SAME MASK WITH STAGE01 RGB
# ============================================================

stage01_np = np.asarray(
    stage01
)


rgba01 = np.zeros(
    (
        H,
        W,
        4
    ),
    dtype=np.uint8
)


rgba01[
    ...,
    :3
] = stage01_np


rgba01[
    ...,
    3
] = (
    combined_mask.astype(
        np.uint8
    )
    *
    255
)


RGBA01_PATH = (
    OUT
    / "03_final_props_stage01_rgb_audit.png"
)


Image.fromarray(
    rgba01,
    mode="RGBA"
).save(
    RGBA01_PATH
)


# ============================================================
# CHECKERBOARD FUNCTION
# ============================================================

def checkerboard(
    width,
    height,
    tile=16
):

    yy, xx = np.indices(
        (
            height,
            width
        )
    )

    pattern = (
        (
            xx // tile
            +
            yy // tile
        )
        %
        2
    )


    gray = np.where(
        pattern == 0,
        220,
        180
    ).astype(
        np.uint8
    )


    return np.stack(
        [
            gray,
            gray,
            gray
        ],
        axis=2
    )


checker = checkerboard(
    W,
    H
)


alpha = (
    combined_mask[
        ...,
        None
    ].astype(
        np.float32
    )
)


preview05 = (

    stage05f_np.astype(
        np.float32
    )
    *
    alpha

    +

    checker.astype(
        np.float32
    )
    *
    (
        1.0
        -
        alpha
    )

).astype(
    np.uint8
)


preview01 = (

    stage01_np.astype(
        np.float32
    )
    *
    alpha

    +

    checker.astype(
        np.float32
    )
    *
    (
        1.0
        -
        alpha
    )

).astype(
    np.uint8
)


PREVIEW05_PATH = (
    OUT
    / "04_stage05f_rgb_checkerboard.png"
)


PREVIEW01_PATH = (
    OUT
    / "05_stage01_rgb_checkerboard_audit.png"
)


Image.fromarray(
    preview05
).save(
    PREVIEW05_PATH
)


Image.fromarray(
    preview01
).save(
    PREVIEW01_PATH
)


# ============================================================
# PER-PROP MASK OVERVIEW
# ============================================================

fig, axes = plt.subplots(
    2,
    3,
    figsize=(
        12,
        10
    )
)


axes = axes.flatten()


for ax, pid in zip(
    axes,
    [
        "P01",
        "P02",
        "P03",
        "P04",
        "P05",
        "P06",
    ]
):

    ax.imshow(
        stage05f
    )


    overlay = np.zeros(
        (
            H,
            W,
            4
        ),
        dtype=np.float32
    )


    overlay[
        prop_masks[
            pid
        ],
        1
    ] = 1.0


    overlay[
        prop_masks[
            pid
        ],
        3
    ] = 0.55


    ax.imshow(
        overlay
    )


    ax.set_title(
        f"{pid} — {PROP_NAMES[pid]}"
    )

    ax.axis(
        "off"
    )


plt.tight_layout()


PER_PROP_PATH = (
    OUT
    / "06_six_prop_mask_overview.png"
)


plt.savefig(
    PER_PROP_PATH,
    dpi=150,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# FINAL 4-PANEL AUDIT
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
# Input
# ------------------------------------------------------------

axes[0].imshow(
    stage05f
)

axes[0].set_title(
    "1. Stage05F Input"
)

axes[0].axis(
    "off"
)


# ------------------------------------------------------------
# Combined mask
# ------------------------------------------------------------

axes[1].imshow(
    combined_mask,
    cmap="gray",
    vmin=0,
    vmax=1
)

axes[1].set_title(
    "2. Final Six-Prop Union Mask"
)

axes[1].axis(
    "off"
)


# ------------------------------------------------------------
# Stage05F RGB extraction
# ------------------------------------------------------------

axes[2].imshow(
    preview05
)

axes[2].set_title(
    "3. Stage05F RGB + Final Alpha"
)

axes[2].axis(
    "off"
)


# ------------------------------------------------------------
# Stage01 RGB audit
# ------------------------------------------------------------

axes[3].imshow(
    preview01
)

axes[3].set_title(
    "4. Stage01 RGB + Same Alpha"
)

axes[3].axis(
    "off"
)


plt.tight_layout()


FINAL_AUDIT_PATH = (
    OUT
    / "07_stage06f4_final_audit.png"
)


plt.savefig(
    FINAL_AUDIT_PATH,
    dpi=150,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# FINAL STATE
# ============================================================

STATE = {

    "stage":
        "06F4",

    "architecture":
        "FINAL_SIX_MAIN_PROP_UNION",

    "image_size": [
        W,
        H
    ],

    "selected_masks": {

        "P01": {
            "name":
                PROP_NAMES["P01"],

            "source":
                "06F3_COMPLETED_VANITY",

            "mask":
                str(
                    MASK_PATHS[
                        "P01"
                    ]
                )
        },

        "P02": {
            "name":
                PROP_NAMES["P02"],

            "source":
                "06D2C1_CANDIDATE_3",

            "mask":
                str(
                    MASK_PATHS[
                        "P02"
                    ]
                )
        },

        "P03": {
            "name":
                PROP_NAMES["P03"],

            "source":
                "06D2C1_CANDIDATE_2",

            "mask":
                str(
                    MASK_PATHS[
                        "P03"
                    ]
                )
        },

        "P04": {
            "name":
                PROP_NAMES["P04"],

            "source":
                "06D2C1_CANDIDATE_1",

            "mask":
                str(
                    MASK_PATHS[
                        "P04"
                    ]
                )
        },

        "P05": {
            "name":
                PROP_NAMES["P05"],

            "source":
                "06D2C1_CANDIDATE_2",

            "mask":
                str(
                    MASK_PATHS[
                        "P05"
                    ]
                )
        },

        "P06": {
            "name":
                PROP_NAMES["P06"],

            "source":
                "06D2C1_CANDIDATE_3",

            "mask":
                str(
                    MASK_PATHS[
                        "P06"
                    ]
                )
        },
    },

    "per_prop_pixels": {

        pid:
            int(
                prop_masks[
                    pid
                ].sum()
            )

        for pid in prop_masks
    },

    "combined_pixels":
        combined_pixels,

    "multi_owned_pixels":
        multi_owned_pixels,

    "outputs": {

        "union_mask":
            str(
                COMBINED_MASK_PATH
            ),

        "stage05f_rgb_transparent":
            str(
                RGBA05_PATH
            ),

        "stage01_rgb_audit":
            str(
                RGBA01_PATH
            ),

        "stage05f_checkerboard":
            str(
                PREVIEW05_PATH
            ),

        "stage01_checkerboard":
            str(
                PREVIEW01_PATH
            ),

        "per_prop_overview":
            str(
                PER_PROP_PATH
            ),

        "final_audit":
            str(
                FINAL_AUDIT_PATH
            ),
    },

    "status":
        "REQUIRES_FINAL_STAGE06_VISUAL_AUDIT",

    "next_if_pass":
        "FREEZE_STAGE06_PROP_LAYER_AND_CONTINUE_STAGE07_EMPTY_ROOM"
}


STATE_PATH = (
    OUT
    / "00_stage06f4_result.json"
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
print("STAGE 06F4 RESULT")
print("=" * 110)

print()
print(
    "CANVAS:",
    f"{W} × {H}"
)

print(
    "COMBINED PROP PIXELS:",
    combined_pixels
)

print(
    "MULTI-OWNED PIXELS:",
    multi_owned_pixels
)


print()
print(
    "SELECTED MASKS:"
)


for pid in [
    "P01",
    "P02",
    "P03",
    "P04",
    "P05",
    "P06",
]:

    print(
        pid,
        "|",
        PROP_NAMES[
            pid
        ],
        "| pixels =",
        int(
            prop_masks[
                pid
            ].sum()
        )
    )


print()
print(
    "UNION MASK:",
    COMBINED_MASK_PATH
)

print(
    "STAGE05F TRANSPARENT:",
    RGBA05_PATH
)

print(
    "STAGE01 RGB AUDIT:",
    RGBA01_PATH
)

print(
    "FINAL AUDIT:",
    FINAL_AUDIT_PATH
)

print(
    "STATE:",
    STATE_PATH
)

print()
print(
    "NO AI MODEL WAS RUN IN 06F4."
)
