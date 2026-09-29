
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
# TRUE METRIC TILED EMPTY ROOM
# ------------------------------------------------------------

TILED_ROOM_PATH = (
    PROD
    / "stage08_tile_application"
    / "08h1_true_metric_tile_projection"
    / "02_lit_glossy_true_metric_projection.png"
)


# ------------------------------------------------------------
# STAGE01 = FINAL RGB SOURCE OF TRUTH
# ------------------------------------------------------------

STAGE01_PATH = (
    PROD
    / "stage01_master"
    / "00_master_input.png"
)


# ------------------------------------------------------------
# FINAL STAGE06 PHYSICAL PROP ALPHA
# ------------------------------------------------------------

PROP_MASK_PATH = (
    PROD
    / "stage06_prop_layer"
    / "06f4_final_six_prop_layer"
    / "01_final_six_prop_union_mask.png"
)


OUT = (
    PROD
    / "stage08_tile_application"
    / "08h2_props_over_metric_floor"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# VALIDATE
# ============================================================

for p in [
    TILED_ROOM_PATH,
    STAGE01_PATH,
    PROP_MASK_PATH,
]:

    if not p.exists():
        raise FileNotFoundError(p)


# ============================================================
# LOAD
# ============================================================

tiled = Image.open(
    TILED_ROOM_PATH
).convert("RGB")


stage01 = Image.open(
    STAGE01_PATH
).convert("RGB")


prop_mask_img = Image.open(
    PROP_MASK_PATH
).convert("L")


if not (
    tiled.size
    ==
    stage01.size
    ==
    prop_mask_img.size
):

    raise RuntimeError(
        (
            "Size mismatch: "
            f"tiled={tiled.size}, "
            f"stage01={stage01.size}, "
            f"mask={prop_mask_img.size}"
        )
    )


W, H = tiled.size


tiled_np = np.asarray(
    tiled
).astype(
    np.uint8
)


stage01_np = np.asarray(
    stage01
).astype(
    np.uint8
)


prop_mask = (
    np.asarray(
        prop_mask_img
    )
    >
    127
)


print("=" * 110)
print("STAGE 08H2 — PROPS OVER TRUE-METRIC FLOOR")
print("=" * 110)

print()
print(
    "CANVAS:",
    f"{W} × {H}"
)

print(
    "PROP PIXELS:",
    int(
        prop_mask.sum()
    )
)


# ============================================================
# EXACT COMPOSITE
#
# Wherever Stage06 alpha is active:
# use exact RGB from Stage01.
# ============================================================

composite = tiled_np.copy()


composite[
    prop_mask
] = stage01_np[
    prop_mask
]


# ============================================================
# TRANSPARENT PROP LAYER
#
# Save again explicitly from Stage01 RGB + Stage06 alpha.
# ============================================================

rgba = np.zeros(
    (
        H,
        W,
        4
    ),
    dtype=np.uint8
)


rgba[
    ...,
    :3
] = stage01_np


rgba[
    ...,
    3
] = (
    prop_mask.astype(
        np.uint8
    )
    *
    255
)


PROP_RGBA_PATH = (
    OUT
    / "01_exact_stage01_rgb_prop_layer.png"
)


Image.fromarray(
    rgba,
    mode="RGBA"
).save(
    PROP_RGBA_PATH
)


# ============================================================
# FINAL COMPOSITE
# ============================================================

COMPOSITE_PATH = (
    OUT
    / "02_metric_floor_plus_physical_props.png"
)


Image.fromarray(
    composite
).save(
    COMPOSITE_PATH
)


# ============================================================
# PROP MASK OVERLAY
# ============================================================

mask_preview = tiled_np.astype(
    np.float32
).copy()


mask_preview[
    prop_mask
] = (

    mask_preview[
        prop_mask
    ]
    *
    0.45

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
    0.55
)


mask_preview = np.clip(
    mask_preview,
    0,
    255
).astype(
    np.uint8
)


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


axes[0].imshow(
    tiled
)

axes[0].set_title(
    "1. 08H1\nTrue-Metric Tiled Room"
)

axes[0].axis(
    "off"
)


axes[1].imshow(
    prop_mask,
    cmap="gray"
)

axes[1].set_title(
    "2. Frozen Stage06\nPhysical Prop Alpha"
)

axes[1].axis(
    "off"
)


axes[2].imshow(
    mask_preview
)

axes[2].set_title(
    "3. Placement Audit\nGREEN=Prop Pixels"
)

axes[2].axis(
    "off"
)


axes[3].imshow(
    composite
)

axes[3].set_title(
    "4. Metric Floor + Props\nNO NEW SHADOWS YET"
)

axes[3].axis(
    "off"
)


plt.tight_layout()


AUDIT_PATH = (
    OUT
    / "03_stage08h2_props_composite_audit.png"
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
        "08H2",

    "purpose":
        "COMPOSITE_PHYSICAL_PROPS_OVER_TRUE_METRIC_TILE",

    "background":
        str(
            TILED_ROOM_PATH
        ),

    "prop_mask":
        str(
            PROP_MASK_PATH
        ),

    "prop_rgb_source":
        str(
            STAGE01_PATH
        ),

    "rules": {

        "prop_rgb":
            "EXACT_STAGE01_RGB",

        "prop_positions":
            "UNCHANGED",

        "new_shadow_generation":
            False,

        "mirror":
            False,

        "glass":
            False
    },

    "statistics": {

        "prop_pixels":
            int(
                prop_mask.sum()
            )
    },

    "outputs": {

        "prop_rgba":
            str(
                PROP_RGBA_PATH
            ),

        "composite":
            str(
                COMPOSITE_PATH
            ),

        "audit":
            str(
                AUDIT_PATH
            )
    },

    "status":
        "REQUIRES_PROP_ALIGNMENT_VISUAL_AUDIT",

    "next_if_pass":
        "GENERATE_CONTACT_AND_CAST_SHADOWS_ON_FINAL_TILED_SURFACES"
}


STATE_PATH = (
    OUT
    / "00_stage08h2_result.json"
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
print("STAGE 08H2 RESULT")
print("=" * 110)

print()

print(
    "PROP PIXELS:",
    int(
        prop_mask.sum()
    )
)

print()

print(
    "PROP RGBA:",
    PROP_RGBA_PATH
)

print(
    "COMPOSITE:",
    COMPOSITE_PATH
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
    "NO AI MODEL WAS RUN."
)

print(
    "NO NEW SHADOWS WERE GENERATED."
)

print(
    "MIRROR AND GLASS REMAIN EXCLUDED."
)
