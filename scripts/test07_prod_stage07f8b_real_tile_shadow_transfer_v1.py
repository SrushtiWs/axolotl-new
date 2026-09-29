
from pathlib import Path
import json

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


TILE_BASE_PATH = (
    PROD
    / "stage08_tile_application"
    / "08h1_true_metric_tile_projection"
    / "02_lit_glossy_true_metric_projection.png"
)


FLOOR_MASK_PATH = (
    PROD
    / "stage08_tile_application"
    / "08b2_floor_plus_screeding_target"
    / "01_main_floor_mask.png"
)


MULTIPLIER_PATH = (
    PROD
    / "stage07_empty_room"
    / "07f7_fsd_soft_multiplier_candidates"
    / "02_multiplier_B_balanced_16bit.png"
)


PROP_LAYER_PATH = (
    PROD
    / "stage08_tile_application"
    / "08h2_props_over_metric_floor"
    / "01_exact_stage01_rgb_prop_layer.png"
)


OLD_H2_RESULT_PATH = (
    PROD
    / "stage08_tile_application"
    / "08h2_props_over_metric_floor"
    / "02_metric_floor_plus_physical_props.png"
)


OUT = (
    PROD
    / "stage08_tile_application"
    / "08f8b_real_tile_shadow_transfer"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# VALIDATE
# ============================================================

for p in [
    TILE_BASE_PATH,
    FLOOR_MASK_PATH,
    MULTIPLIER_PATH,
    PROP_LAYER_PATH,
]:

    if not p.exists():

        raise FileNotFoundError(
            p
        )


# ============================================================
# LOAD
# ============================================================

tile_base = np.asarray(
    Image.open(
        TILE_BASE_PATH
    ).convert("RGB")
).astype(
    np.float32
)


floor = (
    np.asarray(
        Image.open(
            FLOOR_MASK_PATH
        ).convert("L")
    ) > 127
)


mult16 = np.asarray(
    Image.open(
        MULTIPLIER_PATH
    )
).astype(
    np.float32
)


prop_rgba = np.asarray(
    Image.open(
        PROP_LAYER_PATH
    ).convert("RGBA")
).astype(
    np.float32
)


H, W = floor.shape


# ============================================================
# SIZE CHECK
# ============================================================

for name, arr in [

    (
        "tile_base",
        tile_base
    ),

    (
        "multiplier",
        mult16
    ),

    (
        "prop_layer",
        prop_rgba
    ),

]:

    if arr.shape[:2] != (
        H,
        W
    ):

        raise RuntimeError(
            f"{name} shape mismatch: {arr.shape}"
        )


print("=" * 110)
print("07F8B — REAL METRIC TILE SHADOW TRANSFER")
print("=" * 110)

print()

print(
    "IMAGE SIZE:",
    W,
    "x",
    H
)

print(
    "TILE BASE:",
    TILE_BASE_PATH
)

print(
    "MULTIPLIER:",
    MULTIPLIER_PATH
)

print(
    "PROP LAYER:",
    PROP_LAYER_PATH
)


# ============================================================
# DECODE 16-BIT MULTIPLIER
# ============================================================

multiplier = (
    mult16
    /
    65535.0
)


multiplier = np.clip(
    multiplier,
    0.0,
    1.0
)


# Outside true floor:
# no shadow operation.
multiplier[
    ~floor
] = 1.0


print()

print(
    "MULTIPLIER MIN:",
    round(
        float(
            multiplier.min()
        ),
        6
    )
)

print(
    "MULTIPLIER MAX:",
    round(
        float(
            multiplier.max()
        ),
        6
    )
)


active = (
    multiplier
    <
    0.999
)


print(
    "ACTIVE SHADOW PIXELS:",
    int(
        active.sum()
    )
)


# ============================================================
# APPLY PRECOMPUTED SHADOW
#
# THIS IS THE PRODUCTION-RUNTIME OPERATION:
#
# new_surface_rgb *= shadow_multiplier
# ============================================================

tile_shadowed = tile_base.copy()


tile_shadowed[
    floor
] *= multiplier[
    floor,
    None
]


tile_shadowed = np.clip(
    tile_shadowed,
    0,
    255
).astype(
    np.uint8
)


SHADOWED_TILE_PATH = (
    OUT
    / "01_metric_tile_with_precomputed_shadow.png"
)


Image.fromarray(
    tile_shadowed
).save(
    SHADOWED_TILE_PATH
)


# ============================================================
# PROP COMPOSITE
#
# Exact Stage01 RGB prop layer.
# No regenerated prop appearance.
# ============================================================

prop_rgb = (
    prop_rgba[
        ...,
        :3
    ]
)


prop_alpha = (
    prop_rgba[
        ...,
        3
    ]
    /
    255.0
)


final_with_props = (

    tile_shadowed.astype(
        np.float32
    )
    *
    (
        1.0
        -
        prop_alpha[
            ...,
            None
        ]
    )

    +

    prop_rgb
    *
    prop_alpha[
        ...,
        None
    ]

)


final_with_props = np.clip(
    final_with_props,
    0,
    255
).astype(
    np.uint8
)


FINAL_PATH = (
    OUT
    / "02_metric_tile_shadow_plus_exact_props.png"
)


Image.fromarray(
    final_with_props
).save(
    FINAL_PATH
)


# ============================================================
# BASE + PROPS WITHOUT NEW SHADOW
#
# Useful direct control.
# ============================================================

base_with_props = (

    tile_base
    *
    (
        1.0
        -
        prop_alpha[
            ...,
            None
        ]
    )

    +

    prop_rgb
    *
    prop_alpha[
        ...,
        None
    ]

)


base_with_props = np.clip(
    base_with_props,
    0,
    255
).astype(
    np.uint8
)


BASE_PROPS_PATH = (
    OUT
    / "03_metric_tile_plus_props_without_shadow.png"
)


Image.fromarray(
    base_with_props
).save(
    BASE_PROPS_PATH
)


# ============================================================
# OPTIONAL OLD H2 REFERENCE
# ============================================================

old_h2 = None


if OLD_H2_RESULT_PATH.exists():

    old_h2 = np.asarray(
        Image.open(
            OLD_H2_RESULT_PATH
        ).convert("RGB")
    )


# ============================================================
# SHADOW STRENGTH VIS
# ============================================================

strength = (
    1.0
    -
    multiplier
)


strength_vis = np.clip(
    strength
    /
    0.36,
    0.0,
    1.0
)


# ============================================================
# DIFFERENCE
#
# purely diagnostic:
# where did multiplier change tile RGB?
# ============================================================

difference = np.mean(
    np.abs(
        tile_base
        -
        tile_shadowed.astype(
            np.float32
        )
    ),
    axis=2
)


difference_vis = np.clip(
    difference
    /
    60.0,
    0.0,
    1.0
)


# ============================================================
# 8-PANEL AUDIT
# ============================================================

fig, axes = plt.subplots(
    2,
    4,
    figsize=(
        18,
        18
    )
)


axes[0, 0].imshow(
    tile_base.astype(
        np.uint8
    )
)

axes[0, 0].set_title(
    "1. Frozen 08H1\n"
    "Metric Glossy Tile"
)

axes[0, 0].axis(
    "off"
)


axes[0, 1].imshow(
    strength_vis,
    cmap="gray",
    vmin=0,
    vmax=1
)

axes[0, 1].set_title(
    "2. Precomputed\n"
    "Shadow Strength"
)

axes[0, 1].axis(
    "off"
)


axes[0, 2].imshow(
    tile_shadowed
)

axes[0, 2].set_title(
    "3. Metric Tile ×\n"
    "Shadow Multiplier"
)

axes[0, 2].axis(
    "off"
)


axes[0, 3].imshow(
    difference_vis,
    cmap="gray",
    vmin=0,
    vmax=1
)

axes[0, 3].set_title(
    "4. Tile Change\n"
    "Diagnostic"
)

axes[0, 3].axis(
    "off"
)


axes[1, 0].imshow(
    base_with_props
)

axes[1, 0].set_title(
    "5. Tile + Props\n"
    "WITHOUT Shadow"
)

axes[1, 0].axis(
    "off"
)


axes[1, 1].imshow(
    final_with_props
)

axes[1, 1].set_title(
    "6. Tile + Shadow +\n"
    "Exact Props"
)

axes[1, 1].axis(
    "off"
)


if old_h2 is not None:

    axes[1, 2].imshow(
        old_h2
    )

    axes[1, 2].set_title(
        "7. Frozen 08H2\n"
        "Old Result"
    )

else:

    axes[1, 2].imshow(
        np.zeros(
            (
                H,
                W,
                3
            ),
            dtype=np.uint8
        )
    )

    axes[1, 2].set_title(
        "7. 08H2\n"
        "Unavailable"
    )


axes[1, 2].axis(
    "off"
)


# Side-by-side difference crop/visual comparison
comparison = np.concatenate(
    [
        base_with_props,
        final_with_props
    ],
    axis=1
)


axes[1, 3].imshow(
    comparison
)

axes[1, 3].set_title(
    "8. BEFORE | AFTER\n"
    "Runtime Shadow"
)

axes[1, 3].axis(
    "off"
)


plt.tight_layout()


AUDIT_PATH = (
    OUT
    / "04_stage07f8b_real_tile_shadow_transfer_audit.png"
)


plt.savefig(
    AUDIT_PATH,
    dpi=150,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# STATS
# ============================================================

changed_floor = (
    difference
    >
    1.0
)


mean_change = (
    float(
        difference[
            changed_floor
        ].mean()
    )
    if changed_floor.any()
    else 0.0
)


max_change = float(
    difference.max()
)


if active.any():

    mean_active_multiplier = float(
        multiplier[
            active
        ].mean()
    )

    strongest_multiplier = float(
        multiplier[
            active
        ].min()
    )

else:

    mean_active_multiplier = 1.0
    strongest_multiplier = 1.0


print()
print("=" * 110)
print("07F8B RESULT")
print("=" * 110)

print()

print(
    "ACTIVE SHADOW PIXELS:",
    int(
        active.sum()
    )
)


print(
    "MEAN ACTIVE MULTIPLIER:",
    round(
        mean_active_multiplier,
        4
    )
)


print(
    "STRONGEST MULTIPLIER:",
    round(
        strongest_multiplier,
        4
    )
)


print()

print(
    "CHANGED FLOOR PIXELS (>1 RGB):",
    int(
        changed_floor.sum()
    )
)


print(
    "MEAN RGB CHANGE ON CHANGED PIXELS:",
    round(
        mean_change,
        3
    )
)


print(
    "MAX MEAN-RGB CHANGE:",
    round(
        max_change,
        3
    )
)


# ============================================================
# STATE
# ============================================================

STATE = {

    "stage":
        "07F8B",

    "purpose":
        "REAL_TRUE_METRIC_TILE_RUNTIME_SHADOW_VALIDATION",

    "architecture": [

        "frozen 08H1 true-metric glossy tile RGB",

        "multiply by precomputed 07F7-B shadow map",

        "composite exact frozen Stage01 prop layer"
    ],

    "ai_run":
        False,

    "shadow_detection_run":
        False,

    "tile_projection_rerun":
        False,

    "runtime_operation":
        "surface_rgb_times_precomputed_shadow_multiplier",

    "sources": {

        "metric_tile":
            str(
                TILE_BASE_PATH
            ),

        "shadow_multiplier":
            str(
                MULTIPLIER_PATH
            ),

        "prop_layer":
            str(
                PROP_LAYER_PATH
            ),

        "floor_mask":
            str(
                FLOOR_MASK_PATH
            ),
    },

    "statistics": {

        "active_shadow_pixels":
            int(
                active.sum()
            ),

        "mean_active_multiplier":
            mean_active_multiplier,

        "strongest_multiplier":
            strongest_multiplier,

        "changed_floor_pixels":
            int(
                changed_floor.sum()
            ),

        "mean_rgb_change":
            mean_change,

        "max_mean_rgb_change":
            max_change,
    },

    "outputs": {

        "shadowed_tile":
            str(
                SHADOWED_TILE_PATH
            ),

        "final_with_props":
            str(
                FINAL_PATH
            ),

        "control_without_shadow":
            str(
                BASE_PROPS_PATH
            ),

        "audit":
            str(
                AUDIT_PATH
            ),
    },

    "status":
        "RND_REQUIRES_REAL_TILE_VISUAL_AUDIT"
}


STATE_PATH = (
    OUT
    / "00_stage07f8b_result.json"
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

print(
    "SHADOWED TILE:",
    SHADOWED_TILE_PATH
)

print(
    "FINAL WITH PROPS:",
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
    "NO AI WAS RUN."
)

print(
    "NO SHADOW DETECTOR WAS RUN."
)

print(
    "THIS IS THE ACTUAL PRODUCTION-RUNTIME OPERATION."
)
