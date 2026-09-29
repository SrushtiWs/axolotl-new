
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


TILE_PATH = (
    PROD
    / "stage08_tile_application"
    / "08h1_true_metric_tile_projection"
    / "02_lit_glossy_true_metric_projection.png"
)


FLOOR_PATH = (
    PROD
    / "stage08_tile_application"
    / "08b2_floor_plus_screeding_target"
    / "01_main_floor_mask.png"
)


PROP_PATH = (
    PROD
    / "stage08_tile_application"
    / "08h2_props_over_metric_floor"
    / "01_exact_stage01_rgb_prop_layer.png"
)


MULT_DIR = (
    PROD
    / "stage07_empty_room"
    / "07f7_fsd_soft_multiplier_candidates"
)


MULT_A_PATH = (
    MULT_DIR
    / "01_multiplier_A_conservative_16bit.png"
)


MULT_B_PATH = (
    MULT_DIR
    / "02_multiplier_B_balanced_16bit.png"
)


MULT_C_PATH = (
    MULT_DIR
    / "03_multiplier_C_contact_weighted_16bit.png"
)


OUT = (
    PROD
    / "stage08_tile_application"
    / "08f9_real_tile_shadow_candidate_comparison"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# VALIDATE
# ============================================================

for p in [
    TILE_PATH,
    FLOOR_PATH,
    PROP_PATH,
    MULT_A_PATH,
    MULT_B_PATH,
    MULT_C_PATH,
]:

    if not p.exists():
        raise FileNotFoundError(p)


# ============================================================
# LOAD
# ============================================================

tile = np.asarray(
    Image.open(
        TILE_PATH
    ).convert("RGB")
).astype(
    np.float32
)


floor = (
    np.asarray(
        Image.open(
            FLOOR_PATH
        ).convert("L")
    ) > 127
)


prop = np.asarray(
    Image.open(
        PROP_PATH
    ).convert("RGBA")
).astype(
    np.float32
)


H, W = floor.shape


# ============================================================
# MULTIPLIER LOADER
# ============================================================

def load_multiplier(path):

    arr = np.asarray(
        Image.open(
            path
        )
    ).astype(
        np.float32
    )

    if arr.shape != (
        H,
        W
    ):
        raise RuntimeError(
            f"Multiplier shape mismatch: {path} {arr.shape}"
        )

    mult = np.clip(
        arr / 65535.0,
        0.0,
        1.0
    )

    mult[
        ~floor
    ] = 1.0

    return mult


mult_A = load_multiplier(
    MULT_A_PATH
)

mult_B = load_multiplier(
    MULT_B_PATH
)

mult_C = load_multiplier(
    MULT_C_PATH
)


# ============================================================
# PROP COMPOSITING
# ============================================================

prop_rgb = prop[
    ...,
    :3
]

prop_alpha = (
    prop[
        ...,
        3
    ]
    /
    255.0
)


def composite_props(background):

    out = (
        background.astype(
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

    return np.clip(
        out,
        0,
        255
    ).astype(
        np.uint8
    )


# ============================================================
# APPLY RUNTIME SHADOW
# ============================================================

def apply_shadow(multiplier):

    result = tile.copy()

    result[
        floor
    ] *= multiplier[
        floor,
        None
    ]

    return np.clip(
        result,
        0,
        255
    ).astype(
        np.uint8
    )


shadow_A = apply_shadow(
    mult_A
)

shadow_B = apply_shadow(
    mult_B
)

shadow_C = apply_shadow(
    mult_C
)


# ============================================================
# COMPOSITE EXACT PROPS
# ============================================================

control = composite_props(
    tile
)

final_A = composite_props(
    shadow_A
)

final_B = composite_props(
    shadow_B
)

final_C = composite_props(
    shadow_C
)


# ============================================================
# SAVE ALL RESULTS
# ============================================================

CONTROL_PATH = (
    OUT
    / "01_control_no_shadow.png"
)

A_PATH = (
    OUT
    / "02_candidate_A_conservative.png"
)

B_PATH = (
    OUT
    / "03_candidate_B_balanced.png"
)

C_PATH = (
    OUT
    / "04_candidate_C_contact_weighted.png"
)


Image.fromarray(
    control
).save(
    CONTROL_PATH
)


Image.fromarray(
    final_A
).save(
    A_PATH
)


Image.fromarray(
    final_B
).save(
    B_PATH
)


Image.fromarray(
    final_C
).save(
    C_PATH
)


# ============================================================
# STRENGTH MAPS
# ============================================================

strength_A = (
    1.0
    -
    mult_A
)

strength_B = (
    1.0
    -
    mult_B
)

strength_C = (
    1.0
    -
    mult_C
)


# ============================================================
# METRICS
# ============================================================

def get_metrics(
    multiplier
):

    active = (
        multiplier
        <
        0.999
    )

    if not active.any():

        return {
            "active_pixels": 0,
            "mean_multiplier": 1.0,
            "strongest_multiplier": 1.0,
        }

    return {

        "active_pixels":
            int(
                active.sum()
            ),

        "mean_multiplier":
            float(
                multiplier[
                    active
                ].mean()
            ),

        "strongest_multiplier":
            float(
                multiplier[
                    active
                ].min()
            ),
    }


stats_A = get_metrics(
    mult_A
)

stats_B = get_metrics(
    mult_B
)

stats_C = get_metrics(
    mult_C
)


# ============================================================
# DIFFERENCE FROM CONTROL
# ============================================================

def rgb_difference(
    candidate
):

    return np.mean(
        np.abs(
            candidate.astype(
                np.float32
            )
            -
            control.astype(
                np.float32
            )
        ),
        axis=2
    )


diff_A = rgb_difference(
    final_A
)

diff_B = rgb_difference(
    final_B
)

diff_C = rgb_difference(
    final_C
)


# ============================================================
# CROP AROUND SHADOW REGION
#
# Automatically derive union bbox from all active multiplier
# pixels. No hardcoded room coordinates.
# ============================================================

union_active = (
    (mult_A < 0.999)
    |
    (mult_B < 0.999)
    |
    (mult_C < 0.999)
)


ys, xs = np.where(
    union_active
)


if len(xs) > 0:

    margin = 35

    x0 = max(
        int(xs.min()) - margin,
        0
    )

    x1 = min(
        int(xs.max()) + margin + 1,
        W
    )

    y0 = max(
        int(ys.min()) - margin,
        0
    )

    y1 = min(
        int(ys.max()) + margin + 1,
        H
    )

else:

    x0 = 0
    y0 = 0
    x1 = W
    y1 = H


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


# FULL FRAME
axes[0, 0].imshow(
    control
)

axes[0, 0].set_title(
    "1. CONTROL\nNo Runtime Shadow"
)

axes[0, 0].axis(
    "off"
)


axes[0, 1].imshow(
    final_A
)

axes[0, 1].set_title(
    "2. A — Conservative"
)

axes[0, 1].axis(
    "off"
)


axes[0, 2].imshow(
    final_B
)

axes[0, 2].set_title(
    "3. B — Balanced"
)

axes[0, 2].axis(
    "off"
)


axes[0, 3].imshow(
    final_C
)

axes[0, 3].set_title(
    "4. C — Contact-Weighted"
)

axes[0, 3].axis(
    "off"
)


# SHADOW REGION CROPS
axes[1, 0].imshow(
    control[
        y0:y1,
        x0:x1
    ]
)

axes[1, 0].set_title(
    "5. CONTROL Crop"
)

axes[1, 0].axis(
    "off"
)


axes[1, 1].imshow(
    final_A[
        y0:y1,
        x0:x1
    ]
)

axes[1, 1].set_title(
    "6. A Crop"
)

axes[1, 1].axis(
    "off"
)


axes[1, 2].imshow(
    final_B[
        y0:y1,
        x0:x1
    ]
)

axes[1, 2].set_title(
    "7. B Crop"
)

axes[1, 2].axis(
    "off"
)


axes[1, 3].imshow(
    final_C[
        y0:y1,
        x0:x1
    ]
)

axes[1, 3].set_title(
    "8. C Crop"
)

axes[1, 3].axis(
    "off"
)


plt.tight_layout()


AUDIT_PATH = (
    OUT
    / "05_stage07f9_real_tile_candidate_comparison.png"
)


plt.savefig(
    AUDIT_PATH,
    dpi=180,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# SECOND AUDIT — STRENGTH ONLY
# ============================================================

fig2, axes2 = plt.subplots(
    1,
    3,
    figsize=(
        15,
        6
    )
)


for ax, strength, title in [

    (
        axes2[0],
        strength_A,
        "A Conservative"
    ),

    (
        axes2[1],
        strength_B,
        "B Balanced"
    ),

    (
        axes2[2],
        strength_C,
        "C Contact-Weighted"
    ),

]:

    ax.imshow(
        strength,
        cmap="gray",
        vmin=0,
        vmax=0.45
    )

    ax.set_title(
        title
    )

    ax.axis(
        "off"
    )


plt.tight_layout()


STRENGTH_AUDIT_PATH = (
    OUT
    / "06_stage07f9_strength_comparison.png"
)


plt.savefig(
    STRENGTH_AUDIT_PATH,
    dpi=180,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# PRINT
# ============================================================

print()
print("=" * 110)
print("07F9 RESULT")
print("=" * 110)


for name, stats, diff in [

    (
        "A CONSERVATIVE",
        stats_A,
        diff_A
    ),

    (
        "B BALANCED",
        stats_B,
        diff_B
    ),

    (
        "C CONTACT-WEIGHTED",
        stats_C,
        diff_C
    ),

]:

    changed = (
        diff
        >
        1.0
    )

    print()
    print(
        name
    )

    print(
        "  ACTIVE:",
        stats[
            "active_pixels"
        ]
    )

    print(
        "  MEAN MULTIPLIER:",
        round(
            stats[
                "mean_multiplier"
            ],
            4
        )
    )

    print(
        "  STRONGEST:",
        round(
            stats[
                "strongest_multiplier"
            ],
            4
        )
    )

    print(
        "  CHANGED RGB PIXELS:",
        int(
            changed.sum()
        )
    )

    if changed.any():

        print(
            "  MEAN RGB CHANGE:",
            round(
                float(
                    diff[
                        changed
                    ].mean()
                ),
                3
            )
        )


# ============================================================
# STATE
# ============================================================

STATE = {

    "stage":
        "07F9",

    "purpose":
        "COMPARE_PRECOMPUTED_SHADOW_MULTIPLIERS_ON_REAL_TRUE_METRIC_TILE",

    "ai_run":
        False,

    "model_run":
        False,

    "tile_projection_rerun":
        False,

    "candidates": {

        "A":
            stats_A,

        "B":
            stats_B,

        "C":
            stats_C,
    },

    "outputs": {

        "control":
            str(
                CONTROL_PATH
            ),

        "A":
            str(
                A_PATH
            ),

        "B":
            str(
                B_PATH
            ),

        "C":
            str(
                C_PATH
            ),

        "comparison_audit":
            str(
                AUDIT_PATH
            ),

        "strength_audit":
            str(
                STRENGTH_AUDIT_PATH
            ),
    },

    "status":
        "RND_VISUAL_SELECTION_REQUIRED"
}


STATE_PATH = (
    OUT
    / "00_stage07f9_result.json"
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
    "COMPARISON:",
    AUDIT_PATH
)

print(
    "STRENGTH:",
    STRENGTH_AUDIT_PATH
)

print(
    "STATE:",
    STATE_PATH
)

print()
print(
    "NO MODEL WAS RUN."
)

print(
    "THIS IS ONLY A DETERMINISTIC RUNTIME COMPARISON."
)
