
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

ORIGINAL_PATH = (
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

OUT = (
    PROD
    / "stage07_empty_room"
    / "07d3_floor_contact_shadow"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# VALIDATE
# ============================================================

for p in [
    ORIGINAL_PATH,
    FLOOR_MASK_PATH,
    P01_MASK_PATH,
    P02_MASK_PATH,
]:

    if not p.exists():
        raise FileNotFoundError(p)


# ============================================================
# LOAD
# ============================================================

original = np.asarray(
    Image.open(
        ORIGINAL_PATH
    ).convert("RGB")
).astype(
    np.float32
)


floor = (
    np.asarray(
        Image.open(
            FLOOR_MASK_PATH
        ).convert("L")
    )
    >
    127
)


p01 = (
    np.asarray(
        Image.open(
            P01_MASK_PATH
        ).convert("L")
    )
    >
    127
)


p02 = (
    np.asarray(
        Image.open(
            P02_MASK_PATH
        ).convert("L")
    )
    >
    127
)


H, W = floor.shape


# ============================================================
# LUMINANCE
# ============================================================

luma = (
    0.2126 * original[..., 0]
    +
    0.7152 * original[..., 1]
    +
    0.0722 * original[..., 2]
)


# ============================================================
# TRUE FLOOR-ADJACENCY CONTACT
#
# A floor contact seed is a floor pixel lying immediately
# outside the prop mask.
#
# This avoids assumptions about object bottom position.
# ============================================================

def contact_seed_from_floor_adjacency(
    prop_mask,
    floor_mask,
    adjacency_px=5
):

    k = (
        adjacency_px * 2
        +
        1
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            k,
            k
        )
    )


    dilated_prop = (
        cv2.dilate(
            prop_mask.astype(
                np.uint8
            )
            *
            255,
            kernel
        )
        >
        0
    )


    contact_seed = (
        dilated_prop
        &
        floor_mask
        &
        (~prop_mask)
    )


    return contact_seed


p01_contact = contact_seed_from_floor_adjacency(
    p01,
    floor,
    adjacency_px=5
)


p02_contact = contact_seed_from_floor_adjacency(
    p02,
    floor,
    adjacency_px=5
)


# ============================================================
# DISTANCE-BASED SEARCH REGION
#
# Grow naturally outward through floor pixels from contact.
#
# This is not an isotropic prop dilation anymore.
# ============================================================

def build_floor_distance_region(
    contact_seed,
    floor_mask,
    max_distance_px
):

    # distanceTransform measures distance to zeros,
    # so invert the contact seed.

    inv = (
        ~contact_seed
    ).astype(
        np.uint8
    )


    dist = cv2.distanceTransform(
        inv,
        cv2.DIST_L2,
        5
    )


    region = (
        (dist <= float(max_distance_px))
        &
        floor_mask
    )


    return (
        region,
        dist
    )


# Larger region for vanity
p01_region, p01_dist = build_floor_distance_region(
    p01_contact,
    floor,
    max_distance_px=58
)


# More compact region for toilet
p02_region, p02_dist = build_floor_distance_region(
    p02_contact,
    floor,
    max_distance_px=42
)


# ============================================================
# EXCLUDE OBJECTS THEMSELVES
# ============================================================

p01_region &= (
    ~p01
)

p02_region &= (
    ~p02
)


# ============================================================
# LOCAL UNSHADOWED FLOOR ESTIMATE
#
# Exclude both objects.
# ============================================================

safe_floor = (
    floor
    &
    (~p01)
    &
    (~p02)
)


safe_f = safe_floor.astype(
    np.float32
)


weighted = (
    luma
    *
    safe_f
)


sigma = 24.0


num = cv2.GaussianBlur(
    weighted,
    (0, 0),
    sigmaX=sigma,
    sigmaY=sigma
)


den = cv2.GaussianBlur(
    safe_f,
    (0, 0),
    sigmaX=sigma,
    sigmaY=sigma
)


reference = (
    num
    /
    np.maximum(
        den,
        1e-5
    )
)


# ============================================================
# DARKENING RATIO
# ============================================================

ratio = (
    luma
    /
    np.maximum(
        reference,
        1.0
    )
)


ratio = np.clip(
    ratio,
    0.45,
    1.20
)


raw_shadow = np.clip(
    1.0
    -
    ratio,
    0.0,
    0.50
)


# ============================================================
# DISTANCE PRIOR
#
# Physical cast/contact shadow confidence is highest around
# the object contact area and gradually decreases outward.
#
# This does NOT create a shadow.
# It only reduces false floor-texture detections far away.
# ============================================================

def distance_confidence(
    dist,
    max_dist
):

    conf = (
        1.0
        -
        dist
        /
        float(
            max_dist
        )
    )


    return np.clip(
        conf,
        0.0,
        1.0
    )


p01_conf = distance_confidence(
    p01_dist,
    58
)


p02_conf = distance_confidence(
    p02_dist,
    42
)


# ============================================================
# EXTRACT OBJECT SHADOW
# ============================================================

def extract(
    raw,
    region,
    confidence,
    threshold,
    min_area,
    blur
):

    s = raw.copy()

    s[
        ~region
    ] = 0.0


    # Confidence only suppresses remote false positives.
    weighted_strength = (
        s
        *
        (
            0.55
            +
            0.45
            *
            confidence
        )
    )


    weighted_strength[
        weighted_strength
        <
        threshold
    ] = 0.0


    binary = (
        weighted_strength
        >
        0
    ).astype(
        np.uint8
    )


    n, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )
    )


    clean = np.zeros_like(
        binary
    )


    for i in range(
        1,
        n
    ):

        area = int(
            stats[
                i,
                cv2.CC_STAT_AREA
            ]
        )


        if area >= min_area:

            clean[
                labels == i
            ] = 1


    weighted_strength[
        clean == 0
    ] = 0.0


    weighted_strength = cv2.GaussianBlur(
        weighted_strength.astype(
            np.float32
        ),
        (0, 0),
        sigmaX=blur,
        sigmaY=blur
    )


    weighted_strength[
        ~region
    ] = 0.0


    return weighted_strength


p01_shadow = extract(
    raw_shadow,
    p01_region,
    p01_conf,
    threshold=0.032,
    min_area=14,
    blur=1.6
)


p02_shadow = extract(
    raw_shadow,
    p02_region,
    p02_conf,
    threshold=0.040,
    min_area=10,
    blur=1.4
)


# ============================================================
# COMBINE
# ============================================================

combined = np.maximum(
    p01_shadow,
    p02_shadow
)


combined = np.clip(
    combined,
    0.0,
    0.50
)


multiplier = (
    1.0
    -
    combined
)


multiplier = np.clip(
    multiplier,
    0.50,
    1.00
)


multiplier[
    ~floor
] = 1.0


# ============================================================
# SAVE 16-BIT MULTIPLIER
# ============================================================

mult16 = (
    multiplier
    *
    65535.0
).round().astype(
    np.uint16
)


MULT_PATH = (
    OUT
    / "01_floor_shadow_multiplier_16bit.png"
)


Image.fromarray(
    mult16,
    mode="I;16"
).save(
    MULT_PATH
)


# ============================================================
# VIS HELPERS
# ============================================================

def strength_to_u8(x):

    return (
        np.clip(
            x
            /
            0.50,
            0.0,
            1.0
        )
        *
        255
    ).astype(
        np.uint8
    )


# ============================================================
# CONTACT VIS
# ============================================================

contact_vis = np.zeros(
    (
        H,
        W,
        3
    ),
    dtype=np.uint8
)


# P01 contact red
contact_vis[
    p01_contact
] = [
    255,
    0,
    0
]


# P02 contact cyan
contact_vis[
    p02_contact
] = [
    0,
    255,
    255
]


contact_vis[
    p01_contact
    &
    p02_contact
] = [
    255,
    255,
    0
]


# ============================================================
# SEARCH REGION VIS
# ============================================================

region_vis = np.zeros(
    (
        H,
        W,
        3
    ),
    dtype=np.uint8
)


region_vis[
    p01_region
] = [
    255,
    0,
    0
]


region_vis[
    p02_region
] = [
    0,
    255,
    255
]


region_vis[
    p01_region
    &
    p02_region
] = [
    255,
    255,
    0
]


# ============================================================
# LOCATION AUDIT
# ============================================================

overlay = original.copy()


a1 = np.clip(
    p01_shadow
    /
    0.35,
    0.0,
    0.65
)


red = np.zeros_like(
    overlay
)

red[..., 0] = 255


overlay = (
    overlay
    *
    (
        1.0
        -
        a1[..., None]
    )
    +
    red
    *
    a1[..., None]
)


a2 = np.clip(
    p02_shadow
    /
    0.35,
    0.0,
    0.65
)


cyan = np.zeros_like(
    overlay
)

cyan[..., 1] = 255
cyan[..., 2] = 255


overlay = (
    overlay
    *
    (
        1.0
        -
        a2[..., None]
    )
    +
    cyan
    *
    a2[..., None]
)


overlay = np.clip(
    overlay,
    0,
    255
).astype(
    np.uint8
)


# ============================================================
# MATERIAL-INDEPENDENCE TESTS
# ============================================================

neutral = np.full(
    (
        H,
        W,
        3
    ),
    210.0,
    dtype=np.float32
)


blue = np.zeros(
    (
        H,
        W,
        3
    ),
    dtype=np.float32
)


blue[..., 0] = 65
blue[..., 1] = 155
blue[..., 2] = 225


neutral_result = neutral.copy()
blue_result = blue.copy()


neutral_result[
    floor
] *= multiplier[
    floor,
    None
]


blue_result[
    floor
] *= multiplier[
    floor,
    None
]


neutral_result = np.clip(
    neutral_result,
    0,
    255
).astype(
    np.uint8
)


blue_result = np.clip(
    blue_result,
    0,
    255
).astype(
    np.uint8
)


# ============================================================
# SAVE MAPS
# ============================================================

P01_PATH = (
    OUT
    / "02_p01_vanity_shadow.png"
)


P02_PATH = (
    OUT
    / "03_p02_toilet_shadow.png"
)


COMBINED_PATH = (
    OUT
    / "04_combined_shadow.png"
)


Image.fromarray(
    strength_to_u8(
        p01_shadow
    )
).save(
    P01_PATH
)


Image.fromarray(
    strength_to_u8(
        p02_shadow
    )
).save(
    P02_PATH
)


Image.fromarray(
    strength_to_u8(
        combined
    )
).save(
    COMBINED_PATH
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
    original.astype(
        np.uint8
    )
)

axes[0].set_title(
    "1. Original"
)

axes[0].axis(
    "off"
)


axes[1].imshow(
    contact_vis
)

axes[1].set_title(
    "2. TRUE Floor Contact\n"
    "RED=P01 | CYAN=P02"
)

axes[1].axis(
    "off"
)


axes[2].imshow(
    region_vis
)

axes[2].set_title(
    "3. Contact-Grown Regions"
)

axes[2].axis(
    "off"
)


axes[3].imshow(
    strength_to_u8(
        p01_shadow
    ),
    cmap="gray",
    vmin=0,
    vmax=255
)

axes[3].set_title(
    "4. P01 Vanity Shadow"
)

axes[3].axis(
    "off"
)


axes[4].imshow(
    strength_to_u8(
        p02_shadow
    ),
    cmap="gray",
    vmin=0,
    vmax=255
)

axes[4].set_title(
    "5. P02 Toilet Shadow"
)

axes[4].axis(
    "off"
)


axes[5].imshow(
    overlay
)

axes[5].set_title(
    "6. Location Audit"
)

axes[5].axis(
    "off"
)


axes[6].imshow(
    neutral_result
)

axes[6].set_title(
    "7. Neutral Surface"
)

axes[6].axis(
    "off"
)


axes[7].imshow(
    blue_result
)

axes[7].set_title(
    "8. Blue Surface"
)

axes[7].axis(
    "off"
)


plt.tight_layout()


AUDIT_PATH = (
    OUT
    / "05_stage07d3_floor_contact_shadow_audit.png"
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

print()
print("=" * 110)
print("STAGE 07D3 RESULT")
print("=" * 110)

print()

print(
    "P01 CONTACT PIXELS:",
    int(
        p01_contact.sum()
    )
)

print(
    "P01 SEARCH PIXELS:",
    int(
        p01_region.sum()
    )
)

print(
    "P01 ACTIVE SHADOW:",
    int(
        (
            p01_shadow
            >
            0.01
        ).sum()
    )
)

print()

print(
    "P02 CONTACT PIXELS:",
    int(
        p02_contact.sum()
    )
)

print(
    "P02 SEARCH PIXELS:",
    int(
        p02_region.sum()
    )
)

print(
    "P02 ACTIVE SHADOW:",
    int(
        (
            p02_shadow
            >
            0.01
        ).sum()
    )
)

print()

active = (
    combined
    >
    0.01
)


print(
    "COMBINED ACTIVE:",
    int(
        active.sum()
    )
)


if active.any():

    print(
        "MEAN MULTIPLIER:",
        round(
            float(
                multiplier[
                    active
                ].mean()
            ),
            4
        )
    )


    print(
        "STRONGEST MULTIPLIER:",
        round(
            float(
                multiplier[
                    active
                ].min()
            ),
            4
        )
    )


# ============================================================
# STATE
# ============================================================

STATE = {

    "stage":
        "07D3",

    "architecture":
        "PRE_TILE_CONTACT_DERIVED_SHADOW_MAP",

    "method":
        (
            "floor adjacency -> contact seed -> "
            "distance-grown floor search -> "
            "local luminance shadow extraction"
        ),

    "runs_before_tile_selection":
        True,

    "contains_original_floor_rgb":
        False,

    "objects": {

        "P01":
            "vanity",

        "P02":
            "toilet"
    },

    "outputs": {

        "multiplier":
            str(
                MULT_PATH
            ),

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

        "audit":
            str(
                AUDIT_PATH
            )
    },

    "status":
        "RND_REQUIRES_VISUAL_AUDIT"
}


STATE_PATH = (
    OUT
    / "00_stage07d3_result.json"
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
    "AUDIT:",
    AUDIT_PATH
)

print(
    "MULTIPLIER:",
    MULT_PATH
)

print(
    "STATE:",
    STATE_PATH
)

print()
print(
    "NO TILE SELECTION WAS USED."
)

print(
    "NO GENERATIVE MODEL WAS USED."
)
