
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
    / "07d2_object_specific_floor_shadow"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD / VALIDATE
# ============================================================

for p in [
    ORIGINAL_PATH,
    FLOOR_MASK_PATH,
    P01_MASK_PATH,
    P02_MASK_PATH,
]:
    if not p.exists():
        raise FileNotFoundError(p)


original = np.asarray(
    Image.open(ORIGINAL_PATH).convert("RGB")
).astype(np.float32)

floor = (
    np.asarray(
        Image.open(FLOOR_MASK_PATH).convert("L")
    ) > 127
)

p01 = (
    np.asarray(
        Image.open(P01_MASK_PATH).convert("L")
    ) > 127
)

p02 = (
    np.asarray(
        Image.open(P02_MASK_PATH).convert("L")
    ) > 127
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
# OBJECT-SPECIFIC CONTACT-BAND GENERATOR
#
# We do not dilate the whole object equally.
#
# We first locate the object's lowest visible boundary and
# generate a search zone from that contact region.
# ============================================================

def build_contact_search_region(
    object_mask,
    floor_mask,
    downward_px,
    side_px,
    upper_allow_px=3
):
    ys, xs = np.where(object_mask)

    if len(xs) == 0:
        raise RuntimeError("Object mask is empty.")

    y_bottom = int(ys.max())

    # Pixels of object close to its lowest visible region.
    contact_band = (
        object_mask
        &
        (
            np.indices(object_mask.shape)[0]
            >=
            y_bottom - 12
        )
    )

    # Horizontal expansion.
    side_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            side_px * 2 + 1,
            9
        )
    )

    expanded = cv2.dilate(
        contact_band.astype(np.uint8) * 255,
        side_kernel
    ) > 0

    # Directional downward extension.
    region = expanded.copy()

    for dy in range(
        -upper_allow_px,
        downward_px + 1
    ):
        shifted = np.zeros_like(expanded)

        if dy >= 0:
            if dy < H:
                shifted[dy:, :] = expanded[:H-dy, :]
        else:
            d = abs(dy)
            if d < H:
                shifted[:H-d, :] = expanded[d:, :]

        region |= shifted

    region &= floor_mask
    region &= ~object_mask

    return region, contact_band


# ============================================================
# INDIVIDUAL REGIONS
#
# Vanity gets wider + longer search.
# Toilet gets more compact radial/contact region.
# ============================================================

p01_region, p01_contact = build_contact_search_region(
    p01,
    floor,
    downward_px=58,
    side_px=30,
    upper_allow_px=3
)

p02_region, p02_contact = build_contact_search_region(
    p02,
    floor,
    downward_px=38,
    side_px=22,
    upper_allow_px=4
)


# ============================================================
# FLOOR REFERENCE FIELD
#
# Exclude BOTH P01 and P02.
# ============================================================

safe_floor = (
    floor
    &
    (~p01)
    &
    (~p02)
)

safe_float = safe_floor.astype(np.float32)

weighted = (
    luma
    *
    safe_float
)

# Large enough to estimate local unshadowed illumination,
# but not so large that the whole room becomes one value.
sigma = 22.0

num = cv2.GaussianBlur(
    weighted,
    (0, 0),
    sigmaX=sigma,
    sigmaY=sigma
)

den = cv2.GaussianBlur(
    safe_float,
    (0, 0),
    sigmaX=sigma,
    sigmaY=sigma
)

reference_luma = (
    num
    /
    np.maximum(
        den,
        1e-5
    )
)


# ============================================================
# RAW DARKENING
# ============================================================

ratio = (
    luma
    /
    np.maximum(
        reference_luma,
        1.0
    )
)

ratio = np.clip(
    ratio,
    0.45,
    1.20
)

raw_strength = np.clip(
    1.0 - ratio,
    0.0,
    0.50
)


# ============================================================
# OBJECT-SPECIFIC EXTRACTION
# ============================================================

def extract_shadow(
    raw,
    region,
    min_strength,
    blur_sigma,
    min_component_area
):
    s = raw.copy()

    s[~region] = 0.0

    s[
        s < min_strength
    ] = 0.0

    binary = (
        s > 0
    ).astype(np.uint8)


    # Connected components:
    # discard tiny texture fragments.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8
    )

    cleaned = np.zeros_like(binary)

    for i in range(1, n):
        area = stats[
            i,
            cv2.CC_STAT_AREA
        ]

        if area >= min_component_area:
            cleaned[
                labels == i
            ] = 1


    s[
        cleaned == 0
    ] = 0.0


    if blur_sigma > 0:
        s = cv2.GaussianBlur(
            s.astype(np.float32),
            (0, 0),
            sigmaX=blur_sigma,
            sigmaY=blur_sigma
        )

        s[
            ~region
        ] = 0.0


    return s


# Vanity:
# slightly lower threshold because its shadow is weaker.
p01_shadow = extract_shadow(
    raw_strength,
    p01_region,
    min_strength=0.030,
    blur_sigma=1.6,
    min_component_area=10
)

# Toilet:
# stronger visible contact shadow.
p02_shadow = extract_shadow(
    raw_strength,
    p02_region,
    min_strength=0.040,
    blur_sigma=1.4,
    min_component_area=8
)


# ============================================================
# COMBINE
#
# max() is appropriate:
# strongest shadow wins if regions overlap.
# ============================================================

combined_strength = np.maximum(
    p01_shadow,
    p02_shadow
)

combined_strength = np.clip(
    combined_strength,
    0.0,
    0.50
)


shadow_multiplier = (
    1.0 - combined_strength
)

shadow_multiplier = np.clip(
    shadow_multiplier,
    0.50,
    1.00
)

shadow_multiplier[
    ~floor
] = 1.0


# ============================================================
# SAVE PRECISION MAP
# ============================================================

mult16 = (
    shadow_multiplier
    *
    65535.0
).round().astype(np.uint16)

MULT_PATH = (
    OUT
    / "01_combined_shadow_multiplier_16bit.png"
)

Image.fromarray(
    mult16,
    mode="I;16"
).save(
    MULT_PATH
)


# ============================================================
# SAVE INDIVIDUAL SHADOW MAPS
# ============================================================

def strength_to_u8(x):
    return (
        np.clip(
            x / 0.50,
            0.0,
            1.0
        )
        * 255
    ).astype(np.uint8)


P01_PATH = (
    OUT
    / "02_p01_vanity_shadow_strength.png"
)

P02_PATH = (
    OUT
    / "03_p02_toilet_shadow_strength.png"
)

COMBINED_PATH = (
    OUT
    / "04_combined_shadow_strength.png"
)


Image.fromarray(
    strength_to_u8(p01_shadow)
).save(P01_PATH)

Image.fromarray(
    strength_to_u8(p02_shadow)
).save(P02_PATH)

Image.fromarray(
    strength_to_u8(combined_strength)
).save(COMBINED_PATH)


# ============================================================
# SEARCH REGION VISUALIZATION
# ============================================================

region_vis = np.zeros(
    (
        H,
        W,
        3
    ),
    dtype=np.uint8
)

# Vanity = red
region_vis[
    p01_region
] = [
    255,
    0,
    0
]

# Toilet = cyan
region_vis[
    p02_region
] = [
    0,
    255,
    255
]

# overlap = yellow
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
# ORIGINAL LOCATION OVERLAY
# ============================================================

overlay = original.copy()

# vanity shadow = red
a1 = np.clip(
    p01_shadow / 0.35,
    0.0,
    0.65
)

red = np.zeros_like(overlay)
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


# toilet shadow = cyan
a2 = np.clip(
    p02_shadow / 0.35,
    0.0,
    0.65
)

cyan = np.zeros_like(overlay)
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
).astype(np.uint8)


# ============================================================
# MATERIAL-INDEPENDENCE TEST
#
# Apply same map to:
#
# A) neutral gray
# B) synthetic saturated blue
#
# If both preserve their base color/design concept while only
# darkening, the map is material-independent.
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

blue[..., 0] = 70
blue[..., 1] = 155
blue[..., 2] = 220


neutral_shadowed = neutral.copy()
blue_shadowed = blue.copy()


neutral_shadowed[
    floor
] *= shadow_multiplier[
    floor,
    None
]


blue_shadowed[
    floor
] *= shadow_multiplier[
    floor,
    None
]


neutral_shadowed = np.clip(
    neutral_shadowed,
    0,
    255
).astype(np.uint8)

blue_shadowed = np.clip(
    blue_shadowed,
    0,
    255
).astype(np.uint8)


# ============================================================
# 7-PANEL AUDIT
# ============================================================

fig, axes = plt.subplots(
    1,
    7,
    figsize=(30, 7)
)


axes[0].imshow(
    original.astype(np.uint8)
)
axes[0].set_title(
    "1. Original"
)
axes[0].axis("off")


axes[1].imshow(
    region_vis
)
axes[1].set_title(
    "2. Object Search Regions\nRED=P01 | CYAN=P02"
)
axes[1].axis("off")


axes[2].imshow(
    strength_to_u8(p01_shadow),
    cmap="gray",
    vmin=0,
    vmax=255
)
axes[2].set_title(
    "3. P01 Vanity Shadow"
)
axes[2].axis("off")


axes[3].imshow(
    strength_to_u8(p02_shadow),
    cmap="gray",
    vmin=0,
    vmax=255
)
axes[3].set_title(
    "4. P02 Toilet Shadow"
)
axes[3].axis("off")


axes[4].imshow(
    overlay
)
axes[4].set_title(
    "5. Location Audit"
)
axes[4].axis("off")


axes[5].imshow(
    neutral_shadowed
)
axes[5].set_title(
    "6. Neutral Surface Test"
)
axes[5].axis("off")


axes[6].imshow(
    blue_shadowed
)
axes[6].set_title(
    "7. Blue Surface Test"
)
axes[6].axis("off")


plt.tight_layout()

AUDIT_PATH = (
    OUT
    / "05_stage07d2_object_shadow_audit.png"
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

p01_active = int(
    (
        p01_shadow > 0.01
    ).sum()
)

p02_active = int(
    (
        p02_shadow > 0.01
    ).sum()
)

combined_active = int(
    (
        combined_strength > 0.01
    ).sum()
)


print()
print("=" * 110)
print("STAGE 07D2 RESULT")
print("=" * 110)

print()

print(
    "P01 REGION PIXELS:",
    int(p01_region.sum())
)

print(
    "P01 ACTIVE SHADOW:",
    p01_active
)

print()

print(
    "P02 REGION PIXELS:",
    int(p02_region.sum())
)

print(
    "P02 ACTIVE SHADOW:",
    p02_active
)

print()

print(
    "COMBINED ACTIVE:",
    combined_active
)

if combined_active:

    active = combined_strength > 0.01

    print(
        "MEAN MULTIPLIER:",
        round(
            float(
                shadow_multiplier[
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
                shadow_multiplier[
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
        "07D2",

    "architecture":
        "PRE_TILE_OBJECT_SPECIFIC_SHADOW_EXTRACTION",

    "shadow_objects": {

        "P01":
            "complete vanity system",

        "P02":
            "complete toilet system"
    },

    "runs_before_tile_selection":
        True,

    "contains_original_surface_rgb":
        False,

    "runtime_after_tile_selection":
        "NONE",

    "outputs": {

        "multiplier_16bit":
            str(MULT_PATH),

        "p01_shadow":
            str(P01_PATH),

        "p02_shadow":
            str(P02_PATH),

        "combined_shadow":
            str(COMBINED_PATH),

        "audit":
            str(AUDIT_PATH)
    },

    "status":
        "RND_REQUIRES_VISUAL_AUDIT"
}


STATE_PATH = (
    OUT
    / "00_stage07d2_result.json"
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
    "NO TILE WAS SELECTED OR MODIFIED."
)

print(
    "NO GENERATIVE MODEL WAS RUN."
)
