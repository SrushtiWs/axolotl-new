
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

PROD = BASE / "test07" / "production_pipeline"

STAGE05F = (
    PROD
    / "stage05_clean_room_with_props"
    / "12_final_canonical_prop_detection_master.png"
)

STAGE07A = (
    PROD
    / "stage07_empty_room"
    / "07a_qwen_empty_room"
    / "00_stage07_empty_room_candidate.png"
)

FLOOR_MASK_PATH = (
    PROD
    / "stage02_structure"
    / "17_floor_majority_2of3.png"
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
    / "07b2_conservative_floor_shadow"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

# Search around each object.
VANITY_RADIUS = 45
TOILET_RADIUS = 38

# Remove floor-mask boundary zone.
FLOOR_BOUNDARY_ERODE = 8

# Ignore weak differences.
MIN_DROP = 7.0

# Shadow ratio limit.
MAX_SHADOW_RATIO = 0.965

# Prevent extreme darkening.
MIN_MULTIPLIER = 0.58

# Reject structural/high-gradient areas.
GRADIENT_THRESHOLD = 20.0

# Softening.
GAUSSIAN_SIGMA = 3.0


# ============================================================
# LOAD
# ============================================================

for p in [
    STAGE05F,
    STAGE07A,
    FLOOR_MASK_PATH,
    P01_MASK_PATH,
    P02_MASK_PATH,
]:
    if not p.exists():
        raise FileNotFoundError(p)


img_orig = Image.open(
    STAGE05F
).convert("RGB")

img_empty = Image.open(
    STAGE07A
).convert("RGB")


if img_orig.size != img_empty.size:
    raise RuntimeError(
        "Stage05F and Stage07A size mismatch."
    )


W, H = img_orig.size


orig = np.asarray(
    img_orig
).astype(np.float32)

empty = np.asarray(
    img_empty
).astype(np.float32)


def load_mask(path):
    return (
        np.asarray(
            Image.open(path).convert("L")
        )
        > 127
    )


floor = load_mask(
    FLOOR_MASK_PATH
)

p01 = load_mask(
    P01_MASK_PATH
)

p02 = load_mask(
    P02_MASK_PATH
)


# ============================================================
# SAFE FLOOR INTERIOR
#
# Erode floor mask so baseboard / wall-floor boundaries
# cannot become shadow.
# ============================================================

kernel_floor = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (
        FLOOR_BOUNDARY_ERODE * 2 + 1,
        FLOOR_BOUNDARY_ERODE * 2 + 1
    )
)


safe_floor = cv2.erode(
    floor.astype(np.uint8),
    kernel_floor,
    iterations=1
).astype(bool)


# ============================================================
# OBJECT-LOCAL SEARCH REGIONS
# ============================================================

def dilate(mask, radius):

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            radius * 2 + 1,
            radius * 2 + 1
        )
    )

    return cv2.dilate(
        mask.astype(np.uint8),
        kernel,
        iterations=1
    ).astype(bool)


p01_roi = (
    dilate(
        p01,
        VANITY_RADIUS
    )
    &
    ~p01
    &
    safe_floor
)


p02_roi = (
    dilate(
        p02,
        TOILET_RADIUS
    )
    &
    ~p02
    &
    safe_floor
)


combined_roi = (
    p01_roi
    |
    p02_roi
)


# ============================================================
# LUMINANCE
# ============================================================

def luminance(rgb):

    return (
        0.2126 * rgb[..., 0]
        +
        0.7152 * rgb[..., 1]
        +
        0.0722 * rgb[..., 2]
    )


lum_orig = luminance(
    orig
)

lum_empty = luminance(
    empty
)


# ============================================================
# LOCAL ILLUMINATION NORMALIZATION
#
# Qwen slightly changed overall brightness.
# Estimate one robust scale using SAFE floor pixels outside
# the shadow search region.
# ============================================================

reference_region = (
    safe_floor
    &
    ~combined_roi
    &
    ~p01
    &
    ~p02
)


valid_ref = (
    reference_region
    &
    (lum_empty > 20)
)


ratios_ref = (
    lum_orig[valid_ref]
    /
    np.maximum(
        lum_empty[valid_ref],
        1.0
    )
)


if len(ratios_ref) == 0:

    illumination_scale = 1.0

else:

    illumination_scale = float(
        np.median(
            ratios_ref
        )
    )


normalized_empty_luma = (
    lum_empty
    *
    illumination_scale
)


print("=" * 110)
print("STAGE 07B2 — CONSERVATIVE FLOOR SHADOW")
print("=" * 110)

print()
print(
    "LOCAL ILLUMINATION SCALE:",
    round(
        illumination_scale,
        5
    )
)


# ============================================================
# STRUCTURAL EDGE REJECTION
#
# Use empty-room gradient to detect strong geometry edges.
# ============================================================

empty_gray = cv2.cvtColor(
    np.asarray(
        img_empty
    ),
    cv2.COLOR_RGB2GRAY
).astype(np.float32)


gx = cv2.Sobel(
    empty_gray,
    cv2.CV_32F,
    1,
    0,
    ksize=3
)

gy = cv2.Sobel(
    empty_gray,
    cv2.CV_32F,
    0,
    1,
    ksize=3
)


gradient = np.sqrt(
    gx * gx
    +
    gy * gy
)


smooth_surface = (
    gradient
    <
    GRADIENT_THRESHOLD
)


# ============================================================
# SHADOW CANDIDATE
# ============================================================

drop = (
    normalized_empty_luma
    -
    lum_orig
)


ratio = (
    lum_orig
    /
    np.maximum(
        normalized_empty_luma,
        1.0
    )
)


candidate = (
    combined_roi
    &
    smooth_surface
    &
    (drop >= MIN_DROP)
    &
    (ratio <= MAX_SHADOW_RATIO)
)


# ============================================================
# BUILD RAW STRENGTH
# ============================================================

strength = np.zeros(
    (
        H,
        W
    ),
    dtype=np.float32
)


strength[
    candidate
] = (
    1.0
    -
    np.clip(
        ratio[candidate],
        MIN_MULTIPLIER,
        1.0
    )
)


# ============================================================
# REMOVE TINY ISLANDS
# ============================================================

binary_u8 = (
    candidate.astype(
        np.uint8
    )
    *
    255
)


num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
    binary_u8,
    connectivity=8
)


clean_candidate = np.zeros(
    (
        H,
        W
    ),
    dtype=bool
)


MIN_COMPONENT_AREA = 12


for label_id in range(
    1,
    num_labels
):

    area = stats[
        label_id,
        cv2.CC_STAT_AREA
    ]

    if area >= MIN_COMPONENT_AREA:

        clean_candidate[
            labels == label_id
        ] = True


strength *= (
    clean_candidate.astype(
        np.float32
    )
)


# ============================================================
# SOFTEN
# ============================================================

strength = cv2.GaussianBlur(
    strength,
    (0, 0),
    sigmaX=GAUSSIAN_SIGMA,
    sigmaY=GAUSSIAN_SIGMA
)


# Do not let blur escape ROI.
strength *= (
    combined_roi.astype(
        np.float32
    )
)


strength = np.clip(
    strength,
    0.0,
    1.0 - MIN_MULTIPLIER
)


multiplier = (
    1.0
    -
    strength
)


# ============================================================
# SAVE MASKS
# ============================================================

ROI_PATH = (
    OUT
    / "01_floor_shadow_search_roi.png"
)


Image.fromarray(
    combined_roi.astype(np.uint8)
    *
    255
).save(
    ROI_PATH
)


CANDIDATE_PATH = (
    OUT
    / "02_clean_shadow_candidate.png"
)


Image.fromarray(
    clean_candidate.astype(np.uint8)
    *
    255
).save(
    CANDIDATE_PATH
)


STRENGTH_PATH = (
    OUT
    / "03_floor_shadow_strength.png"
)


Image.fromarray(
    (
        strength
        *
        255
    ).astype(np.uint8)
).save(
    STRENGTH_PATH
)


MULTIPLIER_PATH = (
    OUT
    / "04_floor_shadow_multiplier.png"
)


Image.fromarray(
    (
        multiplier
        *
        255
    ).astype(np.uint8)
).save(
    MULTIPLIER_PATH
)


# ============================================================
# REAPPLY TO EMPTY ROOM
# ============================================================

reapplied = (
    empty
    *
    multiplier[
        ...,
        None
    ]
)


reapplied = np.clip(
    reapplied,
    0,
    255
).astype(np.uint8)


REAPPLIED_PATH = (
    OUT
    / "05_stage07_reapplied_floor_shadows.png"
)


Image.fromarray(
    reapplied
).save(
    REAPPLIED_PATH
)


# ============================================================
# 6-PANEL AUDIT
# ============================================================

fig, axes = plt.subplots(
    1,
    6,
    figsize=(
        25,
        7
    )
)


axes[0].imshow(
    img_orig
)

axes[0].set_title(
    "1. Stage05F"
)

axes[0].axis(
    "off"
)


axes[1].imshow(
    img_empty
)

axes[1].set_title(
    "2. Stage07A"
)

axes[1].axis(
    "off"
)


axes[2].imshow(
    combined_roi,
    cmap="gray"
)

axes[2].set_title(
    "3. P01/P02 Floor ROI"
)

axes[2].axis(
    "off"
)


axes[3].imshow(
    clean_candidate,
    cmap="gray"
)

axes[3].set_title(
    "4. Clean Shadow Candidate"
)

axes[3].axis(
    "off"
)


axes[4].imshow(
    strength,
    cmap="gray",
    vmin=0,
    vmax=0.42
)

axes[4].set_title(
    "5. Shadow Strength"
)

axes[4].axis(
    "off"
)


axes[5].imshow(
    reapplied
)

axes[5].set_title(
    "6. Stage07 + Floor Shadows"
)

axes[5].axis(
    "off"
)


plt.tight_layout()


AUDIT_PATH = (
    OUT
    / "06_stage07b2_audit.png"
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
        "07B2",

    "approach":
        "CONSERVATIVE_P01_P02_FLOOR_SHADOW_ONLY",

    "illumination_scale":
        illumination_scale,

    "settings": {

        "vanity_radius":
            VANITY_RADIUS,

        "toilet_radius":
            TOILET_RADIUS,

        "floor_boundary_erode":
            FLOOR_BOUNDARY_ERODE,

        "min_drop":
            MIN_DROP,

        "max_shadow_ratio":
            MAX_SHADOW_RATIO,

        "min_multiplier":
            MIN_MULTIPLIER,

        "gradient_threshold":
            GRADIENT_THRESHOLD,

        "gaussian_sigma":
            GAUSSIAN_SIGMA,
    },

    "statistics": {

        "roi_pixels":
            int(
                combined_roi.sum()
            ),

        "clean_candidate_pixels":
            int(
                clean_candidate.sum()
            ),

        "active_soft_shadow_pixels":
            int(
                (
                    strength
                    >
                    0.01
                ).sum()
            )
    },

    "outputs": {

        "roi":
            str(
                ROI_PATH
            ),

        "candidate":
            str(
                CANDIDATE_PATH
            ),

        "strength":
            str(
                STRENGTH_PATH
            ),

        "multiplier":
            str(
                MULTIPLIER_PATH
            ),

        "reapplied_test":
            str(
                REAPPLIED_PATH
            ),

        "audit":
            str(
                AUDIT_PATH
            )
    },

    "status":
        "REQUIRES_VISUAL_AUDIT"
}


STATE_PATH = (
    OUT
    / "00_stage07b2_result.json"
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
print("STAGE 07B2 RESULT")
print("=" * 110)

print()
print(
    "ILLUMINATION SCALE:",
    round(
        illumination_scale,
        5
    )
)

print(
    "ROI PIXELS:",
    int(
        combined_roi.sum()
    )
)

print(
    "CLEAN SHADOW PIXELS:",
    int(
        clean_candidate.sum()
    )
)

print(
    "ACTIVE SOFT SHADOW PIXELS:",
    int(
        (
            strength
            >
            0.01
        ).sum()
    )
)

print()
print(
    "SHADOW MULTIPLIER:",
    MULTIPLIER_PATH
)

print(
    "AUDIT:",
    AUDIT_PATH
)

print(
    "STATE:",
    STATE_PATH
)
