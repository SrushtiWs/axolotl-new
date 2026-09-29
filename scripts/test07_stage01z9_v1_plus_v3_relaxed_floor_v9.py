
# ============================================================
# TEST07 - ROOM03
# STAGE01Z9
#
# V1 + V3 RELAXED FLOOR REFERENCE
#
# V1:
#   best coherent clean-room reconstruction
#
# V3:
#   semantic floor source
#
# V9:
#   recover both bright green and dark shadowed green
#   using relaxed color thresholds + lower-region connectivity
#
# FINAL:
#   V1 appearance
#   + exact #00C800 on V9 floor mask
#   + exact original RGB on frozen props
#
# NO NEW AI GENERATION.
# ============================================================

from pathlib import Path
from PIL import Image, ImageDraw
from IPython.display import display

import hashlib
import json
import numpy as np
import cv2


# ============================================================
# 1. IDENTIFIERS
# ============================================================

STAGE_NAME = (
    "TEST07_STAGE01Z9_"
    "V1_PLUS_V3_RELAXED_FLOOR_V9"
)

SCRIPT_NAME = (
    "test07_stage01z9_"
    "v1_plus_v3_relaxed_floor_v9.py"
)


# ============================================================
# 2. PATHS
# ============================================================

BASE = Path(
    "/workspace/axolotl"
)

ROOT = (
    BASE
    / "test07"
    / "runs"
    / "room03_bathroom"
    / "stages"
)

OUT = (
    ROOT
    / "01z9_v1_plus_v3_relaxed_floor_v9"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


ORIGINAL_PATH = (
    ROOT
    / "01_structure_inverse_base"
    / "00_input_resized.png"
)


V1_SAFE_PATH = (
    ROOT
    / "01z1_qwen_coherent_cleanroom_v1"
    / "02_safe_props_restored.png"
)


V3_PATH = (
    ROOT
    / "01z3_qwen_short_surface_template_v3"
    / "05_qwen_v3_master_size.png"
)


V8_MASK_PATH = (
    ROOT
    / "01z8_v1_plus_v3_floor_reference_v8"
    / "03_v8_final_floor_mask.png"
)


PROP_MASK_PATH = (
    ROOT
    / "01y3_frozen_canonical_prop_protection"
    / "00_canonical_prop_protection_mask.png"
)


OLD_FLOOR_MASK_PATH = (
    ROOT
    / "01_structure_inverse_base"
    / "04_floor_mask.png"
)


# ============================================================
# 3. TARGET FLOOR COLOR
# ============================================================

FLOOR_HEX = "#00C800"

FLOOR_RGB = np.array(
    [0, 200, 0],
    dtype=np.uint8
)


# ============================================================
# 4. RELAXED GREEN SETTINGS
# ============================================================
#
# V8:
# MIN_GREEN = 100
#
# V9:
# lower brightness threshold to capture shadowed green floor,
# but require green dominance + saturation + lower-image geometry.
# ============================================================

MIN_GREEN = 45

MIN_GREEN_ADVANTAGE_R = 12

MIN_GREEN_ADVANTAGE_B = 12

MIN_SATURATION = 45


# HSV green hue
HUE_MIN = 32
HUE_MAX = 100


# ============================================================
# 5. HELPERS
# ============================================================

def load_rgb(path):

    return np.array(
        Image.open(path).convert("RGB")
    )


def load_rgb_resized(path, size):

    img = Image.open(path).convert("RGB")

    if img.size != size:

        img = img.resize(
            size,
            Image.Resampling.LANCZOS
        )

    return np.array(img)


def load_mask(path, size):

    img = Image.open(path).convert("L")

    if img.size != size:

        img = img.resize(
            size,
            Image.Resampling.NEAREST
        )

    return np.array(img) > 0


def save_mask(mask, path):

    Image.fromarray(
        mask.astype(np.uint8) * 255,
        mode="L"
    ).save(path)


def sha256_file(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:

        while True:

            chunk = f.read(
                1024 * 1024
            )

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


# ============================================================
# 6. VALIDATE INPUTS
# ============================================================

required = {
    "ORIGINAL": ORIGINAL_PATH,
    "V1 SAFE": V1_SAFE_PATH,
    "V3": V3_PATH,
    "V8 FLOOR MASK": V8_MASK_PATH,
    "PROP MASK": PROP_MASK_PATH,
    "OLD FLOOR": OLD_FLOOR_MASK_PATH,
}


print()
print("=" * 100)
print(STAGE_NAME)
print("=" * 100)

print()
print("SCRIPT:", SCRIPT_NAME)
print("OUTPUT:", OUT)
print("NO NEW QWEN GENERATION")

print()

for name, path in required.items():

    ok = path.exists()

    print(
        f"{name:18s}",
        "✅" if ok else "❌",
        path
    )

    if not ok:
        raise FileNotFoundError(path)


# ============================================================
# 7. LOAD ORIGINAL / V1 / V3
# ============================================================

original_pil = Image.open(
    ORIGINAL_PATH
).convert("RGB")

original = np.array(
    original_pil
)

H, W = original.shape[:2]

SIZE = (W, H)

v1 = load_rgb_resized(
    V1_SAFE_PATH,
    SIZE
)

v3 = load_rgb_resized(
    V3_PATH,
    SIZE
)

v8_mask = load_mask(
    V8_MASK_PATH,
    SIZE
)

props = load_mask(
    PROP_MASK_PATH,
    SIZE
)

old_floor = load_mask(
    OLD_FLOOR_MASK_PATH,
    SIZE
)


print()
print(
    "MASTER SIZE:",
    W,
    "x",
    H
)


# ============================================================
# 8. RELAXED GREEN DETECTION
# ============================================================

r = v3[..., 0].astype(
    np.int16
)

g = v3[..., 1].astype(
    np.int16
)

b = v3[..., 2].astype(
    np.int16
)


green_rgb = (
    (g >= MIN_GREEN)
    &
    ((g - r) >= MIN_GREEN_ADVANTAGE_R)
    &
    ((g - b) >= MIN_GREEN_ADVANTAGE_B)
)


v3_bgr = cv2.cvtColor(
    v3,
    cv2.COLOR_RGB2BGR
)

hsv = cv2.cvtColor(
    v3_bgr,
    cv2.COLOR_BGR2HSV
)

hue = hsv[..., 0]

sat = hsv[..., 1]


green_hsv = (
    (hue >= HUE_MIN)
    &
    (hue <= HUE_MAX)
    &
    (sat >= MIN_SATURATION)
)


raw_green_relaxed = (
    green_rgb
    &
    green_hsv
)


print()
print(
    "RAW RELAXED GREEN PIXELS:",
    int(
        raw_green_relaxed.sum()
    )
)


# ============================================================
# 9. REMOVE PROPS
# ============================================================

green_nonprop = (
    raw_green_relaxed
    &
    ~props
)


# ============================================================
# 10. GEOMETRIC LOWER-IMAGE SAFETY GATE
# ============================================================
#
# A real floor region in this room should occupy the lower image.
#
# V3 floor begins in lower-middle area.
# We allow pixels from 48% image height downward.
#
# This prevents green detection from rescuing upper walls.
# ============================================================

LOWER_START = int(
    H * 0.48
)


lower_gate = np.zeros(
    (H, W),
    dtype=bool
)

lower_gate[
    LOWER_START:,
    :
] = True


green_lower = (
    green_nonprop
    &
    lower_gate
)


# ============================================================
# 11. COMBINE WITH V8
# ============================================================
#
# V8 is already trusted for bright floor.
# V9 expands V8 only with relaxed green candidates.
# ============================================================

candidate = (
    v8_mask
    |
    green_lower
)


# ============================================================
# 12. MORPHOLOGICAL CLOSING
# ============================================================
#
# Larger closing helps recover shadowed gaps under vanity/toilet.
# ============================================================

candidate_u8 = (
    candidate.astype(np.uint8)
    * 255
)


kernel_close = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (9, 9)
)


closed = cv2.morphologyEx(
    candidate_u8,
    cv2.MORPH_CLOSE,
    kernel_close,
    iterations=2
) > 0


# ============================================================
# 13. CONNECTED COMPONENT FILTER
# ============================================================

n, labels, stats, centroids = (
    cv2.connectedComponentsWithStats(
        closed.astype(np.uint8),
        connectivity=8
    )
)


floor_connected = np.zeros(
    (H, W),
    dtype=bool
)


BOTTOM_START = int(
    H * 0.94
)


component_info = []


for i in range(
    1,
    n
):

    component = (
        labels == i
    )

    area = int(
        stats[
            i,
            cv2.CC_STAT_AREA
        ]
    )

    x = int(
        stats[
            i,
            cv2.CC_STAT_LEFT
        ]
    )

    y = int(
        stats[
            i,
            cv2.CC_STAT_TOP
        ]
    )

    w = int(
        stats[
            i,
            cv2.CC_STAT_WIDTH
        ]
    )

    h = int(
        stats[
            i,
            cv2.CC_STAT_HEIGHT
        ]
    )


    cy = float(
        centroids[
            i,
            1
        ]
    )


    touches_bottom = bool(
        np.any(
            component[
                BOTTOM_START:,
                :
            ]
        )
    )


    large_lower = (
        area >= 1200
        and
        cy > H * 0.58
    )


    overlaps_v8 = bool(
        np.any(
            component
            &
            v8_mask
        )
    )


    keep = (
        overlaps_v8
        and
        (
            touches_bottom
            or
            large_lower
        )
    )


    component_info.append(
        {
            "id":
                i,

            "area":
                area,

            "bbox":
                [
                    x,
                    y,
                    w,
                    h
                ],

            "centroid_y":
                cy,

            "touches_bottom":
                touches_bottom,

            "large_lower":
                large_lower,

            "overlaps_v8":
                overlaps_v8,

            "kept":
                keep,
        }
    )


    if keep:

        floor_connected |= (
            component
        )


# ============================================================
# 14. HOLE FILL
# ============================================================
#
# Fill enclosed holes inside the connected floor shape.
# ============================================================

floor_u8 = (
    floor_connected.astype(
        np.uint8
    )
    * 255
)


flood = (
    floor_u8.copy()
)


mask_ff = np.zeros(
    (
        H + 2,
        W + 2
    ),
    np.uint8
)


cv2.floodFill(
    flood,
    mask_ff,
    (0, 0),
    255
)


flood_inv = cv2.bitwise_not(
    flood
)


filled = (
    floor_u8
    |
    flood_inv
) > 0


# ============================================================
# 15. FINAL FLOOR MASK
# ============================================================

floor_final = (
    filled
    &
    lower_gate
)


floor_final &= (
    ~props
)


# Always preserve trusted V8 region
floor_final |= (
    v8_mask
    &
    ~props
)


# ============================================================
# 16. SAVE MASK STAGES
# ============================================================

save_mask(
    raw_green_relaxed,
    OUT
    / "00_v3_relaxed_green_pixels.png"
)

save_mask(
    green_lower,
    OUT
    / "01_v3_relaxed_green_lower.png"
)

save_mask(
    candidate,
    OUT
    / "02_v9_candidate_floor.png"
)

save_mask(
    floor_connected,
    OUT
    / "03_v9_connected_floor.png"
)

save_mask(
    floor_final,
    OUT
    / "04_v9_final_floor_mask.png"
)

save_mask(
    v8_mask,
    OUT
    / "05_v8_floor_reference.png"
)


# ============================================================
# 17. V8 VS V9 AUDIT
# ============================================================

v9_added = (
    floor_final
    &
    ~v8_mask
)

v8_lost = (
    v8_mask
    &
    ~floor_final
)


print()
print("MASK AUDIT")
print("-" * 100)

print(
    "V8 pixels:",
    int(
        v8_mask.sum()
    )
)

print(
    "V9 pixels:",
    int(
        floor_final.sum()
    )
)

print(
    "V9 added:",
    int(
        v9_added.sum()
    )
)

print(
    "V8 lost:",
    int(
        v8_lost.sum()
    )
)


# ============================================================
# 18. AUDIT OVERLAY
# ============================================================
#
# GREEN   = final V9 floor
# CYAN    = newly recovered vs V8
# MAGENTA = protected props
# ============================================================

overlay = (
    original.copy()
    .astype(
        np.float32
    )
)


def tint(
    image,
    mask,
    color,
    alpha
):

    out = image.copy()

    c = np.array(
        color,
        dtype=np.float32
    )

    out[
        mask
    ] = (
        out[
            mask
        ]
        *
        (
            1.0
            -
            alpha
        )
        +
        c
        *
        alpha
    )

    return out


overlay = tint(
    overlay,
    floor_final,
    [0, 200, 0],
    0.50
)

overlay = tint(
    overlay,
    v9_added,
    [0, 255, 255],
    0.65
)

overlay = tint(
    overlay,
    props,
    [255, 0, 255],
    0.45
)


overlay = np.clip(
    overlay,
    0,
    255
).astype(
    np.uint8
)


overlay_path = (
    OUT
    / "06_v9_floor_recovery_overlay.png"
)


Image.fromarray(
    overlay
).save(
    overlay_path
)


# ============================================================
# 19. CREATE V9 FROM V1
# ============================================================

v9 = (
    v1.copy()
)


v9[
    floor_final
] = FLOOR_RGB


v9[
    props
] = original[
    props
]


v9_pil = (
    Image.fromarray(
        v9
    )
)


v9_path = (
    OUT
    / "07_v9_final_surface_template.png"
)


v9_pil.save(
    v9_path
)


# ============================================================
# 20. EXACT RGB VERIFICATION
# ============================================================

floor_pixels = (
    v9[
        floor_final
    ]
)


if floor_pixels.size:

    floor_exact_ratio = float(
        np.mean(
            np.all(
                floor_pixels
                ==
                FLOOR_RGB,
                axis=1
            )
        )
    )

else:

    floor_exact_ratio = 1.0


prop_diff = np.abs(

    v9.astype(
        np.int16
    )

    -

    original.astype(
        np.int16
    )

)[
    props
]


max_prop_rgb_error = (

    int(
        prop_diff.max()
    )

    if prop_diff.size

    else 0
)


must_match_v1 = (
    ~floor_final
    &
    ~props
)


nonfloor_diff = np.abs(

    v9.astype(
        np.int16
    )

    -

    v1.astype(
        np.int16
    )

)[
    must_match_v1
]


max_nonfloor_error = (

    int(
        nonfloor_diff.max()
    )

    if nonfloor_diff.size

    else 0
)


print()
print("V9 VERIFICATION")
print("-" * 100)

print(
    "Floor exact #00C800 %:",
    round(
        floor_exact_ratio
        * 100,
        6
    )
)

print(
    "Max prop RGB error:",
    max_prop_rgb_error
)

print(
    "Max non-floor change vs V1:",
    max_nonfloor_error
)


# ============================================================
# 21. COMPARISON: V1 / V3 / V8 / V9
# ============================================================

LABEL_H = 40


comparison = Image.new(
    "RGB",
    (
        W * 4,
        H + LABEL_H
    ),
    (
        255,
        255,
        255
    )
)


comparison.paste(
    Image.fromarray(
        v1
    ),
    (
        0,
        LABEL_H
    )
)


comparison.paste(
    Image.fromarray(
        v3
    ),
    (
        W,
        LABEL_H
    )
)


v8_result_path = (
    ROOT
    / "01z8_v1_plus_v3_floor_reference_v8"
    / "07_v8_final_surface_template.png"
)


v8_result = load_rgb_resized(
    v8_result_path,
    SIZE
)


comparison.paste(
    Image.fromarray(
        v8_result
    ),
    (
        W * 2,
        LABEL_H
    )
)


comparison.paste(
    v9_pil,
    (
        W * 3,
        LABEL_H
    )
)


draw = ImageDraw.Draw(
    comparison
)


labels = [

    (
        8,
        "V1"
    ),

    (
        W + 8,
        "V3 FLOOR REFERENCE"
    ),

    (
        W * 2 + 8,
        "V8"
    ),

    (
        W * 3 + 8,
        "V9 RELAXED FLOOR"
    ),
]


for x, text in labels:

    draw.text(
        (
            x,
            10
        ),
        text,
        fill=(
            0,
            0,
            0
        )
    )


comparison_path = (
    OUT
    / "08_v1_v3_v8_v9_comparison.png"
)


comparison.save(
    comparison_path
)


# ============================================================
# 22. V8 VS V9
# ============================================================

compare2 = Image.new(
    "RGB",
    (
        W * 2,
        H + LABEL_H
    ),
    (
        255,
        255,
        255
    )
)


compare2.paste(
    Image.fromarray(
        v8_result
    ),
    (
        0,
        LABEL_H
    )
)


compare2.paste(
    v9_pil,
    (
        W,
        LABEL_H
    )
)


draw2 = ImageDraw.Draw(
    compare2
)


draw2.text(
    (
        8,
        10
    ),
    "V8",
    fill=(
        0,
        0,
        0
    )
)


draw2.text(
    (
        W + 8,
        10
    ),
    "V9",
    fill=(
        0,
        0,
        0
    )
)


compare2_path = (
    OUT
    / "09_v8_vs_v9.png"
)


compare2.save(
    compare2_path
)


# ============================================================
# 23. REPORT
# ============================================================

report = {

    "stage":
        STAGE_NAME,

    "status":
        "COMPLETED",

    "script":
        SCRIPT_NAME,

    "new_ai_generation":
        False,

    "strategy":
        (
            "Expand V8 floor mask by detecting darker/shadowed "
            "green regions in V3 while requiring lower-region "
            "geometry and connectivity to trusted V8 floor."
        ),

    "thresholds": {

        "min_green":
            MIN_GREEN,

        "min_green_advantage_r":
            MIN_GREEN_ADVANTAGE_R,

        "min_green_advantage_b":
            MIN_GREEN_ADVANTAGE_B,

        "min_saturation":
            MIN_SATURATION,

        "hue_min":
            HUE_MIN,

        "hue_max":
            HUE_MAX,

        "lower_start_fraction":
            0.48,
    },

    "counts": {

        "raw_relaxed_green":
            int(
                raw_green_relaxed.sum()
            ),

        "green_lower":
            int(
                green_lower.sum()
            ),

        "candidate":
            int(
                candidate.sum()
            ),

        "connected_floor":
            int(
                floor_connected.sum()
            ),

        "v8_floor":
            int(
                v8_mask.sum()
            ),

        "v9_final":
            int(
                floor_final.sum()
            ),

        "v9_added_vs_v8":
            int(
                v9_added.sum()
            ),

        "v8_lost":
            int(
                v8_lost.sum()
            ),
    },

    "component_audit":
        component_info,

    "verification": {

        "floor_exact_color_ratio":
            floor_exact_ratio,

        "max_prop_rgb_error":
            max_prop_rgb_error,

        "max_nonfloor_error_vs_v1":
            max_nonfloor_error,
    },

    "sources": {

        "original":
            str(
                ORIGINAL_PATH
            ),

        "v1":
            str(
                V1_SAFE_PATH
            ),

        "v3":
            str(
                V3_PATH
            ),

        "v8_floor_mask":
            str(
                V8_MASK_PATH
            ),

        "props":
            str(
                PROP_MASK_PATH
            ),
    },

    "hashes": {

        "original":
            sha256_file(
                ORIGINAL_PATH
            ),

        "v1":
            sha256_file(
                V1_SAFE_PATH
            ),

        "v3":
            sha256_file(
                V3_PATH
            ),

        "v8_floor":
            sha256_file(
                V8_MASK_PATH
            ),
    },

    "outputs": {

        "raw_relaxed_green":
            str(
                OUT
                / "00_v3_relaxed_green_pixels.png"
            ),

        "green_lower":
            str(
                OUT
                / "01_v3_relaxed_green_lower.png"
            ),

        "candidate":
            str(
                OUT
                / "02_v9_candidate_floor.png"
            ),

        "connected":
            str(
                OUT
                / "03_v9_connected_floor.png"
            ),

        "final_floor":
            str(
                OUT
                / "04_v9_final_floor_mask.png"
            ),

        "audit_overlay":
            str(
                overlay_path
            ),

        "v9":
            str(
                v9_path
            ),

        "comparison":
            str(
                comparison_path
            ),

        "v8_vs_v9":
            str(
                compare2_path
            ),
    },

    "benchmark": {

        "V1":
            "GOOD RECONSTRUCTION BASE",

        "V3":
            "GOOD GENERATIVE FLOOR REFERENCE",

        "V8":
            "GOOD FLOOR SHAPE BUT SHADOWED FLOOR PARTLY MISSED",

        "V9":
            "RELAXED SHADOWED-FLOOR RECOVERY",
    },
}


report_path = (
    OUT
    / "00_stage01z9_report.json"
)


report_path.write_text(
    json.dumps(
        report,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# ============================================================
# 24. FINAL
# ============================================================

print()
print("=" * 100)
print("STAGE01Z9 COMPLETE")
print("=" * 100)

print()
print(
    "NO NEW QWEN GENERATION"
)

print(
    "V9 RESULT:",
    v9_path
)

print(
    "AUDIT OVERLAY:",
    overlay_path
)

print(
    "COMPARISON:",
    comparison_path
)

print(
    "REPORT:",
    report_path
)


# ============================================================
# 25. DISPLAY
# ============================================================

print()
print(
    "V9 FLOOR RECOVERY OVERLAY"
)

display(
    Image.open(
        overlay_path
    )
)

print()
print(
    "V1 / V3 / V8 / V9"
)

display(
    comparison
)

print()
print(
    "V8 VS V9"
)

display(
    compare2
)
