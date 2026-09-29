
# ============================================================
# TEST07 - ROOM03
# STAGE01Z8
#
# V1 + V3 FLOOR REFERENCE
#
# V1:
#   best coherent clean-room reconstruction
#
# V3:
#   used ONLY to infer semantic floor region
#
# FINAL:
#   V1 appearance
#   + exact #00C800 on V3-derived floor
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
    "TEST07_STAGE01Z8_"
    "V1_PLUS_V3_FLOOR_REFERENCE_V8"
)

SCRIPT_NAME = (
    "test07_stage01z8_"
    "v1_plus_v3_floor_reference_v8.py"
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
    / "01z8_v1_plus_v3_floor_reference_v8"
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


# V3 RAW MASTER-SIZE IMAGE:
# use Qwen's semantic green-floor result BEFORE exact prop
# restoration as the semantic floor reference.
V3_PATH = (
    ROOT
    / "01z3_qwen_short_surface_template_v3"
    / "05_qwen_v3_master_size.png"
)


PROP_MASK_PATH = (
    ROOT
    / "01y3_frozen_canonical_prop_protection"
    / "00_canonical_prop_protection_mask.png"
)


# Original Stage01 floor retained only for comparison,
# NOT used to build the V8 mask.
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
# 4. V3 GREEN-DETECTION SETTINGS
# ============================================================
#
# V3 floor was clearly bright/saturated green.
#
# We deliberately use broad semantic-green criteria instead
# of requiring exact RGB because Qwen output includes shading.
# ============================================================

MIN_GREEN = 100

MIN_GREEN_ADVANTAGE_R = 25

MIN_GREEN_ADVANTAGE_B = 25

MIN_SATURATION = 70


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
# 6. INPUT VALIDATION
# ============================================================

required = {

    "ORIGINAL":
        ORIGINAL_PATH,

    "V1 SAFE":
        V1_SAFE_PATH,

    "V3 FLOOR REFERENCE":
        V3_PATH,

    "FROZEN PROP MASK":
        PROP_MASK_PATH,

    "OLD FLOOR MASK":
        OLD_FLOOR_MASK_PATH,
}


print()
print("=" * 100)
print(STAGE_NAME)
print("=" * 100)

print()
print("SCRIPT:", SCRIPT_NAME)
print("OUTPUT:", OUT)

print()
print("NO NEW QWEN GENERATION")


for name, path in required.items():

    ok = path.exists()

    print(
        f"{name:20s}",
        "✅" if ok else "❌",
        path
    )

    if not ok:

        raise FileNotFoundError(
            path
        )


# ============================================================
# 7. LOAD ORIGINAL
# ============================================================

original_pil = Image.open(
    ORIGINAL_PATH
).convert("RGB")

original = np.array(
    original_pil
)

H, W = original.shape[:2]

SIZE = (
    W,
    H
)

print()
print(
    "MASTER SIZE:",
    W,
    "x",
    H
)


# ============================================================
# 8. LOAD V1 / V3 / MASKS
# ============================================================

v1 = load_rgb_resized(
    V1_SAFE_PATH,
    SIZE
)

v3 = load_rgb_resized(
    V3_PATH,
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


# ============================================================
# 9. RAW GREEN DETECTION FROM V3
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


# RGB dominance test
green_rgb = (

    (g >= MIN_GREEN)

    &

    (
        g - r
        >= MIN_GREEN_ADVANTAGE_R
    )

    &

    (
        g - b
        >= MIN_GREEN_ADVANTAGE_B
    )
)


# HSV saturation test
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


# OpenCV green hue roughly 35–95
green_hsv = (

    (hue >= 35)

    &

    (hue <= 95)

    &

    (sat >= MIN_SATURATION)
)


raw_green = (
    green_rgb
    &
    green_hsv
)


print()
print("RAW GREEN PIXELS:", int(raw_green.sum()))


# ============================================================
# 10. REMOVE FROZEN PROPS BEFORE CONNECTIVITY
# ============================================================

green_nonprop = (
    raw_green
    &
    ~props
)


# ============================================================
# 11. MORPHOLOGICAL CLEANUP
# ============================================================
#
# Close small gaps in the generated green floor.
# ============================================================

binary = (
    green_nonprop.astype(np.uint8)
    * 255
)

kernel_close = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (7, 7)
)

closed = cv2.morphologyEx(
    binary,
    cv2.MORPH_CLOSE,
    kernel_close,
    iterations=2
)


kernel_open = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (3, 3)
)

cleaned = cv2.morphologyEx(
    closed,
    cv2.MORPH_OPEN,
    kernel_open,
    iterations=1
) > 0


# ============================================================
# 12. CONNECTED-COMPONENT FLOOR SELECTION
# ============================================================
#
# Real floor should connect to / approach the bottom image edge.
#
# We select components that:
# - touch bottom band, OR
# - are very large and located mainly in lower image.
# ============================================================

n, labels, stats, centroids = (
    cv2.connectedComponentsWithStats(
        cleaned.astype(np.uint8),
        connectivity=8
    )
)

floor_connected = np.zeros(
    (H, W),
    dtype=bool
)

BOTTOM_BAND = max(
    8,
    int(
        H * 0.05
    )
)

bottom_start = (
    H
    -
    BOTTOM_BAND
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

    touches_bottom = bool(
        np.any(
            component[
                bottom_start:,
                :
            ]
        )
    )

    cy = float(
        centroids[
            i,
            1
        ]
    )

    large_lower_region = (

        area >= 1500

        and

        cy > H * 0.55
    )


    keep = (
        touches_bottom
        or
        large_lower_region
    )


    component_info.append(
        {
            "id": i,
            "area": area,
            "bbox": [
                x,
                y,
                w,
                h
            ],
            "centroid_y": cy,
            "touches_bottom": touches_bottom,
            "large_lower_region": large_lower_region,
            "kept": keep,
        }
    )


    if keep:

        floor_connected |= component


# ============================================================
# 13. FINAL SMALL HOLE FILL
# ============================================================

floor_u8 = (
    floor_connected.astype(np.uint8)
    * 255
)


kernel_final = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (5, 5)
)


floor_final = cv2.morphologyEx(
    floor_u8,
    cv2.MORPH_CLOSE,
    kernel_final,
    iterations=1
) > 0


# Props always excluded
floor_final &= ~props


# ============================================================
# 14. SANITY CONSTRAINT
# ============================================================
#
# We DO NOT force the old floor mask.
#
# But to protect against strange upper-image green hallucination,
# reject floor pixels far above the upper extent indicated by
# BOTH V3 semantic floor and the old floor.
#
# This is only a broad safety gate.
# ============================================================

candidate_rows = np.where(
    floor_final
)[0]


if candidate_rows.size:

    semantic_top = int(
        candidate_rows.min()
    )

else:

    semantic_top = H


old_rows = np.where(
    old_floor
)[0]


if old_rows.size:

    old_top = int(
        old_rows.min()
    )

else:

    old_top = H


safe_top = max(
    0,
    min(
        semantic_top,
        old_top
    )
    - 12
)


row_gate = np.zeros(
    (H, W),
    dtype=bool
)

row_gate[
    safe_top:,
    :
] = True


floor_final &= row_gate


# ============================================================
# 15. SAVE FLOOR MASK STAGES
# ============================================================

save_mask(
    raw_green,
    OUT
    / "00_v3_raw_green_pixels.png"
)

save_mask(
    cleaned,
    OUT
    / "01_v3_green_cleaned.png"
)

save_mask(
    floor_connected,
    OUT
    / "02_v3_floor_connected_components.png"
)

save_mask(
    floor_final,
    OUT
    / "03_v8_final_floor_mask.png"
)

save_mask(
    old_floor,
    OUT
    / "04_old_stage01_floor_mask_reference.png"
)

save_mask(
    props,
    OUT
    / "05_frozen_prop_mask.png"
)


# ============================================================
# 16. MASK COMPARISON AUDIT
# ============================================================

intersection = (
    floor_final
    &
    old_floor
)

v8_only = (
    floor_final
    &
    ~old_floor
)

old_only = (
    old_floor
    &
    ~floor_final
)


print()
print("FLOOR MASK AUDIT")
print("-" * 100)

print(
    "Old Stage01 floor:",
    int(
        old_floor.sum()
    )
)

print(
    "V8 floor:",
    int(
        floor_final.sum()
    )
)

print(
    "Intersection:",
    int(
        intersection.sum()
    )
)

print(
    "V8 only:",
    int(
        v8_only.sum()
    )
)

print(
    "Old only:",
    int(
        old_only.sum()
    )
)


# ============================================================
# 17. CREATE FLOOR-MASK DIAGNOSTIC OVERLAY
# ============================================================
#
# GREEN   = V8 floor
# RED     = old floor missed by V8
# CYAN    = V8 added beyond old floor
# MAGENTA = protected props
# ============================================================

overlay = (
    original.copy()
    .astype(
        np.float32
    )
)


def tint(
    img,
    mask,
    color,
    alpha
):

    result = img.copy()

    c = np.array(
        color,
        dtype=np.float32
    )

    result[
        mask
    ] = (

        result[
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

    return result


overlay = tint(
    overlay,
    floor_final,
    [0, 200, 0],
    0.55
)

overlay = tint(
    overlay,
    v8_only,
    [0, 255, 255],
    0.55
)

overlay = tint(
    overlay,
    old_only,
    [255, 0, 0],
    0.35
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
    / "06_v8_floor_mask_audit_overlay.png"
)

Image.fromarray(
    overlay
).save(
    overlay_path
)


# ============================================================
# 18. CREATE V8 FROM V1
# ============================================================

v8 = (
    v1.copy()
)


v8[
    floor_final
] = FLOOR_RGB


# Restore frozen props exactly
v8[
    props
] = original[
    props
]


v8_pil = Image.fromarray(
    v8
)


v8_path = (
    OUT
    / "07_v8_final_surface_template.png"
)


v8_pil.save(
    v8_path
)


# ============================================================
# 19. EXACT COLOR VERIFICATION
# ============================================================

floor_pixels = (
    v8[
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


# ============================================================
# 20. PROP VERIFICATION
# ============================================================

prop_diff = np.abs(

    v8.astype(
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


# ============================================================
# 21. NON-FLOOR V1 PRESERVATION
# ============================================================

must_match_v1 = (

    ~floor_final

    &

    ~props
)


nonfloor_diff = np.abs(

    v8.astype(
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
print("V8 VERIFICATION")
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
# 22. COMPARISON: ORIGINAL / V1 / V3 / V8
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
    original_pil,
    (
        0,
        LABEL_H
    )
)


comparison.paste(
    Image.fromarray(
        v1
    ),
    (
        W,
        LABEL_H
    )
)


comparison.paste(
    Image.fromarray(
        v3
    ),
    (
        W * 2,
        LABEL_H
    )
)


comparison.paste(
    v8_pil,
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
        "ORIGINAL"
    ),

    (
        W + 8,
        "V1 GOOD REFERENCE"
    ),

    (
        W * 2 + 8,
        "V3 FLOOR REFERENCE"
    ),

    (
        W * 3 + 8,
        "V8 = V1 + V3 FLOOR MASK"
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
    / "08_original_v1_v3_v8_comparison.png"
)


comparison.save(
    comparison_path
)


# ============================================================
# 23. V1 VS V8
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
        v1
    ),
    (
        0,
        LABEL_H
    )
)


compare2.paste(
    v8_pil,
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
    "V1",
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
    "V8",
    fill=(
        0,
        0,
        0
    )
)


compare2_path = (
    OUT
    / "09_v1_vs_v8.png"
)


compare2.save(
    compare2_path
)


# ============================================================
# 24. REPORT
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
            "Use V1 as final clean-room appearance and use V3 "
            "only as semantic floor-reference image."
        ),

    "green_detection": {

        "min_green":
            MIN_GREEN,

        "min_green_advantage_r":
            MIN_GREEN_ADVANTAGE_R,

        "min_green_advantage_b":
            MIN_GREEN_ADVANTAGE_B,

        "min_saturation":
            MIN_SATURATION,

        "hsv_green_hue_range":
            [
                35,
                95
            ],
    },

    "mask_counts": {

        "raw_green":
            int(
                raw_green.sum()
            ),

        "cleaned_green":
            int(
                cleaned.sum()
            ),

        "floor_connected":
            int(
                floor_connected.sum()
            ),

        "final_v8_floor":
            int(
                floor_final.sum()
            ),

        "old_stage01_floor":
            int(
                old_floor.sum()
            ),

        "intersection":
            int(
                intersection.sum()
            ),

        "v8_only":
            int(
                v8_only.sum()
            ),

        "old_only":
            int(
                old_only.sum()
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

        "v1_safe":
            str(
                V1_SAFE_PATH
            ),

        "v3_reference":
            str(
                V3_PATH
            ),

        "props":
            str(
                PROP_MASK_PATH
            ),

        "old_floor_reference":
            str(
                OLD_FLOOR_MASK_PATH
            ),
    },

    "source_hashes": {

        "original":
            sha256_file(
                ORIGINAL_PATH
            ),

        "v1_safe":
            sha256_file(
                V1_SAFE_PATH
            ),

        "v3_reference":
            sha256_file(
                V3_PATH
            ),
    },

    "outputs": {

        "raw_green":
            str(
                OUT
                / "00_v3_raw_green_pixels.png"
            ),

        "cleaned_green":
            str(
                OUT
                / "01_v3_green_cleaned.png"
            ),

        "connected_floor":
            str(
                OUT
                / "02_v3_floor_connected_components.png"
            ),

        "final_floor_mask":
            str(
                OUT
                / "03_v8_final_floor_mask.png"
            ),

        "audit_overlay":
            str(
                overlay_path
            ),

        "v8":
            str(
                v8_path
            ),

        "comparison":
            str(
                comparison_path
            ),

        "v1_vs_v8":
            str(
                compare2_path
            ),
    },

    "benchmark": {

        "V1":
            "GOOD RECONSTRUCTION BASE",

        "V3":
            "GOOD SEMANTIC FLOOR REFERENCE",

        "V7":
            "FAILED DUE TO OLD FLOOR MASK",

        "V8":
            "V1 + V3-DERIVED FLOOR MASK",
    },
}


report_path = (
    OUT
    / "00_stage01z8_report.json"
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
# 25. FINAL
# ============================================================

print()
print("=" * 100)
print("STAGE01Z8 COMPLETE")
print("=" * 100)

print()
print(
    "NO NEW QWEN GENERATION"
)

print()
print(
    "FINAL FLOOR MASK:",
    OUT
    / "03_v8_final_floor_mask.png"
)

print(
    "V8 RESULT:",
    v8_path
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
# 26. DISPLAY
# ============================================================

print()
print(
    "V8 FLOOR MASK AUDIT"
)

display(
    Image.open(
        overlay_path
    )
)

print()
print(
    "ORIGINAL / V1 / V3 / V8"
)

display(
    comparison
)

print()
print(
    "V1 VS V8"
)

display(
    compare2
)
