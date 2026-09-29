
# ============================================================
# TEST07 - ROOM03
# STAGE01Z6
#
# WALL MASK REFINEMENT V6
#
# METHOD:
# base semantic wall
# + mirror -> wall
# + glass-background -> wall
# + conservative morphology / connected cleanup
# - frozen props
# - floor
# - ceiling
#
# NO GENERATIVE MODEL USED.
# ============================================================

from pathlib import Path
from PIL import Image, ImageDraw
from IPython.display import display

import json
import hashlib
import numpy as np
import cv2


# ============================================================
# 1. PATHS
# ============================================================

STAGE_NAME = "TEST07_STAGE01Z6_WALL_MASK_REFINEMENT_V6"
SCRIPT_NAME = "test07_stage01z6_wall_mask_refinement_v6.py"

BASE = Path("/workspace/axolotl")

ROOT = (
    BASE
    / "test07"
    / "runs"
    / "room03_bathroom"
    / "stages"
)

OUT = (
    ROOT
    / "01z6_wall_mask_refinement_v6"
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

BASE_WALL_PATH = (
    ROOT
    / "01_structure_inverse_base"
    / "02_wall_mask.png"
)

CEILING_PATH = (
    ROOT
    / "01_structure_inverse_base"
    / "03_ceiling_mask.png"
)

FLOOR_PATH = (
    ROOT
    / "01_structure_inverse_base"
    / "04_floor_mask.png"
)

MIRROR_PATH = (
    ROOT
    / "01f_mirror_shape_candidate"
    / "00_candidate02_mirror_shape_mask.png"
)

GLASS_TO_WALL_PATH = (
    ROOT
    / "01w2_v3_glass_background_classification"
    / "00_glass_to_wall.png"
)

PROP_PATH = (
    ROOT
    / "01y3_frozen_canonical_prop_protection"
    / "00_canonical_prop_protection_mask.png"
)


# ============================================================
# 2. HELPERS
# ============================================================

def load_rgb(path):
    return np.array(
        Image.open(path).convert("RGB")
    )


def load_mask(path, size=None):
    img = Image.open(path).convert("L")

    if size is not None and img.size != size:
        img = img.resize(
            size,
            Image.Resampling.NEAREST
        )

    return np.array(img) > 0


def save_mask(mask, path):
    Image.fromarray(
        (mask.astype(np.uint8) * 255),
        mode="L"
    ).save(path)


def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def remove_small_components(mask, min_area=40):
    binary = (
        mask.astype(np.uint8) * 255
    )

    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8
    )

    out = np.zeros_like(mask, dtype=bool)

    for i in range(1, n):
        area = stats[
            i,
            cv2.CC_STAT_AREA
        ]

        if area >= min_area:
            out[labels == i] = True

    return out


# ============================================================
# 3. VALIDATION
# ============================================================

required = {
    "ORIGINAL": ORIGINAL_PATH,
    "BASE WALL": BASE_WALL_PATH,
    "CEILING": CEILING_PATH,
    "FLOOR": FLOOR_PATH,
    "MIRROR": MIRROR_PATH,
    "GLASS->WALL": GLASS_TO_WALL_PATH,
    "PROPS": PROP_PATH,
}

print()
print("=" * 100)
print(STAGE_NAME)
print("=" * 100)

for name, path in required.items():
    ok = path.exists()

    print(
        f"{name:16s}",
        "✅" if ok else "❌",
        path
    )

    if not ok:
        raise FileNotFoundError(path)


# ============================================================
# 4. LOAD
# ============================================================

original = load_rgb(
    ORIGINAL_PATH
)

H, W = original.shape[:2]
SIZE = (W, H)

wall = load_mask(
    BASE_WALL_PATH,
    SIZE
)

ceiling = load_mask(
    CEILING_PATH,
    SIZE
)

floor = load_mask(
    FLOOR_PATH,
    SIZE
)

mirror = load_mask(
    MIRROR_PATH,
    SIZE
)

glass_wall = load_mask(
    GLASS_TO_WALL_PATH,
    SIZE
)

props = load_mask(
    PROP_PATH,
    SIZE
)


print()
print("MASTER SIZE:", W, "x", H)

print()
print("INPUT COUNTS")
print("-" * 80)

print("base wall   :", int(wall.sum()))
print("mirror      :", int(mirror.sum()))
print("glass->wall :", int(glass_wall.sum()))
print("ceiling     :", int(ceiling.sum()))
print("floor       :", int(floor.sum()))
print("props       :", int(props.sum()))


# ============================================================
# 5. INITIAL WALL AUGMENTATION
# ============================================================

wall_aug = (
    wall
    |
    mirror
    |
    glass_wall
)


# ============================================================
# 6. REMOVE KNOWN NON-WALL CLASSES
# ============================================================

wall_aug &= ~ceiling
wall_aug &= ~floor
wall_aug &= ~props


# ============================================================
# 7. CONSERVATIVE MORPHOLOGY
# ============================================================
#
# Small closing:
# fills tiny holes / seams without aggressively expanding.
# ============================================================

kernel_close = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (5, 5)
)

closed = cv2.morphologyEx(
    (
        wall_aug.astype(np.uint8)
        * 255
    ),
    cv2.MORPH_CLOSE,
    kernel_close,
    iterations=1
) > 0


# ============================================================
# 8. VERY SMALL DILATION TO CONNECT WALL FRAGMENTS
# ============================================================
#
# Only 1 iteration and 3x3 kernel.
# After dilation we again subtract floor/ceiling/props.
# ============================================================

kernel_dilate = cv2.getStructuringElement(
    cv2.MORPH_RECT,
    (3, 3)
)

dilated = cv2.dilate(
    (
        closed.astype(np.uint8)
        * 255
    ),
    kernel_dilate,
    iterations=1
) > 0


# ============================================================
# 9. HARD EXCLUSIONS AGAIN
# ============================================================

refined = dilated.copy()

refined &= ~ceiling
refined &= ~floor
refined &= ~props


# ============================================================
# 10. REMOVE TINY DISCONNECTED NOISE
# ============================================================

refined = remove_small_components(
    refined,
    min_area=40
)


# ============================================================
# 11. FINAL HARD EXCLUSIONS
# ============================================================

refined &= ~ceiling
refined &= ~floor
refined &= ~props


# ============================================================
# 12. CHANGE AUDIT
# ============================================================

added = (
    refined
    &
    ~wall
)

removed = (
    wall
    &
    ~refined
)

print()
print("REFINEMENT COUNTS")
print("-" * 80)

print(
    "refined wall:",
    int(refined.sum())
)

print(
    "added vs base:",
    int(added.sum())
)

print(
    "removed vs base:",
    int(removed.sum())
)


# ============================================================
# 13. SAVE MASKS
# ============================================================

save_mask(
    wall,
    OUT / "00_base_wall_mask.png"
)

save_mask(
    refined,
    OUT / "01_refined_wall_mask.png"
)

save_mask(
    added,
    OUT / "02_added_wall_pixels.png"
)

save_mask(
    removed,
    OUT / "03_removed_wall_pixels.png"
)

save_mask(
    glass_wall,
    OUT / "04_glass_to_wall_source.png"
)

save_mask(
    mirror,
    OUT / "05_mirror_to_wall_source.png"
)


# ============================================================
# 14. OVERLAY - BASE WALL
# ============================================================

base_overlay = original.copy().astype(
    np.float32
)

blue = np.array(
    [0, 102, 255],
    dtype=np.float32
)

base_overlay[
    wall
] = (
    base_overlay[
        wall
    ] * 0.45
    +
    blue * 0.55
)

base_overlay = np.clip(
    base_overlay,
    0,
    255
).astype(np.uint8)

Image.fromarray(
    base_overlay
).save(
    OUT
    / "06_base_wall_overlay.png"
)


# ============================================================
# 15. OVERLAY - REFINED WALL
# ============================================================

refined_overlay = original.copy().astype(
    np.float32
)

refined_overlay[
    refined
] = (
    refined_overlay[
        refined
    ] * 0.35
    +
    blue * 0.65
)

refined_overlay = np.clip(
    refined_overlay,
    0,
    255
).astype(np.uint8)

Image.fromarray(
    refined_overlay
).save(
    OUT
    / "07_refined_wall_overlay.png"
)


# ============================================================
# 16. CHANGE OVERLAY
# ============================================================
#
# BLUE   = final wall
# GREEN  = newly added wall pixels
# RED    = removed old wall pixels
# MAGENTA = protected props
# ============================================================

change = original.copy().astype(
    np.float32
)

change[
    refined
] = (
    change[
        refined
    ] * 0.55
    +
    np.array(
        [0, 102, 255],
        dtype=np.float32
    ) * 0.45
)

change[
    added
] = (
    change[
        added
    ] * 0.25
    +
    np.array(
        [0, 255, 0],
        dtype=np.float32
    ) * 0.75
)

change[
    removed
] = (
    change[
        removed
    ] * 0.25
    +
    np.array(
        [255, 0, 0],
        dtype=np.float32
    ) * 0.75
)

change[
    props
] = (
    change[
        props
    ] * 0.45
    +
    np.array(
        [255, 0, 255],
        dtype=np.float32
    ) * 0.55
)

change = np.clip(
    change,
    0,
    255
).astype(np.uint8)

change_path = (
    OUT
    / "08_wall_refinement_change_overlay.png"
)

Image.fromarray(
    change
).save(
    change_path
)


# ============================================================
# 17. SIDE-BY-SIDE
# ============================================================

LABEL_H = 38

comparison = Image.new(
    "RGB",
    (
        W * 3,
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
        original
    ),
    (
        0,
        LABEL_H
    )
)

comparison.paste(
    Image.fromarray(
        base_overlay
    ),
    (
        W,
        LABEL_H
    )
)

comparison.paste(
    Image.fromarray(
        refined_overlay
    ),
    (
        W * 2,
        LABEL_H
    )
)

draw = ImageDraw.Draw(
    comparison
)

draw.text(
    (8, 10),
    "ORIGINAL",
    fill=(0, 0, 0)
)

draw.text(
    (W + 8, 10),
    "BASE WALL MASK",
    fill=(0, 0, 0)
)

draw.text(
    (W * 2 + 8, 10),
    "REFINED WALL MASK V6",
    fill=(0, 0, 0)
)

comparison_path = (
    OUT
    / "09_original_base_refined_comparison.png"
)

comparison.save(
    comparison_path
)


# ============================================================
# 18. REPORT
# ============================================================

report = {
    "stage":
        STAGE_NAME,

    "status":
        "COMPLETED",

    "script":
        SCRIPT_NAME,

    "method":
        (
            "base wall + mirror-to-wall + glass-to-wall "
            "+ conservative morphology - floor - ceiling "
            "- frozen props"
        ),

    "inputs": {
        "original":
            str(ORIGINAL_PATH),

        "base_wall":
            str(BASE_WALL_PATH),

        "ceiling":
            str(CEILING_PATH),

        "floor":
            str(FLOOR_PATH),

        "mirror":
            str(MIRROR_PATH),

        "glass_to_wall":
            str(GLASS_TO_WALL_PATH),

        "props":
            str(PROP_PATH),
    },

    "hashes": {
        "original":
            sha256_file(ORIGINAL_PATH),

        "base_wall":
            sha256_file(BASE_WALL_PATH),

        "glass_to_wall":
            sha256_file(GLASS_TO_WALL_PATH),
    },

    "counts": {
        "base_wall":
            int(wall.sum()),

        "mirror":
            int(mirror.sum()),

        "glass_to_wall":
            int(glass_wall.sum()),

        "refined_wall":
            int(refined.sum()),

        "added_vs_base":
            int(added.sum()),

        "removed_vs_base":
            int(removed.sum()),
    },

    "operations": {
        "closing_kernel":
            [5, 5],

        "closing_iterations":
            1,

        "dilation_kernel":
            [3, 3],

        "dilation_iterations":
            1,

        "min_component_area":
            40,
    },

    "outputs": {
        "refined_wall_mask":
            str(
                OUT
                / "01_refined_wall_mask.png"
            ),

        "added_pixels":
            str(
                OUT
                / "02_added_wall_pixels.png"
            ),

        "removed_pixels":
            str(
                OUT
                / "03_removed_wall_pixels.png"
            ),

        "refined_overlay":
            str(
                OUT
                / "07_refined_wall_overlay.png"
            ),

        "change_overlay":
            str(
                change_path
            ),

        "comparison":
            str(
                comparison_path
            ),
    },

    "benchmark_status": {
        "V5":
            "Hybrid architecture valid, wall mask incomplete",

        "V6":
            "Pending visual review",
    },
}

report_path = (
    OUT
    / "00_stage01z6_report.json"
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
# 19. FINAL
# ============================================================

print()
print("=" * 100)
print("STAGE01Z6 COMPLETE")
print("=" * 100)

print()
print(
    "REFINED WALL MASK:"
)

print(
    OUT
    / "01_refined_wall_mask.png"
)

print()
print(
    "CHANGE OVERLAY:"
)

print(
    change_path
)

print()
print(
    "COMPARISON:"
)

print(
    comparison_path
)

print()
print(
    "REPORT:"
)

print(
    report_path
)


# ============================================================
# 20. DISPLAY
# ============================================================

print()
print(
    "WALL REFINEMENT CHANGE OVERLAY"
)

display(
    Image.open(
        change_path
    )
)

print()
print(
    "ORIGINAL / BASE WALL / REFINED WALL"
)

display(
    comparison
)
