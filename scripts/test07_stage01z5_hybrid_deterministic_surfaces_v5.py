
# ============================================================
# TEST07 - ROOM03
# STAGE01Z5
#
# HYBRID QWEN RECONSTRUCTION +
# DETERMINISTIC SURFACE CLASSES
#
# IMPORTANT:
# Qwen does NOT decide final wall/floor/ceiling colors.
#
# Qwen V1 is used only as a coherent reconstructed background
# where mirror / shower-glass removal already worked well.
#
# Surface classes are taken from frozen TEST07 masks.
#
# WALL    -> #0066FF
# CEILING -> #FFFFFF
# FLOOR   -> #00C800
#
# Props are restored EXACTLY from ORIGINAL RGB.
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

from pathlib import Path
from PIL import Image, ImageDraw
from IPython.display import display

import json
import hashlib
import numpy as np


# ============================================================
# 2. IDENTIFIERS / PATHS
# ============================================================

STAGE_NAME = (
    "TEST07_STAGE01Z5_"
    "HYBRID_DETERMINISTIC_SURFACES_V5"
)

SCRIPT_NAME = (
    "test07_stage01z5_"
    "hybrid_deterministic_surfaces_v5.py"
)

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
    / "01z5_hybrid_deterministic_surfaces_v5"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 3. INPUTS
# ============================================================

ORIGINAL_PATH = (
    ROOT
    / "01_structure_inverse_base"
    / "00_input_resized.png"
)

WALL_MASK_PATH = (
    ROOT
    / "01_structure_inverse_base"
    / "02_wall_mask.png"
)

CEILING_MASK_PATH = (
    ROOT
    / "01_structure_inverse_base"
    / "03_ceiling_mask.png"
)

FLOOR_MASK_PATH = (
    ROOT
    / "01_structure_inverse_base"
    / "04_floor_mask.png"
)

STRUCTURE_MASK_PATH = (
    ROOT
    / "01_structure_inverse_base"
    / "05_structure_mask.png"
)

MIRROR_MASK_PATH = (
    ROOT
    / "01f_mirror_shape_candidate"
    / "00_candidate02_mirror_shape_mask.png"
)

GLASS_MASK_PATH = (
    ROOT
    / "01n_canonical_glass_layer"
    / "00_canonical_glass_system_mask.png"
)

GLASS_TO_WALL_PATH = (
    ROOT
    / "01w2_v3_glass_background_classification"
    / "00_glass_to_wall.png"
)

GLASS_TO_FLOOR_PATH = (
    ROOT
    / "01w2_v3_glass_background_classification"
    / "01_glass_to_floor.png"
)

GLASS_TO_CEILING_PATH = (
    ROOT
    / "01w2_v3_glass_background_classification"
    / "02_glass_to_ceiling.png"
)

PROP_MASK_PATH = (
    ROOT
    / "01y3_frozen_canonical_prop_protection"
    / "00_canonical_prop_protection_mask.png"
)


# ============================================================
# 4. V1 QWEN RECONSTRUCTION SOURCE
# ============================================================
#
# Use RAW V1 Qwen result as reconstructed architecture.
#
# We do NOT use its final colors.
# Surface colors are replaced deterministically below.
# ============================================================

V1_RAW_PATH = (
    ROOT
    / "01z1_qwen_coherent_cleanroom_v1"
    / "00_raw_qwen_cleanroom.png"
)


# ============================================================
# 5. TARGET DIAGNOSTIC COLORS
# ============================================================

WALL_RGB = np.array(
    [0, 102, 255],
    dtype=np.uint8
)

CEILING_RGB = np.array(
    [255, 255, 255],
    dtype=np.uint8
)

FLOOR_RGB = np.array(
    [0, 200, 0],
    dtype=np.uint8
)


# ============================================================
# 6. HELPERS
# ============================================================

def load_rgb(path):

    return np.array(
        Image.open(
            path
        ).convert(
            "RGB"
        )
    )


def load_mask(path, size=None):

    img = Image.open(
        path
    ).convert(
        "L"
    )

    if size is not None and img.size != size:

        img = img.resize(
            size,
            Image.Resampling.NEAREST
        )

    return (
        np.array(img) > 0
    )


def save_mask(mask, path):

    Image.fromarray(
        (
            mask.astype(
                np.uint8
            )
            * 255
        ),
        mode="L"
    ).save(
        path
    )


def sha256_file(path):

    h = hashlib.sha256()

    with open(
        path,
        "rb"
    ) as f:

        while True:

            chunk = f.read(
                1024 * 1024
            )

            if not chunk:
                break

            h.update(
                chunk
            )

    return h.hexdigest()


# ============================================================
# 7. START
# ============================================================

print()
print("=" * 100)
print(STAGE_NAME)
print("=" * 100)

print()
print("SCRIPT:", SCRIPT_NAME)
print("OUTPUT:", OUT)

print()
print("NO NEW QWEN GENERATION IN V5")
print("Using V1 as reconstructed architecture source.")


# ============================================================
# 8. VALIDATE INPUTS
# ============================================================

required = {

    "ORIGINAL":
        ORIGINAL_PATH,

    "V1 RAW QWEN":
        V1_RAW_PATH,

    "WALL MASK":
        WALL_MASK_PATH,

    "CEILING MASK":
        CEILING_MASK_PATH,

    "FLOOR MASK":
        FLOOR_MASK_PATH,

    "STRUCTURE MASK":
        STRUCTURE_MASK_PATH,

    "MIRROR MASK":
        MIRROR_MASK_PATH,

    "GLASS MASK":
        GLASS_MASK_PATH,

    "GLASS->WALL":
        GLASS_TO_WALL_PATH,

    "GLASS->FLOOR":
        GLASS_TO_FLOOR_PATH,

    "GLASS->CEILING":
        GLASS_TO_CEILING_PATH,

    "PROP MASK":
        PROP_MASK_PATH,
}


print()
print("INPUT CHECK")
print("-" * 100)

for name, path in required.items():

    ok = path.exists()

    print(
        f"{name:18s}",
        "✅" if ok else "❌",
        path
    )

    if not ok:

        raise FileNotFoundError(
            path
        )


# ============================================================
# 9. LOAD ORIGINAL
# ============================================================

original = load_rgb(
    ORIGINAL_PATH
)

H, W = original.shape[:2]

MASTER_SIZE = (
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
# 10. LOAD V1 QWEN BASE
# ============================================================

v1_pil = Image.open(
    V1_RAW_PATH
).convert(
    "RGB"
)

if v1_pil.size != MASTER_SIZE:

    v1_pil = v1_pil.resize(
        MASTER_SIZE,
        Image.Resampling.LANCZOS
    )

v1 = np.array(
    v1_pil
)


# ============================================================
# 11. LOAD MASKS
# ============================================================

wall = load_mask(
    WALL_MASK_PATH,
    MASTER_SIZE
)

ceiling = load_mask(
    CEILING_MASK_PATH,
    MASTER_SIZE
)

floor = load_mask(
    FLOOR_MASK_PATH,
    MASTER_SIZE
)

structure = load_mask(
    STRUCTURE_MASK_PATH,
    MASTER_SIZE
)

mirror = load_mask(
    MIRROR_MASK_PATH,
    MASTER_SIZE
)

glass = load_mask(
    GLASS_MASK_PATH,
    MASTER_SIZE
)

glass_wall = load_mask(
    GLASS_TO_WALL_PATH,
    MASTER_SIZE
)

glass_floor = load_mask(
    GLASS_TO_FLOOR_PATH,
    MASTER_SIZE
)

glass_ceiling = load_mask(
    GLASS_TO_CEILING_PATH,
    MASTER_SIZE
)

props = load_mask(
    PROP_MASK_PATH,
    MASTER_SIZE
)


# ============================================================
# 12. PRINT RAW MASK COUNTS
# ============================================================

print()
print("RAW MASK PIXEL COUNTS")
print("-" * 100)

for name, m in {

    "wall":
        wall,

    "ceiling":
        ceiling,

    "floor":
        floor,

    "mirror":
        mirror,

    "glass":
        glass,

    "glass_wall":
        glass_wall,

    "glass_floor":
        glass_floor,

    "glass_ceiling":
        glass_ceiling,

    "props":
        props,

}.items():

    print(
        f"{name:18s}",
        int(
            m.sum()
        )
    )


# ============================================================
# 13. BUILD AUGMENTED SURFACE CLASSES
# ============================================================
#
# MIRROR:
# mirror covers wall, therefore mirror region -> WALL.
#
# GLASS:
# use our frozen background classification.
# ============================================================

wall_aug = (
    wall
    |
    mirror
    |
    glass_wall
)

floor_aug = (
    floor
    |
    glass_floor
)

ceiling_aug = (
    ceiling
    |
    glass_ceiling
)


# ============================================================
# 14. REMOVE FROZEN PHYSICAL PROPS FROM ALL SURFACES
# ============================================================

wall_aug &= ~props

floor_aug &= ~props

ceiling_aug &= ~props


# ============================================================
# 15. CLASS OVERLAP AUDIT
# ============================================================

wall_floor_overlap = (
    wall_aug
    &
    floor_aug
)

wall_ceiling_overlap = (
    wall_aug
    &
    ceiling_aug
)

floor_ceiling_overlap = (
    floor_aug
    &
    ceiling_aug
)

triple_overlap = (
    wall_aug
    &
    floor_aug
    &
    ceiling_aug
)


print()
print("CLASS OVERLAP AUDIT")
print("-" * 100)

print(
    "wall/floor overlap  :",
    int(
        wall_floor_overlap.sum()
    )
)

print(
    "wall/ceiling overlap:",
    int(
        wall_ceiling_overlap.sum()
    )
)

print(
    "floor/ceiling overlap:",
    int(
        floor_ceiling_overlap.sum()
    )
)

print(
    "triple overlap      :",
    int(
        triple_overlap.sum()
    )
)


# ============================================================
# 16. DETERMINISTIC CONFLICT RESOLUTION
# ============================================================
#
# Existing semantic classes should largely be exclusive.
#
# For rare overlaps:
#
# FLOOR gets priority at floor boundary.
# CEILING gets priority above wall.
# WALL receives remaining pixels.
#
# Priority:
# FLOOR > CEILING > WALL
#
# Props override EVERYTHING later.
# ============================================================

final_floor = (
    floor_aug.copy()
)

final_ceiling = (
    ceiling_aug
    &
    ~final_floor
)

final_wall = (
    wall_aug
    &
    ~final_floor
    &
    ~final_ceiling
)


# ============================================================
# 17. VERIFY FINAL CLASSES EXCLUSIVE
# ============================================================

assert not np.any(
    final_wall
    &
    final_floor
)

assert not np.any(
    final_wall
    &
    final_ceiling
)

assert not np.any(
    final_floor
    &
    final_ceiling
)


print()
print("FINAL CLASS PIXELS")
print("-" * 100)

print(
    "WALL   :",
    int(
        final_wall.sum()
    )
)

print(
    "CEILING:",
    int(
        final_ceiling.sum()
    )
)

print(
    "FLOOR  :",
    int(
        final_floor.sum()
    )
)


# ============================================================
# 18. SAVE FINAL CLASS MASKS
# ============================================================

save_mask(
    final_wall,
    OUT / "00_final_wall_mask.png"
)

save_mask(
    final_ceiling,
    OUT / "01_final_ceiling_mask.png"
)

save_mask(
    final_floor,
    OUT / "02_final_floor_mask.png"
)

save_mask(
    props,
    OUT / "03_frozen_prop_mask.png"
)

save_mask(
    mirror,
    OUT / "04_mirror_mask.png"
)

save_mask(
    glass,
    OUT / "05_glass_mask.png"
)


# ============================================================
# 19. CREATE CLASS LABEL MAP
# ============================================================
#
# 0 = other/background
# 1 = wall
# 2 = ceiling
# 3 = floor
# 4 = prop
# ============================================================

label_map = np.zeros(
    (
        H,
        W
    ),
    dtype=np.uint8
)

label_map[
    final_wall
] = 1

label_map[
    final_ceiling
] = 2

label_map[
    final_floor
] = 3

label_map[
    props
] = 4


Image.fromarray(
    label_map,
    mode="L"
).save(
    OUT
    / "06_surface_class_label_map.png"
)


# ============================================================
# 20. BUILD HYBRID DIAGNOSTIC IMAGE
# ============================================================
#
# Start from V1 reconstruction.
#
# This preserves coherent reconstruction for non-class regions.
# ============================================================

hybrid = (
    v1.copy()
)


# ------------------------------------------------------------
# WALL = BLUE
# ------------------------------------------------------------

hybrid[
    final_wall
] = WALL_RGB


# ------------------------------------------------------------
# CEILING = WHITE
# ------------------------------------------------------------

hybrid[
    final_ceiling
] = CEILING_RGB


# ------------------------------------------------------------
# FLOOR = GREEN
# ------------------------------------------------------------

hybrid[
    final_floor
] = FLOOR_RGB


# ============================================================
# 21. RESTORE ORIGINAL PROP RGB EXACTLY
# ============================================================

hybrid[
    props
] = original[
    props
]


# ============================================================
# 22. SAVE HYBRID RESULT
# ============================================================

hybrid_pil = Image.fromarray(
    hybrid
)

hybrid_path = (
    OUT
    / "07_hybrid_deterministic_surface_template.png"
)

hybrid_pil.save(
    hybrid_path
)


# ============================================================
# 23. EXACT COLOR VERIFICATION
# ============================================================

def class_exact_ratio(
    image,
    mask,
    target_rgb
):

    if not np.any(
        mask
    ):

        return 1.0

    pixels = (
        image[
            mask
        ]
    )

    correct = np.all(
        pixels
        ==
        target_rgb,
        axis=1
    )

    return float(
        correct.mean()
    )


wall_exact = class_exact_ratio(
    hybrid,
    final_wall,
    WALL_RGB
)

ceiling_exact = class_exact_ratio(
    hybrid,
    final_ceiling,
    CEILING_RGB
)

floor_exact = class_exact_ratio(
    hybrid,
    final_floor,
    FLOOR_RGB
)


# ============================================================
# 24. PROP RGB VERIFICATION
# ============================================================

prop_diff = np.abs(

    hybrid.astype(
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


print()
print("DETERMINISTIC RGB VERIFICATION")
print("-" * 100)

print(
    "Wall exact color %   :",
    round(
        wall_exact * 100,
        4
    )
)

print(
    "Ceiling exact color %:",
    round(
        ceiling_exact * 100,
        4
    )
)

print(
    "Floor exact color %  :",
    round(
        floor_exact * 100,
        4
    )
)

print(
    "Prop max RGB error   :",
    max_prop_rgb_error
)


# ============================================================
# 25. CREATE CLASS OVERLAY ON ORIGINAL
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
    alpha=0.55
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
    final_wall,
    WALL_RGB,
    0.55
)

overlay = tint(
    overlay,
    final_ceiling,
    CEILING_RGB,
    0.45
)

overlay = tint(
    overlay,
    final_floor,
    FLOOR_RGB,
    0.55
)

# Props shown magenta for diagnostic visibility
overlay = tint(
    overlay,
    props,
    [255, 0, 255],
    0.40
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
    / "08_surface_class_overlay_on_original.png"
)

Image.fromarray(
    overlay
).save(
    overlay_path
)


# ============================================================
# 26. CREATE 3-PANEL COMPARISON
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
        v1
    ),
    (
        W,
        LABEL_H
    )
)


comparison.paste(
    hybrid_pil,
    (
        W * 2,
        LABEL_H
    )
)


draw = ImageDraw.Draw(
    comparison
)


draw.text(
    (
        8,
        10
    ),
    "ORIGINAL",
    fill=(
        0,
        0,
        0
    )
)


draw.text(
    (
        W + 8,
        10
    ),
    "V1 QWEN RECONSTRUCTION",
    fill=(
        0,
        0,
        0
    )
)


draw.text(
    (
        W * 2 + 8,
        10
    ),
    "V5 HYBRID",
    fill=(
        0,
        0,
        0
    )
)


comparison_path = (
    OUT
    / "09_original_v1_v5_comparison.png"
)

comparison.save(
    comparison_path
)


# ============================================================
# 27. ORIGINAL VS V5
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
        original
    ),
    (
        0,
        LABEL_H
    )
)


compare2.paste(
    hybrid_pil,
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
    "ORIGINAL",
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
    "V5 HYBRID SURFACE CLASSES",
    fill=(
        0,
        0,
        0
    )
)


compare2_path = (
    OUT
    / "10_original_vs_v5.png"
)

compare2.save(
    compare2_path
)


# ============================================================
# 28. REPORT
# ============================================================

report = {

    "stage":
        STAGE_NAME,

    "status":
        "COMPLETED",

    "script":
        SCRIPT_NAME,

    "architecture":
        (
            "V1 Qwen reconstruction + deterministic "
            "surface-class application + exact prop RGB restoration"
        ),

    "new_qwen_generation":
        False,

    "sources": {

        "original":
            str(
                ORIGINAL_PATH
            ),

        "v1_qwen_reconstruction":
            str(
                V1_RAW_PATH
            ),

        "wall_mask":
            str(
                WALL_MASK_PATH
            ),

        "ceiling_mask":
            str(
                CEILING_MASK_PATH
            ),

        "floor_mask":
            str(
                FLOOR_MASK_PATH
            ),

        "mirror_mask":
            str(
                MIRROR_MASK_PATH
            ),

        "glass_mask":
            str(
                GLASS_MASK_PATH
            ),

        "glass_to_wall":
            str(
                GLASS_TO_WALL_PATH
            ),

        "glass_to_floor":
            str(
                GLASS_TO_FLOOR_PATH
            ),

        "glass_to_ceiling":
            str(
                GLASS_TO_CEILING_PATH
            ),

        "frozen_props":
            str(
                PROP_MASK_PATH
            ),
    },

    "source_hashes": {

        "original":
            sha256_file(
                ORIGINAL_PATH
            ),

        "v1_qwen":
            sha256_file(
                V1_RAW_PATH
            ),

        "wall_mask":
            sha256_file(
                WALL_MASK_PATH
            ),

        "floor_mask":
            sha256_file(
                FLOOR_MASK_PATH
            ),

        "ceiling_mask":
            sha256_file(
                CEILING_MASK_PATH
            ),
    },

    "surface_colors": {

        "wall": {
            "hex":
                "#0066FF",

            "rgb":
                WALL_RGB.tolist(),
        },

        "ceiling": {
            "hex":
                "#FFFFFF",

            "rgb":
                CEILING_RGB.tolist(),
        },

        "floor": {
            "hex":
                "#00C800",

            "rgb":
                FLOOR_RGB.tolist(),
        },
    },

    "raw_counts": {

        "wall":
            int(
                wall.sum()
            ),

        "ceiling":
            int(
                ceiling.sum()
            ),

        "floor":
            int(
                floor.sum()
            ),

        "mirror":
            int(
                mirror.sum()
            ),

        "glass":
            int(
                glass.sum()
            ),

        "glass_to_wall":
            int(
                glass_wall.sum()
            ),

        "glass_to_floor":
            int(
                glass_floor.sum()
            ),

        "glass_to_ceiling":
            int(
                glass_ceiling.sum()
            ),

        "props":
            int(
                props.sum()
            ),
    },

    "final_counts": {

        "wall":
            int(
                final_wall.sum()
            ),

        "ceiling":
            int(
                final_ceiling.sum()
            ),

        "floor":
            int(
                final_floor.sum()
            ),
    },

    "overlap_audit": {

        "wall_floor":
            int(
                wall_floor_overlap.sum()
            ),

        "wall_ceiling":
            int(
                wall_ceiling_overlap.sum()
            ),

        "floor_ceiling":
            int(
                floor_ceiling_overlap.sum()
            ),

        "triple":
            int(
                triple_overlap.sum()
            ),
    },

    "verification": {

        "wall_exact_color_ratio":
            wall_exact,

        "ceiling_exact_color_ratio":
            ceiling_exact,

        "floor_exact_color_ratio":
            floor_exact,

        "max_prop_rgb_error":
            max_prop_rgb_error,
    },

    "outputs": {

        "wall_mask":
            str(
                OUT
                / "00_final_wall_mask.png"
            ),

        "ceiling_mask":
            str(
                OUT
                / "01_final_ceiling_mask.png"
            ),

        "floor_mask":
            str(
                OUT
                / "02_final_floor_mask.png"
            ),

        "class_map":
            str(
                OUT
                / "06_surface_class_label_map.png"
            ),

        "hybrid":
            str(
                hybrid_path
            ),

        "overlay":
            str(
                overlay_path
            ),

        "comparison":
            str(
                comparison_path
            ),

        "original_vs_v5":
            str(
                compare2_path
            ),
    },

    "benchmark": {

        "V1":
            "Qwen coherent reconstruction reference",

        "V2":
            "failed long prompt",

        "V3":
            "successful white-wall / green-floor generative test",

        "V4":
            "surface-class prompt diagnostic",

        "V5":
            "hybrid deterministic surface-class architecture",
    }
}


report_path = (
    OUT
    / "00_stage01z5_report.json"
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
# 29. FINAL STATUS
# ============================================================

print()
print("=" * 100)
print("STAGE01Z5 COMPLETE")
print("=" * 100)

print()
print(
    "NO QWEN GENERATION WAS REQUIRED."
)

print()
print(
    "HYBRID RESULT:"
)

print(
    hybrid_path
)

print()
print(
    "CLASS OVERLAY:"
)

print(
    overlay_path
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

print()
print(
    "MAX PROP RGB ERROR:",
    max_prop_rgb_error
)


# ============================================================
# 30. DISPLAY
# ============================================================

print()
print(
    "SURFACE CLASS OVERLAY ON ORIGINAL"
)

display(
    Image.open(
        overlay_path
    )
)

print()
print(
    "ORIGINAL / V1 RECONSTRUCTION / V5 HYBRID"
)

display(
    comparison
)

print()
print(
    "ORIGINAL VS V5"
)

display(
    compare2
)
