
from pathlib import Path
from PIL import Image, ImageDraw
from IPython.display import display

import hashlib
import json
import numpy as np
from datetime import datetime


# ============================================================
# 1. PATHS
# ============================================================

BASE = Path("/workspace/axolotl")

PROJECT = (
    BASE
    / "test07"
)

PROD = (
    PROJECT
    / "production_pipeline"
)

STAGE01 = (
    PROD
    / "stage01_master"
)

STAGE04 = (
    PROD
    / "stage04_glass"
)

STAGE04.mkdir(
    parents=True,
    exist_ok=True
)


MASTER_PATH = (
    STAGE01
    / "00_master_input.png"
)


# ============================================================
# VERIFIED TEST07 GLASS ARTIFACTS
# ============================================================

FROZEN_GLASS_MASK = (
    PROJECT
    / "runs"
    / "room03_bathroom"
    / "stages"
    / "01n_canonical_glass_layer"
    / "00_canonical_glass_system_mask.png"
)


FROZEN_GLASS_STATE = (
    PROJECT
    / "runs"
    / "room03_bathroom"
    / "stages"
    / "01o_v3_frozen_glass_state"
    / "00_frozen_glass_state.json"
)


# Existing hidden-background classifications.
# We preserve references to these because they will be useful
# later when Stage05 reconstructs the room without glass.

GLASS_TO_WALL = (
    PROJECT
    / "runs"
    / "room03_bathroom"
    / "stages"
    / "01w2_v3_glass_background_classification"
    / "00_glass_to_wall.png"
)


GLASS_TO_FLOOR = (
    PROJECT
    / "runs"
    / "room03_bathroom"
    / "stages"
    / "01w2_v3_glass_background_classification"
    / "01_glass_to_floor.png"
)


GLASS_TO_CEILING = (
    PROJECT
    / "runs"
    / "room03_bathroom"
    / "stages"
    / "01w2_v3_glass_background_classification"
    / "02_glass_to_ceiling.png"
)


# ============================================================
# 2. OUTPUTS
# ============================================================

GLASS_MASK_PATH = (
    STAGE04
    / "00_glass_mask.png"
)


GLASS_REFERENCE_LAYER_PATH = (
    STAGE04
    / "01_original_glass_reference_rgba.png"
)


BBOX_PREVIEW_PATH = (
    STAGE04
    / "02_glass_bbox_preview.png"
)


PLACEHOLDER_PATH = (
    STAGE04
    / "03_master_without_glass_placeholder.png"
)


REPORT_PATH = (
    STAGE04
    / "00_stage04_glass_state.json"
)


# ============================================================
# 3. HELPERS
# ============================================================

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


def bbox_from_mask(mask):

    ys, xs = np.where(
        mask
    )

    if len(xs) == 0:

        return None

    return [
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    ]


def load_binary_mask(
    path,
    expected_size
):

    img = Image.open(
        path
    ).convert("L")

    if img.size != expected_size:

        raise RuntimeError(
            f"Coordinate mismatch for {path.name}: "
            f"{img.size} != {expected_size}"
        )

    return (
        np.array(img) > 0
    )


# ============================================================
# 4. START
# ============================================================

print()
print("=" * 110)
print("TEST07 PRODUCTION")
print("STAGE 04A - GLASS PARTITION STATE + REFERENCE LAYER")
print("=" * 110)


# ============================================================
# 5. REQUIRED INPUT CHECK
# ============================================================

required = {
    "MASTER":
        MASTER_PATH,

    "CANONICAL GLASS MASK":
        FROZEN_GLASS_MASK,

    "FROZEN GLASS STATE":
        FROZEN_GLASS_STATE,

    "GLASS TO WALL":
        GLASS_TO_WALL,

    "GLASS TO FLOOR":
        GLASS_TO_FLOOR,

    "GLASS TO CEILING":
        GLASS_TO_CEILING,
}


print()
print("INPUT CHECK")
print("-" * 110)


for name, path in required.items():

    exists = path.exists()

    print(
        f"{name:24s}",
        "✅" if exists else "❌",
        path
    )

    if not exists:

        raise FileNotFoundError(
            path
        )


# ============================================================
# 6. LOAD PRODUCTION MASTER
# ============================================================

master_pil = Image.open(
    MASTER_PATH
).convert("RGB")

master = np.array(
    master_pil
)

H, W = master.shape[:2]

EXPECTED_SIZE = (
    W,
    H
)


print()
print(
    "PRODUCTION MASTER:",
    W,
    "x",
    H
)


# ============================================================
# 7. LOAD CANONICAL GLASS MASK
# ============================================================

glass_mask = load_binary_mask(
    FROZEN_GLASS_MASK,
    EXPECTED_SIZE
)


glass_pixels = int(
    glass_mask.sum()
)


if glass_pixels == 0:

    raise RuntimeError(
        "Canonical glass mask is empty."
    )


bbox = bbox_from_mask(
    glass_mask
)


print()
print(
    "GLASS PIXELS:",
    glass_pixels
)

print(
    "GLASS BBOX:",
    bbox
)


# ============================================================
# 8. LOAD HIDDEN BACKGROUND CLASSIFICATIONS
# ============================================================

glass_to_wall = load_binary_mask(
    GLASS_TO_WALL,
    EXPECTED_SIZE
)


glass_to_floor = load_binary_mask(
    GLASS_TO_FLOOR,
    EXPECTED_SIZE
)


glass_to_ceiling = load_binary_mask(
    GLASS_TO_CEILING,
    EXPECTED_SIZE
)


wall_pixels = int(
    glass_to_wall.sum()
)

floor_pixels = int(
    glass_to_floor.sum()
)

ceiling_pixels = int(
    glass_to_ceiling.sum()
)


classified_union = (
    glass_to_wall
    |
    glass_to_floor
    |
    glass_to_ceiling
)


classified_pixels = int(
    classified_union.sum()
)


print()
print("HIDDEN BACKGROUND CLASSIFICATION")

print(
    "GLASS → WALL   :",
    wall_pixels
)

print(
    "GLASS → FLOOR  :",
    floor_pixels
)

print(
    "GLASS → CEILING:",
    ceiling_pixels
)

print(
    "CLASSIFIED UNION:",
    classified_pixels
)


# ============================================================
# 9. SAVE PRODUCTION GLASS MASK
# ============================================================

Image.fromarray(
    glass_mask.astype(
        np.uint8
    ) * 255,
    mode="L"
).save(
    GLASS_MASK_PATH
)


# ============================================================
# 10. CREATE EXACT ORIGINAL-RGB GLASS REFERENCE LAYER
# ============================================================
#
# This is NOT yet the recreated production glass.
#
# It simply preserves the original glass-system appearance
# and exact location for analysis/reference.
#
# RGB = exact Stage01 master pixels
# Alpha = canonical glass mask
# ============================================================

glass_rgba = np.zeros(
    (
        H,
        W,
        4
    ),
    dtype=np.uint8
)


glass_rgba[
    glass_mask,
    0:3
] = master[
    glass_mask
]


glass_rgba[
    glass_mask,
    3
] = 255


Image.fromarray(
    glass_rgba,
    mode="RGBA"
).save(
    GLASS_REFERENCE_LAYER_PATH
)


# ============================================================
# 11. VERIFY EXACT RGB
# ============================================================

rgb_error = np.abs(
    glass_rgba[
        :,
        :,
        :3
    ].astype(
        np.int16
    )
    -
    master.astype(
        np.int16
    )
)


max_rgb_error = int(
    rgb_error[
        glass_mask
    ].max()
)


print()
print(
    "MAX RGB ERROR INSIDE GLASS MASK:",
    max_rgb_error
)


if max_rgb_error != 0:

    raise RuntimeError(
        "Glass reference layer RGB differs from master."
    )


# ============================================================
# 12. BBOX PREVIEW
# ============================================================

preview = master_pil.copy()

draw = ImageDraw.Draw(
    preview
)


x1, y1, x2, y2 = bbox


draw.rectangle(
    [
        x1,
        y1,
        x2 - 1,
        y2 - 1
    ],
    outline=(
        255,
        0,
        0
    ),
    width=2
)


draw.text(
    (
        x1,
        max(
            0,
            y1 - 16
        )
    ),
    "GLASS PARTITION",
    fill=(
        255,
        0,
        0
    )
)


preview.save(
    BBOX_PREVIEW_PATH
)


# ============================================================
# 13. GLASS-REMOVAL PLACEHOLDER
# ============================================================
#
# Diagnostic only.
#
# White indicates the region that Stage05 must reconstruct
# after removing the glass system.
#
# This is NOT final background.
# ============================================================

placeholder = master.copy()


placeholder[
    glass_mask
] = [
    255,
    255,
    255
]


Image.fromarray(
    placeholder
).save(
    PLACEHOLDER_PATH
)


# ============================================================
# 14. READ EXISTING GLASS STATE
# ============================================================

legacy_state = json.loads(
    FROZEN_GLASS_STATE.read_text(
        encoding="utf-8"
    )
)


# ============================================================
# 15. PRODUCTION GLASS STATE
# ============================================================

report = {

    "stage":
        "PRODUCTION_STAGE04A_GLASS_STATE",

    "status":
        "PASS_MIGRATED_PENDING_AI_RECREATION",

    "created":
        datetime.now().isoformat(),

    "role":
        (
            "Canonicalize the glass partition as a separate "
            "special system and preserve all verified geometry "
            "and hidden-background knowledge."
        ),

    "production_master": {

        "path":
            str(
                MASTER_PATH
            ),

        "width":
            W,

        "height":
            H,

        "sha256":
            sha256_file(
                MASTER_PATH
            ),
    },

    "verified_source": {

        "canonical_glass_mask":
            str(
                FROZEN_GLASS_MASK
            ),

        "frozen_glass_state":
            str(
                FROZEN_GLASS_STATE
            ),
    },

    "glass": {

        "pixels":
            glass_pixels,

        "bbox_xyxy":
            bbox,

        "master_fraction":
            (
                glass_pixels
                /
                float(
                    W * H
                )
            ),

        "max_rgb_error_reference_layer":
            max_rgb_error,
    },

    "hidden_background": {

        "wall_pixels":
            wall_pixels,

        "floor_pixels":
            floor_pixels,

        "ceiling_pixels":
            ceiling_pixels,

        "classified_union_pixels":
            classified_pixels,

        "glass_to_wall":
            str(
                GLASS_TO_WALL
            ),

        "glass_to_floor":
            str(
                GLASS_TO_FLOOR
            ),

        "glass_to_ceiling":
            str(
                GLASS_TO_CEILING
            ),
    },

    "known_room03_glass_description": {

        "system":
            "two-panel shower enclosure",

        "main_panel":
            (
                "Main panel approximately x224-x320; "
                "upper region primarily clear and lower "
                "region frosted/translucent."
            ),

        "secondary_panel":
            (
                "Narrow right panel approximately x320-x358; "
                "predominantly clear."
            ),

        "hardware":
            (
                "Visible rails/frame and horizontal handle "
                "belong to the glass system."
            ),
    },

    "legacy_state":
        legacy_state,

    "outputs": {

        "glass_mask":
            str(
                GLASS_MASK_PATH
            ),

        "original_glass_reference_rgba":
            str(
                GLASS_REFERENCE_LAYER_PATH
            ),

        "bbox_preview":
            str(
                BBOX_PREVIEW_PATH
            ),

        "reconstruction_placeholder":
            str(
                PLACEHOLDER_PATH
            ),
    },

    "next_step":
        (
            "Stage04B will recreate the same glass partition "
            "as a reusable production layer while preserving "
            "the detected position, panel arrangement, "
            "clear/frosted character and hardware."
        ),

    "frozen_rules": [

        "Glass is a separate special layer.",

        "Original glass geometry remains tied to Stage01 coordinates.",

        "The original-glass RGBA file is a reference layer, not the final recreated layer.",

        "Stage05 receives the room without the glass system.",

        "Stage05 reconstruction uses hidden-background wall/floor/ceiling knowledge.",

        "Glass is reintroduced only during final compositing.",

        "Existing verified TEST07 glass artifacts are never overwritten.",
    ],
}


REPORT_PATH.write_text(
    json.dumps(
        report,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# ============================================================
# 16. FINAL
# ============================================================

print()
print("=" * 110)
print("PRODUCTION STAGE 04A COMPLETE")
print("=" * 110)

print()

print(
    "GLASS PIXELS:",
    glass_pixels
)

print(
    "GLASS BBOX:",
    bbox
)

print(
    "MAX RGB ERROR:",
    max_rgb_error
)

print()

print(
    "GLASS → WALL:",
    wall_pixels
)

print(
    "GLASS → FLOOR:",
    floor_pixels
)

print(
    "GLASS → CEILING:",
    ceiling_pixels
)

print()

print(
    "MASK:",
    GLASS_MASK_PATH
)

print(
    "REFERENCE RGBA:",
    GLASS_REFERENCE_LAYER_PATH
)

print(
    "PLACEHOLDER:",
    PLACEHOLDER_PATH
)

print(
    "STATE:",
    REPORT_PATH
)


print()
print("GLASS BBOX PREVIEW")

display(
    Image.open(
        BBOX_PREVIEW_PATH
    )
)


print()
print("ORIGINAL GLASS REFERENCE LAYER")

display(
    Image.open(
        GLASS_REFERENCE_LAYER_PATH
    )
)


print()
print("GLASS-REMOVAL PLACEHOLDER")

display(
    Image.open(
        PLACEHOLDER_PATH
    )
)
