
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

STAGE03 = (
    PROD
    / "stage03_mirror"
)

STAGE03.mkdir(
    parents=True,
    exist_ok=True
)


MASTER_PATH = (
    STAGE01
    / "00_master_input.png"
)


# ------------------------------------------------------------
# Frozen successful TEST07 Room03 mirror result
# ------------------------------------------------------------

FROZEN_MIRROR_MASK = (
    PROJECT
    / "runs"
    / "room03_bathroom"
    / "stages"
    / "01f_mirror_shape_candidate"
    / "00_candidate02_mirror_shape_mask.png"
)


FROZEN_MIRROR_STATE = (
    PROJECT
    / "runs"
    / "room03_bathroom"
    / "stages"
    / "01b_mirror_boundary"
    / "01_canonical_mirror_state.json"
)


# ============================================================
# 2. OUTPUTS
# ============================================================

MIRROR_MASK_PATH = (
    STAGE03
    / "00_mirror_mask.png"
)

MIRROR_LAYER_PATH = (
    STAGE03
    / "01_mirror_layer_rgba.png"
)

BBOX_PREVIEW_PATH = (
    STAGE03
    / "02_mirror_bbox_preview.png"
)

PLACEHOLDER_PATH = (
    STAGE03
    / "03_master_without_mirror_placeholder.png"
)

REPORT_PATH = (
    STAGE03
    / "00_stage03_mirror_state.json"
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

    ys, xs = np.where(mask)

    if len(xs) == 0:

        return None

    return [
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    ]


# ============================================================
# 4. START
# ============================================================

print()
print("=" * 110)
print("TEST07 PRODUCTION")
print("STAGE 03 - MIRROR SPECIAL LAYER")
print("=" * 110)


# ============================================================
# 5. INPUT CHECK
# ============================================================

required = {
    "MASTER":
        MASTER_PATH,

    "FROZEN MIRROR MASK":
        FROZEN_MIRROR_MASK,
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
# 6. LOAD MASTER
# ============================================================

master_pil = Image.open(
    MASTER_PATH
).convert("RGB")

master = np.array(
    master_pil
)

H, W = master.shape[:2]


print()
print(
    "PRODUCTION MASTER:",
    W,
    "x",
    H
)


# ============================================================
# 7. LOAD VERIFIED MIRROR MASK
# ============================================================

mirror_mask_pil = Image.open(
    FROZEN_MIRROR_MASK
).convert("L")


print(
    "FROZEN MIRROR MASK SIZE:",
    mirror_mask_pil.size
)


# ============================================================
# 8. STRICT COORDINATE CHECK
# ============================================================
#
# We do NOT resize a verified mask silently.
# Production master and frozen mirror mask must already match.
# ============================================================

if mirror_mask_pil.size != (
    W,
    H
):

    raise RuntimeError(
        "Mirror mask coordinate mismatch. "
        f"MASTER={W}x{H}, "
        f"MIRROR={mirror_mask_pil.size}. "
        "Do not resize the mirror mask automatically."
    )


mirror_mask = (
    np.array(
        mirror_mask_pil
    ) > 0
)


mirror_pixels = int(
    mirror_mask.sum()
)


if mirror_pixels == 0:

    raise RuntimeError(
        "Verified mirror mask is empty."
    )


# ============================================================
# 9. SAVE PRODUCTION MIRROR MASK
# ============================================================

Image.fromarray(
    mirror_mask.astype(
        np.uint8
    ) * 255,
    mode="L"
).save(
    MIRROR_MASK_PATH
)


# ============================================================
# 10. MIRROR BBOX
# ============================================================

bbox = bbox_from_mask(
    mirror_mask
)


print()
print(
    "MIRROR PIXELS:",
    mirror_pixels
)

print(
    "MIRROR BBOX:",
    bbox
)


# ============================================================
# 11. CREATE EXACT ORIGINAL-RGB RGBA MIRROR LAYER
# ============================================================
#
# RGB:
#   exact pixels from Production Stage01 master
#
# Alpha:
#   mirror mask
#
# Everything outside mirror:
#   transparent
# ============================================================

rgba = np.zeros(
    (
        H,
        W,
        4
    ),
    dtype=np.uint8
)


rgba[
    mirror_mask,
    0:3
] = master[
    mirror_mask
]


rgba[
    mirror_mask,
    3
] = 255


Image.fromarray(
    rgba,
    mode="RGBA"
).save(
    MIRROR_LAYER_PATH
)


# ============================================================
# 12. VERIFY EXACT RGB
# ============================================================

layer_rgb = rgba[:, :, :3]

rgb_error = np.abs(
    layer_rgb.astype(np.int16)
    -
    master.astype(np.int16)
)


if mirror_mask.any():

    max_rgb_error = int(
        rgb_error[
            mirror_mask
        ].max()
    )

else:

    max_rgb_error = 0


print(
    "MAX RGB ERROR INSIDE MIRROR:",
    max_rgb_error
)


if max_rgb_error != 0:

    raise RuntimeError(
        "Mirror layer RGB is not exact master RGB."
    )


# ============================================================
# 13. BBOX PREVIEW
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
    "MIRROR",
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
# 14. MASTER WITHOUT MIRROR PLACEHOLDER
# ============================================================
#
# IMPORTANT:
# This is ONLY a pipeline diagnostic.
#
# Mirror pixels are replaced with neutral white so later
# inspection can clearly see which region requires background
# reconstruction.
#
# THIS IMAGE MUST NOT BE USED AS FINAL BACKGROUND.
# Stage05 will reconstruct/simplify the real surface.
# ============================================================

placeholder = master.copy()


placeholder[
    mirror_mask
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
# 15. READ LEGACY STATE IF AVAILABLE
# ============================================================

legacy_state = None


if FROZEN_MIRROR_STATE.exists():

    try:

        legacy_state = json.loads(
            FROZEN_MIRROR_STATE.read_text(
                encoding="utf-8"
            )
        )

    except Exception as e:

        legacy_state = {
            "read_error":
                str(e)
        }


# ============================================================
# 16. PRODUCTION STATE
# ============================================================

report = {

    "stage":
        "PRODUCTION_STAGE03_MIRROR",

    "status":
        "PASS_FROZEN_FOR_ROOM03",

    "created":
        datetime.now().isoformat(),

    "role":
        (
            "Detect/canonicalize mirror and store it as "
            "a separate special layer."
        ),

    "production_input": {

        "master":
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

    "mirror_source": {

        "method":
            (
                "Migrated from previously verified "
                "TEST07 Room03 mirror result."
            ),

        "frozen_mask":
            str(
                FROZEN_MIRROR_MASK
            ),

        "frozen_state":
            str(
                FROZEN_MIRROR_STATE
            ),

        "legacy_state_loaded":
            legacy_state is not None,
    },

    "mirror": {

        "pixels":
            mirror_pixels,

        "bbox_xyxy":
            bbox,

        "master_fraction":
            (
                mirror_pixels
                /
                float(
                    W * H
                )
            ),

        "layer_rgb_source":
            "Production Stage01 master",

        "max_rgb_error_inside_mask":
            max_rgb_error,
    },

    "outputs": {

        "mirror_mask":
            str(
                MIRROR_MASK_PATH
            ),

        "mirror_rgba_layer":
            str(
                MIRROR_LAYER_PATH
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

    "frozen_rules": [

        "Mirror is a separate special layer.",

        "Mirror reflections must not become physical props.",

        "Mirror RGB must come from Stage01 master.",

        "Mirror mask must stay in Stage01 master coordinates.",

        "The white placeholder is diagnostic only.",

        "Actual hidden wall/background reconstruction occurs later.",

        "Experimental verified mirror artifacts are not overwritten.",
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
# 17. FINAL
# ============================================================

print()
print("=" * 110)
print("PRODUCTION STAGE 03 COMPLETE")
print("=" * 110)

print()

print(
    "MIRROR PIXELS:",
    mirror_pixels
)

print(
    "MIRROR BBOX:",
    bbox
)

print(
    "MAX RGB ERROR:",
    max_rgb_error
)

print()

print(
    "MASK:",
    MIRROR_MASK_PATH
)

print(
    "RGBA LAYER:",
    MIRROR_LAYER_PATH
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
print("MIRROR BBOX PREVIEW")

display(
    Image.open(
        BBOX_PREVIEW_PATH
    )
)


print()
print("MIRROR LAYER")

display(
    Image.open(
        MIRROR_LAYER_PATH
    )
)


print()
print("MIRROR-REMOVAL PLACEHOLDER")

display(
    Image.open(
        PLACEHOLDER_PATH
    )
)
