
from pathlib import Path
from PIL import Image
import hashlib
import json
from datetime import datetime


BASE = Path("/workspace/axolotl")
PROJECT = BASE / "test07"
PROD = PROJECT / "production_pipeline"

STAGE_DIR = PROD / "stage01_master"
STAGE_DIR.mkdir(parents=True, exist_ok=True)


SOURCE_IMAGE = (
    PROJECT
    / "runs"
    / "room03_bathroom"
    / "stages"
    / "01_structure_inverse_base"
    / "00_input_resized.png"
)


MAX_LONG_SIDE = 1536


MASTER_PATH = (
    STAGE_DIR
    / "00_master_input.png"
)

REPORT_PATH = (
    STAGE_DIR
    / "00_stage01_master_report.json"
)


def sha256_file(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:

        while True:

            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


print("=" * 100)
print("TEST07 PRODUCTION - STAGE 01 MASTER INPUT")
print("=" * 100)

print()
print("SOURCE:")
print(SOURCE_IMAGE)


if not SOURCE_IMAGE.exists():

    raise FileNotFoundError(
        f"Source image not found: {SOURCE_IMAGE}"
    )


img = Image.open(
    SOURCE_IMAGE
).convert("RGB")


original_w, original_h = img.size

original_long_side = max(
    original_w,
    original_h
)


print()
print(
    "SOURCE SIZE:",
    original_w,
    "x",
    original_h
)


# ============================================================
# RESIZE ONLY IF REQUIRED
# ============================================================

if original_long_side > MAX_LONG_SIDE:

    scale = (
        MAX_LONG_SIDE
        / original_long_side
    )

    new_w = max(
        1,
        round(
            original_w * scale
        )
    )

    new_h = max(
        1,
        round(
            original_h * scale
        )
    )

    master = img.resize(
        (
            new_w,
            new_h
        ),
        Image.Resampling.LANCZOS
    )

    resized = True

else:

    master = img.copy()

    new_w = original_w
    new_h = original_h

    resized = False


# ============================================================
# SAVE MASTER
# ============================================================

master.save(
    MASTER_PATH,
    format="PNG"
)


master_hash = sha256_file(
    MASTER_PATH
)


# ============================================================
# REPORT
# ============================================================

report = {

    "stage":
        "PRODUCTION_STAGE01_MASTER",

    "status":
        "COMPLETED",

    "created":
        datetime.now().isoformat(),

    "source_image":
        str(
            SOURCE_IMAGE
        ),

    "master_image":
        str(
            MASTER_PATH
        ),

    "resize_policy": {
        "max_long_side":
            MAX_LONG_SIDE,
        "preserve_aspect_ratio":
            True,
    },

    "source_dimensions": {
        "width":
            original_w,
        "height":
            original_h,
    },

    "master_dimensions": {
        "width":
            new_w,
        "height":
            new_h,
    },

    "resized":
        resized,

    "master_sha256":
        master_hash,

    "frozen_rules": [
        "This master image defines the production coordinate system.",
        "All later masks and layers must map exactly to this master.",
        "Final prop RGB must come from this master input.",
        "Generated helper images must never replace this geometry/RGB source of truth."
    ]
}


REPORT_PATH.write_text(
    json.dumps(
        report,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


print()
print("=" * 100)
print("STAGE 01 COMPLETE")
print("=" * 100)

print()
print(
    "RESIZED:",
    resized
)

print(
    "MASTER SIZE:",
    new_w,
    "x",
    new_h
)

print(
    "MASTER SHA256:",
    master_hash
)

print()
print(
    "MASTER:",
    MASTER_PATH
)

print(
    "REPORT:",
    REPORT_PATH
)
