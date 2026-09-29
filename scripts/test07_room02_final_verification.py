
from pathlib import Path
import json
import cv2
import numpy as np


ROOT = Path(
    "/workspace/axolotl/test07/runs/"
    "room02_kitchen/stages"
)

MASTER = (
    ROOT
    / "01_structure_inverse_base"
    / "00_input_resized.png"
)

STAGE08_MASK = (
    ROOT
    / "08_base_seeded_local_recovery"
    / "03_final_props_mask.png"
)

STAGE10_MASK = (
    ROOT
    / "10_v2_local_crop_recovery_verification"
    / "02_final_props_mask.png"
)

STAGE10_REPORT = (
    ROOT
    / "10_v2_local_crop_recovery_verification"
    / "stage10_report.json"
)

STAGE11_MASK = (
    ROOT
    / "11_rejected_candidate_recovery"
    / "02_final_props_mask.png"
)

STAGE11_REPORT = (
    ROOT
    / "11_rejected_candidate_recovery"
    / "stage11_report.json"
)

STAGE095_REPORT = (
    ROOT
    / "09_5_instance_collision_audit"
    / "stage095_report.json"
)

OUT = (
    ROOT
    / "12_room02_final_verification"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def load_mask(path):

    img = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE
    )

    if img is None:
        raise RuntimeError(
            f"Could not read mask: {path}"
        )

    return img > 127


def count_pixels(mask):

    return int(
        mask.sum()
    )


print("=" * 100)
print("TEST07 ROOM02 FINAL REFERENCE VERIFICATION")
print("=" * 100)


# ============================================================
# FILE EXISTENCE
# ============================================================

required = [
    MASTER,
    STAGE08_MASK,
    STAGE10_MASK,
    STAGE10_REPORT,
    STAGE11_MASK,
    STAGE11_REPORT,
    STAGE095_REPORT,
]

missing_files = [
    str(p)
    for p in required
    if not p.exists()
]

if missing_files:

    print()
    print("❌ MISSING REQUIRED FILES")

    for p in missing_files:
        print(p)

    raise SystemExit(1)


# ============================================================
# LOAD
# ============================================================

master = cv2.imread(
    str(MASTER),
    cv2.IMREAD_COLOR
)

if master is None:
    raise RuntimeError(
        "Could not read locked master."
    )

stage08 = load_mask(
    STAGE08_MASK
)

stage10 = load_mask(
    STAGE10_MASK
)

stage11 = load_mask(
    STAGE11_MASK
)

s10 = load_json(
    STAGE10_REPORT
)

s11 = load_json(
    STAGE11_REPORT
)

s095 = load_json(
    STAGE095_REPORT
)


# ============================================================
# DIMENSION CHECK
# ============================================================

H, W = master.shape[:2]

dimension_pass = (
    stage08.shape == (H, W)
    and
    stage10.shape == (H, W)
    and
    stage11.shape == (H, W)
)


# ============================================================
# PIXEL COUNTS
# ============================================================

stage08_pixels = count_pixels(
    stage08
)

stage10_pixels = count_pixels(
    stage10
)

stage11_pixels = count_pixels(
    stage11
)

stage10_added = (
    stage10_pixels
    -
    stage08_pixels
)

stage11_added = (
    stage11_pixels
    -
    stage10_pixels
)


# ============================================================
# REPORT COUNTS
# ============================================================

stage10_report_added = int(
    s10.get(
        "accepted_recovery_pixels",
        -1
    )
)

stage11_report_added = int(
    s11.get(
        "accepted_recovery_pixels",
        -1
    )
)

stage11_remaining = [
    int(x)
    for x in s11.get(
        "remaining_unresolved_ids",
        []
    )
]

stage11_accepted_ids = sorted(
    int(x)
    for x in s11.get(
        "accepted_ids",
        []
    )
)

expected_stage11_targets = [
    11,
    12,
    14,
    48,
    49,
    52,
]


# ============================================================
# ALIAS CHECK
# ============================================================

alias_pairs = s095.get(
    "same_instance_duplicate_pairs",
    []
)

normalized_alias_pairs = sorted(
    sorted(
        [
            int(pair[0]),
            int(pair[1])
        ]
    )
    for pair in alias_pairs
)

expected_alias_pairs = sorted([
    [3, 7],
    [27, 58],
])

alias_pass = (
    normalized_alias_pairs
    ==
    expected_alias_pairs
)


# ============================================================
# MONOTONIC ADDITIVE CHECK
# ============================================================

stage08_lost_in_stage10 = int(
    np.logical_and(
        stage08,
        ~stage10
    ).sum()
)

stage10_lost_in_stage11 = int(
    np.logical_and(
        stage10,
        ~stage11
    ).sum()
)

additive_pass = (
    stage08_lost_in_stage10 == 0
    and
    stage10_lost_in_stage11 == 0
)


# ============================================================
# EXACT RGB CHECK
# ============================================================

# Construct canonical final RGBA directly from:
# locked master RGB + Stage11 final mask.
#
# This guarantees no regenerated/redrawn object RGB.

rgba = np.zeros(
    (H, W, 4),
    dtype=np.uint8
)

rgba[:, :, :3] = master

rgba[:, :, 3] = (
    stage11.astype(
        np.uint8
    )
    *
    255
)

RGBA_PATH = (
    OUT
    / "00_room02_final_props_rgba.png"
)

cv2.imwrite(
    str(RGBA_PATH),
    rgba
)

# Verify every visible RGB pixel equals master exactly.

saved_rgba = cv2.imread(
    str(RGBA_PATH),
    cv2.IMREAD_UNCHANGED
)

visible = (
    saved_rgba[:, :, 3]
    >
    0
)

rgb_difference = np.abs(
    saved_rgba[:, :, :3].astype(
        np.int16
    )
    -
    master.astype(
        np.int16
    )
)

visible_rgb_diff = rgb_difference[
    visible
]

if visible_rgb_diff.size:

    max_rgb_error = int(
        visible_rgb_diff.max()
    )

else:

    max_rgb_error = 0

exact_rgb_pass = (
    max_rgb_error == 0
)


# ============================================================
# FINAL LOGIC
# ============================================================

checks = {
    "required_files_exist":
        len(missing_files) == 0,

    "dimensions_match_master":
        dimension_pass,

    "stage08_pixels_expected":
        stage08_pixels == 87877,

    "stage10_pixels_expected":
        stage10_pixels == 90094,

    "stage11_pixels_expected":
        stage11_pixels == 90253,

    "stage10_added_pixels_expected":
        stage10_added == 2217,

    "stage11_added_pixels_expected":
        stage11_added == 159,

    "stage10_report_matches_mask":
        stage10_report_added
        ==
        stage10_added,

    "stage11_report_matches_mask":
        stage11_report_added
        ==
        stage11_added,

    "stage11_targets_correct":
        stage11_accepted_ids
        ==
        expected_stage11_targets,

    "no_remaining_unresolved":
        stage11_remaining == [],

    "same_instance_aliases_correct":
        alias_pass,

    "additive_foreground_preserved":
        additive_pass,

    "exact_master_rgb_preserved":
        exact_rgb_pass,
}


failed = [
    key
    for key, passed
    in checks.items()
    if not passed
]


reference_pass = (
    len(failed) == 0
)


# ============================================================
# SAVE JSON REPORT
# ============================================================

report = {

    "experiment":
        "TEST07_ROOM02_FINAL_REFERENCE_VERIFICATION",

    "room_id":
        "room02_kitchen",

    "result":
        (
            "REFERENCE_PASS"
            if reference_pass
            else
            "REFERENCE_FAIL"
        ),

    "master_size": {
        "width": W,
        "height": H,
    },

    "pixel_accounting": {

        "stage08_pixels":
            stage08_pixels,

        "stage10_pixels":
            stage10_pixels,

        "stage10_added_pixels":
            stage10_added,

        "stage11_pixels":
            stage11_pixels,

        "stage11_added_pixels":
            stage11_added,
    },

    "physical_instance_resolution": {

        "stage11_input_unresolved":
            expected_stage11_targets,

        "stage11_accepted":
            stage11_accepted_ids,

        "remaining_unresolved":
            stage11_remaining,

        "resolved_by_existing_base": [
            27
        ],

        "same_instance_alias_map": {
            "3": [7],
            "27": [58],
        },
    },

    "safety": {

        "stage08_pixels_lost_in_stage10":
            stage08_lost_in_stage10,

        "stage10_pixels_lost_in_stage11":
            stage10_lost_in_stage11,

        "max_visible_rgb_error":
            max_rgb_error,

        "final_rgb_source":
            "locked Stage01 resized master",
    },

    "checks":
        checks,

    "failed_checks":
        failed,

    "important_final_mask":
        str(
            STAGE11_MASK
        ),

    "canonical_final_rgba":
        str(
            RGBA_PATH
        ),
}


REPORT_JSON = (
    OUT
    / "ROOM02_REFERENCE_VERIFICATION.json"
)

REPORT_JSON.write_text(
    json.dumps(
        report,
        indent=2
    )
)


# ============================================================
# SAVE TXT REPORT
# ============================================================

txt = []

txt.append(
    "=" * 100
)

txt.append(
    "TEST07 ROOM02 FINAL REFERENCE VERIFICATION"
)

txt.append(
    "=" * 100
)

txt.append("")

txt.append(
    f'RESULT: {report["result"]}'
)

txt.append("")

txt.append(
    f"LOCKED MASTER: {W} x {H}"
)

txt.append("")

txt.append(
    f"STAGE08 TRUSTED PIXELS: {stage08_pixels}"
)

txt.append(
    f"STAGE10 FINAL PIXELS:   {stage10_pixels}"
)

txt.append(
    f"STAGE10 RECOVERY:       +{stage10_added}"
)

txt.append(
    f"STAGE11 FINAL PIXELS:   {stage11_pixels}"
)

txt.append(
    f"STAGE11 RECOVERY:       +{stage11_added}"
)

txt.append("")

txt.append(
    "STAGE11 TARGETS: "
    +
    str(
        expected_stage11_targets
    )
)

txt.append(
    "STAGE11 ACCEPTED: "
    +
    str(
        stage11_accepted_ids
    )
)

txt.append(
    "REMAINING UNRESOLVED: "
    +
    str(
        stage11_remaining
    )
)

txt.append("")

txt.append(
    "RESOLVED BY EXISTING BASE: [27]"
)

txt.append(
    "SAME-INSTANCE ALIASES:"
)

txt.append(
    "  07 -> 03"
)

txt.append(
    "  58 -> 27"
)

txt.append("")

txt.append(
    "ADDITIVE MASK CHECK:"
)

txt.append(
    f"  Stage08 pixels lost in Stage10: "
    f"{stage08_lost_in_stage10}"
)

txt.append(
    f"  Stage10 pixels lost in Stage11: "
    f"{stage10_lost_in_stage11}"
)

txt.append("")

txt.append(
    "EXACT RGB CHECK:"
)

txt.append(
    f"  Maximum visible RGB error: "
    f"{max_rgb_error}"
)

txt.append("")

txt.append(
    "CHECKS:"
)

for key, passed in checks.items():

    txt.append(
        f'  {"PASS" if passed else "FAIL"}  {key}'
    )

txt.append("")

if failed:

    txt.append(
        "FAILED CHECKS:"
    )

    for key in failed:

        txt.append(
            f"  - {key}"
        )

else:

    txt.append(
        "ALL FINAL REFERENCE CHECKS PASSED."
    )

txt.append("")

txt.append(
    f"FINAL MASK:"
)

txt.append(
    str(
        STAGE11_MASK
    )
)

txt.append("")

txt.append(
    "CANONICAL FINAL RGBA:"
)

txt.append(
    str(
        RGBA_PATH
    )
)


REPORT_TXT = (
    OUT
    / "ROOM02_REFERENCE_VERIFICATION.txt"
)

REPORT_TXT.write_text(
    "\n".join(txt),
    encoding="utf-8"
)


# ============================================================
# PRINT
# ============================================================

print()
print("=" * 100)

print(
    "ROOM02 RESULT:",
    report["result"]
)

print("=" * 100)

print()

print(
    "MASTER SIZE:",
    W,
    "x",
    H
)

print()

print(
    "STAGE08 PIXELS:",
    stage08_pixels
)

print(
    "STAGE10 PIXELS:",
    stage10_pixels,
    "| added:",
    stage10_added
)

print(
    "STAGE11 PIXELS:",
    stage11_pixels,
    "| added:",
    stage11_added
)

print()

print(
    "STAGE11 ACCEPTED IDS:",
    stage11_accepted_ids
)

print(
    "REMAINING UNRESOLVED:",
    stage11_remaining
)

print()

print(
    "ALIASES:",
    {
        3: [7],
        27: [58]
    }
)

print(
    "RESOLVED BY BASE:",
    [27]
)

print()

print(
    "STAGE08 → STAGE10 LOST PIXELS:",
    stage08_lost_in_stage10
)

print(
    "STAGE10 → STAGE11 LOST PIXELS:",
    stage10_lost_in_stage11
)

print(
    "MAX RGB ERROR:",
    max_rgb_error
)

print()

for key, passed in checks.items():

    print(
        "✅" if passed else "❌",
        key
    )

print()

print(
    "REPORT:",
    REPORT_TXT
)

print(
    "JSON:",
    REPORT_JSON
)

print(
    "FINAL RGBA:",
    RGBA_PATH
)
