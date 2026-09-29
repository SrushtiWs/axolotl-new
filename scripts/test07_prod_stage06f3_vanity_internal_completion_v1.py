
from pathlib import Path
import json
import gc

import numpy as np
import torch

from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.patches as patches


# ============================================================
# PATHS
# ============================================================

BASE = Path("/workspace/axolotl")

PROD = (
    BASE
    / "test07"
    / "production_pipeline"
)

STAGE05F = (
    PROD
    / "stage05_clean_room_with_props"
    / "12_final_canonical_prop_detection_master.png"
)

F2_STATE = (
    PROD
    / "stage06_prop_layer"
    / "06f2_complete_vanity_mask_v2"
    / "00_stage06f2_v2_result.json"
)

D2B3_STATE = (
    PROD
    / "stage06_prop_layer"
    / "06d2b3_final_verified_main_prop_localization"
    / "00_stage06d2b3_result.json"
)

SAM2_CHECKPOINT = (
    BASE
    / "test07"
    / "models"
    / "sam2"
    / "sam2.1_hiera_large.pt"
)

SAM2_CONFIG = (
    "configs/sam2.1/sam2.1_hiera_l.yaml"
)

OUT = (
    PROD
    / "stage06_prop_layer"
    / "06f3_vanity_internal_completion"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# VALIDATE
# ============================================================

for p in [
    STAGE05F,
    F2_STATE,
    D2B3_STATE,
    SAM2_CHECKPOINT,
]:

    if not p.exists():
        raise FileNotFoundError(p)


# ============================================================
# LOAD
# ============================================================

image = Image.open(
    STAGE05F
).convert("RGB")

image_np = np.asarray(image)

H, W = image_np.shape[:2]


f2 = json.loads(
    F2_STATE.read_text(
        encoding="utf-8"
    )
)

d2b3 = json.loads(
    D2B3_STATE.read_text(
        encoding="utf-8"
    )
)


# ============================================================
# BASE = VISUALLY SELECTED CANDIDATE #3
# ============================================================

BASE_CANDIDATE = 3


candidate_row = next(
    row
    for row in f2["candidate_masks"]
    if row["candidate"] == BASE_CANDIDATE
)


base_mask_path = Path(
    candidate_row["mask"]
)


if not base_mask_path.exists():
    raise FileNotFoundError(
        base_mask_path
    )


base_mask = (
    np.asarray(
        Image.open(
            base_mask_path
        ).convert("L")
    )
    > 127
)


# ============================================================
# P01 VERIFIED RECORD
# ============================================================

p01 = next(
    row
    for row in d2b3["results"]
    if row.get("main_prop_id") == "P01"
)


P01_BBOX = p01["final_bbox"]

x1, y1, x2, y2 = map(
    float,
    P01_BBOX
)


# ============================================================
# ACCEPTED COMPLETION EVIDENCE
# ============================================================

accepted = [
    row
    for row in p01[
        "candidate_member_evidence"
    ]
    if row.get(
        "accepted_for_completion"
    ) is True
]


# We only need internal completion around the upper assembly.
TARGET_TERMS = [
    "sink",
    "basin",
    "faucet",
    "bottle",
    "countertop",
]


completion_evidence = []


for row in accepted:

    name = str(
        row.get(
            "name",
            ""
        )
    ).lower()

    if any(
        term in name
        for term in TARGET_TERMS
    ):
        completion_evidence.append(
            row
        )


print("=" * 110)
print("STAGE 06F3 — P01 VANITY INTERNAL COMPLETION")
print("=" * 110)

print()
print(
    "BASE CANDIDATE:",
    BASE_CANDIDATE
)

print(
    "BASE AREA:",
    int(
        base_mask.sum()
    )
)

print(
    "COMPLETION EVIDENCE:",
    len(
        completion_evidence
    )
)


for row in completion_evidence:

    print(
        "✅",
        row["evidence_id"],
        "|",
        row["name"],
        "|",
        row["bbox"]
    )


# ============================================================
# LOAD SAM2
# ============================================================

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


print()
print("Loading SAM2...")


sam2_model = build_sam2(
    SAM2_CONFIG,
    str(
        SAM2_CHECKPOINT
    ),
    device="cuda"
)


predictor = SAM2ImagePredictor(
    sam2_model
)


predictor.set_image(
    image_np
)


# ============================================================
# P01 SPATIAL ALLOWANCE MASK
#
# Prevent internal completion from escaping far outside vanity.
# ============================================================

PAD = 8


px1 = max(
    0,
    int(
        np.floor(
            x1 - PAD
        )
    )
)

py1 = max(
    0,
    int(
        np.floor(
            y1 - PAD
        )
    )
)

px2 = min(
    W,
    int(
        np.ceil(
            x2 + PAD
        )
    )
)

py2 = min(
    H,
    int(
        np.ceil(
            y2 + PAD
        )
    )
)


allowed_region = np.zeros(
    (
        H,
        W
    ),
    dtype=bool
)


allowed_region[
    py1:py2,
    px1:px2
] = True


# ============================================================
# INTERNAL COMPLETION
# ============================================================

union_mask = base_mask.copy()

completion_records = []


for idx, row in enumerate(
    completion_evidence,
    start=1
):

    bbox = np.asarray(
        row["bbox"],
        dtype=np.float32
    )


    bx1, by1, bx2, by2 = bbox


    center = np.asarray(
        [
            [
                (
                    bx1 + bx2
                )
                /
                2.0,

                (
                    by1 + by2
                )
                /
                2.0
            ]
        ],
        dtype=np.float32
    )


    labels = np.asarray(
        [1],
        dtype=np.int32
    )


    masks, scores, logits = predictor.predict(
        point_coords=center,
        point_labels=labels,
        box=bbox,
        multimask_output=True
    )


    # --------------------------------------------------------
    # Choose candidate by:
    #
    # 1. high SAM score
    # 2. must stay mostly within P01 allowed region
    # 3. prefer reasonable/localized component mask
    # --------------------------------------------------------

    evaluated = []


    for candidate_index, (
        mask,
        score
    ) in enumerate(
        zip(
            masks,
            scores
        ),
        start=1
    ):

        mask_bool = mask.astype(
            bool
        )


        total_area = int(
            mask_bool.sum()
        )


        if total_area == 0:
            continue


        inside = int(
            (
                mask_bool
                &
                allowed_region
            ).sum()
        )


        inside_fraction = (
            inside
            /
            total_area
        )


        # overlap with current vanity union
        overlap = int(
            (
                mask_bool
                &
                union_mask
            ).sum()
        )


        # small nearby components may not yet overlap,
        # therefore overlap is diagnostic rather than mandatory.
        rank_score = (
            float(score)
            +
            0.20
            *
            inside_fraction
        )


        evaluated.append(
            {
                "candidate_index":
                    candidate_index,

                "mask":
                    mask_bool,

                "sam_score":
                    float(score),

                "area":
                    total_area,

                "inside_fraction":
                    float(
                        inside_fraction
                    ),

                "overlap_with_current":
                    overlap,

                "rank_score":
                    float(
                        rank_score
                    )
            }
        )


    if not evaluated:
        continue


    # reject obvious escape masks
    viable = [
        e
        for e in evaluated
        if e["inside_fraction"] >= 0.85
    ]


    if not viable:
        viable = evaluated


    viable.sort(
        key=lambda e: e[
            "rank_score"
        ],
        reverse=True
    )


    chosen = viable[0]


    chosen_mask = (
        chosen["mask"]
        &
        allowed_region
    )


    before_area = int(
        union_mask.sum()
    )


    union_mask |= chosen_mask


    after_area = int(
        union_mask.sum()
    )


    added_pixels = (
        after_area
        -
        before_area
    )


    completion_records.append(
        {
            "evidence_id":
                row["evidence_id"],

            "name":
                row["name"],

            "bbox":
                row["bbox"],

            "selected_candidate":
                chosen[
                    "candidate_index"
                ],

            "sam_score":
                chosen[
                    "sam_score"
                ],

            "inside_fraction":
                chosen[
                    "inside_fraction"
                ],

            "overlap_with_current":
                chosen[
                    "overlap_with_current"
                ],

            "added_pixels":
                added_pixels
        }
    )


# ============================================================
# SAVE COMPLETED MASK
# ============================================================

FINAL_MASK_PATH = (
    OUT
    / "01_p01_completed_vanity_mask.png"
)


Image.fromarray(
    union_mask.astype(
        np.uint8
    )
    *
    255
).save(
    FINAL_MASK_PATH
)


# ============================================================
# TRANSPARENT P01 PREVIEW
#
# Stage05F RGB only for visual audit.
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
    ...,
    :3
] = image_np


rgba[
    ...,
    3
] = (
    union_mask.astype(
        np.uint8
    )
    *
    255
)


RGBA_PATH = (
    OUT
    / "02_p01_completed_vanity_transparent.png"
)


Image.fromarray(
    rgba,
    mode="RGBA"
).save(
    RGBA_PATH
)


# ============================================================
# VISUAL COMPARISON
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 8)
)


# -------------------------
# Base
# -------------------------

axes[0].imshow(
    image
)


overlay_base = np.zeros(
    (
        H,
        W,
        4
    ),
    dtype=np.float32
)


overlay_base[
    base_mask,
    1
] = 1.0

overlay_base[
    base_mask,
    3
] = 0.45


axes[0].imshow(
    overlay_base
)

axes[0].set_title(
    "1. 06F2 Candidate #3 Base"
)

axes[0].axis(
    "off"
)


# -------------------------
# Completed
# -------------------------

axes[1].imshow(
    image
)


overlay_final = np.zeros(
    (
        H,
        W,
        4
    ),
    dtype=np.float32
)


overlay_final[
    union_mask,
    1
] = 1.0

overlay_final[
    union_mask,
    3
] = 0.45


axes[1].imshow(
    overlay_final
)

axes[1].set_title(
    "2. 06F3 Completed Vanity Mask"
)

axes[1].axis(
    "off"
)


# -------------------------
# Newly added pixels
# -------------------------

added_mask = (
    union_mask
    &
    ~base_mask
)


axes[2].imshow(
    image
)


overlay_added = np.zeros(
    (
        H,
        W,
        4
    ),
    dtype=np.float32
)


overlay_added[
    added_mask,
    0
] = 1.0

overlay_added[
    added_mask,
    3
] = 0.65


axes[2].imshow(
    overlay_added
)

axes[2].set_title(
    "3. Newly Added Completion Pixels"
)

axes[2].axis(
    "off"
)


plt.tight_layout()


COMPARISON_PATH = (
    OUT
    / "03_p01_completion_comparison.png"
)


plt.savefig(
    COMPARISON_PATH,
    dpi=150,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# STATE
# ============================================================

STATE = {

    "stage":
        "06F3",

    "target":
        "P01_COMPLETE_VANITY_SYSTEM",

    "base_candidate":
        BASE_CANDIDATE,

    "base_area":
        int(
            base_mask.sum()
        ),

    "completed_area":
        int(
            union_mask.sum()
        ),

    "added_pixels":
        int(
            (
                union_mask
                &
                ~base_mask
            ).sum()
        ),

    "internal_completion":
        completion_records,

    "final_mask":
        str(
            FINAL_MASK_PATH
        ),

    "transparent_preview":
        str(
            RGBA_PATH
        ),

    "comparison":
        str(
            COMPARISON_PATH
        ),

    "status":
        "REQUIRES_VISUAL_FINAL_P01_AUDIT",

    "production_rule":
        (
            "All component segmentation is internal only. "
            "The output remains one single P01 vanity-system mask."
        )
}


STATE_PATH = (
    OUT
    / "00_stage06f3_result.json"
)


STATE_PATH.write_text(
    json.dumps(
        STATE,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# ============================================================
# PRINT
# ============================================================

print()
print("=" * 110)
print("STAGE 06F3 RESULT")
print("=" * 110)

print()

print(
    "BASE AREA:",
    int(
        base_mask.sum()
    )
)

print(
    "COMPLETED AREA:",
    int(
        union_mask.sum()
    )
)

print(
    "TOTAL ADDED PIXELS:",
    int(
        (
            union_mask
            &
            ~base_mask
        ).sum()
    )
)


print()
print(
    "INTERNAL COMPLETION RESULTS:"
)


for row in completion_records:

    print(
        row["evidence_id"],
        "|",
        row["name"],
        "| candidate",
        row["selected_candidate"],
        "| SAM",
        round(
            row["sam_score"],
            4
        ),
        "| added",
        row["added_pixels"]
    )


print()
print(
    "FINAL MASK:",
    FINAL_MASK_PATH
)

print(
    "TRANSPARENT:",
    RGBA_PATH
)

print(
    "COMPARISON:",
    COMPARISON_PATH
)

print(
    "STATE:",
    STATE_PATH
)


# ============================================================
# CLEANUP
# ============================================================

del predictor
del sam2_model

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()
