
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
    / "06f2_complete_vanity_mask_v2"
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

image_np = np.asarray(
    image
)

H, W = image_np.shape[:2]


d2b3 = json.loads(
    D2B3_STATE.read_text(
        encoding="utf-8"
    )
)


# ============================================================
# EXACT VERIFIED P01 RECORD
# ============================================================

if "results" not in d2b3:

    raise RuntimeError(
        "06D2B3 has no 'results' field."
    )


p01_row = next(
    (
        row
        for row in d2b3["results"]
        if row.get("main_prop_id") == "P01"
    ),
    None
)


if p01_row is None:

    raise RuntimeError(
        "P01 not found in 06D2B3 results."
    )


P01_BBOX = p01_row["final_bbox"]

x1, y1, x2, y2 = map(
    float,
    P01_BBOX
)


print("=" * 110)
print("STAGE 06F2 V2 — COMPLETE VANITY MASK")
print("=" * 110)

print()
print(
    "P01:",
    p01_row["main_prop_name"]
)

print(
    "SOURCE:",
    p01_row["source"]
)

print(
    "FINAL BBOX:",
    P01_BBOX
)


# ============================================================
# ACCEPTED MEMBER EVIDENCE
# ============================================================

evidence_rows = p01_row.get(
    "candidate_member_evidence",
    []
)


accepted = [
    row
    for row in evidence_rows
    if row.get("accepted_for_completion") is True
]


print()
print(
    "TOTAL MEMBER EVIDENCE:",
    len(evidence_rows)
)

print(
    "ACCEPTED FOR COMPLETION:",
    len(accepted)
)


for row in accepted:

    print(
        "✅",
        row.get("evidence_id"),
        "|",
        row.get("name"),
        "|",
        row.get("bbox")
    )


# ============================================================
# BUILD SEMANTIC POSITIVE POINTS
#
# IMPORTANT:
# component locations are prompt evidence only.
# No child output layers are created.
# ============================================================

def center_of_bbox(bbox):

    bx1, by1, bx2, by2 = map(
        float,
        bbox
    )

    return [
        (bx1 + bx2) / 2.0,
        (by1 + by2) / 2.0
    ]


# Use only accepted P01 evidence.
#
# For duplicate semantic evidence,
# choose the SMALLEST accepted box because it usually gives
# the most localized positive prompt.
wanted_components = [
    "sink",
    "countertop",
    "bottle",
    "faucet",
    "sink faucet",
    "drawer handle",
    "door handle",
]


positive_points = []


for wanted in wanted_components:

    matches = []

    for row in accepted:

        name = str(
            row.get(
                "name",
                ""
            )
        ).strip().lower()

        bbox = row.get(
            "bbox"
        )

        if bbox is None:
            continue

        if (
            name == wanted
            or
            wanted in name
        ):

            bx1, by1, bx2, by2 = map(
                float,
                bbox
            )

            area = (
                max(
                    0.0,
                    bx2 - bx1
                )
                *
                max(
                    0.0,
                    by2 - by1
                )
            )

            matches.append(
                (
                    area,
                    row
                )
            )


    if matches:

        matches.sort(
            key=lambda x: x[0]
        )

        row = matches[0][1]

        positive_points.append(
            {
                "label":
                    row["name"],

                "source_id":
                    row["evidence_id"],

                "point":
                    center_of_bbox(
                        row["bbox"]
                    )
            }
        )


# ============================================================
# SYSTEM-LEVEL VANITY BODY SUPPORT
#
# These ensure SAM2 receives evidence across the complete
# cabinet/body instead of only upper sink components.
# ============================================================

bw = x2 - x1
bh = y2 - y1


system_points = [

    {
        "label":
            "vanity_upper_body",

        "source_id":
            "SYSTEM_RELATIVE",

        "point": [
            x1 + 0.46 * bw,
            y1 + 0.43 * bh
        ]
    },

    {
        "label":
            "vanity_middle_body",

        "source_id":
            "SYSTEM_RELATIVE",

        "point": [
            x1 + 0.48 * bw,
            y1 + 0.61 * bh
        ]
    },

    {
        "label":
            "vanity_lower_body",

        "source_id":
            "SYSTEM_RELATIVE",

        "point": [
            x1 + 0.48 * bw,
            y1 + 0.80 * bh
        ]
    },

    {
        "label":
            "vanity_left_body",

        "source_id":
            "SYSTEM_RELATIVE",

        "point": [
            x1 + 0.22 * bw,
            y1 + 0.62 * bh
        ]
    },

    {
        "label":
            "vanity_right_body",

        "source_id":
            "SYSTEM_RELATIVE",

        "point": [
            x1 + 0.75 * bw,
            y1 + 0.62 * bh
        ]
    },
]


positive_points.extend(
    system_points
)


print()
print(
    "TOTAL POSITIVE POINTS:",
    len(positive_points)
)


for i, row in enumerate(
    positive_points,
    start=1
):

    print(
        i,
        "|",
        row["label"],
        "|",
        row["source_id"],
        "|",
        [
            round(
                v,
                2
            )
            for v in row["point"]
        ]
    )


# ============================================================
# PROMPT POINT PREVIEW
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 10)
)

ax.imshow(
    image
)


rect = patches.Rectangle(
    (x1, y1),
    x2 - x1,
    y2 - y1,
    fill=False,
    linewidth=2,
    edgecolor="yellow"
)

ax.add_patch(
    rect
)


for i, row in enumerate(
    positive_points,
    start=1
):

    px, py = row["point"]

    ax.scatter(
        px,
        py,
        s=55,
        c="lime",
        edgecolors="black"
    )

    ax.text(
        px + 2,
        py,
        str(i),
        color="red",
        fontsize=8,
        fontweight="bold"
    )


ax.set_title(
    "06F2 V2 — P01 VANITY POSITIVE PROMPT POINTS\n"
    "GREEN = positive | YELLOW = verified complete vanity bbox"
)

ax.axis(
    "off"
)


POINT_PREVIEW = (
    OUT
    / "01_p01_positive_point_preview.png"
)


plt.tight_layout()

plt.savefig(
    POINT_PREVIEW,
    dpi=150,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# LOAD SAM2
# ============================================================

print()
print(
    "Loading SAM2..."
)


from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


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
# ONE COMPLETE-SYSTEM SAM2 CALL
#
# No child masks.
# ============================================================

point_coords = np.asarray(
    [
        row["point"]
        for row in positive_points
    ],
    dtype=np.float32
)


point_labels = np.ones(
    len(
        positive_points
    ),
    dtype=np.int32
)


box = np.asarray(
    P01_BBOX,
    dtype=np.float32
)


masks, scores, logits = predictor.predict(
    point_coords=point_coords,
    point_labels=point_labels,
    box=box,
    multimask_output=True
)


print()
print(
    "SAM2 CANDIDATES:",
    len(masks)
)


# ============================================================
# SAVE ALL THREE CANDIDATES
# ============================================================

candidate_records = []


for i, (
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


    area = int(
        mask_bool.sum()
    )


    mask_path = (
        OUT
        / f"02_p01_candidate_{i}_mask.png"
    )


    Image.fromarray(
        mask_bool.astype(
            np.uint8
        )
        *
        255
    ).save(
        mask_path
    )


    # --------------------------------------------------------
    # Preview
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(8, 10)
    )

    ax.imshow(
        image
    )


    overlay = np.zeros(
        (
            H,
            W,
            4
        ),
        dtype=np.float32
    )

    overlay[
        mask_bool,
        1
    ] = 1.0

    overlay[
        mask_bool,
        3
    ] = 0.45


    ax.imshow(
        overlay
    )


    rect = patches.Rectangle(
        (x1, y1),
        x2 - x1,
        y2 - y1,
        fill=False,
        linewidth=2,
        edgecolor="yellow"
    )

    ax.add_patch(
        rect
    )


    for j, row in enumerate(
        positive_points,
        start=1
    ):

        px, py = row["point"]

        ax.scatter(
            px,
            py,
            s=35,
            c="red"
        )

        ax.text(
            px + 2,
            py,
            str(j),
            fontsize=7,
            color="red"
        )


    ax.set_title(
        (
            f"P01 COMPLETE VANITY SYSTEM\n"
            f"CANDIDATE #{i} | "
            f"SAM={float(score):.4f} | "
            f"AREA={area}"
        )
    )

    ax.axis(
        "off"
    )


    preview_path = (
        OUT
        / f"03_p01_candidate_{i}_preview.png"
    )


    plt.tight_layout()

    plt.savefig(
        preview_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.show()


    candidate_records.append(
        {
            "candidate":
                i,

            "sam_score":
                float(score),

            "area":
                area,

            "mask":
                str(mask_path),

            "preview":
                str(preview_path)
        }
    )


# ============================================================
# SAVE STATE
# ============================================================

STATE = {

    "stage":
        "06F2_V2",

    "target":
        "P01_COMPLETE_VANITY_SYSTEM",

    "input":
        str(STAGE05F),

    "source_localization_state":
        str(D2B3_STATE),

    "p01_source":
        p01_row["source"],

    "verified_bbox":
        P01_BBOX,

    "accepted_member_evidence_ids":
        p01_row.get(
            "accepted_member_evidence_ids",
            []
        ),

    "positive_points":
        positive_points,

    "candidate_masks":
        candidate_records,

    "final_selection":
        None,

    "status":
        "REQUIRES_VISUAL_CANDIDATE_AUDIT",

    "production_rule":
        (
            "Component evidence is used only internally to "
            "complete one P01 vanity-system mask. No child "
            "component becomes an independent final prop."
        )
}


STATE_PATH = (
    OUT
    / "00_stage06f2_v2_result.json"
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
print("STAGE 06F2 V2 RESULT")
print("=" * 110)

print()
print(
    "P01:",
    p01_row["main_prop_name"]
)

print(
    "BBOX:",
    P01_BBOX
)

print(
    "ACCEPTED MEMBER EVIDENCE:",
    len(accepted)
)

print(
    "POSITIVE POINTS:",
    len(positive_points)
)

print()


for row in candidate_records:

    print(
        "CANDIDATE",
        row["candidate"],
        "| SAM =",
        round(
            row["sam_score"],
            4
        ),
        "| AREA =",
        row["area"]
    )


print()
print(
    "POINT PREVIEW:",
    POINT_PREVIEW
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
