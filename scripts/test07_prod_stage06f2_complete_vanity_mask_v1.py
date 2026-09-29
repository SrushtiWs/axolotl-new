
from pathlib import Path
import json
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

STAGE06F1_STATE = (
    PROD
    / "stage06_prop_layer"
    / "06f1_six_main_prop_state"
    / "00_stage06f1_state.json"
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
    / "06f2_complete_vanity_mask"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD INPUTS
# ============================================================

for p in [
    STAGE05F,
    STAGE06F1_STATE,
    D2B3_STATE,
    SAM2_CHECKPOINT,
]:

    if not p.exists():
        raise FileNotFoundError(p)


image = Image.open(
    STAGE05F
).convert("RGB")

image_np = np.asarray(
    image
)

H, W = image_np.shape[:2]


state06f1 = json.loads(
    STAGE06F1_STATE.read_text(
        encoding="utf-8"
    )
)

d2b3 = json.loads(
    D2B3_STATE.read_text(
        encoding="utf-8"
    )
)


P01_BBOX = (
    state06f1["main_props"]["P01"]["bbox"]
)

x1, y1, x2, y2 = map(
    float,
    P01_BBOX
)


print("=" * 110)
print("STAGE 06F2 — COMPLETE VANITY MASK")
print("=" * 110)

print()
print(
    "P01 BBOX:",
    P01_BBOX
)

print(
    "IMAGE SIZE:",
    W,
    "x",
    H
)


# ============================================================
# FIND P01 COMPLETION EVIDENCE FROM 06D2B3
# ============================================================

def walk(obj):

    if isinstance(obj, dict):

        yield obj

        for v in obj.values():
            yield from walk(v)

    elif isinstance(obj, list):

        for item in obj:
            yield from walk(item)


p01_row = None

for row in walk(
    d2b3
):

    if (
        isinstance(row, dict)
        and
        row.get("prop_id") == "P01"
    ):
        p01_row = row
        break


if p01_row is None:

    raise RuntimeError(
        "Could not find P01 row in 06D2B3 state."
    )


evidence_rows = (
    p01_row.get(
        "candidate_member_evidence",
        []
    )
)


accepted = []

for row in evidence_rows:

    if row.get(
        "accepted_for_completion"
    ) is True:

        accepted.append(
            row
        )


print()
print(
    "ACCEPTED P01 EVIDENCE:",
    len(accepted)
)


# ============================================================
# BUILD COMPONENT CENTERS
#
# These are INTERNAL prompt locations only.
# They do NOT become final child layers.
# ============================================================

preferred_names = [
    "sink",
    "basin",
    "sink faucet",
    "faucet",
    "bottle",
    "countertop",
    "drawer handle",
    "door handle",
    "cabinet",
]


positive_points = []


def bbox_center(
    bbox
):

    bx1, by1, bx2, by2 = map(
        float,
        bbox
    )

    return [
        (bx1 + bx2) / 2.0,
        (by1 + by2) / 2.0
    ]


# pick smallest accepted region per semantic component
for name in preferred_names:

    matches = []

    for row in accepted:

        ev_name = str(
            row.get(
                "evidence_name",
                row.get(
                    "name",
                    ""
                )
            )
        ).lower()

        bbox = (
            row.get("bbox")
            or
            row.get("box")
        )

        if bbox is None:
            continue

        if name in ev_name:

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
                    bbox,
                    ev_name
                )
            )


    if matches:

        matches.sort(
            key=lambda x: x[0]
        )

        _, bbox, ev_name = matches[0]

        point = bbox_center(
            bbox
        )

        positive_points.append(
            {
                "label":
                    name,

                "source":
                    ev_name,

                "point":
                    point
            }
        )


# ============================================================
# ADD STRONG MANUAL SYSTEM-LEVEL SUPPORT POINTS
#
# Still generic relative to P01 bbox.
# No hardcoded absolute scene coordinates.
# ============================================================

bw = x2 - x1
bh = y2 - y1


system_points = [

    (
        "upper_vanity_body",
        [
            x1 + 0.46 * bw,
            y1 + 0.44 * bh
        ]
    ),

    (
        "middle_vanity_body",
        [
            x1 + 0.50 * bw,
            y1 + 0.61 * bh
        ]
    ),

    (
        "lower_vanity_body",
        [
            x1 + 0.50 * bw,
            y1 + 0.80 * bh
        ]
    ),

    (
        "left_vanity_body",
        [
            x1 + 0.22 * bw,
            y1 + 0.63 * bh
        ]
    ),

    (
        "right_vanity_body",
        [
            x1 + 0.76 * bw,
            y1 + 0.63 * bh
        ]
    ),
]


for name, point in system_points:

    positive_points.append(
        {
            "label":
                name,

            "source":
                "SYSTEM_RELATIVE_POINT",

            "point":
                point
        }
    )


# ============================================================
# PREVIEW PROMPT POINTS
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
    linewidth=2
)

ax.add_patch(
    rect
)


for i, item in enumerate(
    positive_points,
    start=1
):

    px, py = item["point"]

    ax.scatter(
        px,
        py,
        s=45
    )

    ax.text(
        px + 2,
        py,
        str(i),
        fontsize=8
    )


ax.set_title(
    "06F2 — P01 VANITY POSITIVE PROMPT POINTS"
)

ax.axis(
    "off"
)


POINT_PREVIEW = (
    OUT
    / "01_p01_prompt_points.png"
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
# PASS 1 — WHOLE SYSTEM MULTIPOINT
# ============================================================

point_coords = np.array(
    [
        item["point"]
        for item in positive_points
    ],
    dtype=np.float32
)

point_labels = np.ones(
    len(
        positive_points
    ),
    dtype=np.int32
)


box = np.array(
    P01_BBOX,
    dtype=np.float32
)


masks, scores, logits = predictor.predict(
    point_coords=point_coords,
    point_labels=point_labels,
    box=box,
    multimask_output=True
)


# ============================================================
# SAVE ALL 3 CANDIDATES
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
        linewidth=2
    )

    ax.add_patch(
        rect
    )


    for j, item in enumerate(
        positive_points,
        start=1
    ):

        px, py = item["point"]

        ax.scatter(
            px,
            py,
            s=35
        )

        ax.text(
            px + 2,
            py,
            str(j),
            fontsize=7
        )


    ax.set_title(
        (
            f"P01 COMPLETE VANITY — CANDIDATE {i}\n"
            f"SAM={float(score):.4f} | AREA={area}"
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
                str(
                    mask_path
                ),

            "preview":
                str(
                    preview_path
                )
        }
    )


# ============================================================
# SAVE RESULT
# ============================================================

STATE = {

    "stage":
        "06F2",

    "target":
        "P01_COMPLETE_VANITY_SYSTEM",

    "input":
        str(
            STAGE05F
        ),

    "bbox":
        P01_BBOX,

    "positive_points":
        positive_points,

    "candidate_masks":
        candidate_records,

    "final_selection":
        None,

    "status":
        "REQUIRES_VISUAL_CANDIDATE_AUDIT",

    "important_rule":
        (
            "Internal component positions were used only as "
            "prompt evidence. No child item becomes a final "
            "independent prop layer."
        )
}


STATE_PATH = (
    OUT
    / "00_stage06f2_result.json"
)


STATE_PATH.write_text(
    json.dumps(
        STATE,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


print()
print("=" * 110)
print("STAGE 06F2 RESULT")
print("=" * 110)

print()
print(
    "TARGET:",
    "P01 COMPLETE VANITY SYSTEM"
)

print(
    "PROMPT POINTS:",
    len(
        positive_points
    )
)

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
    "STATE:",
    STATE_PATH
)

print(
    "PROMPT PREVIEW:",
    POINT_PREVIEW
)
