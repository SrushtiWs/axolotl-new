
from pathlib import Path
import json
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw


# ============================================================
# SAM2 IMPORT
# ============================================================

for candidate in [
    Path("/workspace/sam2_src"),
    Path("/workspace/axolotl/sam2"),
]:

    if (
        candidate.exists()
        and str(candidate) not in sys.path
    ):

        sys.path.insert(
            0,
            str(candidate)
        )


from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# ============================================================
# PATHS
# ============================================================

BASE = Path("/workspace/axolotl")

PROD = (
    BASE
    / "test07"
    / "production_pipeline"
)

STAGE06 = (
    PROD
    / "stage06_prop_layer"
)

MASTER = (
    PROD
    / "stage01_master"
    / "00_master_input.png"
)

CONSENSUS = (
    STAGE06
    / "06c6_multi_route_candidate_consensus"
    / "00_multi_route_consensus_all.json"
)

CHECKPOINT = (
    BASE
    / "test07"
    / "models"
    / "sam2"
    / "sam2.1_hiera_large.pt"
)

OUT = (
    STAGE06
    / "06c7b_faucet_box_expansion"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


SAM2_CONFIG = (
    "configs/sam2.1/"
    "sam2.1_hiera_l.yaml"
)


# ============================================================
# EXPANSIONS
# ============================================================

EXPANSIONS = [
    0.00,
    0.25,
    0.50,
    1.00,
    1.50,
]


# ============================================================
# LOAD
# ============================================================

for p in [
    MASTER,
    CONSENSUS,
    CHECKPOINT,
]:

    if not p.exists():
        raise FileNotFoundError(p)


master_pil = Image.open(
    MASTER
).convert("RGB")

master = np.asarray(
    master_pil
)

W, H = master_pil.size


data = json.loads(
    CONSENSUS.read_text(
        encoding="utf-8"
    )
)


instance = next(
    row
    for row in data
    if int(row["inventory_id"]) == 10
)


base_box = [
    float(v)
    for v in instance[
        "clusters"
    ][0][
        "bbox"
    ]
]


print("=" * 100)
print("06C7B FAUCET BOX EXPANSION")
print("=" * 100)

print(
    "ORIGINAL:",
    [
        round(v, 2)
        for v in base_box
    ]
)


# ============================================================
# SAM2
# ============================================================

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


model = build_sam2(
    SAM2_CONFIG,
    str(CHECKPOINT),
    device=device
)


predictor = SAM2ImagePredictor(
    model
)

predictor.set_image(
    master
)


# ============================================================
# HELPERS
# ============================================================

def expand_box(
    box,
    ratio
):

    x1, y1, x2, y2 = box

    cx = (
        x1 + x2
    ) / 2

    cy = (
        y1 + y2
    ) / 2

    bw = max(
        2.0,
        x2 - x1
    )

    bh = max(
        2.0,
        y2 - y1
    )


    new_w = (
        bw
        *
        (
            1.0
            +
            2.0 * ratio
        )
    )

    new_h = (
        bh
        *
        (
            1.0
            +
            2.0 * ratio
        )
    )


    return np.asarray(
        [
            max(
                0,
                cx - new_w / 2
            ),
            max(
                0,
                cy - new_h / 2
            ),
            min(
                W,
                cx + new_w / 2
            ),
            min(
                H,
                cy + new_h / 2
            ),
        ],
        dtype=np.float32
    )


def mask_outline(mask):

    interior = (
        np.roll(mask, 1, 0)
        &
        np.roll(mask, -1, 0)
        &
        np.roll(mask, 1, 1)
        &
        np.roll(mask, -1, 1)
    )

    return (
        mask
        &
        ~interior
    )


# ============================================================
# TEST
# ============================================================

results = []


for ratio in EXPANSIONS:

    box = expand_box(
        base_box,
        ratio
    )


    with torch.inference_mode():

        masks, scores, _ = (
            predictor.predict(
                box=box,
                multimask_output=True
            )
        )


    order = np.argsort(
        scores
    )[::-1]


    best_idx = int(
        order[0]
    )

    mask = (
        masks[
            best_idx
        ]
        >
        0
    )


    score = float(
        scores[
            best_idx
        ]
    )

    area = int(
        mask.sum()
    )


    label = int(
        round(
            ratio * 100
        )
    )


    # ========================================================
    # MASK
    # ========================================================

    mask_path = (
        OUT
        /
        f"010_faucet_expand_{label:03d}_mask.png"
    )


    Image.fromarray(
        mask.astype(
            np.uint8
        )
        *
        255
    ).save(
        mask_path
    )


    # ========================================================
    # PREVIEW
    # ========================================================

    arr = master.copy()

    outline = mask_outline(
        mask
    )

    arr[
        outline
    ] = [
        255,
        0,
        0
    ]


    preview = Image.fromarray(
        arr
    )

    draw = ImageDraw.Draw(
        preview
    )


    x1, y1, x2, y2 = [
        int(
            round(
                float(v)
            )
        )
        for v in box
    ]


    draw.rectangle(
        [
            x1,
            y1,
            x2,
            y2
        ],
        outline="red",
        width=2
    )


    preview_path = (
        OUT
        /
        f"010_faucet_expand_{label:03d}_preview.png"
    )


    preview.save(
        preview_path
    )


    results.append({

        "expansion_ratio":
            ratio,

        "bbox":
            [
                float(v)
                for v in box
            ],

        "sam2_score":
            score,

        "mask_area":
            area,

        "mask_path":
            str(mask_path),

        "preview_path":
            str(preview_path),
    })


    print(
        "expand={:>4}% | SAM={:.4f} | area={} | bbox={}".format(

            label,

            score,

            area,

            [
                round(
                    float(v),
                    1
                )
                for v in box
            ],
        )
    )


# ============================================================
# SAVE
# ============================================================

RESULT_PATH = (
    OUT
    / "00_expansion_results.json"
)


RESULT_PATH.write_text(
    json.dumps(
        results,
        indent=2
    ),
    encoding="utf-8"
)


print()
print(
    "OUTPUT:",
    OUT
)

print(
    "RESULT JSON:",
    RESULT_PATH
)

print()
print(
    "DIAGNOSTIC ONLY."
)
