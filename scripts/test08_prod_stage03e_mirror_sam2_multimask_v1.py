
from pathlib import Path
import json
import sys
import gc
import hashlib

import numpy as np
import torch

from PIL import Image, ImageDraw

from IPython.display import display


# ============================================================
# SAM2 IMPORT
#
# Same proven import strategy used in TEST07.
# ============================================================

for source in [

    Path(
        "/workspace/sam2_src"
    ),

    Path(
        "/workspace/axolotl/sam2"
    ),

]:

    if (
        source.exists()
        and
        str(
            source
        )
        not in sys.path
    ):

        sys.path.insert(
            0,
            str(
                source
            )
        )


from sam2.build_sam import build_sam2

from sam2.sam2_image_predictor import (
    SAM2ImagePredictor
)


# ============================================================
# PATHS
# ============================================================

BASE = Path(
    "/workspace/axolotl"
)

PROD = (
    BASE
    / "test08"
    / "production_pipeline"
)


MASTER_PATH = (
    PROD
    / "stage01_master"
    / "00_master_input.png"
)


DINO_DIR = (
    PROD
    / "stage03_mirror"
    / "03d_dino_mirror_discovery"
)


DINO_JSON = (
    DINO_DIR
    / "01_deduplicated_mirror_candidates.json"
)


OUT = (
    PROD
    / "stage03_mirror"
    / "03e_sam2_mirror_multimask"
)


OBJECTS = (
    OUT
    / "objects"
)


OUT.mkdir(
    parents=True,
    exist_ok=True
)


OBJECTS.mkdir(
    parents=True,
    exist_ok=True
)


SAM2_CHECKPOINT = Path(
    "/workspace/axolotl/test07/models/sam2/"
    "sam2.1_hiera_large.pt"
)


SAM2_CONFIG = (
    "configs/sam2.1/"
    "sam2.1_hiera_l.yaml"
)


# ============================================================
# HELPERS
# ============================================================

def sha256_file(
    path
):

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


def bbox_from_mask(
    mask
):

    ys, xs = np.where(
        mask
    )

    if len(
        xs
    ) == 0:

        return None

    return [

        int(
            xs.min()
        ),

        int(
            ys.min()
        ),

        int(
            xs.max()
        )
        + 1,

        int(
            ys.max()
        )
        + 1,
    ]


def largest_component(
    mask
):

    import cv2

    mask_u8 = (
        mask.astype(
            np.uint8
        )
    )


    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask_u8,
            connectivity=8
        )
    )


    if num_labels <= 1:

        return mask.astype(
            bool
        )


    areas = stats[
        1:,
        cv2.CC_STAT_AREA
    ]


    largest = (
        int(
            np.argmax(
                areas
            )
        )
        + 1
    )


    return (
        labels
        ==
        largest
    )


# ============================================================
# PROTECT EXISTING RESULT
# ============================================================

existing = [

    p
    for p
    in OUT.rglob(
        "*"
    )
    if p.is_file()
]


if existing:

    print(
        "EXISTING STAGE03E FILES:"
    )

    for p in existing:

        print(
            " -",
            p
        )

    raise FileExistsError(
        "TEST08 Stage03E already contains files. "
        "Refusing to overwrite them."
    )


# ============================================================
# VALIDATE INPUTS
# ============================================================

required = [

    MASTER_PATH,
    DINO_JSON,
    SAM2_CHECKPOINT,
]


for path in required:

    print(
        (
            "✅"
            if path.exists()
            else
            "❌"
        ),
        path
    )

    if not path.exists():

        raise FileNotFoundError(
            path
        )


# ============================================================
# LOAD MASTER
# ============================================================

master_pil = Image.open(
    MASTER_PATH
).convert(
    "RGB"
)


master = np.array(
    master_pil
)


H, W = master.shape[
    :2
]


print()
print("=" * 110)
print("TEST08 STAGE03E — MIRROR SAM2 MULTIMASK")
print("=" * 110)

print()
print(
    "MASTER:",
    MASTER_PATH
)

print(
    "SIZE:",
    W,
    "x",
    H
)

print(
    "MASTER SHA256:",
    sha256_file(
        MASTER_PATH
    )
)


# ============================================================
# LOAD DINO CANDIDATES
# ============================================================

candidates = json.loads(
    DINO_JSON.read_text(
        encoding="utf-8"
    )
)


if not candidates:

    raise RuntimeError(
        "Stage03D has no deduplicated mirror candidates."
    )


# ============================================================
# AUTOMATIC CANONICAL BBOX
#
# Highest-scoring deduplicated mirror-family candidate.
#
# From current TEST08 Stage03D this should be:
#   phrase = round mirror
#   score ~0.6534
#
# NO hardcoded coordinates.
# ============================================================

candidates = sorted(
    candidates,
    key=lambda row:
        float(
            row[
                "score"
            ]
        ),
    reverse=True
)


selected = candidates[
    0
]


prompt_box = [

    float(
        v
    )

    for v
    in selected[
        "bbox"
    ]
]


print()
print(
    "AUTO SELECTED DINO CANDIDATE:"
)

print(
    "  phrase:",
    selected.get(
        "phrase"
    )
)

print(
    "  score:",
    round(
        float(
            selected[
                "score"
            ]
        ),
        6
    )
)

print(
    "  bbox:",
    [
        round(
            v,
            2
        )
        for v
        in prompt_box
    ]
)


# ============================================================
# SAFETY ASSERTION
#
# We are deliberately testing Stage03D's strongest result.
# This does NOT encode coordinates.
# ============================================================

if float(
    selected[
        "score"
    ]
) <= 0:

    raise RuntimeError(
        "Selected DINO candidate has invalid score."
    )


# ============================================================
# LOAD SAM2
# ============================================================

device = (
    "cuda"
    if torch.cuda.is_available()
    else
    "cpu"
)


print()
print(
    "DEVICE:",
    device
)

print(
    "Loading SAM2..."
)


sam2_model = build_sam2(

    SAM2_CONFIG,

    str(
        SAM2_CHECKPOINT
    ),

    device=device
)


predictor = SAM2ImagePredictor(
    sam2_model
)


predictor.set_image(
    master
)


print(
    "✅ SAM2 READY"
)


# ============================================================
# SAM2 MULTIMASK
#
# IMPORTANT:
#   Prompt box is used exactly.
#   No expansion.
# ============================================================

with torch.inference_mode():

    masks, scores, logits = (
        predictor.predict(

            box=np.asarray(
                prompt_box,
                dtype=np.float32
            ),

            multimask_output=True
        )
    )


print()
print(
    "SAM2 MASK COUNT:",
    len(
        masks
    )
)


# ============================================================
# SAVE PROMPT PREVIEW
# ============================================================

prompt_preview = (
    master_pil.copy()
)


draw = ImageDraw.Draw(
    prompt_preview
)


x1, y1, x2, y2 = prompt_box


draw.rectangle(
    [
        x1,
        y1,
        x2,
        y2
    ],
    outline="lime",
    width=3
)


draw.text(
    (
        x1 + 3,
        max(
            0,
            y1 - 16
        )
    ),
    (
        f"{selected.get('phrase')} "
        f"{float(selected['score']):.3f}"
    ),
    fill="lime"
)


PROMPT_PREVIEW_PATH = (
    OUT
    / "01_selected_dino_prompt_bbox.png"
)


prompt_preview.save(
    PROMPT_PREVIEW_PATH
)


# ============================================================
# PROCESS EVERY SAM2 CANDIDATE
#
# We only:
#   - threshold mask
#   - keep largest connected component
#
# We DO NOT rank/select final mirror.
# ============================================================

rows = []


for idx in range(
    len(
        masks
    )
):

    raw_mask = (
        masks[
            idx
        ]
        >
        0
    )


    cleaned = largest_component(
        raw_mask
    )


    pixels = int(
        cleaned.sum()
    )


    bbox = bbox_from_mask(
        cleaned
    )


    sam_score = float(
        scores[
            idx
        ]
    )


    row = {

        "candidate":
            idx + 1,

        "sam2_index":
            idx,

        "sam2_score":
            sam_score,

        "pixels":
            pixels,

        "mask_bbox":
            bbox,

        "prompt_bbox":
            prompt_box,

        "semantic_source_phrase":
            selected.get(
                "phrase"
            ),

        "semantic_source_score":
            float(
                selected[
                    "score"
                ]
            ),
    }


    rows.append(
        row
    )


    # --------------------------------------------------------
    # MASK
    # --------------------------------------------------------

    MASK_PATH = (
        OBJECTS
        / f"candidate_{idx + 1:02d}_mask.png"
    )


    Image.fromarray(
        cleaned.astype(
            np.uint8
        )
        *
        255,
        mode="L"
    ).save(
        MASK_PATH
    )


    # --------------------------------------------------------
    # EXACT STAGE01 RGB RGBA
    # --------------------------------------------------------

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
    ] = master


    rgba[
        ...,
        3
    ] = (
        cleaned.astype(
            np.uint8
        )
        *
        255
    )


    RGBA_PATH = (
        OBJECTS
        / f"candidate_{idx + 1:02d}_stage01_rgb_rgba.png"
    )


    Image.fromarray(
        rgba,
        mode="RGBA"
    ).save(
        RGBA_PATH
    )


    # --------------------------------------------------------
    # OVERLAY
    # --------------------------------------------------------

    overlay = (
        master_pil.copy()
        .convert(
            "RGBA"
        )
    )


    tint = np.zeros(
        (
            H,
            W,
            4
        ),
        dtype=np.uint8
    )


    tint[
        cleaned
    ] = [
        0,
        255,
        0,
        105
    ]


    overlay = Image.alpha_composite(
        overlay,
        Image.fromarray(
            tint,
            mode="RGBA"
        )
    )


    draw = ImageDraw.Draw(
        overlay
    )


    draw.rectangle(
        prompt_box,
        outline="red",
        width=2
    )


    if bbox is not None:

        draw.rectangle(
            bbox,
            outline="lime",
            width=2
        )


    draw.text(
        (
            8,
            8
        ),
        (
            f"SAM2 candidate {idx + 1} | "
            f"score={sam_score:.4f} | "
            f"pixels={pixels}"
        ),
        fill="lime"
    )


    OVERLAY_PATH = (
        OBJECTS
        / f"candidate_{idx + 1:02d}_overlay.png"
    )


    overlay.convert(
        "RGB"
    ).save(
        OVERLAY_PATH
    )


# ============================================================
# SAVE RESULT JSON
# ============================================================

RESULT = {

    "stage":
        "TEST08_STAGE03E_MIRROR_SAM2_MULTIMASK",

    "status":
        "RND_REQUIRES_VISUAL_MASK_SELECTION",

    "master": {

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

    "semantic_bbox_source": {

        "stage":
            "TEST08_STAGE03D_DINO_MIRROR_DISCOVERY",

        "source_json":
            str(
                DINO_JSON
            ),

        "selection_rule":
            "highest-scoring deduplicated mirror-family candidate",

        "phrase":
            selected.get(
                "phrase"
            ),

        "score":
            float(
                selected[
                    "score"
                ]
            ),

        "bbox":
            prompt_box,

        "manual_coordinates":
            False,

        "bbox_expansion":
            0.0,
    },

    "sam2": {

        "config":
            SAM2_CONFIG,

        "checkpoint":
            str(
                SAM2_CHECKPOINT
            ),

        "multimask_output":
            True,

        "candidate_count":
            len(
                rows
            ),
    },

    "candidates":
        rows,

    "rules": [

        "DINO semantic bbox used automatically.",

        "No manually entered mirror coordinates.",

        "Prompt bbox not expanded.",

        "All SAM2 multimask candidates preserved.",

        "No final mirror candidate selected automatically.",

        "Exact RGB source is TEST08 Stage01 master.",

        "SAM2 geometry is not semantic proof.",
    ],
}


RESULT_PATH = (
    OUT
    / "00_stage03e_result.json"
)


RESULT_PATH.write_text(
    json.dumps(
        RESULT,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# ============================================================
# BUILD CONTACT SHEET
# ============================================================

candidate_images = []


for idx in range(
    len(
        rows
    )
):

    p = (
        OBJECTS
        / f"candidate_{idx + 1:02d}_overlay.png"
    )

    candidate_images.append(
        Image.open(
            p
        ).convert(
            "RGB"
        )
    )


if candidate_images:

    thumb_w = 420

    thumb_h = int(
        round(
            H
            *
            (
                thumb_w
                /
                W
            )
        )
    )


    resized = []


    for img in candidate_images:

        resized.append(
            img.resize(
                (
                    thumb_w,
                    thumb_h
                ),
                Image.Resampling.LANCZOS
            )
        )


    sheet = Image.new(
        "RGB",
        (
            thumb_w
            *
            len(
                resized
            ),
            thumb_h
        ),
        "white"
    )


    for idx, img in enumerate(
        resized
    ):

        sheet.paste(
            img,
            (
                idx
                *
                thumb_w,
                0
            )
        )


    SHEET_PATH = (
        OUT
        / "02_sam2_candidate_comparison.png"
    )


    sheet.save(
        SHEET_PATH
    )


else:

    SHEET_PATH = None


# ============================================================
# PRINT
# ============================================================

print()
print("=" * 110)
print("TEST08 STAGE03E RESULT")
print("=" * 110)

print()

print(
    "DINO PHRASE:",
    selected.get(
        "phrase"
    )
)

print(
    "DINO SCORE:",
    round(
        float(
            selected[
                "score"
            ]
        ),
        6
    )
)

print(
    "PROMPT BBOX:",
    [
        round(
            v,
            2
        )
        for v
        in prompt_box
    ]
)


print()
print(
    "SAM2 CANDIDATES:"
)


for row in rows:

    print(
        f"  candidate {row['candidate']}",
        "| score:",
        round(
            row[
                "sam2_score"
            ],
            6
        ),
        "| pixels:",
        row[
            "pixels"
        ],
        "| bbox:",
        row[
            "mask_bbox"
        ]
    )


print()
print(
    "RESULT:",
    RESULT_PATH
)

print(
    "PROMPT PREVIEW:",
    PROMPT_PREVIEW_PATH
)

print(
    "COMPARISON:",
    SHEET_PATH
)


# ============================================================
# DISPLAY
# ============================================================

print()
print(
    "DINO PROMPT BOX"
)

display(
    Image.open(
        PROMPT_PREVIEW_PATH
    )
)


if SHEET_PATH is not None:

    print()
    print(
        "SAM2 MULTIMASK COMPARISON"
    )

    display(
        Image.open(
            SHEET_PATH
        )
    )


# ============================================================
# CLEANUP
# ============================================================

del predictor
del sam2_model

gc.collect()


if torch.cuda.is_available():

    torch.cuda.empty_cache()
