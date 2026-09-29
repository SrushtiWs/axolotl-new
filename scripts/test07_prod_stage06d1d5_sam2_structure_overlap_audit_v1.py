
from pathlib import Path
import json
import gc

import cv2
import numpy as np

import torch

from PIL import (
    Image,
    ImageDraw
)

import matplotlib.pyplot as plt


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


MASTER_PATH = (
    PROD
    / "stage05_clean_room_with_props"
    / "12_final_canonical_prop_detection_master.png"
)


D1D3_PATH = (
    STAGE06
    / "06d1d3_crop_reverify_structural_cleanup"
    / "00_stage06d1d3_result.json"
)


D1D4_PATH = (
    STAGE06
    / "06d1d4_crossmodel_localization_support"
    / "00_stage06d1d4_result.json"
)


STAGE02 = (
    PROD
    / "stage02_structure"
)


# ------------------------------------------------------------
# FROZEN 3/3 CONSENSUS
# ------------------------------------------------------------

WALL_3 = (
    STAGE02
    / "13_wall_consensus_3of3.png"
)

FLOOR_3 = (
    STAGE02
    / "14_floor_consensus_3of3.png"
)

CEILING_3 = (
    STAGE02
    / "15_ceiling_consensus_3of3.png"
)


# ------------------------------------------------------------
# FROZEN >=2/3 MAJORITY
# ------------------------------------------------------------

WALL_2 = (
    STAGE02
    / "16_wall_majority_2of3.png"
)

FLOOR_2 = (
    STAGE02
    / "17_floor_majority_2of3.png"
)

CEILING_2 = (
    STAGE02
    / "18_ceiling_majority_2of3.png"
)


DISAGREEMENT = (
    STAGE02
    / "19_surface_disagreement_mask.png"
)


OUT = (
    STAGE06
    / "06d1d5_sam2_structure_overlap_audit"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


OBJECT_DIR = (
    OUT
    / "objects"
)

OBJECT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


PREVIEW_DIR = (
    OUT
    / "previews"
)

PREVIEW_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SAM2
# ============================================================

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


# ============================================================
# GEOMETRY CONFIG
# ============================================================

PROMPT_EXPANSION = 0.06

LOCAL_GATE_EXPANSION = 0.12

MIN_SAM_SCORE = 0.50

MIN_CONTAINMENT = 0.82

MAX_BORDER_TOUCH = 0.28

MIN_MASK_PIXELS = 12


# ============================================================
# STRUCTURE STATE THRESHOLDS
#
# These classify diagnostic behavior only.
# They do NOT permanently delete evidence.
# ============================================================

STRUCTURE_DOMINANT_MAJORITY = 0.80

STRUCTURE_DOMINANT_CONSENSUS = 0.55

OBJECT_DOMINANT_MAJORITY = 0.35

OBJECT_DOMINANT_CONSENSUS = 0.20


# ============================================================
# IMPORTANT EVIDENCE IDS TO ALWAYS PRINT INLINE
# ============================================================

FORCED_REVIEW_IDS = {

    "3X3_030",   # alleged toilet tank

    "4X4_001",   # alleged wire

    "4X4_002",   # alleged ceiling cable

    "4X4_026",   # alleged sink

    "3X3_013",   # alleged toilet-seat handle
}


# ============================================================
# VALIDATE
# ============================================================

required = [

    MASTER_PATH,

    D1D3_PATH,

    D1D4_PATH,

    WALL_3,

    FLOOR_3,

    CEILING_3,

    WALL_2,

    FLOOR_2,

    CEILING_2,

    DISAGREEMENT,

    SAM2_CHECKPOINT,
]


for path in required:

    if not path.exists():

        raise FileNotFoundError(
            path
        )


# ============================================================
# LOAD MASTER + STATES
# ============================================================

master_pil = Image.open(
    MASTER_PATH
).convert(
    "RGB"
)

master_np = np.asarray(
    master_pil
)

H, W = master_np.shape[:2]


d1d3 = json.loads(
    D1D3_PATH.read_text(
        encoding="utf-8"
    )
)


d1d4 = json.loads(
    D1D4_PATH.read_text(
        encoding="utf-8"
    )
)


physical = d1d3[
    "physical_object_evidence"
]


d1d4_by_id = {

    row[
        "evidence_id"
    ]:
        row

    for row in d1d4[
        "results"
    ]
}


# ============================================================
# LOAD STRUCTURE MASKS
# ============================================================

def load_mask(
    path
):

    mask = (
        np.asarray(
            Image.open(
                path
            ).convert(
                "L"
            )
        )
        >
        0
    )

    if mask.shape != (
        H,
        W
    ):

        raise RuntimeError(
            f"Mask size mismatch: {path}"
        )

    return mask


wall3 = load_mask(
    WALL_3
)

floor3 = load_mask(
    FLOOR_3
)

ceiling3 = load_mask(
    CEILING_3
)


wall2 = load_mask(
    WALL_2
)

floor2 = load_mask(
    FLOOR_2
)

ceiling2 = load_mask(
    CEILING_2
)


disagreement = load_mask(
    DISAGREEMENT
)


structure3 = (
    wall3
    |
    floor3
    |
    ceiling3
)


structure2 = (
    wall2
    |
    floor2
    |
    ceiling2
)


# ============================================================
# HELPERS
# ============================================================

def show(
    image_or_path,
    title,
    figsize=(7, 8)
):

    if isinstance(
        image_or_path,
        (str, Path)
    ):

        image = Image.open(
            image_or_path
        )

    else:

        image = image_or_path


    plt.figure(
        figsize=figsize
    )

    plt.imshow(
        image
    )

    plt.title(
        title
    )

    plt.axis(
        "off"
    )

    plt.show()


def clamp_box(
    box
):

    x1, y1, x2, y2 = [
        float(v)
        for v in box
    ]


    x1 = max(
        0.0,
        min(
            float(W - 1),
            x1
        )
    )

    y1 = max(
        0.0,
        min(
            float(H - 1),
            y1
        )
    )

    x2 = max(
        x1 + 1.0,
        min(
            float(W),
            x2
        )
    )

    y2 = max(
        y1 + 1.0,
        min(
            float(H),
            y2
        )
    )


    return [
        x1,
        y1,
        x2,
        y2
    ]


def expand_box(
    box,
    ratio
):

    x1, y1, x2, y2 = box


    bw = max(
        1.0,
        x2 - x1
    )

    bh = max(
        1.0,
        y2 - y1
    )


    return clamp_box(
        [
            x1 - bw * ratio,
            y1 - bh * ratio,
            x2 + bw * ratio,
            y2 + bh * ratio,
        ]
    )


def box_mask(
    box
):

    result = np.zeros(
        (
            H,
            W
        ),
        dtype=bool
    )


    x1, y1, x2, y2 = [
        int(
            round(v)
        )
        for v in box
    ]


    x1 = max(
        0,
        min(
            W - 1,
            x1
        )
    )

    y1 = max(
        0,
        min(
            H - 1,
            y1
        )
    )

    x2 = max(
        x1 + 1,
        min(
            W,
            x2
        )
    )

    y2 = max(
        y1 + 1,
        min(
            H,
            y2
        )
    )


    result[
        y1:y2,
        x1:x2
    ] = True


    return result


def mask_fraction(
    mask,
    reference
):

    area = int(
        mask.sum()
    )


    if area <= 0:

        return 0.0


    return float(
        (
            mask
            &
            reference
        ).sum()
        /
        area
    )


def border_touch_fraction(
    mask
):

    ys, xs = np.where(
        mask
    )


    if len(xs) == 0:

        return 1.0


    x1 = int(
        xs.min()
    )

    x2 = int(
        xs.max()
    )

    y1 = int(
        ys.min()
    )

    y2 = int(
        ys.max()
    )


    perimeter_pixels = np.zeros_like(
        mask
    )


    perimeter_pixels[
        y1,
        x1:x2 + 1
    ] = True


    perimeter_pixels[
        y2,
        x1:x2 + 1
    ] = True


    perimeter_pixels[
        y1:y2 + 1,
        x1
    ] = True


    perimeter_pixels[
        y1:y2 + 1,
        x2
    ] = True


    denom = int(
        perimeter_pixels.sum()
    )


    if denom <= 0:

        return 0.0


    return float(
        (
            mask
            &
            perimeter_pixels
        ).sum()
        /
        denom
    )


def choose_structure_state(
    majority_fraction,
    consensus_fraction,
):

    if (
        majority_fraction
        >=
        STRUCTURE_DOMINANT_MAJORITY

        and

        consensus_fraction
        >=
        STRUCTURE_DOMINANT_CONSENSUS
    ):

        return (
            "STRUCTURE_DOMINANT"
        )


    if (
        majority_fraction
        <=
        OBJECT_DOMINANT_MAJORITY

        and

        consensus_fraction
        <=
        OBJECT_DOMINANT_CONSENSUS
    ):

        return (
            "OBJECT_DOMINANT"
        )


    return (
        "MIXED_STRUCTURE_OBJECT"
    )


def draw_mask_outline(
    image,
    mask,
    color,
    width=2
):

    result = image.copy()

    draw = ImageDraw.Draw(
        result
    )


    contours, _ = cv2.findContours(

        (
            mask.astype(
                np.uint8
            )
            *
            255
        ),

        cv2.RETR_EXTERNAL,

        cv2.CHAIN_APPROX_SIMPLE,
    )


    for contour in contours:

        points = [

            (
                int(p[0][0]),
                int(p[0][1])
            )

            for p in contour
        ]


        if len(points) >= 2:

            draw.line(
                points + [
                    points[0]
                ],
                fill=color,
                width=width
            )


    return result


# ============================================================
# LOAD SAM2
# ============================================================

print("=" * 110)
print("PRODUCTION STAGE 06D1D5")
print("SAM2 + FROZEN STAGE02 STRUCTURE-OVERLAP AUDIT")
print("=" * 110)

print()
print(
    "PHYSICAL EVIDENCE:",
    len(
        physical
    )
)


print()
print(
    "Loading SAM2..."
)


from sam2.build_sam import (
    build_sam2,
)

from sam2.sam2_image_predictor import (
    SAM2ImagePredictor,
)


sam2_model = build_sam2(
    SAM2_CONFIG,
    str(
        SAM2_CHECKPOINT
    ),
    device=(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    ),
)


predictor = SAM2ImagePredictor(
    sam2_model
)


predictor.set_image(
    master_np
)


print(
    "✅ SAM2 READY"
)


# ============================================================
# RUN
# ============================================================

results = []


for index, row in enumerate(
    physical,
    start=1
):

    evidence_id = row[
        "evidence_id"
    ]


    loc = d1d4_by_id.get(
        evidence_id
    )


    final_name = (

        row.get(
            "final_name"
        )

        or

        row.get(
            "corrected_name"
        )

        or

        row.get(
            "name"
        )
    )


    print()
    print("=" * 100)

    print(
        f"{index:02d}/{len(physical):02d} "
        f"{evidence_id} — {final_name}"
    )

    print("=" * 100)


    if loc is None:

        result = {

            "evidence_id":
                evidence_id,

            "name":
                final_name,

            "state":
                "SAM2_GEOMETRY_FAIL",

            "reason":
                "Missing 06D1D4 localization record.",
        }

        results.append(
            result
        )

        print(
            "MISSING D1D4 LOCALIZATION"
        )

        continue


    canonical_box = loc.get(
        "crossmodel_canonical_bbox"
    )


    if not canonical_box:

        result = {

            "evidence_id":
                evidence_id,

            "name":
                final_name,

            "state":
                "SAM2_GEOMETRY_FAIL",

            "reason":
                "No cross-model canonical bbox.",
        }

        results.append(
            result
        )

        print(
            "NO CANONICAL BOX"
        )

        continue


    canonical_box = clamp_box(
        canonical_box
    )


    prompt_box = expand_box(
        canonical_box,
        PROMPT_EXPANSION
    )


    local_gate_box = expand_box(
        prompt_box,
        LOCAL_GATE_EXPANSION
    )


    local_gate = box_mask(
        local_gate_box
    )


    # --------------------------------------------------------
    # SAM2 MULTIMASK
    # --------------------------------------------------------

    masks, scores, logits = (
        predictor.predict(

            point_coords=None,

            point_labels=None,

            box=np.array(
                prompt_box,
                dtype=np.float32
            ),

            multimask_output=True,
        )
    )


    candidate_rows = []


    for candidate_index in range(
        len(
            masks
        )
    ):

        raw_mask = (
            masks[
                candidate_index
            ]
            >
            0
        )


        raw_area = int(
            raw_mask.sum()
        )


        local_mask = (
            raw_mask
            &
            local_gate
        )


        area = int(
            local_mask.sum()
        )


        if raw_area > 0:

            containment = float(
                area
                /
                raw_area
            )

        else:

            containment = 0.0


        border_touch = (
            border_touch_fraction(
                local_mask
            )
        )


        sam_score = float(
            scores[
                candidate_index
            ]
        )


        # ----------------------------------------------------
        # Geometry quality
        # ----------------------------------------------------

        geometry_pass = bool(

            area
            >=
            MIN_MASK_PIXELS

            and

            sam_score
            >=
            MIN_SAM_SCORE

            and

            containment
            >=
            MIN_CONTAINMENT

            and

            border_touch
            <=
            MAX_BORDER_TOUCH
        )


        geometry_quality = (

            0.55
            *
            sam_score

            +

            0.30
            *
            containment

            +

            0.15
            *
            (
                1.0
                -
                min(
                    1.0,
                    border_touch
                )
            )
        )


        # ----------------------------------------------------
        # Structure overlaps
        # ----------------------------------------------------

        consensus_fraction = (
            mask_fraction(
                local_mask,
                structure3
            )
        )


        majority_fraction = (
            mask_fraction(
                local_mask,
                structure2
            )
        )


        disagreement_fraction = (
            mask_fraction(
                local_mask,
                disagreement
            )
        )


        wall_majority = (
            mask_fraction(
                local_mask,
                wall2
            )
        )


        floor_majority = (
            mask_fraction(
                local_mask,
                floor2
            )
        )


        ceiling_majority = (
            mask_fraction(
                local_mask,
                ceiling2
            )
        )


        candidate_rows.append({

            "candidate_index":
                candidate_index + 1,

            "sam_score":
                sam_score,

            "raw_area":
                raw_area,

            "area":
                area,

            "containment":
                containment,

            "border_touch":
                border_touch,

            "geometry_quality":
                geometry_quality,

            "geometry_pass":
                geometry_pass,

            "consensus3_structure_fraction":
                consensus_fraction,

            "majority2_structure_fraction":
                majority_fraction,

            "surface_disagreement_fraction":
                disagreement_fraction,

            "wall_majority_fraction":
                wall_majority,

            "floor_majority_fraction":
                floor_majority,

            "ceiling_majority_fraction":
                ceiling_majority,

            "mask":
                local_mask,
        })


    # --------------------------------------------------------
    # Choose best geometry-safe mask.
    #
    # Geometry comes before structure semantics.
    # --------------------------------------------------------

    geometry_candidates = [

        candidate

        for candidate
        in candidate_rows

        if candidate[
            "geometry_pass"
        ]
    ]


    if geometry_candidates:

        selected = max(

            geometry_candidates,

            key=lambda candidate:
                candidate[
                    "geometry_quality"
                ]
        )


    else:

        selected = max(

            candidate_rows,

            key=lambda candidate:
                candidate[
                    "geometry_quality"
                ]
        )


    selected_mask = selected[
        "mask"
    ]


    # ========================================================
    # CLASSIFY DIAGNOSTIC STRUCTURE STATE
    # ========================================================

    if not selected[
        "geometry_pass"
    ]:

        state_name = (
            "SAM2_GEOMETRY_FAIL"
        )


    else:

        state_name = choose_structure_state(

            selected[
                "majority2_structure_fraction"
            ],

            selected[
                "consensus3_structure_fraction"
            ],
        )


    # ========================================================
    # SAVE MASK
    # ========================================================

    safe_name = (
        evidence_id
        .replace(
            "/",
            "_"
        )
    )


    MASK_PATH = (
        OBJECT_DIR
        /
        f"{safe_name}_mask.png"
    )


    RGBA_PATH = (
        OBJECT_DIR
        /
        f"{safe_name}_rgba.png"
    )


    PREVIEW_PATH = (
        PREVIEW_DIR
        /
        f"{safe_name}_preview.png"
    )


    Image.fromarray(

        selected_mask.astype(
            np.uint8
        )
        *
        255

    ).save(
        MASK_PATH
    )


    # --------------------------------------------------------
    # Detection-master RGB only for diagnostic RGBA.
    # NOT final RGB source.
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
    ] = master_np


    rgba[
        ...,
        3
    ] = (
        selected_mask.astype(
            np.uint8
        )
        *
        255
    )


    Image.fromarray(
        rgba,
        mode="RGBA"
    ).save(
        RGBA_PATH
    )


    # ========================================================
    # PREVIEW
    #
    # GREEN  = SAM2 mask
    # RED    = canonical localization bbox
    # YELLOW = prompt bbox
    # ========================================================

    preview = master_pil.copy()


    preview = draw_mask_outline(
        preview,
        selected_mask,
        "lime",
        3
    )


    draw = ImageDraw.Draw(
        preview
    )


    draw.rectangle(
        canonical_box,
        outline="red",
        width=2
    )


    draw.rectangle(
        prompt_box,
        outline="yellow",
        width=2
    )


    preview.save(
        PREVIEW_PATH
    )


    # ========================================================
    # RESULT
    # ========================================================

    result = {

        "evidence_id":
            evidence_id,

        "name":
            final_name,

        "canonical_bbox":
            canonical_box,

        "prompt_bbox":
            prompt_box,

        "selected_candidate_index":
            selected[
                "candidate_index"
            ],

        "sam_score":
            selected[
                "sam_score"
            ],

        "area":
            selected[
                "area"
            ],

        "containment":
            selected[
                "containment"
            ],

        "border_touch":
            selected[
                "border_touch"
            ],

        "geometry_quality":
            selected[
                "geometry_quality"
            ],

        "geometry_pass":
            selected[
                "geometry_pass"
            ],

        "consensus3_structure_fraction":
            selected[
                "consensus3_structure_fraction"
            ],

        "majority2_structure_fraction":
            selected[
                "majority2_structure_fraction"
            ],

        "surface_disagreement_fraction":
            selected[
                "surface_disagreement_fraction"
            ],

        "wall_majority_fraction":
            selected[
                "wall_majority_fraction"
            ],

        "floor_majority_fraction":
            selected[
                "floor_majority_fraction"
            ],

        "ceiling_majority_fraction":
            selected[
                "ceiling_majority_fraction"
            ],

        "state":
            state_name,

        "mask_path":
            str(
                MASK_PATH
            ),

        "rgba_path":
            str(
                RGBA_PATH
            ),

        "preview_path":
            str(
                PREVIEW_PATH
            ),

        "all_sam_candidates":
            [

                {
                    k:
                        v

                    for k, v
                    in candidate.items()

                    if k != "mask"
                }

                for candidate
                in candidate_rows
            ],

        "original_d1d3_evidence":
            row,
    }


    results.append(
        result
    )


    print(
        "SAM={:.4f} | area={} | contain={:.4f} | "
        "border={:.4f}".format(

            result[
                "sam_score"
            ],

            result[
                "area"
            ],

            result[
                "containment"
            ],

            result[
                "border_touch"
            ],
        )
    )


    print(
        "STRUCTURE 3/3={:.4f} | 2/3={:.4f} | "
        "wall={:.4f} floor={:.4f} ceiling={:.4f}".format(

            result[
                "consensus3_structure_fraction"
            ],

            result[
                "majority2_structure_fraction"
            ],

            result[
                "wall_majority_fraction"
            ],

            result[
                "floor_majority_fraction"
            ],

            result[
                "ceiling_majority_fraction"
            ],
        )
    )


    print(
        "STATE:",
        state_name
    )


# ============================================================
# SUMMARY COUNTS
# ============================================================

state_counts = {}


for result in results:

    state_name = result[
        "state"
    ]


    state_counts[
        state_name
    ] = (
        state_counts.get(
            state_name,
            0
        )
        +
        1
    )


# ============================================================
# SAVE RESULT JSON
# ============================================================

FINAL_STATE = {

    "stage":
        "06D1D5",

    "input_stage":
        "06D1D3 + 06D1D4",

    "master":
        str(
            MASTER_PATH
        ),

    "stage02_structure_masks": {

        "wall_consensus_3of3":
            str(
                WALL_3
            ),

        "floor_consensus_3of3":
            str(
                FLOOR_3
            ),

        "ceiling_consensus_3of3":
            str(
                CEILING_3
            ),

        "wall_majority_2of3":
            str(
                WALL_2
            ),

        "floor_majority_2of3":
            str(
                FLOOR_2
            ),

        "ceiling_majority_2of3":
            str(
                CEILING_2
            ),

        "disagreement":
            str(
                DISAGREEMENT
            ),
    },

    "sam2": {

        "checkpoint":
            str(
                SAM2_CHECKPOINT
            ),

        "config":
            SAM2_CONFIG,

        "min_sam_score":
            MIN_SAM_SCORE,

        "min_containment":
            MIN_CONTAINMENT,

        "max_border_touch":
            MAX_BORDER_TOUCH,
    },

    "structure_thresholds": {

        "structure_dominant_majority":
            STRUCTURE_DOMINANT_MAJORITY,

        "structure_dominant_consensus":
            STRUCTURE_DOMINANT_CONSENSUS,

        "object_dominant_majority":
            OBJECT_DOMINANT_MAJORITY,

        "object_dominant_consensus":
            OBJECT_DOMINANT_CONSENSUS,
    },

    "input_count":
        len(
            physical
        ),

    "state_counts":
        state_counts,

    "results":
        results,

    "status":
        "REQUIRES_VISUAL_STRUCTURE_AUDIT",

    "rules": [

        "Stage02 is coarse structure evidence only",

        "STRUCTURE_DOMINANT does not automatically delete evidence",

        "OBJECT_DOMINANT does not automatically prove semantic correctness",

        "SAM2 geometry is diagnostic",

        "no instance deduplication performed",

        "no connected grouping performed",

        "no final prop mask union created",

        "Stage01 remains final RGB source",
    ],
}


RESULT_PATH = (
    OUT
    / "00_stage06d1d5_result.json"
)


RESULT_PATH.write_text(
    json.dumps(
        FINAL_STATE,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print()
print("=" * 110)
print("PRODUCTION STAGE 06D1D5 RESULT")
print("=" * 110)


print(
    "INPUT:",
    len(
        physical
    )
)


for state_name in [

    "OBJECT_DOMINANT",

    "MIXED_STRUCTURE_OBJECT",

    "STRUCTURE_DOMINANT",

    "SAM2_GEOMETRY_FAIL",
]:

    print(
        "{:26s}: {}".format(

            state_name,

            state_counts.get(
                state_name,
                0
            ),
        )
    )


print()
print("=" * 110)
print("PER-EVIDENCE STRUCTURE STATE")
print("=" * 110)


for result in results:

    print(

        "{:9s} | {:25s} | {:23s} | "
        "SAM={:.3f} | 3/3={:.3f} | 2/3={:.3f}".format(

            result[
                "evidence_id"
            ],

            str(
                result[
                    "name"
                ]
            )[:25],

            result[
                "state"
            ],

            result[
                "sam_score"
            ],

            result[
                "consensus3_structure_fraction"
            ],

            result[
                "majority2_structure_fraction"
            ],
        )
    )


print()
print(
    "RESULT JSON:",
    RESULT_PATH
)

print()
print(
    "NO EVIDENCE WAS PERMANENTLY REJECTED."
)

print(
    "NO INSTANCE DEDUPLICATION WAS PERFORMED."
)

print(
    "NO CONNECTED GROUPING WAS PERFORMED."
)

print(
    "NO FINAL PROP UNION WAS CREATED."
)


# ============================================================
# INLINE AUDIT
# ============================================================

print()
print("=" * 110)
print("INLINE 06D1D5 VISUAL AUDIT")
print("=" * 110)


# ------------------------------------------------------------
# Always show the five specifically suspicious records.
# ------------------------------------------------------------

for result in results:

    if result[
        "evidence_id"
    ] in FORCED_REVIEW_IDS:

        show(

            result[
                "preview_path"
            ],

            (
                f'{result["evidence_id"]} | '
                f'{result["name"]} | '
                f'{result["state"]} | '
                f'3/3={result["consensus3_structure_fraction"]:.2f} | '
                f'2/3={result["majority2_structure_fraction"]:.2f}'
            ),
        )


# ------------------------------------------------------------
# Show every structure-dominant result.
# ------------------------------------------------------------

for result in results:

    if (
        result[
            "state"
        ]
        ==
        "STRUCTURE_DOMINANT"

        and

        result[
            "evidence_id"
        ]
        not in
        FORCED_REVIEW_IDS
    ):

        show(

            result[
                "preview_path"
            ],

            (
                f'{result["evidence_id"]} | '
                f'{result["name"]} | '
                'STRUCTURE_DOMINANT'
            ),
        )


# ------------------------------------------------------------
# Show every geometry failure.
# ------------------------------------------------------------

for result in results:

    if (
        result[
            "state"
        ]
        ==
        "SAM2_GEOMETRY_FAIL"

        and

        result[
            "evidence_id"
        ]
        not in
        FORCED_REVIEW_IDS
    ):

        show(

            result[
                "preview_path"
            ],

            (
                f'{result["evidence_id"]} | '
                f'{result["name"]} | '
                'SAM2_GEOMETRY_FAIL'
            ),
        )


# ------------------------------------------------------------
# Sanity-check first six object-dominant masks.
# ------------------------------------------------------------

object_examples = [

    result

    for result in results

    if result[
        "state"
    ]
    ==
    "OBJECT_DOMINANT"
]


for result in object_examples[:6]:

    if result[
        "evidence_id"
    ] in FORCED_REVIEW_IDS:

        continue


    show(

        result[
            "preview_path"
        ],

        (
            f'{result["evidence_id"]} | '
            f'{result["name"]} | '
            'OBJECT_DOMINANT SANITY CHECK'
        ),
    )


# ============================================================
# CLEANUP
# ============================================================

del predictor
del sam2_model

gc.collect()

if torch.cuda.is_available():

    torch.cuda.empty_cache()
