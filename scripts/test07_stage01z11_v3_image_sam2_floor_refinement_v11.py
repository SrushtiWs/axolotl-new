
# ============================================================
# TEST07 - ROOM03
# STAGE01Z11
#
# V3 IMAGE -> SAM2 FLOOR REFINEMENT
#
# SOURCES
# ------------------------------------------------------------
# V1 = coherent clean reconstruction / final appearance base
# V3 = simplified semantic room used as SAM2 input
# V9 = trusted semantic floor proposal for automatic prompts
# Frozen props = exact original RGB restoration
#
# FINAL
# ------------------------------------------------------------
# V1
# + V11 SAM2 floor mask
# + exact #00C800
# + original frozen props
#
# NO NEW QWEN GENERATION.
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

from pathlib import Path
from PIL import Image, ImageDraw
from IPython.display import display

import gc
import hashlib
import json
import numpy as np
import cv2
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# ============================================================
# 2. IDENTIFIERS
# ============================================================

STAGE_NAME = (
    "TEST07_STAGE01Z11_"
    "V3_IMAGE_SAM2_FLOOR_REFINEMENT_V11"
)

SCRIPT_NAME = (
    "test07_stage01z11_"
    "v3_image_sam2_floor_refinement_v11.py"
)


# ============================================================
# 3. PATHS
# ============================================================

BASE = Path(
    "/workspace/axolotl"
)

ROOT = (
    BASE
    / "test07"
    / "runs"
    / "room03_bathroom"
    / "stages"
)

OUT = (
    ROOT
    / "01z11_v3_image_sam2_floor_refinement_v11"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


ORIGINAL_PATH = (
    ROOT
    / "01_structure_inverse_base"
    / "00_input_resized.png"
)


V1_SAFE_PATH = (
    ROOT
    / "01z1_qwen_coherent_cleanroom_v1"
    / "02_safe_props_restored.png"
)


V3_PATH = (
    ROOT
    / "01z3_qwen_short_surface_template_v3"
    / "05_qwen_v3_master_size.png"
)


V9_MASK_PATH = (
    ROOT
    / "01z9_v1_plus_v3_relaxed_floor_v9"
    / "04_v9_final_floor_mask.png"
)


V10_MASK_PATH = (
    ROOT
    / "01z10_v3_guided_sam2_floor_refinement_v10"
    / "04_v10_final_sam2_floor_mask.png"
)


PROP_MASK_PATH = (
    ROOT
    / "01y3_frozen_canonical_prop_protection"
    / "00_canonical_prop_protection_mask.png"
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
# 4. TARGET FLOOR
# ============================================================

FLOOR_HEX = "#00C800"

FLOOR_RGB = np.array(
    [0, 200, 0],
    dtype=np.uint8
)


# ============================================================
# 5. HELPERS
# ============================================================

def load_rgb(path, size=None):

    img = Image.open(
        path
    ).convert("RGB")

    if size is not None and img.size != size:

        img = img.resize(
            size,
            Image.Resampling.LANCZOS
        )

    return np.array(img)


def load_mask(path, size):

    img = Image.open(
        path
    ).convert("L")

    if img.size != size:

        img = img.resize(
            size,
            Image.Resampling.NEAREST
        )

    return np.array(img) > 0


def save_mask(mask, path):

    Image.fromarray(
        mask.astype(np.uint8) * 255,
        mode="L"
    ).save(path)


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


def mask_bbox(mask):

    ys, xs = np.where(mask)

    if len(xs) == 0:
        return None

    return [
        int(xs.min()),
        int(ys.min()),
        int(xs.max()),
        int(ys.max())
    ]


def mask_iou(a, b):

    inter = np.logical_and(
        a,
        b
    ).sum()

    union = np.logical_or(
        a,
        b
    ).sum()

    if union == 0:
        return 0.0

    return float(
        inter / union
    )


# ============================================================
# 6. VALIDATE INPUTS
# ============================================================

required = {

    "ORIGINAL":
        ORIGINAL_PATH,

    "V1":
        V1_SAFE_PATH,

    "V3":
        V3_PATH,

    "V9 MASK":
        V9_MASK_PATH,

    "V10 MASK":
        V10_MASK_PATH,

    "PROP MASK":
        PROP_MASK_PATH,

    "SAM2 CHECKPOINT":
        SAM2_CHECKPOINT,
}


print()
print("=" * 100)
print(STAGE_NAME)
print("=" * 100)

print()
print("SCRIPT:", SCRIPT_NAME)
print("OUTPUT:", OUT)

print()
print(
    "IMPORTANT: SAM2 IMAGE INPUT = V3 SIMPLIFIED ROOM"
)

print(
    "NO NEW QWEN GENERATION"
)

print()


for name, path in required.items():

    ok = path.exists()

    print(
        f"{name:18s}",
        "✅" if ok else "❌",
        path
    )

    if not ok:

        raise FileNotFoundError(
            path
        )


# ============================================================
# 7. LOAD MASTER INPUTS
# ============================================================

original_pil = Image.open(
    ORIGINAL_PATH
).convert("RGB")

original = np.array(
    original_pil
)

H, W = original.shape[:2]

SIZE = (
    W,
    H
)


v1 = load_rgb(
    V1_SAFE_PATH,
    SIZE
)


v3 = load_rgb(
    V3_PATH,
    SIZE
)


v9 = load_mask(
    V9_MASK_PATH,
    SIZE
)


v10 = load_mask(
    V10_MASK_PATH,
    SIZE
)


props = load_mask(
    PROP_MASK_PATH,
    SIZE
)


proposal = (
    v9
    &
    ~props
)


print()
print(
    "MASTER SIZE:",
    W,
    "x",
    H
)

print(
    "V9 PROPOSAL PIXELS:",
    int(
        proposal.sum()
    )
)


# ============================================================
# 8. BUILD BOX FROM V9 PROPOSAL
# ============================================================

bbox = mask_bbox(
    proposal
)

if bbox is None:

    raise RuntimeError(
        "V9 proposal is empty."
    )


x1, y1, x2, y2 = bbox


pad_x = max(
    6,
    int(
        W * 0.03
    )
)

pad_y = max(
    6,
    int(
        H * 0.03
    )
)


box = np.array(
    [
        max(
            0,
            x1 - pad_x
        ),

        max(
            0,
            y1 - pad_y
        ),

        min(
            W - 1,
            x2 + pad_x
        ),

        min(
            H - 1,
            y2 + pad_y
        ),
    ],
    dtype=np.float32
)


print(
    "SAM2 BOX:",
    box.tolist()
)


# ============================================================
# 9. POSITIVE POINTS FROM PROPOSAL INTERIOR
# ============================================================

proposal_u8 = (
    proposal.astype(np.uint8)
    * 255
)


distance = cv2.distanceTransform(
    proposal_u8,
    cv2.DIST_L2,
    5
)


positive_points = []

distance_work = (
    distance.copy()
)


# Select interior maxima.
for _ in range(10):

    _, max_val, _, max_loc = (
        cv2.minMaxLoc(
            distance_work
        )
    )

    if max_val < 2.0:

        break


    px, py = max_loc


    positive_points.append(
        [
            float(px),
            float(py)
        ]
    )


    # Suppress neighbourhood so prompts spread across floor.
    cv2.circle(
        distance_work,
        (
            px,
            py
        ),
        24,
        0,
        -1
    )


# ============================================================
# 10. ADD LOWER-FLOOR POSITIVE POINTS
# ============================================================

for frac_x in [
    0.12,
    0.28,
    0.45,
    0.62,
    0.78,
    0.92,
]:

    px = int(
        W * frac_x
    )


    ys = np.where(
        proposal[
            :,
            px
        ]
    )[0]


    if len(ys) > 0:

        # Use a point slightly inside from bottom.
        py = int(
            np.percentile(
                ys,
                80
            )
        )

        positive_points.append(
            [
                float(px),
                float(py)
            ]
        )


# ============================================================
# 11. NEGATIVE POINTS
# ============================================================
#
# Because SAM2 sees V3, wall and ceiling are visually simple.
#
# Put clear negatives in:
# - ceiling
# - upper wall
# - middle wall
# - door/wall side
#
# We avoid placing negatives close to expected floor boundary.
# ============================================================

negative_points = [

    [
        W * 0.50,
        H * 0.10
    ],

    [
        W * 0.25,
        H * 0.28
    ],

    [
        W * 0.50,
        H * 0.30
    ],

    [
        W * 0.75,
        H * 0.28
    ],

    [
        W * 0.12,
        H * 0.48
    ],

    [
        W * 0.88,
        H * 0.45
    ],
]


if len(
    positive_points
) == 0:

    raise RuntimeError(
        "No positive SAM2 floor points created."
    )


point_coords = np.array(
    positive_points
    +
    negative_points,
    dtype=np.float32
)


point_labels = np.array(

    [1]
    *
    len(
        positive_points
    )

    +

    [0]
    *
    len(
        negative_points
    ),

    dtype=np.int32
)


print()
print(
    "POSITIVE POINTS:",
    len(
        positive_points
    )
)

print(
    "NEGATIVE POINTS:",
    len(
        negative_points
    )
)


# ============================================================
# 12. SAVE V3 SAM2 PROMPT PREVIEW
# ============================================================

prompt_preview = (
    v3.copy()
)


for x, y in positive_points:

    cv2.circle(
        prompt_preview,
        (
            int(x),
            int(y)
        ),
        4,
        (
            0,
            255,
            0
        ),
        -1
    )


for x, y in negative_points:

    cv2.circle(
        prompt_preview,
        (
            int(x),
            int(y)
        ),
        4,
        (
            255,
            0,
            0
        ),
        -1
    )


cv2.rectangle(
    prompt_preview,

    (
        int(
            box[0]
        ),
        int(
            box[1]
        )
    ),

    (
        int(
            box[2]
        ),
        int(
            box[3]
        )
    ),

    (
        255,
        255,
        0
    ),

    2
)


prompt_preview_path = (
    OUT
    / "00_v3_sam2_prompt_preview.png"
)


Image.fromarray(
    prompt_preview
).save(
    prompt_preview_path
)


# ============================================================
# 13. CLEAN GPU
# ============================================================

gc.collect()


if torch.cuda.is_available():

    torch.cuda.empty_cache()

    torch.cuda.ipc_collect()


device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print()
print(
    "SAM2 DEVICE:",
    device
)


# ============================================================
# 14. LOAD SAM2
# ============================================================

print()
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


# ============================================================
# IMPORTANT:
# SAM2 IMAGE = V3, NOT ORIGINAL.
# ============================================================

predictor.set_image(
    v3
)


print(
    "✅ SAM2 READY ON V3 IMAGE"
)


# ============================================================
# 15. PREDICT
# ============================================================

masks, scores, logits = predictor.predict(

    point_coords=
        point_coords,

    point_labels=
        point_labels,

    box=
        box,

    multimask_output=
        True
)


print()
print(
    "SAM2 CANDIDATES:",
    len(
        masks
    )
)


# ============================================================
# 16. CANDIDATE SCORING
# ============================================================
#
# Score against:
#
# 1. V9 semantic proposal
# 2. V3 green appearance
# 3. lower-image behaviour
# 4. upper leakage
# 5. bottom connectivity
#
# Unlike V10, SAM2 now sees the simplified V3 scene.
# ============================================================

# ----------------------------------------------
# Build broad V3 green reference
# ----------------------------------------------

r = v3[..., 0].astype(
    np.int16
)

g = v3[..., 1].astype(
    np.int16
)

b = v3[..., 2].astype(
    np.int16
)


green_reference = (

    (g > r + 10)

    &

    (g > b + 10)

    &

    (g > 40)
)


candidate_rows = []


for idx, (
    raw_mask,
    sam_score
) in enumerate(
    zip(
        masks,
        scores
    )
):

    m = (
        raw_mask > 0
    )


    # Physical props are never editable floor.
    m &= ~props


    area = int(
        m.sum()
    )


    if area == 0:

        candidate_rows.append(
            {
                "index":
                    int(idx),

                "sam2_score":
                    float(
                        sam_score
                    ),

                "pixels":
                    0,

                "custom_score":
                    -999.0,
            }
        )

        continue


    # ------------------------------------------
    # Proposal agreement
    # ------------------------------------------

    intersection_v9 = int(
        np.logical_and(
            m,
            proposal
        ).sum()
    )


    v9_recall = float(
        intersection_v9
        /
        max(
            1,
            int(
                proposal.sum()
            )
        )
    )


    precision_v9 = float(
        intersection_v9
        /
        area
    )


    iou_v9 = mask_iou(
        m,
        proposal
    )


    # ------------------------------------------
    # Green-reference agreement
    # ------------------------------------------

    green_intersection = int(
        np.logical_and(
            m,
            green_reference
        ).sum()
    )


    green_ratio = float(
        green_intersection
        /
        area
    )


    # ------------------------------------------
    # Upper leakage
    # ------------------------------------------

    upper_pixels = int(
        m[
            :int(
                H * 0.45
            ),
            :
        ].sum()
    )


    upper_ratio = float(
        upper_pixels
        /
        area
    )


    # ------------------------------------------
    # Lower occupancy
    # ------------------------------------------

    lower_pixels = int(
        m[
            int(
                H * 0.50
            ):,
            :
        ].sum()
    )


    lower_ratio = float(
        lower_pixels
        /
        area
    )


    # ------------------------------------------
    # Bottom connectivity
    # ------------------------------------------

    bottom_contact = bool(
        np.any(
            m[
                int(
                    H * 0.94
                ):,
                :
            ]
        )
    )


    # ------------------------------------------
    # Area sanity
    # ------------------------------------------

    proposal_area = max(
        1,
        int(
            proposal.sum()
        )
    )


    area_ratio_to_proposal = float(
        area
        /
        proposal_area
    )


    # Penalize absurdly large candidates.
    area_penalty = 0.0

    if area_ratio_to_proposal > 2.0:

        area_penalty = (
            area_ratio_to_proposal
            -
            2.0
        )


    # ------------------------------------------
    # Custom score
    # ------------------------------------------

    custom_score = (

        2.2
        *
        v9_recall

        +

        1.2
        *
        iou_v9

        +

        0.7
        *
        precision_v9

        +

        1.0
        *
        green_ratio

        +

        0.35
        *
        lower_ratio

        +

        (
            0.25
            if bottom_contact
            else 0.0
        )

        +

        0.15
        *
        float(
            sam_score
        )

        -

        2.5
        *
        upper_ratio

        -

        0.5
        *
        area_penalty
    )


    candidate_rows.append(
        {
            "index":
                int(
                    idx
                ),

            "sam2_score":
                float(
                    sam_score
                ),

            "pixels":
                area,

            "v9_recall":
                v9_recall,

            "precision_vs_v9":
                precision_v9,

            "iou_vs_v9":
                iou_v9,

            "green_ratio":
                green_ratio,

            "upper_ratio":
                upper_ratio,

            "lower_ratio":
                lower_ratio,

            "bottom_contact":
                bottom_contact,

            "area_ratio_to_v9":
                area_ratio_to_proposal,

            "custom_score":
                float(
                    custom_score
                ),
        }
    )


# ============================================================
# 17. PRINT CANDIDATE AUDIT
# ============================================================

print()
print("=" * 110)
print("V11 SAM2 FLOOR CANDIDATE AUDIT — SAM2 RUN ON V3")
print("=" * 110)


for row in candidate_rows:

    print(
        f"C{row['index']} | "
        f"SAM={row.get('sam2_score', 0):.4f} | "
        f"PIX={row.get('pixels', 0):6d} | "
        f"V9_REC={row.get('v9_recall', 0):.4f} | "
        f"IOU={row.get('iou_vs_v9', 0):.4f} | "
        f"GREEN={row.get('green_ratio', 0):.4f} | "
        f"UP={row.get('upper_ratio', 0):.4f} | "
        f"LOW={row.get('lower_ratio', 0):.4f} | "
        f"CUSTOM={row.get('custom_score', -999):.4f}"
    )


# ============================================================
# 18. SAVE ALL SAM2 CANDIDATES
# ============================================================

for idx, raw_mask in enumerate(
    masks
):

    m = (
        raw_mask > 0
    )

    m &= ~props


    save_mask(
        m,

        OUT
        / f"01_candidate_{idx+1:02d}_mask.png"
    )


# ============================================================
# 19. SELECT BEST
# ============================================================

best_row = max(
    candidate_rows,
    key=lambda x: x.get(
        "custom_score",
        -999
    )
)


best_index = int(
    best_row[
        "index"
    ]
)


selected = (
    masks[
        best_index
    ] > 0
)


selected &= ~props


print()
print(
    "SELECTED CANDIDATE:",
    best_index
)


# ============================================================
# 20. FLOOR REGION SAFETY GATE
# ============================================================
#
# Use V9's known top floor extent as a conservative guard.
# ============================================================

proposal_rows = np.where(
    proposal
)[0]


if len(
    proposal_rows
) > 0:

    proposal_top = int(
        proposal_rows.min()
    )

else:

    proposal_top = int(
        H * 0.50
    )


safe_top = max(
    0,
    proposal_top
    -
    12
)


row_gate = np.zeros(
    (
        H,
        W
    ),
    dtype=bool
)


row_gate[
    safe_top:,
    :
] = True


selected &= (
    row_gate
)


# ============================================================
# 21. CONNECTED COMPONENT FILTER
# ============================================================

n, labels, stats, centroids = (
    cv2.connectedComponentsWithStats(
        selected.astype(
            np.uint8
        ),
        connectivity=8
    )
)


connected = np.zeros(
    (
        H,
        W
    ),
    dtype=bool
)


for i in range(
    1,
    n
):

    component = (
        labels == i
    )


    area = int(
        stats[
            i,
            cv2.CC_STAT_AREA
        ]
    )


    overlap_v9 = bool(
        np.any(
            component
            &
            proposal
        )
    )


    touches_bottom = bool(
        np.any(
            component[
                int(
                    H * 0.92
                ):,
                :
            ]
        )
    )


    if (
        overlap_v9

        and

        (
            touches_bottom
            or
            area >= 1200
        )
    ):

        connected |= (
            component
        )


# ============================================================
# 22. CLEANUP
# ============================================================

connected_u8 = (
    connected.astype(
        np.uint8
    )
    * 255
)


kernel_close = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (
        5,
        5
    )
)


final_floor = cv2.morphologyEx(
    connected_u8,
    cv2.MORPH_CLOSE,
    kernel_close,
    iterations=1
) > 0


final_floor &= ~props


# ============================================================
# 23. IMPORTANT:
# DO NOT AUTOMATICALLY UNION V9
# ============================================================
#
# V10 unioned V9 back into the final mask.
#
# In V11 we want to know whether SAM2 itself produces a better
# complete floor boundary on V3.
#
# Therefore V9 is ONLY a reference/prompt source.
# ============================================================


# ============================================================
# 24. SAVE FINAL MASK
# ============================================================

save_mask(
    final_floor,

    OUT
    / "04_v11_final_sam2_floor_mask.png"
)


# ============================================================
# 25. AUDIT AGAINST V9 AND V10
# ============================================================

v11_added_vs_v9 = (
    final_floor
    &
    ~v9
)


v9_only = (
    v9
    &
    ~final_floor
)


v11_added_vs_v10 = (
    final_floor
    &
    ~v10
)


v10_only = (
    v10
    &
    ~final_floor
)


print()
print("=" * 100)
print("V11 MASK COMPARISON")
print("=" * 100)

print(
    "V9 pixels :",
    int(
        v9.sum()
    )
)

print(
    "V10 pixels:",
    int(
        v10.sum()
    )
)

print(
    "V11 pixels:",
    int(
        final_floor.sum()
    )
)

print(
    "V11 added vs V9:",
    int(
        v11_added_vs_v9.sum()
    )
)

print(
    "V9 only:",
    int(
        v9_only.sum()
    )
)

print(
    "V11 added vs V10:",
    int(
        v11_added_vs_v10.sum()
    )
)

print(
    "V10 only:",
    int(
        v10_only.sum()
    )
)


# ============================================================
# 26. FLOOR AUDIT OVERLAY
# ============================================================
#
# GREEN   = V11 final floor
# CYAN    = V11 added beyond V9
# RED     = V9 pixels not in V11
# MAGENTA = frozen props
# ============================================================

overlay = (
    original.copy()
    .astype(
        np.float32
    )
)


def tint(
    image,
    mask,
    color,
    alpha
):

    result = (
        image.copy()
    )


    c = np.array(
        color,
        dtype=np.float32
    )


    result[
        mask
    ] = (

        result[
            mask
        ]
        *
        (
            1.0
            -
            alpha
        )

        +

        c
        *
        alpha
    )


    return result


overlay = tint(
    overlay,
    final_floor,
    [
        0,
        200,
        0
    ],
    0.45
)


overlay = tint(
    overlay,
    v11_added_vs_v9,
    [
        0,
        255,
        255
    ],
    0.70
)


overlay = tint(
    overlay,
    v9_only,
    [
        255,
        0,
        0
    ],
    0.45
)


overlay = tint(
    overlay,
    props,
    [
        255,
        0,
        255
    ],
    0.45
)


overlay = np.clip(
    overlay,
    0,
    255
).astype(
    np.uint8
)


overlay_path = (
    OUT
    / "05_v11_floor_audit_overlay.png"
)


Image.fromarray(
    overlay
).save(
    overlay_path
)


# ============================================================
# 27. CREATE V11 FROM V1
# ============================================================

v11 = (
    v1.copy()
)


v11[
    final_floor
] = FLOOR_RGB


# Original props always win.
v11[
    props
] = original[
    props
]


v11_pil = (
    Image.fromarray(
        v11
    )
)


v11_path = (
    OUT
    / "06_v11_final_surface_template.png"
)


v11_pil.save(
    v11_path
)


# ============================================================
# 28. EXACT RGB VERIFICATION
# ============================================================

floor_pixels = (
    v11[
        final_floor
    ]
)


if floor_pixels.size:

    floor_exact_ratio = float(
        np.mean(
            np.all(
                floor_pixels
                ==
                FLOOR_RGB,
                axis=1
            )
        )
    )

else:

    floor_exact_ratio = 1.0


prop_diff = np.abs(

    v11.astype(
        np.int16
    )

    -

    original.astype(
        np.int16
    )

)[
    props
]


max_prop_error = (

    int(
        prop_diff.max()
    )

    if prop_diff.size

    else 0
)


must_match_v1 = (
    ~final_floor
    &
    ~props
)


nonfloor_diff = np.abs(

    v11.astype(
        np.int16
    )

    -

    v1.astype(
        np.int16
    )

)[
    must_match_v1
]


max_nonfloor_error = (

    int(
        nonfloor_diff.max()
    )

    if nonfloor_diff.size

    else 0
)


print()
print("=" * 100)
print("V11 RGB VERIFICATION")
print("=" * 100)

print(
    "Floor exact #00C800 %:",
    round(
        floor_exact_ratio
        * 100,
        6
    )
)

print(
    "Max prop RGB error:",
    max_prop_error
)

print(
    "Max non-floor change vs V1:",
    max_nonfloor_error
)


# ============================================================
# 29. LOAD V9 / V10 RESULTS FOR VISUAL COMPARISON
# ============================================================

V9_RESULT_PATH = (
    ROOT
    / "01z9_v1_plus_v3_relaxed_floor_v9"
    / "07_v9_final_surface_template.png"
)


V10_RESULT_PATH = (
    ROOT
    / "01z10_v3_guided_sam2_floor_refinement_v10"
    / "06_v10_final_surface_template.png"
)


v9_result = load_rgb(
    V9_RESULT_PATH,
    SIZE
)


v10_result = load_rgb(
    V10_RESULT_PATH,
    SIZE
)


# ============================================================
# 30. V1 / V3 / V10 / V11 COMPARISON
# ============================================================

LABEL_H = 40


comparison = Image.new(
    "RGB",
    (
        W * 4,
        H + LABEL_H
    ),
    (
        255,
        255,
        255
    )
)


comparison.paste(
    Image.fromarray(
        v1
    ),
    (
        0,
        LABEL_H
    )
)


comparison.paste(
    Image.fromarray(
        v3
    ),
    (
        W,
        LABEL_H
    )
)


comparison.paste(
    Image.fromarray(
        v10_result
    ),
    (
        W * 2,
        LABEL_H
    )
)


comparison.paste(
    v11_pil,
    (
        W * 3,
        LABEL_H
    )
)


draw = ImageDraw.Draw(
    comparison
)


for x, text in [

    (
        8,
        "V1"
    ),

    (
        W + 8,
        "V3 SAM2 INPUT"
    ),

    (
        W * 2 + 8,
        "V10 SAM2 ON ORIGINAL"
    ),

    (
        W * 3 + 8,
        "V11 SAM2 ON V3"
    ),
]:

    draw.text(
        (
            x,
            10
        ),
        text,
        fill=(
            0,
            0,
            0
        )
    )


comparison_path = (
    OUT
    / "07_v1_v3_v10_v11_comparison.png"
)


comparison.save(
    comparison_path
)


# ============================================================
# 31. V9 VS V10 VS V11
# ============================================================

comparison2 = Image.new(
    "RGB",
    (
        W * 3,
        H + LABEL_H
    ),
    (
        255,
        255,
        255
    )
)


comparison2.paste(
    Image.fromarray(
        v9_result
    ),
    (
        0,
        LABEL_H
    )
)


comparison2.paste(
    Image.fromarray(
        v10_result
    ),
    (
        W,
        LABEL_H
    )
)


comparison2.paste(
    v11_pil,
    (
        W * 2,
        LABEL_H
    )
)


draw2 = ImageDraw.Draw(
    comparison2
)


for x, text in [

    (
        8,
        "V9"
    ),

    (
        W + 8,
        "V10"
    ),

    (
        W * 2 + 8,
        "V11"
    ),
]:

    draw2.text(
        (
            x,
            10
        ),
        text,
        fill=(
            0,
            0,
            0
        )
    )


comparison2_path = (
    OUT
    / "08_v9_v10_v11_comparison.png"
)


comparison2.save(
    comparison2_path
)


# ============================================================
# 32. SAVE V3 MASK PREVIEW
# ============================================================

v3_overlay = (
    v3.copy()
    .astype(
        np.float32
    )
)


v3_overlay = tint(
    v3_overlay,
    final_floor,
    [
        255,
        0,
        255
    ],
    0.40
)


v3_overlay = np.clip(
    v3_overlay,
    0,
    255
).astype(
    np.uint8
)


v3_overlay_path = (
    OUT
    / "09_v11_mask_on_v3.png"
)


Image.fromarray(
    v3_overlay
).save(
    v3_overlay_path
)


# ============================================================
# 33. REPORT
# ============================================================

report = {

    "stage":
        STAGE_NAME,

    "status":
        "COMPLETED",

    "script":
        SCRIPT_NAME,

    "new_qwen_generation":
        False,

    "sam2_input_image":
        "V3 simplified surface-template image",

    "sam2_checkpoint":
        str(
            SAM2_CHECKPOINT
        ),

    "sam2_config":
        SAM2_CONFIG,

    "sam2_device":
        device,

    "strategy":
        (
            "Run SAM2 on V3 simplified image rather than original "
            "textured bathroom, using V9 semantic floor mask to "
            "automatically derive prompts."
        ),

    "prompting": {

        "box":
            box.tolist(),

        "positive_points":
            positive_points,

        "negative_points":
            negative_points,

        "safe_top":
            int(
                safe_top
            ),
    },

    "candidates":
        candidate_rows,

    "selected_candidate":
        int(
            best_index
        ),

    "mask_counts": {

        "v9":
            int(
                v9.sum()
            ),

        "v10":
            int(
                v10.sum()
            ),

        "v11":
            int(
                final_floor.sum()
            ),

        "v11_added_vs_v9":
            int(
                v11_added_vs_v9.sum()
            ),

        "v9_only":
            int(
                v9_only.sum()
            ),

        "v11_added_vs_v10":
            int(
                v11_added_vs_v10.sum()
            ),

        "v10_only":
            int(
                v10_only.sum()
            ),
    },

    "verification": {

        "floor_exact_color_ratio":
            floor_exact_ratio,

        "max_prop_rgb_error":
            max_prop_error,

        "max_nonfloor_error_vs_v1":
            max_nonfloor_error,
    },

    "sources": {

        "original":
            str(
                ORIGINAL_PATH
            ),

        "v1":
            str(
                V1_SAFE_PATH
            ),

        "v3":
            str(
                V3_PATH
            ),

        "v9_floor":
            str(
                V9_MASK_PATH
            ),

        "v10_floor":
            str(
                V10_MASK_PATH
            ),

        "props":
            str(
                PROP_MASK_PATH
            ),
    },

    "hashes": {

        "original":
            sha256_file(
                ORIGINAL_PATH
            ),

        "v1":
            sha256_file(
                V1_SAFE_PATH
            ),

        "v3":
            sha256_file(
                V3_PATH
            ),

        "v9":
            sha256_file(
                V9_MASK_PATH
            ),

        "v10":
            sha256_file(
                V10_MASK_PATH
            ),
    },

    "outputs": {

        "sam2_prompt_preview":
            str(
                prompt_preview_path
            ),

        "final_floor_mask":
            str(
                OUT
                / "04_v11_final_sam2_floor_mask.png"
            ),

        "audit_overlay":
            str(
                overlay_path
            ),

        "v11":
            str(
                v11_path
            ),

        "v1_v3_v10_v11":
            str(
                comparison_path
            ),

        "v9_v10_v11":
            str(
                comparison2_path
            ),

        "mask_on_v3":
            str(
                v3_overlay_path
            ),
    },

    "benchmark": {

        "V10":
            "SAM2 ON ORIGINAL — LITTLE IMPROVEMENT",

        "V11":
            "SAM2 ON SIMPLIFIED V3 — CURRENT TEST",
    },
}


report_path = (
    OUT
    / "00_stage01z11_report.json"
)


report_path.write_text(
    json.dumps(
        report,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# ============================================================
# 34. RELEASE SAM2
# ============================================================

try:
    del predictor
except Exception:
    pass

try:
    del sam2_model
except Exception:
    pass


gc.collect()


if torch.cuda.is_available():

    torch.cuda.empty_cache()


# ============================================================
# 35. FINAL
# ============================================================

print()
print("=" * 100)
print("STAGE01Z11 COMPLETE")
print("=" * 100)

print()
print(
    "SAM2 RAN ON V3 SIMPLIFIED IMAGE"
)

print(
    "SELECTED CANDIDATE:",
    best_index
)

print(
    "V11 FLOOR MASK:",
    OUT
    / "04_v11_final_sam2_floor_mask.png"
)

print(
    "V11 RESULT:",
    v11_path
)

print(
    "REPORT:",
    report_path
)


# ============================================================
# 36. DISPLAY
# ============================================================

print()
print(
    "V3 + SAM2 PROMPTS"
)

display(
    Image.open(
        prompt_preview_path
    )
)

print()
print(
    "V11 FLOOR AUDIT ON ORIGINAL"
)

display(
    Image.open(
        overlay_path
    )
)

print()
print(
    "V11 MASK ON V3"
)

display(
    Image.open(
        v3_overlay_path
    )
)

print()
print(
    "V1 / V3 / V10 / V11"
)

display(
    comparison
)

print()
print(
    "V9 / V10 / V11"
)

display(
    comparison2
)
