
# ============================================================
# TEST07 - ROOM03
# STAGE01Z10
#
# V3/V9-GUIDED SAM2 FLOOR REFINEMENT
#
# ARCHITECTURE
# ------------------------------------------------------------
# V1 = coherent clean-room appearance
# V3/V9 = semantic floor proposal
# SAM2 = actual floor-boundary refinement
# Frozen props = hard exclusion
#
# FINAL:
# V1 + exact #00C800 on SAM2 refined floor
#
# NO QWEN GENERATION.
# ============================================================

from pathlib import Path
from PIL import Image, ImageDraw
from IPython.display import display

import gc
import json
import hashlib
import numpy as np
import cv2
import torch


# ============================================================
# 1. SAM2 IMPORTS
# ============================================================

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except Exception as e:
    raise RuntimeError(
        "SAM2 import failed. Existing TEST07 SAM2 environment is required.\n"
        + str(e)
    )


# ============================================================
# 2. IDENTIFIERS
# ============================================================

STAGE_NAME = (
    "TEST07_STAGE01Z10_"
    "V3_GUIDED_SAM2_FLOOR_REFINEMENT_V10"
)

SCRIPT_NAME = (
    "test07_stage01z10_"
    "v3_guided_sam2_floor_refinement_v10.py"
)


# ============================================================
# 3. PATHS
# ============================================================

BASE = Path("/workspace/axolotl")

ROOT = (
    BASE
    / "test07"
    / "runs"
    / "room03_bathroom"
    / "stages"
)

OUT = (
    ROOT
    / "01z10_v3_guided_sam2_floor_refinement_v10"
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

PROP_MASK_PATH = (
    ROOT
    / "01y3_frozen_canonical_prop_protection"
    / "00_canonical_prop_protection_mask.png"
)


# ============================================================
# 4. SAM2 MODEL
# ============================================================

SAM2_CHECKPOINT = Path(
    "/workspace/axolotl/test07/models/sam2/"
    "sam2.1_hiera_large.pt"
)

SAM2_CONFIG = (
    "configs/sam2.1/"
    "sam2.1_hiera_l.yaml"
)


# ============================================================
# 5. TARGET
# ============================================================

FLOOR_RGB = np.array(
    [0, 200, 0],
    dtype=np.uint8
)

FLOOR_HEX = "#00C800"


# ============================================================
# 6. HELPERS
# ============================================================

def load_rgb(path, size=None):

    img = Image.open(path).convert("RGB")

    if size is not None and img.size != size:
        img = img.resize(
            size,
            Image.Resampling.LANCZOS
        )

    return np.array(img)


def load_mask(path, size):

    img = Image.open(path).convert("L")

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

            chunk = f.read(1024 * 1024)

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


def iou(a, b):

    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()

    if union == 0:
        return 0.0

    return float(inter / union)


# ============================================================
# 7. VALIDATE
# ============================================================

required = {
    "ORIGINAL": ORIGINAL_PATH,
    "V1 SAFE": V1_SAFE_PATH,
    "V3": V3_PATH,
    "V9 MASK": V9_MASK_PATH,
    "PROP MASK": PROP_MASK_PATH,
    "SAM2 CHECKPOINT": SAM2_CHECKPOINT,
}

print()
print("=" * 100)
print(STAGE_NAME)
print("=" * 100)

for name, path in required.items():

    ok = path.exists()

    print(
        f"{name:20s}",
        "✅" if ok else "❌",
        path
    )

    if not ok:
        raise FileNotFoundError(path)


# ============================================================
# 8. LOAD INPUTS
# ============================================================

original_pil = Image.open(
    ORIGINAL_PATH
).convert("RGB")

original = np.array(
    original_pil
)

H, W = original.shape[:2]

SIZE = (W, H)


v1 = load_rgb(
    V1_SAFE_PATH,
    SIZE
)

v3 = load_rgb(
    V3_PATH,
    SIZE
)

proposal = load_mask(
    V9_MASK_PATH,
    SIZE
)

props = load_mask(
    PROP_MASK_PATH,
    SIZE
)


# Never prompt SAM2 on protected props
proposal_nonprop = (
    proposal
    &
    ~props
)


print()
print("MASTER:", W, "x", H)
print("V9 proposal pixels:", int(proposal_nonprop.sum()))


# ============================================================
# 9. BUILD SAM2 BOX FROM FLOOR PROPOSAL
# ============================================================

bbox = mask_bbox(
    proposal_nonprop
)

if bbox is None:
    raise RuntimeError(
        "V9 floor proposal is empty."
    )

x1, y1, x2, y2 = bbox


# Slight expansion around proposal
pad_x = max(
    4,
    int(W * 0.02)
)

pad_y = max(
    4,
    int(H * 0.02)
)

box = np.array(
    [
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(W - 1, x2 + pad_x),
        min(H - 1, y2 + pad_y)
    ],
    dtype=np.float32
)


print("SAM2 BOX:", box.tolist())


# ============================================================
# 10. GENERATE POSITIVE POINTS FROM INTERIOR
# ============================================================
#
# Use distance transform so positive points are safely inside
# the semantic floor region, not directly on the boundary.
# ============================================================

proposal_u8 = (
    proposal_nonprop.astype(np.uint8)
    * 255
)

distance = cv2.distanceTransform(
    proposal_u8,
    cv2.DIST_L2,
    5
)


positive_points = []

dist_copy = distance.copy()


# Choose several widely separated interior points
for _ in range(7):

    _, max_val, _, max_loc = cv2.minMaxLoc(
        dist_copy
    )

    if max_val < 2:
        break

    px, py = max_loc

    positive_points.append(
        [float(px), float(py)]
    )

    cv2.circle(
        dist_copy,
        (px, py),
        28,
        0,
        -1
    )


# ============================================================
# 11. ADD BOTTOM-EDGE FLOOR POSITIVES
# ============================================================
#
# Floor should reach the lower image region.
# Add additional proposal-supported points near bottom.
# ============================================================

for frac_x in [
    0.20,
    0.40,
    0.60,
    0.80
]:

    px = int(
        W * frac_x
    )

    column_y = np.where(
        proposal_nonprop[:, px]
    )[0]

    if len(column_y) > 0:

        py = int(
            column_y.max()
        )

        positive_points.append(
            [float(px), float(py)]
        )


if len(positive_points) == 0:
    raise RuntimeError(
        "Could not create positive SAM2 floor prompts."
    )


# ============================================================
# 12. NEGATIVE POINTS
# ============================================================
#
# Give SAM2 explicit non-floor guidance:
# - upper wall/ceiling
# - visible wall regions near left/right
#
# These are intentionally conservative.
# ============================================================

negative_points = [
    [W * 0.50, H * 0.15],
    [W * 0.35, H * 0.35],
    [W * 0.75, H * 0.35],
    [W * 0.08, H * 0.55],
    [W * 0.92, H * 0.45],
]


point_coords = np.array(
    positive_points
    +
    negative_points,
    dtype=np.float32
)


point_labels = np.array(
    [1] * len(positive_points)
    +
    [0] * len(negative_points),
    dtype=np.int32
)


print()
print(
    "POSITIVE POINTS:",
    len(positive_points)
)

print(
    "NEGATIVE POINTS:",
    len(negative_points)
)


# ============================================================
# 13. SAVE PROMPT PREVIEW
# ============================================================

prompt_preview = original.copy()

for x, y in positive_points:

    cv2.circle(
        prompt_preview,
        (int(x), int(y)),
        4,
        (0, 255, 0),
        -1
    )


for x, y in negative_points:

    cv2.circle(
        prompt_preview,
        (int(x), int(y)),
        4,
        (255, 0, 0),
        -1
    )


cv2.rectangle(
    prompt_preview,
    (
        int(box[0]),
        int(box[1])
    ),
    (
        int(box[2]),
        int(box[3])
    ),
    (255, 255, 0),
    2
)


prompt_preview_path = (
    OUT
    / "00_sam2_prompt_preview.png"
)

Image.fromarray(
    prompt_preview
).save(
    prompt_preview_path
)


# ============================================================
# 14. CLEAN GPU BEFORE SAM2
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
# 15. LOAD SAM2
# ============================================================

print()
print("Loading SAM2...")


sam2_model = build_sam2(
    SAM2_CONFIG,
    str(SAM2_CHECKPOINT),
    device=device
)


predictor = SAM2ImagePredictor(
    sam2_model
)


predictor.set_image(
    original
)


print("✅ SAM2 READY")


# ============================================================
# 16. RUN MULTIMASK PREDICTION
# ============================================================

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
# 17. SCORE FLOOR CANDIDATES
# ============================================================
#
# We do NOT blindly choose SAM2's own confidence.
#
# A good floor mask should:
# - overlap semantic V9 proposal
# - cover proposal
# - reach lower edge
# - stay mainly in lower image
# - avoid excessive upper-wall leakage
# - avoid protected props
# ============================================================

candidate_rows = []


for idx, (mask, sam_score) in enumerate(
    zip(
        masks,
        scores
    )
):

    m = (
        mask > 0
    )

    # Remove known physical props
    m &= ~props


    area = int(
        m.sum()
    )


    intersection = int(
        np.logical_and(
            m,
            proposal_nonprop
        ).sum()
    )


    proposal_pixels = max(
        1,
        int(
            proposal_nonprop.sum()
        )
    )


    proposal_recall = float(
        intersection
        /
        proposal_pixels
    )


    candidate_precision = float(
        intersection
        /
        max(
            1,
            area
        )
    )


    candidate_iou = iou(
        m,
        proposal_nonprop
    )


    # Bottom contact
    bottom_band = m[
        int(
            H * 0.94
        ):,
        :
    ]

    bottom_contact = bool(
        np.any(
            bottom_band
        )
    )


    # Upper leakage
    upper_pixels = int(
        m[
            :int(H * 0.45),
            :
        ].sum()
    )

    upper_ratio = float(
        upper_pixels
        /
        max(
            1,
            area
        )
    )


    # Lower-region ratio
    lower_pixels = int(
        m[
            int(H * 0.50):,
            :
        ].sum()
    )

    lower_ratio = float(
        lower_pixels
        /
        max(
            1,
            area
        )
    )


    # Broad custom ranking
    custom_score = (

        2.0
        *
        proposal_recall

        +

        1.0
        *
        candidate_iou

        +

        0.5
        *
        candidate_precision

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

        -

        2.0
        *
        upper_ratio
    )


    candidate_rows.append(
        {
            "index":
                int(idx),

            "sam2_score":
                float(sam_score),

            "pixels":
                area,

            "proposal_intersection":
                intersection,

            "proposal_recall":
                proposal_recall,

            "precision_vs_proposal":
                candidate_precision,

            "iou_vs_proposal":
                candidate_iou,

            "bottom_contact":
                bottom_contact,

            "upper_ratio":
                upper_ratio,

            "lower_ratio":
                lower_ratio,

            "custom_score":
                float(custom_score),
        }
    )


# ============================================================
# 18. PRINT CANDIDATE AUDIT
# ============================================================

print()
print("=" * 100)
print("SAM2 FLOOR CANDIDATE AUDIT")
print("=" * 100)

for row in candidate_rows:

    print(
        f"C{row['index']} | "
        f"SAM={row['sam2_score']:.4f} | "
        f"PIX={row['pixels']} | "
        f"RECALL={row['proposal_recall']:.4f} | "
        f"IOU={row['iou_vs_proposal']:.4f} | "
        f"UPPER={row['upper_ratio']:.4f} | "
        f"LOWER={row['lower_ratio']:.4f} | "
        f"CUSTOM={row['custom_score']:.4f}"
    )


# ============================================================
# 19. SAVE ALL 3 RAW SAM2 CANDIDATES
# ============================================================

for idx, mask in enumerate(masks):

    m = (
        mask > 0
    )

    m &= ~props

    save_mask(
        m,
        OUT
        / f"01_candidate_{idx+1:02d}_mask.png"
    )


# ============================================================
# 20. SELECT BEST CANDIDATE
# ============================================================

best_row = max(
    candidate_rows,
    key=lambda x: x["custom_score"]
)

best_index = int(
    best_row["index"]
)


sam_floor = (
    masks[
        best_index
    ] > 0
)


sam_floor &= ~props


print()
print(
    "SELECTED SAM2 CANDIDATE:",
    best_index
)


# ============================================================
# 21. LOWER-FLOOR SAFETY GATE
# ============================================================
#
# SAM2 may occasionally include walls.
#
# We preserve the proposal's highest known floor region
# and allow only a modest amount above it.
# ============================================================

proposal_rows = np.where(
    proposal_nonprop
)[0]


if len(proposal_rows) > 0:

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
    (H, W),
    dtype=bool
)

row_gate[
    safe_top:,
    :
] = True


sam_floor &= row_gate


# ============================================================
# 22. KEEP ONLY BOTTOM-CONNECTED SAM2 COMPONENTS
# ============================================================

n, labels, stats, centroids = (
    cv2.connectedComponentsWithStats(
        sam_floor.astype(np.uint8),
        connectivity=8
    )
)


connected = np.zeros(
    (H, W),
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


    touches_bottom = bool(
        np.any(
            component[
                int(H * 0.92):,
                :
            ]
        )
    )


    overlaps_proposal = bool(
        np.any(
            component
            &
            proposal_nonprop
        )
    )


    if (
        overlaps_proposal
        and
        (
            touches_bottom
            or
            area > 1500
        )
    ):

        connected |= component


# ============================================================
# 23. CONSERVATIVE FINAL CLEANUP
# ============================================================

connected_u8 = (
    connected.astype(np.uint8)
    * 255
)


kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (5, 5)
)


final_floor = cv2.morphologyEx(
    connected_u8,
    cv2.MORPH_CLOSE,
    kernel,
    iterations=1
) > 0


final_floor &= ~props


# Preserve trusted V9 proposal if SAM2 accidentally clips
# a small internal floor region.
final_floor |= (
    proposal_nonprop
    &
    row_gate
)


# ============================================================
# 24. SAVE FINAL MASK
# ============================================================

save_mask(
    final_floor,
    OUT
    / "04_v10_final_sam2_floor_mask.png"
)


# ============================================================
# 25. MASK AUDIT VS V9
# ============================================================

added_vs_v9 = (
    final_floor
    &
    ~proposal_nonprop
)

lost_vs_v9 = (
    proposal_nonprop
    &
    ~final_floor
)


print()
print("V10 MASK AUDIT")
print("-" * 100)

print(
    "V9 pixels:",
    int(
        proposal_nonprop.sum()
    )
)

print(
    "V10 pixels:",
    int(
        final_floor.sum()
    )
)

print(
    "Added vs V9:",
    int(
        added_vs_v9.sum()
    )
)

print(
    "Lost vs V9:",
    int(
        lost_vs_v9.sum()
    )
)


# ============================================================
# 26. CREATE MASK AUDIT OVERLAY
# ============================================================
#
# GREEN = V10 final floor
# CYAN  = new pixels added by SAM2
# RED   = V9 pixels lost
# MAGENTA = frozen props
# ============================================================

overlay = (
    original.copy()
    .astype(np.float32)
)


def tint(
    image,
    mask,
    color,
    alpha
):

    out = image.copy()

    c = np.array(
        color,
        dtype=np.float32
    )

    out[
        mask
    ] = (
        out[
            mask
        ]
        *
        (1.0 - alpha)

        +

        c
        *
        alpha
    )

    return out


overlay = tint(
    overlay,
    final_floor,
    [0, 200, 0],
    0.45
)

overlay = tint(
    overlay,
    added_vs_v9,
    [0, 255, 255],
    0.65
)

overlay = tint(
    overlay,
    lost_vs_v9,
    [255, 0, 0],
    0.50
)

overlay = tint(
    overlay,
    props,
    [255, 0, 255],
    0.45
)


overlay = np.clip(
    overlay,
    0,
    255
).astype(np.uint8)


overlay_path = (
    OUT
    / "05_v10_floor_audit_overlay.png"
)


Image.fromarray(
    overlay
).save(
    overlay_path
)


# ============================================================
# 27. CREATE V10 FROM V1
# ============================================================

v10 = (
    v1.copy()
)


v10[
    final_floor
] = FLOOR_RGB


# Restore exact original props
v10[
    props
] = original[
    props
]


v10_pil = (
    Image.fromarray(
        v10
    )
)


v10_path = (
    OUT
    / "06_v10_final_surface_template.png"
)


v10_pil.save(
    v10_path
)


# ============================================================
# 28. VERIFY
# ============================================================

floor_pixels = (
    v10[
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

    v10.astype(np.int16)

    -

    original.astype(np.int16)

)[
    props
]


max_prop_error = (
    int(prop_diff.max())
    if prop_diff.size
    else 0
)


must_match_v1 = (
    ~final_floor
    &
    ~props
)


nonfloor_diff = np.abs(

    v10.astype(np.int16)

    -

    v1.astype(np.int16)

)[
    must_match_v1
]


max_nonfloor_error = (
    int(nonfloor_diff.max())
    if nonfloor_diff.size
    else 0
)


print()
print("V10 VERIFICATION")
print("-" * 100)

print(
    "Floor exact #00C800 %:",
    round(
        floor_exact_ratio * 100,
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
# 29. COMPARISON V1 / V3 / V9 / V10
# ============================================================

LABEL_H = 40


v9_result_path = (
    ROOT
    / "01z9_v1_plus_v3_relaxed_floor_v9"
    / "07_v9_final_surface_template.png"
)


v9_result = load_rgb(
    v9_result_path,
    SIZE
)


comparison = Image.new(
    "RGB",
    (
        W * 4,
        H + LABEL_H
    ),
    (255, 255, 255)
)


comparison.paste(
    Image.fromarray(v1),
    (0, LABEL_H)
)

comparison.paste(
    Image.fromarray(v3),
    (W, LABEL_H)
)

comparison.paste(
    Image.fromarray(v9_result),
    (W * 2, LABEL_H)
)

comparison.paste(
    v10_pil,
    (W * 3, LABEL_H)
)


draw = ImageDraw.Draw(
    comparison
)


for x, text in [
    (8, "V1"),
    (W + 8, "V3 FLOOR REFERENCE"),
    (W * 2 + 8, "V9"),
    (W * 3 + 8, "V10 SAM2 FLOOR"),
]:

    draw.text(
        (x, 10),
        text,
        fill=(0, 0, 0)
    )


comparison_path = (
    OUT
    / "07_v1_v3_v9_v10_comparison.png"
)


comparison.save(
    comparison_path
)


# ============================================================
# 30. V9 VS V10
# ============================================================

compare2 = Image.new(
    "RGB",
    (
        W * 2,
        H + LABEL_H
    ),
    (255, 255, 255)
)


compare2.paste(
    Image.fromarray(v9_result),
    (0, LABEL_H)
)

compare2.paste(
    v10_pil,
    (W, LABEL_H)
)


draw2 = ImageDraw.Draw(
    compare2
)


draw2.text(
    (8, 10),
    "V9",
    fill=(0, 0, 0)
)

draw2.text(
    (W + 8, 10),
    "V10",
    fill=(0, 0, 0)
)


compare2_path = (
    OUT
    / "08_v9_vs_v10.png"
)


compare2.save(
    compare2_path
)


# ============================================================
# 31. REPORT
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

    "sam2_checkpoint":
        str(SAM2_CHECKPOINT),

    "sam2_config":
        SAM2_CONFIG,

    "sam2_device":
        device,

    "strategy":
        (
            "Use V9/V3 semantic floor proposal to create "
            "automatic box + positive/negative prompts for SAM2, "
            "then use SAM2 to refine the continuous floor boundary."
        ),

    "prompting": {

        "bbox":
            box.tolist(),

        "positive_points":
            positive_points,

        "negative_points":
            negative_points,

        "safe_top":
            int(safe_top),
    },

    "candidates":
        candidate_rows,

    "selected_candidate":
        int(best_index),

    "counts": {

        "v9_proposal":
            int(
                proposal_nonprop.sum()
            ),

        "v10_final":
            int(
                final_floor.sum()
            ),

        "added_vs_v9":
            int(
                added_vs_v9.sum()
            ),

        "lost_vs_v9":
            int(
                lost_vs_v9.sum()
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
            str(ORIGINAL_PATH),

        "v1":
            str(V1_SAFE_PATH),

        "v3":
            str(V3_PATH),

        "v9_floor":
            str(V9_MASK_PATH),

        "props":
            str(PROP_MASK_PATH),
    },

    "hashes": {

        "original":
            sha256_file(ORIGINAL_PATH),

        "v1":
            sha256_file(V1_SAFE_PATH),

        "v3":
            sha256_file(V3_PATH),

        "v9_floor":
            sha256_file(V9_MASK_PATH),
    },

    "outputs": {

        "prompt_preview":
            str(prompt_preview_path),

        "final_floor_mask":
            str(
                OUT
                / "04_v10_final_sam2_floor_mask.png"
            ),

        "audit_overlay":
            str(overlay_path),

        "v10":
            str(v10_path),

        "comparison":
            str(comparison_path),

        "v9_vs_v10":
            str(compare2_path),
    },

    "benchmark": {

        "V1":
            "GOOD RECONSTRUCTION BASE",

        "V3":
            "SEMANTIC GREEN FLOOR REFERENCE",

        "V8":
            "BRIGHT GREEN FLOOR EXTRACTION",

        "V9":
            "RELAXED GREEN FLOOR EXTRACTION",

        "V10":
            "SAM2 REFINED FLOOR BOUNDARY",
    },
}


report_path = (
    OUT
    / "00_stage01z10_report.json"
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
# 32. RELEASE SAM2
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
# 33. FINAL
# ============================================================

print()
print("=" * 100)
print("STAGE01Z10 COMPLETE")
print("=" * 100)

print()
print(
    "NO NEW QWEN GENERATION"
)

print(
    "SELECTED SAM2 CANDIDATE:",
    best_index
)

print(
    "FINAL FLOOR MASK:",
    OUT
    / "04_v10_final_sam2_floor_mask.png"
)

print(
    "V10:",
    v10_path
)

print(
    "REPORT:",
    report_path
)


# ============================================================
# 34. DISPLAY
# ============================================================

print()
print(
    "SAM2 PROMPTS"
)

display(
    Image.open(
        prompt_preview_path
    )
)

print()
print(
    "V10 FLOOR AUDIT"
)

display(
    Image.open(
        overlay_path
    )
)

print()
print(
    "V1 / V3 / V9 / V10"
)

display(
    comparison
)

print()
print(
    "V9 VS V10"
)

display(
    compare2
)
