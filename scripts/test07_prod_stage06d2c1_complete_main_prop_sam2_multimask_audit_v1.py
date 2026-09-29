
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


LOCALIZATION_PATH = (
    STAGE06
    / "06d2b3_final_verified_main_prop_localization"
    / "00_stage06d2b3_result.json"
)


INVENTORY_PATH = (
    STAGE06
    / "06d2a2_verified_main_prop_state"
    / "00_stage06d2a2_verified_main_prop_state.json"
)


OUT = (
    STAGE06
    / "06d2c1_complete_main_prop_sam2_multimask_audit"
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
# PROMPT CONFIG
# ============================================================

PROMPT_EXPANSION = 0.025

LOCAL_GATE_EXPANSION = 0.10


# ============================================================
# VALIDATE
# ============================================================

for path in [
    MASTER_PATH,
    LOCALIZATION_PATH,
    INVENTORY_PATH,
    SAM2_CHECKPOINT,
]:

    if not path.exists():

        raise FileNotFoundError(
            path
        )


# ============================================================
# LOAD
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


localization = json.loads(
    LOCALIZATION_PATH.read_text(
        encoding="utf-8"
    )
)


inventory = json.loads(
    INVENTORY_PATH.read_text(
        encoding="utf-8"
    )
)


inventory_by_id = {

    row["main_prop_id"]:
        row

    for row in inventory["main_props"]
}


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

    mask = np.zeros(
        (
            H,
            W
        ),
        dtype=bool
    )


    x1, y1, x2, y2 = [
        int(round(v))
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


    mask[
        y1:y2,
        x1:x2
    ] = True


    return mask


def mask_bbox(
    mask
):

    ys, xs = np.where(
        mask
    )

    if len(xs) == 0:

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
        ) + 1,

        int(
            ys.max()
        ) + 1,
    ]


def border_touch_fraction(
    mask
):

    bbox = mask_bbox(
        mask
    )

    if bbox is None:

        return 1.0


    x1, y1, x2, y2 = bbox


    border = np.zeros_like(
        mask
    )


    border[
        y1,
        x1:x2
    ] = True


    border[
        y2 - 1,
        x1:x2
    ] = True


    border[
        y1:y2,
        x1
    ] = True


    border[
        y1:y2,
        x2 - 1
    ] = True


    denominator = int(
        border.sum()
    )


    if denominator <= 0:

        return 0.0


    return float(
        (
            mask
            &
            border
        ).sum()
        /
        denominator
    )


def draw_mask_outline(
    image,
    mask,
    color="lime",
    width=3
):

    result = image.copy()


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


    draw = ImageDraw.Draw(
        result
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
                points
                +
                [
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
print("PRODUCTION STAGE 06D2C1")
print("COMPLETE MAIN-PROP SAM2 MULTIMASK AUDIT")
print("=" * 110)


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
# RUN SIX FINAL MAIN PROPS
# ============================================================

results = []


for localization_row in localization[
    "results"
]:

    prop_id = localization_row[
        "main_prop_id"
    ]


    prop_name = localization_row[
        "main_prop_name"
    ]


    prop_info = inventory_by_id[
        prop_id
    ]


    final_bbox = clamp_box(
        localization_row[
            "final_bbox"
        ]
    )


    prompt_bbox = expand_box(
        final_bbox,
        PROMPT_EXPANSION
    )


    gate_bbox = expand_box(
        prompt_bbox,
        LOCAL_GATE_EXPANSION
    )


    gate_mask = box_mask(
        gate_bbox
    )


    print()
    print("=" * 110)

    print(
        f"{prop_id} — {prop_name}"
    )

    print("=" * 110)

    print(
        "FINAL LOCALIZATION:",
        [
            round(v, 1)
            for v in final_bbox
        ]
    )

    print(
        "SAM2 PROMPT:",
        [
            round(v, 1)
            for v in prompt_bbox
        ]
    )


    # ========================================================
    # SAM2 MULTIMASK
    # ========================================================

    masks, scores, logits = predictor.predict(

        point_coords=None,

        point_labels=None,

        box=np.asarray(
            prompt_bbox,
            dtype=np.float32
        ),

        multimask_output=True,
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


        gated_mask = (
            raw_mask
            &
            gate_mask
        )


        area = int(
            gated_mask.sum()
        )


        containment = (

            float(
                area
                /
                raw_area
            )

            if raw_area > 0

            else 0.0
        )


        sam_score = float(
            scores[
                candidate_index
            ]
        )


        border_touch = (
            border_touch_fraction(
                gated_mask
            )
        )


        result_bbox = mask_bbox(
            gated_mask
        )


        # ====================================================
        # SAVE MASK
        # ====================================================

        mask_path = (
            OBJECT_DIR
            /
            f"{prop_id}_candidate_{candidate_index + 1}_mask.png"
        )


        Image.fromarray(
            gated_mask.astype(
                np.uint8
            )
            *
            255
        ).save(
            mask_path
        )


        # ====================================================
        # PREVIEW
        #
        # GREEN = SAM2 mask
        # RED   = final localization
        # YELLOW = actual SAM2 prompt
        # ====================================================

        preview = draw_mask_outline(
            master_pil,
            gated_mask,
            "lime",
            3
        )


        draw = ImageDraw.Draw(
            preview
        )


        draw.rectangle(
            final_bbox,
            outline="red",
            width=2
        )


        draw.rectangle(
            prompt_bbox,
            outline="yellow",
            width=2
        )


        preview_path = (
            OBJECT_DIR
            /
            f"{prop_id}_candidate_{candidate_index + 1}_preview.png"
        )


        preview.save(
            preview_path
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

            "mask_bbox":
                result_bbox,

            "mask_path":
                str(
                    mask_path
                ),

            "preview_path":
                str(
                    preview_path
                ),
        })


        print(
            "candidate #{} | SAM={:.4f} | "
            "area={} | containment={:.4f} | "
            "border={:.4f} | mask_bbox={}".format(

                candidate_index + 1,

                sam_score,

                area,

                containment,

                border_touch,

                result_bbox,
            )
        )


    results.append({

        "main_prop_id":
            prop_id,

        "main_prop_name":
            prop_name,

        "prop_type":
            prop_info[
                "prop_type"
            ],

        "visible_members":
            prop_info.get(
                "visible_members",
                []
            ),

        "final_localization_bbox":
            final_bbox,

        "sam2_prompt_bbox":
            prompt_bbox,

        "candidate_count":
            len(
                candidate_rows
            ),

        "candidates":
            candidate_rows,

        "status":
            "REQUIRES_VISUAL_CANDIDATE_SELECTION",
    })


# ============================================================
# SAVE
# ============================================================

FINAL_STATE = {

    "stage":
        "06D2C1",

    "input_stage":
        "06D2B3",

    "segmentation_image":
        str(
            MASTER_PATH
        ),

    "main_prop_count":
        len(
            results
        ),

    "results":
        results,

    "status":
        "REQUIRES_COMPLETE_MAIN_PROP_MASK_AUDIT",

    "rules": [

        "exactly one final segmentation target per verified main prop",

        "child components are not independent segmentation targets",

        "SAM2 multimask candidates are diagnostic until visually audited",

        "no automatic mask candidate is accepted",

        "no masks were unioned",

        "no final prop layer was created",

        "Stage05F is segmentation image",

        "Stage01 remains final exact RGB source",
    ],
}


RESULT_PATH = (
    OUT
    / "00_stage06d2c1_result.json"
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
print("PRODUCTION STAGE 06D2C1 RESULT")
print("=" * 110)


for row in results:

    print()

    print(
        "{} — {}".format(
            row[
                "main_prop_id"
            ],
            row[
                "main_prop_name"
            ],
        )
    )


    for candidate in row[
        "candidates"
    ]:

        print(
            "   candidate #{} | SAM={:.4f} | area={} | "
            "contain={:.4f} | border={:.4f}".format(

                candidate[
                    "candidate_index"
                ],

                candidate[
                    "sam_score"
                ],

                candidate[
                    "area"
                ],

                candidate[
                    "containment"
                ],

                candidate[
                    "border_touch"
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
    "NO MASK CANDIDATE HAS BEEN ACCEPTED YET."
)

print(
    "NO FINAL PROP MASK UNION WAS CREATED."
)


# ============================================================
# INLINE VISUAL AUDIT
#
# PRINT EVERY CANDIDATE.
# 6 props × 3 SAM2 masks = 18 previews.
# ============================================================

print()
print("=" * 110)
print("INLINE 06D2C1 COMPLETE MAIN-PROP MASK AUDIT")
print("=" * 110)


for row in results:

    print()
    print(
        "-" * 110
    )

    print(
        "{} — {}".format(

            row[
                "main_prop_id"
            ],

            row[
                "main_prop_name"
            ],
        )
    )

    print(
        "EXPECTED MEMBERS:",
        row[
            "visible_members"
        ]
    )

    print(
        "-" * 110
    )


    for candidate in row[
        "candidates"
    ]:

        show(

            candidate[
                "preview_path"
            ],

            (
                f'{row["main_prop_id"]} — '
                f'{row["main_prop_name"]}\n'
                f'SAM2 CANDIDATE #{candidate["candidate_index"]} | '
                f'SAM={candidate["sam_score"]:.3f} | '
                f'AREA={candidate["area"]}'
            ),

            figsize=(7, 8)
        )


# ============================================================
# CLEANUP
# ============================================================

del predictor
del sam2_model

gc.collect()

if torch.cuda.is_available():

    torch.cuda.empty_cache()
