
from pathlib import Path
import json
import gc

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

INPUT_STATE = (
    STAGE06
    / "06d1d3_crop_reverify_structural_cleanup"
    / "00_stage06d1d3_result.json"
)

OUT = (
    STAGE06
    / "06d1d4_crossmodel_localization_support"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

CROP_DIR = (
    OUT
    / "crops"
)

CROP_DIR.mkdir(
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

CACHE = Path(
    "/workspace/data/huggingface-cache"
)


# ============================================================
# MODEL CONFIG — PRESERVED WORKING VALUES
# ============================================================

FLORENCE_MODEL_ID = (
    "florence-community/Florence-2-large"
)

DINO_MODEL_ID = (
    "IDEA-Research/grounding-dino-base"
)

DINO_BOX_THRESHOLD = 0.15

DINO_TEXT_THRESHOLD = 0.12


# ============================================================
# LOCALIZATION CONFIG
# ============================================================

CROP_EXPANSION = 0.08

CROP_UPSCALE = 2

TOP_K_PER_MODEL = 5


# ============================================================
# CROSS-MODEL AGREEMENT
# ============================================================

MIN_PAIR_IOU = 0.20

MIN_PAIR_CONTAINMENT = 0.55

MAX_NORMALIZED_CENTER_DISTANCE = 0.60


# ============================================================
# VALIDATE
# ============================================================

for path in [
    MASTER_PATH,
    INPUT_STATE,
]:

    if not path.exists():

        raise FileNotFoundError(
            path
        )


master = Image.open(
    MASTER_PATH
).convert(
    "RGB"
)

W, H = master.size


state = json.loads(
    INPUT_STATE.read_text(
        encoding="utf-8"
    )
)


physical = state[
    "physical_object_evidence"
]


print("=" * 110)
print("PRODUCTION STAGE 06D1D4")
print("FLORENCE + DINO SOURCE-CROP LOCALIZATION SUPPORT")
print("=" * 110)

print()
print(
    "PHYSICAL EVIDENCE INPUT:",
    len(
        physical
    )
)

print(
    "MASTER:",
    MASTER_PATH
)

print(
    "SIZE:",
    (W, H)
)


# ============================================================
# HELPERS
# ============================================================

def show(
    image_or_path,
    title,
    figsize=(7, 7)
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
    box,
    width,
    height
):

    x1, y1, x2, y2 = [
        float(v)
        for v in box
    ]

    x1 = max(
        0.0,
        min(
            float(width - 1),
            x1
        )
    )

    y1 = max(
        0.0,
        min(
            float(height - 1),
            y1
        )
    )

    x2 = max(
        x1 + 1.0,
        min(
            float(width),
            x2
        )
    )

    y2 = max(
        y1 + 1.0,
        min(
            float(height),
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

    x1, y1, x2, y2 = [
        float(v)
        for v in box
    ]

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
        ],
        W,
        H
    )


def bbox_area(
    box
):

    return (
        max(
            0.0,
            box[2] - box[0]
        )
        *
        max(
            0.0,
            box[3] - box[1]
        )
    )


def bbox_intersection(
    a,
    b
):

    ix1 = max(
        a[0],
        b[0]
    )

    iy1 = max(
        a[1],
        b[1]
    )

    ix2 = min(
        a[2],
        b[2]
    )

    iy2 = min(
        a[3],
        b[3]
    )

    return (
        max(
            0.0,
            ix2 - ix1
        )
        *
        max(
            0.0,
            iy2 - iy1
        )
    )


def bbox_iou(
    a,
    b
):

    inter = bbox_intersection(
        a,
        b
    )

    union = (
        bbox_area(a)
        +
        bbox_area(b)
        -
        inter
    )

    return float(
        inter
        /
        max(
            1.0,
            union
        )
    )


def bbox_containment(
    a,
    b
):

    inter = bbox_intersection(
        a,
        b
    )

    smaller = min(
        bbox_area(a),
        bbox_area(b)
    )

    return float(
        inter
        /
        max(
            1.0,
            smaller
        )
    )


def normalized_center_distance(
    a,
    b
):

    acx = (
        a[0] + a[2]
    ) / 2.0

    acy = (
        a[1] + a[3]
    ) / 2.0


    bcx = (
        b[0] + b[2]
    ) / 2.0

    bcy = (
        b[1] + b[3]
    ) / 2.0


    distance = (
        (
            acx - bcx
        ) ** 2
        +
        (
            acy - bcy
        ) ** 2
    ) ** 0.5


    diag_a = (
        (
            a[2] - a[0]
        ) ** 2
        +
        (
            a[3] - a[1]
        ) ** 2
    ) ** 0.5


    diag_b = (
        (
            b[2] - b[0]
        ) ** 2
        +
        (
            b[3] - b[1]
        ) ** 2
    ) ** 0.5


    reference = max(
        1.0,
        min(
            diag_a,
            diag_b
        )
    )


    return float(
        distance
        /
        reference
    )


def map_crop_box_to_master(
    local_box,
    crop_box,
    scale
):

    cx1, cy1, _, _ = crop_box

    lx1, ly1, lx2, ly2 = [
        float(v)
        /
        float(scale)
        for v in local_box
    ]

    return clamp_box(
        [
            cx1 + lx1,
            cy1 + ly1,
            cx1 + lx2,
            cy1 + ly2,
        ],
        W,
        H
    )


def choose_phrase(
    row
):

    name = (
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

    name = str(
        name
    ).strip()


    original_phrase = str(
        row.get(
            "grounding_phrase",
            ""
        )
    ).strip()


    # --------------------------------------------------------
    # The corrected semantic name must lead the query.
    # Original phrase is auxiliary evidence only because it
    # may contain the old incorrect label.
    # --------------------------------------------------------

    if original_phrase:

        return (
            f"{name}. {original_phrase}"
        )


    return name


def boxes_agree(
    a,
    b
):

    iou = bbox_iou(
        a,
        b
    )

    containment = bbox_containment(
        a,
        b
    )

    center_distance = normalized_center_distance(
        a,
        b
    )


    agreed = bool(

        iou >= MIN_PAIR_IOU

        or

        containment >= MIN_PAIR_CONTAINMENT

        or

        center_distance
        <=
        MAX_NORMALIZED_CENTER_DISTANCE
    )


    return {

        "agree":
            agreed,

        "iou":
            iou,

        "containment":
            containment,

        "normalized_center_distance":
            center_distance,
    }


# ============================================================
# PREPARE CROPS
# ============================================================

prepared = []


for row in physical:

    evidence_id = row[
        "evidence_id"
    ]


    source_bbox = row.get(
        "source_bbox"
    )


    if (
        not source_bbox
        or
        len(source_bbox) != 4
    ):

        prepared.append({

            "row":
                row,

            "evidence_id":
                evidence_id,

            "status":
                "NO_SOURCE_BBOX",
        })

        continue


    crop_box = expand_box(
        source_bbox,
        CROP_EXPANSION
    )


    x1, y1, x2, y2 = [
        int(round(v))
        for v in crop_box
    ]


    crop = master.crop(
        (
            x1,
            y1,
            x2,
            y2
        )
    )


    highres = crop.resize(
        (
            crop.width
            *
            CROP_UPSCALE,

            crop.height
            *
            CROP_UPSCALE
        ),
        Image.Resampling.LANCZOS
    )


    crop_path = (
        CROP_DIR
        /
        f"{evidence_id}.png"
    )


    highres.save(
        crop_path
    )


    prepared.append({

        "row":
            row,

        "evidence_id":
            evidence_id,

        "status":
            "READY",

        "crop_box":
            crop_box,

        "crop_path":
            str(
                crop_path
            ),

        "phrase":
            choose_phrase(
                row
            ),
    })


print()
print(
    "PREPARED CROPS:",
    sum(
        1
        for x in prepared
        if x[
            "status"
        ]
        ==
        "READY"
    )
)


# ============================================================
# ============================================================
# PHASE A — FLORENCE
# ============================================================
# ============================================================

print()
print("=" * 110)
print("PHASE A — FLORENCE-2 LOCALIZATION")
print("=" * 110)


from transformers import (
    AutoProcessor,
    AutoModelForMultimodalLM,
)


florence_processor = (
    AutoProcessor
    .from_pretrained(
        FLORENCE_MODEL_ID,
        cache_dir=str(
            CACHE
        )
    )
)


florence_model = (
    AutoModelForMultimodalLM
    .from_pretrained(
        FLORENCE_MODEL_ID,
        dtype=torch.float16,
        device_map="auto",
        cache_dir=str(
            CACHE
        )
    )
)


florence_model.eval()


def extract_florence_boxes(
    parsed
):

    output = []


    if not isinstance(
        parsed,
        dict
    ):

        return output


    for _, value in parsed.items():

        if not isinstance(
            value,
            dict
        ):

            continue


        boxes = value.get(
            "bboxes",
            []
        )

        labels = value.get(
            "labels",
            []
        )

        scores = value.get(
            "scores",
            []
        )


        for i, box in enumerate(
            boxes
        ):

            if (
                not isinstance(
                    box,
                    (list, tuple)
                )
                or
                len(box) != 4
            ):

                continue


            output.append({

                "bbox":
                    [
                        float(v)
                        for v in box
                    ],

                "label":
                    (
                        str(
                            labels[i]
                        )
                        if
                        i < len(
                            labels
                        )
                        else ""
                    ),

                "score":
                    (
                        float(
                            scores[i]
                        )
                        if
                        i < len(
                            scores
                        )
                        else None
                    ),
            })


    return output


for idx, item in enumerate(
    prepared,
    start=1
):

    if item[
        "status"
    ] != "READY":

        item[
            "florence"
        ] = []

        continue


    crop = Image.open(
        item[
            "crop_path"
        ]
    ).convert(
        "RGB"
    )


    phrase = item[
        "phrase"
    ]


    task_prompt = (
        "<CAPTION_TO_PHRASE_GROUNDING>"
    )


    inputs = florence_processor(
        text=(
            task_prompt
            +
            phrase
        ),
        images=crop,
        return_tensors="pt"
    )


    inputs = {

        k:
            (
                v.to(
                    florence_model.device
                )
                if hasattr(
                    v,
                    "to"
                )
                else v
            )

        for k, v in inputs.items()
    }


    with torch.inference_mode():

        generated_ids = (
            florence_model.generate(
                **inputs,
                max_new_tokens=256,
                num_beams=3,
                do_sample=False
            )
        )


    generated_text = (
        florence_processor
        .batch_decode(
            generated_ids,
            skip_special_tokens=False
        )[0]
    )


    parsed = (
        florence_processor
        .post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=crop.size
        )
    )


    local_boxes = (
        extract_florence_boxes(
            parsed
        )
    )


    mapped = []


    for box_row in local_boxes[
        :TOP_K_PER_MODEL
    ]:

        local_box = clamp_box(
            box_row[
                "bbox"
            ],
            crop.width,
            crop.height
        )


        master_box = map_crop_box_to_master(
            local_box,
            item[
                "crop_box"
            ],
            CROP_UPSCALE
        )


        mapped.append({

            **box_row,

            "local_bbox":
                local_box,

            "master_bbox":
                master_box,
        })


    item[
        "florence"
    ] = mapped


    print(
        "{:03d}/{:03d} {} | Florence={}".format(

            idx,

            len(
                prepared
            ),

            item[
                "evidence_id"
            ],

            len(
                mapped
            ),
        )
    )


# ============================================================
# FREE FLORENCE
# ============================================================

del florence_model
del florence_processor

gc.collect()

if torch.cuda.is_available():

    torch.cuda.empty_cache()


print()
print(
    "✅ FLORENCE UNLOADED"
)


# ============================================================
# ============================================================
# PHASE B — GROUNDING DINO
# ============================================================
# ============================================================

print()
print("=" * 110)
print("PHASE B — GROUNDING-DINO LOCALIZATION")
print("=" * 110)


from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
)


device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


dino_processor = (
    AutoProcessor
    .from_pretrained(
        DINO_MODEL_ID,
        cache_dir=str(
            CACHE
        )
    )
)


dino_model = (
    AutoModelForZeroShotObjectDetection
    .from_pretrained(
        DINO_MODEL_ID,
        cache_dir=str(
            CACHE
        )
    )
    .to(
        device
    )
)


dino_model.eval()


def run_dino(
    image,
    phrase
):

    text = (
        phrase
        .strip()
        .lower()
    )


    if not text.endswith(
        "."
    ):

        text += "."


    inputs = dino_processor(
        images=image,
        text=text,
        return_tensors="pt"
    )


    inputs = {

        k:
            (
                v.to(
                    device
                )
                if hasattr(
                    v,
                    "to"
                )
                else v
            )

        for k, v in inputs.items()
    }


    with torch.inference_mode():

        outputs = dino_model(
            **inputs
        )


    processed = (
        dino_processor
        .post_process_grounded_object_detection(
            outputs,
            inputs[
                "input_ids"
            ],
            threshold=
                DINO_BOX_THRESHOLD,
            text_threshold=
                DINO_TEXT_THRESHOLD,
            target_sizes=[
                image.size[::-1]
            ]
        )[0]
    )


    boxes = processed.get(
        "boxes",
        []
    )

    scores = processed.get(
        "scores",
        []
    )

    labels = processed.get(
        "labels",
        []
    )


    rows = []


    for i in range(
        min(
            len(boxes),
            TOP_K_PER_MODEL
        )
    ):

        box = (
            boxes[i]
            .detach()
            .cpu()
            .tolist()
        )


        score = float(
            scores[i]
            .detach()
            .cpu()
            .item()
        )


        label = (
            str(
                labels[i]
            )
            if
            i < len(
                labels
            )
            else ""
        )


        rows.append({

            "bbox":
                [
                    float(v)
                    for v in box
                ],

            "score":
                score,

            "label":
                label,
        })


    return rows


for idx, item in enumerate(
    prepared,
    start=1
):

    if item[
        "status"
    ] != "READY":

        item[
            "dino"
        ] = []

        continue


    crop = Image.open(
        item[
            "crop_path"
        ]
    ).convert(
        "RGB"
    )


    rows = run_dino(
        crop,
        item[
            "phrase"
        ]
    )


    mapped = []


    for row in rows:

        local_box = clamp_box(
            row[
                "bbox"
            ],
            crop.width,
            crop.height
        )


        master_box = map_crop_box_to_master(
            local_box,
            item[
                "crop_box"
            ],
            CROP_UPSCALE
        )


        mapped.append({

            **row,

            "local_bbox":
                local_box,

            "master_bbox":
                master_box,
        })


    item[
        "dino"
    ] = mapped


    print(
        "{:03d}/{:03d} {} | DINO={}".format(

            idx,

            len(
                prepared
            ),

            item[
                "evidence_id"
            ],

            len(
                mapped
            ),
        )
    )


# ============================================================
# FREE DINO
# ============================================================

del dino_model
del dino_processor

gc.collect()

if torch.cuda.is_available():

    torch.cuda.empty_cache()


print()
print(
    "✅ DINO UNLOADED"
)


# ============================================================
# ============================================================
# PHASE C — CROSS-MODEL AGREEMENT
# ============================================================
# ============================================================

print()
print("=" * 110)
print("PHASE C — SPATIAL AGREEMENT")
print("=" * 110)


results = []


for item in prepared:

    row = item[
        "row"
    ]


    florence = item.get(
        "florence",
        []
    )

    dino = item.get(
        "dino",
        []
    )


    pair_rows = []


    for fi, frow in enumerate(
        florence
    ):

        for di, drow in enumerate(
            dino
        ):

            metrics = boxes_agree(
                frow[
                    "master_bbox"
                ],
                drow[
                    "master_bbox"
                ]
            )


            pair_rows.append({

                "florence_index":
                    fi,

                "dino_index":
                    di,

                **metrics,
            })


    agreeing_pairs = [

        p

        for p in pair_rows

        if p[
            "agree"
        ]
    ]


    # --------------------------------------------------------
    # Rank agreement pairs:
    # higher IoU
    # higher containment
    # lower center distance
    # --------------------------------------------------------

    agreeing_pairs.sort(

        key=lambda p:
            (
                p[
                    "iou"
                ],
                p[
                    "containment"
                ],
                -p[
                    "normalized_center_distance"
                ],
            ),

        reverse=True
    )


    best_pair = (
        agreeing_pairs[0]
        if agreeing_pairs
        else None
    )


    if (
        florence
        and
        dino
        and
        best_pair is not None
    ):

        support_state = (
            "SUPPORTED_BY_BOTH"
        )


    elif (
        florence
        and
        dino
    ):

        support_state = (
            "CROSS_MODEL_CONFLICT"
        )


    elif (
        florence
        or
        dino
    ):

        support_state = (
            "SINGLE_MODEL_SUPPORT"
        )


    else:

        support_state = (
            "NO_LOCALIZATION_SUPPORT"
        )


    result = {

        "evidence_id":
            item[
                "evidence_id"
            ],

        "input_name":
            row.get(
                "name"
            ),

        "final_name":
            (
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
            ),

        "phrase":
            item.get(
                "phrase"
            ),

        "source_bbox":
            row.get(
                "source_bbox"
            ),

        "crop_box":
            item.get(
                "crop_box"
            ),

        "crop_path":
            item.get(
                "crop_path"
            ),

        "florence_candidates":
            florence,

        "dino_candidates":
            dino,

        "cross_model_pairs":
            pair_rows,

        "best_agreement_pair":
            best_pair,

        "support_state":
            support_state,

        "original_evidence":
            row,
    }


    # --------------------------------------------------------
    # Candidate canonical localization from cross-model pair
    # only when both models agree.
    # --------------------------------------------------------

    if best_pair is not None:

        fbox = florence[
            best_pair[
                "florence_index"
            ]
        ][
            "master_bbox"
        ]


        dbox = dino[
            best_pair[
                "dino_index"
            ]
        ][
            "master_bbox"
        ]


        canonical = [

            (
                fbox[i]
                +
                dbox[i]
            )
            /
            2.0

            for i in range(4)
        ]


        result[
            "crossmodel_canonical_bbox"
        ] = canonical


    else:

        result[
            "crossmodel_canonical_bbox"
        ] = None


    results.append(
        result
    )


# ============================================================
# PREVIEWS
# ============================================================

for result in results:

    preview = master.copy()

    draw = ImageDraw.Draw(
        preview
    )


    # source crop = yellow
    if result[
        "crop_box"
    ]:

        draw.rectangle(
            result[
                "crop_box"
            ],
            outline="yellow",
            width=2
        )


    # Florence = blue
    for idx, row in enumerate(
        result[
            "florence_candidates"
        ],
        start=1
    ):

        box = row[
            "master_bbox"
        ]

        draw.rectangle(
            box,
            outline="blue",
            width=2
        )

        draw.text(
            (
                box[0] + 2,
                box[1] + 2
            ),
            f"F{idx}",
            fill="blue"
        )


    # DINO = red
    for idx, row in enumerate(
        result[
            "dino_candidates"
        ],
        start=1
    ):

        box = row[
            "master_bbox"
        ]

        draw.rectangle(
            box,
            outline="red",
            width=2
        )

        draw.text(
            (
                box[0] + 2,
                box[1] + 12
            ),
            f"D{idx}",
            fill="red"
        )


    # accepted cross-model canonical = lime
    canonical = result[
        "crossmodel_canonical_bbox"
    ]


    if canonical:

        draw.rectangle(
            canonical,
            outline="lime",
            width=3
        )


    preview_path = (
        PREVIEW_DIR
        /
        f'{result["evidence_id"]}.png'
    )


    preview.save(
        preview_path
    )


    result[
        "preview_path"
    ] = str(
        preview_path
    )


# ============================================================
# SUMMARY
# ============================================================

state_counts = {}


for result in results:

    key = result[
        "support_state"
    ]

    state_counts[
        key
    ] = (
        state_counts.get(
            key,
            0
        )
        +
        1
    )


# ============================================================
# SAVE
# ============================================================

FINAL_STATE = {

    "stage":
        "06D1D4",

    "input_stage":
        "06D1D3",

    "detection_master":
        str(
            MASTER_PATH
        ),

    "models": {

        "florence":
            FLORENCE_MODEL_ID,

        "grounding_dino":
            DINO_MODEL_ID,

        "dino_box_threshold":
            DINO_BOX_THRESHOLD,

        "dino_text_threshold":
            DINO_TEXT_THRESHOLD,
    },

    "agreement_thresholds": {

        "min_iou":
            MIN_PAIR_IOU,

        "min_containment":
            MIN_PAIR_CONTAINMENT,

        "max_normalized_center_distance":
            MAX_NORMALIZED_CENTER_DISTANCE,
    },

    "input_physical_count":
        len(
            physical
        ),

    "support_counts":
        state_counts,

    "results":
        results,

    "status":
        "REQUIRES_CROSSMODEL_VISUAL_AUDIT",

    "rules": [

        "no result is rejected solely because one model fails",

        "cross-model localization support is evidence, not final semantic truth",

        "SUPPORTED_BY_BOTH requires spatial agreement",

        "SINGLE_MODEL_SUPPORT remains reviewable",

        "CROSS_MODEL_CONFLICT remains unresolved",

        "NO_LOCALIZATION_SUPPORT remains unresolved",

        "no instance deduplication performed",

        "no connected grouping performed",

        "no SAM2 performed",
    ],
}


RESULT_PATH = (
    OUT
    / "00_stage06d1d4_result.json"
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
# PRINT
# ============================================================

print()
print("=" * 110)
print("PRODUCTION STAGE 06D1D4 RESULT")
print("=" * 110)

print()
print(
    "INPUT PHYSICAL EVIDENCE:",
    len(
        physical
    )
)


for state_name in [

    "SUPPORTED_BY_BOTH",

    "SINGLE_MODEL_SUPPORT",

    "CROSS_MODEL_CONFLICT",

    "NO_LOCALIZATION_SUPPORT",
]:

    print(
        "{:28s}: {}".format(

            state_name,

            state_counts.get(
                state_name,
                0
            ),
        )
    )


print()
print("=" * 110)
print("PER-EVIDENCE SUPPORT STATE")
print("=" * 110)


for result in results:

    best = result[
        "best_agreement_pair"
    ]


    metric_text = ""


    if best:

        metric_text = (

            " | IoU={:.3f}"
            " contain={:.3f}"
            " center={:.3f}"

        ).format(

            best[
                "iou"
            ],

            best[
                "containment"
            ],

            best[
                "normalized_center_distance"
            ],
        )


    print(

        "{:9s} | {:26s} | F={} D={} | {}{}".format(

            result[
                "evidence_id"
            ],

            str(
                result[
                    "final_name"
                ]
            )[:26],

            len(
                result[
                    "florence_candidates"
                ]
            ),

            len(
                result[
                    "dino_candidates"
                ]
            ),

            result[
                "support_state"
            ],

            metric_text,
        )
    )


print()
print(
    "RESULT JSON:",
    RESULT_PATH
)

print()
print(
    "NO INSTANCE DEDUPLICATION WAS PERFORMED."
)

print(
    "NO CONNECTED GROUPING WAS PERFORMED."
)

print(
    "NO SAM2 WAS RUN."
)


# ============================================================
# INLINE REVIEW
#
# Print every problematic item plus a small sample of strong
# supported evidence.
# ============================================================

problem_states = {

    "CROSS_MODEL_CONFLICT",

    "NO_LOCALIZATION_SUPPORT",
}


problem_results = [

    result

    for result in results

    if result[
        "support_state"
    ]
    in
    problem_states
]


single_results = [

    result

    for result in results

    if result[
        "support_state"
    ]
    ==
    "SINGLE_MODEL_SUPPORT"
]


strong_results = [

    result

    for result in results

    if result[
        "support_state"
    ]
    ==
    "SUPPORTED_BY_BOTH"
]


print()
print("=" * 110)
print("INLINE 06D1D4 VISUAL AUDIT")
print("=" * 110)


# ------------------------------------------------------------
# All conflict / unsupported items
# ------------------------------------------------------------

for result in problem_results:

    show(

        result[
            "preview_path"
        ],

        (
            f'{result["evidence_id"]} | '
            f'{result["final_name"]} | '
            f'{result["support_state"]} | '
            'BLUE=Florence RED=DINO GREEN=Agreement'
        ),

        figsize=(7, 7)
    )


# ------------------------------------------------------------
# Single-model items are also important.
# ------------------------------------------------------------

for result in single_results:

    show(

        result[
            "preview_path"
        ],

        (
            f'{result["evidence_id"]} | '
            f'{result["final_name"]} | '
            f'SINGLE_MODEL_SUPPORT'
        ),

        figsize=(7, 7)
    )


# ------------------------------------------------------------
# Show first 8 strong examples as sanity check.
# ------------------------------------------------------------

for result in strong_results[:8]:

    show(

        result[
            "preview_path"
        ],

        (
            f'{result["evidence_id"]} | '
            f'{result["final_name"]} | '
            f'SUPPORTED_BY_BOTH'
        ),

        figsize=(7, 7)
    )
