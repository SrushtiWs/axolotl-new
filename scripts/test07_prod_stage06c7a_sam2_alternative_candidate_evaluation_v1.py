
from pathlib import Path
import json
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw


# ============================================================
# SAM2 IMPORT
# ============================================================

SAM2_SOURCE_CANDIDATES = [
    Path("/workspace/sam2_src"),
    Path("/workspace/axolotl/sam2"),
]

for candidate in SAM2_SOURCE_CANDIDATES:

    if (
        candidate.exists()
        and
        str(candidate) not in sys.path
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

CONSENSUS_JSON = (
    STAGE06
    / "06c6_multi_route_candidate_consensus"
    / "00_multi_route_consensus_all.json"
)

SAM2_CHECKPOINT = Path(
    "/workspace/axolotl/test07/models/sam2/"
    "sam2.1_hiera_large.pt"
)

OUT = (
    STAGE06
    / "06c7a_sam2_alternative_candidate_evaluation"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

OBJECTS = (
    OUT
    / "objects"
)

OBJECTS.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIG
# ============================================================

SAM2_CONFIG = (
    "configs/sam2.1/"
    "sam2.1_hiera_l.yaml"
)

LOCAL_EXPANSION = 0.08

MIN_SAM_SCORE = 0.55
MIN_LOCAL_CONTAINMENT = 0.88
MAX_LOCAL_BORDER_TOUCH = 0.20
MIN_MASK_PIXELS = 20


TARGET_RANKS = {
    10: 3,
    21: 4,
    24: 3,
    25: 4,
}


# ============================================================
# HELPERS
# ============================================================

def safe_name(text):

    return "".join(
        c
        if c.isalnum() or c in "-_"
        else "_"
        for c in str(text)
    )[:60]


def make_box_gate(
    box,
    W,
    H,
    expansion=0.0
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

    px = (
        bw
        *
        expansion
    )

    py = (
        bh
        *
        expansion
    )

    x1 = int(
        max(
            0,
            np.floor(
                x1 - px
            )
        )
    )

    y1 = int(
        max(
            0,
            np.floor(
                y1 - py
            )
        )
    )

    x2 = int(
        min(
            W,
            np.ceil(
                x2 + px
            )
        )
    )

    y2 = int(
        min(
            H,
            np.ceil(
                y2 + py
            )
        )
    )

    gate = np.zeros(
        (
            H,
            W
        ),
        dtype=bool
    )

    gate[
        y1:y2,
        x1:x2
    ] = True

    return (
        gate,
        [
            x1,
            y1,
            x2,
            y2
        ]
    )


def border_touch_ratio(
    mask,
    gate_box
):

    x1, y1, x2, y2 = (
        gate_box
    )

    if mask.sum() == 0:
        return 1.0

    border = np.zeros_like(
        mask,
        dtype=bool
    )

    thickness = 3

    border[
        y1:min(
            y2,
            y1 + thickness
        ),
        x1:x2
    ] = True

    border[
        max(
            y1,
            y2 - thickness
        ):y2,
        x1:x2
    ] = True

    border[
        y1:y2,
        x1:min(
            x2,
            x1 + thickness
        )
    ] = True

    border[
        y1:y2,
        max(
            x1,
            x2 - thickness
        ):x2
    ] = True

    touched = (
        mask
        &
        border
    ).sum()

    return float(
        touched
        /
        max(
            1,
            mask.sum()
        )
    )


# ============================================================
# VALIDATE
# ============================================================

for p in [
    MASTER,
    CONSENSUS_JSON,
    SAM2_CHECKPOINT,
]:

    print(
        "✅" if p.exists() else "❌",
        p
    )

    if not p.exists():
        raise FileNotFoundError(p)


# ============================================================
# LOAD MASTER / CONSENSUS
# ============================================================

master_pil = Image.open(
    MASTER
).convert(
    "RGB"
)

master = np.asarray(
    master_pil
)

W, H = master_pil.size

consensus = json.loads(
    CONSENSUS_JSON.read_text(
        encoding="utf-8"
    )
)


# ============================================================
# LOAD SAM2
# ============================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print()
print(
    "Loading SAM2..."
)

sam2_model = build_sam2(
    SAM2_CONFIG,
    str(
        SAM2_CHECKPOINT
    ),
    device=DEVICE
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
# PROCESS
# ============================================================

results = []


for instance in consensus:

    iid = int(
        instance[
            "inventory_id"
        ]
    )

    if iid not in TARGET_RANKS:
        continue

    name = instance[
        "inventory_name"
    ]

    clusters = instance.get(
        "clusters",
        []
    )[
        :TARGET_RANKS[iid]
    ]

    print()
    print("=" * 100)
    print(
        f"{iid:03d}. {name}"
    )
    print("=" * 100)


    for rank, cluster in enumerate(
        clusters,
        start=1
    ):

        box = np.asarray(
            cluster[
                "bbox"
            ],
            dtype=np.float32
        )

        print()
        print(
            f"CLUSTER #{rank}"
        )

        print(
            "BOX:",
            [
                round(
                    float(v),
                    2
                )
                for v in box
            ]
        )


        with torch.inference_mode():

            masks, scores, logits = (
                predictor.predict(
                    box=box,
                    multimask_output=True
                )
            )


        local_gate, local_box = (
            make_box_gate(
                box,
                W,
                H,
                LOCAL_EXPANSION
            )
        )

        original_gate, _ = (
            make_box_gate(
                box,
                W,
                H,
                0.0
            )
        )


        evaluated = []


        for m_idx in range(
            len(
                masks
            )
        ):

            raw_mask = (
                masks[
                    m_idx
                ]
                >
                0
            )

            raw_area = int(
                raw_mask.sum()
            )

            if raw_area == 0:
                continue


            local_pixels = (
                raw_mask
                &
                local_gate
            )

            containment = float(
                local_pixels.sum()
                /
                raw_area
            )

            mask = (
                local_pixels
            )

            area = int(
                mask.sum()
            )

            if area == 0:
                continue


            inside_original = int(
                (
                    mask
                    &
                    original_gate
                ).sum()
            )

            original_box_fraction = float(
                inside_original
                /
                max(
                    1,
                    area
                )
            )


            border_touch = (
                border_touch_ratio(
                    mask,
                    local_box
                )
            )


            sam_score = float(
                scores[
                    m_idx
                ]
            )


            quality = (
                0.65
                *
                sam_score
                +
                0.20
                *
                containment
                +
                0.10
                *
                original_box_fraction
                +
                0.05
                *
                (
                    1.0
                    -
                    min(
                        1.0,
                        border_touch
                        /
                        0.20
                    )
                )
            )


            evaluated.append({

                "mask_index":
                    int(
                        m_idx
                    ),

                "mask":
                    mask,

                "sam_score":
                    sam_score,

                "quality":
                    quality,

                "area":
                    area,

                "containment":
                    containment,

                "original_box_fraction":
                    original_box_fraction,

                "border_touch":
                    border_touch,
            })


        if not evaluated:

            print(
                "❌ NO SAM2 MASK"
            )

            results.append({

                "inventory_id":
                    iid,

                "inventory_name":
                    name,

                "cluster_rank":
                    rank,

                "bbox":
                    cluster[
                        "bbox"
                    ],

                "status":
                    "NO_MASK",
            })

            continue


        evaluated.sort(
            key=lambda x:
                x[
                    "quality"
                ],
            reverse=True
        )

        best = evaluated[
            0
        ]

        mask = best[
            "mask"
        ]


        safe = bool(

            best[
                "area"
            ]
            >=
            MIN_MASK_PIXELS

            and

            best[
                "sam_score"
            ]
            >=
            MIN_SAM_SCORE

            and

            best[
                "containment"
            ]
            >=
            MIN_LOCAL_CONTAINMENT

            and

            best[
                "border_touch"
            ]
            <=
            MAX_LOCAL_BORDER_TOUCH
        )


        prefix = (
            f"{iid:03d}_"
            f"{safe_name(name)}_"
            f"cluster{rank}"
        )


        # -----------------------------
        # MASK
        # -----------------------------

        MASK_PATH = (
            OBJECTS
            /
            f"{prefix}_mask.png"
        )

        Image.fromarray(
            mask.astype(
                np.uint8
            )
            *
            255
        ).save(
            MASK_PATH
        )


        # -----------------------------
        # RGBA
        # -----------------------------

        rgba = np.zeros(
            (
                H,
                W,
                4
            ),
            dtype=np.uint8
        )

        rgba[
            :,
            :,
            :3
        ] = master

        rgba[
            :,
            :,
            3
        ] = (
            mask.astype(
                np.uint8
            )
            *
            255
        )


        RGBA_PATH = (
            OBJECTS
            /
            f"{prefix}_rgba.png"
        )

        Image.fromarray(
            rgba,
            mode="RGBA"
        ).save(
            RGBA_PATH
        )


        # -----------------------------
        # VISUAL PREVIEW
        # -----------------------------

        diagnostic = master_pil.copy()

        draw = ImageDraw.Draw(
            diagnostic
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
            width=3
        )

        mask_outline = (
            mask
            &
            ~(
                np.roll(
                    mask,
                    1,
                    axis=0
                )
                &
                np.roll(
                    mask,
                    -1,
                    axis=0
                )
                &
                np.roll(
                    mask,
                    1,
                    axis=1
                )
                &
                np.roll(
                    mask,
                    -1,
                    axis=1
                )
            )
        )

        arr = np.asarray(
            diagnostic
        ).copy()

        arr[
            mask_outline
        ] = [
            255,
            0,
            0
        ]

        PREVIEW_PATH = (
            OBJECTS
            /
            f"{prefix}_preview.png"
        )

        Image.fromarray(
            arr
        ).save(
            PREVIEW_PATH
        )


        result = {

            "inventory_id":
                iid,

            "inventory_name":
                name,

            "cluster_rank":
                rank,

            "bbox":
                cluster[
                    "bbox"
                ],

            "dino_max_score":
                cluster[
                    "max_dino_score"
                ],

            "route_count":
                cluster[
                    "route_count"
                ],

            "proposal_count":
                cluster[
                    "proposal_count"
                ],

            "sam2_selected_mask_index":
                best[
                    "mask_index"
                ],

            "sam2_score":
                best[
                    "sam_score"
                ],

            "sam2_quality":
                best[
                    "quality"
                ],

            "mask_area":
                best[
                    "area"
                ],

            "local_containment":
                best[
                    "containment"
                ],

            "original_box_fraction":
                best[
                    "original_box_fraction"
                ],

            "border_touch":
                best[
                    "border_touch"
                ],

            "passes_geometry_gate":
                safe,

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
        }


        results.append(
            result
        )


        print(
            "SAM2 score:",
            round(
                best[
                    "sam_score"
                ],
                4
            )
        )

        print(
            "Quality:",
            round(
                best[
                    "quality"
                ],
                4
            )
        )

        print(
            "Area:",
            best[
                "area"
            ]
        )

        print(
            "Containment:",
            round(
                best[
                    "containment"
                ],
                4
            )
        )

        print(
            "Box fraction:",
            round(
                best[
                    "original_box_fraction"
                ],
                4
            )
        )

        print(
            "Border touch:",
            round(
                best[
                    "border_touch"
                ],
                4
            )
        )

        print(
            "GEOMETRY GATE:",
            safe
        )


# ============================================================
# SAVE RESULTS
# ============================================================

RESULT_PATH = (
    OUT
    / "00_sam2_candidate_evaluation.json"
)

RESULT_PATH.write_text(
    json.dumps(
        results,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


print()
print("=" * 110)
print("PRODUCTION STAGE 06C7A RESULT")
print("=" * 110)


for iid in TARGET_RANKS:

    rows = [
        r
        for r in results
        if int(
            r[
                "inventory_id"
            ]
        ) == iid
    ]

    print()
    print(
        f"{iid:03d}"
    )

    for row in rows:

        if row.get(
            "status"
        ) == "NO_MASK":

            print(
                f'   cluster #{row["cluster_rank"]} | NO MASK'
            )

            continue

        print(
            "   cluster #{} | SAM={:.4f} | quality={:.4f} | area={} | contain={:.4f} | border={:.4f} | gate={}".format(

                row[
                    "cluster_rank"
                ],

                row[
                    "sam2_score"
                ],

                row[
                    "sam2_quality"
                ],

                row[
                    "mask_area"
                ],

                row[
                    "local_containment"
                ],

                row[
                    "border_touch"
                ],

                row[
                    "passes_geometry_gate"
                ],
            )
        )


print()
print(
    "RESULT JSON:",
    RESULT_PATH
)

print(
    "OBJECT PREVIEWS:",
    OBJECTS
)

print()
print(
    "DIAGNOSTIC ONLY — NO FINAL PROP MASK CREATED."
)
