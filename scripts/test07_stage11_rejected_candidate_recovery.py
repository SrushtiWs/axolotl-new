
from pathlib import Path
import argparse
import gc
import json
import sys

import cv2
import numpy as np
import torch
from PIL import Image


# ============================================================
# SAM2 IMPORT
# ============================================================

for source in [
    Path("/workspace/sam2_src"),
    Path("/workspace/axolotl/sam2"),
]:
    if source.exists() and str(source) not in sys.path:
        sys.path.insert(0, str(source))

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# ============================================================
# TEST07 — STAGE11
#
# REJECTED-CANDIDATE RECOVERY
# + FRESH LOCAL SAM2 SEGMENTATION
#
# PURPOSE:
#
# For inventory instances still unresolved after Stage10:
#
#   Stage03 original proposals
#       ↓
#   recover strongest alternative localization
#       ↓
#   reject giant / structural proposals
#       ↓
#   fresh SAM2 from ORIGINAL RGB crop
#       ↓
#   structure + geometry safety
#       ↓
#   merge only missing pixels into Stage10 trusted base
#
# NO object-specific rules.
# NO manual coordinates.
# ============================================================


SAM2_CONFIG = (
    "configs/sam2.1/"
    "sam2.1_hiera_l.yaml"
)

CROP_MARGIN = 0.18
GATE_EXPANSION = 0.10

MIN_DINO_SCORE = 0.20

MAX_SOURCE_AREA_RATIO = 0.22
MAX_SOURCE_WIDTH_RATIO = 0.80
MAX_SOURCE_HEIGHT_RATIO = 0.90

MIN_RECOVERY_PIXELS = 5

MAX_CROP_FILL = 0.74
MAX_GATE_OCCUPANCY = 0.88
MAX_OUTSIDE_GATE_RATIO = 0.22

MAX_GLOBAL_RECOVERY_RATIO = 0.06


# ============================================================
# HELPERS
# ============================================================

def find_checkpoint(explicit=None):

    candidates = []

    if explicit:
        candidates.append(Path(explicit))

    candidates += [
        Path(
            "/workspace/axolotl/test06a_temp/"
            "models/sam2.1_hiera_large.pt"
        ),
        Path(
            "/workspace/models/"
            "sam2.1_hiera_large.pt"
        ),
        Path(
            "/workspace/sam2_src/checkpoints/"
            "sam2.1_hiera_large.pt"
        ),
    ]

    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        "SAM2 checkpoint not found."
    )


def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def clip_box(
    box,
    W,
    H
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
    ratio,
    W,
    H
):

    x1, y1, x2, y2 = box

    bw = x2 - x1
    bh = y2 - y1

    dx = bw * ratio
    dy = bh * ratio

    return [
        max(
            0,
            int(
                np.floor(
                    x1 - dx
                )
            )
        ),

        max(
            0,
            int(
                np.floor(
                    y1 - dy
                )
            )
        ),

        min(
            W,
            int(
                np.ceil(
                    x2 + dx
                )
            )
        ),

        min(
            H,
            int(
                np.ceil(
                    y2 + dy
                )
            )
        )
    ]


def box_mask(
    box,
    H,
    W
):

    x1, y1, x2, y2 = box

    mask = np.zeros(
        (
            H,
            W
        ),
        dtype=bool
    )

    mask[
        y1:y2,
        x1:x2
    ] = True

    return mask


def largest_component(mask):

    n, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask.astype(
                np.uint8
            ),
            8
        )
    )

    if n <= 1:
        return mask

    idx = (
        1
        +
        np.argmax(
            stats[
                1:,
                cv2.CC_STAT_AREA
            ]
        )
    )

    return (
        labels
        ==
        idx
    )


def crop_edge_count(mask):

    if not mask.any():
        return 4

    values = [
        float(
            mask[
                0,
                :
            ].mean()
        ),

        float(
            mask[
                -1,
                :
            ].mean()
        ),

        float(
            mask[
                :,
                0
            ].mean()
        ),

        float(
            mask[
                :,
                -1
            ].mean()
        ),
    ]

    return sum(
        v > 0.15
        for v in values
    )


# ============================================================
# SOURCE CANDIDATE SCORING
# ============================================================

def source_candidate_score(
    row,
    W,
    H
):

    box = clip_box(
        row[
            "bbox"
        ],
        W,
        H
    )

    x1, y1, x2, y2 = box

    bw = (
        x2 - x1
    )

    bh = (
        y2 - y1
    )

    area = (
        bw
        *
        bh
    )

    image_area = float(
        W
        *
        H
    )

    area_ratio = (
        area
        /
        image_area
    )

    width_ratio = (
        bw
        /
        float(W)
    )

    height_ratio = (
        bh
        /
        float(H)
    )

    dino_score = float(
        row.get(
            "score",
            0.0
        )
    )

    dino_label = str(
        row.get(
            "dino_label",
            ""
        )
    ).lower()

    inventory_name = str(
        row.get(
            "inventory_name",
            ""
        )
    ).lower()

    grounding_phrase = str(
        row.get(
            "grounding_phrase",
            ""
        )
    ).lower()


    # --------------------------------------------------------
    # Semantic label agreement
    # --------------------------------------------------------

    semantic_bonus = 0.0

    inventory_tokens = {
        t
        for t in inventory_name.split()
        if len(t) >= 3
    }

    label_tokens = {
        t
        for t in dino_label.split()
        if len(t) >= 3
    }

    phrase_tokens = {
        t
        for t in grounding_phrase.split()
        if len(t) >= 3
    }


    if (
        inventory_tokens
        &
        label_tokens
    ):

        semantic_bonus += 0.75


    if (
        inventory_tokens
        &
        phrase_tokens
    ):

        semantic_bonus += 0.25


    # --------------------------------------------------------
    # Geometry penalty
    # --------------------------------------------------------

    geometry_penalty = 0.0


    if area_ratio > 0.25:
        geometry_penalty += 2.5

    elif area_ratio > 0.15:
        geometry_penalty += 1.2


    if width_ratio > 0.80:
        geometry_penalty += 2.0


    if height_ratio > 0.90:
        geometry_penalty += 1.5


    # Prefer localized proposals when detector evidence is good.
    compact_bonus = 0.0

    if (
        area_ratio < 0.08
        and
        dino_score >= 0.30
    ):

        compact_bonus += 0.65


    score = (

        dino_score
        *
        2.0

        +

        semantic_bonus

        +

        compact_bonus

        -

        geometry_penalty
    )


    metrics = {

        "bbox":
            box,

        "dino_score":
            dino_score,

        "dino_label":
            dino_label,

        "area_ratio":
            area_ratio,

        "width_ratio":
            width_ratio,

        "height_ratio":
            height_ratio,

        "semantic_bonus":
            semantic_bonus,

        "compact_bonus":
            compact_bonus,

        "geometry_penalty":
            geometry_penalty,

        "source_score":
            score
    }


    return (
        score,
        metrics
    )


# ============================================================
# MAIN
# ============================================================

def run(
    master_path,
    base_mask_path,
    stage03_json,
    stage10_report,
    output_dir,
    checkpoint_path=None
):

    master_path = Path(
        master_path
    )

    base_mask_path = Path(
        base_mask_path
    )

    stage03_json = Path(
        stage03_json
    )

    stage10_report = Path(
        stage10_report
    )

    output_dir = Path(
        output_dir
    )


    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    objects_dir = (
        output_dir
        /
        "objects"
    )


    objects_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    for p in [
        master_path,
        base_mask_path,
        stage03_json,
        stage10_report
    ]:

        if not p.exists():
            raise FileNotFoundError(p)


    checkpoint = (
        find_checkpoint(
            checkpoint_path
        )
    )


    # ========================================================
    # LOAD
    # ========================================================

    master_pil = (
        Image.open(
            master_path
        )
        .convert(
            "RGB"
        )
    )


    master = np.asarray(
        master_pil
    )


    H, W = (
        master.shape[:2]
    )


    base = (
        np.asarray(
            Image.open(
                base_mask_path
            )
            .convert(
                "L"
            )
        )
        >
        127
    )


    if base.shape != (
        H,
        W
    ):

        raise RuntimeError(
            "Base mask size mismatch."
        )


    stage03 = load_json(
        stage03_json
    )


    stage10 = load_json(
        stage10_report
    )


    unresolved_ids = [
        int(x)
        for x in stage10.get(
            "remaining_unresolved_inventory_ids",
            []
        )
    ]


    print(
        "=" * 90
    )

    print(
        "TEST07 STAGE11"
    )

    print(
        "REJECTED-CANDIDATE RECOVERY"
    )

    print(
        "=" * 90
    )


    print(
        "BASE PIXELS:",
        int(
            base.sum()
        )
    )


    print(
        "UNRESOLVED IDS:",
        unresolved_ids
    )


    # ========================================================
    # LOAD SAM2
    # ========================================================

    device = (
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
            checkpoint
        ),
        device=device
    )


    predictor = (
        SAM2ImagePredictor(
            sam2_model
        )
    )


    print(
        "✅ SAM2 READY"
    )


    accepted_union = np.zeros(
        (
            H,
            W
        ),
        dtype=bool
    )


    rejected_union = np.zeros(
        (
            H,
            W
        ),
        dtype=bool
    )


    results = []


    # ========================================================
    # PER UNRESOLVED INSTANCE
    # ========================================================

    for iid in unresolved_ids:

        rows = [
            r
            for r in stage03
            if int(
                r.get(
                    "inventory_id",
                    -1
                )
            )
            ==
            iid
        ]


        print()
        print(
            "-" * 90
        )

        print(
            "INSTANCE:",
            iid
        )


        if not rows:

            print(
                "❌ No Stage03 candidates"
            )

            results.append({

                "inventory_id":
                    iid,

                "decision":
                    "REJECT",

                "reason":
                    "no_stage03_candidates"
            })

            continue


        # ====================================================
        # SCORE ALL ORIGINAL CANDIDATES
        # ====================================================

        scored = []


        for row in rows:

            score, metrics = (
                source_candidate_score(
                    row,
                    W,
                    H
                )
            )


            scored.append({

                "row":
                    row,

                "score":
                    score,

                "metrics":
                    metrics
            })


            print(
                "candidate",
                [
                    round(
                        float(v),
                        1
                    )
                    for v
                    in row[
                        "bbox"
                    ]
                ],
                "| DINO:",
                round(
                    float(
                        row.get(
                            "score",
                            0
                        )
                    ),
                    3
                ),
                "| label:",
                row.get(
                    "dino_label"
                ),
                "| source_score:",
                round(
                    score,
                    3
                )
            )


        scored.sort(
            key=lambda x:
                x[
                    "score"
                ],
            reverse=True
        )


        best_source = (
            scored[
                0
            ]
        )


        row = (
            best_source[
                "row"
            ]
        )


        metrics = (
            best_source[
                "metrics"
            ]
        )


        name = str(
            row.get(
                "inventory_name",
                "object"
            )
        )


        box = (
            metrics[
                "bbox"
            ]
        )


        # ====================================================
        # SOURCE SAFETY
        # ====================================================

        source_safe = bool(

            metrics[
                "dino_score"
            ]
            >=
            MIN_DINO_SCORE

            and

            metrics[
                "area_ratio"
            ]
            <=
            MAX_SOURCE_AREA_RATIO

            and

            metrics[
                "width_ratio"
            ]
            <=
            MAX_SOURCE_WIDTH_RATIO

            and

            metrics[
                "height_ratio"
            ]
            <=
            MAX_SOURCE_HEIGHT_RATIO
        )


        if not source_safe:

            print(
                "❌ Best source localization still unsafe"
            )


            results.append({

                "inventory_id":
                    iid,

                "inventory_name":
                    name,

                "decision":
                    "REJECT",

                "reason":
                    "no_safe_source_localization",

                "source_candidates": [
                    {
                        "bbox":
                            x[
                                "metrics"
                            ][
                                "bbox"
                            ],

                        "source_score":
                            x[
                                "score"
                            ],

                        "dino_score":
                            x[
                                "metrics"
                            ][
                                "dino_score"
                            ],

                        "dino_label":
                            x[
                                "metrics"
                            ][
                                "dino_label"
                            ]
                    }
                    for x
                    in scored
                ]
            })

            continue


        print(
            "✅ RECOVERED SOURCE BOX:",
            [
                round(
                    float(v),
                    1
                )
                for v
                in box
            ]
        )


        # ====================================================
        # CROP + GATE
        # ====================================================

        crop_box = expand_box(
            box,
            CROP_MARGIN,
            W,
            H
        )


        gate_box = expand_box(
            box,
            GATE_EXPANSION,
            W,
            H
        )


        cx1, cy1, cx2, cy2 = (
            crop_box
        )


        crop = (
            master[
                cy1:cy2,
                cx1:cx2
            ]
        )


        ch, cw = (
            crop.shape[:2]
        )


        x1, y1, x2, y2 = (
            box
        )


        local_box = np.array(
            [
                x1 - cx1,
                y1 - cy1,
                x2 - cx1,
                y2 - cy1
            ],
            dtype=np.float32
        )


        global_gate = box_mask(
            gate_box,
            H,
            W
        )


        predictor.set_image(
            crop
        )


        # ====================================================
        # FRESH SAM2 PROMPTS
        # ====================================================

        proposals = []


        with torch.inference_mode():

            masks_a, scores_a, _ = (
                predictor.predict(
                    box=
                        local_box,
                    multimask_output=
                        True
                )
            )


        for mi in range(
            len(
                masks_a
            )
        ):

            proposals.append(
                (
                    "box",
                    mi,
                    masks_a[
                        mi
                    ],
                    float(
                        scores_a[
                            mi
                        ]
                    )
                )
            )


        center_x = (
            (
                x1
                +
                x2
            )
            /
            2
            -
            cx1
        )


        center_y = (
            (
                y1
                +
                y2
            )
            /
            2
            -
            cy1
        )


        with torch.inference_mode():

            masks_b, scores_b, _ = (
                predictor.predict(
                    point_coords=
                        np.array(
                            [
                                [
                                    center_x,
                                    center_y
                                ]
                            ],
                            dtype=
                                np.float32
                        ),

                    point_labels=
                        np.array(
                            [
                                1
                            ],
                            dtype=
                                np.int32
                        ),

                    box=
                        local_box,

                    multimask_output=
                        True
                )
            )


        for mi in range(
            len(
                masks_b
            )
        ):

            proposals.append(
                (
                    "box+center",
                    mi,
                    masks_b[
                        mi
                    ],
                    float(
                        scores_b[
                            mi
                        ]
                    )
                )
            )


        # ====================================================
        # EVALUATE
        # ====================================================

        evaluated = []


        for (
            prompt_type,
            mask_index,
            local_mask,
            sam_score
        ) in proposals:

            local_mask = (
                local_mask
                >
                0
            )


            if not local_mask.any():
                continue


            local_mask = (
                largest_component(
                    local_mask
                )
            )


            local_area = int(
                local_mask.sum()
            )


            crop_fill = (
                local_area
                /
                max(
                    1,
                    ch
                    *
                    cw
                )
            )


            edge_count = (
                crop_edge_count(
                    local_mask
                )
            )


            global_mask = np.zeros(
                (
                    H,
                    W
                ),
                dtype=bool
            )


            global_mask[
                cy1:cy2,
                cx1:cx2
            ] = (
                local_mask
            )


            pre_gate_area = int(
                global_mask.sum()
            )


            outside_gate = int(
                (
                    global_mask
                    &
                    (~global_gate)
                ).sum()
            )


            outside_gate_ratio = (
                outside_gate
                /
                max(
                    1,
                    pre_gate_area
                )
            )


            global_mask &= (
                global_gate
            )


            area = int(
                global_mask.sum()
            )


            if area == 0:
                continue


            gate_occupancy = (
                area
                /
                max(
                    1,
                    int(
                        global_gate.sum()
                    )
                )
            )


            base_inside_gate = (
                base
                &
                global_gate
            )


            base_pixels = int(
                base_inside_gate.sum()
            )


            base_covered = int(
                (
                    global_mask
                    &
                    base_inside_gate
                ).sum()
            )


            if base_pixels > 0:

                base_coverage = (
                    base_covered
                    /
                    base_pixels
                )

            else:

                base_coverage = 0.0


            recovery = (
                global_mask
                &
                (~base)
            )


            recovery_pixels = int(
                recovery.sum()
            )


            # ================================================
            # STRUCTURE RISK
            # ================================================

            structural_risk = 0.0


            if crop_fill > 0.74:

                structural_risk += 4.0

            elif crop_fill > 0.58:

                structural_risk += 1.8


            if edge_count >= 3:

                structural_risk += 3.0

            elif edge_count == 2:

                structural_risk += 0.8


            if gate_occupancy > 0.90:

                structural_risk += 2.0


            if outside_gate_ratio > 0.20:

                structural_risk += 1.5


            quality = (

                sam_score
                *
                1.4

                +

                base_coverage
                *
                1.8

                +

                (
                    1.0
                    -
                    min(
                        crop_fill,
                        1.0
                    )
                )
                *
                0.25

                -

                structural_risk
            )


            evaluated.append({

                "prompt_type":
                    prompt_type,

                "mask_index":
                    mask_index,

                "sam_score":
                    sam_score,

                "quality":
                    quality,

                "mask":
                    global_mask,

                "recovery":
                    recovery,

                "recovery_pixels":
                    recovery_pixels,

                "crop_fill":
                    crop_fill,

                "edge_count":
                    edge_count,

                "gate_occupancy":
                    gate_occupancy,

                "outside_gate_ratio":
                    outside_gate_ratio,

                "base_pixels":
                    base_pixels,

                "base_coverage":
                    base_coverage,

                "structural_risk":
                    structural_risk
            })


        if not evaluated:

            results.append({

                "inventory_id":
                    iid,

                "inventory_name":
                    name,

                "decision":
                    "REJECT",

                "reason":
                    "no_fresh_sam2_proposal"
            })

            continue


        evaluated.sort(
            key=lambda x:
                x[
                    "quality"
                ],
            reverse=True
        )


        best = (
            evaluated[
                0
            ]
        )


        # ====================================================
        # FINAL SAFETY
        # ====================================================

        decision = (
            "ACCEPT"
        )


        reasons = []


        # ====================================================
        # RECOVERY / ALREADY-PRESENT LOGIC
        #
        # A verified fresh object mask may already be almost
        # completely contained in the trusted base.
        #
        # In that situation recovery_pixels can legitimately
        # be zero. That means there is nothing missing to add;
        # it does NOT mean the object is unresolved.
        # ====================================================

        already_present = bool(

            best[
                "recovery_pixels"
            ]
            <
            MIN_RECOVERY_PIXELS

            and

            best[
                "base_coverage"
            ]
            >=
            0.85
        )


        if (
            best[
                "recovery_pixels"
            ]
            <
            MIN_RECOVERY_PIXELS

            and

            not already_present
        ):

            decision = (
                "REJECT"
            )

            reasons.append(
                "almost_no_new_pixels_without_base_support"
            )


        if (
            best[
                "crop_fill"
            ]
            >
            MAX_CROP_FILL
        ):

            decision = (
                "REJECT"
            )

            reasons.append(
                "crop_filling_surface"
            )


        if (
            best[
                "gate_occupancy"
            ]
            >
            MAX_GATE_OCCUPANCY
        ):

            decision = (
                "REJECT"
            )

            reasons.append(
                "large_gate_fill"
            )


        if (
            best[
                "outside_gate_ratio"
            ]
            >
            MAX_OUTSIDE_GATE_RATIO
        ):

            decision = (
                "REJECT"
            )

            reasons.append(
                "extends_outside_verified_gate"
            )


        if (
            best[
                "edge_count"
            ]
            >=
            3
        ):

            decision = (
                "REJECT"
            )

            reasons.append(
                "multi_edge_architectural_mask"
            )


        recovery_ratio = (
            best[
                "recovery_pixels"
            ]
            /
            float(
                W
                *
                H
            )
        )


        if (
            recovery_ratio
            >
            MAX_GLOBAL_RECOVERY_RATIO
        ):

            decision = (
                "REJECT"
            )

            reasons.append(
                "catastrophic_global_recovery"
            )


        # If the fresh segmentation is strongly supported by
        # the existing trusted base, zero new recovery pixels
        # means "already preserved", not failure.
        if (
            already_present
            and
            decision == "ACCEPT"
        ):

            reasons.append(
                "verified_object_already_present_in_base"
            )


        recovery = (
            best[
                "recovery"
            ]
        )


        if decision == "ACCEPT":

            if already_present:

                resolution_status = (
                    "RESOLVED_ALREADY_PRESENT"
                )

            else:

                resolution_status = (
                    "RESOLVED_RECOVERED"
                )

                accepted_union |= (
                    recovery
                )

        else:

            resolution_status = (
                "UNRESOLVED"
            )

            rejected_union |= (
                recovery
            )


        print(
            "RESULT:",
            decision
        )


        print(
            "SAM:",
            round(
                best[
                    "sam_score"
                ],
                3
            ),
            "| recovery:",
            best[
                "recovery_pixels"
            ],
            "| crop_fill:",
            round(
                best[
                    "crop_fill"
                ],
                3
            ),
            "| gate:",
            round(
                best[
                    "gate_occupancy"
                ],
                3
            ),
            "| outside:",
            round(
                best[
                    "outside_gate_ratio"
                ],
                3
            ),
            "| base:",
            round(
                best[
                    "base_coverage"
                ],
                3
            ),
            "| quality:",
            round(
                best[
                    "quality"
                ],
                3
            )
        )


        if reasons:

            print(
                "REASONS:",
                reasons
            )


        # ====================================================
        # SAVE OBJECT DEBUG
        # ====================================================

        prefix = (
            objects_dir
            /
            f"{iid:02d}_{name.replace(' ', '_')}"
        )


        Image.fromarray(
            crop
        ).save(
            str(
                prefix
            )
            +
            "_crop.png"
        )


        Image.fromarray(
            best[
                "mask"
            ].astype(
                np.uint8
            )
            *
            255
        ).save(
            str(
                prefix
            )
            +
            "_fresh_mask.png"
        )


        Image.fromarray(
            recovery.astype(
                np.uint8
            )
            *
            255
        ).save(
            str(
                prefix
            )
            +
            "_recovery.png"
        )


        results.append({

            "inventory_id":
                iid,

            "inventory_name":
                name,

            "decision":
                decision,

            "resolution_status":
                resolution_status,

            "already_present_in_base":
                bool(
                    already_present
                ),

            "reasons":
                reasons,

            "source_bbox":
                box,

            "source_dino_score":
                metrics[
                    "dino_score"
                ],

            "source_dino_label":
                metrics[
                    "dino_label"
                ],

            "source_score":
                metrics[
                    "source_score"
                ],

            "best_prompt":
                best[
                    "prompt_type"
                ],

            "sam_score":
                best[
                    "sam_score"
                ],

            "quality":
                best[
                    "quality"
                ],

            "recovery_pixels":
                best[
                    "recovery_pixels"
                ],

            "crop_fill":
                best[
                    "crop_fill"
                ],

            "gate_occupancy":
                best[
                    "gate_occupancy"
                ],

            "outside_gate_ratio":
                best[
                    "outside_gate_ratio"
                ],

            "base_coverage":
                best[
                    "base_coverage"
                ],

            "edge_count":
                best[
                    "edge_count"
                ]
        })


    # ========================================================
    # FINAL MASK
    # ========================================================

    final_mask = (
        base
        |
        accepted_union
    )


    accepted_ids = {
        int(
            r[
                "inventory_id"
            ]
        )
        for r in results
        if r.get(
            "decision"
        )
        ==
        "ACCEPT"
    }


    remaining_unresolved = sorted(
        set(
            unresolved_ids
        )
        -
        accepted_ids
    )


    # ========================================================
    # SAVE
    # ========================================================

    accepted_path = (
        output_dir
        /
        "00_recovered_candidate_accepted.png"
    )


    rejected_path = (
        output_dir
        /
        "01_recovered_candidate_rejected.png"
    )


    final_path = (
        output_dir
        /
        "02_final_props_mask.png"
    )


    Image.fromarray(
        accepted_union.astype(
            np.uint8
        )
        *
        255
    ).save(
        accepted_path
    )


    Image.fromarray(
        rejected_union.astype(
            np.uint8
        )
        *
        255
    ).save(
        rejected_path
    )


    Image.fromarray(
        final_mask.astype(
            np.uint8
        )
        *
        255
    ).save(
        final_path
    )


    # ========================================================
    # TRANSPARENCY PREVIEW
    # ========================================================

    tile = 24

    yy, xx = np.indices(
        (
            H,
            W
        )
    )


    checker = (
        (
            (
                xx // tile
                +
                yy // tile
            )
            %
            2
        )
        *
        55
        +
        200
    ).astype(
        np.uint8
    )


    checker_rgb = np.stack(
        [
            checker,
            checker,
            checker
        ],
        axis=-1
    )


    preview = (
        checker_rgb.copy()
    )


    preview[
        final_mask
    ] = (
        master[
            final_mask
        ]
    )


    preview_path = (
        output_dir
        /
        "03_transparency_preview.png"
    )


    Image.fromarray(
        preview
    ).save(
        preview_path
    )


    results_path = (
        output_dir
        /
        "04_results.json"
    )


    results_path.write_text(
        json.dumps(
            results,
            indent=2
        )
    )


    report = {

        "experiment":
            "TEST07_STAGE11",

        "method":
            (
                "recover unresolved object from original "
                "Stage03 candidate alternatives, then fresh "
                "local SAM2 segmentation"
            ),

        "base_pixels":
            int(
                base.sum()
            ),

        "input_unresolved_ids":
            unresolved_ids,

        "accepted_ids":
            sorted(
                accepted_ids
            ),

        "remaining_unresolved_ids":
            remaining_unresolved,

        "accepted_recovery_pixels":
            int(
                accepted_union.sum()
            ),

        "rejected_recovery_pixels":
            int(
                rejected_union.sum()
            ),

        "final_pixels":
            int(
                final_mask.sum()
            ),

        "objects":
            results
    }


    report_path = (
        output_dir
        /
        "stage11_report.json"
    )


    report_path.write_text(
        json.dumps(
            report,
            indent=2
        )
    )


    # ========================================================
    # RESULT
    # ========================================================

    print()
    print(
        "=" * 90
    )

    print(
        "TEST07 STAGE11 RESULT"
    )

    print(
        "=" * 90
    )


    print(
        "BASE PIXELS:",
        int(
            base.sum()
        )
    )


    print(
        "INPUT UNRESOLVED:",
        unresolved_ids
    )


    print(
        "ACCEPTED IDS:",
        sorted(
            accepted_ids
        )
    )


    print(
        "REMAINING UNRESOLVED:",
        remaining_unresolved
    )


    print(
        "ACCEPTED RECOVERY PIXELS:",
        int(
            accepted_union.sum()
        )
    )


    print(
        "FINAL PIXELS:",
        int(
            final_mask.sum()
        )
    )


    print()
    print(
        "IMPORTANT OUTPUT:"
    )

    print(
        preview_path
    )


    del predictor
    del sam2_model

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--master",
        required=True
    )


    parser.add_argument(
        "--base-mask",
        required=True
    )


    parser.add_argument(
        "--stage03-json",
        required=True
    )


    parser.add_argument(
        "--stage10-report",
        required=True
    )


    parser.add_argument(
        "--output-dir",
        required=True
    )


    parser.add_argument(
        "--sam2-checkpoint",
        default=None
    )


    args = parser.parse_args()


    run(
        master_path=
            args.master,

        base_mask_path=
            args.base_mask,

        stage03_json=
            args.stage03_json,

        stage10_report=
            args.stage10_report,

        output_dir=
            args.output_dir,

        checkpoint_path=
            args.sam2_checkpoint
    )
