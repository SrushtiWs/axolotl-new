
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
# TEST07 — STAGE10
#
# LOCAL-CROP SAM2 + ARCHITECTURAL SURFACE REJECTION
#
# INPUT:
#   locked resized master
#   Stage08 trusted final foreground
#   Stage09 unresolved multimask results
#   Stage06 unresolved candidates
#
# PURPOSE:
#   Verify Stage09 plausible recovery proposals locally before
#   allowing their missing pixels into the trusted foreground.
#
# IMPORTANT:
#   - Stage08 is primary and never replaced
#   - Stage09 catastrophic proposals are excluded
#   - SAM2 reruns LOCALLY
#   - large crop-filling / multi-edge masks rejected
#   - only missing pixels outside Stage08 may be added
#   - mirror's bad giant box is NOT reused
#   - exact RGB from locked master
# ============================================================


SAM2_CONFIG = (
    "configs/sam2.1/"
    "sam2.1_hiera_l.yaml"
)

CROP_MARGIN = 0.12

MIN_RECOVERY_PIXELS = 8

# Same catastrophic rule introduced after Stage09.
MAX_GLOBAL_RECOVERY_RATIO = 0.08
MAX_GLOBAL_MASK_RATIO = 0.18
MIN_GLOBAL_BAD_MISSING_RATIO = 0.45


# ============================================================
# HELPERS
# ============================================================

def find_checkpoint(explicit=None):

    candidates = []

    if explicit:
        candidates.append(
            Path(explicit)
        )

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


def touches_edges(mask):

    h, w = mask.shape

    return {
        "top":
            float(
                mask[0, :].mean()
            ),

        "bottom":
            float(
                mask[-1, :].mean()
            ),

        "left":
            float(
                mask[:, 0].mean()
            ),

        "right":
            float(
                mask[:, -1].mean()
            )
    }


def edge_touch_score(mask):

    values = touches_edges(
        mask
    )

    edge_count = sum(
        v > 0.20
        for v in values.values()
    )

    return (
        edge_count,
        values
    )


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


def contour_complexity(mask):

    contours, _ = cv2.findContours(
        mask.astype(
            np.uint8
        ),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return 0.0

    cnt = max(
        contours,
        key=cv2.contourArea
    )

    area = cv2.contourArea(
        cnt
    )

    perimeter = cv2.arcLength(
        cnt,
        True
    )

    if area <= 1:
        return 0.0

    return float(
        (
            perimeter
            *
            perimeter
        )
        /
        (
            4.0
            *
            np.pi
            *
            area
        )
    )


def central_support(mask):

    h, w = mask.shape

    y1 = int(
        h * 0.20
    )

    y2 = int(
        h * 0.80
    )

    x1 = int(
        w * 0.20
    )

    x2 = int(
        w * 0.80
    )

    total = int(
        mask.sum()
    )

    if total == 0:
        return 0.0

    center = (
        mask[
            y1:y2,
            x1:x2
        ]
    )

    return float(
        center.sum()
        /
        total
    )


def clip_box(
    box,
    W,
    H
):

    x1, y1, x2, y2 = [
        int(
            round(
                float(v)
            )
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

    return [
        x1,
        y1,
        x2,
        y2
    ]


def make_crop(
    box,
    W,
    H,
    margin_ratio=0.12
):

    x1, y1, x2, y2 = box

    bw = (
        x2 - x1
    )

    bh = (
        y2 - y1
    )

    mx = max(
        4,
        int(
            round(
                bw
                *
                margin_ratio
            )
        )
    )

    my = max(
        4,
        int(
            round(
                bh
                *
                margin_ratio
            )
        )
    )

    return (
        max(
            0,
            x1 - mx
        ),

        max(
            0,
            y1 - my
        ),

        min(
            W,
            x2 + mx
        ),

        min(
            H,
            y2 + my
        )
    )


def local_prompt_box(
    original_box,
    crop_box
):

    x1, y1, x2, y2 = (
        original_box
    )

    cx1, cy1, cx2, cy2 = (
        crop_box
    )

    return np.array(
        [
            x1 - cx1,
            y1 - cy1,
            x2 - cx1,
            y2 - cy1
        ],
        dtype=np.float32
    )


def safe_name(text):

    return "".join(
        c
        if c.isalnum() or c in "-_"
        else "_"
        for c in str(text)
    )[:50]


# ============================================================
# MAIN
# ============================================================

def run(
    master_path,
    base_mask_path,
    stage09_json,
    unresolved_json,
    output_dir,
    checkpoint_path=None
):

    master_path = Path(
        master_path
    )

    base_mask_path = Path(
        base_mask_path
    )

    stage09_json = Path(
        stage09_json
    )

    unresolved_json = Path(
        unresolved_json
    )

    output_dir = Path(
        output_dir
    )


    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    object_dir = (
        output_dir
        /
        "objects"
    )

    object_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    for p in [
        master_path,
        base_mask_path,
        stage09_json,
        unresolved_json
    ]:

        if not p.exists():
            raise FileNotFoundError(p)


    checkpoint = (
        find_checkpoint(
            checkpoint_path
        )
    )


    # ========================================================
    # LOAD MASTER + BASE
    # ========================================================

    master_bgr = cv2.imread(
        str(
            master_path
        ),
        cv2.IMREAD_COLOR
    )

    if master_bgr is None:
        raise RuntimeError(
            "Could not read master."
        )

    master_rgb = cv2.cvtColor(
        master_bgr,
        cv2.COLOR_BGR2RGB
    )

    H, W = (
        master_rgb.shape[:2]
    )


    base_u8 = cv2.imread(
        str(
            base_mask_path
        ),
        cv2.IMREAD_GRAYSCALE
    )

    if base_u8 is None:
        raise RuntimeError(
            "Could not read base mask."
        )

    if base_u8.shape != (
        H,
        W
    ):
        raise RuntimeError(
            "Base mask shape mismatch."
        )

    base = (
        base_u8
        >
        127
    )


    with open(
        stage09_json,
        "r",
        encoding="utf-8"
    ) as f:

        stage09 = json.load(
            f
        )


    with open(
        unresolved_json,
        "r",
        encoding="utf-8"
    ) as f:

        unresolved = json.load(
            f
        )


    unresolved_by_id = {}

    for row in unresolved:

        iid = int(
            row[
                "inventory_id"
            ]
        )

        unresolved_by_id[
            iid
        ] = row


    # ========================================================
    # APPLY CORRECTED STAGE09 GATE
    # ========================================================

    eligible = []

    excluded = []


    for row in stage09:

        iid = int(
            row[
                "inventory_id"
            ]
        )

        recovery_pixels = int(
            row.get(
                "recovery_pixels",
                0
            )
        )

        metrics = row.get(
            "metrics",
            {}
        )

        mask_area = int(
            metrics.get(
                "area",
                0
            )
        )

        missing_ratio = float(
            metrics.get(
                "missing_ratio",
                0
            )
        )

        recovery_ratio = (
            recovery_pixels
            /
            float(
                W * H
            )
        )

        mask_ratio = (
            mask_area
            /
            float(
                W * H
            )
        )

        catastrophic = (

            recovery_ratio
            >
            MAX_GLOBAL_RECOVERY_RATIO

            or

            (
                mask_ratio
                >
                MAX_GLOBAL_MASK_RATIO

                and

                missing_ratio
                >
                MIN_GLOBAL_BAD_MISSING_RATIO
            )
        )


        stage09_accept = (
            row.get(
                "decision"
            )
            ==
            "PROPOSAL_ACCEPT"
        )


        if (
            stage09_accept
            and
            not catastrophic
        ):

            source = (
                unresolved_by_id.get(
                    iid
                )
            )

            if source is None:

                excluded.append({
                    "inventory_id":
                        iid,

                    "inventory_name":
                        row.get(
                            "inventory_name",
                            "unknown"
                        ),

                    "reason":
                        "missing_original_unresolved_record"
                })

                continue


            eligible.append({
                "inventory_id":
                    iid,

                "inventory_name":
                    row[
                        "inventory_name"
                    ],

                "box":
                    clip_box(
                        source[
                            "bbox"
                        ],
                        W,
                        H
                    ),

                "stage09":
                    row
            })


        else:

            excluded.append({

                "inventory_id":
                    iid,

                "inventory_name":
                    row.get(
                        "inventory_name",
                        "unknown"
                    ),

                "reason":
                    (
                        "stage09_catastrophic_proposal"
                        if catastrophic
                        else
                        "stage09_not_accepted"
                    ),

                "recovery_ratio":
                    recovery_ratio,

                "mask_ratio":
                    mask_ratio,

                "missing_ratio":
                    missing_ratio
            })


    print(
        "=" * 90
    )

    print(
        "TEST07 STAGE10"
    )

    print(
        "LOCAL CROP SAM2 + STRUCTURE REJECTION"
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
        "STAGE09 RECORDS:",
        len(
            stage09
        )
    )


    print(
        "ELIGIBLE FOR LOCAL VERIFICATION:",
        len(
            eligible
        )
    )


    print(
        "EXCLUDED BEFORE SAM2:",
        len(
            excluded
        )
    )


    for row in excluded:

        print(
            "EXCLUDED:",
            f'{int(row["inventory_id"]):02d}',
            row[
                "inventory_name"
            ],
            "|",
            row[
                "reason"
            ]
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
    # LOCAL VERIFICATION
    # ========================================================

    for index, obj in enumerate(
        eligible,
        start=1
    ):

        iid = int(
            obj[
                "inventory_id"
            ]
        )

        name = str(
            obj[
                "inventory_name"
            ]
        )

        box = (
            obj[
                "box"
            ]
        )


        print()
        print(
            "-" * 90
        )

        print(
            f"[{index}/{len(eligible)}]",
            f"ID={iid:02d}",
            name,
            "| box:",
            box
        )


        crop_box = (
            make_crop(
                box,
                W,
                H,
                margin_ratio=
                    CROP_MARGIN
            )
        )


        cx1, cy1, cx2, cy2 = (
            crop_box
        )


        crop_rgb = (
            master_rgb[
                cy1:cy2,
                cx1:cx2
            ]
        )


        ch, cw = (
            crop_rgb.shape[:2]
        )


        if (
            ch < 5
            or
            cw < 5
        ):

            results.append({
                "inventory_id":
                    iid,

                "inventory_name":
                    name,

                "decision":
                    "REJECT",

                "reason":
                    "crop_too_small"
            })

            continue


        local_box = (
            local_prompt_box(
                box,
                crop_box
            )
        )


        predictor.set_image(
            crop_rgb
        )


        with torch.inference_mode():

            masks, scores, logits = (
                predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=local_box,
                    multimask_output=True
                )
            )


        candidate_infos = []


        for mi in range(
            len(
                masks
            )
        ):

            m = (
                masks[
                    mi
                ]
                .astype(bool)
            )


            if m.sum() == 0:
                continue


            m = (
                largest_component(
                    m
                )
            )


            area = int(
                m.sum()
            )


            crop_area = int(
                ch
                *
                cw
            )


            area_ratio = (
                area
                /
                max(
                    1,
                    crop_area
                )
            )


            edge_count, edge_values = (
                edge_touch_score(
                    m
                )
            )


            center_ratio = (
                central_support(
                    m
                )
            )


            complexity = (
                contour_complexity(
                    m
                )
            )


            # ================================================
            # ARCHITECTURAL SURFACE RISK
            # ================================================

            surface_risk = 0.0


            if area_ratio > 0.82:

                surface_risk += 4.0

            elif area_ratio > 0.68:

                surface_risk += 2.5

            elif area_ratio > 0.55:

                surface_risk += 1.0


            if edge_count >= 4:

                surface_risk += 4.0

            elif edge_count == 3:

                surface_risk += 2.5

            elif edge_count == 2:

                surface_risk += 0.8


            if (
                area_ratio > 0.45
                and
                complexity < 1.8
            ):

                surface_risk += 2.0


            # ================================================
            # OBJECT SUPPORT
            # ================================================

            object_support = 0.0


            object_support += (
                float(
                    scores[
                        mi
                    ]
                )
                *
                2.0
            )


            object_support += (
                min(
                    center_ratio,
                    0.80
                )
                *
                1.5
            )


            if (
                0.03
                <=
                area_ratio
                <=
                0.55
            ):

                object_support += 1.0


            if complexity >= 1.3:

                object_support += 0.5


            final_score = (
                object_support
                -
                surface_risk
            )


            candidate_infos.append({

                "mask_index":
                    int(
                        mi
                    ),

                "sam_score":
                    float(
                        scores[
                            mi
                        ]
                    ),

                "area":
                    area,

                "area_ratio":
                    float(
                        area_ratio
                    ),

                "edge_count":
                    int(
                        edge_count
                    ),

                "edge_values":
                    edge_values,

                "central_support":
                    float(
                        center_ratio
                    ),

                "complexity":
                    float(
                        complexity
                    ),

                "surface_risk":
                    float(
                        surface_risk
                    ),

                "object_support":
                    float(
                        object_support
                    ),

                "final_score":
                    float(
                        final_score
                    ),

                "_mask":
                    m
            })


        if not candidate_infos:

            results.append({

                "inventory_id":
                    iid,

                "inventory_name":
                    name,

                "box":
                    box,

                "crop_box":
                    list(
                        crop_box
                    ),

                "decision":
                    "REJECT",

                "reason":
                    "no_usable_mask"
            })

            continue


        candidate_infos.sort(
            key=lambda x:
                x[
                    "final_score"
                ],
            reverse=True
        )


        best = (
            candidate_infos[
                0
            ]
        )


        local_mask = (
            best[
                "_mask"
            ]
        )


        # ====================================================
        # FINAL LOCAL REJECTION
        # ====================================================

        reject_reason = None


        if (
            best[
                "area_ratio"
            ]
            >
            0.86
        ):

            reject_reason = (
                "fills_crop"
            )


        elif (
            best[
                "edge_count"
            ]
            >=
            3

            and

            best[
                "area_ratio"
            ]
            >
            0.45
        ):

            reject_reason = (
                "multi_edge_large_surface"
            )


        elif (
            best[
                "surface_risk"
            ]
            >=
            5.0

            and

            best[
                "final_score"
            ]
            <
            0.0
        ):

            reject_reason = (
                "architectural_surface_risk"
            )


        # ====================================================
        # MAP TO MASTER
        # ====================================================

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


        recovery = (
            global_mask
            &
            (~base)
        )


        recovery_pixels = int(
            recovery.sum()
        )


        if (
            recovery_pixels
            <
            MIN_RECOVERY_PIXELS
        ):

            reject_reason = (
                reject_reason
                or
                "almost_no_new_pixels"
            )


        # ====================================================
        # EXTRA GLOBAL CATASTROPHIC CHECK
        # ====================================================

        recovery_image_ratio = (
            recovery_pixels
            /
            float(
                W * H
            )
        )


        full_mask_ratio = (
            global_mask.sum()
            /
            float(
                W * H
            )
        )


        missing_ratio = (
            recovery_pixels
            /
            max(
                1,
                int(
                    global_mask.sum()
                )
            )
        )


        if (
            recovery_image_ratio
            >
            MAX_GLOBAL_RECOVERY_RATIO
        ):

            reject_reason = (
                reject_reason
                or
                "catastrophic_global_recovery"
            )


        if (
            full_mask_ratio
            >
            MAX_GLOBAL_MASK_RATIO

            and

            missing_ratio
            >
            MIN_GLOBAL_BAD_MISSING_RATIO
        ):

            reject_reason = (
                reject_reason
                or
                "broad_structural_mask"
            )


        # ====================================================
        # SAVE DEBUG
        # ====================================================

        prefix = (
            object_dir
            /
            f"{iid:02d}_{safe_name(name)}"
        )


        cv2.imwrite(
            str(
                prefix
            )
            +
            "_crop.png",
            cv2.cvtColor(
                crop_rgb,
                cv2.COLOR_RGB2BGR
            )
        )


        cv2.imwrite(
            str(
                prefix
            )
            +
            "_local_mask.png",
            local_mask.astype(
                np.uint8
            )
            *
            255
        )


        cv2.imwrite(
            str(
                prefix
            )
            +
            "_recovery.png",
            recovery.astype(
                np.uint8
            )
            *
            255
        )


        if reject_reason is None:

            accepted_union |= (
                recovery
            )

            decision = (
                "ACCEPT"
            )

        else:

            rejected_union |= (
                recovery
            )

            decision = (
                "REJECT"
            )


        print(
            decision,
            "| recovery:",
            recovery_pixels,
            "| area_ratio:",
            round(
                best[
                    "area_ratio"
                ],
                3
            ),
            "| edges:",
            best[
                "edge_count"
            ],
            "| surface:",
            round(
                best[
                    "surface_risk"
                ],
                3
            ),
            "| score:",
            round(
                best[
                    "final_score"
                ],
                3
            ),
            "| reason:",
            reject_reason
        )


        clean_candidates = []

        for candidate in candidate_infos:

            clean_candidates.append({
                k: v
                for k, v
                in candidate.items()
                if k
                !=
                "_mask"
            })


        results.append({

            "inventory_id":
                iid,

            "inventory_name":
                name,

            "box":
                box,

            "crop_box":
                list(
                    crop_box
                ),

            "decision":
                decision,

            "reason":
                reject_reason,

            "recovery_pixels":
                recovery_pixels,

            "global_recovery_ratio":
                float(
                    recovery_image_ratio
                ),

            "global_mask_ratio":
                float(
                    full_mask_ratio
                ),

            "missing_ratio":
                float(
                    missing_ratio
                ),

            "best_mask": {
                k: v
                for k, v
                in best.items()
                if k
                !=
                "_mask"
            },

            "all_masks":
                clean_candidates
        })


    # ========================================================
    # FINAL MASK
    # ========================================================

    final_mask = (
        base
        |
        accepted_union
    )


    # ========================================================
    # REMAINING UNRESOLVED INSTANCE IDS
    # ========================================================

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


    original_unresolved_ids = {
        int(
            row[
                "inventory_id"
            ]
        )
        for row in unresolved
    }


    remaining_unresolved_ids = sorted(
        original_unresolved_ids
        -
        accepted_ids
    )


    # ========================================================
    # SAVE
    # ========================================================

    accepted_path = (
        output_dir
        /
        "00_local_crop_accepted_recovery.png"
    )


    rejected_path = (
        output_dir
        /
        "01_local_crop_rejected_recovery.png"
    )


    final_mask_path = (
        output_dir
        /
        "02_final_props_mask.png"
    )


    cv2.imwrite(
        str(
            accepted_path
        ),
        accepted_union.astype(
            np.uint8
        )
        *
        255
    )


    cv2.imwrite(
        str(
            rejected_path
        ),
        rejected_union.astype(
            np.uint8
        )
        *
        255
    )


    cv2.imwrite(
        str(
            final_mask_path
        ),
        final_mask.astype(
            np.uint8
        )
        *
        255
    )


    # ========================================================
    # RGBA
    # ========================================================

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
    ] = (
        master_rgb
    )


    rgba[
        :,
        :,
        3
    ] = (
        final_mask.astype(
            np.uint8
        )
        *
        255
    )


    rgba_path = (
        output_dir
        /
        "03_all_props_rgba.png"
    )


    Image.fromarray(
        rgba,
        mode="RGBA"
    ).save(
        rgba_path
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
        master_rgb[
            final_mask
        ]
    )


    preview_path = (
        output_dir
        /
        "04_transparency_preview.png"
    )


    Image.fromarray(
        preview
    ).save(
        preview_path
    )


    # ========================================================
    # ACCEPTED OVERLAY
    # ========================================================

    overlay = (
        master_rgb.copy()
    )


    red = np.array(
        [
            255,
            0,
            0
        ],
        dtype=np.uint8
    )


    overlay[
        accepted_union
    ] = (
        overlay[
            accepted_union
        ].astype(
            np.float32
        )
        *
        0.35
        +
        red.astype(
            np.float32
        )
        *
        0.65
    ).astype(
        np.uint8
    )


    overlay_path = (
        output_dir
        /
        "05_accepted_recovery_overlay.png"
    )


    Image.fromarray(
        overlay
    ).save(
        overlay_path
    )


    results_path = (
        output_dir
        /
        "06_object_results.json"
    )


    results_path.write_text(
        json.dumps(
            results,
            indent=2
        )
    )


    # ========================================================
    # REPORT
    # ========================================================

    accepted_objects = [
        r
        for r in results
        if r.get(
            "decision"
        )
        ==
        "ACCEPT"
    ]


    rejected_objects = [
        r
        for r in results
        if r.get(
            "decision"
        )
        ==
        "REJECT"
    ]


    report = {

        "experiment":
            "TEST07_STAGE10",

        "method":
            (
                "local crop SAM2 verification "
                "with architectural surface rejection"
            ),

        "master":
            str(
                master_path
            ),

        "base_mask":
            str(
                base_mask_path
            ),

        "stage09_source":
            str(
                stage09_json
            ),

        "base_pixels":
            int(
                base.sum()
            ),

        "eligible_objects":
            len(
                eligible
            ),

        "excluded_before_sam2":
            excluded,

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

        "accepted_objects":
            len(
                accepted_objects
            ),

        "rejected_objects":
            len(
                rejected_objects
            ),

        "remaining_unresolved_inventory_ids":
            remaining_unresolved_ids,

        "objects":
            results,

        "rules": [
            "Stage08 trusted foreground was not replaced",
            "corrected Stage09 catastrophic gate applied before local SAM2",
            "SAM2 rerun locally per plausible unresolved instance",
            "large crop-filling multi-edge masks rejected",
            "only pixels outside Stage08 foreground added",
            "catastrophic global recovery rejected",
            "mirror giant source proposal was not reused",
            "RGB copied exactly from locked resized master"
        ]
    }


    report_path = (
        output_dir
        /
        "stage10_report.json"
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
        "TEST07 STAGE10 RESULT"
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
        "ELIGIBLE OBJECTS:",
        len(
            eligible
        )
    )


    print(
        "EXCLUDED BEFORE SAM2:",
        len(
            excluded
        )
    )


    print(
        "ACCEPTED:",
        len(
            accepted_objects
        )
    )


    print(
        "REJECTED:",
        len(
            rejected_objects
        )
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
        "REMAINING UNRESOLVED:",
        remaining_unresolved_ids
    )


    print()
    print(
        "IMPORTANT OUTPUT:"
    )

    print(
        preview_path
    )


    print()
    print(
        "ACCEPTED RECOVERY:"
    )

    print(
        accepted_path
    )


    print()
    print(
        "STAGE08 BASE WAS NOT REPLACED."
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
        "--stage09-json",
        required=True
    )


    parser.add_argument(
        "--unresolved-json",
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

        stage09_json=
            args.stage09_json,

        unresolved_json=
            args.unresolved_json,

        output_dir=
            args.output_dir,

        checkpoint_path=
            args.sam2_checkpoint
    )
