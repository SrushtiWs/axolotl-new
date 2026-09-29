
from pathlib import Path
import argparse
import json

from PIL import Image
import cv2
import numpy as np


# ============================================================
# TEST07 — STAGE 05
#
# GENERALIZED CANDIDATE GEOMETRY FILTER
#
# RECONSTRUCTED FROM TEST06 BB4 BEHAVIOR
#
# PURPOSE:
#   Filter Stage04 verified object boxes before tight-box
#   verification / SAM2 segmentation.
#
# INPUT:
#   - resized master image
#   - Stage04 verified candidate JSON
#   - optional AQ foreground/base mask statistics if available
#
# OUTPUT:
#   - 01_filtered_candidates.json
#   - 02_rejected_candidates.json
#   - stage05_report.json
#
# IMPORTANT:
#   - NO TEST06 pixel locks
#   - NO fixed coordinates
#   - NO fixed object names
#   - NO per-room tuning
#   - purely normalized geometry / confidence filtering
# ============================================================


# ============================================================
# GENERALIZED THRESHOLDS
# ============================================================

# Giant detections.
MAX_AREA_RATIO = 0.34

# Almost full image dimension.
MAX_WIDTH_RATIO = 0.93
MAX_HEIGHT_RATIO = 0.93

# Large region with weak foreground support.
LARGE_REGION_RATIO = 0.18
MIN_AQ_COVERAGE_FOR_LARGE_REGION = 0.12

# Moderate region with extremely low foreground support.
MEDIUM_REGION_RATIO = 0.07
MIN_AQ_COVERAGE_FOR_MEDIUM_REGION = 0.015

# Extremely weak detector results.
MIN_DETECTOR_SCORE = 0.16

# Candidate quality.
MIN_QUALITY_SCORE = 0.42

# Duplicate suppression for same semantic inventory concept.
DUPLICATE_IOU_THRESHOLD = 0.72

# Maximum candidates retained per semantic concept.
MAX_PER_CONCEPT = 3


# ============================================================
# HELPERS
# ============================================================

def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def save_json(path, data):

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=2
        )


def box_iou(a, b):

    ax1, ay1, ax2, ay2 = [
        float(v)
        for v in a
    ]

    bx1, by1, bx2, by2 = [
        float(v)
        for v in b
    ]

    ix1 = max(
        ax1,
        bx1
    )

    iy1 = max(
        ay1,
        by1
    )

    ix2 = min(
        ax2,
        bx2
    )

    iy2 = min(
        ay2,
        by2
    )

    iw = max(
        0.0,
        ix2 - ix1
    )

    ih = max(
        0.0,
        iy2 - iy1
    )

    inter = (
        iw * ih
    )

    area_a = max(
        0.0,
        ax2 - ax1
    ) * max(
        0.0,
        ay2 - ay1
    )

    area_b = max(
        0.0,
        bx2 - bx1
    ) * max(
        0.0,
        by2 - by1
    )

    union = (
        area_a
        +
        area_b
        -
        inter
    )

    if union <= 0:
        return 0.0

    return (
        inter
        /
        union
    )


def normalize_box(
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
            float(W),
            x1
        )
    )

    y1 = max(
        0.0,
        min(
            float(H),
            y1
        )
    )

    x2 = max(
        0.0,
        min(
            float(W),
            x2
        )
    )

    y2 = max(
        0.0,
        min(
            float(H),
            y2
        )
    )

    if x2 < x1:
        x1, x2 = x2, x1

    if y2 < y1:
        y1, y2 = y2, y1

    return [
        x1,
        y1,
        x2,
        y2
    ]


def derive_quality(
    detector_score,
    aq_coverage,
    area_ratio,
    width_ratio,
    height_ratio
):

    """
    Generic candidate score.

    High detector confidence and reasonable AQ support help.
    Giant image-spanning boxes are penalized.
    """

    detector_score = max(
        0.0,
        min(
            1.0,
            float(detector_score)
        )
    )

    aq_coverage = max(
        0.0,
        min(
            1.0,
            float(aq_coverage)
        )
    )

    size_penalty = 0.0

    if area_ratio > 0.18:
        size_penalty += (
            area_ratio
            * 0.8
        )

    if width_ratio > 0.75:
        size_penalty += (
            width_ratio
            - 0.75
        )

    if height_ratio > 0.75:
        size_penalty += (
            height_ratio
            - 0.75
        )

    quality = (
        detector_score * 0.62
        +
        aq_coverage * 0.38
        -
        size_penalty
    )

    return float(
        quality
    )


def get_candidate_list(data):

    """
    Accept several possible Stage04 JSON layouts.
    """

    if isinstance(
        data,
        list
    ):

        return data

    if not isinstance(
        data,
        dict
    ):

        raise RuntimeError(
            "Unsupported Stage04 JSON structure."
        )

    keys = [
        "candidates",
        "verified_candidates",
        "objects",
        "results",
        "detections"
    ]

    for key in keys:

        value = data.get(
            key
        )

        if isinstance(
            value,
            list
        ):

            return value

    raise RuntimeError(
        "Could not find candidate list in Stage04 JSON."
    )


# ============================================================
# MAIN FILTER
# ============================================================

def run(
    master_path,
    stage04_json,
    aq_mask_path,
    output_dir
):

    master_path = Path(
        master_path
    )

    stage04_json = Path(
        stage04_json
    )

    aq_mask_path = Path(
        aq_mask_path
    )

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    if not master_path.exists():

        raise FileNotFoundError(
            master_path
        )


    if not stage04_json.exists():

        raise FileNotFoundError(
            stage04_json
        )

    if not aq_mask_path.exists():

        raise FileNotFoundError(
            aq_mask_path
        )


    with Image.open(
        master_path
    ) as img:

        W, H = img.size


    aq_img = cv2.imread(
        str(aq_mask_path),
        cv2.IMREAD_GRAYSCALE
    )

    if aq_img is None:

        raise RuntimeError(
            f"Could not read AQ mask: {aq_mask_path}"
        )

    if aq_img.shape != (H, W):

        raise RuntimeError(
            f"AQ mask shape mismatch: "
            f"{aq_img.shape} != {(H, W)}"
        )

    aq_mask = (
        aq_img > 127
    )


    stage04_data = load_json(
        stage04_json
    )


    candidates = get_candidate_list(
        stage04_data
    )


    print(
        "=" * 90
    )

    print(
        "TEST07 STAGE05"
    )

    print(
        "GENERALIZED CANDIDATE GEOMETRY FILTER"
    )

    print(
        "=" * 90
    )

    print(
        "Resolution:",
        W,
        "x",
        H
    )

    print(
        "Input candidates:",
        len(
            candidates
        )
    )


    provisional_keep = []
    rejected = []


    for index, row in enumerate(
        candidates,
        start=1
    ):

        item = dict(
            row
        )


        box = (
            item.get(
                "bbox"
            )
            or
            item.get(
                "box"
            )
            or
            item.get(
                "raw_box"
            )
        )


        if (
            not isinstance(
                box,
                (list, tuple)
            )
            or
            len(box) != 4
        ):

            item[
                "stage05_decision"
            ] = "REJECT"

            item[
                "stage05_reject_reasons"
            ] = [
                "missing_valid_bbox"
            ]

            rejected.append(
                item
            )

            continue


        box = normalize_box(
            box,
            W,
            H
        )


        x1, y1, x2, y2 = (
            box
        )


        bw = max(
            0.0,
            x2 - x1
        )

        bh = max(
            0.0,
            y2 - y1
        )


        area = (
            bw * bh
        )


        image_area = max(
            1.0,
            float(
                W * H
            )
        )


        area_ratio = (
            area
            /
            image_area
        )


        width_ratio = (
            bw
            /
            max(
                1.0,
                float(W)
            )
        )


        height_ratio = (
            bh
            /
            max(
                1.0,
                float(H)
            )
        )


        detector_score = float(
            item.get(
                "score",
                item.get(
                    "detector_score",
                    item.get(
                        "confidence",
                        0.0
                    )
                )
            )
            or 0.0
        )


        # ====================================================
        # DYNAMIC AQ COVERAGE
        #
        # Fraction of this DINO bbox already supported by the
        # Stage01 trusted foreground candidate.
        # ====================================================

        ix1 = max(
            0,
            int(np.floor(x1))
        )

        iy1 = max(
            0,
            int(np.floor(y1))
        )

        ix2 = min(
            W,
            int(np.ceil(x2))
        )

        iy2 = min(
            H,
            int(np.ceil(y2))
        )

        if (
            ix2 <= ix1
            or iy2 <= iy1
        ):

            aq_coverage = 0.0

        else:

            aq_crop = aq_mask[
                iy1:iy2,
                ix1:ix2
            ]

            aq_coverage = float(
                aq_crop.mean()
            )


        quality = derive_quality(
            detector_score,
            aq_coverage,
            area_ratio,
            width_ratio,
            height_ratio
        )


        reasons = []


        if (
            area <= 1.0
        ):

            reasons.append(
                "degenerate_bbox"
            )


        if (
            area_ratio
            > MAX_AREA_RATIO
        ):

            reasons.append(
                "giant_area"
            )


        if (
            width_ratio
            > MAX_WIDTH_RATIO
        ):

            reasons.append(
                "near_full_width"
            )


        if (
            height_ratio
            > MAX_HEIGHT_RATIO
        ):

            # Tall narrow objects can legitimately span almost
            # the full image height. Only reject when the bbox
            # is also broad OR has weak Stage01 foreground
            # support.
            tall_object_supported = (
                width_ratio < 0.30
                and
                aq_coverage >= 0.55
            )

            if not tall_object_supported:

                reasons.append(
                    "near_full_height"
                )


        if (
            area_ratio
            > LARGE_REGION_RATIO
            and
            aq_coverage
            < MIN_AQ_COVERAGE_FOR_LARGE_REGION
        ):

            reasons.append(
                "large_low_aq"
            )


        if (
            area_ratio
            > MEDIUM_REGION_RATIO
            and
            aq_coverage
            < MIN_AQ_COVERAGE_FOR_MEDIUM_REGION
        ):

            reasons.append(
                "medium_low_aq"
            )


        if (
            detector_score
            < MIN_DETECTOR_SCORE
        ):

            reasons.append(
                "low_detector_score"
            )


        if (
            quality
            < MIN_QUALITY_SCORE
        ):

            strong_foreground_support = (
                aq_coverage >= 0.75
                and
                detector_score >= 0.28
                and
                area_ratio < 0.25
            )

            if not strong_foreground_support:

                reasons.append(
                    "low_quality_score"
                )


        item[
            "bbox"
        ] = [
            float(v)
            for v in box
        ]


        item[
            "area_ratio_stage05"
        ] = float(
            area_ratio
        )


        item[
            "width_ratio_stage05"
        ] = float(
            width_ratio
        )


        item[
            "height_ratio_stage05"
        ] = float(
            height_ratio
        )


        item[
            "aq_coverage_stage05"
        ] = float(
            aq_coverage
        )


        item[
            "stage05_quality_score"
        ] = float(
            quality
        )


        item[
            "stage05_reject_reasons"
        ] = list(
            reasons
        )


        if reasons:

            item[
                "stage05_decision"
            ] = "REJECT"

            rejected.append(
                item
            )

        else:

            item[
                "stage05_decision"
            ] = "KEEP"

            provisional_keep.append(
                item
            )


    # ========================================================
    # SAME-CONCEPT DUPLICATE CONTROL
    # ========================================================

    grouped = {}


    for row in provisional_keep:

        # ====================================================
        # INSTANCE-LEVEL GROUPING
        #
        # Stage02 may contain several physically separate
        # objects with the same generic name:
        #
        # framed artwork x6
        # lamp x2
        # bedside table x2
        #
        # Therefore inventory_name must NOT define identity.
        # inventory_id represents one physical instance.
        # ====================================================

        inventory_id = str(
            row.get(
                "inventory_id",
                "unknown"
            )
        )


        grouped.setdefault(
            inventory_id,
            []
        ).append(
            row
        )


    final_candidates = []


    for inventory_id, rows in grouped.items():

        rows.sort(
            key=lambda r:
                float(
                    r.get(
                        "stage05_quality_score",
                        0.0
                    )
                ),
            reverse=True
        )


        accepted_for_concept = []


        for row in rows:

            duplicate = False


            for existing in accepted_for_concept:

                iou = box_iou(
                    row[
                        "bbox"
                    ],
                    existing[
                        "bbox"
                    ]
                )


                if (
                    iou
                    >= DUPLICATE_IOU_THRESHOLD
                ):

                    duplicate = True
                    break


            if duplicate:

                row[
                    "stage05_decision"
                ] = "REJECT"

                row[
                    "stage05_reject_reasons"
                ] = [
                    "duplicate_same_concept"
                ]

                rejected.append(
                    row
                )

                continue


            if (
                len(
                    accepted_for_concept
                )
                >= MAX_PER_CONCEPT
            ):

                row[
                    "stage05_decision"
                ] = "REJECT"

                row[
                    "stage05_reject_reasons"
                ] = [
                    "too_many_candidates_for_instance"
                ]

                rejected.append(
                    row
                )

                continue


            accepted_for_concept.append(
                row
            )


        final_candidates.extend(
            accepted_for_concept
        )


    final_candidates.sort(
        key=lambda r:
            float(
                r.get(
                    "stage05_quality_score",
                    0.0
                )
            ),
        reverse=True
    )


    # ========================================================
    # INSTANCE-LEVEL RESCUE
    #
    # Stage04 has already semantically verified these boxes.
    # Stage05 geometry filtering must not silently erase an
    # entire physical inventory instance.
    #
    # If all proposals for one instance were rejected, rescue
    # its best verified candidate as LOW TRUST so Stage06 can
    # tighten or recover it.
    #
    # Truly catastrophic near-full-image proposals remain
    # rejected.
    # ========================================================

    input_by_instance = {}

    for row in candidates:

        iid = str(
            row.get(
                "inventory_id",
                "unknown"
            )
        )

        input_by_instance.setdefault(
            iid,
            []
        ).append(
            row
        )


    surviving_ids = {
        str(
            row.get(
                "inventory_id",
                "unknown"
            )
        )
        for row in final_candidates
    }


    rescue_candidates = []


    for iid, rows in input_by_instance.items():

        if iid in surviving_ids:
            continue


        # Search enriched rejected rows first because they
        # already contain Stage05 measurements.
        rejected_for_instance = [
            row
            for row in rejected
            if str(
                row.get(
                    "inventory_id",
                    "unknown"
                )
            ) == iid
        ]


        pool = (
            rejected_for_instance
            if rejected_for_instance
            else rows
        )


        # Best semantic/DINO proposal first.
        pool = sorted(
            pool,
            key=lambda r:
                (
                    float(
                        r.get(
                            "score",
                            0.0
                        )
                    ),
                    float(
                        r.get(
                            "stage05_quality_score",
                            -999.0
                        )
                    )
                ),
            reverse=True
        )


        rescued = None


        for candidate in pool:

            area_r = float(
                candidate.get(
                    "area_ratio_stage05",
                    candidate.get(
                        "area_ratio",
                        0.0
                    )
                )
            )

            width_r = float(
                candidate.get(
                    "width_ratio_stage05",
                    0.0
                )
            )

            height_r = float(
                candidate.get(
                    "height_ratio_stage05",
                    0.0
                )
            )


            # Catastrophic proposals are too unsafe even for
            # a rescue handoff.
            catastrophic = (
                area_r > 0.72
                or
                width_r > 0.985
                or
                height_r > 0.985
            )

            if catastrophic:
                continue


            rescued = dict(
                candidate
            )

            break


        if rescued is None:
            continue


        rescued[
            "stage05_decision"
        ] = "RESCUE_LOW_TRUST"

        rescued[
            "stage05_rescue"
        ] = True

        rescued[
            "stage05_rescue_reason"
        ] = (
            "no_strict_candidate_survived_for_inventory_instance"
        )

        rescued[
            "stage05_original_reject_reasons"
        ] = list(
            rescued.get(
                "stage05_reject_reasons",
                []
            )
        )


        final_candidates.append(
            rescued
        )

        rescue_candidates.append(
            rescued
        )


    final_candidates.sort(
        key=lambda r:
            (
                0
                if r.get(
                    "stage05_rescue",
                    False
                )
                else 1,

                float(
                    r.get(
                        "stage05_quality_score",
                        0.0
                    )
                )
            ),
        reverse=True
    )


    # ========================================================
    # OUTPUT
    # ========================================================

    final_path = (
        output_dir
        / "01_filtered_candidates.json"
    )

    rejected_path = (
        output_dir
        / "02_rejected_candidates.json"
    )

    report_path = (
        output_dir
        / "stage05_report.json"
    )


    save_json(
        final_path,
        final_candidates
    )

    save_json(
        rejected_path,
        rejected
    )


    concept_counts = {}


    for row in final_candidates:

        concept = str(
            row.get(
                "inventory_name",
                row.get(
                    "name",
                    row.get(
                        "label",
                        "unknown"
                    )
                )
            )
        )


        concept_counts[
            concept
        ] = (
            concept_counts.get(
                concept,
                0
            )
            + 1
        )


    report = {
        "experiment":
            "TEST07_STAGE05",

        "method":
            (
                "generalized normalized candidate geometry "
                "and quality filtering reconstructed from "
                "TEST06 BB4"
            ),

        "master":
            str(
                master_path
            ),

        "stage04_input":
            str(
                stage04_json
            ),

        "resolution": {
            "width":
                int(W),

            "height":
                int(H)
        },

        "input_candidates":
            len(
                candidates
            ),

        "final_candidates":
            len(
                final_candidates
            ),

        "rejected_candidates":
            len(
                rejected
            ),

        "per_concept_final":
            concept_counts,

        "thresholds": {
            "max_area_ratio":
                MAX_AREA_RATIO,

            "max_width_ratio":
                MAX_WIDTH_RATIO,

            "max_height_ratio":
                MAX_HEIGHT_RATIO,

            "large_region_ratio":
                LARGE_REGION_RATIO,

            "min_aq_coverage_large":
                MIN_AQ_COVERAGE_FOR_LARGE_REGION,

            "min_quality_score":
                MIN_QUALITY_SCORE,

            "duplicate_iou":
                DUPLICATE_IOU_THRESHOLD,

            "max_per_concept":
                MAX_PER_CONCEPT
        },

        "guarantees": [
            "no fixed image coordinates",
            "no fixed object names",
            "no TEST06 pixel counts",
            "normalized geometry only",
            "same thresholds intended for all TEST07 rooms"
        ],

        "outputs": {
            "filtered_candidates":
                str(
                    final_path
                ),

            "rejected_candidates":
                str(
                    rejected_path
                )
        }
    }


    save_json(
        report_path,
        report
    )


    print()
    print(
        "=" * 90
    )

    print(
        "TEST07 STAGE05 RESULT"
    )

    print(
        "=" * 90
    )

    print(
        "INPUT CANDIDATES:",
        len(
            candidates
        )
    )

    print(
        "FINAL CANDIDATES:",
        len(
            final_candidates
        )
    )

    print(
        "REJECTED:",
        len(
            rejected
        )
    )

    print()
    print(
        "FILTERED JSON:"
    )

    print(
        final_path
    )

    print()
    print(
        "REPORT:"
    )

    print(
        report_path
    )


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
        "--stage04-json",
        required=True
    )

    parser.add_argument(
        "--aq-mask",
        required=True
    )


    parser.add_argument(
        "--output-dir",
        required=True
    )

    args = parser.parse_args()


    run(
        master_path=
            args.master,

        stage04_json=
            args.stage04_json,

        aq_mask_path=
            args.aq_mask,

        output_dir=
            args.output_dir
    )
