
from pathlib import Path
import argparse
import json

import cv2
import numpy as np
from PIL import Image


# ============================================================
# TEST07 — STAGE 10
#
# GENERALIZED COMPONENT STRUCTURE FILTER
#
# RECONSTRUCTED FROM TEST06 BB11_FIX
#
# PURPOSE:
#   Take:
#       AQ trusted foreground base
#       +
#       BB10 recovered props mask
#
#   Then filter ONLY the additional BB10 recovery components.
#
# IMPORTANT:
#   AQ IS THE TRUSTED BASE AND CAN NEVER LOSE PIXELS.
#
# METHOD:
#
#   recovery = BB10 & ~AQ
#       ↓
#   connected components
#       ↓
#   normalized geometry / attachment / shape analysis
#       ↓
#   reject structural contamination
#       ↓
#   final = AQ | accepted_recovery
#
# OUTPUT:
#   00_accepted_recovery_mask.png
#   01_rejected_recovery_mask.png
#   02_final_props_mask.png
#   03_all_props_rgba.png
#   04_transparency_preview.png
#   05_recovery_decision_overlay.png
#   06_component_report.json
#   stage10_report.json
#
# GUARANTEES:
#   - AQ pixels never removed
#   - only BB10 recovery filtered
#   - no fixed image coordinates
#   - no fixed pixel counts
#   - original RGB copied exactly
# ============================================================


# ============================================================
# GENERALIZED THRESHOLDS
# ============================================================

# --------------------------------
# Noise
# --------------------------------

MIN_COMPONENT_PIXELS = 3

# --------------------------------
# Giant / structural components
# --------------------------------

MAX_COMPONENT_AREA_RATIO = 0.040

MAX_BBOX_AREA_RATIO = 0.085

MAX_WIDTH_RATIO = 0.82

MAX_HEIGHT_RATIO = 0.75

# --------------------------------
# Broad surface-like regions
# --------------------------------

BROAD_COMPONENT_AREA_RATIO = 0.015

BROAD_BBOX_AREA_RATIO = 0.040

MIN_FILL_FOR_BROAD = 0.32

# --------------------------------
# Thin-object handling
# --------------------------------

THIN_ASPECT_RATIO = 4.0

VERY_THIN_ASPECT_RATIO = 9.0

# Long thin objects are allowed if sufficiently connected
# to the trusted AQ foreground.
MIN_AQ_TOUCH_FOR_VERY_THIN = 0.015

# --------------------------------
# AQ connection
# --------------------------------

AQ_TOUCH_DILATION = 2

# Recovery disconnected from AQ is not automatically rejected,
# because entirely missed props can exist. But large disconnected
# components are suspicious.
MAX_DISCONNECTED_AREA_RATIO = 0.006

# --------------------------------
# Structural straight-line test
# --------------------------------

MAX_GENERIC_LINE_SCORE = 0.90

LINE_TEST_MIN_PIXELS = 18

# --------------------------------
# Border risk
# --------------------------------

IMAGE_BORDER_MARGIN_RATIO = 0.008

MAX_BORDER_FRACTION = 0.40

# --------------------------------
# Flat structural surface test
# --------------------------------

MIN_COMPONENT_STD = 5.0

# --------------------------------
# Safety
# --------------------------------

MAX_TOTAL_ACCEPTED_RECOVERY_RATIO = 0.18


# ============================================================
# HELPERS
# ============================================================

def load_mask(
    path,
    H,
    W
):

    path = Path(path)

    img = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE
    )

    if img is None:

        raise RuntimeError(
            f"Could not read mask: {path}"
        )

    if img.shape != (
        H,
        W
    ):

        raise RuntimeError(
            f"Mask shape mismatch: {path}: "
            f"{img.shape} != {(H, W)}"
        )

    return (
        img > 127
    )


def save_mask(
    path,
    mask
):

    Image.fromarray(
        mask.astype(
            np.uint8
        )
        * 255
    ).save(
        path
    )


def line_score(
    mask
):

    ys, xs = np.where(
        mask
    )

    if len(xs) < LINE_TEST_MIN_PIXELS:

        return 0.0


    coords = np.column_stack(
        [
            xs,
            ys
        ]
    ).astype(
        np.float32
    )


    coords -= coords.mean(
        axis=0,
        keepdims=True
    )


    cov = np.cov(
        coords,
        rowvar=False
    )


    try:

        eigvals = (
            np.linalg.eigvalsh(
                cov
            )
        )

    except Exception:

        return 0.0


    eigvals = np.maximum(
        eigvals,
        0
    )


    total = float(
        eigvals.sum()
    )


    if total <= 1e-8:

        return 0.0


    return float(
        eigvals[-1]
        /
        total
    )


def component_border_fraction(
    component,
    H,
    W
):

    margin = max(
        2,
        int(
            round(
                min(
                    H,
                    W
                )
                *
                IMAGE_BORDER_MARGIN_RATIO
            )
        )
    )


    border = np.zeros(
        (
            H,
            W
        ),
        dtype=bool
    )


    border[
        :margin,
        :
    ] = True

    border[
        H - margin:,
        :
    ] = True

    border[
        :,
        :margin
    ] = True

    border[
        :,
        W - margin:
    ] = True


    area = int(
        component.sum()
    )


    if area == 0:

        return 1.0


    touched = int(
        (
            component
            &
            border
        ).sum()
    )


    return (
        touched
        /
        max(
            1,
            area
        )
    )


# ============================================================
# MAIN
# ============================================================

def run(
    master_path,
    aq_mask_path,
    bb10_mask_path,
    output_dir
):

    master_path = Path(
        master_path
    )

    aq_mask_path = Path(
        aq_mask_path
    )

    bb10_mask_path = Path(
        bb10_mask_path
    )

    output_dir = Path(
        output_dir
    )


    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    # ========================================================
    # LOAD MASTER
    # ========================================================

    if not master_path.exists():

        raise FileNotFoundError(
            master_path
        )


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


    W, H = (
        master_pil.size
    )


    # ========================================================
    # LOAD MASKS
    # ========================================================

    aq = load_mask(
        aq_mask_path,
        H,
        W
    )


    bb10 = load_mask(
        bb10_mask_path,
        H,
        W
    )


    aq_pixels = int(
        aq.sum()
    )


    bb10_pixels = int(
        bb10.sum()
    )


    # ========================================================
    # IMPORTANT DYNAMIC SAFETY CHECK
    # ========================================================

    # BB10 must contain all AQ pixels.
    lost_aq_before_filter = (
        aq
        &
        (~bb10)
    )


    if lost_aq_before_filter.any():

        raise RuntimeError(
            "STOP: BB10 does not preserve all AQ pixels."
        )


    recovery = (
        bb10
        &
        (~aq)
    )


    recovery_pixels = int(
        recovery.sum()
    )


    print(
        "=" * 90
    )

    print(
        "TEST07 STAGE10"
    )

    print(
        "GENERALIZED COMPONENT STRUCTURE FILTER"
    )

    print(
        "=" * 90
    )


    print()
    print(
        "Resolution:",
        W,
        "x",
        H
    )

    print(
        "AQ pixels:",
        aq_pixels
    )

    print(
        "BB10 pixels:",
        bb10_pixels
    )

    print(
        "Recovery pixels to filter:",
        recovery_pixels
    )


    # ========================================================
    # AQ TOUCH REGION
    # ========================================================

    aq_touch_region = (
        cv2.dilate(
            aq.astype(
                np.uint8
            ),
            np.ones(
                (
                    AQ_TOUCH_DILATION * 2 + 1,
                    AQ_TOUCH_DILATION * 2 + 1
                ),
                np.uint8
            ),
            iterations=1
        )
        > 0
    )


    # ========================================================
    # CONNECTED COMPONENTS
    # ========================================================

    n_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            recovery.astype(
                np.uint8
            ),
            connectivity=8
        )
    )


    accepted_recovery = np.zeros(
        (
            H,
            W
        ),
        dtype=bool
    )


    rejected_recovery = np.zeros(
        (
            H,
            W
        ),
        dtype=bool
    )


    component_report = []


    image_area = float(
        H * W
    )


    # ========================================================
    # COMPONENT LOOP
    # ========================================================

    for label_id in range(
        1,
        n_labels
    ):

        component = (
            labels
            ==
            label_id
        )


        area = int(
            stats[
                label_id,
                cv2.CC_STAT_AREA
            ]
        )


        x = int(
            stats[
                label_id,
                cv2.CC_STAT_LEFT
            ]
        )


        y = int(
            stats[
                label_id,
                cv2.CC_STAT_TOP
            ]
        )


        width = int(
            stats[
                label_id,
                cv2.CC_STAT_WIDTH
            ]
        )


        height = int(
            stats[
                label_id,
                cv2.CC_STAT_HEIGHT
            ]
        )


        x2 = (
            x + width
        )


        y2 = (
            y + height
        )


        bbox_area = max(
            1,
            width * height
        )


        area_ratio = (
            area
            /
            image_area
        )


        bbox_ratio = (
            bbox_area
            /
            image_area
        )


        fill_ratio = (
            area
            /
            max(
                1,
                bbox_area
            )
        )


        aspect_ratio = (
            max(
                width,
                height
            )
            /
            max(
                1,
                min(
                    width,
                    height
                )
            )
        )


        width_ratio = (
            width
            /
            max(
                1,
                W
            )
        )


        height_ratio = (
            height
            /
            max(
                1,
                H
            )
        )


        # ====================================================
        # AQ TOUCH
        # ====================================================

        aq_touch_pixels = int(
            (
                component
                &
                aq_touch_region
            ).sum()
        )


        aq_touch_ratio = (
            aq_touch_pixels
            /
            max(
                1,
                area
            )
        )


        touches_aq = (
            aq_touch_pixels
            > 0
        )


        # ====================================================
        # SHAPE
        # ====================================================

        thin_component = (
            aspect_ratio
            >= THIN_ASPECT_RATIO
        )


        very_thin_component = (
            aspect_ratio
            >= VERY_THIN_ASPECT_RATIO
        )


        component_line_score = (
            line_score(
                component
            )
        )


        border_fraction = (
            component_border_fraction(
                component,
                H,
                W
            )
        )


        # ====================================================
        # RGB VARIATION
        # ====================================================

        values = cv2.cvtColor(
            master,
            cv2.COLOR_RGB2GRAY
        )[
            component
        ]


        component_std = (
            float(
                values.std()
            )
            if len(values)
            > 0
            else 0.0
        )


        # ====================================================
        # DECISION
        # ====================================================

        reasons = []


        # --------------------------------
        # Tiny noise
        # --------------------------------

        if (
            area
            <
            MIN_COMPONENT_PIXELS
        ):

            reasons.append(
                "tiny_noise"
            )


        # --------------------------------
        # Huge area
        # --------------------------------

        if (
            area_ratio
            >
            MAX_COMPONENT_AREA_RATIO
        ):

            reasons.append(
                "component_too_large"
            )


        # --------------------------------
        # Huge bbox
        # --------------------------------

        if (
            bbox_ratio
            >
            MAX_BBOX_AREA_RATIO
        ):

            reasons.append(
                "bbox_too_large"
            )


        # --------------------------------
        # Image spanning
        # --------------------------------

        if (
            width_ratio
            >
            MAX_WIDTH_RATIO
        ):

            reasons.append(
                "near_full_width"
            )


        if (
            height_ratio
            >
            MAX_HEIGHT_RATIO
        ):

            reasons.append(
                "near_full_height"
            )


        # --------------------------------
        # Broad structure
        # --------------------------------

        broad_region = (
            (
                area_ratio
                >
                BROAD_COMPONENT_AREA_RATIO
            )
            and
            (
                bbox_ratio
                >
                BROAD_BBOX_AREA_RATIO
            )
            and
            (
                fill_ratio
                >
                MIN_FILL_FOR_BROAD
            )
        )


        if broad_region:

            reasons.append(
                "broad_structural_region"
            )


        # --------------------------------
        # Large disconnected recovery
        # --------------------------------

        if (
            not touches_aq
            and
            area_ratio
            >
            MAX_DISCONNECTED_AREA_RATIO
        ):

            reasons.append(
                "large_disconnected_component"
            )


        # --------------------------------
        # Very thin + poor attachment
        # --------------------------------

        if (
            very_thin_component
            and
            aq_touch_ratio
            <
            MIN_AQ_TOUCH_FOR_VERY_THIN
            and
            area
            >
            20
        ):

            reasons.append(
                "unsupported_very_thin_structure"
            )


        # --------------------------------
        # Strong straight line
        # --------------------------------

        if (
            component_line_score
            >
            MAX_GENERIC_LINE_SCORE
            and
            aspect_ratio
            >
            4.0
            and
            area
            >
            LINE_TEST_MIN_PIXELS
            and
            aq_touch_ratio
            <
            0.08
        ):

            reasons.append(
                "line_like_structural_component"
            )


        # --------------------------------
        # Border dominated
        # --------------------------------

        if (
            border_fraction
            >
            MAX_BORDER_FRACTION
            and
            area
            >
            20
            and
            aq_touch_ratio
            <
            0.05
        ):

            reasons.append(
                "image_border_structure"
            )


        # --------------------------------
        # Large flat patch
        # --------------------------------

        if (
            component_std
            <
            MIN_COMPONENT_STD
            and
            area
            >
            40
            and
            fill_ratio
            >
            0.35
        ):

            reasons.append(
                "flat_surface_component"
            )


        # ====================================================
        # FINAL COMPONENT DECISION
        # ====================================================

        if reasons:

            decision = (
                "REJECT"
            )

            rejected_recovery |= (
                component
            )

        else:

            decision = (
                "KEEP"
            )

            accepted_recovery |= (
                component
            )


        component_report.append({
            "id":
                int(
                    label_id
                ),

            "bbox": [
                int(x),
                int(y),
                int(x2),
                int(y2)
            ],

            "width":
                int(
                    width
                ),

            "height":
                int(
                    height
                ),

            "area":
                int(
                    area
                ),

            "area_ratio":
                float(
                    area_ratio
                ),

            "bbox_ratio":
                float(
                    bbox_ratio
                ),

            "fill_ratio":
                float(
                    fill_ratio
                ),

            "aspect_ratio":
                float(
                    aspect_ratio
                ),

            "width_ratio":
                float(
                    width_ratio
                ),

            "height_ratio":
                float(
                    height_ratio
                ),

            "touches_aq":
                bool(
                    touches_aq
                ),

            "aq_touch_pixels":
                int(
                    aq_touch_pixels
                ),

            "aq_touch_ratio":
                float(
                    aq_touch_ratio
                ),

            "thin_component":
                bool(
                    thin_component
                ),

            "very_thin_component":
                bool(
                    very_thin_component
                ),

            "broad_region":
                bool(
                    broad_region
                ),

            "line_score":
                float(
                    component_line_score
                ),

            "border_fraction":
                float(
                    border_fraction
                ),

            "component_std":
                float(
                    component_std
                ),

            "decision":
                decision,

            "reasons":
                reasons
        })


    # ========================================================
    # GLOBAL SAFETY CAP
    # ========================================================

    accepted_pixels = int(
        accepted_recovery.sum()
    )


    accepted_ratio = (
        accepted_pixels
        /
        max(
            1,
            int(
                aq.sum()
            )
        )
    )


    if (
        accepted_ratio
        >
        MAX_TOTAL_ACCEPTED_RECOVERY_RATIO
    ):

        raise RuntimeError(
            "STOP: accepted recovery exceeds generalized "
            "global safety ratio. "
            f"Accepted/AQ={accepted_ratio:.4f}"
        )


    # ========================================================
    # FINAL MASK
    # ========================================================

    final_mask = (
        aq
        |
        accepted_recovery
    )


    # AQ can NEVER lose pixels.
    lost_aq = (
        aq
        &
        (~final_mask)
    )


    if lost_aq.any():

        raise RuntimeError(
            "SAFETY FAILURE: AQ pixels were removed."
        )


    final_pixels = int(
        final_mask.sum()
    )


    rejected_pixels = int(
        rejected_recovery.sum()
    )


    # ========================================================
    # SAVE MASKS
    # ========================================================

    accepted_path = (
        output_dir
        / "00_accepted_recovery_mask.png"
    )


    rejected_path = (
        output_dir
        / "01_rejected_recovery_mask.png"
    )


    final_path = (
        output_dir
        / "02_final_props_mask.png"
    )


    save_mask(
        accepted_path,
        accepted_recovery
    )


    save_mask(
        rejected_path,
        rejected_recovery
    )


    save_mask(
        final_path,
        final_mask
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
        master
    )


    rgba[
        :,
        :,
        3
    ] = (
        final_mask.astype(
            np.uint8
        )
        * 255
    )


    rgba_path = (
        output_dir
        / "03_all_props_rgba.png"
    )


    Image.fromarray(
        rgba
    ).save(
        rgba_path
    )


    # ========================================================
    # TRANSPARENCY PREVIEW
    # ========================================================

    tile = 32


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
            % 2
        )
        * 55
        + 200
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
        / "04_transparency_preview.png"
    )


    Image.fromarray(
        preview
    ).save(
        preview_path
    )


    # ========================================================
    # DECISION OVERLAY
    # ========================================================

    overlay = (
        master.copy()
    )


    if (
        accepted_recovery.any()
    ):

        overlay[
            accepted_recovery
        ] = (
            0.35
            *
            overlay[
                accepted_recovery
            ].astype(
                np.float32
            )

            +

            0.65
            *
            np.array(
                [
                    0,
                    255,
                    0
                ],
                dtype=np.float32
            )
        ).astype(
            np.uint8
        )


    rejected_only = (
        rejected_recovery
        &
        (~accepted_recovery)
    )


    if (
        rejected_only.any()
    ):

        overlay[
            rejected_only
        ] = (
            0.35
            *
            overlay[
                rejected_only
            ].astype(
                np.float32
            )

            +

            0.65
            *
            np.array(
                [
                    255,
                    0,
                    0
                ],
                dtype=np.float32
            )
        ).astype(
            np.uint8
        )


    overlay_path = (
        output_dir
        / "05_recovery_decision_overlay.png"
    )


    Image.fromarray(
        overlay
    ).save(
        overlay_path
    )


    # ========================================================
    # COMPONENT REPORT
    # ========================================================

    component_report_path = (
        output_dir
        / "06_component_report.json"
    )


    with open(
        component_report_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            component_report,
            f,
            indent=2
        )


    # ========================================================
    # MAIN REPORT
    # ========================================================

    kept_components = sum(
        1
        for row
        in component_report
        if row[
            "decision"
        ]
        == "KEEP"
    )


    rejected_components = sum(
        1
        for row
        in component_report
        if row[
            "decision"
        ]
        == "REJECT"
    )


    report = {
        "experiment":
            "TEST07_STAGE10",

        "method":
            (
                "generalized connected-component filtering "
                "of BB10 recovery while preserving AQ base"
            ),

        "master":
            str(
                master_path
            ),

        "aq_mask":
            str(
                aq_mask_path
            ),

        "bb10_mask":
            str(
                bb10_mask_path
            ),

        "aq_pixels":
            int(
                aq_pixels
            ),

        "bb10_pixels":
            int(
                bb10_pixels
            ),

        "raw_recovery_pixels":
            int(
                recovery_pixels
            ),

        "accepted_recovery_pixels":
            int(
                accepted_pixels
            ),

        "rejected_recovery_pixels":
            int(
                rejected_pixels
            ),

        "final_pixels":
            int(
                final_pixels
            ),

        "components_total":
            len(
                component_report
            ),

        "components_accepted":
            int(
                kept_components
            ),

        "components_rejected":
            int(
                rejected_components
            ),

        "dynamic_integrity": {
            "aq_preserved":
                True,

            "bb10_contains_aq":
                True,

            "accepted_recovery_to_aq_ratio":
                float(
                    accepted_ratio
                )
        },

        "guarantees": [
            "AQ pixels never removed",
            "only BB10 recovery filtered",
            "no fixed TEST06 counts",
            "no fixed coordinates",
            "normalized geometry used",
            "RGB copied exactly from resized master"
        ],

        "outputs": {
            "accepted_recovery":
                str(
                    accepted_path
                ),

            "rejected_recovery":
                str(
                    rejected_path
                ),

            "final_mask":
                str(
                    final_path
                ),

            "rgba":
                str(
                    rgba_path
                ),

            "preview":
                str(
                    preview_path
                ),

            "decision_overlay":
                str(
                    overlay_path
                ),

            "component_report":
                str(
                    component_report_path
                )
        }
    }


    report_path = (
        output_dir
        / "stage10_report.json"
    )


    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=2
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
        "AQ BASE:",
        aq_pixels
    )


    print(
        "BB10 INPUT:",
        bb10_pixels
    )


    print(
        "RAW RECOVERY:",
        recovery_pixels
    )


    print(
        "ACCEPTED RECOVERY:",
        accepted_pixels
    )


    print(
        "REJECTED RECOVERY:",
        rejected_pixels
    )


    print(
        "FINAL PIXELS:",
        final_pixels
    )


    print()
    print(
        "COMPONENTS"
    )


    print(
        "TOTAL:",
        len(
            component_report
        )
    )


    print(
        "ACCEPTED:",
        kept_components
    )


    print(
        "REJECTED:",
        rejected_components
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
        "REPORT:"
    )


    print(
        report_path
    )


    print()
    print(
        "AQ WAS NOT MODIFIED."
    )


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    parser = (
        argparse.ArgumentParser()
    )


    parser.add_argument(
        "--master",
        required=True
    )


    parser.add_argument(
        "--aq-mask",
        required=True
    )


    parser.add_argument(
        "--bb10-mask",
        required=True
    )


    parser.add_argument(
        "--output-dir",
        required=True
    )


    args = (
        parser.parse_args()
    )


    run(
        master_path=
            args.master,

        aq_mask_path=
            args.aq_mask,

        bb10_mask_path=
            args.bb10_mask,

        output_dir=
            args.output_dir
    )
