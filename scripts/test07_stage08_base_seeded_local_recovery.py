
from pathlib import Path
import argparse
import json

import cv2
import numpy as np
from PIL import Image


# ============================================================
# TEST07 — STAGE 08
#
# STAGE01-SEEDED LOCAL RECOVERY AUDIT
#
# INPUT:
#   locked resized master
#   Stage01 clean foreground mask
#   Stage07 missing recovery source
#   Stage06 unresolved candidate JSON
#
# PURPOSE:
#   Re-audit SAM2 recovery pixels before allowing them into the
#   foreground.
#
# IMPORTANT:
#   - Stage01 is NEVER replaced
#   - Stage07 recovery is only a candidate recovery source
#   - only recovery spatially supported by Stage01 is accepted
#   - broad/giant recovery is rejected
#   - unresolved Stage06 instances remain unresolved
#   - no Qwen coordinates
#   - no SAM2 execution in this stage
# ============================================================


VERY_CLOSE = 4.0
CLOSE = 10.0
MEDIUM = 20.0
MAX_DISTANCE = 32.0


def save_mask(
    path,
    mask
):
    Image.fromarray(
        mask.astype(np.uint8)
        * 255
    ).save(
        path
    )


def run(
    master_path,
    base_mask_path,
    recovery_path,
    unresolved_json,
    output_dir
):

    master_path = Path(
        master_path
    )

    base_mask_path = Path(
        base_mask_path
    )

    recovery_path = Path(
        recovery_path
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


    for required in [
        master_path,
        base_mask_path,
        recovery_path,
        unresolved_json
    ]:

        if not required.exists():

            raise FileNotFoundError(
                required
            )


    # ========================================================
    # LOAD
    # ========================================================

    master = np.asarray(
        Image.open(
            master_path
        )
        .convert(
            "RGB"
        )
    )


    H, W = (
        master.shape[:2]
    )


    base_u8 = cv2.imread(
        str(
            base_mask_path
        ),
        cv2.IMREAD_GRAYSCALE
    )


    recovery_u8 = cv2.imread(
        str(
            recovery_path
        ),
        cv2.IMREAD_GRAYSCALE
    )


    if base_u8 is None:

        raise RuntimeError(
            f"Could not read base mask: {base_mask_path}"
        )


    if recovery_u8 is None:

        raise RuntimeError(
            f"Could not read recovery mask: {recovery_path}"
        )


    if base_u8.shape != (
        H,
        W
    ):

        raise RuntimeError(
            f"Base mask shape mismatch: "
            f"{base_u8.shape} != {(H, W)}"
        )


    if recovery_u8.shape != (
        H,
        W
    ):

        raise RuntimeError(
            f"Recovery shape mismatch: "
            f"{recovery_u8.shape} != {(H, W)}"
        )


    base = (
        base_u8
        >
        127
    )


    recovery = (
        recovery_u8
        >
        127
    )


    with open(
        unresolved_json,
        "r",
        encoding="utf-8"
    ) as f:

        unresolved = json.load(
            f
        )


    unresolved_ids = sorted(
        {
            int(
                row[
                    "inventory_id"
                ]
            )
            for row in unresolved
        }
    )


    print(
        "=" * 90
    )

    print(
        "TEST07 STAGE08"
    )

    print(
        "STAGE01-SEEDED LOCAL RECOVERY AUDIT"
    )

    print(
        "=" * 90
    )


    print(
        "MASTER:",
        master_path
    )

    print(
        "SIZE:",
        W,
        "x",
        H
    )


    print(
        "BASE MASK:",
        base_mask_path
    )


    print(
        "RECOVERY SOURCE:",
        recovery_path
    )


    print(
        "BASE PIXELS:",
        int(
            base.sum()
        )
    )


    print(
        "RECOVERY SOURCE PIXELS:",
        int(
            recovery.sum()
        )
    )


    print(
        "UNRESOLVED INSTANCE IDS:",
        unresolved_ids
    )


    # ========================================================
    # DISTANCE FROM TRUSTED BASE
    # ========================================================

    inverse_base = (
        ~base
    ).astype(
        np.uint8
    )


    distance = cv2.distanceTransform(
        inverse_base,
        cv2.DIST_L2,
        5
    )


    # ========================================================
    # LEVEL 1 — IMMEDIATE BOUNDARY COMPLETION
    # ========================================================

    accepted = np.zeros(
        (
            H,
            W
        ),
        dtype=bool
    )


    level1 = (
        recovery
        &
        (
            distance
            <=
            VERY_CLOSE
        )
    )


    accepted |= (
        level1
    )


    # ========================================================
    # CONNECTED COMPONENT AUDIT
    # ========================================================

    (
        num_labels,
        labels,
        stats,
        centroids
    ) = cv2.connectedComponentsWithStats(
        recovery.astype(
            np.uint8
        ),
        connectivity=8
    )


    component_report = []


    for label_id in range(
        1,
        num_labels
    ):

        component = (
            labels
            ==
            label_id
        )


        area = int(
            component.sum()
        )


        if area < 3:

            continue


        ys, xs = np.where(
            component
        )


        x1 = int(
            xs.min()
        )

        x2 = int(
            xs.max()
        ) + 1

        y1 = int(
            ys.min()
        )

        y2 = int(
            ys.max()
        ) + 1


        bw = (
            x2 - x1
        )

        bh = (
            y2 - y1
        )


        # ====================================================
        # LOCAL BASE SUPPORT
        # ====================================================

        pad = max(
            6,
            int(
                round(
                    max(
                        bw,
                        bh
                    )
                    *
                    0.10
                )
            )
        )


        rx1 = max(
            0,
            x1 - pad
        )

        ry1 = max(
            0,
            y1 - pad
        )

        rx2 = min(
            W,
            x2 + pad
        )

        ry2 = min(
            H,
            y2 + pad
        )


        local_base = (
            base[
                ry1:ry2,
                rx1:rx2
            ]
        )


        base_support = int(
            local_base.sum()
        )


        component_dist = (
            distance[
                component
            ]
        )


        min_dist = float(
            component_dist.min()
        )


        mean_dist = float(
            component_dist.mean()
        )


        p90_dist = float(
            np.percentile(
                component_dist,
                90
            )
        )


        # ====================================================
        # GEOMETRY
        # ====================================================

        bbox_area = max(
            1,
            bw
            *
            bh
        )


        fill_ratio = (
            area
            /
            bbox_area
        )


        image_fraction = (
            area
            /
            float(
                W
                *
                H
            )
        )


        giant = bool(

            image_fraction
            >
            0.08

            or

            (
                bw
                >
                0.55
                *
                W

                and

                bh
                >
                0.20
                *
                H
            )

            or

            (
                bh
                >
                0.45
                *
                H

                and

                bw
                >
                0.35
                *
                W
            )
        )


        # ====================================================
        # DECISION
        # ====================================================

        decision = (
            "REJECT"
        )

        reason = (
            ""
        )


        # Case A:
        # tightly connected completion.
        if (
            min_dist
            <=
            VERY_CLOSE

            and

            p90_dist
            <=
            CLOSE
        ):

            decision = (
                "KEEP"
            )

            reason = (
                "tight_base_connection"
            )


        # Case B:
        # moderate but local extension.
        elif (
            min_dist
            <=
            VERY_CLOSE

            and

            mean_dist
            <=
            CLOSE

            and

            p90_dist
            <=
            MEDIUM

            and

            not giant
        ):

            decision = (
                "KEEP"
            )

            reason = (
                "local_base_extension"
            )


        # Case C:
        # small isolated completion near trusted foreground.
        elif (
            area
            <=
            1200

            and

            min_dist
            <=
            CLOSE

            and

            mean_dist
            <=
            MEDIUM

            and

            not giant
        ):

            decision = (
                "KEEP"
            )

            reason = (
                "small_near_base"
            )


        else:

            if giant:

                reason = (
                    "broad_or_giant_region"
                )

            elif (
                min_dist
                >
                CLOSE
            ):

                reason = (
                    "too_far_from_base"
                )

            elif (
                p90_dist
                >
                MAX_DISTANCE
            ):

                reason = (
                    "extends_too_far"
                )

            else:

                reason = (
                    "insufficient_base_support"
                )


        # ====================================================
        # ACCEPT ONLY LOCAL PORTION
        # ====================================================

        if (
            decision
            ==
            "KEEP"
        ):

            local_component = (
                component
                &
                (
                    distance
                    <=
                    MAX_DISTANCE
                )
            )


            accepted |= (
                local_component
            )


        component_report.append({

            "component":
                int(
                    label_id
                ),

            "area":
                area,

            "bbox": [
                x1,
                y1,
                x2,
                y2
            ],

            "bbox_width":
                bw,

            "bbox_height":
                bh,

            "fill_ratio":
                round(
                    float(
                        fill_ratio
                    ),
                    4
                ),

            "image_fraction":
                round(
                    float(
                        image_fraction
                    ),
                    5
                ),

            "base_support_nearby":
                base_support,

            "min_distance_to_base":
                round(
                    min_dist,
                    3
                ),

            "mean_distance_to_base":
                round(
                    mean_dist,
                    3
                ),

            "p90_distance_to_base":
                round(
                    p90_dist,
                    3
                ),

            "giant":
                giant,

            "decision":
                decision,

            "reason":
                reason
        })


    # ========================================================
    # NEW PIXELS ONLY
    # ========================================================

    accepted_recovery = (
        accepted
        &
        (~base)
    )


    rejected_recovery = (
        recovery
        &
        (~accepted_recovery)
        &
        (~base)
    )


    final_mask = (
        base
        |
        accepted_recovery
    )


    # ========================================================
    # SAVE MASKS
    # ========================================================

    recovery_source_path = (
        output_dir
        /
        "00_stage07_recovery_source.png"
    )


    accepted_path = (
        output_dir
        /
        "01_base_seeded_accepted_recovery.png"
    )


    rejected_path = (
        output_dir
        /
        "02_base_seeded_rejected_recovery.png"
    )


    final_mask_path = (
        output_dir
        /
        "03_final_props_mask.png"
    )


    save_mask(
        recovery_source_path,
        recovery
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
        final_mask_path,
        final_mask
    )


    # ========================================================
    # EXACT RGB RGBA
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
        *
        255
    )


    rgba_path = (
        output_dir
        /
        "04_all_props_rgba.png"
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
        master[
            final_mask
        ]
    )


    preview_path = (
        output_dir
        /
        "05_transparency_preview.png"
    )


    Image.fromarray(
        preview
    ).save(
        preview_path
    )


    # ========================================================
    # ACCEPTED RECOVERY OVERLAY
    # ========================================================

    overlay = (
        master.copy()
    )


    highlighted = (
        overlay.copy()
    )


    highlighted[
        accepted_recovery
    ] = [
        255,
        255,
        255
    ]


    overlay = cv2.addWeighted(
        overlay,
        0.65,
        highlighted,
        0.35,
        0
    )


    overlay_path = (
        output_dir
        /
        "06_accepted_recovery_overlay.png"
    )


    Image.fromarray(
        overlay
    ).save(
        overlay_path
    )


    # ========================================================
    # REPORT
    # ========================================================

    accepted_components = sum(
        1
        for row in component_report
        if row[
            "decision"
        ]
        ==
        "KEEP"
    )


    rejected_components = sum(
        1
        for row in component_report
        if row[
            "decision"
        ]
        ==
        "REJECT"
    )


    report = {

        "experiment":
            "TEST07_STAGE08",

        "method":
            "Stage01-seeded local recovery audit",

        "master":
            str(
                master_path
            ),

        "base_mask":
            str(
                base_mask_path
            ),

        "recovery_source":
            str(
                recovery_path
            ),

        "unresolved_json":
            str(
                unresolved_json
            ),

        "master_size": [
            W,
            H
        ],

        "base_pixels":
            int(
                base.sum()
            ),

        "stage07_recovery_pixels":
            int(
                recovery.sum()
            ),

        "accepted_recovery_pixels":
            int(
                accepted_recovery.sum()
            ),

        "rejected_recovery_pixels":
            int(
                rejected_recovery.sum()
            ),

        "final_pixels":
            int(
                final_mask.sum()
            ),

        "components_total":
            len(
                component_report
            ),

        "components_accepted":
            accepted_components,

        "components_rejected":
            rejected_components,

        "unresolved_inventory_instances":
            unresolved_ids,

        "thresholds": {

            "very_close":
                VERY_CLOSE,

            "close":
                CLOSE,

            "medium":
                MEDIUM,

            "max_distance":
                MAX_DISTANCE
        },

        "components":
            component_report,

        "rules": [
            "Stage01 is never replaced",
            "Stage01 is the trusted foreground base",
            "Stage07 SAM2 recovery is candidate recovery only",
            "recovery must remain spatially local to Stage01",
            "broad/giant regions are rejected",
            "unresolved Stage06 instances are not claimed recovered",
            "exact RGB comes only from locked resized master"
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
                    final_mask_path
                ),

            "final_rgba":
                str(
                    rgba_path
                ),

            "transparency_preview":
                str(
                    preview_path
                ),

            "recovery_overlay":
                str(
                    overlay_path
                )
        }
    }


    report_path = (
        output_dir
        /
        "stage08_report.json"
    )


    report_path.write_text(
        json.dumps(
            report,
            indent=2
        )
    )


    # ========================================================
    # FINISH
    # ========================================================

    print()
    print(
        "=" * 90
    )

    print(
        "TEST07 STAGE08 RESULT"
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
        "STAGE07 RECOVERY PIXELS:",
        int(
            recovery.sum()
        )
    )


    print(
        "ACCEPTED RECOVERY:",
        int(
            accepted_recovery.sum()
        )
    )


    print(
        "REJECTED RECOVERY:",
        int(
            rejected_recovery.sum()
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
        accepted_components
    )


    print(
        "REJECTED:",
        rejected_components
    )


    print()
    print(
        "UNRESOLVED INSTANCES:",
        unresolved_ids
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
        "RECOVERY MASK:"
    )


    print(
        accepted_path
    )


    print()
    print(
        "REJECTED MASK:"
    )


    print(
        rejected_path
    )


    print()
    print(
        "STAGE01 WAS NOT REPLACED."
    )


    print(
        "ONLY LOCAL STAGE01-SUPPORTED RECOVERY WAS ADDED."
    )


    print(
        "UNRESOLVED INSTANCES REMAIN UNRESOLVED."
    )


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
        "--recovery-mask",
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


    args = parser.parse_args()


    run(
        master_path=
            args.master,

        base_mask_path=
            args.base_mask,

        recovery_path=
            args.recovery_mask,

        unresolved_json=
            args.unresolved_json,

        output_dir=
            args.output_dir
    )
