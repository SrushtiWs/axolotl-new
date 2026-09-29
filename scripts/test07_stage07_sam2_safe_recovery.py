
from pathlib import Path
import argparse
import gc
import json
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw


# ============================================================
# SAM2 IMPORT
# ============================================================

# Current workspace has previously used SAM2 source here.
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
# TEST07 — STAGE 07
#
# SEGMENTATION-SAFE BOXES
#       ↓
# SAM2 MULTIMASK
#       ↓
# LOCAL / QUALITY SAFETY GATES
#       ↓
# ONLY PIXELS MISSING FROM STAGE01 BASE ARE ADDED
#
# IMPORTANT:
#   - Stage01 clean foreground remains PRIMARY
#   - SAM2 never replaces Stage01
#   - only Stage06 SAFE candidates processed
#   - Stage06 NON-SAFE instances are preserved unresolved
#   - exact RGB comes only from locked resized master
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


def find_checkpoint(
    explicit=None
):

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


    for path in candidates:

        if path.exists():

            return path


    raise FileNotFoundError(
        "SAM2 checkpoint not found. Checked:\n"
        +
        "\n".join(
            str(p)
            for p in candidates
        )
    )


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


    bw = (
        x2 - x1
    )

    bh = (
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


    if (
        mask.sum()
        ==
        0
    ):

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
# MAIN
# ============================================================

def run(
    master_path,
    base_mask_path,
    safe_json,
    unsafe_json,
    output_dir,
    checkpoint_path=None
):

    master_path = Path(
        master_path
    )

    base_mask_path = Path(
        base_mask_path
    )

    safe_json = Path(
        safe_json
    )

    unsafe_json = Path(
        unsafe_json
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


    for required in [
        master_path,
        base_mask_path,
        safe_json,
        unsafe_json
    ]:

        if not required.exists():

            raise FileNotFoundError(
                required
            )


    checkpoint = (
        find_checkpoint(
            checkpoint_path
        )
    )


    # ========================================================
    # INPUTS
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


    W, H = (
        master_pil.size
    )


    base_mask = (
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


    if (
        base_mask.shape
        !=
        (
            H,
            W
        )
    ):

        raise RuntimeError(
            f"Base-mask shape mismatch: "
            f"{base_mask.shape} != {(H, W)}"
        )


    with open(
        safe_json,
        "r",
        encoding="utf-8"
    ) as f:

        candidates = json.load(
            f
        )


    with open(
        unsafe_json,
        "r",
        encoding="utf-8"
    ) as f:

        unresolved = json.load(
            f
        )


    print(
        "=" * 90
    )

    print(
        "TEST07 STAGE07"
    )

    print(
        "SAM2 SAFE-CANDIDATE RECOVERY"
    )

    print(
        "=" * 90
    )


    print(
        "MASTER:",
        master_path
    )

    print(
        "Resolution:",
        W,
        "x",
        H
    )

    print(
        "SAFE CANDIDATES:",
        len(
            candidates
        )
    )

    print(
        "UNRESOLVED CANDIDATES:",
        len(
            unresolved
        )
    )

    print(
        "PRIMARY BASE MASK:",
        base_mask_path
    )

    print(
        "SAM2 CHECKPOINT:",
        checkpoint
    )


    print()
    print(
        "Stage01 remains primary: YES"
    )

    print(
        "SAM2 replaces Stage01: NO"
    )

    print(
        "Only missing pixels added: YES"
    )


    # ========================================================
    # LOAD SAM2
    # ========================================================

    DEVICE = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    print()
    print(
        "Device:",
        DEVICE
    )


    if (
        DEVICE
        ==
        "cuda"
    ):

        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            )
        )


    print()
    print(
        "Loading SAM2..."
    )


    sam2_model = (
        build_sam2(
            SAM2_CONFIG,
            str(
                checkpoint
            ),
            device=
                DEVICE
        )
    )


    predictor = (
        SAM2ImagePredictor(
            sam2_model
        )
    )


    predictor.set_image(
        master
    )


    print(
        "✅ SAM2 READY"
    )


    # ========================================================
    # GLOBAL ACCUMULATORS
    # ========================================================

    global_recovery = np.zeros(
        (
            H,
            W
        ),
        dtype=bool
    )


    selected_union = np.zeros(
        (
            H,
            W
        ),
        dtype=bool
    )


    results = []


    # ========================================================
    # PROCESS SAFE CANDIDATES
    # ========================================================

    for index, row in enumerate(
        candidates,
        start=1
    ):

        name = str(
            row.get(
                "inventory_name",
                "object"
            )
        )


        iid = int(
            row.get(
                "inventory_id",
                index
            )
        )


        box = np.asarray(
            row[
                "bbox"
            ],
            dtype=np.float32
        )


        print()
        print(
            "-" * 90
        )


        print(
            f"[{index}/{len(candidates)}]",
            f"ID={iid:02d}",
            name
        )


        print(
            "BOX:",
            [
                round(
                    float(v),
                    1
                )
                for v in box
            ]
        )


        # ====================================================
        # SAM2 MULTIMASK
        # ====================================================

        with torch.inference_mode():

            masks, scores, logits = (
                predictor.predict(
                    box=
                        box,
                    multimask_output=
                        True
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


            if (
                raw_area
                ==
                0
            ):

                continue


            # --------------------------------------------
            # LOCAL CONTAINMENT
            # --------------------------------------------

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


            # SAM2 is not allowed outside local neighborhood.
            mask = (
                local_pixels
            )


            area = int(
                mask.sum()
            )


            if (
                area
                ==
                0
            ):

                continue


            # --------------------------------------------
            # ORIGINAL BOX FRACTION
            # --------------------------------------------

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


            # --------------------------------------------
            # BORDER TOUCH
            # --------------------------------------------

            border_touch = (
                border_touch_ratio(
                    mask,
                    local_box
                )
            )


            # --------------------------------------------
            # BASE MASK SUPPORT
            # --------------------------------------------

            base_overlap = int(
                (
                    mask
                    &
                    base_mask
                ).sum()
            )


            base_coverage = float(
                base_overlap
                /
                max(
                    1,
                    area
                )
            )


            # --------------------------------------------
            # MISSING PIXELS
            # --------------------------------------------

            missing = (
                mask
                &
                (~base_mask)
            )


            missing_ratio = float(
                missing.sum()
                /
                max(
                    1,
                    area
                )
            )


            # --------------------------------------------
            # QUALITY
            # --------------------------------------------

            sam_score = float(
                scores[
                    m_idx
                ]
            )


            quality = (
                0.55
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
                0.10
                *
                min(
                    1.0,
                    base_coverage
                    /
                    0.60
                )
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

                "missing":
                    missing,

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

                "base_coverage":
                    base_coverage,

                "missing_ratio":
                    missing_ratio
            })


        # ====================================================
        # SELECT BEST
        # ====================================================

        if not evaluated:

            print(
                "❌ No usable SAM2 mask"
            )


            result = dict(
                row
            )


            result.update({
                "stage07_selected":
                    False,

                "stage07_reason":
                    "no_usable_mask"
            })


            results.append(
                result
            )

            continue


        evaluated = sorted(
            evaluated,
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


        selected_mask = (
            best[
                "mask"
            ]
        )


        missing = (
            best[
                "missing"
            ]
        )


        # ====================================================
        # SAFETY GATE
        # ====================================================

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


        print(
            "SAM score:",
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
            "Base coverage:",
            round(
                best[
                    "base_coverage"
                ],
                4
            )
        )


        print(
            "Missing ratio:",
            round(
                best[
                    "missing_ratio"
                ],
                4
            )
        )


        print(
            "SAFE:",
            safe
        )


        if safe:

            selected_union |= (
                selected_mask
            )


            global_recovery |= (
                missing
            )


        # ====================================================
        # INDIVIDUAL OUTPUTS
        # ====================================================

        prefix = (
            f"{index:03d}_"
            f"{iid:02d}_"
            +
            safe_name(
                name
            )
        )


        Image.fromarray(
            selected_mask.astype(
                np.uint8
            )
            *
            255
        ).save(
            objects_dir
            /
            f"{prefix}_mask.png"
        )


        Image.fromarray(
            missing.astype(
                np.uint8
            )
            *
            255
        ).save(
            objects_dir
            /
            f"{prefix}_missing_vs_base.png"
        )


        # Exact master RGB.
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
            selected_mask.astype(
                np.uint8
            )
            *
            255
        )


        Image.fromarray(
            rgba,
            mode="RGBA"
        ).save(
            objects_dir
            /
            f"{prefix}_rgba.png"
        )


        diagnostic = (
            master_pil.copy()
        )


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


        diagnostic.save(
            objects_dir
            /
            f"{prefix}_box.png"
        )


        result = dict(
            row
        )


        result.update({

            "stage07_selected":
                safe,

            "stage07_sam_score":
                best[
                    "sam_score"
                ],

            "stage07_quality":
                best[
                    "quality"
                ],

            "stage07_area":
                best[
                    "area"
                ],

            "stage07_local_containment":
                best[
                    "containment"
                ],

            "stage07_original_box_fraction":
                best[
                    "original_box_fraction"
                ],

            "stage07_border_touch":
                best[
                    "border_touch"
                ],

            "stage07_base_coverage":
                best[
                    "base_coverage"
                ],

            "stage07_missing_ratio":
                best[
                    "missing_ratio"
                ],

            "stage07_selected_mask_index":
                best[
                    "mask_index"
                ]
        })


        results.append(
            result
        )


    # ========================================================
    # SAVE INPUTS
    # ========================================================

    input_copy_path = (
        output_dir
        /
        "00_input_safe_candidates.json"
    )


    input_copy_path.write_text(
        json.dumps(
            candidates,
            indent=2
        )
    )


    unresolved_copy_path = (
        output_dir
        /
        "00_unresolved_stage06_candidates.json"
    )


    unresolved_copy_path.write_text(
        json.dumps(
            unresolved,
            indent=2
        )
    )


    # ========================================================
    # UNION MASKS
    # ========================================================

    selected_union_path = (
        output_dir
        /
        "01_sam2_selected_union.png"
    )


    Image.fromarray(
        selected_union.astype(
            np.uint8
        )
        *
        255
    ).save(
        selected_union_path
    )


    recovery_path = (
        output_dir
        /
        "02_missing_recovery_mask.png"
    )


    Image.fromarray(
        global_recovery.astype(
            np.uint8
        )
        *
        255
    ).save(
        recovery_path
    )


    # ========================================================
    # BASE + RECOVERY
    # ========================================================

    final_mask = (
        base_mask
        |
        global_recovery
    )


    final_mask_path = (
        output_dir
        /
        "03_all_props_mask.png"
    )


    Image.fromarray(
        final_mask.astype(
            np.uint8
        )
        *
        255
    ).save(
        final_mask_path
    )


    # ========================================================
    # EXACT RGB RGBA
    # ========================================================

    final_rgba = np.zeros(
        (
            H,
            W,
            4
        ),
        dtype=np.uint8
    )


    final_rgba[
        :,
        :,
        :3
    ] = (
        master
    )


    final_rgba[
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


    final_rgba_path = (
        output_dir
        /
        "04_all_props_rgba.png"
    )


    Image.fromarray(
        final_rgba,
        mode="RGBA"
    ).save(
        final_rgba_path
    )


    # ========================================================
    # TRANSPARENCY PREVIEW
    # ========================================================

    yy, xx = np.indices(
        (
            H,
            W
        )
    )


    tile = 24


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
    # RESULTS
    # ========================================================

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


    selected_count = sum(
        bool(
            row.get(
                "stage07_selected",
                False
            )
        )
        for row in results
    )


    # ========================================================
    # INSTANCE COVERAGE
    # ========================================================

    safe_input_ids = {
        int(
            row[
                "inventory_id"
            ]
        )
        for row in candidates
    }


    selected_instance_ids = {
        int(
            row[
                "inventory_id"
            ]
        )
        for row in results
        if row.get(
            "stage07_selected",
            False
        )
    }


    sam2_instances_missing = (
        safe_input_ids
        -
        selected_instance_ids
    )


    unresolved_instance_ids = {
        int(
            row[
                "inventory_id"
            ]
        )
        for row in unresolved
    }


    # ========================================================
    # REPORT
    # ========================================================

    report = {

        "experiment":
            "TEST07_STAGE07",

        "method":
            (
                "SAM2 multimask segmentation of Stage06 "
                "segmentation-safe boxes with Stage01 "
                "missing-pixel recovery"
            ),

        "sam2_checkpoint":
            str(
                checkpoint
            ),

        "sam2_config":
            SAM2_CONFIG,

        "input_candidates":
            len(
                candidates
            ),

        "selected_candidates":
            selected_count,

        "rejected_candidates":
            (
                len(
                    candidates
                )
                -
                selected_count
            ),

        "safe_input_inventory_instances":
            len(
                safe_input_ids
            ),

        "sam2_selected_inventory_instances":
            len(
                selected_instance_ids
            ),

        "sam2_missing_inventory_instances":
            sorted(
                sam2_instances_missing
            ),

        "stage06_unresolved_inventory_instances":
            sorted(
                unresolved_instance_ids
            ),

        "base_pixels":
            int(
                base_mask.sum()
            ),

        "sam2_union_pixels":
            int(
                selected_union.sum()
            ),

        "recovered_pixels":
            int(
                global_recovery.sum()
            ),

        "final_pixels":
            int(
                final_mask.sum()
            ),

        "thresholds": {

            "local_expansion":
                LOCAL_EXPANSION,

            "min_sam_score":
                MIN_SAM_SCORE,

            "min_local_containment":
                MIN_LOCAL_CONTAINMENT,

            "max_local_border_touch":
                MAX_LOCAL_BORDER_TOUCH,

            "min_mask_pixels":
                MIN_MASK_PIXELS
        },

        "guarantees": [
            "Stage01 clean foreground mask remains primary",
            "SAM2 never replaces Stage01",
            "only pixels missing from Stage01 are added",
            "Stage06 non-safe candidates are not segmented here",
            "RGB copied only from locked resized master"
        ],

        "outputs": {

            "selected_union":
                str(
                    selected_union_path
                ),

            "recovery_mask":
                str(
                    recovery_path
                ),

            "final_mask":
                str(
                    final_mask_path
                ),

            "final_rgba":
                str(
                    final_rgba_path
                ),

            "transparency_preview":
                str(
                    preview_path
                ),

            "object_results":
                str(
                    results_path
                )
        }
    }


    report_path = (
        output_dir
        /
        "stage07_report.json"
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
        "TEST07 STAGE07 RESULT"
    )

    print(
        "=" * 90
    )


    print(
        "INPUT SAFE CANDIDATES:",
        len(
            candidates
        )
    )


    print(
        "SAM2 SAFE:",
        selected_count
    )


    print(
        "SAM2 REJECTED:",
        len(
            candidates
        )
        -
        selected_count
    )


    print()
    print(
        "SAFE INPUT INSTANCES:",
        len(
            safe_input_ids
        )
    )


    print(
        "INSTANCES WITH ACCEPTED SAM2 MASK:",
        len(
            selected_instance_ids
        )
    )


    print(
        "SAFE INSTANCES LOST BY SAM2:",
        sorted(
            sam2_instances_missing
        )
    )


    print(
        "STAGE06 UNRESOLVED INSTANCES:",
        sorted(
            unresolved_instance_ids
        )
    )


    print()
    print(
        "BASE PIXELS:",
        int(
            base_mask.sum()
        )
    )


    print(
        "RECOVERED PIXELS:",
        int(
            global_recovery.sum()
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
        "IMPORTANT OUTPUTS:"
    )


    print(
        recovery_path
    )

    print(
        final_rgba_path
    )

    print(
        preview_path
    )


    print()
    print(
        "STAGE01 WAS NEVER REPLACED."
    )

    print(
        "ONLY SAM2 PIXELS MISSING FROM STAGE01 WERE ADDED."
    )

    print(
        "STAGE06 NON-SAFE INSTANCES WERE NOT SEGMENTED."
    )


    # ========================================================
    # CLEANUP
    # ========================================================

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
        "--safe-json",
        required=True
    )


    parser.add_argument(
        "--unsafe-json",
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

        safe_json=
            args.safe_json,

        unsafe_json=
            args.unsafe_json,

        output_dir=
            args.output_dir,

        checkpoint_path=
            args.sam2_checkpoint
    )
