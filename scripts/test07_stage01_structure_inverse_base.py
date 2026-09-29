
from pathlib import Path
import argparse
import gc
import json

import cv2
import numpy as np
import torch
from PIL import Image

from transformers.models.segformer.image_processing_segformer import (
    SegformerImageProcessor
)

from transformers.models.segformer.modeling_segformer import (
    SegformerForSemanticSegmentation
)


# ============================================================
# TEST07 — STAGE 01
#
# GENERALIZED STRUCTURE-INVERSE BASE
#
# PURPOSE:
#   Detect only architectural surfaces:
#       WALL
#       FLOOR
#       CEILING
#
#   Remove those surfaces.
#
#   Everything else becomes the initial foreground candidate.
#
# MODEL:
#   NVIDIA SegFormer B5 ADE20K
#
# IMPORTANT:
#   - no fixed room type
#   - no TEST06 pixel counts
#   - no fixed image size
#   - aspect ratio preserved
#   - longest side resized to 1024
#   - no object inventory
#   - no Grounding DINO
#   - no SAM2
#   - exact resized-master RGB preserved
# ============================================================


MODEL_ID = (
    "nvidia/"
    "segformer-b5-finetuned-ade-640-640"
)

LONGEST_SIDE = 1024

# Small morphology only.
STRUCTURE_CLOSE_KERNEL = 5
STRUCTURE_OPEN_KERNEL = 3

# Remove tiny candidate noise while preserving actual objects.
MIN_FOREGROUND_COMPONENT_RATIO = 0.000003


# ============================================================
# HELPERS
# ============================================================

def normalize_label(text):

    return (
        str(text)
        .lower()
        .strip()
        .replace("-", " ")
        .replace("_", " ")
    )


def resize_preserve_aspect(
    image,
    longest_side
):

    W, H = image.size

    scale = (
        float(longest_side)
        /
        max(W, H)
    )

    # Do not upscale tiny inputs unnecessarily.
    if scale > 1.0:
        scale = 1.0

    new_w = max(
        1,
        int(round(W * scale))
    )

    new_h = max(
        1,
        int(round(H * scale))
    )

    if (
        new_w == W
        and
        new_h == H
    ):
        return image.copy()

    return image.resize(
        (new_w, new_h),
        Image.Resampling.LANCZOS
    )


def remove_tiny_components(
    mask,
    min_pixels
):

    n, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask.astype(np.uint8),
            connectivity=8
        )
    )

    output = np.zeros_like(
        mask,
        dtype=bool
    )

    kept = 0
    rejected = 0

    for label_id in range(
        1,
        n
    ):

        area = int(
            stats[
                label_id,
                cv2.CC_STAT_AREA
            ]
        )

        if area >= min_pixels:

            output[
                labels == label_id
            ] = True

            kept += 1

        else:

            rejected += 1

    return (
        output,
        kept,
        rejected
    )


def save_mask(
    path,
    mask
):

    Image.fromarray(
        mask.astype(np.uint8)
        * 255
    ).save(path)


# ============================================================
# MAIN
# ============================================================

def run(
    input_image,
    output_dir,
    model_cache
):

    input_image = Path(
        input_image
    )

    output_dir = Path(
        output_dir
    )

    model_cache = Path(
        model_cache
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    model_cache.mkdir(
        parents=True,
        exist_ok=True
    )


    if not input_image.exists():

        raise FileNotFoundError(
            input_image
        )


    print(
        "=" * 90
    )

    print(
        "TEST07 STAGE01"
    )

    print(
        "GENERALIZED STRUCTURE-INVERSE BASE"
    )

    print(
        "=" * 90
    )


    # ========================================================
    # LOAD ORIGINAL
    # ========================================================

    original = (
        Image.open(
            input_image
        )
        .convert("RGB")
    )


    original_w, original_h = (
        original.size
    )


    resized = resize_preserve_aspect(
        original,
        LONGEST_SIDE
    )


    W, H = resized.size


    master_path = (
        output_dir
        / "00_input_resized.png"
    )


    resized.save(
        master_path
    )


    master = np.asarray(
        resized
    )


    print()
    print(
        "Original:",
        original_w,
        "x",
        original_h
    )

    print(
        "Working:",
        W,
        "x",
        H
    )


    DEVICE = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    print(
        "Device:",
        DEVICE
    )


    if DEVICE == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )


    # ========================================================
    # LOAD MODEL
    # ========================================================

    print()
    print(
        "Loading SegFormer..."
    )


    processor = (
        SegformerImageProcessor
        .from_pretrained(
            MODEL_ID,
            cache_dir=str(
                model_cache
            )
        )
    )


    model = (
        SegformerForSemanticSegmentation
        .from_pretrained(
            MODEL_ID,
            cache_dir=str(
                model_cache
            )
        )
    )


    model = model.to(
        DEVICE
    )

    model.eval()


    print(
        "✅ SEGFORMER READY"
    )


    # ========================================================
    # LABEL IDS
    # ========================================================

    id2label = {
        int(k): v
        for k, v
        in model.config.id2label.items()
    }


    wall_ids = []
    floor_ids = []
    ceiling_ids = []


    for class_id, label in (
        id2label.items()
    ):

        normalized = normalize_label(
            label
        )


        if normalized == "wall":

            wall_ids.append(
                class_id
            )


        elif normalized == "floor":

            floor_ids.append(
                class_id
            )


        elif normalized == "ceiling":

            ceiling_ids.append(
                class_id
            )


    print()
    print(
        "Structural IDs:"
    )

    print(
        "WALL:",
        wall_ids
    )

    print(
        "FLOOR:",
        floor_ids
    )

    print(
        "CEILING:",
        ceiling_ids
    )


    if not wall_ids:

        raise RuntimeError(
            "ADE20K wall class not found."
        )


    if not floor_ids:

        raise RuntimeError(
            "ADE20K floor class not found."
        )


    # Ceiling may be absent from a particular config,
    # although expected in ADE20K.


    # ========================================================
    # SEMANTIC INFERENCE
    # ========================================================

    inputs = processor(
        images=resized,
        return_tensors="pt"
    )


    inputs = {
        key:
            value.to(
                DEVICE
            )
        for key, value
        in inputs.items()
    }


    with torch.inference_mode():

        outputs = model(
            **inputs
        )


    logits = (
        outputs.logits
    )


    upsampled_logits = (
        torch.nn.functional.interpolate(
            logits,
            size=(
                H,
                W
            ),
            mode="bilinear",
            align_corners=False
        )
    )


    semantic_map = (
        upsampled_logits
        .argmax(
            dim=1
        )[0]
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.int32
        )
    )


    print(
        "✅ SEMANTIC SEGMENTATION COMPLETE"
    )


    # ========================================================
    # SAVE SEMANTIC MAP
    # ========================================================

    semantic_path = (
        output_dir
        / "01_semantic_label_map.png"
    )


    cv2.imwrite(
        str(
            semantic_path
        ),
        semantic_map.astype(
            np.uint16
        )
    )


    # ========================================================
    # STRUCTURE MASKS
    # ========================================================

    wall_mask = np.zeros(
        (
            H,
            W
        ),
        dtype=bool
    )


    floor_mask = np.zeros(
        (
            H,
            W
        ),
        dtype=bool
    )


    ceiling_mask = np.zeros(
        (
            H,
            W
        ),
        dtype=bool
    )


    for class_id in wall_ids:

        wall_mask |= (
            semantic_map
            == class_id
        )


    for class_id in floor_ids:

        floor_mask |= (
            semantic_map
            == class_id
        )


    for class_id in ceiling_ids:

        ceiling_mask |= (
            semantic_map
            == class_id
        )


    save_mask(
        output_dir
        / "02_wall_mask.png",
        wall_mask
    )


    save_mask(
        output_dir
        / "03_ceiling_mask.png",
        ceiling_mask
    )


    save_mask(
        output_dir
        / "04_floor_mask.png",
        floor_mask
    )


    # ========================================================
    # COMBINE STRUCTURE
    # ========================================================

    structure_mask = (
        wall_mask
        |
        floor_mask
        |
        ceiling_mask
    )


    # Small morphology only.
    close_kernel = np.ones(
        (
            STRUCTURE_CLOSE_KERNEL,
            STRUCTURE_CLOSE_KERNEL
        ),
        np.uint8
    )


    open_kernel = np.ones(
        (
            STRUCTURE_OPEN_KERNEL,
            STRUCTURE_OPEN_KERNEL
        ),
        np.uint8
    )


    structure_u8 = (
        structure_mask.astype(
            np.uint8
        )
        * 255
    )


    structure_clean = cv2.morphologyEx(
        structure_u8,
        cv2.MORPH_CLOSE,
        close_kernel
    )


    structure_clean = cv2.morphologyEx(
        structure_clean,
        cv2.MORPH_OPEN,
        open_kernel
    )


    structure_clean = (
        structure_clean
        > 127
    )


    save_mask(
        output_dir
        / "05_structure_mask.png",
        structure_clean
    )


    # ========================================================
    # FOREGROUND INVERSION
    # ========================================================

    raw_candidate = (
        ~structure_clean
    )


    save_mask(
        output_dir
        / "07_foreground_candidate_raw.png",
        raw_candidate
    )


    min_component_pixels = max(
        2,
        int(
            round(
                H
                *
                W
                *
                MIN_FOREGROUND_COMPONENT_RATIO
            )
        )
    )


    foreground_clean, kept_components, rejected_components = (
        remove_tiny_components(
            raw_candidate,
            min_component_pixels
        )
    )


    save_mask(
        output_dir
        / "06_foreground_mask_clean.png",
        foreground_clean
    )


    save_mask(
        output_dir
        / "08_foreground_candidate_clean.png",
        foreground_clean
    )


    # ========================================================
    # FOREGROUND RGBA
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
    ] = master


    rgba[
        :,
        :,
        3
    ] = (
        foreground_clean.astype(
            np.uint8
        )
        * 255
    )


    rgba_path = (
        output_dir
        / "09_foreground_rgba.png"
    )


    Image.fromarray(
        rgba
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


    preview = checker_rgb.copy()


    preview[
        foreground_clean
    ] = master[
        foreground_clean
    ]


    preview_path = (
        output_dir
        / "10_transparency_preview.png"
    )


    Image.fromarray(
        preview
    ).save(
        preview_path
    )


    # ========================================================
    # REPORT
    # ========================================================

    report = {
        "experiment":
            "TEST07_STAGE01",

        "method":
            "SegFormer ADE20K wall/floor/ceiling inversion",

        "model_id":
            MODEL_ID,

        "input":
            str(
                input_image
            ),

        "resized_master":
            str(
                master_path
            ),

        "original_resolution": {
            "width":
                int(
                    original_w
                ),

            "height":
                int(
                    original_h
                )
        },

        "working_resolution": {
            "width":
                int(W),

            "height":
                int(H)
        },

        "structural_class_ids": {
            "wall":
                wall_ids,

            "floor":
                floor_ids,

            "ceiling":
                ceiling_ids
        },

        "pixels": {
            "image":
                int(
                    H * W
                ),

            "wall":
                int(
                    wall_mask.sum()
                ),

            "floor":
                int(
                    floor_mask.sum()
                ),

            "ceiling":
                int(
                    ceiling_mask.sum()
                ),

            "structure_clean":
                int(
                    structure_clean.sum()
                ),

            "foreground_candidate":
                int(
                    foreground_clean.sum()
                )
        },

        "foreground_components": {
            "min_component_pixels":
                int(
                    min_component_pixels
                ),

            "kept":
                int(
                    kept_components
                ),

            "tiny_rejected":
                int(
                    rejected_components
                )
        },

        "guarantees": [
            "aspect ratio preserved",
            "no fixed TEST06 counts",
            "no fixed room type",
            "only wall floor ceiling removed",
            "exact resized master RGB preserved",
            "no Grounding DINO",
            "no SAM2",
            "no object prompts"
        ]
    }


    report_path = (
        output_dir
        / "stage01_report.json"
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
        "TEST07 STAGE01 RESULT"
    )

    print(
        "=" * 90
    )


    print(
        "WALL PIXELS:",
        int(
            wall_mask.sum()
        )
    )


    print(
        "FLOOR PIXELS:",
        int(
            floor_mask.sum()
        )
    )


    print(
        "CEILING PIXELS:",
        int(
            ceiling_mask.sum()
        )
    )


    print(
        "STRUCTURE PIXELS:",
        int(
            structure_clean.sum()
        )
    )


    print(
        "FOREGROUND CANDIDATE PIXELS:",
        int(
            foreground_clean.sum()
        )
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
        "MASK:"
    )

    print(
        output_dir
        / "08_foreground_candidate_clean.png"
    )


    print()
    print(
        "REPORT:"
    )

    print(
        report_path
    )


    del model
    del processor

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--input",
        required=True
    )


    parser.add_argument(
        "--output-dir",
        required=True
    )


    parser.add_argument(
        "--model-cache",
        default=
            "/workspace/axolotl/test07/models/huggingface"
    )


    args = parser.parse_args()


    run(
        input_image=
            args.input,

        output_dir=
            args.output_dir,

        model_cache=
            args.model_cache
    )
