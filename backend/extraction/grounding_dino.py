"""
Grounding DINO and SAM detection stage — independent module.

Produces the common detection structure:

    {
        "label": "chair",
        "confidence": 0.92,
        "box": [x1, y1, x2, y2],
        "mask": ...
    }

`detect_to_dicts` and `run_independent` stop at boxes. `detect_and_segment` and
`run_with_masks` continue through SAM 2, refine the masks, and remove duplicates.
"""

from __future__ import annotations

import argparse
import cv2
import json
import numpy as np

from extraction import cleanup, dino, sam2
from extraction.config import ExtractionConfig

DUPLICATE_MASK_IOU = 0.92


def detect_to_dicts(rgb: np.ndarray, config: ExtractionConfig | None = None) -> list[dict]:
    """
    Run Grounding DINO over the original photo and return detections
    as the common dict structure.
    """
    if config is None:
        config = ExtractionConfig()
    detections = dino.detect(rgb, config)
    return [
        {
            "label": d.label,
            "confidence": round(d.confidence, 3),
            "box": list(d.box),
        }
        for d in detections
    ]


def detect_and_segment(
    rgb: np.ndarray,
    config: ExtractionConfig | None = None,
    surface: np.ndarray | None = None,
) -> list[dict]:
    """Run Grounding DINO, segment each box, and refine the SAM masks."""
    if config is None:
        config = ExtractionConfig()

    detections = dino.detect(rgb, config)

    if not detections:
        return []

    embedding = sam2.encode(rgb)
    logits, scores, _ = sam2.masks_for_boxes(embedding, [item.box for item in detections])

    refined: list[dict] = []

    for index, detection in enumerate(detections):
        mask = sam2.to_full_resolution(
            logits[index, int(np.argmax(scores[index]))],
            embedding,
            config.segmentation.mask_logit_threshold,
        )
        mask, _ = cleanup.clean(mask, config, surface)

        if not mask.any():
            continue

        duplicate = False

        for previous in refined:
            union = int((mask | previous["mask"]).sum())

            if union and int((mask & previous["mask"]).sum()) / union >= DUPLICATE_MASK_IOU:
                duplicate = True
                break

        if duplicate:
            continue

        refined.append(
            {
                "label": detection.label,
                "confidence": round(detection.confidence, 3),
                "box": list(detection.box),
                "mask": mask,
            }
        )

    return refined


def run_independent(image_path: str, config: ExtractionConfig | None = None) -> list[dict]:
    """
    Independent entry point: original photo -> Grounding DINO -> detections.
    """
    rgb = cv2.imread(image_path)
    if rgb is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    return detect_to_dicts(rgb, config)


def run_with_masks(image_path: str, config: ExtractionConfig | None = None) -> list[dict]:
    """Independent entry point: original photo -> DINO boxes -> SAM masks."""
    rgb = cv2.imread(image_path)
    if rgb is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    return detect_and_segment(rgb, config)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Grounding DINO on an image and print bounding-box detections as JSON."
    )
    parser.add_argument("image", help="Path to the original photo")
    args = parser.parse_args()

    print(json.dumps(run_independent(args.image), indent=2))


if __name__ == "__main__":
    main()