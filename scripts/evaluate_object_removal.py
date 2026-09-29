"""Evaluate object extraction and LaMa room reconstruction without tile rendering."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from extraction import inpaint
from extraction.config import load
from extraction.extractor import extract, to_original


DEFAULT_IMAGES = (
    ("living_room", "frontend/public/demo/rooms/living-1.jpg"),
    ("bedroom", "frontend/public/demo/rooms/bedroom-1.jpg"),
    ("kitchen", "frontend/public/demo/rooms/kitchen-1.jpg"),
    ("bathroom", "frontend/public/demo/rooms/bath-1.jpg"),
    ("mirrors", "frontend/public/demo/rooms/bath-2.jpg"),
    ("thin_objects", "frontend/public/demo/rooms/living-3.jpg"),
    ("wall_floor_contact", "frontend/public/demo/rooms/bedroom-2.jpg"),
)


def _read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if image is None:
        raise FileNotFoundError(path)

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _write_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode="L").save(path)


def _overlay(rgb: np.ndarray, masks: list[tuple[np.ndarray, tuple[int, int, int]]]) -> np.ndarray:
    canvas = rgb.astype(np.float32).copy()

    for mask, colour in masks:
        canvas[mask] = canvas[mask] * 0.45 + np.asarray(colour, dtype=np.float32) * 0.55

    return np.clip(canvas, 0, 255).astype(np.uint8)


def _detections_image(rgb: np.ndarray, result) -> np.ndarray:
    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()

    for detection in result.trace.get("raw_detections", []):
        x0, y0, x1, y1 = detection.box
        cv2.rectangle(image, (x0, y0), (x1, y1), (150, 150, 150), 1)

    for detection in result.trace.get("kept_detections", []):
        x0, y0, x1, y1 = detection.box
        cv2.rectangle(image, (x0, y0), (x1, y1), (0, 220, 0), 2)
        cv2.putText(
            image,
            f"{detection.label} {detection.confidence:.2f}",
            (x0, max(14, y0 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 220, 0),
            1,
            cv2.LINE_AA,
        )

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _box_record(detection) -> dict:
    x0, y0, x1, y1 = detection.box
    return {
        "label": detection.label,
        "confidence": round(float(detection.confidence), 4),
        "box": [int(x0), int(y0), int(x1), int(y1)],
        "box_area": int(detection.area),
    }


def evaluate(name: str, path: Path, destination: Path) -> dict:
    rgb = _read_rgb(path)
    config = load(debug=True)
    result = extract(rgb, config)

    object_masks = [
        to_original(item.mask, rgb.shape[:2])
        for item in result.objects
        if not item.is_mirror
    ]
    mirror_masks = [
        to_original(item.mask, rgb.shape[:2])
        for item in result.objects
        if item.is_mirror
    ]
    all_masks = [to_original(item.mask, rgb.shape[:2]) for item in result.objects]
    object_mask = np.logical_or.reduce(all_masks) if all_masks else np.zeros(rgb.shape[:2], dtype=bool)
    all_objects_mask = np.logical_or.reduce(object_masks) if object_masks else np.zeros(rgb.shape[:2], dtype=bool)
    mirrors_mask = np.logical_or.reduce(mirror_masks) if mirror_masks else np.zeros(rgb.shape[:2], dtype=bool)
    mirrors_mask &= object_mask

    hole = object_mask.copy()
    clean, clean_info = inpaint.clean_room(rgb, hole)

    folder = destination / name
    debug_folder = folder / "debug"
    folder.mkdir(parents=True, exist_ok=True)
    debug_folder.mkdir(parents=True, exist_ok=True)

    Image.fromarray(rgb, mode="RGB").save(folder / "original.png")
    _write_mask(folder / "current_mask.png", object_mask)
    Image.fromarray(clean, mode="RGB").save(folder / "current_CLEAN_ROOM.png")

    Image.fromarray(_detections_image(rgb, result), mode="RGB").save(debug_folder / "detections.png")
    Image.fromarray(
        _overlay(rgb, [(mask, (255, 80, 40)) for mask in all_masks]), mode="RGB"
    ).save(debug_folder / "mask_overlay.png")
    _write_mask(debug_folder / "all_objects_mask.png", all_objects_mask)
    _write_mask(debug_folder / "mirrors_mask.png", mirrors_mask)

    if hole.any():
        radius = clean_info["grow_px"]
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
        )
        written = cv2.dilate(hole.astype(np.uint8), kernel).astype(bool)
    else:
        written = hole

    _write_mask(debug_folder / "inpaint_mask.png", written)
    Image.fromarray(clean, mode="RGB").save(debug_folder / "clean_room.png")

    rejected_reasons = [item.get("reason", "unknown") for item in result.rejected]
    duplicate_count = sum("same physical object" in reason for reason in rejected_reasons)
    sam_rejected_count = sum("no SAM candidate" in reason for reason in rejected_reasons)
    surface_rejected_count = sum(
        "surface" in reason or "wall/floor/ceiling" in reason for reason in rejected_reasons
    )
    boxes = [_box_record(item) for item in result.trace.get("raw_detections", [])]
    (debug_folder / "dino_boxes.json").write_text(json.dumps(boxes, indent=2), encoding="utf-8")

    report = {
        "image": str(path),
        "size": [int(rgb.shape[1]), int(rgb.shape[0])],
        "raw_detections": result.detections_raw,
        "accepted_objects": len(result.objects),
        "accepted_mask_pixels": int(object_mask.sum()),
        "rejected": len(result.rejected),
        "rejected_reasons": rejected_reasons,
        "duplicate_detections": duplicate_count,
        "sam_rejected_candidates": sam_rejected_count,
        "surface_contaminated_candidates": surface_rejected_count,
        "accepted_physical_objects": sum(not item.is_mirror for item in result.objects),
        "mirror_detections": sum(item.is_mirror for item in result.objects),
        "clean_room": clean_info,
        "clean_room_checks": inpaint.verify(rgb, clean, written),
        "artifacts": {
            "original": "original.png",
            "mask": "current_mask.png",
            "clean_room": "current_CLEAN_ROOM.png",
        },
        "debug_artifacts": {
            "detections": "debug/detections.png",
            "mask_overlay": "debug/mask_overlay.png",
            "all_objects_mask": "debug/all_objects_mask.png",
            "mirrors_mask": "debug/mirrors_mask.png",
            "inpaint_mask": "debug/inpaint_mask.png",
            "clean_room": "debug/clean_room.png",
            "dino_boxes": "debug/dino_boxes.json",
        },
        "failure_classification": {
            "A_object_not_detected": "manual review of original.png against current_mask.png",
            "B_mask_incomplete": "manual review of current_mask.png against original.png",
            "C_wrong_pixels": "manual review of current_mask.png for floor/wall contamination",
            "D_lama_reconstruction": "manual review of current_CLEAN_ROOM.png and clean_room_checks",
        },
    }

    (folder / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("backend/jobs/object_removal_eval"),
        help="Directory for per-room artifacts and reports",
    )
    parser.add_argument(
        "--image",
        action="append",
        metavar="NAME=PATH",
        help="Override the representative set; may be repeated",
    )
    args = parser.parse_args()

    if args.image:
        images = [tuple(value.split("=", 1)) for value in args.image]
    else:
        images = DEFAULT_IMAGES

    reports = []

    for name, filename in images:
        path = Path(filename)
        print(f"[{name}] {path}", flush=True)
        reports.append(evaluate(name, path, args.output))

    summary = {"rooms": reports, "output": str(args.output)}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()