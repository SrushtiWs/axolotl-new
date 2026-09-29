"""Diagnostic-only SegFormer confidence analysis for reusable segment jobs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import reuse  # noqa: E402
import surfaces  # noqa: E402


def unpack(bundle, label):
    for item in bundle.surfaces:
        if item.label == label:
            return item.mask.copy()
    return np.zeros(bundle.room.shape[:2], dtype=bool)


def resize(values, shape):
    return cv2.resize(
        values.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR
    )


def save_gray(path, values):
    Image.fromarray(np.clip(values * 255.0, 0, 255).astype(np.uint8)).save(path)


def stats(values, mask):
    selected = values[mask]
    if selected.size == 0:
        return {"pixel_count": 0}
    return {
        "pixel_count": int(selected.size),
        "mean": float(selected.mean()),
        "median": float(np.median(selected)),
        "p10": float(np.percentile(selected, 10)),
        "p25": float(np.percentile(selected, 25)),
        "p75": float(np.percentile(selected, 75)),
        "p90": float(np.percentile(selected, 90)),
    }


def analyze_job(job_id, output_root):
    job_dir = ROOT / "backend" / "jobs" / job_id
    state_path = job_dir / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    source_job = state.get("timings", {}).get("source_job") or job_id
    bundle = reuse.load(ROOT / "backend" / "jobs" / source_job)
    if bundle is None:
        raise RuntimeError(f"No reusable bundle for {source_job}")

    confidence = surfaces.confidence_maps(bundle.clean)
    shape = bundle.clean.shape[:2]
    wall = unpack(bundle, "wall")
    ceiling = unpack(bundle, "ceiling")
    floor = unpack(bundle, "floor")
    wall_boundary = cv2.morphologyEx(
        wall.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)
    ).astype(bool)
    gray = cv2.cvtColor(bundle.clean, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150) > 0
    boundary_candidates = wall_boundary & edges
    wall_probability = resize(confidence["wall_probability"], shape)
    wall_margin = resize(confidence["wall_margin"], shape)
    ceiling_probability = resize(confidence["ceiling_probability"], shape)
    floor_probability = resize(confidence["floor_probability"], shape)

    output = output_root / job_id
    output.mkdir(parents=True, exist_ok=True)
    save_gray(output / "wall_probability.png", wall_probability)
    save_gray(output / "wall_margin.png", np.clip((wall_margin + 1.0) / 2.0, 0, 1))
    save_gray(output / "wall_ceiling_margin.png", np.clip(wall_probability - ceiling_probability + 1, 0, 2) / 2)
    save_gray(output / "wall_floor_margin.png", np.clip(wall_probability - floor_probability + 1, 0, 2) / 2)

    overlay = bundle.clean.astype(np.float32) * 0.55
    heat = np.zeros((*shape, 3), dtype=np.float32)
    heat[..., 0] = wall_probability * 255
    heat[..., 1] = (1.0 - wall_probability) * 255
    overlay += heat * 0.45
    overlay[boundary_candidates] = [255, 255, 255]
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(
        output / "confidence_overlay.png"
    )

    regions = {
        "accepted_wall": wall,
        "ceiling": ceiling,
        "floor": floor,
        "wall_ceiling_boundary": wall_boundary & cv2.dilate(ceiling.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool),
        "wall_floor_boundary": wall_boundary & cv2.dilate(floor.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool),
        "boundary_candidates": boundary_candidates,
    }
    region_report = {}
    for name, mask in regions.items():
        wall_stats = stats(wall_probability, mask)
        margin_stats = stats(wall_margin, mask)
        region_report[name] = {**wall_stats, "wall_margin": margin_stats}

    report = {
        "job": job_id,
        "segment_job": source_job,
        "image_size": [shape[1], shape[0]],
        "native_model_shape": confidence["native_shape"],
        "native_to_image_mapping": "wall probabilities and margins resized to image resolution with cv2.INTER_LINEAR for regional comparison; native tensors are not altered",
        "class_ids": confidence["class_ids"],
        "class_labels": {str(k): v for k, v in confidence["labels"].items()},
        "regions": region_report,
        "confidence_available": True,
        "mask_behavior_changed": False,
    }
    (output / "confidence_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("jobs", nargs="+", help="render job ids")
    parser.add_argument("--output", default="debug/segmentation_confidence")
    args = parser.parse_args()
    output_root = ROOT / args.output
    reports = [analyze_job(job_id, output_root) for job_id in args.jobs]
    combined = {"jobs": reports, "mask_behavior_changed": False}
    (output_root / "confidence_report_all.json").write_text(
        json.dumps(combined, indent=2), encoding="utf-8"
    )
    for report in reports:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
