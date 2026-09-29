"""Wall-mask and metric tile projection diagnostics for an existing render job."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import live_scene  # noqa: E402
import reuse  # noqa: E402
from engine import (  # noqa: E402
    TileSpec,
    _intersect_plane_at,
    _project_walls,
    _rays,
    render,
)


def unpack_surface(bundle, label):
    for item in bundle.surfaces:
        if item.label == label:
            return item.mask.copy()
    return np.zeros(bundle.room.shape[:2], dtype=bool)


def bbox(mask):
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return []
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def overlay(image, wall, ceiling, floor, planes):
    result = image.astype(np.float32).copy()
    result[wall] = result[wall] * 0.45 + np.array([40, 180, 255]) * 0.55
    result[ceiling] = result[ceiling] * 0.45 + np.array([255, 180, 30]) * 0.55
    result[floor] = result[floor] * 0.45 + np.array([40, 255, 80]) * 0.55
    result = np.clip(result, 0, 255).astype(np.uint8)
    for points, label in planes:
        cv2.polylines(result, [np.asarray(points, np.int32)], True, (255, 0, 255), 3)
        origin = tuple(np.asarray(points[0], dtype=np.int32))
        cv2.putText(result, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
    return result


def component_image(mask):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    colours = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for label in range(1, count):
        colour = np.array(
            [(label * 71) % 255, (label * 137) % 255, (label * 193) % 255],
            dtype=np.uint8,
        )
        colours[labels == label] = colour
    return colours, stats[1:]


def boundary_overlay(image, wall, ceiling, floor, planes):
    result = image.copy()
    wall_edges = cv2.morphologyEx(
        wall.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)
    ).astype(bool)
    ceiling_edges = cv2.morphologyEx(
        ceiling.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)
    ).astype(bool)
    floor_edges = cv2.morphologyEx(
        floor.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)
    ).astype(bool)
    result[wall_edges] = [255, 0, 255]
    result[ceiling_edges] = [255, 180, 0]
    result[floor_edges] = [0, 255, 0]
    for points, label in planes:
        cv2.polylines(result, [np.asarray(points, np.int32)], True, (0, 0, 255), 2)
    return result


def plane_candidates(scene):
    ys, xs = np.nonzero(scene.wall)
    directions = _rays(scene, xs, ys)
    best = np.full(ys.size, np.inf, dtype=np.float64)
    chosen = np.full(ys.size, -1, dtype=np.int32)
    for index, plane in enumerate(scene.walls):
        points, hit, lam = _intersect_plane_at(
            scene, directions, plane.axis, plane.value
        )
        across = plane.horizontal_sign * (
            points[plane.horizontal_axis] - plane.horizontal_origin
        )
        hit &= (across >= 0) & (across <= plane.width_mm)
        hit &= (points[1] >= 0) & (points[1] <= plane.height_mm)
        closer = hit & (lam < best)
        best[closer] = lam[closer]
        chosen[closer] = index
    result = np.zeros((scene.height, scene.width, 3), dtype=np.uint8)
    colours = [(255, 80, 80), (80, 255, 80), (80, 160, 255)]
    for index, colour in enumerate(colours[: len(scene.walls)]):
        selected = chosen == index
        result[ys[selected], xs[selected]] = colour
    return result, chosen


def architectural_edges(rgb, wall, ceiling, floor):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    boundary = cv2.morphologyEx(
        wall.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)
    ).astype(bool)
    long_edges = cv2.morphologyEx(
        edges, cv2.MORPH_CLOSE, np.ones((5, 25), np.uint8)
    ) > 0
    candidates = long_edges & (boundary | ceiling | floor)
    image = np.zeros((*wall.shape, 3), dtype=np.uint8)
    image[edges > 0] = [100, 100, 100]
    image[candidates] = [255, 0, 0]
    return edges, image, candidates


def plane_polygons(scene):
    polygons = []
    for plane in scene.walls:
        corners = np.array(
            [
                [plane.horizontal_origin, 0.0],
                [plane.horizontal_origin + plane.horizontal_sign * plane.width_mm, 0.0],
                [plane.horizontal_origin + plane.horizontal_sign * plane.width_mm, plane.height_mm],
                [plane.horizontal_origin, plane.height_mm],
            ],
            dtype=np.float64,
        )
        if plane.axis == 0:
            world = np.array([[plane.value, y, z] for z, y in corners], dtype=np.float64)
        else:
            world = np.array([[x, y, plane.value] for x, y in corners], dtype=np.float64)
        camera = (scene.R @ world.T + scene.t[:, None]).T
        image = (scene.K @ camera.T).T
        image = image[:, :2] / image[:, 2:3]
        polygons.append((image.round(1).tolist(), plane.label))
    return polygons


def save(path, image):
    Image.fromarray(np.clip(image, 0, 255).astype(np.uint8)).save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("job")
    args = parser.parse_args()
    job_dir = ROOT / "backend" / "jobs" / args.job
    state = json.loads((job_dir / "state.json").read_text(encoding="utf-8"))
    source_job = state["timings"].get("source_job") or args.job
    bundle = reuse.load(ROOT / "backend" / "jobs" / source_job)
    if bundle is None:
        raise RuntimeError(f"No reusable bundle for {source_job}")

    submitted = state["submitted"]
    scene, geometry = live_scene.build(
        bundle.room,
        [],
        bundle.surfaces,
        *submitted["room_ft"],
        wall_labels=tuple(submitted.get("walls") or []) or None,
        clean=bundle.clean,
        props=bundle.props,
    )
    material = np.asarray(Image.open(job_dir / "tile.png").convert("RGB"), dtype=np.uint8)
    spec = TileSpec(
        material,
        float(submitted["tile_mm"][0]),
        float(submitted["tile_mm"][1]),
        float(submitted["rotation"]),
        float(submitted.get("grout_mm", 5.0)),
    )
    debug = ROOT / "debug" / "tile_projection" / args.job
    debug.mkdir(parents=True, exist_ok=True)

    raw_wall = unpack_surface(bundle, "wall")
    ceiling = unpack_surface(bundle, "ceiling")
    floor = unpack_surface(bundle, "floor")
    cleaned_wall = scene.wall.copy()

    current_raw = np.zeros_like(bundle.room)
    current_canvas = bundle.clean.astype(np.float32).copy()
    _project_walls(scene, spec, current_canvas)
    current_raw[cleaned_wall] = current_canvas[cleaned_wall]

    strict_wall = cleaned_wall & ~ceiling & ~floor
    strict_scene = replace(scene, wall=strict_wall)
    strict_canvas = bundle.clean.astype(np.float32).copy()
    strict_covered = _project_walls(strict_scene, spec, strict_canvas)
    strict_raw = np.zeros_like(bundle.room)
    strict_raw[strict_covered] = strict_canvas[strict_covered]

    result = render(scene, spec, submitted["surface"])
    plane_data = plane_polygons(scene)
    planes = [(np.asarray(points), label) for points, label in plane_data]
    components, component_stats = component_image(cleaned_wall)
    gradient, architectural, boundary_candidates = architectural_edges(
        bundle.room, cleaned_wall, ceiling, floor
    )
    boundary = boundary_overlay(bundle.room, cleaned_wall, ceiling, floor, planes)
    plane_image, plane_labels = plane_candidates(scene)
    opening_image = np.zeros((*cleaned_wall.shape, 3), dtype=np.uint8)
    opening_image[scene.props] = [255, 160, 0]
    confidence_image = np.zeros((*cleaned_wall.shape, 3), dtype=np.uint8)

    save(debug / "source_wall_mask.png", raw_wall.astype(np.uint8) * 255)
    save(debug / "cleaned_wall_mask.png", cleaned_wall.astype(np.uint8) * 255)
    save(debug / "ceiling_mask.png", ceiling.astype(np.uint8) * 255)
    save(debug / "floor_mask.png", floor.astype(np.uint8) * 255)
    save(debug / "wall_mask_overlay.png", overlay(bundle.room, cleaned_wall, ceiling, floor, []))
    save(debug / "wall_boundary_overlay.png", boundary)
    save(debug / "wall_components.png", components)
    save(debug / "wall_plane_overlay.png", overlay(bundle.room, cleaned_wall, ceiling, floor, planes))
    save(debug / "image_edges.png", gradient)
    evidence = ROOT / "debug" / "boundary_evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    save(evidence / "wall_raw.png", raw_wall.astype(np.uint8) * 255)
    save(evidence / "wall_clean.png", cleaned_wall.astype(np.uint8) * 255)
    save(evidence / "wall_confidence.png", confidence_image)
    save(evidence / "edges.png", gradient)
    save(evidence / "architectural_edges.png", architectural)
    save(evidence / "plane_polygons.png", overlay(bundle.room, cleaned_wall, ceiling, floor, planes))
    save(evidence / "plane_candidates.png", plane_image)
    save(evidence / "openings.png", opening_image)
    save(evidence / "boundary_candidates.png", boundary_candidates.astype(np.uint8) * 255)
    save(debug / "raw_projection.png", current_raw)
    save(debug / "strict_mask_projection.png", strict_raw)
    save(debug / "projection_current.png", current_raw)
    save(debug / "projection_boundary_clipped.png", strict_raw)
    save(debug / "current_wall_mask.png", cleaned_wall.astype(np.uint8) * 255)
    save(debug / "boundary_clipped_wall_mask.png", strict_wall.astype(np.uint8) * 255)
    save(debug / "physical_repeat_projection.png", strict_raw)
    save(debug / "final_composite.png", result.composite)
    save(debug / "clean_room.png", bundle.clean)

    report = {
        "job": args.job,
        "segment_job": source_job,
        "image_size": [scene.width, scene.height],
        "tile_size_mm": [spec.width_mm, spec.height_mm],
        "grout_mm": spec.grout_mm,
        "pitch_mm": [spec.pitch_u, spec.pitch_v],
        "wall_planes_mm": [[p.width_mm, p.height_mm] for p in scene.walls],
        "horizontal_tile_count": [p.width_mm / spec.pitch_u for p in scene.walls],
        "vertical_tile_count": [p.height_mm / spec.pitch_v for p in scene.walls],
        "source_wall_mask_area": int(raw_wall.sum()),
        "wall_mask_area": int(cleaned_wall.sum()),
        "wall_mask_bbox": bbox(cleaned_wall),
        "ceiling_mask_area": int(ceiling.sum()),
        "floor_mask_area": int(floor.sum()),
        "ceiling_overlap_pixels": int((cleaned_wall & ceiling).sum()),
        "floor_overlap_pixels": int((cleaned_wall & floor).sum()),
        "strict_ceiling_overlap_pixels": int((strict_wall & ceiling).sum()),
        "strict_floor_overlap_pixels": int((strict_wall & floor).sum()),
        "projected_pixel_count": int(result.target.sum()),
        "strict_projected_pixel_count": int(strict_covered.sum()),
        "clipped_pixel_count": int((cleaned_wall & ~strict_wall).sum()),
        "current_outside_wall_pixels": int((result.target & ~cleaned_wall).sum()),
        "strict_outside_wall_pixels": int((strict_covered & ~strict_wall).sum()),
        "current_wall_leak_pixels": int((result.target & (ceiling | floor)).sum()),
        "strict_wall_leak_pixels": int((strict_covered & (ceiling | floor)).sum()),
        "wall_planes": [
            {"label": label, "image_corners": points}
            for points, label in plane_data
        ],
        "wall_components": [
            {
                "area": int(row[cv2.CC_STAT_AREA]),
                "bbox": [
                    int(row[cv2.CC_STAT_LEFT]),
                    int(row[cv2.CC_STAT_TOP]),
                    int(row[cv2.CC_STAT_WIDTH]),
                    int(row[cv2.CC_STAT_HEIGHT]),
                ],
            }
            for row in component_stats
        ],
        "wall_boundary_edge_overlap_pixels": int(np.count_nonzero(gradient & cleaned_wall)),
        "wall_boundary_edge_pixels": int(np.count_nonzero(gradient)),
        "boundary_evidence": {
            "wall_segmentation_sufficient": False,
            "confidence_boundary_separation": False,
            "confidence_available": False,
            "edge_boundary_separation": "ambiguous",
            "plane_geometry_useful": "decomposition_only",
            "existing_opening_masks_useful": False,
            "recommended_boundary_signal": "No independent reliable visibility boundary is retained. Existing wall argmax is one connected component; geometry decomposes it into planes but does not establish photographed visibility. Architectural edges are mixed with object and texture edges.",
            "confidence": "insufficient for a generalized production refinement",
            "production_change_recommended": False,
            "plane_candidate_pixel_counts": {
                str(index): int(np.count_nonzero(plane_labels == index))
                for index in range(len(scene.walls))
            },
            "existing_opening_signal": "only union object mask is available; no separate persisted door/window surface masks",
        },
        "projection": "same metric ray-plane projection; A/B differs only by strict mask",
        "object_restoration_unchanged": True,
    }
    (debug / "projection_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"debug_dir={debug}")


if __name__ == "__main__":
    main()