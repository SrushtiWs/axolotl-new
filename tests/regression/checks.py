"""
The per-render checks of the regression harness. Each returns a dict with
`ok` plus the numbers behind it, so a failure says by how much.

  pixel_parity   result.png against the stored baseline, pixel for pixel
  mask_clip      no tile pixel outside FLOOR_MASK.png / WALL_MASK.png, and no
                 pixel changed outside the tiled region
  three_json     three.json survives a JSON round-trip, and its geometry
                 round-trips: plane -> quad corners -> tile grid (mm), and
                 every tiled pixel's camera ray lands inside the quad
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from tiles_backend.perspective_engine.core.uv import rotate

#: Plane / grid round-trip tolerances. Floats in three.json are float64, so
#: these are rounding bounds, not fudge factors.
PLANE_TOL_UNITS = 1e-6
GRID_TOL_MM = 1e-3
MPU_REL_TOL = 1e-9

#: Outside the tiles a pixel is original*a + clean*(1-a), a = the object alpha.
#: three/objects.png stores a in 8 bits, so the blend is reproducible to
#: within rounding: 1 level from the pixel, 1 from the alpha.
BLEND_TOL_LEVELS = 2


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def load_mask(path: Path, shape: tuple[int, int] | None = None) -> np.ndarray:
    image = Image.open(path).convert("L")
    if shape is not None and image.size != (shape[1], shape[0]):
        image = image.resize((shape[1], shape[0]), Image.NEAREST)
    return np.asarray(image) > 127


def pixel_hash(rgb: np.ndarray) -> str:
    """Hash of the decoded pixels (not the PNG bytes), with the shape."""
    digest = hashlib.sha256(str(rgb.shape).encode())
    digest.update(np.ascontiguousarray(rgb).tobytes())
    return digest.hexdigest()


def pixel_parity(result: np.ndarray, baseline: np.ndarray | None, baseline_hash: str) -> dict:
    current = pixel_hash(result)
    if current == baseline_hash:
        return {"ok": True, "differing_pixels": 0, "max_abs_diff": 0}
    out = {"ok": False, "hash": current, "baseline_hash": baseline_hash}
    if baseline is None:
        out["note"] = "baseline image not on disk; only the hash could be compared"
    elif baseline.shape != result.shape:
        out["note"] = f"shape {result.shape} != baseline {baseline.shape}"
    else:
        diff = np.abs(result.astype(np.int16) - baseline.astype(np.int16)).max(axis=2)
        out["differing_pixels"] = int((diff > 0).sum())
        out["differing_fraction"] = round(float((diff > 0).mean()), 6)
        out["max_abs_diff"] = int(diff.max())
    return out


def mask_clip(tiled: np.ndarray, allowed: np.ndarray, result: np.ndarray,
              clean: np.ndarray, original: np.ndarray, reported: dict | None,
              objects_alpha: np.ndarray | None = None) -> dict:
    """
    `tiled` is the render's own tile coverage (three/<surface>.png); `allowed`
    the surface's mask from the Clean Room job at render resolution.

    Outside the tiled region a pixel may only show the clean room or, where an
    object was restored over it, the original photograph -- blended by the
    object's alpha (three/objects.png) at its anti-aliased edges.
    """
    outside_mask = int((tiled & ~allowed).sum())
    untiled = ~tiled
    if objects_alpha is None:
        changed = untiled & np.any(result != clean, axis=2) & np.any(result != original, axis=2)
    else:
        a = (objects_alpha.astype(np.float64) / 255.0)[..., None]
        expected = original.astype(np.float64) * a + clean.astype(np.float64) * (1.0 - a)
        off = np.abs(result.astype(np.float64) - expected).max(axis=2)
        changed = untiled & (off > BLEND_TOL_LEVELS)
    changed_outside = int(changed.sum())
    reported_outside = None if reported is None else reported.get("outside_mask_after_clip")
    return {
        "ok": outside_mask == 0 and changed_outside == 0 and not reported_outside,
        "tile_pixels": int(tiled.sum()),
        "tile_pixels_outside_mask": outside_mask,
        "pixels_changed_outside_tiles": changed_outside,
        "engine_reported_outside": reported_outside,
    }


def _unit(v) -> np.ndarray:
    v = np.asarray(v, np.float64)
    return v / (np.linalg.norm(v) or 1.0)


def _same(a, b, rel=1e-6) -> bool:
    return abs(float(a) - float(b)) <= rel * max(1.0, abs(float(a)), abs(float(b)))


def room_consistency(doc: dict, key: str, tile_mm=None) -> list[str]:
    """
    10a: the record is the room object's own values -- camera, single mm scale,
    tile mm. The same rules as frontend/src/3js/threeTiles.ts roomConsistency().
    """
    s, room = doc["surfaces"][key], doc.get("room") or {}
    K = (room.get("camera") or {}).get("K")
    if not K:
        return ["no room geometry in three.json"]
    out = []
    if not _same(s["camera"]["focal_px"], K[0][0]):
        out.append(f"focal {s['camera']['focal_px']} != room {K[0][0]}")
    if not (_same(s["camera"]["cx"], K[0][2]) and _same(s["camera"]["cy"], K[1][2])):
        out.append("principal point differs from the room camera")
    mpu = s["grid"]["mm_per_unit"]
    if not _same(s["meters_per_unit"] * 1000.0, mpu):
        out.append("two scales inside one record")
    if s["kind"] == "floor" and not _same(s["plane"]["d_units"] * mpu, room["camera_height_mm"]):
        out.append(f"floor plane at {s['plane']['d_units'] * mpu:.3f} mm, room camera height {room['camera_height_mm']:.3f} mm")
    if s["kind"] == "wall" and not _same(mpu, 1.0):
        out.append(f"wall on its own scale ({mpu} mm/unit)")
    if tile_mm is not None and (not _same(s["tile"]["width_mm"], tile_mm[0]) or not _same(s["tile"]["height_mm"], tile_mm[1])):
        out.append(f"tile {s['tile']['width_mm']} x {s['tile']['height_mm']} mm != requested {tile_mm[0]} x {tile_mm[1]}")
    return out


def three_json(job_dir: Path, key: str, image_shape: tuple[int, int], tile_mm=None) -> dict:
    text = (job_dir / "three.json").read_text(encoding="utf-8")
    doc = json.loads(text)
    problems: list[str] = []

    # ---- JSON round-trip ----
    if json.loads(json.dumps(doc)) != doc:
        problems.append("json round-trip changed the document")

    surface = (doc.get("surfaces") or {}).get(key)
    if surface is None:
        return {"ok": False, "problems": [f"no surface {key!r} in three.json"]}

    for field in ("kind", "camera", "plane", "meters_per_unit", "quad_units", "quad_grid_mm",
                  "grid", "tile", "lighting", "mask"):
        if field not in surface:
            problems.append(f"missing field {field}")
    if problems:
        return {"ok": False, "problems": problems}

    cam, plane, grid = surface["camera"], surface["plane"], surface["grid"]
    h, w = image_shape
    if list(cam["image_size"]) != [w, h]:
        problems.append(f"camera image_size {cam['image_size']} != render {[w, h]}")
    if doc.get("room") is None:
        problems.append("no room in three.json")

    mpu = float(grid["mm_per_unit"])
    if abs(surface["meters_per_unit"] * 1000.0 - mpu) > MPU_REL_TOL * mpu:
        problems.append("meters_per_unit != grid.mm_per_unit / 1000")
    problems += room_consistency(doc, key, tile_mm)

    n = _unit(plane["normal"])
    d = float(plane["d_units"])
    eu = np.asarray(plane["e_u"], np.float64)
    ev = np.asarray(plane["e_v"], np.float64)
    rad = float(grid["rotation_rad"])
    off = [float(o) for o in grid["offset_units"]]

    # ---- plane -> quad corners -> tile grid (mm) ----
    quad = np.asarray(surface["quad_units"], np.float64)
    plane_err = float(np.abs(quad @ n + d).max())
    if plane_err > PLANE_TOL_UNITS * max(1.0, abs(d)):
        problems.append(f"quad corners off the plane by {plane_err:.3g} units")

    grid_err = 0.0
    for X, want in zip(quad, surface["quad_grid_mm"]):
        ur, vr = rotate(np.float64(X @ eu), np.float64(X @ ev), rad)
        got = ((float(ur) + off[0]) * mpu, (float(vr) + off[1]) * mpu)
        grid_err = max(grid_err, abs(got[0] - want[0]), abs(got[1] - want[1]))
    if grid_err > GRID_TOL_MM:
        problems.append(f"quad_grid_mm does not round-trip: {grid_err:.3g} mm")

    # ---- every tiled pixel: camera ray -> plane -> inside the quad ----
    mask_path = job_dir / surface["mask"]
    if not mask_path.is_file():
        problems.append(f"mask file {surface['mask']} missing")
        return {"ok": False, "problems": problems}
    mask = load_mask(mask_path)
    ys, xs = np.nonzero(mask)
    f, cx, cy = float(cam["focal_px"]), float(cam["cx"]), float(cam["cy"])
    rays = np.stack([(xs - cx) / f, (ys - cy) / f, np.ones_like(xs, np.float64)], axis=1)
    denom = rays @ n
    with np.errstate(divide="ignore", invalid="ignore"):
        t = -d / denom
    pts = rays * t[:, None]
    u, v = pts @ eu, pts @ ev
    qu, qv = quad @ eu, quad @ ev
    eps = 1e-9 * max(1.0, float(np.abs(quad).max()))
    inside = (np.isfinite(t) & (t > 0)
              & (u >= qu.min() - eps) & (u <= qu.max() + eps)
              & (v >= qv.min() - eps) & (v <= qv.max() + eps))
    rays_outside = int((~inside).sum())
    if rays_outside:
        problems.append(f"{rays_outside} tiled pixels' rays miss the quad")

    return {
        "ok": not problems,
        "problems": problems,
        "plane_error_units": plane_err,
        "grid_error_mm": grid_err,
        "mask_pixels": int(mask.sum()),
        "rays_outside_quad": rays_outside,
    }
