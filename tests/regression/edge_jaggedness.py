"""
Jagged tiled-wall edges, DIAGNOSIS ONLY (no code or pixel change).

    backend/.venv/bin/python tests/regression/edge_jaggedness.py <out_dir> [room ...]

Renders every wall of each baseline room as the Studio does (current code, from a
copy of the job; tests/fixtures/tile.png 600 x 600 mm, 5 mm grout) and, per wall
region, splits its boundary pixels by what lies just outside them:
  ceiling   neither WALL_MASK nor FLOOR_MASK nor an object
  floor     FLOOR_MASK
  corner    another wall's tiled region
  object    an object (props)  -- reported, never fitted
Per edge (ceiling / floor / corner of a wall, >= 1.5 % of the diagonal long):
  line        RANSAC (2 px or 0.3 % of the diagonal) + least squares on inliers
  rms_px      RMS distance of ALL edge pixels to the line (and of the inliers)
  steps       the edge profile d(u) (mean distance per 1 px along the line);
              steps_1px = places where it jumps by >= 1 px (a rasterised straight
              line never does), steps_2px = jumps >= 2 px
  causes      (a) grid: share of the residual's power at the SegFormer grid period
                  (128 cells over the image) projected on the edge -- linear
                  logit upsampling leaves a wave at that period; reported with
                  the dominant period found
              (b) noise: residual RMS left after removing that wave
              (c) hard composite: share of the edge's inside pixels whose output
                  is exactly the tile render and outside pixels exactly the photo
                  (no blended pixel = a hard 0/1 mask)
              (d) objects: share of steps within 3 px of an object
  photo edge  DeepLSD segments within 2 deg and max(3 px, 0.2 % diag) of the line
              covering >= 50 % of the edge's extent = supported
  verdict     straight+supported / straight, not supported / curved / irregular
Writes <out_dir>/<room>__edges.png (boundary + fitted lines), crops for wide_angle,
and <out_dir>/edge_jaggedness.json.
"""

from __future__ import annotations

import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "backend"), str(ROOT / "tests/regression")]

import perspective_engine as pe  # noqa: E402
import reuse  # noqa: E402
from baseline_rooms import ROOMS, TILE, TILE_W_MM, TILE_H_MM, GROUT_MM  # noqa: E402
from engine import TileSpec  # noqa: E402
from perspective_engine import masks  # noqa: E402
from wall_direction_diagnose import deeplsd  # noqa: E402

GRID = 128
MIN_LEN, RANSAC, SUPPORT_DEG, SUPPORT_OFF, SUPPORT_SHARE = 0.015, 0.003, 2.0, 0.002, 0.5
CURVED_SAG, MIN_INLIER_SHARE = 0.005, 0.6
EDGE_COLORS = {"ceiling": (255, 60, 60), "floor": (60, 140, 255), "corner": (255, 200, 0), "object": (200, 80, 255)}


def fit(pts, thr, rng):
    best = None
    for _ in range(400):
        i, j = rng.choice(len(pts), 2, replace=False)
        d = pts[j] - pts[i]
        if np.hypot(*d) < 1e-6:
            continue
        n = np.array([-d[1], d[0]]) / np.hypot(*d)
        inl = np.abs((pts - pts[i]) @ n) < thr
        if best is None or inl.sum() > best.sum():
            best = inl
    c = pts[best].mean(axis=0)
    nrm = np.linalg.svd(pts[best] - c)[2][1]
    return nrm, c, best


def main() -> int:
    out = Path(sys.argv[1])
    names = sys.argv[2:] or list(ROOMS)
    out.mkdir(parents=True, exist_ok=True)
    spec = TileSpec(artwork=np.asarray(Image.open(TILE).convert("RGB")), width_mm=TILE_W_MM,
                    height_mm=TILE_H_MM, rotation_deg=0, grout_mm=GROUT_MM)
    report = {}
    for name in names:
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as tmp:
            job = Path(tmp) / ROOMS[name].name
            shutil.copytree(ROOMS[name], job)
            b = reuse.load(job)
            fl, wl = pe.load(job / "segments")
            geo = pe.ensure_geometry(job / "segments", b.room, fl, wl, clean=b.clean)
            regions, raws, comp = {}, {}, b.room.copy()
            for wd in geo["walls"]:
                idx = int(wd["index"])
                try:
                    r = pe.render_tiled_room(b.room, fl, wl, spec, "wall", (None, None, None), clean=b.clean,
                                             props=b.props, wall_index=idx, geometry=geo)
                except Exception:  # noqa: BLE001
                    continue
                m = r.regions.get(f"wall-{idx}")
                if m is not None:
                    regions[idx], raws[idx] = m, r.result.raw
                    comp[m] = r.result.composite[m]
        h, w = b.room.shape[:2]
        diag = float(np.hypot(h, w))
        floor_m, wall_m, _ = masks.prepare(fl, wl, (h, w))
        props = np.asarray(b.props, bool)
        ceiling = ~wall_m & ~floor_m & ~props
        owner = np.full((h, w), -1, int)
        for idx, m in regions.items():
            owner[m] = idx
        segs = deeplsd(b.room)
        segs = segs[np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]) >= MIN_LEN * diag]
        near_obj = cv2.dilate(props.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
        img = (b.room.astype(np.float32) * 0.6).astype(np.uint8)
        edges_out = {}
        for idx, region in sorted(regions.items()):
            inner = region & ~cv2.erode(region.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            ys, xs = np.nonzero(inner)
            cls = np.full(len(ys), "", object)
            outside_px = {}
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                yy, xx = np.clip(ys + dy, 0, h - 1), np.clip(xs + dx, 0, w - 1)
                out_of_region = ~region[yy, xx]
                for label, test in (("object", props[yy, xx]), ("corner", (owner[yy, xx] >= 0) & (owner[yy, xx] != idx)),
                                    ("floor", floor_m[yy, xx]), ("ceiling", ceiling[yy, xx])):
                    sel = out_of_region & test & (cls == "")
                    cls[sel] = label
                    for k in np.nonzero(sel)[0]:
                        outside_px.setdefault(label, []).append((yy[k], xx[k]))
            for (y, x), c in zip(zip(ys, xs), cls):
                if c:
                    img[y, x] = EDGE_COLORS[c]
            for label in ("ceiling", "floor", "corner"):
                sel = cls == label
                if sel.sum() < MIN_LEN * diag:
                    continue
                pts = np.stack([xs[sel], ys[sel]], 1).astype(float)
                if np.ptp(pts, axis=0).max() < MIN_LEN * diag:
                    continue
                nrm, c, inl = fit(pts, max(2.0, RANSAC * diag), rng)
                tang = np.array([nrm[1], -nrm[0]])
                d = (pts - c) @ nrm
                u = (pts - c) @ tang
                rms_all = float(np.sqrt((d ** 2).mean()))
                rms_in = float(np.sqrt((d[inl] ** 2).mean()))
                coef = np.polyfit(u[inl], d[inl], 2)
                uu = np.linspace(u[inl].min(), u[inl].max(), 50)
                sag = float(np.abs(np.polyval(coef, uu) - np.polyval(np.polyfit(u[inl], d[inl], 1), uu)).max())
                # profile d(u) on the inliers, 1 px bins
                ub = np.round(u[inl]).astype(int)
                order = np.unique(ub)
                prof = np.array([d[inl][ub == k].mean() for k in order])
                jumps = np.abs(np.diff(prof))
                gaps = np.diff(order) == 1
                steps1 = int(((jumps >= 1.0) & gaps).sum())
                steps2 = int(((jumps >= 2.0) & gaps).sum())
                step_u = order[1:][(jumps >= 1.0) & gaps]
                step_pts = c[None] + step_u[:, None] * tang[None] + 0 * nrm[None]
                sy = np.clip(np.round(step_pts[:, 1]).astype(int), 0, h - 1)
                sx = np.clip(np.round(step_pts[:, 0]).astype(int), 0, w - 1)
                obj_share = float(near_obj[sy, sx].mean()) if len(step_u) else 0.0
                # (a) grid wave: power at the SegFormer cell length along this edge
                cell = 1.0 / math.hypot(tang[0] / (w / GRID), tang[1] / (h / GRID))
                grid_share, dominant = None, None
                full = np.arange(order.min(), order.max() + 1)
                if len(full) >= 4 * cell and len(prof) > 8:
                    pr = np.interp(full, order, prof)
                    pr = pr - np.polyval(np.polyfit(full, pr, 1), full)
                    spec_p = np.abs(np.fft.rfft(pr)) ** 2
                    freqs = np.fft.rfftfreq(len(pr))
                    band = (freqs > 0)
                    if band.sum():
                        target = 1.0 / cell
                        near = band & (np.abs(freqs - target) <= 0.25 * target)
                        grid_share = round(float(spec_p[near].sum() / spec_p[band].sum()), 3)
                        dominant = round(float(1.0 / freqs[band][np.argmax(spec_p[band])]), 1)
                    # (b) what is left after removing the grid-period band
                    F = np.fft.rfft(pr)
                    F[near] = 0
                    noise = float(np.sqrt((np.fft.irfft(F, len(pr)) ** 2).mean()))
                else:
                    noise = None
                # (c) hard composite: inside pixels == tile render, outside pixels == photo
                in_y, in_x = ys[sel], xs[sel]
                hard_in = np.all(comp[in_y, in_x] == raws[idx][in_y, in_x], axis=1) | props[in_y, in_x]
                oy, ox = np.array(outside_px[label]).T
                hard_out = np.all(comp[oy, ox] == b.room[oy, ox], axis=1)
                # photo support
                ext = (u[inl].min(), u[inl].max())
                covered = np.zeros(int(ext[1] - ext[0]) + 1, bool)
                for s in segs:
                    p1, p2 = s[:2], s[2:]
                    dv = p2 - p1
                    ang = math.degrees(math.acos(min(1.0, abs(float(dv @ tang)) / (np.hypot(*dv) + 1e-9))))
                    off = max(abs(float((p1 - c) @ nrm)), abs(float((p2 - c) @ nrm)))
                    if ang <= SUPPORT_DEG and off <= max(3.0, SUPPORT_OFF * diag):
                        a, bb = sorted([float((p1 - c) @ tang), float((p2 - c) @ tang)])
                        a, bb = max(a, ext[0]), min(bb, ext[1])
                        if bb > a:
                            covered[int(a - ext[0]):int(bb - ext[0]) + 1] = True
                support = float(covered.mean())
                if inl.mean() < MIN_INLIER_SHARE:
                    verdict = "irregular (no single line)"
                elif sag > CURVED_SAG * diag:
                    verdict = "curved"
                elif support >= SUPPORT_SHARE:
                    verdict = "straight, supported by a photo edge"
                else:
                    verdict = "straight fit, NOT supported by a photo edge"
                causes = []
                if grid_share is not None and grid_share >= 0.25:
                    causes.append("(a) SegFormer grid wave")
                if rms_in >= 0.75:
                    causes.append("(b) noisy boundary")
                if hard_in.mean() > 0.99 and hard_out.mean() > 0.99:
                    causes.append("(c) hard 0/1 composite (no anti-aliasing)")
                if obj_share >= 0.3:
                    causes.append("(d) object edges")
                edges_out[f"wall-{idx} {label}"] = {
                    "pixels": int(sel.sum()), "length_px": round(float(ext[1] - ext[0]), 1),
                    "angle_deg": round(math.degrees(math.atan2(tang[1], tang[0])), 2),
                    "inlier_share": round(float(inl.mean()), 3),
                    "rms_px_all": round(rms_all, 2), "rms_px_inliers": round(rms_in, 2),
                    "max_dev_px_inliers": round(float(np.abs(d[inl]).max()), 2),
                    "sagitta_px": round(sag, 2),
                    "steps_1px": steps1, "steps_2px": steps2,
                    "steps_per_100px": round(100.0 * steps1 / max(len(order), 1), 1),
                    "a_grid_cell_px": round(cell, 1), "a_grid_power_share": grid_share,
                    "a_dominant_period_px": dominant,
                    "b_noise_rms_px": round(noise, 2) if noise is not None else None,
                    "c_inside_equals_tile": round(float(hard_in.mean()), 3),
                    "c_outside_equals_photo": round(float(hard_out.mean()), 3),
                    "d_steps_near_object": round(obj_share, 3),
                    "photo_edge_support": round(support, 3),
                    "verdict": verdict, "causes": causes,
                    "line": [[round(v, 4) for v in nrm], [round(v, 1) for v in c]],
                }
                if verdict.startswith("straight"):
                    p0, p1 = c + (ext[0] - 20) * tang, c + (ext[1] + 20) * tang
                    col = (255, 255, 255) if "supported" in verdict and "NOT" not in verdict else (120, 255, 120)
                    cv2.line(img, tuple(int(v) for v in p0), tuple(int(v) for v in p1), col, 1, cv2.LINE_AA)
        report[name] = {"size": [w, h], "edges": edges_out}
        Image.fromarray(img).save(out / f"{name}__edges.png")
        if name == "wide_angle":
            for tag, (x0, y0, x1, y1) in {"right_wall_top": (950, 30, 1160, 345),
                                          "right_wall_bottom": (975, 1030, 1160, 1450)}.items():
                both = np.hstack([img[y0:y1, x0:x1], comp[y0:y1, x0:x1]])
                Image.fromarray(both).resize((both.shape[1] * 3, both.shape[0] * 3), Image.NEAREST).save(
                    out / f"{name}__crop_{tag}__overlay_render_x3.png")
        print(name, "done", flush=True)
    (out / "edge_jaggedness.json").write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
