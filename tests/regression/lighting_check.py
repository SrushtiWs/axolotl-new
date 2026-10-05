"""
Tile lighting before / after (engine.LIGHTING_MODE "per-pixel" vs "lowfreq"), one room.

    backend/.venv/bin/python tests/regression/lighting_check.py <name> <segment_job> <tile_image> <tile_w_mm> <tile_h_mm> <out_dir> [y_line ...]

Renders the floor and every wall alone (as the Studio does), composes them onto
the photo region by region (as /compose does), for both lighting modes.
Reports per mode:
  ratio          per surface: mean luminance render / original (object-free pixels)
  blown          share of tiled pixels above 250
  halo ring      pixels brighter by >= 30 levels in the 1-3 px ring around opaque objects
  lines          for each y_line: mean |vertical gradient| on that floor row vs the
                 rows 12-20 px away (1.0 = no line there)
  outside        pixels changed outside the tiled regions between the modes (must be 0)
Images: <out_dir>/<name>__{per-pixel,lowfreq}.png and crops.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "backend")]

import perspective_engine as pe  # noqa: E402
import reuse  # noqa: E402
from engine import TileSpec  # noqa: E402
from tiles_backend.perspective_engine import engine  # noqa: E402

CROPS = {"left_glass_edge": (150, 0, 330, 420), "slat_wall": (340, 40, 720, 260), "round_artwork": (1050, 250, 1240, 470),
         "rug_edge": (380, 500, 900, 620), "floor_top_edge": (250, 470, 1100, 560)}


def lum(a):
    return a.astype(float).mean(axis=2)


def render(job: Path, spec, mode: str):
    engine.LIGHTING_MODE = mode
    b = reuse.load(job)
    fl, wl = pe.load(job / "segments")
    geo = pe.ensure_geometry(job / "segments", b.room, fl, wl, clean=b.clean)
    out = b.room.copy()
    regions, refused = {}, {}
    for surface, idx in [("floor", None)] + [("wall", int(w["index"])) for w in geo["walls"]]:
        try:
            r = pe.render_tiled_room(b.room, fl, wl, spec, surface, (None, None, None), clean=b.clean,
                                     props=b.props, wall_index=idx, geometry=geo)
        except Exception as e:  # noqa: BLE001
            refused[f"{surface}{'' if idx is None else f'-{idx}'}"] = str(e)[:80]
            continue
        for key, m in r.regions.items():
            out[m] = r.result.composite[m]
            regions[key] = m
    engine.LIGHTING_MODE = "lowfreq"
    alpha = None
    return b, out, regions, refused


def main() -> int:
    name, job, tile_path = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
    tw, th, out_dir = float(sys.argv[4]), float(sys.argv[5]), Path(sys.argv[6])
    y_lines = [int(v) for v in sys.argv[7:]]
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = TileSpec(artwork=np.asarray(Image.open(tile_path).convert("RGB")), width_mm=tw, height_mm=th,
                    rotation_deg=0, grout_mm=5)
    res, imgs = {}, {}
    for mode in ("per-pixel", "lowfreq"):
        b, comp, regions, refused = render(job, spec, mode)
        photo = b.room
        tiled = np.zeros(photo.shape[:2], bool)
        for m in regions.values():
            tiled |= m
        props = np.asarray(b.props, bool) if b.props is not None else np.zeros_like(tiled)
        ratios = {}
        for key, m in sorted(regions.items()):
            mm = m & ~props
            if mm.sum() > 500:
                ratios[key] = round(float(lum(comp)[mm].mean() / max(lum(photo)[mm].mean(), 1)), 2)
        full = props
        ring = cv2.dilate(full.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool) & ~full & tiled
        lines = {}
        floor = regions.get("floor")
        if floor is not None:
            g = np.abs(np.diff(lum(comp), axis=0))
            for y in y_lines:
                def row_strength(yy):
                    cols = floor[yy, :] & floor[yy + 1, :]
                    return float(g[yy, cols].mean()) if cols.sum() > 50 else None
                at = max((row_strength(yy) or 0) for yy in range(y - 3, y + 4))
                near = [row_strength(yy) for yy in list(range(y - 20, y - 12)) + list(range(y + 12, y + 20))]
                near = [v for v in near if v]
                lines[y] = round(at / max(np.mean(near), 1e-3), 2) if near else None
        res[mode] = {"ratio_render_vs_original": ratios,
                     "blown_share_pct": round(100.0 * float(((comp.max(axis=2) > 250) & tiled).sum()) / max(tiled.sum(), 1), 2),
                     "blown_in_photo_pct": round(100.0 * float(((photo.max(axis=2) > 250) & tiled).sum()) / max(tiled.sum(), 1), 2),
                     "halo_ring_brighter_30": int(((lum(comp) - lum(photo)) >= 30)[ring].sum()), "ring_px": int(ring.sum()),
                     "floor_line_strength": lines, "refused": refused}
        imgs[mode] = (comp, tiled, photo)
        Image.fromarray(comp).save(out_dir / f"{name}__{mode}.png")
    a, b_ = imgs["per-pixel"][0].astype(int), imgs["lowfreq"][0].astype(int)
    tiled = imgs["per-pixel"][1] | imgs["lowfreq"][1]
    res["changed_outside_tiled_between_modes"] = int(((np.abs(a - b_).max(axis=2) > 0) & ~tiled).sum())
    photo = imgs["lowfreq"][2]
    for n, (x0, y0, x1, y1) in CROPS.items():
        h, w = photo.shape[:2]
        if x1 <= w and y1 <= h:
            strip = np.hstack([photo[y0:y1, x0:x1], imgs["per-pixel"][0][y0:y1, x0:x1], imgs["lowfreq"][0][y0:y1, x0:x1]])
            Image.fromarray(strip).save(out_dir / f"{name}__crop_{n}__photo_before_after.png")
    (out_dir / f"{name}__lighting.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
