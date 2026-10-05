"""
How much of the OLD surface shows through the rendered tiles (read only).

    backend/.venv/bin/python tests/regression/neutralize_audit.py <segment_job> <out_dir> [floor|wall]

Re-creates the inputs composite() gets for the surface (the clean room it
blends onto, the mask factor, the per-pixel light factor) exactly as the
render computes them, and reports:

  mask edge     width of the band where mask_factor is between 0.1 and 0.9,
                and its pixel count
  light_factor  min / median / p95 / max inside the surface (clean-room
                brightness / the surface's mean brightness, clipped 0.15..2.2)
  old texture   the light factor's fine detail (it divided by its own large
                Gaussian blur): pixels where it moves the tile more than 5% /
                10% are pixels where the old surface's pattern, veins or
                reflections are still visible in the tile
Debug image: <out_dir>/<surface>__old_surface_visible.png
  left   the clean room's surface (what the tile is multiplied by)
  right  heat map of |fine detail| (black = none, red/yellow = old surface showing),
         with partial-alpha pixels in cyan
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "backend")]

import perspective_engine as pe  # noqa: E402
import reuse  # noqa: E402
from perspective_engine import masks as pe_masks  # noqa: E402
from tiles_backend.perspective_engine.core.composite import average_masked_brightness, decode_mask  # noqa: E402
from tiles_backend.perspective_engine.engine import REFERENCE_FRONTEND_OPTIONS, _mask_bgr  # noqa: E402


def main() -> int:
    job, out = Path(sys.argv[1]), Path(sys.argv[2])
    surface = sys.argv[3] if len(sys.argv) > 3 else "floor"
    out.mkdir(parents=True, exist_ok=True)
    b = reuse.load(job)
    fm, wm = pe.load(job / "segments")
    floor, wall, _ = pe_masks.prepare(fm, wm, b.clean.shape[:2])
    m = floor if surface == "floor" else wall
    clean_bgr = cv2.cvtColor(b.clean, cv2.COLOR_RGB2BGR)
    mb = _mask_bgr(m)
    had_alpha = mb.ndim == 3 and mb.shape[2] == 4
    if not had_alpha:
        mb = np.dstack([mb, np.full(mb.shape[:2], 255, np.uint8)])
    surf, factor = decode_mask(mb, had_alpha, surface)
    ys, xs = np.nonzero(surf)
    avg = average_masked_brightness(clean_bgr, ys, xs)
    bright = clean_bgr.astype(np.float32).mean(axis=2)
    lf = np.clip(bright / (avg + 0.1), 0.15, 2.2)
    blend = REFERENCE_FRONTEND_OPTIONS.get("lighting_blend", 0.5)
    mult = (1 - blend) + blend * lf                      # what each tile pixel is multiplied by
    h, w = m.shape
    sigma = 0.03 * float(np.hypot(h, w))
    smooth = cv2.GaussianBlur(mult * surf, (0, 0), sigma) / np.maximum(cv2.GaussianBlur(surf.astype(np.float32), (0, 0), sigma), 1e-3)
    fine = np.abs(mult / np.maximum(smooth, 1e-3) - 1.0)
    partial = (factor > 0.1) & (factor < 0.9)
    band_cols = 0
    if partial.any():
        dist_in = cv2.distanceTransform(surf.astype(np.uint8), cv2.DIST_L2, 3)
        band_cols = float(np.percentile(dist_in[partial], 95))
    inside = surf
    lf_in = lf[inside]
    res = {
        "surface": surface, "pixels": int(inside.sum()), "lighting_blend": blend,
        "base": "clean room (CLEAN_ROOM.png, objects removed)",
        "mask_edge": {"partial_alpha_px": int(partial.sum()), "band_width_px_p95": round(band_cols, 1),
                      "mask_factor_values": sorted({round(float(v), 2) for v in np.unique(factor[inside])})[:6]},
        "light_factor": {k: round(float(v), 3) for k, v in (("min", lf_in.min()), ("median", np.median(lf_in)),
                                                           ("p95", np.percentile(lf_in, 95)), ("max", lf_in.max()))},
        "tile_multiplier": {k: round(float(v), 3) for k, v in (("min", mult[inside].min()), ("median", np.median(mult[inside])),
                                                              ("p95", np.percentile(mult[inside], 95)), ("max", mult[inside].max()))},
        "old_surface_visible": {
            "px_over_5pct": int((fine[inside] > 0.05).sum()), "share_over_5pct": round(float((fine[inside] > 0.05).mean()), 3),
            "px_over_10pct": int((fine[inside] > 0.10).sum()), "share_over_10pct": round(float((fine[inside] > 0.10).mean()), 3),
        },
    }
    left = clean_bgr.copy()
    left[~inside] = (left[~inside] * 0.3).astype(np.uint8)
    heat = cv2.applyColorMap(np.clip(fine / 0.3 * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_HOT)
    right = (clean_bgr * 0.25).astype(np.uint8)
    right[inside] = heat[inside]
    right[partial] = (255, 255, 0)
    cv2.imwrite(str(out / f"{surface}__old_surface_visible.png"), np.hstack([left, right]))
    (out / f"{surface}__neutralize_audit.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
