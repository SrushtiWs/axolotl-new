"""
Object edge halo, before / after (render.OBJECT_EDGE "matte" vs "tight"), one room.

    backend/.venv/bin/python tests/regression/edge_check.py <name> <segment_job> <tile_image> <tile_w_mm> <tile_h_mm> <out_dir>

Both renders use the current lighting; only the object edge differs. Rings are
defined from the ORIGINAL matte (the same pixels before and after):
  ring           1-3 px outside the opaque object (alpha >= 0.98), inside a tiled area
  vs photo       ring pixels brighter than the photo by >= 30 levels (the asked metric;
                 it also counts the tile simply being lighter than the old surface)
  vs tile        ring pixels brighter by >= 30 than the tile 4-8 px further out
                 (the rim itself, independent of the tile's exposure)
  outside band   pixels changed between the two renders more than 6 px from any
                 object edge (must be 0)
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
from perspective_engine import render as pe_render  # noqa: E402

CROPS = {"left_glass_edge": (150, 0, 330, 420), "round_artwork": (1050, 250, 1240, 470),
         "rug_edge": (380, 500, 900, 620), "sofa_edge": (560, 380, 800, 520)}


def compose(job, spec, edge):
    pe_render.OBJECT_EDGE = edge
    b = reuse.load(job)
    fl, wl = pe.load(job / "segments")
    geo = pe.ensure_geometry(job / "segments", b.room, fl, wl, clean=b.clean)
    out = b.room.copy()
    tiled = np.zeros(out.shape[:2], bool)
    alpha = None
    for surface, idx in [("floor", None)] + [("wall", int(w["index"])) for w in geo["walls"]]:
        try:
            r = pe.render_tiled_room(b.room, fl, wl, spec, surface, (None, None, None), clean=b.clean,
                                     props=b.props, wall_index=idx, geometry=geo)
        except Exception:  # noqa: BLE001
            continue
        for m in r.regions.values():
            out[m] = r.result.composite[m]
            tiled |= m
    pe_render.OBJECT_EDGE = "tight"
    return b, out, tiled


def main() -> int:
    name, job, tile_path = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
    tw, th, out_dir = float(sys.argv[4]), float(sys.argv[5]), Path(sys.argv[6])
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = TileSpec(artwork=np.asarray(Image.open(tile_path).convert("RGB")), width_mm=tw, height_mm=th,
                    rotation_deg=0, grout_mm=5)
    b, before, tiled = compose(job, spec, "matte")
    _, after, _ = compose(job, spec, "tight")
    from live_scene import _props_alpha
    alpha = _props_alpha(np.asarray(b.props, bool), b.room) if b.props is not None else None
    alpha = np.zeros(tiled.shape, np.float32) if alpha is None else alpha
    photo = b.room.astype(float).mean(axis=2)
    opaque = alpha >= 0.98
    near = cv2.dilate(opaque.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    ring = near & ~opaque & tiled
    farther = cv2.dilate(opaque.astype(np.uint8), np.ones((17, 17), np.uint8)).astype(bool) & ~cv2.dilate(
        opaque.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) & tiled & (alpha == 0)
    obj = alpha > 0
    edge = cv2.dilate(obj.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~cv2.erode(
        obj.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    band = cv2.dilate(edge.astype(np.uint8), np.ones((13, 13), np.uint8)).astype(bool)
    res = {}
    for label, img in (("before (matte)", before), ("after (tight)", after)):
        lum = img.astype(float).mean(axis=2)
        local = cv2.blur(np.where(farther, lum, 0), (31, 31)) / np.maximum(cv2.blur(farther.astype(float), (31, 31)), 1e-3)
        has_local = cv2.blur(farther.astype(float), (31, 31)) > 1e-3
        res[label] = {"ring_px": int(ring.sum()),
                      "ring_brighter_than_photo_30": int(((lum - photo) >= 30)[ring].sum()),
                      "ring_brighter_than_nearby_tile_30": int((((lum - local) >= 30) & has_local)[ring].sum())}
    diff = np.abs(after.astype(int) - before.astype(int)).max(axis=2) > 0
    res["changed_px"] = int(diff.sum())
    res["changed_outside_6px_object_band"] = int((diff & ~band).sum())
    for n, (x0, y0, x1, y1) in CROPS.items():
        Image.fromarray(np.hstack([b.room[y0:y1, x0:x1], before[y0:y1, x0:x1], after[y0:y1, x0:x1]])).save(
            out_dir / f"{name}__crop_{n}__photo_before_after.png")
    Image.fromarray(after).save(out_dir / f"{name}__after_tight.png")
    (out_dir / f"{name}__edge.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
