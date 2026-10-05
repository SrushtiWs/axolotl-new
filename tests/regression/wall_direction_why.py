"""
Why walls fail wall_direction (read only): where do the rendered seams converge?

    backend/.venv/bin/python tests/regression/wall_direction_why.py

Per wall that renders, as wall_direction.py renders it: the rendered checker's
along-wall lines (the 60 longest) are fitted to ONE vanishing point
(boundary_vp.fit_vp, RANSAC),
and that point is compared with
  stored     the wall's stored VP (what wall_direction tests against)
  level row  the same direction on the principal-point row (cy) -- where a
             level camera puts every horizontal VP
plus the room's detected horizon row and cy.
"""

from __future__ import annotations

import math
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "backend"), str(ROOT / "tests" / "regression")]

import perspective_engine as pe  # noqa: E402
import wall_direction as wd  # noqa: E402
from perspective_engine import masks as pe_masks  # noqa: E402
from reuse import load as reuse_load  # noqa: E402
from tiles_backend.perspective_engine import TileRequest, render_room  # noqa: E402
from tiles_backend.perspective_engine.camera import boundary_vp  # noqa: E402

WORK = ROOT / "tests" / ".work" / "wall_direction_why"


def main() -> int:
    tile = wd._checker()
    for room in wd.ROOMS:
        seg = WORK / room.name / "segments"
        shutil.rmtree(seg.parent, ignore_errors=True)
        shutil.copytree(room / "segments", seg)
        for stored in ("floor", "wall", "room"):
            shutil.rmtree(seg / stored, ignore_errors=True)
        bundle = reuse_load(room)
        photo = next(room.glob("photo.*")).read_bytes()
        fm, wm = pe.load(seg)
        geo = pe.ensure_geometry(seg, bundle.room, fm, wm, clean=bundle.clean, photo_bytes=photo)
        floor, wall, _ = pe_masks.prepare(fm, wm, bundle.clean.shape[:2])
        cx, cy = geo["camera"]["principal_point"]
        check = (geo.get("wall_split") or {}).get("direction_check") or {}
        print(f"== {room.name}: horizon y={check.get('horizon_y')}  principal row cy={cy}  "
              f"gap={None if check.get('horizon_y') is None else round(check['horizon_y'] - cy, 1)} px")
        for w in geo["walls"]:
            d = w.get("direction")
            if not d:
                continue
            try:
                rw = render_room(bundle.clean, floor, wall, tile, "wall", TileRequest(tile_width_mm=600, tile_height_mm=600, grout_mm=0),
                                 wall_index=w["index"], geometry=geo)
            except Exception as e:  # noqa: BLE001
                print(f"   {w['id']}: render refused ({str(e)[:50]})")
                continue
            segs = wd._render_lines(rw.image, rw.wall_tiled)
            if len(segs) < 5:
                continue
            vp = np.array(d["vanishing_point"]["homogeneous"], float)
            res_stored = float(np.median(wd._residuals(segs, vp)))
            longest = sorted(segs, key=lambda s: -math.hypot(s[2] - s[0], s[3] - s[1]))[:60]
            fit = boundary_vp.fit_vp([list(map(float, s)) for s in longest])
            if fit is None:
                print(f"   {w['id']}: rendered lines fit no single VP")
                continue
            fh = np.array(fit["homogeneous"], float)
            res_fit = float(np.median(wd._residuals(segs, fh)))
            if abs(vp[2]) > 1e-9:
                sx, sy = vp[0] / vp[2], vp[1] / vp[2]
                level = np.array([sx, cy, 1.0])
                res_level = float(np.median(wd._residuals(segs, level)))
                stored_txt = f"stored VP ({sx:.0f},{sy:.0f})"
            else:
                res_level, stored_txt = None, "stored VP at infinity"
            fit_txt = "infinity" if fit["image"] is None else f"({fit['image'][0]:.0f},{fit['image'][1]:.0f})"
            print(f"   {w['id']} faces {d.get('faces'):<10} snapped {str(d.get('snapped_to_depth_vp', (check.get('walls') or [{}])[0].get('snapped_to_depth_vp'))):<5} | "
                  f"{stored_txt}: median {res_stored:.2f} deg | rendered seams converge at {fit_txt} "
                  f"(median {res_fit:.2f} deg, {fit['inliers']}/{fit['runs']} lines) | "
                  f"same x on row cy: median {'-' if res_level is None else f'{res_level:.2f}'} deg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
