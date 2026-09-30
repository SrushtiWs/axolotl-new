"""
Floor/wall seam gap and tile-scale ratio, per wall, on the frozen rooms.

    backend/.venv/bin/python tests/regression/seam_scale.py [--out results.json]

For every wall the floor/wall junction pixels (the wall's bottom edge on the
floor) are cast onto the rendered floor plane and the rendered wall plane:

  seam gap     median distance between the two 3D points / distance from the
               camera. 0 = the wall stands exactly on the floor.
  scale ratio  junction length measured on the wall plane / on the floor plane.
               1.00 = floor and wall tiles share one millimetre scale.

Rendered through the engine (render_room) as the backend does, one wall at a
time, in three modes: room size AUTO; a typed W x L x H; and the typed size
with the room geometry withheld (the path a room RoomGeometry cannot frame takes).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "backend")]

import perspective_engine as pe  # noqa: E402
from tiles_backend.perspective_engine import TileRequest, render_room  # noqa: E402
from tiles_backend.perspective_engine.surface.wall.junction_layout import junction_points  # noqa: E402

ROOMS = ROOT / "tests" / "fixtures" / "rooms"
#: The size Step 6 was specified with; typed, so the legacy wall path has its anchor.
TYPED_MM = {"width": 4000.0, "length": 5000.0, "height": 3000.0}
MIN_JUNCTION = 10


def _hit(plane, f, cx, cy, xs, ys):
    r = np.stack([(xs - cx) / f, (ys - cy) / f, np.ones_like(xs)], -1)
    n = np.asarray(plane[:3], np.float64)
    den = r @ n
    t = -plane[3] / np.where(np.abs(den) < 1e-12, 1e-12, den)
    return r * t[:, None], t > 0


def seam(fplane, ff, wplane, wf, J, cx, cy):
    if len(J) < MIN_JUNCTION:
        return None
    J = J[np.argsort(J[:, 0])]
    Pf, a = _hit(fplane, ff, cx, cy, J[:, 0], J[:, 1])
    Pw, b = _hit(wplane, wf, cx, cy, J[:, 0], J[:, 1] - 2)      # 2 px up: on the wall
    ok = a & b
    if ok.sum() < MIN_JUNCTION:
        return None
    Pf, Pw = Pf[ok], Pw[ok]

    def span(P):
        Q = P - P.mean(0)
        return float(np.ptp(Q @ np.linalg.svd(Q, full_matrices=False)[2][0]))

    return (float(np.median(np.linalg.norm(Pf - Pw, axis=1) / np.linalg.norm(Pf, axis=1))),
            span(Pw) / max(span(Pf), 1e-9), int(ok.sum()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, help="write the numbers as JSON")
    parser.add_argument("--modes", nargs="*", default=["auto", "manual", "no-geometry"])
    args = parser.parse_args()

    tile = cv2.cvtColor(cv2.imread(str(ROOT / "tests" / "fixtures" / "tile.png")), cv2.COLOR_BGR2RGB)
    results = {}
    for room in sorted(p for p in ROOMS.iterdir() if (p / "fixture.json").is_file()):
        seg = room / "segments"
        clean = cv2.imread(str(seg / "CLEAN_ROOM.png"))
        rgb = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)
        Fm, Wm = cv2.imread(str(seg / "FLOOR_MASK.png"), 0), cv2.imread(str(seg / "WALL_MASK.png"), 0)
        F, W = Fm > 127, Wm > 127
        geo = pe.ensure_geometry(seg, rgb, Fm, Wm, clean=rgb)
        h, w = F.shape
        cx, cy = w / 2.0, h / 2.0

        modes = {"auto": ("AUTO", {}, geo), "manual": ("MANUAL 4000x5000x3000", TYPED_MM, geo),
                 "no-geometry": ("MANUAL, no room geometry", TYPED_MM,
                                 {k: v for k, v in geo.items() if k != "room_frame"})}
        for mode, dims, g in (modes[m] for m in args.modes):
            req = TileRequest(tile_width_mm=1200, tile_height_mm=1800, grout_mm=5,
                              room_width_mm=dims.get("width"), room_length_mm=dims.get("length"),
                              room_height_mm=dims.get("height"))
            rf = render_room(rgb, F, W, tile, "floor", req, geometry=g)
            fi = rf.info["surfaces"]["floor"]
            fplane = [*fi["plane"][:3], fi["plane"][3] * fi["mm_per_unit"]]
            print(f"== {room.name} | {mode}: floor camera height {fplane[3]:.0f} mm")

            for wd in geo["walls"]:
                i = wd["index"]
                key = f"{room.name}|{mode}|wall-{i}"
                try:
                    rw = render_room(rgb, F, W, tile, "wall", req, wall_index=i, geometry=g)
                except Exception as ex:                     # noqa: BLE001 (reported, not hidden)
                    results[key] = {"error": str(ex)[:120]}
                    print(f"   wall-{i}: not rendered: {str(ex)[:100]}")
                    continue
                winfo = rw.info["surfaces"]["wall"]
                e = winfo["walls"][0]
                mpu = e["mm_per_unit"]
                wplane = [*e["plane"][:3], e["plane"][3] * mpu]
                J = junction_points(F, geo["_wall_masks"][i])
                s = seam(fplane, fi["focal_px"], wplane, winfo["focal_px"], J, cx, cy)
                results[key] = {
                    "plane_source": e.get("plane_source"), "scale_source": winfo.get("scale_source"),
                    "placement": e.get("placement"), "validated": e.get("validated"),
                    "wall_mm": [round(e["wall_width_mm"]), round(e["wall_height_mm"])],
                    "seam_gap": None if s is None else round(s[0], 4),
                    "scale_ratio": None if s is None else round(s[1], 3),
                    "junction_px": 0 if s is None else s[2],
                }
                r = results[key]
                print(f"   wall-{i}: plane {r['plane_source']:<28} scale {str(r['scale_source']):<16} "
                      f"{r['wall_mm'][0] / 1000:5.2f} x {r['wall_mm'][1] / 1000:4.2f} m | "
                      f"seam gap {'-' if s is None else f'{100 * s[0]:5.1f}%'}  "
                      f"scale ratio {'-' if s is None else f'{s[1]:.2f}'}"
                      + ("" if r["validated"] is None else f" | {'validated' if r['validated'] else 'NOT VALIDATED'}"))

    if args.out:
        args.out.write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
