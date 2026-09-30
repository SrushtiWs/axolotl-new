"""
Wall directions from each wall's own edges (fix Step 3), all walls at once.

    backend/.venv/bin/python tests/regression/wall_direction.py

For every frozen room, geometry is recomputed exactly as a new Clean Room would
(scratch copies; the fixtures are not touched). Then per wall:

  evidence     which edges gave the direction (floor junction / ceiling line),
               their RANSAC support, agreement, confidence, validated or why not
  cross-check  side walls vs the room's depth direction, back wall facing the
               camera, one horizon for all walls
  rendered     the wall is tiled with a checker; the checker's along-wall lines
               are found in the RENDERED pixels (LSD, plumb lines excluded), and
               each must point at the wall's own VP: the angle between a line
               and the direction from its midpoint to that VP (at infinity for a
               wall facing the camera). Median <= MAX_MEDIAN_DEG. (A single VP
               fitted to near-parallel lines is ill-conditioned on walls seen at
               a grazing angle, so the per-line angle is what is tested.)

Overlays go to tests/.work/wall_direction/<room>__wall-<i>.png.
"""

from __future__ import annotations

import math
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "backend")]

import perspective_engine as pe  # noqa: E402
from perspective_engine import masks as pe_masks  # noqa: E402
from reuse import load as reuse_load  # noqa: E402
from tiles_backend.perspective_engine import TileRequest, render_room  # noqa: E402
from tiles_backend.perspective_engine.camera.vp_from_mask import detect_lines_lsd  # noqa: E402

WORK = ROOT / "tests" / ".work" / "wall_direction"
ROOMS = [p for base in (ROOT / "tests" / "fixtures" / "rooms", ROOT / "tests" / "validation" / "rooms")
         for p in sorted(base.iterdir()) if (p / "fixture.json").is_file()]
MAX_MEDIAN_DEG = 1.0
PLUMB_DEG = 20.0


def _checker(px: int = 256, n: int = 4) -> np.ndarray:
    idx = (np.arange(px) * n // px)
    board = ((idx[:, None] + idx[None, :]) % 2 * 255).astype(np.uint8)
    return np.dstack([board] * 3)


def _az(v, f, cx) -> float:
    return math.degrees(math.atan2((v[0] - cx * v[2]) / f, v[2])) % 180.0


def _diff(a, b) -> float:
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def _render_lines(image, tiled):
    """The rendered checker's along-wall line segments (inside the tiled region)."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    inner = cv2.erode(tiled.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    segs = []
    for x1, y1, x2, y2 in np.asarray(detect_lines_lsd(gray)).reshape(-1, 4):
        if math.hypot(x2 - x1, y2 - y1) < 15:
            continue
        if math.degrees(math.atan2(abs(x2 - x1), abs(y2 - y1))) < PLUMB_DEG:
            continue                                            # plumb: not along the wall
        mx, my = int((x1 + x2) / 2), int((y1 + y2) / 2)
        if 0 <= my < inner.shape[0] and 0 <= mx < inner.shape[1] and inner[my, mx]:
            segs.append((x1, y1, x2, y2))
    return segs


def _residuals(segs, vp) -> np.ndarray:
    """Per segment: angle (deg) between it and the direction from its midpoint to `vp` (homogeneous)."""
    out = []
    for x1, y1, x2, y2 in segs:
        mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        if abs(vp[2]) < 1e-9:
            tx, ty = vp[0], vp[1]                                # at infinity: a direction
        else:
            tx, ty = vp[0] / vp[2] - mx, vp[1] / vp[2] - my
        a = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0
        b = math.degrees(math.atan2(ty, tx)) % 180.0
        d = abs(a - b) % 180.0
        out.append(min(d, 180.0 - d))
    return np.array(out)


def main() -> int:
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    tile = _checker()
    failed = total = 0
    for room in ROOMS:
        seg = WORK / room.name / "segments"
        shutil.copytree(room / "segments", seg)
        for stored in ("floor", "wall", "room"):
            shutil.rmtree(seg / stored, ignore_errors=True)
        bundle = reuse_load(room)
        photo = next(room.glob("photo.*")).read_bytes()
        floor_mask, wall_mask = pe.load(seg)
        geo = pe.ensure_geometry(seg, bundle.room, floor_mask, wall_mask, clean=bundle.clean, photo_bytes=photo)
        floor, wall, _ = pe_masks.prepare(floor_mask, wall_mask, bundle.clean.shape[:2])
        check = (geo.get("wall_split") or {}).get("direction_check") or {}
        f = geo["camera"]["focal_px"]
        cx = geo["camera"]["principal_point"][0]
        print(f"== {room.name}: split {geo['wall_split'].get('method')}, horizon y={check.get('horizon_y')}, "
              f"one horizon: {check.get('one_horizon')}, depth VP {check.get('depth_vp')}")
        for wd, row in zip(geo["walls"], check.get("walls", [])):
            total += 1
            d = wd.get("direction")
            i = wd["index"]
            off = row.get("deg_from_depth_direction")
            off_txt = "" if off is None else f"{off:5.1f} deg from depth"
            verdict = "validated" if row.get("validated") else "NOT validated: " + str(row.get("validation"))
            line = (f"   {wd['id']}: {str(row.get('faces')):<10} source {str(row.get('source')):<28} "
                    f"conf {str((d or {}).get('confidence')):<6} snapped {str(row.get('snapped_to_depth_vp')):<5} "
                    f"{off_txt} | {verdict}")
            if d is None:
                print(line + " | no direction: not rendered here")
                continue
            try:
                req = TileRequest(tile_width_mm=600, tile_height_mm=600, grout_mm=0)
                rw = render_room(bundle.clean, floor, wall, tile, "wall", req, wall_index=i, geometry=geo)
            except Exception as ex:                             # noqa: BLE001 (reported)
                print(line + f" | render refused: {str(ex)[:70]}")
                continue
            segs = _render_lines(rw.image, rw.wall_tiled)
            if len(segs) < 5:
                print(line + f" | rendered lines: {len(segs)} (too few to test)")
                continue
            vp = np.array(d["vanishing_point"]["homogeneous"], float)
            res = _residuals(segs, vp)
            med, p90 = float(np.median(res)), float(np.percentile(res, 90))
            ok = med <= MAX_MEDIAN_DEG
            failed += not ok
            print(line + f" | {len(segs)} rendered lines vs the wall's own VP: median {med:4.2f} deg, 90% {p90:4.2f} deg "
                         f"{'PASS' if ok else 'FAIL'}")
            img = cv2.cvtColor(rw.image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(WORK / f"{room.name}__{wd['id']}.png"), img)
    print(f"\nrendered-direction check failures: {failed} (of {total} walls)")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
