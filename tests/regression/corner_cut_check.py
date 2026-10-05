"""
Step 2 (one pixel = one plane): before / after, per room.

    backend/.venv/bin/python tests/regression/corner_cut_check.py

Geometry is recomputed twice in scratch copies (stored rooms are untouched):
corner_cut.ENABLED = False (the split as it was) and True. Each result is
measured with plane_measure (corner lines found from the whole wall mask,
independently of the split) plus:

  claimed>1   wall pixels in more than one wall piece (target 0)
  fit         per wall, the median angle between its own floor-junction /
              ceiling runs and the direction to its own vanishing point
              (degrees from its own plane; at infinity: its direction)
  dots        each dot inside its own wall, not on a removed object

Overlays: tests/.work/corner_cut/{before,after}/<room>__<wall>.png
Numbers:  tests/.work/corner_cut/report.json
"""

from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "backend"), str(ROOT / "tests" / "regression")]

import perspective_engine as pe  # noqa: E402
import plane_measure  # noqa: E402
from reuse import load as reuse_load  # noqa: E402
from tiles_backend.perspective_engine.surface.wall import corner_cut, edge_layout  # noqa: E402
from tiles_backend.perspective_engine.surface.wall import junction_layout as jl  # noqa: E402

WORK = ROOT / "tests" / ".work" / "corner_cut"


def rooms() -> dict:
    out = {}
    for base in (ROOT / "tests" / "fixtures" / "rooms", ROOT / "tests" / "validation" / "rooms"):
        for p in sorted(base.iterdir()):
            if (p / "fixture.json").is_file():
                out[p.name] = (p, p / "segments", next(p.glob("photo.*")))
    for p in sorted((ROOT / "room-data" / "rooms").glob("room_*")):
        if (p / "job" / "reuse").is_dir():
            out[p.name] = (p / "job", p / "job" / "segments", next(p.glob("original.*")))
    return out


def recompute(name, job, seg, photo, enabled: bool, where: Path) -> Path:
    target = where / name / "segments"
    shutil.rmtree(target.parent, ignore_errors=True)
    shutil.copytree(seg, target)
    for stored in ("floor", "wall", "room"):
        shutil.rmtree(target / stored, ignore_errors=True)
    corner_cut.ENABLED = enabled
    bundle = reuse_load(job)
    floor_mask, wall_mask = pe.load(target)
    pe.ensure_geometry(target, bundle.room, floor_mask, wall_mask, clean=bundle.clean,
                       photo_bytes=photo.read_bytes())
    return target


def _angle_to(seg, vp_h) -> float:
    (x1, y1), (x2, y2) = seg
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    if abs(vp_h[2]) < 1e-9:
        tx, ty = vp_h[0], vp_h[1]
    else:
        tx, ty = vp_h[0] / vp_h[2] - mx, vp_h[1] / vp_h[2] - my
    a = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180
    b = math.degrees(math.atan2(ty, tx)) % 180
    d = abs(a - b) % 180
    return min(d, 180 - d)


def extra(seg: Path) -> dict:
    """claimed>1, per-wall direction fit, dot checks."""
    geo = json.loads((seg / "wall" / "wall_geometry.json").read_text())
    floor = plane_measure._read(seg / "floor" / "floor_mask.png")
    wall = plane_measure._read(seg / "wall" / "wall_mask.png")
    h, w = wall.shape
    diag = math.hypot(h, w)
    objects = plane_measure._read(seg / "ALL_OBJECTS.png")
    if objects is not None and objects.shape != wall.shape:
        objects = cv2.resize(objects.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    count = np.zeros((h, w), np.int32)
    walls = {}
    for wd in geo["walls"]:
        piece = plane_measure._read(seg / "wall" / "walls" / f"{wd['id']}.png")
        count += piece
        d = wd.get("direction") or {}
        vp = (d.get("vanishing_point") or {}).get("homogeneous")
        fits = []
        if vp is not None:
            for pts in (jl.junction_points(floor, piece, objects), edge_layout.ceiling_points(piece, objects)):
                for r in corner_cut.runs(pts, w, diag):
                    p = r["pts"]
                    fits.append(_angle_to(((p[0, 0], p[0, 1]), (p[-1, 0], p[-1, 1])), vp))
        x, y = wd["select_point"]
        walls[wd["id"]] = {
            "fit_deg": round(float(np.median(fits)), 2) if fits else None,
            "fit_runs": len(fits),
            "dot_inside_own_wall": bool(piece[int(y), int(x)]),
            "dot_on_object": bool(objects[int(y), int(x)]) if objects is not None else None,
        }
    split = geo.get("wall_split") or {}
    return {"claimed_more_than_once_px": int((count > 1).sum()), "walls": walls,
            "corner_cut": split.get("corner_cut"), "split_method": split.get("method")}


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    report = []
    for name, (job, seg, photo) in rooms().items():
        row = {"room": name}
        for label, enabled in (("before", False), ("after", True)):
            target = recompute(name, job, seg, photo, enabled, WORK / "scratch" / label)
            plane_measure.WORK = WORK / label
            plane_measure.WORK.mkdir(parents=True, exist_ok=True)
            row[label] = {"measure": plane_measure.measure(name, target), **extra(target)}
        corner_cut.ENABLED = True
        report.append(row)
        b, a = row["before"], row["after"]
        cc = a["corner_cut"] or {}
        print(f"== {name}: walls {len(b['measure']['walls'])} -> {len(a['measure']['walls'])} "
              f"| spill {sum(x['spill_px'] for x in b['measure']['walls'])} -> "
              f"{sum(x['spill_px'] for x in a['measure']['walls'])} px "
              f"| claimed>1 {b['claimed_more_than_once_px']} -> {a['claimed_more_than_once_px']} "
              f"| confirmed corners {len(cc.get('corners', []))}, uncertain {len(cc.get('uncertain_corners', []))}, "
              f"two-direction slots {cc.get('slots_with_two_directions')} -> uncertain={cc.get('uncertain')}")
        for label in ("before", "after"):
            for wl in row[label]["measure"]["walls"]:
                ex = row[label]["walls"][wl["id"]]
                print(f"   {label:<6} {wl['id']}: {wl['pixels']:>7} px faces {str(wl['faces']):<10} "
                      f"spill {wl['spill_px']:>6} ({wl['spill_pct']}%) two-dir {str(wl['two_directions']):<5} "
                      f"fit {ex['fit_deg']} deg ({ex['fit_runs']} runs) dot in own wall {ex['dot_inside_own_wall']} "
                      f"on object {ex['dot_on_object']}")
    (WORK / "report.json").write_text(json.dumps(report, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
