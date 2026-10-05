"""
Camera step 1: one camera for floor and walls -- before / after, per room.

    backend/.venv/bin/python tests/regression/camera_check.py [name=job_dir ...]

Each room's geometry is recomputed in a scratch copy twice (stored rooms are
not touched), one room at a time:
  before  focal_estimate.USE_LONG_SIDE = False, engine.JOINT_CAMERA = False
          (the camera as it was: width-based prior, image-centre principal point)
  after   both True (long-side prior, joint camera)

Per room: focal_source, focal_px, hfov, horizon row, principal point,
confidence, the joint camera's decision. Per wall (rendered alone with a
checker, as wall_direction.py does): median angle between the rendered
along-wall seams and the wall's own vanishing point (fitted from its own
lines). Floor: each rendered floor seam is matched to the wall VP it points
at best; per wall the median miss, and the share pointing at no wall (> 10 deg).
Renders (all walls + floor, checker): tests/.work/camera_step1/<room>__{before,after}.png
Numbers: tests/.work/camera_step1/report.json
"""

from __future__ import annotations

import gc
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
import wall_direction as wd  # noqa: E402
from perspective_engine import masks as pe_masks  # noqa: E402
from reuse import load as reuse_load  # noqa: E402
from tiles_backend.perspective_engine import TileRequest, engine, render_room  # noqa: E402
from tiles_backend.perspective_engine.camera import focal_estimate  # noqa: E402
from tiles_backend.perspective_engine.camera.vp_from_mask import detect_lines_lsd  # noqa: E402
from tiles_backend.perspective_engine.surface.wall import junction_layout as jl  # noqa: E402

WORK = ROOT / "tests" / ".work" / "camera_step1"
# Room size for the test renders only: scale does not change seam DIRECTIONS.
REQ = TileRequest(tile_width_mm=600, tile_height_mm=600, grout_mm=0,
                  room_width_mm=3658, room_length_mm=3658, room_height_mm=3048)


def rooms() -> dict:
    out = {}
    for base in (ROOT / "tests" / "fixtures" / "rooms", ROOT / "tests" / "validation" / "rooms"):
        for p in sorted(base.iterdir()):
            if (p / "fixture.json").is_file():
                out[p.name] = (p, next(p.glob("photo.*")))
    for p in sorted((ROOT / "room-data" / "rooms").glob("room_*")):
        if (p / "job" / "reuse").is_dir():
            out[p.name] = (p / "job", next(p.glob("original.*")))
    for job in ("97ed8cbb4108", "6a6b7dfd26b6", "8ae65553d5c9", "9deeb8d81154"):
        j = ROOT / "backend" / "jobs" / job
        if (j / "reuse").is_dir():
            out["job_" + job] = (j, j / "original.png")
    return out


def recompute(job: Path, photo: Path, target: Path, after: bool):
    seg = target / "segments"
    shutil.rmtree(target, ignore_errors=True)
    shutil.copytree(job / "segments", seg)
    for stored in ("floor", "wall", "room"):
        shutil.rmtree(seg / stored, ignore_errors=True)
    focal_estimate.USE_LONG_SIDE = after
    engine.JOINT_CAMERA = after
    bundle = reuse_load(job)
    fm, wm = pe.load(seg)
    # The uploaded bytes when the room keeps them (EXIF); a job's original.png is a re-encode without EXIF.
    photo_bytes = photo.read_bytes()
    geo = pe.ensure_geometry(seg, bundle.room, fm, wm, clean=bundle.clean, photo_bytes=photo_bytes)
    focal_estimate.USE_LONG_SIDE = True
    engine.JOINT_CAMERA = True
    return bundle, fm, wm, geo


def floor_vs_walls(rendered, floor_tiled, walls, h) -> dict:
    """
    Floor seams vs the walls' own vanishing points (fitted from each wall's own
    lines, so they come from the photo, not from the floor camera). Each
    rendered floor seam is given to the wall VP it points at best; per wall the
    median miss in degrees, plus the share of seams that point at no wall (> 10 deg).
    """
    gray = cv2.cvtColor(rendered, cv2.COLOR_RGB2GRAY)
    inner = cv2.erode(floor_tiled.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    vps = {w["id"]: np.array(w["direction"]["vanishing_point"]["homogeneous"], float)
           for w in walls if (w.get("direction") or {}).get("vanishing_point")}
    if not vps:
        return {"no wall VPs": True}
    segs = []
    for x1, y1, x2, y2 in np.asarray(detect_lines_lsd(gray)).reshape(-1, 4):
        if math.hypot(x2 - x1, y2 - y1) < 20:
            continue
        mx, my = int((x1 + x2) / 2), int((y1 + y2) / 2)
        if inner[min(my, h - 1), min(mx, inner.shape[1] - 1)]:
            segs.append((x1, y1, x2, y2))
    if len(segs) < 5:
        return {"lines": len(segs)}
    res = np.stack([wd._residuals(segs, v) for v in vps.values()])          # walls x lines
    best = res.argmin(axis=0)
    miss = res.min(axis=0)
    out = {"lines": len(segs), "pointing_at_no_wall_share": round(float((miss > 10).mean()), 3)}
    for k, wid in enumerate(vps):
        mine = miss[(best == k) & (miss <= 10)]
        if len(mine) >= 5:
            out[wid] = {"median_deg": round(float(np.median(mine)), 2), "lines": int(len(mine))}
    return out


def measure(name, job, photo, after: bool) -> dict:
    label = "after" if after else "before"
    bundle, fm, wm, geo = recompute(job, photo, WORK / "scratch" / name / label, after)
    floor, wall, _ = pe_masks.prepare(fm, wm, bundle.clean.shape[:2])
    h = floor.shape[0]
    cam = geo["camera"]
    joint = cam.get("joint") or {}
    res = {"focal_source": cam.get("focal_source"), "focal_px": round(cam["focal_px"], 1),
           "hfov_deg": round(focal_estimate.focal_to_hfov_degrees(cam["focal_px"], floor.shape[1]), 1),
           "principal_point": [round(v, 1) for v in cam["principal_point"]],
           "horizon_y": joint.get("horizon_y", (geo["floor"].get("vanishing_points") or {}).get("horizon_y")),
           "confidence": cam.get("confidence"), "decision": joint.get("decision"),
           "floor_status": geo["floor"].get("status"), "walls": {}}
    tile = wd._checker()
    canvas = bundle.clean.copy()
    for w in geo["walls"]:
        d = w.get("direction")
        if not d:
            continue
        try:
            rw = render_room(bundle.clean, floor, wall, tile, "wall", REQ, wall_index=w["index"], geometry=geo)
        except Exception as e:  # noqa: BLE001
            res["walls"][w["id"]] = {"refused": str(e)[:80]}
            continue
        canvas[rw.wall_tiled] = rw.image[rw.wall_tiled]
        segs = wd._render_lines(rw.image, rw.wall_tiled)
        if len(segs) < 5:
            res["walls"][w["id"]] = {"lines": len(segs)}
            continue
        vp = np.array(d["vanishing_point"]["homogeneous"], float)
        r = wd._residuals(segs, vp)
        res["walls"][w["id"]] = {"seam_median_deg": round(float(np.median(r)), 2), "lines": len(segs),
                                 "faces": d.get("faces")}
    try:
        rf = render_room(bundle.clean, floor, wall, tile, "floor", REQ, geometry=geo)
        canvas[rf.floor_tiled] = rf.image[rf.floor_tiled]
        res["floor_vs_wall_base"] = floor_vs_walls(rf.image, rf.floor_tiled, geo["walls"], h)
    except Exception as e:  # noqa: BLE001
        res["floor_vs_wall_base"] = {"refused": str(e)[:80]}
    cv2.imwrite(str(WORK / f"{name}__{label}.png"), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    del bundle, canvas
    gc.collect()
    return res


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    chosen = rooms()
    if len(sys.argv) > 1:
        chosen = {k: v for k, v in chosen.items() if k in sys.argv[1:]}
    report = {}
    path = WORK / "report.json"
    for name, (job, photo) in chosen.items():
        report[name] = {"before": measure(name, job, photo, False), "after": measure(name, job, photo, True)}
        path.write_text(json.dumps(report, indent=1, default=str))       # after every room: resumable reading
        b, a = report[name]["before"], report[name]["after"]
        print(f"== {name}: focal {b['focal_source']} {b['focal_px']} ({b['hfov_deg']} deg) -> {a['focal_source']} "
              f"{a['focal_px']} ({a['hfov_deg']} deg) | principal {b['principal_point']} -> {a['principal_point']} | "
              f"horizon {a['horizon_y']} | confidence {a['confidence']}", flush=True)
        for wid in sorted(set(b["walls"]) | set(a["walls"])):
            print(f"   {wid}: seam error {b['walls'].get(wid, {}).get('seam_median_deg')} -> "
                  f"{a['walls'].get(wid, {}).get('seam_median_deg')} deg", flush=True)
        fb, fa = b.get("floor_vs_wall_base") or {}, a.get("floor_vs_wall_base") or {}
        for wid in sorted(k for k in set(fb) | set(fa) if k.startswith("wall-")):
            print(f"   floor seams toward {wid}'s VP: {(fb.get(wid) or {}).get('median_deg')} -> "
                  f"{(fa.get(wid) or {}).get('median_deg')} deg", flush=True)
        print(f"   floor seams pointing at no wall: {fb.get('pointing_at_no_wall_share')} -> "
              f"{fa.get('pointing_at_no_wall_share')}", flush=True)
        for line in a.get("decision") or []:
            print(f"   camera: {line}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
