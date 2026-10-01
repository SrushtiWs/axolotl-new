"""
Depth VPs judged on the room's long structural lines (fix Step 4).

    backend/.venv/bin/python tests/regression/weak_texture.py

Per frozen room, geometry is recomputed as a new Clean Room would (scratch
copies; fixtures untouched). Printed per VP (the floor detector's, and the
long-line pool's own): lines in the pool, inliers, support, angular spread,
inlier ratio, confident or not -- and the decision taken.

Rules checked:
  * a depth VP the room then USES is confident (inliers, support, spread);
  * a VP that is not confident is not used: the floor is rejected instead;
  * short / plumb / image-horizontal / object lines never enter the pool.
"""

from __future__ import annotations

import math
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "backend")]

import perspective_engine as pe  # noqa: E402
from reuse import load as reuse_load  # noqa: E402
from tiles_backend.perspective_engine.camera import boundary_vp as bv  # noqa: E402

WORK = ROOT / "tests" / ".work" / "weak_texture"
ROOMS = [p for base in (ROOT / "tests" / "fixtures" / "rooms", ROOT / "tests" / "validation" / "rooms")
         for p in sorted(base.iterdir()) if (p / "fixture.json").is_file()]


def _fmt(s: dict) -> str:
    if not s or s.get("point") is None:
        return "none"
    return (f"({s['point'][0]:.0f},{s['point'][1]:.0f}) inliers {s['inliers']}/{s['lines']} support {s['support']} "
            f"spread {s['spread_deg']} deg -> {'CONFIDENT' if s['confident'] else 'low'}")


def main() -> int:
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    failed = 0
    for room in ROOMS:
        seg = WORK / room.name / "segments"
        shutil.copytree(room / "segments", seg)
        for stored in ("floor", "wall", "room"):
            shutil.rmtree(seg / stored, ignore_errors=True)
        bundle = reuse_load(room)
        photo = next(room.glob("photo.*")).read_bytes()
        floor_mask, wall_mask = pe.load(seg)
        geo = pe.ensure_geometry(seg, bundle.room, floor_mask, wall_mask, clean=bundle.clean, photo_bytes=photo)
        rep = geo.get("vp_report") or {}
        fl = geo["floor"]
        used = fl.get("status") == "detected"
        used_vp = (fl.get("vanishing_points") or {}).get("depth_vp") if used else None

        problems = []
        if used:
            decision = rep.get("decision", "")
            chosen = rep["floor_vp"] if decision.startswith("floor VP kept") else rep.get("pool_vp")
            gate_kept = "passed the floor detector's gate" in decision and rep["floor_vp"].get("validated") is False
            if not chosen or not (chosen.get("confident") or gate_kept):
                problems.append("a depth VP is used that is neither confident nor above the detector's gate (flagged)")
        elif rep.get("floor_vp", {}).get("point") is not None and "rejected" not in rep.get("decision", ""):
            problems.append("floor has no VP but the decision does not say it was rejected")

        ok = not problems
        failed += not ok
        frame = (geo.get("room_frame") or {}).get("status")
        print(f"{'PASS' if ok else 'FAIL'}  {room.name:<8} pool {rep.get('pool_lines')} long lines")
        print(f"        floor detector VP ({rep.get('floor_vp', {}).get('origin')}, detector conf "
              f"{rep.get('floor_vp', {}).get('detector_confidence')}): {_fmt(rep.get('floor_vp'))}")
        print(f"        long-line pool VP: {_fmt(rep.get('pool_vp'))}")
        print(f"        decision: {rep.get('decision')} | floor now {fl.get('status')}"
              f"{'' if used_vp is None else f', depth VP ({used_vp[0]:.0f},{used_vp[1]:.0f}), horizon y={used_vp[1]:.0f}'}"
              f" | room frame {frame}")
        for p in problems:
            print(f"        - {p}")
    print(f"\n{len(ROOMS) - failed}/{len(ROOMS)} passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
