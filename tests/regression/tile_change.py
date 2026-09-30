"""
10d: a tile change must not re-run detection or the camera.

    backend/.venv/bin/python tests/regression/tile_change.py

For a floor and a wall of each frozen room: render with one tile, then with
other tile sizes. During every render after the first:

  * the memoised per-room steps (MiDaS depth, line detection, vanishing
    points, EXIF focal, the estimated camera) must all be cache hits, 0 misses;
  * the room-level detectors (segmentation, room geometry, room frame, VP_Y,
    wall instances) must not be called at all.

Then the cache must be exact: the last render is repeated with every cache
cleared, and the two result.png files must be byte-identical.
"""

from __future__ import annotations

import collections
import functools
import importlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness  # noqa: E402

TILES = [(1200, 1800), (600, 600), (800, 1600)]
DETECTORS = {
    "tiles_backend.perspective_engine.engine": ["detect_room_geometry", "detect_surfaces", "wall_instances"],
    "tiles_backend.perspective_engine.room.geometry": ["frame"],
    "tiles_backend.perspective_engine.camera.vertical_vp": ["detect"],
    "surfaces": ["instances"],
}
MEMOISED = [
    ("tiles_backend.perspective_engine.depth", "estimate_depth_heatmap"),
    ("tiles_backend.perspective_engine.camera.vp_from_mask", "detect_lines_lsd"),
    ("tiles_backend.perspective_engine.camera.vp_from_mask", "compute_vanishing_points"),
    ("tiles_backend.perspective_engine.camera.focal_estimate", "estimate_focal_from_exif"),
    ("live_scene", "build"),
]


def main() -> int:
    work = harness.WORK / "tile_change"
    if work.exists():
        shutil.rmtree(work)
    (work / "jobs").mkdir(parents=True)

    import app as backend
    from fastapi.testclient import TestClient
    from tiles_backend.perspective_engine.core import memo

    calls = collections.Counter()
    for modname, names in DETECTORS.items():
        mod = importlib.import_module(modname)
        for name in names:
            original = getattr(mod, name)

            @functools.wraps(original)
            def counted(*a, _o=original, _n=f"{modname.split('.')[-1]}.{name}", **k):
                calls[_n] += 1
                return _o(*a, **k)
            for m in list(sys.modules.values()):
                if m is not None and getattr(m, name, None) is original:
                    setattr(m, name, counted)

    backend.JOBS_DIR = work / "jobs"
    client = TestClient(backend.app)
    tile = (harness.FIXTURES / "tile.png").read_bytes()
    failed, total = 0, 0

    for room in harness._rooms(None):
        job = json.loads((room / "fixture.json").read_text())["clean_room_job"]
        shutil.copytree(room, work / "jobs" / job)
        photo = harness._photo(room).read_bytes()
        walls = sorted(p.stem for p in (room / "segments" / "wall" / "walls").glob("wall-*.png"))
        for surface, extra in (("floor", {}), ("wall", {"wall_id": walls[0]})):
            def render(tw, th):
                return client.post("/generate", data={
                    "tile_width": tw, "tile_height": th, "grout_mm": 5, "rotation": 0, "job_id": job,
                    "surface": surface, "room_mode": "auto", **extra},
                    files={"room_image": ("p.png", photo), "tile_image": ("t.png", tile)})

            render(*TILES[0])
            for tw, th in TILES[1:]:
                total += 1
                memo.reset_stats(); calls.clear()
                r = render(tw, th)
                s = memo.stats()
                misses = {k.split(".")[-1]: v["misses"] for k, v in s.items() if v["misses"]}
                hits = sum(v["hits"] for v in s.values())
                ok = r.status_code == 200 and not misses and not calls and hits > 0
                failed += not ok
                print(f"{'PASS' if ok else 'FAIL'}  {room.name:<8} {surface}{'/' + extra.get('wall_id', '') if extra else ''}"
                      f" tile {tw}x{th}: HTTP {r.status_code}, memo hits {hits}, misses {misses or 0}, "
                      f"detectors called {dict(calls) or 0}")
            last = work / "jobs" / r.json()["job_id"] / "result.png"
            warm = last.read_bytes()

            # Cold: every cache cleared, same render again.
            for modname, name in MEMOISED:
                getattr(importlib.import_module(modname), name).cache_clear()
            engine = importlib.import_module("tiles_backend.perspective_engine.engine")
            engine._INSTANCE_CACHE.clear()
            cold = (work / "jobs" / render(*TILES[-1]).json()["job_id"] / "result.png").read_bytes()
            total += 1
            same = warm == cold
            failed += not same
            print(f"{'PASS' if same else 'FAIL'}  {room.name:<8} {surface}: cached render == cold render "
                  f"(result.png {len(warm)} bytes, byte-identical: {same})")

    shutil.rmtree(work, ignore_errors=True)
    print(f"\n{total - failed}/{total} passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
