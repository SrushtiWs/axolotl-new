"""
10c: the 3D view against the 2D render, pixel by pixel, on the frozen rooms.

    backend/.venv/bin/python tests/regression/three_parity.py [--rotations 0 45 90 135]

Every surface x rotation is rendered in 2D through POST /generate (in-process),
then drawn by the app's own Three.js renderer (frontend/src/3js/threeTiles.ts,
bundled by tests/threejs/) at the photo's resolution in headless Chrome with
software WebGL, and compared with the 2D engine's tiled image (lit.png):

  mask leaks     3D pixels drawn outside the surface's tile mask   must be 0
  mask missing   tile-mask pixels the 3D view left empty           must be 0
  colour         per-pixel max channel difference inside the mask  every pixel <= 8
  room object    roomConsistency() in the browser: camera, scale, tile mm  no problems
"""

from __future__ import annotations

import argparse
import json
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness  # noqa: E402

ROOT = harness.ROOT
THREEJS = ROOT / "tests" / "threejs"
ESBUILD = ROOT / "frontend" / "node_modules" / ".bin" / "esbuild"
MAX_LEVELS = 8


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rotations", nargs="*", type=int, default=list(harness.ROTATIONS))
    args = parser.parse_args()

    site = harness.WORK / "three"
    if site.exists():
        shutil.rmtree(site)
    (site / "jobs").mkdir(parents=True)
    (site / "out").mkdir()

    # ---- 2D renders ----
    import app as backend
    from fastapi.testclient import TestClient

    backend.JOBS_DIR = site / "jobs"
    client = TestClient(backend.app)
    tile = (harness.FIXTURES / "tile.png").read_bytes()
    cases = []
    for room in harness._rooms(None):
        job = json.loads((room / "fixture.json").read_text())["clean_room_job"]
        shutil.copytree(room, site / "jobs" / job)
        photo = harness._photo(room).read_bytes()
        walls = sorted(p.stem for p in (room / "segments" / "wall" / "walls").glob("wall-*.png"))
        for sid in ["floor", *walls]:
            for rotation in args.rotations:
                data = {"tile_width": harness.SETTINGS["tile_width"], "tile_height": harness.SETTINGS["tile_height"],
                        "grout_mm": harness.SETTINGS["grout_mm"], "room_mode": "auto", "rotation": rotation,
                        "job_id": job, "surface": "floor" if sid == "floor" else "wall",
                        **({} if sid == "floor" else {"wall_id": sid})}
                r = client.post("/generate", data=data,
                                files={"room_image": ("p.png", photo), "tile_image": ("t.png", tile)})
                name = f"{room.name}__{sid}__r{rotation}"
                if r.status_code != 200:
                    print(f"SKIP  {name}: 2D HTTP {r.status_code}")
                    continue
                cases.append({"name": name, "surface": sid, "jobBase": f"jobs/{r.json()['job_id']}/"})
    (site / "cases.json").write_text(json.dumps(cases))

    # ---- 3D renders: the real renderer, bundled ----
    subprocess.run([str(ESBUILD), str(THREEJS / "entry.ts"), "--bundle", "--format=iife",
                    f"--outfile={site / 'bundle.js'}", "--log-level=warning"], check=True)
    shutil.copy2(THREEJS / "page.html", site / "index.html")
    port = _free_port()
    server = subprocess.Popen([sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1",
                               "--directory", str(site)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(0.8)
        subprocess.run(["node", str(THREEJS / "run.mjs"), f"http://127.0.0.1:{port}/index.html",
                        str(site / "cases.json"), str(site / "out")], check=True, cwd=THREEJS)
    finally:
        server.terminate()
    browser = json.loads((site / "out" / "results.json").read_text())

    # ---- compare ----
    failed = 0
    totals = {"pixels": 0, "le2": 0, "le8": 0}
    for c in cases:
        job_dir = site / c["jobBase"]
        mask = np.asarray(Image.open(job_dir / "three" / f"{c['surface']}.png").convert("L")) > 127
        lit = np.asarray(Image.open(job_dir / "lit.png").convert("RGB")).astype(np.int16)
        rgba = np.asarray(Image.open(site / "out" / f"{c['name']}.png").convert("RGBA"))
        drawn = rgba[..., 3] > 0
        leaks = int((drawn & ~mask).sum())
        missing = int((mask & ~drawn).sum())
        diff = np.abs(rgba[..., :3].astype(np.int16) - lit).max(axis=2)[mask]
        worst = int(diff.max()) if diff.size else 0
        le2, le8 = float((diff <= 2).mean()) if diff.size else 1.0, float((diff <= MAX_LEVELS).mean()) if diff.size else 1.0
        problems = browser["results"][c["name"]]["problems"]
        ok = leaks == 0 and missing == 0 and worst <= MAX_LEVELS and not problems
        failed += not ok
        totals["pixels"] += int(mask.sum()); totals["le2"] += int((diff <= 2).sum()); totals["le8"] += int((diff <= MAX_LEVELS).sum())
        print(f"{'PASS' if ok else 'FAIL'}  {c['name']:<22} mask {int(mask.sum()):>7} px  leaks {leaks}  missing {missing}  "
              f"colour: {100 * le2:6.2f}% <=2, {100 * le8:6.2f}% <=8, max {worst}"
              + (f"  room: {problems}" if problems else ""))

    print(f"\nall surfaces: {totals['pixels']} tile pixels, {100 * totals['le2'] / max(totals['pixels'], 1):.2f}% within 2 levels, "
          f"{100 * totals['le8'] / max(totals['pixels'], 1):.2f}% within 8; page errors: {browser['errors'] or 'none'}")
    print(f"{len(cases) - failed}/{len(cases)} passed")
    return 0 if failed == 0 and not browser["errors"] else 1


if __name__ == "__main__":
    sys.exit(main())
