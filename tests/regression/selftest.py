"""
Negative controls: prove each check in checks.py can fail.

    backend/.venv/bin/python tests/regression/selftest.py

Renders one fixture case, then feeds each check a deliberately broken input
(one changed pixel, a tile mask grown past the surface mask, a pixel changed
outside the tiles, a shifted three.json grid, a three.json plane moved) and
expects FAIL. The untouched render must PASS. Exit status 0 only if all hold.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness  # noqa: E402  (sets sys.path for the backend)
import checks  # noqa: E402


def main() -> int:
    work = harness.WORK / "selftest"
    if work.exists():
        shutil.rmtree(work)
    (work / "jobs").mkdir(parents=True)

    import app as backend
    from fastapi.testclient import TestClient

    backend.JOBS_DIR = work / "jobs"
    client = TestClient(backend.app)

    room = harness.FIXTURES / "rooms" / "living"
    job = json.loads((room / "fixture.json").read_text())["clean_room_job"]
    shutil.copytree(room, work / "jobs" / job)
    photo = harness._photo(room)
    response = client.post("/generate", data={
        "tile_width": 1200, "tile_height": 1800, "grout_mm": 5, "room_mode": "auto",
        "rotation": 45, "job_id": job, "surface": "floor"},
        files={"room_image": (photo.name, photo.read_bytes()),
               "tile_image": ("tile.png", (harness.FIXTURES / "tile.png").read_bytes())})
    response.raise_for_status()
    render = work / "jobs" / response.json()["job_id"]

    result = checks.load_rgb(render / "result.png")
    shape = result.shape[:2]
    tiled = checks.load_mask(render / "three" / "floor.png", shape)
    allowed = checks.load_mask(room / "segments" / "FLOOR_MASK.png", shape)
    clean = checks.load_rgb(render / "CLEAN_ROOM.png")
    original = checks.load_rgb(render / "original.png")
    from PIL import Image
    alpha = np.asarray(Image.open(render / "three" / "objects.png"))[..., 3]

    outcomes = []

    def expect(name, report, want_ok):
        good = bool(report["ok"]) == want_ok
        outcomes.append(good)
        print(f"{'OK ' if good else 'BAD'}  {name:<46} check={'PASS' if report['ok'] else 'FAIL'} "
              f"(expected {'PASS' if want_ok else 'FAIL'})")

    base_hash = checks.pixel_hash(result)
    expect("parity: untouched render", checks.pixel_parity(result, result, base_hash), True)
    one = result.copy()
    one[shape[0] // 2, shape[1] // 2, 0] ^= 1
    expect("parity: one channel of one pixel changed", checks.pixel_parity(one, result, base_hash), False)

    expect("mask clip: untouched render (room with objects)",
           checks.mask_clip(tiled, allowed, result, clean, original, None, alpha), True)
    grown = tiled.copy()
    ys, xs = np.nonzero(~allowed)
    grown[ys[0], xs[0]] = True
    expect("mask clip: one tile pixel outside the mask",
           checks.mask_clip(grown, allowed, result, clean, original, None, alpha), False)
    stray = result.copy()
    ys, xs = np.nonzero(~tiled & (alpha == 0))
    stray[ys[0], xs[0]] = (clean[ys[0], xs[0]].astype(int) + 7) % 256
    expect("mask clip: one pixel changed outside the tiles by 7 levels",
           checks.mask_clip(tiled, allowed, stray, clean, original, None, alpha), False)
    leak = result.copy()
    ys, xs = np.nonzero(~tiled & (alpha > 0) & (alpha < 255))
    leak[ys[0], xs[0]] = result[np.nonzero(tiled)][0]
    expect("mask clip: tile colour at an object's soft edge",
           checks.mask_clip(tiled, allowed, leak, clean, original, None, alpha), False)

    expect("three.json: untouched", checks.three_json(render, "floor", shape, (1200, 1800)), True)
    three_path = render / "three.json"
    pristine = three_path.read_text()
    for label, mutate in (
        ("three.json: grid offset shifted 1 mm", lambda s: s["grid"].__setitem__(
            "offset_units", [s["grid"]["offset_units"][0] + 1.0 / s["grid"]["mm_per_unit"],
                             s["grid"]["offset_units"][1]])),
        ("three.json: plane moved 1% further away", lambda s: s["plane"].__setitem__(
            "d_units", s["plane"]["d_units"] * 1.01)),
        ("three.json: record camera focal != room camera", lambda s: s["camera"].__setitem__(
            "focal_px", s["camera"]["focal_px"] * 1.001)),
        ("three.json: a second scale (mm_per_unit x1.05)", lambda s: (
            s["grid"].__setitem__("mm_per_unit", s["grid"]["mm_per_unit"] * 1.05),
            s.__setitem__("meters_per_unit", s["meters_per_unit"] * 1.05))),
        ("three.json: tile mm not the requested tile", lambda s: s["tile"].__setitem__(
            "width_mm", s["tile"]["width_mm"] + 1)),
    ):
        doc = json.loads(pristine)
        mutate(doc["surfaces"]["floor"])
        three_path.write_text(json.dumps(doc))
        expect(label, checks.three_json(render, "floor", shape, (1200, 1800)), False)
    three_path.write_text(pristine)

    shutil.rmtree(work, ignore_errors=True)
    print(f"\n{sum(outcomes)}/{len(outcomes)} controls behaved as expected")
    return 0 if all(outcomes) else 1


if __name__ == "__main__":
    sys.exit(main())
