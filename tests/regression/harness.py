"""
Regression harness: floor + every wall x rotations 0/45/90/135, on the frozen
rooms in tests/fixtures/rooms/.

    backend/.venv/bin/python tests/regression/harness.py                    # compare
    backend/.venv/bin/python tests/regression/harness.py --update-baseline  # save baseline

Each case goes through the real POST /generate, in-process (FastAPI
TestClient), against a scratch jobs folder holding copies of the fixture
Clean Room jobs; backend/jobs/ and a running server are not involved.

Per case:
  pixel parity   result.png and the tile coverage vs tests/baseline/
  mask clip      0 tile pixels outside FLOOR_MASK.png / WALL_MASK.png, and
                 nothing changed outside the tiles
  three.json     JSON and geometry round-trip (see checks.py)

Exit status 0 only when every case passes.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "backend", ROOT / "tests" / "regression"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import checks  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures"
BASELINE = ROOT / "tests" / "baseline"
WORK = ROOT / "tests" / ".work"

ROTATIONS = (0, 45, 90, 135)

#: The render every case uses. AUTO room size (no dimensions sent), so the
#: baseline covers the default path; the tile is the fixture artwork.
SETTINGS = {
    "tile_width": 1200,
    "tile_height": 1800,
    "grout_mm": 5,
    "room_mode": "auto",
}


def _rooms(only: list[str] | None) -> list[Path]:
    rooms = sorted(p for p in (FIXTURES / "rooms").iterdir() if (p / "fixture.json").is_file())
    return [p for p in rooms if not only or p.name in only]


def _photo(room: Path) -> Path:
    return next(room.glob("photo.*"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--update-baseline", action="store_true", help="save this run as the baseline")
    parser.add_argument("--rooms", nargs="*", help="only these fixture rooms")
    parser.add_argument("--rotations", nargs="*", type=int, default=list(ROTATIONS))
    parser.add_argument("--manifest", type=Path, default=BASELINE / "manifest.json",
                        help="baseline to compare to / save (its images sit beside it)")
    parser.add_argument("--set", action="append", default=[], metavar="MODULE.ATTR=VALUE",
                        help="test-only: override a module constant for this run (e.g. a gate)")
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    images_dir = manifest_path.parent / "images"
    baseline = {} if args.update_baseline else json.loads(manifest_path.read_text())["cases"]

    # Scratch jobs folder with the fixture Clean Room jobs under their own ids.
    if WORK.exists():
        shutil.rmtree(WORK)
    jobs = WORK / "jobs"
    jobs.mkdir(parents=True)

    import app as backend  # noqa: E402  (the FastAPI app, imported after sys.path)
    from fastapi.testclient import TestClient

    backend.JOBS_DIR = jobs

    import importlib
    for item in args.set:
        target, value = item.split("=", 1)
        module_name, attr = target.rsplit(".", 1)
        module = importlib.import_module(module_name)
        if not hasattr(module, attr):
            sys.exit(f"--set: {module_name} has no {attr}")
        setattr(module, attr, type(getattr(module, attr))(value))
        print(f"override: {module_name}.{attr} = {getattr(module, attr)!r}")
    client = TestClient(backend.app)
    tile_bytes = (FIXTURES / "tile.png").read_bytes()

    cases: dict[str, dict] = {}
    failures = 0
    started = time.time()

    for room in _rooms(args.rooms):
        meta = json.loads((room / "fixture.json").read_text())
        job = meta["clean_room_job"]
        shutil.copytree(room, jobs / job)
        photo = _photo(room)
        photo_bytes = photo.read_bytes()

        listed = client.post("/surfaces", files={"room_image": (photo.name, photo_bytes)}, data={"job_id": job})
        listed.raise_for_status()
        surface_ids = [item["id"] for item in listed.json()["surfaces"]]

        for sid in surface_ids:
            for rotation in args.rotations:
                name = f"{room.name}/{sid}/r{rotation}"
                data = {
                    "tile_width": SETTINGS["tile_width"], "tile_height": SETTINGS["tile_height"],
                    "grout_mm": SETTINGS["grout_mm"], "room_mode": SETTINGS["room_mode"],
                    "rotation": rotation, "job_id": job,
                    "surface": "floor" if sid == "floor" else "wall",
                }
                if sid != "floor":
                    data["wall_id"] = sid
                response = client.post("/generate", data=data, files={
                    "room_image": (photo.name, photo_bytes), "tile_image": ("tile.png", tile_bytes)})

                case = {"status": response.status_code}
                report = {"status": response.status_code}

                if response.status_code != 200:
                    case["detail"] = response.json().get("detail")
                else:
                    body = response.json()
                    render_dir = jobs / body["job_id"]
                    result = checks.load_rgb(render_dir / "result.png")
                    shape = result.shape[:2]
                    tiled_path = render_dir / "three" / f"{sid}.png"
                    tiled = checks.load_mask(tiled_path, shape) if tiled_path.is_file() else np.zeros(shape, bool)
                    allowed = checks.load_mask(
                        room / "segments" / ("FLOOR_MASK.png" if sid == "floor" else "WALL_MASK.png"), shape)

                    case.update({
                        "result_hash": checks.pixel_hash(result),
                        "tile_mask_hash": checks.pixel_hash(tiled.astype(np.uint8)),
                        "tile_pixels": int(tiled.sum()),
                        "room_status": ((body.get("geometry") or {}).get("room") or {}).get("status"),
                    })
                    objects_png = render_dir / "three" / "objects.png"
                    report["mask_clip"] = checks.mask_clip(
                        tiled, allowed, result,
                        checks.load_rgb(render_dir / "CLEAN_ROOM.png"),
                        checks.load_rgb(render_dir / "original.png"),
                        (body.get("geometry") or {}).get("mask_clip"),
                        np.asarray(Image.open(objects_png))[..., 3] if objects_png.is_file() else None)
                    report["three_json"] = (checks.three_json(render_dir, sid, shape,
                                                              (SETTINGS["tile_width"], SETTINGS["tile_height"]))
                                            if (render_dir / "three.json").is_file()
                                            else {"ok": False, "problems": ["no three.json"]})

                    image_path = images_dir / f"{name.replace('/', '__')}.png"
                    if args.update_baseline:
                        image_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(render_dir / "result.png", image_path)

                if not args.update_baseline:
                    want = baseline.get(name)
                    if want is None:
                        report["pixel_parity"] = {"ok": False, "note": "case not in baseline"}
                    elif want["status"] != case["status"]:
                        report["pixel_parity"] = {"ok": False, "note": f"HTTP {case['status']} != baseline {want['status']}"}
                    elif case["status"] != 200:
                        report["pixel_parity"] = {"ok": want.get("detail") == case.get("detail")}
                    else:
                        image_path = images_dir / f"{name.replace('/', '__')}.png"
                        stored = checks.load_rgb(image_path) if image_path.is_file() else None
                        parity = checks.pixel_parity(result, stored, want["result_hash"])
                        parity["tile_mask_equal"] = want["tile_mask_hash"] == case["tile_mask_hash"]
                        parity["ok"] = parity["ok"] and parity["tile_mask_equal"]
                        # Byte level too: the PNG file itself, when the baseline image is on disk.
                        if image_path.is_file():
                            parity["file_identical"] = (image_path.read_bytes()
                                                        == (render_dir / "result.png").read_bytes())
                            parity["ok"] = parity["ok"] and parity["file_identical"]
                        report["pixel_parity"] = parity

                ok = all(v.get("ok", True) for k, v in report.items() if isinstance(v, dict))
                failures += not ok
                case_report = {**case, **report, "ok": ok}
                cases[name] = case_report

                cols = " ".join(
                    f"{k}={'PASS' if v.get('ok') else 'FAIL'}"
                    for k, v in report.items() if isinstance(v, dict))
                extra = "" if response.status_code == 200 else f" ({case.get('detail')})"
                print(f"{'PASS' if ok else 'FAIL'}  {name:<24} HTTP {response.status_code} "
                      f"tiles={case.get('tile_pixels', 0):>7} {cols}{extra}", flush=True)

                # A render's folder is only needed for its checks.
                if response.status_code == 200:
                    shutil.rmtree(jobs / response.json()["job_id"], ignore_errors=True)

    elapsed = round(time.time() - started, 1)
    (WORK / "last_report.json").write_text(json.dumps(cases, indent=2, default=str))

    if args.update_baseline:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps({
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "settings": {**SETTINGS, "rotations": args.rotations, "tile": "tests/fixtures/tile.png",
                         "overrides": args.set},
            "rooms": {p.name: json.loads((p / "fixture.json").read_text()) for p in _rooms(args.rooms)},
            "cases": {k: {f: v[f] for f in ("status", "detail", "result_hash", "tile_mask_hash",
                                             "tile_pixels", "room_status") if f in v}
                      for k, v in cases.items()},
        }, indent=2))
        print(f"\nbaseline saved: {len(cases)} cases -> {manifest_path}")

    print(f"\n{len(cases) - failures}/{len(cases)} cases passed in {elapsed} s "
          f"(report: {(WORK / 'last_report.json').relative_to(ROOT)})")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
