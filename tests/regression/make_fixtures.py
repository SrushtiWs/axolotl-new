"""
Freeze Clean Room jobs as regression fixtures.

    backend/.venv/bin/python tests/regression/make_fixtures.py NAME=PHOTO [NAME=PHOTO ...]

Runs Clean Room (POST /segment/start) for each photo on a running backend, then
copies that job's folder (original.png, reuse/, segments/) and the exact photo
bytes into tests/fixtures/rooms/NAME/. The harness renders from these copies,
so its baseline never depends on backend/jobs/ or on the ML models re-running.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
ROOMS = ROOT / "tests" / "fixtures" / "rooms"
JOBS = ROOT / "backend" / "jobs"


def clean_room(api: str, photo: Path) -> str:
    with photo.open("rb") as handle:
        started = httpx.post(f"{api}/segment/start", files={"room_image": (photo.name, handle)}, timeout=120)
    started.raise_for_status()
    token = started.json()["job_id"]

    while True:
        status = httpx.get(f"{api}/segment/status/{token}", timeout=60)
        status.raise_for_status()
        body = status.json()
        if body["status"] == "done":
            return body.get("job_id") or token
        print(f"  {photo.name}: cleaning {body.get('elapsed_s')} s", flush=True)
        time.sleep(5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rooms", nargs="+", help="NAME=PATH/TO/PHOTO")
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--into", type=Path, default=ROOMS,
                        help="where to freeze them (default: the harness fixtures)")
    args = parser.parse_args()

    for item in args.rooms:
        name, photo = item.split("=", 1)
        photo = Path(photo).resolve()
        job = clean_room(args.api, photo)
        source = JOBS / job
        if not (source / "reuse").is_dir() or not (source / "segments" / "room" / "room_frame.json").is_file():
            sys.exit(f"{name}: job {job} has no reuse/ or stored room geometry")

        target = args.into.resolve() / name
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)
        for part in ("original.png", "reuse", "segments"):
            src = source / part
            (shutil.copytree if src.is_dir() else shutil.copy2)(src, target / part)
        shutil.copy2(photo, target / f"photo{photo.suffix.lower()}")
        (target / "fixture.json").write_text(json.dumps(
            {"name": name, "source_photo": photo.name, "clean_room_job": job,
             "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))
        print(f"{name}: job {job} -> {target}", flush=True)


if __name__ == "__main__":
    main()
