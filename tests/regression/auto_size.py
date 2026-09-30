"""
Auto room size (Step 7), on the frozen rooms: nothing typed.

    backend/.venv/bin/python tests/regression/auto_size.py

Per room, a floor render through POST /generate with room_mode=auto. Rules:

  prior scale        the scale is the 1500 mm camera-height prior, flagged
                     is_estimated, every dimension ESTIMATED or UNKNOWN
  returned           width / length / height (mm), sources, confidence
  lower bounds       length is always a lower bound (the front wall is behind
                     the camera); when the back wall is not measured it comes
                     from the farthest visible floor; a width with no pair of
                     side walls is a lower bound from the visible floor
  UI line            "Estimated dimensions: ..." when all three exist, else the
                     render is refused with "Please enter one known measurement"
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness  # noqa: E402

from tiles_backend.perspective_engine.room.geometry import CAMERA_HEIGHT_PRIOR_MM  # noqa: E402

PROMPT = "Please enter one known measurement"


def main() -> int:
    work = harness.WORK / "auto_size"
    if work.exists():
        shutil.rmtree(work)
    (work / "jobs").mkdir(parents=True)

    import app as backend
    from fastapi.testclient import TestClient

    backend.JOBS_DIR = work / "jobs"
    client = TestClient(backend.app)
    tile = (harness.FIXTURES / "tile.png").read_bytes()
    failed = 0

    for room in harness._rooms(None):
        job = json.loads((room / "fixture.json").read_text())["clean_room_job"]
        shutil.copytree(room, work / "jobs" / job)
        photo = harness._photo(room)
        frame = json.loads((room / "segments" / "room" / "room_frame.json").read_text())
        box = frame.get("box_units") or {}

        response = client.post("/generate", data={
            "tile_width": 1200, "tile_height": 1800, "grout_mm": 5, "rotation": 0,
            "job_id": job, "surface": "floor", "room_mode": "auto"},
            files={"room_image": (photo.name, photo.read_bytes()), "tile_image": ("tile.png", tile)})

        problems, r = [], {}
        if response.status_code != 200:
            detail = response.json().get("detail") or ""
            if not (response.status_code == 422 and detail.startswith(PROMPT)):
                problems.append(f"HTTP {response.status_code}: {detail}")
            line = detail
        else:
            body = response.json()
            r = body["geometry"].get("room") or {}
            line = r.get("room_check") or ""
            shutil.rmtree(work / "jobs" / body["job_id"], ignore_errors=True)

            if not r.get("is_estimated"):
                problems.append("is_estimated is not set")
            if r.get("camera_height_source") != "CAMERA_HEIGHT_PRIOR" or r.get("camera_height_mm") != CAMERA_HEIGHT_PRIOR_MM:
                problems.append(f"scale {r.get('camera_height_mm')} from {r.get('camera_height_source')}")
            for name in ("width", "length", "height"):
                src = r.get(f"{name}_source")
                if src not in ("ESTIMATED", "UNKNOWN"):
                    problems.append(f"{name} source {src}")
                if (r.get(f"{name}_mm") is None) != (src == "UNKNOWN"):
                    problems.append(f"{name} value/source mismatch")
            if r.get("confidence") is None:
                problems.append("no confidence")
            if r.get("length_mm") is not None:
                if not r.get("length_is_lower_bound"):
                    problems.append("length not flagged as a lower bound")
                want = "back wall" if box.get("back_z") is not None else "farthest visible floor"
                if r.get("length_basis") != want:
                    problems.append(f"length basis {r.get('length_basis')!r}, expected {want!r}")
            no_pair = box.get("left_x") is None or box.get("right_x") is None
            if r.get("width_mm") is not None and bool(r.get("width_is_lower_bound")) != no_pair:
                problems.append("width lower-bound flag does not match the side walls seen")
            all_three = all(r.get(f"{n}_mm") for n in ("width", "length", "height"))
            if not line.startswith("Estimated dimensions" if all_three else PROMPT):
                problems.append(f"UI line {line!r}")

        failed += bool(problems)
        seen = (f"sides {'L' if box.get('left_x') is not None else '-'}{'R' if box.get('right_x') is not None else '-'}"
                f" back {'yes' if box.get('back_z') is not None else 'no'}")
        dims = " ".join(
            f"{n[0].upper()}={'?' if r.get(f'{n}_mm') is None else round(r[f'{n}_mm'])}"
            f"{'+' if r.get(f'{n}_is_lower_bound') else ''}" for n in ("width", "length", "height")) if r else "-"
        conf = r.get("confidence")
        print(f"{'PASS' if not problems else 'FAIL'}  {room.name:<8} HTTP {response.status_code}  {seen:<16} "
              f"{dims:<26} conf {conf if conf is not None else '-'}  length: {r.get('length_basis') or '-'}")
        print(f"        UI: {line}")
        for p in problems:
            print(f"        - {p}")

    shutil.rmtree(work, ignore_errors=True)
    print("\n(+ = lower bound, mm)")
    print(f"{3 - failed}/3 passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
