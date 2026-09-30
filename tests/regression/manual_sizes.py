"""
Manual room size as a hard constraint (Step 6), on the frozen rooms.

    backend/.venv/bin/python tests/regression/manual_sizes.py

Each size is typed in full (W x L x H mm, sent in feet as the UI does) and a
floor is rendered through POST /generate. Checked per case, as rules rather
than expected numbers:

  never overridden   RoomGeometry reports exactly the typed W / L / H
  drives the scale   every typed dimension the photo measures is in the
                     scale fit (camera_height_source); the 1500 mm camera prior
                     is used only when the photo measures none of them
  12% rule           status CONFLICT  <=>  some |residual| > 0.12 or the length
                     is shorter than the photo's back wall; residuals reported
  status line        "Consistent" / "Conflict with photo" / "Not checked
                     against photo", matching the status
  tile layout        the floor's tile count uses the typed width and length
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness  # noqa: E402  (sets sys.path for the backend)

from tiles_backend.perspective_engine.room.geometry import CONFLICT_TOLERANCE, MM_PER_FOOT  # noqa: E402

SIZES_MM = [(4000, 5000, 3000), (6000, 8000, 3000), (3000, 4000, 2700)]
LINES = {"OK": "Consistent", "CONFLICT": "Conflict with photo", "UNCHECKED": "Not checked against photo"}


def main() -> int:
    work = harness.WORK / "manual_sizes"
    if work.exists():
        shutil.rmtree(work)
    (work / "jobs").mkdir(parents=True)

    import app as backend
    from fastapi.testclient import TestClient

    backend.JOBS_DIR = work / "jobs"
    client = TestClient(backend.app)
    tile = (harness.FIXTURES / "tile.png").read_bytes()

    failed, total, rows = 0, 0, []
    for room in harness._rooms(None):
        job = json.loads((room / "fixture.json").read_text())["clean_room_job"]
        shutil.copytree(room, work / "jobs" / job)
        photo = harness._photo(room)

        for w, l, h in SIZES_MM:
            total += 1
            label = f"{room.name:<8} {w}x{l}x{h}"
            response = client.post("/generate", data={
                "tile_width": 1200, "tile_height": 1800, "grout_mm": 5, "rotation": 0,
                "job_id": job, "surface": "floor", "room_mode": "manual",
                "room_width": w / MM_PER_FOOT, "room_length": l / MM_PER_FOOT, "room_height": h / MM_PER_FOOT},
                files={"room_image": (photo.name, photo.read_bytes()), "tile_image": ("tile.png", tile)})

            problems = []
            if response.status_code != 200:
                problems.append(f"HTTP {response.status_code}: {response.json().get('detail')}")
                room_geo, layout = {}, {}
            else:
                body = response.json()
                room_geo = body["geometry"].get("room") or {}
                layout = (body["geometry"].get("tile_layout") or {}).get("surfaces", {}).get("floor", {})
                shutil.rmtree(work / "jobs" / body["job_id"], ignore_errors=True)

            if room_geo:
                typed = {"width": w, "length": l, "height": h}
                for name, value in typed.items():
                    if abs((room_geo.get(f"{name}_mm") or 0) - value) > 1e-6:
                        problems.append(f"{name} reported {room_geo.get(f'{name}_mm')} != typed {value}")

                checks = room_geo.get("input_checks") or {}
                if set(checks) != set(typed):
                    problems.append(f"input_checks for {sorted(checks)}")
                used = [n for n, e in checks.items() if e.get("used_for_scale")]
                src = room_geo.get("camera_height_source") or ""
                if used and not all(n in src for n in used):
                    problems.append(f"scale source {src!r} misses {used}")
                if not used and src != "CAMERA_HEIGHT_PRIOR":
                    problems.append(f"no measurable input, yet scale source {src!r}")

                over = [n for n, e in checks.items()
                        if e.get("residual") is not None and abs(e["residual"]) > CONFLICT_TOLERANCE]
                length_short = checks.get("length", {}).get("check") == "CONFLICT"
                want_conflict = bool(over) or length_short
                if (room_geo.get("status") == "CONFLICT") != want_conflict:
                    problems.append(f"status {room_geo.get('status')} but conflict={want_conflict}")
                if room_geo.get("status") == "CONFLICT" and not room_geo.get("residuals"):
                    problems.append("CONFLICT without residuals")

                line = room_geo.get("room_check") or ""
                if not line.startswith(LINES.get(room_geo.get("status"), "?")):
                    problems.append(f"status line {line!r} for status {room_geo.get('status')}")

                if (layout.get("span_x_mm"), layout.get("span_z_mm")) != (float(w), float(l)):
                    problems.append(f"tile layout spans {layout.get('span_x_mm')} x {layout.get('span_z_mm')}")

            ok = not problems
            failed += not ok
            per_dim = ", ".join(
                f"{n[0].upper()} {e['check'].lower()}"
                + (f" {e['residual']:+.1%}" if e.get("residual") is not None else "")
                + (" (scale)" if e.get("used_for_scale") else "")
                for n, e in (room_geo.get("input_checks") or {}).items())
            rows.append((ok, label, room_geo.get("status"), room_geo.get("room_check"),
                         room_geo.get("camera_height_mm"), per_dim,
                         f"{layout.get('tile_count_x')}x{layout.get('tile_count_z')}={layout.get('total_tiles')}", problems))

    for ok, label, status, line, cam, per_dim, tiles, problems in rows:
        cam_txt = "-" if cam is None else f"{cam:.0f}"
        print(f"{'PASS' if ok else 'FAIL'}  {label}  {str(status):<9} {str(line):<34} cam {cam_txt:>5} mm  "
              f"floor tiles {tiles:<9} | {per_dim}")
        for p in problems:
            print(f"        - {p}")
    shutil.rmtree(work, ignore_errors=True)
    print(f"\n{total - failed}/{total} passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
