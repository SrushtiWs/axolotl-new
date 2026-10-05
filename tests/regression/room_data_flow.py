"""
room-data Step 3: open / reopen / edit / broken profile, through the real endpoints.

    backend/.venv/bin/python tests/regression/room_data_flow.py first  <room_id> <out>
    backend/.venv/bin/python tests/regression/room_data_flow.py reopen <room_id> <out>

first   opens the room as the Studio does (POST /segment/start, use_saved=true).
        With no profile.json, Clean Room runs once and the result is saved.
        Then renders the floor (POST /generate) and stores the result bytes.
reopen  a FRESH process with an EMPTY jobs folder: opening the room must come
        from room-data (no model is ever loaded), the response must equal the
        first one, and the same floor render must be byte-identical. Then:
        manual edit (backup + index), invalid edit refused, broken profile.json
        falls back to the backup.

Uses the real room-data/ folder; jobs go to a scratch folder (backend/jobs untouched).
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "backend")]

import app as app_module  # noqa: E402
import room_data  # noqa: E402
import surfaces  # noqa: E402
from extraction import dino, inpaint, sam2  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

failures = []
# (name, module, the module's cached session); all None until a model is first used.
MODELS = [("SegFormer", surfaces, "_session"), ("Grounding DINO", dino, "_session"),
          ("SAM2", sam2, "_encoder"), ("LaMa", inpaint, "_session")]


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'' if ok else '  ' + str(detail)}", flush=True)
    if not ok:
        failures.append(name)


def strip(response: dict) -> dict:
    return {k: v for k, v in response.items() if k != "room_data"}


def open_room(client, photo: Path, use_saved=True) -> tuple[dict, float]:
    """The same core as /segment/start (which the Studio polls), called synchronously."""
    t = time.perf_counter()
    r = client.post("/segment", files={"room_image": (photo.name, photo.read_bytes(), "image/jpeg")},
                    data={"use_saved": "true" if use_saved else "false"})
    return r.json(), time.perf_counter() - t


def render_floor(client, photo: Path, job_id: str) -> bytes:
    tile = ROOT / "tests" / "fixtures" / "tile.png"
    r = client.post("/generate", files={"room_image": (photo.name, photo.read_bytes(), "image/jpeg"),
                                        "tile_image": ("tile.png", tile.read_bytes(), "image/png")},
                    data={"tile_width": "600", "tile_height": "600", "rotation": "0", "surface": "floor",
                          "grout_mm": "3", "job_id": job_id})
    body = r.json()
    if r.status_code != 200:
        return f"HTTP {r.status_code}: {body}".encode()
    url = body["result_image_url"].split("/jobs/", 1)[1]
    return (app_module.JOBS_DIR / url).read_bytes()


def main() -> int:
    mode, room_id, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    app_module.JOBS_DIR = Path(tempfile.mkdtemp(prefix="room_data_jobs_"))
    client = TestClient(app_module.app)
    entry = next(r for r in room_data.load_index()["rooms"] if r["id"] == room_id)
    photo = room_data.room_dir(room_id) / entry["original"]
    folder = room_data.room_dir(room_id)

    if mode == "first":
        check("no profile before first open", not (folder / "profile.json").exists())
        body, seconds = open_room(client, photo)
        print(f"      first open: {seconds:.1f} s, source {body.get('source')}, room_data {body.get('room_data')}")
        check("Clean Room ran (live)", body.get("source") == "live", body.get("detail"))
        check("saved now", (body.get("room_data") or {}).get("source") == "saved now", body.get("room_data"))
        for rel in ("profile.json", "job/segment_response.json", "masks/floor.png", "masks/walls.png",
                    "masks/objects.png", "result/preview.jpg"):
            check(f"wrote {rel}", (folder / rel).exists())
        profile = json.loads((folder / "profile.json").read_text())
        check("profile validates", not room_data.validate_profile(profile), room_data.validate_profile(profile))
        index = {r["id"]: r for r in room_data.load_index()["rooms"]}
        check("index updated", index[room_id]["status"] == profile["status"]
              and index[room_id]["wall_count"] == profile["wall_count"], index[room_id])
        render = render_floor(client, photo, body["job_id"])
        (out / "first_render.png").write_bytes(render)
        (out / "first_response.json").write_text(json.dumps(strip(body), sort_keys=True))
        (out / "first_open_seconds.txt").write_text(f"{seconds:.1f}")
        (out / "first_profile.json").write_text((folder / "profile.json").read_text())
    else:
        first = json.loads((out / "first_response.json").read_text())
        check("jobs folder starts empty", not any(app_module.JOBS_DIR.iterdir()))
        body, seconds = open_room(client, photo)
        print(f"      reopen: {seconds:.2f} s (first open {(out / 'first_open_seconds.txt').read_text()} s), "
              f"room_data {body.get('room_data')}")
        check("came from room-data", (body.get("room_data") or {}).get("source") == "file", body.get("room_data"))
        loaded = {name: getattr(module, attr) is not None for name, module, attr in MODELS}
        check("no model loaded (no detection)", not any(loaded.values()), loaded)
        check("response identical to the first open", strip(body) == first)
        check("job restored under its own id", (app_module.JOBS_DIR / first["job_id"]).is_dir())
        render = render_floor(client, photo, body["job_id"])
        same = render == (out / "first_render.png").read_bytes()
        check("floor render byte-identical to the first", same,
              hashlib.sha256(render).hexdigest()[:12])

        # ---- manual edit: backup + index
        backups_before = len(room_data._backups(room_id))
        r = client.put(f"/room-data/rooms/{room_id}/profile", json={"group": "D", "room_size_ft": [12, 14, 10]})
        check("manual edit accepted", r.status_code == 200, r.text[:200])
        check("edit backed up the previous profile", len(room_data._backups(room_id)) == backups_before + 1)
        p = json.loads((folder / "profile.json").read_text())
        check("edit saved (group D manual, size)", p["group"] == "D" and p["group_source"] == "manual"
              and p["room_size_ft"] == [12, 14, 10])
        index = {x["id"]: x for x in room_data.load_index()["rooms"]}
        check("index follows the edit", index[room_id]["group"] == "D")
        good = (folder / "profile.json").read_bytes()
        r = client.put(f"/room-data/rooms/{room_id}/profile", json={"status": "finished"})
        check("invalid edit refused (422)", r.status_code == 422, r.status_code)
        check("invalid edit wrote nothing", (folder / "profile.json").read_bytes() == good)
        r = client.put(f"/room-data/rooms/{room_id}/profile", json={"walls": []})
        check("non-editable field refused", r.status_code == 422, r.status_code)

        # ---- broken profile -> backup
        (folder / "profile.json").write_text('{"room_id": "' + room_id + '", "status": ')
        r = client.get(f"/room-data/rooms/{room_id}")
        body = r.json()
        print(f"      broken file -> {body.get('source')}: {body.get('message')}")
        check("broken profile falls back to a backup", body.get("source") == "backup" and body.get("profile"))
        check("clear message", "broken" in (body.get("message") or "") and "restored" in body["message"])
        check("profile.json repaired on disk", not room_data.validate_profile(json.loads((folder / "profile.json").read_text())))
        check("broken file kept", any(room_data._backup_dir(room_id).glob("broken-*.json")))
        r = client.get(f"/room-data/rooms/{room_id}")
        check("next load is from the file again", r.json().get("source") == "file")
        # Leave the room as the first open saved it (the edits above stay in _backup/).
        restored = json.loads((out / "first_profile.json").read_text())
        restored["updated_at"] = room_data.now_iso()
        room_data.save_profile(restored)
    print(f"\n{'all passed' if not failures else f'{len(failures)} failed'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
