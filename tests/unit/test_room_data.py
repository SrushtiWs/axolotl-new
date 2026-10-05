"""
room-data Step 1 + 2: folder import and the profile.json format.

    backend/.venv/bin/python tests/unit/test_room_data.py

Runs against a temporary room-data folder (the real one is not touched):
  * import: room_01, room_02 ... ; SHA-256 duplicates skipped; bytes unchanged;
    non-jpg keeps its own extension; non-images ignored
  * build_profile from a frozen Clean Room job -> validate_profile passes
  * every validation rule rejects a broken profile
  * Step 3: save from a job, backup before overwrite, broken file -> backup,
    a manual camera survives a re-run, saved response rebased to the server
"""

from __future__ import annotations

import copy
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "backend")]

import room_data  # noqa: E402

failures = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'' if ok else '  ' + str(detail)}")
    if not ok:
        failures.append(name)


def use_root(tmp: Path) -> None:
    room_data.ROOT = tmp / "room-data"
    room_data.ROOMS = room_data.ROOT / "rooms"
    room_data.BACKUP = room_data.ROOT / "_backup"
    room_data.INDEX = room_data.ROOT / "index.json"
    room_data.ERRORS_LOG = room_data.ROOT / "errors.log"


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="room_data_test_"))
    try:
        use_root(tmp)
        src = tmp / "photos"
        src.mkdir()
        demo = ROOT / "frontend" / "public" / "demo" / "rooms"
        shutil.copy(demo / "living-1.jpg", src / "a_living.jpg")
        shutil.copy(demo / "living-1.jpg", src / "b_same_bytes.jpeg")            # duplicate
        shutil.copy(ROOT / "tests" / "fixtures" / "rooms" / "empty" / "photo.png", src / "c_empty.png")
        (src / "d_bytes.webp").write_bytes(b"RIFF....WEBPVP8 fake")                 # kept as-is
        (src / "notes.txt").write_text("not an image")

        first = room_data.import_rooms(room_data.images_in_folder(src))
        ids = [a["id"] for a in first["added"]]
        check("import numbers rooms room_01..", ids == ["room_01", "room_02", "room_03"], ids)
        check("duplicate bytes skipped", [s["file"] for s in first["skipped"]] == ["b_same_bytes.jpeg"], first["skipped"])
        check("jpg stored as original.jpg", (room_data.room_dir("room_01") / "original.jpg").read_bytes()
              == (src / "a_living.jpg").read_bytes())
        check("png keeps its extension and bytes", (room_data.room_dir("room_02") / "original.png").read_bytes()
              == (src / "c_empty.png").read_bytes())
        check("webp keeps its extension", (room_data.room_dir("room_03") / "original.webp").exists())
        check("masks/ and result/ created", all((room_data.room_dir(i) / d).is_dir() for i in ids for d in ("masks", "result")))
        check("_backup/ created", room_data.BACKUP.is_dir())
        again = room_data.import_rooms(room_data.images_in_folder(src))
        check("re-import adds nothing", not again["added"] and len(again["skipped"]) == 4, again)
        index = json.loads(room_data.INDEX.read_text())
        check("index lists 3 rooms, status new", [r["status"] for r in index["rooms"]] == ["new"] * 3, index)

        # ---- profile from a frozen Clean Room job
        segments = ROOT / "tests" / "fixtures" / "rooms" / "living" / "segments"
        profile = room_data.build_profile("room_01", segments, name="Living")
        problems = room_data.validate_profile(profile)
        check("profile from a real job validates", not problems, problems)
        check("one polygon entry per wall", set(profile["wall_polygons"]) == {w["id"] for w in profile["walls"]})
        check("sides only back/left/right/unknown", all(w["side"] in room_data.SIDES for w in profile["walls"]))
        expected_group = room_data.GROUP_BY_WALL_COUNT.get(profile["wall_count"])
        check("auto group follows wall count", profile["group"] == expected_group and profile["group_source"] == "auto",
              (profile["wall_count"], profile["group"]))
        check("floor_corners not invented", profile["floor_corners"] is None)
        check("JSON round-trips", json.loads(json.dumps(profile)) == profile)
        manual = room_data.build_profile("room_01", segments, group="D")
        check("manual group kept", manual["group"] == "D" and manual["group_source"] == "manual")

        # ---- every rule rejects a broken profile
        def broken(mutate):
            p = copy.deepcopy(profile)
            mutate(p)
            return room_data.validate_profile(p)

        cases = {
            "missing status": lambda p: p.pop("status"),
            "bad status": lambda p: p.update(status="done"),
            "bad group": lambda p: p.update(group="Z"),
            "bad side": lambda p: p["walls"][0].update(side="front"),
            "duplicate wall id": lambda p: p["walls"].append(dict(p["walls"][0])),
            "polygon for unknown wall": lambda p: p["wall_polygons"].update({"wall-99": []}),
            "dot outside photo": lambda p: p["walls"][0].update(dot=[-500, 10]),
            "3 floor corners": lambda p: p.update(floor_corners=[[0, 0], [1, 1], [2, 2]]),
            "negative room size": lambda p: p.update(room_size_ft=[10, -1, None]),
            "camera source": lambda p: p["camera"].update(source="guess"),
            "bad date": lambda p: p.update(updated_at="yesterday"),
        }
        for name, mutate in cases.items():
            check(f"rejects: {name}", bool(broken(mutate)))
        check("rejects: not an object", bool(room_data.validate_profile([1, 2])))

        # ---- Step 3: save / backup / broken file / manual camera kept on re-run
        job = tmp / "jobs" / "abc123def456"
        shutil.copytree(segments.parent, job, ignore=shutil.ignore_patterns("photo.*", "fixture.json"))
        room = room_data.load_index()["rooms"][0]
        first_saved = room_data.save_from_job(room, job, {"room_url": "http://h:1/jobs/x"}, "http://h:1")
        check("save_from_job writes profile, masks, preview, job", all(
            (room_data.room_dir("room_01") / rel).exists() for rel in
            ("profile.json", "masks/floor.png", "masks/walls.png", "result/preview.jpg", "job/segment_response.json")))
        check("response stored without the server base", "http://h:1" not in
              (room_data.room_dir("room_01") / "job" / "segment_response.json").read_text())
        manual = json.loads(json.dumps(first_saved))
        manual["camera"].update(source="manual", pitch_deg=0.0)
        manual["status"] = "approved"
        room_data.save_profile(manual)
        check("overwrite backed up the old profile", len(room_data._backups("room_01")) == 1)
        rerun = room_data.save_from_job(room, job, {"room_url": "http://h:1/jobs/y"}, "http://h:1")
        check("re-run keeps a manual camera and its status", rerun["camera"]["source"] == "manual"
              and rerun["camera"]["pitch_deg"] == 0.0 and rerun["status"] == "approved", rerun["camera"])
        check("re-run backed up again", len(room_data._backups("room_01")) == 2)
        (room_data.room_dir("room_01") / "profile.json").write_text("{broken")
        loaded = room_data.load_profile("room_01")
        check("broken profile -> newest valid backup", loaded["source"] == "backup" and loaded["profile"]
              and "restored" in loaded["message"], loaded["message"])
        try:
            room_data.update_profile("room_01", {"status": "bogus"})
            check("invalid edit refused", False)
        except room_data.ProfileError:
            check("invalid edit refused", True)
        sha = room_data.load_index()["rooms"][0]["sha256"]
        restored = room_data.saved_response(sha, tmp / "jobs2", "http://other:9")
        check("saved_response restores the job and rebases URLs",
              restored and restored["room_url"] == "http://other:9/jobs/y" and (tmp / "jobs2").exists() is True)

        # ---- concurrent opens of one room (double click, two tabs, React dev double-start)
        from concurrent.futures import ThreadPoolExecutor
        jobs3 = tmp / "jobs3"
        with ThreadPoolExecutor(8) as pool:
            results = list(pool.map(lambda _: room_data.saved_response(sha, jobs3, "http://c:1"), range(8)))
        check("8 concurrent opens all get the saved result", all(r and r["room_url"] == "http://c:1/jobs/y" for r in results))
        check("job restored once, no staging folders left",
              len([p for p in jobs3.iterdir()]) == 1 and not list(jobs3.glob("*.incoming-*")) and not list(jobs3.glob("*.old-*")))

        # ---- atomic write leaves no temp file behind
        room_data.write_json_atomic(room_data.ROOT / "x.json", {"a": 1})
        check("atomic write, no temp left", not list(room_data.ROOT.glob(".x.json.*.tmp"))
              and json.loads((room_data.ROOT / "x.json").read_text()) == {"a": 1})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{'all passed' if not failures else f'{len(failures)} failed'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
