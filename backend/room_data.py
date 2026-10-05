"""
room-data/: one folder per room, holding everything the app knows about it.

    room-data/
      index.json                 every room: id, name, status, group, wall count
      rooms/
        room_01/
          original.jpg           the input photo, byte for byte (never modified)
          profile.json           camera + walls + polygons + status   (PROFILE FORMAT below)
          masks/                 floor.png, walls.png, objects.png
          result/                preview.jpg (contact-sheet thumbnail)
          job/                   the Clean Room outputs, so reopening never re-runs it
      _backup/                   the previous profile.json of a room, before each overwrite
      errors.log                 batch failures

Steps 1-3: the folder layout, the room import, the profile.json format
(build + validate), and loading/saving (backup before overwrite, atomic
writes, fallback to a backup when a profile is broken, the saved Clean Room
result for a known photo). The batch run (Step 4) comes later. Nothing here detects
anything: a profile is read from what Clean Room already stored.

The folder sits at the project root, found relative to this file.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROOT = PROJECT_ROOT / "room-data"
ROOMS = ROOT / "rooms"
BACKUP = ROOT / "_backup"
INDEX = ROOT / "index.json"
ERRORS_LOG = ROOT / "errors.log"

SCHEMA_VERSION = 1
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".avif")
STATUSES = ("auto", "needs_fix", "approved")
# In index.json only: a room imported but not processed yet (no profile.json).
INDEX_STATUSES = ("new",) + STATUSES
GROUPS = ("A", "B", "C", "D", "E")
# A, B, C follow the wall count; D (high ceiling / stairs) and E (not a room)
# are only ever set by hand.
GROUP_BY_WALL_COUNT = {1: "A", 2: "B", 3: "C"}
SIDES = ("back", "left", "right", "unknown")
# Two walls whose normals are this close are one plane split in two (needs_review).
NEAR_PARALLEL_DEG = 5.0

# Polygon outlines: pieces smaller than this share of the mask are left out,
# and outlines are simplified to this share of the image diagonal.
MIN_PIECE_SHARE = 0.005
SIMPLIFY_SHARE = 0.002


# --------------------------------------------------------------------- utils
def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def digest(data: bytes) -> str:
    """The photo's identity: SHA-256 of its exact bytes (the same rule as reuse.digest)."""
    return hashlib.sha256(data).hexdigest()


def write_json_atomic(path: Path, payload) -> None:
    """Write to a temp file in the same folder, then rename over the target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def room_dir(room_id: str) -> Path:
    return ROOMS / room_id


def original_path(room_id: str) -> Path | None:
    folder = room_dir(room_id)
    for suffix in IMAGE_SUFFIXES:
        candidate = folder / f"original{suffix}"
        if candidate.exists():
            return candidate
    return None


# ----------------------------------------------------------------- the index
def load_index() -> dict:
    if not INDEX.exists():
        return {"schema_version": SCHEMA_VERSION, "updated_at": None, "rooms": []}
    return json.loads(INDEX.read_text("utf-8"))


def save_index(index: dict) -> None:
    index["schema_version"] = SCHEMA_VERSION
    index["updated_at"] = now_iso()
    write_json_atomic(INDEX, index)


def _next_room_id(index: dict) -> str:
    taken = [int(m.group(1)) for r in index["rooms"] if (m := re.fullmatch(r"room_(\d+)", r["id"]))]
    return f"room_{(max(taken) + 1 if taken else 1):02d}"


def ensure_layout() -> None:
    """Create room-data/, rooms/ and _backup/, and an empty index.json if missing."""
    ROOMS.mkdir(parents=True, exist_ok=True)
    BACKUP.mkdir(parents=True, exist_ok=True)
    if not INDEX.exists():
        save_index({"rooms": []})


def sniff_format(data: bytes) -> str | None:
    """The real image format from the file's first bytes (magic number), or None."""
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    if data[4:8] == b"ftyp":
        # ISO-BMFF: the major brand, then the compatible brands, up to the box size.
        size = int.from_bytes(data[:4], "big")
        brands = {data[i:i + 4] for i in range(8, min(max(size, 16), len(data)), 4) if i != 12}
        if brands & {b"avif", b"avis"}:
            return ".avif"
    return None


def import_rooms(images: list[tuple[Path, str]]) -> dict:
    """
    Add photos as new rooms (room_01, room_02, ...), skipping exact duplicates.

    `images` is [(path, display name)]. A photo whose SHA-256 is already in the
    index is skipped. The bytes are copied unchanged; the stored name is
    original.<real format>, read from the file header rather than the file
    name (a ".jpeg" that is really AVIF becomes original.avif). A file that is
    not jpg / png / webp / avif inside is skipped.
    """
    ensure_layout()
    index = load_index()
    known = {r.get("sha256"): r["id"] for r in index["rooms"]}
    added, skipped = [], []
    for path, name in images:
        data = path.read_bytes()
        sha = digest(data)
        if sha in known:
            skipped.append({"file": path.name, "reason": f"duplicate of {known[sha]}"})
            continue
        suffix = sniff_format(data)
        if suffix is None:
            skipped.append({"file": path.name, "reason": "not a jpg, png, webp or avif image (file header)"})
            continue
        room_id = _next_room_id(index)
        folder = room_dir(room_id)
        (folder / "masks").mkdir(parents=True, exist_ok=True)
        (folder / "result").mkdir(parents=True, exist_ok=True)
        (folder / f"original{suffix}").write_bytes(data)
        entry = {"id": room_id, "name": name, "status": "new", "group": None, "wall_count": None,
                 "needs_review": None, "original": f"original{suffix}", "sha256": sha,
                 "source_file": path.name}
        index["rooms"].append(entry)
        known[sha] = room_id
        added.append(entry)
    save_index(index)
    return {"added": added, "skipped": skipped}


def images_in_folder(folder: Path) -> list[tuple[Path, str]]:
    """Every jpg/png/webp/avif in `folder` (not recursive), sorted by name; name = file stem."""
    return [(p, p.stem) for p in sorted(Path(folder).iterdir())
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES and not p.name.startswith(".")]


def demo_rooms() -> list[tuple[Path, str]]:
    """The demo rooms of frontend/public/demo/catalogue.json, with their catalogue names."""
    public = PROJECT_ROOT / "frontend" / "public"
    catalogue = json.loads((public / "demo" / "catalogue.json").read_text("utf-8"))
    return [(public / room["src"].lstrip("/"), room["name"]) for room in catalogue.get("rooms", [])]


# ------------------------------------------------------- profile.json format
def _outlines(mask: np.ndarray) -> list[list[list[int]]]:
    """A mask's outer outline(s) as [[x, y], ...]; a wall split by objects has several."""
    m = mask.astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    total = max(int(mask.sum()), 1)
    eps = SIMPLIFY_SHARE * float(np.hypot(*mask.shape))
    out = []
    for c in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(c) < MIN_PIECE_SHARE * total:
            continue
        poly = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(poly) >= 3:
            out.append([[int(x), int(y)] for x, y in poly])
    return out


def _read_mask(path: Path) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return None if image is None else image > 127


def _side(frame_wall: dict | None, direction: dict | None) -> str:
    """
    back / left / right only when the stored geometry says so unambiguously:

      back    room frame class "back" AND the wall's own direction faces front
      left    room frame class "side" with a placed plane left of the camera  (x < 0)
      right   room frame class "side" with a placed plane right of the camera (x > 0)

    Anything else -- oblique, no direction, the two sources disagreeing, a side
    wall that could not be placed -- is "unknown". Names are never guessed.
    """
    cls = (frame_wall or {}).get("class")
    faces = (direction or {}).get("faces")
    if cls == "back":
        return "back" if faces == "front" else "unknown"
    if cls == "side" and faces != "front" and (frame_wall or {}).get("x") is not None:
        return "left" if frame_wall["x"] < 0 else "right" if frame_wall["x"] > 0 else "unknown"
    return "unknown"


def _near_parallel_pairs(walls_doc: dict, max_deg: float) -> list[list[str]]:
    """Pairs of walls whose stored plane normals are within `max_deg` of each other."""
    normals = []
    for w in walls_doc.get("walls", []):
        n = (w.get("direction") or {}).get("normal")
        if n is not None and np.linalg.norm(n) > 0:
            normals.append((w["id"], np.asarray(n, float) / np.linalg.norm(n)))
    pairs = []
    for i in range(len(normals)):
        for j in range(i + 1, len(normals)):
            cos = abs(float(normals[i][1] @ normals[j][1]))
            if np.degrees(np.arccos(min(1.0, cos))) <= max_deg:
                pairs.append([normals[i][0], normals[j][0]])
    return pairs


def dot_problems(profile: dict) -> list[str]:
    """Every wall whose dot is not inside that wall's own outline (within the outline's simplification)."""
    canvas = profile.get("canvas") or [0, 0]
    tolerance = SIMPLIFY_SHARE * float(np.hypot(*canvas)) + 1.0
    problems = []
    for w in profile.get("walls") or []:
        if not isinstance(w, dict) or not w.get("dot"):
            continue
        polys = (profile.get("wall_polygons") or {}).get(w.get("id")) or []
        x, y = float(w["dot"][0]), float(w["dot"][1])
        inside = any(cv2.pointPolygonTest(np.asarray(poly, np.float32).reshape(-1, 1, 2), (x, y), True) >= -tolerance
                     for poly in polys if len(poly) >= 3)
        if not inside:
            problems.append(f"{w.get('id')}: dot {w['dot']} is not inside its own wall")
    return problems


def build_profile(room_id: str, segments: Path, *, name: str | None = None, status: str = "auto",
                  group: str | None = None, room_size_ft=None, job_id: str | None = None) -> dict:
    """
    profile.json for one room, read from a finished Clean Room job's `segments/`
    (the geometry it stored; nothing is re-estimated).

    `group` None means automatic: A / B / C from the wall count, or None when the
    count is 0 or more than 3 (left for the contact sheet). `room_size_ft` is
    [width, length, height] in feet; None per value = not entered (AUTO).

    needs_review is true when the detection looks unreliable: more than 3
    walls, or 2+ walls whose normals are within NEAR_PARALLEL_DEG (one real
    plane split in two). The automatic group is then marked "auto-unreliable".
    A wall dot that is not inside its own wall makes an automatic profile
    "needs_fix" (it is reported, never moved).
    """
    segments = Path(segments)
    floor = json.loads((segments / "floor" / "floor_geometry.json").read_text("utf-8"))
    walls_doc = json.loads((segments / "wall" / "wall_geometry.json").read_text("utf-8"))
    frame_path = segments / "room" / "room_frame.json"
    frame = json.loads(frame_path.read_text("utf-8")) if frame_path.exists() else {}
    frame_walls = frame.get("walls") if isinstance(frame.get("walls"), dict) else {}

    vertical = frame.get("vertical_vp") or {}
    pitch_lines = vertical.get("pitch_deg") if vertical.get("reliable") else None
    roll_lines = vertical.get("roll_deg") if vertical.get("reliable") else None
    pitch_floor = (floor.get("camera") or {}).get("pitch_deg") if floor.get("status") == "detected" else None
    # The floor tiles are placed from the floor plane, so that pitch is the one in use.
    if pitch_floor is not None:
        pitch, pitch_source = pitch_floor, "floor plane"
    elif pitch_lines is not None:
        pitch, pitch_source = pitch_lines, "vertical lines"
    else:
        pitch, pitch_source = None, None
    focal_px = floor.get("focal_px")
    canvas = floor.get("canvas")

    walls, wall_polygons = [], {}
    for w in sorted(walls_doc.get("walls", []), key=lambda item: item["index"]):
        mask = _read_mask(segments / "wall" / "walls" / f"{w['id']}.png")
        walls.append({
            "id": w["id"],
            "side": _side(frame_walls.get(w["id"]), w.get("direction")),
            "pixels": int(w.get("pixels", 0)),
            "dot": [int(v) for v in w.get("select_point", [])],
        })
        wall_polygons[w["id"]] = _outlines(mask) if mask is not None else []

    floor_mask = _read_mask(segments / "floor" / "floor_mask.png")
    count = len(walls)
    # The split's own count before the corner cut (corner_cut.py), when it ran.
    cut = (walls_doc.get("wall_split") or {}).get("corner_cut") or {}
    plane_merge_before = ((walls_doc.get("wall_split") or {}).get("plane_merge") or {}).get("walls_before")
    count_raw = int(cut.get("walls_before", plane_merge_before or count))
    parallel = _near_parallel_pairs(walls_doc, NEAR_PARALLEL_DEG)
    reasons = ([f"{count} walls detected (more than 3)"] if count > 3 else []) + [
        f"{a} and {b} have normals within {NEAR_PARALLEL_DEG:g} deg" for a, b in parallel]
    # Not 95% sure of a corner: never guessed, reported (and the room needs a fix).
    corner_doubts = [f"corner at x={u['x']} seen in the {u['seen_in']} (no second witness)"
                     for u in cut.get("uncertain_corners", [])]
    if cut.get("slots_with_two_directions"):
        corner_doubts.append("two walls with no confirmed corner between them")
    plane_merge = (walls_doc.get("wall_split") or {}).get("plane_merge") or {}
    if plane_merge.get("uncertain"):
        corner_doubts.append("a pillar was merged into its wall (its edges lack two witnesses)")
    reasons += corner_doubts
    needs_review = bool(reasons)
    auto_group = GROUP_BY_WALL_COUNT.get(count)
    rnd = lambda v: None if v is None else round(float(v), 2)  # noqa: E731

    profile = {
        "schema_version": SCHEMA_VERSION,
        "room_id": room_id,
        "name": name,
        "group": group if group is not None else auto_group,
        "group_source": "manual" if group is not None else ("auto-unreliable" if needs_review else "auto"),
        # The split before the corner cut, and the walls after it (corner_cut.py).
        "wall_count_raw": count_raw,
        "wall_count": count,
        "needs_review": needs_review,
        "review_reasons": reasons,
        "walls": walls,
        "camera": {
            "pitch_deg": rnd(pitch),
            "pitch_source": pitch_source,
            "pitch_vertical_lines_deg": rnd(pitch_lines),
            "pitch_floor_plane_deg": rnd(pitch_floor),
            "roll_deg": rnd(roll_lines),
            "focal_px": None if focal_px is None else round(float(focal_px), 1),
            # A real focal length in mm only exists when the photo's EXIF had one.
            "focal_mm": None,
            "focal_source": floor.get("focal_source"),
            # The joint camera: "low" when the focal came from the lens prior or the
            # camera is not verified level (absent in geometry stored before it existed).
            "confidence": floor.get("confidence"),
            "principal_point": floor.get("principal_point"),
            "source": "auto",
        },
        # Set by the corner editor (Step 6); none are detected automatically.
        "floor_corners": None,
        "floor_polygons": _outlines(floor_mask) if floor_mask is not None else [],
        "wall_polygons": wall_polygons,
        "room_size_ft": list(room_size_ft) if room_size_ft is not None else [None, None, None],
        "canvas": canvas,
        "job_id": job_id,
        "status": status,
        "updated_at": now_iso(),
    }
    dots = dot_problems(profile)
    if dots:
        profile["review_reasons"] += dots
    if (dots or corner_doubts) and status == "auto":
        profile["status"] = "needs_fix"
    return profile


def _is_point(value, canvas) -> bool:
    if not (isinstance(value, list) and len(value) == 2 and all(isinstance(v, (int, float)) for v in value)):
        return False
    if canvas:
        return -1 <= value[0] <= canvas[0] + 1 and -1 <= value[1] <= canvas[1] + 1
    return True


def validate_profile(profile) -> list[str]:
    """
    Every problem with a profile (empty list = valid).

    A wall dot outside its own wall is an error, except in a profile already
    marked "needs_fix" (that is exactly what the status records).
    """
    if not isinstance(profile, dict):
        return ["profile is not a JSON object"]
    errors = []
    required = ("schema_version", "room_id", "group", "group_source", "wall_count_raw", "needs_review",
                "walls", "camera", "floor_corners", "wall_polygons", "room_size_ft", "status", "updated_at")
    errors += [f"missing {key}" for key in required if key not in profile]
    if errors:
        return errors
    canvas = profile.get("canvas")
    if profile["status"] not in STATUSES:
        errors.append(f"status {profile['status']!r} not in {STATUSES}")
    if profile["group"] is not None and profile["group"] not in GROUPS:
        errors.append(f"group {profile['group']!r} not in {GROUPS}")
    if profile["group_source"] not in ("auto", "auto-unreliable", "manual"):
        errors.append(f"group_source {profile['group_source']!r} not auto / auto-unreliable / manual")
    if not isinstance(profile["needs_review"], bool):
        errors.append("needs_review must be true or false")
    walls = profile["walls"]
    if not isinstance(walls, list) or len(walls) > 3 and profile["status"] == "approved":
        errors.append("walls must be a list (an approved room has at most 3)")
    else:
        ids = [w.get("id") for w in walls if isinstance(w, dict)]
        if len(ids) != len(walls) or len(set(ids)) != len(ids):
            errors.append("every wall needs a unique id")
        for w in walls:
            if isinstance(w, dict) and w.get("side") not in SIDES:
                errors.append(f"{w.get('id')}: side {w.get('side')!r} not in {SIDES}")
            if isinstance(w, dict) and w.get("dot") and not _is_point(w["dot"], canvas):
                errors.append(f"{w.get('id')}: dot {w.get('dot')} is outside the photo")
        if set(ids) != set(profile["wall_polygons"] or {}):
            errors.append("wall_polygons must have exactly one entry per wall id")
        elif profile["status"] != "needs_fix":
            errors += dot_problems(profile)
    for wid, polys in (profile["wall_polygons"] or {}).items():
        for poly in polys:
            if not all(_is_point(p, canvas) for p in poly):
                errors.append(f"{wid}: polygon point outside the photo")
                break
    corners = profile["floor_corners"]
    if corners is not None and not (isinstance(corners, list) and len(corners) == 4
                                    and all(_is_point(p, canvas) for p in corners)):
        errors.append("floor_corners must be null or 4 [x, y] points inside the photo")
    size = profile["room_size_ft"]
    if not (isinstance(size, list) and len(size) == 3
            and all(v is None or (isinstance(v, (int, float)) and v > 0) for v in size)):
        errors.append("room_size_ft must be [w, l, h] with positive numbers or null")
    camera = profile["camera"]
    if not isinstance(camera, dict) or camera.get("source") not in ("auto", "manual"):
        errors.append("camera.source must be auto or manual")
    try:
        datetime.fromisoformat(str(profile["updated_at"]))
    except ValueError:
        errors.append("updated_at is not an ISO date")
    return errors


# ------------------------------------------------- Step 3: load / save logic
#: Previous profile.json versions kept per room in _backup/<room>/.
BACKUPS_KEPT = 10
#: Saved /segment response: absolute URLs are stored with this in place of the server's base.
BASE_TOKEN = "{BASE}"
RESPONSE_FILE = "segment_response.json"
PREVIEW_EDGE = 360


class ProfileError(ValueError):
    """A profile that does not validate; never written."""


def find_room(sha: str) -> dict | None:
    """The index entry of the room whose original has this SHA-256, or None."""
    if not INDEX.exists():
        return None
    return next((r for r in load_index()["rooms"] if r.get("sha256") == sha), None)


def _backup_dir(room_id: str) -> Path:
    return BACKUP / room_id


def _backups(room_id: str) -> list[Path]:
    """This room's backed-up profiles, newest first."""
    folder = _backup_dir(room_id)
    return sorted(folder.glob("profile-*.json"), reverse=True) if folder.exists() else []


def _backup_current(room_id: str) -> Path | None:
    """Copy the current profile.json to _backup/<room>/profile-<time>.json; keep the newest BACKUPS_KEPT."""
    current = room_dir(room_id) / "profile.json"
    if not current.exists():
        return None
    folder = _backup_dir(room_id)
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    target = folder / f"profile-{stamp}.json"
    target.write_bytes(current.read_bytes())
    for old in _backups(room_id)[BACKUPS_KEPT:]:
        old.unlink(missing_ok=True)
    return target


def _update_index(profile: dict) -> None:
    index = load_index()
    for entry in index["rooms"]:
        if entry["id"] == profile["room_id"]:
            entry.update(status=profile["status"], group=profile["group"], wall_count=profile["wall_count"],
                         needs_review=profile["needs_review"], updated_at=profile["updated_at"])
            break
    else:
        raise ProfileError(f"{profile['room_id']} is not in index.json")
    save_index(index)


def save_profile(profile: dict) -> dict:
    """
    Validate, back up the old profile.json, write the new one atomically, update index.json.
    An invalid profile raises ProfileError and nothing is written.
    """
    problems = validate_profile(profile)
    if problems:
        raise ProfileError("; ".join(problems))
    _backup_current(profile["room_id"])
    write_json_atomic(room_dir(profile["room_id"]) / "profile.json", profile)
    _update_index(profile)
    return profile


def load_profile(room_id: str) -> dict:
    """
    {"profile": dict | None, "source": "file" | "backup" | "none", "message": str | None}

    A profile.json that is missing nothing but fails to parse or validate is
    replaced by the newest valid backup (the broken file is kept beside the
    backups as broken-<time>.json), and the message says so.
    """
    path = room_dir(room_id) / "profile.json"
    if not path.exists():
        return {"profile": None, "source": "none", "message": None}
    try:
        profile = json.loads(path.read_text("utf-8"))
        problems = validate_profile(profile)
    except (ValueError, UnicodeDecodeError) as error:
        profile, problems = None, [f"not valid JSON ({error})"]
    if not problems:
        return {"profile": profile, "source": "file", "message": None}

    reason = "; ".join(problems)
    for backup in _backups(room_id):
        try:
            candidate = json.loads(backup.read_text("utf-8"))
        except (ValueError, UnicodeDecodeError):
            continue
        if not validate_profile(candidate):
            folder = _backup_dir(room_id)
            folder.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            (folder / f"broken-{stamp}.json").write_bytes(path.read_bytes())
            write_json_atomic(path, candidate)
            _update_index(candidate)
            return {"profile": candidate, "source": "backup",
                    "message": f"{room_id}/profile.json was broken ({reason}); restored the backup "
                               f"{backup.name}. The broken file is kept as {folder.name}/broken-{stamp}.json."}
    return {"profile": None, "source": "none",
            "message": f"{room_id}/profile.json is broken ({reason}) and no valid backup exists."}


def _preview(original: Path, floor: Path, wall: Path, target: Path) -> None:
    """A small JPEG of the photo with the floor (orange) and walls (blue) lightly tinted."""
    from PIL import Image, ImageOps

    image = ImageOps.exif_transpose(Image.open(original)).convert("RGB")
    rgb = np.asarray(image, dtype=np.float32)
    h, w = rgb.shape[:2]
    for path, colour in ((floor, (255, 150, 40)), (wall, (40, 120, 255))):
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST) > 127
        rgb[mask] = rgb[mask] * 0.65 + np.array(colour, np.float32) * 0.35
    out = Image.fromarray(rgb.clip(0, 255).astype(np.uint8))
    out.thumbnail((PREVIEW_EDGE, PREVIEW_EDGE))
    target.parent.mkdir(parents=True, exist_ok=True)
    out.save(target, "JPEG", quality=85)


# One lock per room: two opens of the same room at once (a double click, two
# tabs, React's development double-start) must not restore or replace its job
# at the same time.
_ROOM_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _room_lock(room_id: str) -> threading.Lock:
    with _LOCKS_GUARD:
        return _ROOM_LOCKS.setdefault(room_id, threading.Lock())


def _copy_dir(source: Path, target: Path) -> None:
    """Replace `target` with a copy of `source`, never leaving a half-copied folder in place."""
    import shutil

    tag = uuid.uuid4().hex[:8]
    staging = target.with_name(f"{target.name}.incoming-{tag}")
    shutil.rmtree(staging, ignore_errors=True)
    shutil.copytree(source, staging)
    old = target.with_name(f"{target.name}.old-{tag}")
    shutil.rmtree(old, ignore_errors=True)
    if target.exists():
        target.rename(old)
    staging.rename(target)
    shutil.rmtree(old, ignore_errors=True)


def save_from_job(room: dict, job_dir: Path, response: dict, base: str) -> dict:
    """
    After a Clean Room run on this room's photo: keep the job, the masks, a
    preview, the response and a fresh automatic profile.

    A manual group (D / E) and the entered room size carry over from the
    previous profile; everything detected is new (status "auto", or
    "needs_fix" when a dot is outside its wall). The old profile goes to _backup/.
    """
    with _room_lock(room["id"]):
        return _save_from_job(room, job_dir, response, base)


def _save_from_job(room: dict, job_dir: Path, response: dict, base: str) -> dict:
    room_id = room["id"]
    folder = room_dir(room_id)
    job_dir = Path(job_dir)
    segments = job_dir / "segments"
    _copy_dir(job_dir, folder / "job")
    stored = json.loads(json.dumps(response).replace(base, BASE_TOKEN))
    write_json_atomic(folder / "job" / RESPONSE_FILE, stored)
    masks = folder / "masks"
    masks.mkdir(parents=True, exist_ok=True)
    import shutil

    for name, target in (("FLOOR_MASK.png", "floor.png"), ("WALL_MASK.png", "walls.png"),
                         ("ALL_OBJECTS.png", "objects.png")):
        if (segments / name).exists():
            shutil.copyfile(segments / name, masks / target)
    _preview(folder / room["original"], masks / "floor.png", masks / "walls.png", folder / "result" / "preview.jpg")

    previous = load_profile(room_id)["profile"] or {}
    manual_group = previous.get("group") if previous.get("group_source") == "manual" else None
    profile = build_profile(room_id, segments, name=room.get("name"), group=manual_group,
                            room_size_ft=previous.get("room_size_ft"), job_id=job_dir.name)
    # A hand-set camera (corner editor) is never replaced by automatic values:
    # its camera, floor corners and status carry over; the detection is saved beside it.
    if (previous.get("camera") or {}).get("source") == "manual":
        profile.update(camera=previous["camera"], floor_corners=previous.get("floor_corners"),
                       status=previous["status"])
    return save_profile(profile)


def saved_response(sha: str, jobs_dir: Path, base: str) -> dict | None:
    """
    The Clean Room result saved for this photo, ready to return from /segment
    -- or None when there is none (no room, no valid profile, no job).

    The job is put back under `jobs_dir` with its original id if it is not
    there, so every /jobs URL, /surfaces and /generate work exactly as after
    the original run. Nothing is detected.
    """
    room = find_room(sha)
    if room is None:
        return None
    with _room_lock(room["id"]):
        return _saved_response(room, jobs_dir, base)


def _saved_response(room: dict, jobs_dir: Path, base: str) -> dict | None:
    loaded = load_profile(room["id"])
    profile = loaded["profile"]
    job = room_dir(room["id"]) / "job"
    if profile is None or not profile.get("job_id") or not (job / RESPONSE_FILE).exists():
        return None
    target = Path(jobs_dir) / profile["job_id"]
    if not target.exists():
        _copy_dir(job, target)
    response = json.loads((job / RESPONSE_FILE).read_text("utf-8").replace(BASE_TOKEN, base))
    response["room_data"] = {"room_id": room["id"], "source": loaded["source"], "message": loaded["message"],
                             "status": profile["status"]}
    return response


def update_profile(room_id: str, changes: dict) -> dict:
    """
    A manual edit: only status, group, room_size_ft, floor_corners and wall
    sides can change. A group set here becomes group_source "manual".
    """
    allowed = {"status", "group", "room_size_ft", "floor_corners", "sides"}
    unknown = set(changes) - allowed
    if unknown:
        raise ProfileError(f"cannot edit {sorted(unknown)}; editable: {sorted(allowed)}")
    profile = load_profile(room_id)["profile"]
    if profile is None:
        raise ProfileError(f"{room_id} has no profile yet")
    profile = json.loads(json.dumps(profile))
    if "group" in changes:
        profile["group"], profile["group_source"] = changes["group"], "manual"
    for key in ("status", "room_size_ft", "floor_corners"):
        if key in changes:
            profile[key] = changes[key]
    for wall in profile["walls"]:
        if wall["id"] in (changes.get("sides") or {}):
            wall["side"] = changes["sides"][wall["id"]]
    profile["updated_at"] = now_iso()
    return save_profile(profile)


if __name__ == "__main__":
    # python backend/room_data.py import-demo           -> the 15 demo rooms
    # python backend/room_data.py import-folder <folder> -> every jpg/png/webp/avif in it
    import sys

    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "import-demo":
        print(json.dumps(import_rooms(demo_rooms()), indent=1))
    elif command == "import-folder" and len(sys.argv) > 2:
        print(json.dumps(import_rooms(images_in_folder(Path(sys.argv[2]))), indent=1))
    else:
        print(__doc__)
