"""
A cleaned room's floor and wall geometry, on disk beside its masks.

    segments/
      floor/
        floor_mask.png               FLOOR_MASK.png — floor white, all else black
        floor_geometry.json          camera, floor plane, pitch, direction, status
        floor_vanishing_points.json  the floor's vanishing points and horizon
      wall/
        wall_mask.png                WALL_MASK.png — every wall white
        wall_geometry.json           camera, and per wall: region, orientation
        wall_vanishing_points.json   room-level and per-wall vanishing points
        walls/wall-<i>.png           each wall on its own

Written once, when the room is cleaned, by `write`; read by `load` for every
tile render of that room, so the detected geometry — not a fresh detection per
render — is what the tiles are projected with. The masks are FLOOR_MASK.png and
WALL_MASK.png unchanged, at the clean room's own resolution; the geometry is
measured at the render resolution and says so in `canvas`.
"""

from __future__ import annotations

import ast
import filecmp
import functools
import hashlib
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from floor_wall import FLOOR_MASK_FILENAME, WALL_MASK_FILENAME
from scene import PROJECT_ROOT
from tiles_backend.perspective_engine.engine import GEOMETRY_VERSION

FLOOR_DIR = "floor"
WALL_DIR = "wall"
ROOM_DIR = "room"

# What the stored geometry was detected from, beside it rather than inside it,
# so a room whose geometry comes out the same keeps its geometry files
# byte-identical. Geometry that no longer matches is moved, never overwritten.
PROVENANCE_FILENAME = "geometry_provenance.json"
SUPERSEDED_DIR = "geometry_superseded"

# The code that detects geometry. Any change to what it does invalidates every
# stored geometry, because GEOMETRY_VERSION alone was not bumped when the floor
# vanishing point logic changed. Hashed by syntax tree, so comments and
# docstrings do not count as a change.
SOURCE_ROOTS = (PROJECT_ROOT / "tiles_backend" / "perspective_engine", Path(__file__).resolve().parent)


def _without_docstrings(tree: ast.AST) -> ast.AST:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                node.body = node.body[1:] or [ast.Pass()]
    return tree


@functools.lru_cache(maxsize=1)
def code_hash() -> str:
    digest = hashlib.sha256()
    for root in SOURCE_ROOTS:
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            digest.update(str(path.relative_to(root)).encode())
            digest.update(ast.dump(_without_docstrings(ast.parse(path.read_text("utf-8")))).encode())
    return digest.hexdigest()


def provenance(floor: np.ndarray, wall: np.ndarray, objects, image: np.ndarray) -> dict:
    """
    The detection's inputs and code, hashed. The photo bytes (EXIF focal) are
    left out: not every caller has them, and the photo of a job never changes.
    """
    digest = hashlib.sha256()
    for array in (floor, wall, objects, image):
        if array is None:
            digest.update(b"none")
            continue
        array = np.ascontiguousarray(array)
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(np.packbits(array).tobytes() if array.dtype == bool else array.tobytes())
    return {"geometry_version": GEOMETRY_VERSION, "code_sha256": code_hash(), "inputs_sha256": digest.hexdigest()}


def is_current(segments: Path, expected: dict) -> tuple[bool, str]:
    """Whether the stored geometry was detected from `expected`, and why not."""
    stored = _optional_json(segments / PROVENANCE_FILENAME)
    if stored is None:
        return False, "no provenance stored (geometry from before this check)"
    changed = [k for k in ("geometry_version", "code_sha256", "inputs_sha256") if stored.get(k) != expected[k]]
    if changed:
        return False, "changed: " + ", ".join(changed)
    return True, "current"


def write_provenance(segments: Path, expected: dict, reason: str) -> None:
    _json(segments / PROVENANCE_FILENAME, {
        **expected, "reason": reason, "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })


def same_as_stored(segments: Path, geometry: dict, wall_masks: dict) -> bool:
    """Whether writing `geometry` would reproduce the stored files byte for byte."""
    with tempfile.TemporaryDirectory() as tmp:
        probe = Path(tmp)
        for name in (FLOOR_MASK_FILENAME, WALL_MASK_FILENAME):
            shutil.copyfile(segments / name, probe / name)
        write(probe, geometry, wall_masks)
        for sub in (FLOOR_DIR, WALL_DIR, ROOM_DIR):
            new, old = probe / sub, segments / sub
            if new.exists() != old.exists():
                return False
            if not new.exists():
                continue
            names = sorted(p.relative_to(new) for p in new.rglob("*") if p.is_file())
            if names != sorted(p.relative_to(old) for p in old.rglob("*") if p.is_file()):
                return False
            if any(not filecmp.cmp(new / n, old / n, shallow=False) for n in names):
                return False
    return True


def supersede(segments: Path, reason: str) -> Path:
    """Move the stored geometry into a new dated folder, keeping it."""
    target = segments / SUPERSEDED_DIR / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    target.mkdir(parents=True)
    for name in (FLOOR_DIR, WALL_DIR, ROOM_DIR, PROVENANCE_FILENAME):
        if (segments / name).exists():
            shutil.move(str(segments / name), str(target / name))
    _json(target / "superseded.json", {"reason": reason})
    return target


def _json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write(segments: Path, geometry: dict, wall_masks: dict[int, np.ndarray]) -> None:
    """Write the floor/ and wall/ folders for one cleaned room."""
    floor_dir = segments / FLOOR_DIR
    wall_dir = segments / WALL_DIR
    (wall_dir / "walls").mkdir(parents=True, exist_ok=True)
    floor_dir.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(segments / FLOOR_MASK_FILENAME, floor_dir / "floor_mask.png")
    shutil.copyfile(segments / WALL_MASK_FILENAME, wall_dir / "wall_mask.png")

    camera = {"canvas": geometry["canvas"], **geometry["camera"]}
    floor = geometry["floor"]

    _json(floor_dir / "floor_geometry.json", {
        **camera,
        **{k: v for k, v in floor.items() if k != "vanishing_points"},
    })
    _json(floor_dir / "floor_vanishing_points.json", {
        "canvas": geometry["canvas"],
        "principal_point": geometry["camera"]["principal_point"],
        "focal_px": geometry["camera"]["focal_px"],
        **floor["vanishing_points"],
    })

    _json(wall_dir / "wall_geometry.json", {
        **camera,
        "version": geometry.get("version", 1),
        "wall_split": geometry.get("wall_split"),
        "walls": geometry["walls"],
    })
    _json(wall_dir / "wall_vanishing_points.json", {
        "canvas": geometry["canvas"],
        "principal_point": geometry["camera"]["principal_point"],
        "focal_px": geometry["camera"]["focal_px"],
        "room_horizontal": geometry["room_vps"],
        "walls": [
            {
                "id": w["id"],
                "vanishing_point": (w["direction"] or {}).get("vanishing_point"),
                "source": w["orientation_source"],
            }
            for w in geometry["walls"]
        ],
    })

    # The canonical room frame (image-only measurements, camera-height units).
    if geometry.get("room_frame") is not None:
        (segments / "room").mkdir(exist_ok=True)
        _json(segments / "room" / "room_frame.json", geometry["room_frame"])

    # Each wall on its own, at the clean room's resolution like the masks.
    full = np.asarray(Image.open(segments / WALL_MASK_FILENAME).convert("L")).shape

    for index, mask in wall_masks.items():
        white = mask.astype(np.uint8) * 255
        if white.shape != full:
            white = ((cv2.resize(white.astype(np.float32), (full[1], full[0]),
                                 interpolation=cv2.INTER_LINEAR) >= 127.5) * 255).astype(np.uint8)
        Image.fromarray(white, mode="L").save(wall_dir / "walls" / f"wall-{index}.png")


def _optional_json(path: Path):
    try:
        return json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        return None


def load(segments: Path) -> dict | None:
    """
    The geometry `write` stored, reassembled into the dict the tile engine
    takes, or None when this room has none yet.
    """
    try:
        floor = json.loads((segments / FLOOR_DIR / "floor_geometry.json").read_text("utf-8"))
        floor_vps = json.loads((segments / FLOOR_DIR / "floor_vanishing_points.json").read_text("utf-8"))
        walls = json.loads((segments / WALL_DIR / "wall_geometry.json").read_text("utf-8"))
        wall_vps = json.loads((segments / WALL_DIR / "wall_vanishing_points.json").read_text("utf-8"))
    except (OSError, ValueError):
        return None

    # Geometry from before the stored wall split is detected again.
    if walls.get("version", 1) < GEOMETRY_VERSION:
        return None

    wall_masks = {}
    for w in walls["walls"]:
        path = segments / WALL_DIR / "walls" / f"{w['id']}.png"
        if not path.exists():
            return None
        wall_masks[int(w["index"])] = np.asarray(Image.open(path).convert("L")) > 127

    camera_keys = ("focal_px", "focal_source", "principal_point")

    floor_part = {k: v for k, v in floor.items() if k not in camera_keys}
    floor_part["vanishing_points"] = {
        k: v for k, v in floor_vps.items()
        if k not in ("canvas", "principal_point", "focal_px")
    }

    return {
        "canvas": floor["canvas"],
        # The joint camera's report and confidence (geometry stored before it existed has neither).
        "camera": {**{k: floor[k] for k in camera_keys},
                   **{k: floor[k] for k in ("joint", "confidence") if k in floor}},
        "floor": floor_part,
        "walls": walls["walls"],
        "room_vps": wall_vps.get("room_horizontal", {}),
        "version": walls.get("version", 1),
        "room_frame": _optional_json(segments / "room" / "room_frame.json"),
        # In memory only (never written back as JSON): the stored split the
        # tile engine renders instead of splitting the walls again.
        "_wall_masks": wall_masks,
    }


def exists(segments: Path) -> bool:
    return (segments / FLOOR_DIR / "floor_geometry.json").exists() and (
        segments / WALL_DIR / "wall_geometry.json"
    ).exists()
