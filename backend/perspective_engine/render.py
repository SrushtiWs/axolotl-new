"""
Floor/wall masks in, tiled room out — the backend's connection to the tile engine.

    room + clean room + FLOOR_MASK + WALL_MASK + tile + dimensions
      -> estimated camera    live_scene.build          (only for its horizon)
      -> tile engine         tiles_backend.perspective_engine.render_room
                               the reference project's floor and wall
                               pipelines, clipped to each surface's mask
      -> objects back        engine._restore_props     (from the original photo)
      -> strict clip         compositing.clip          (no tile outside its mask)
      -> final tiled room

The tile placement — vanishing points, focal length, plane, metric scale,
rotation, anchoring, the millimetre grid, cut tiles, grout, lighting — is the
migrated engine in `tiles_backend/perspective_engine/`, used unchanged. This
module only adapts the backend's inputs to it and its output back to the
`engine.Result` shape `/generate` already writes to disk.

What the backend contributes, and why
-------------------------------------

* **The masks**, as the strict boundary. The engine is handed FLOOR_MASK and
  WALL_MASK verbatim and clips to them.

* **A fallback vanishing point.** The engine takes a caller's manual vanishing
  point as its fallback when its own detection is rejected, which happens
  often; without one it fits the floor plane to depth, and on one measured room
  a 600 mm tile came out over a metre across. `live_scene` already estimates
  this room's camera from the floor's far edge and the room length you entered,
  so its horizon is handed over as that fallback. An accepted detection still
  wins.

* **The objects**, restored over the tiles from the original photograph
  through their soft alpha, exactly as on every other path.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
from dataclasses import dataclass, field, replace

import cv2
import numpy as np

import engine
import live_scene
from engine import Result, TileSpec
from scene import PROJECT_ROOT, Scene
from surfaces import Instance

from perspective_engine import compositing, masks

# tiles_backend/ lives at the project root, beside backend/.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tiles_backend.perspective_engine import (  # noqa: E402
    TileRequest,
    detect_room_geometry,
    detect_surfaces,
    interior_point,
    render_room,
)

from perspective_engine import geometry as geometry_store  # noqa: E402



@dataclass
class Rendered:
    """One render, with the scene and geometry that produced it."""

    result: Result
    scene: Scene
    geometry: dict
    clip: dict
    # Each surface's tiled pixels: "floor", and "wall-<i>" per wall the tile
    # engine rendered. What `/compose` shows or hides one surface at a time.
    regions: dict
    # Per surface id, the render state a 3D view replays to draw the same
    # tiles (tiles_backend.perspective_engine.three_layer.record).
    three: dict = field(default_factory=dict)
    # Validation overlay of the room geometry (room/debug.py); never composited.
    room_debug: object = None


def _instance(label: str, mask: np.ndarray) -> Instance:
    ys, xs = np.nonzero(mask)

    bbox = (
        (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        if ys.size
        else (0, 0, 0, 0)
    )

    return Instance(label=label, mask=mask, bbox=bbox, pixels=int(mask.sum()))


def _room_box_mm(room_mm: tuple, geometry: dict | None,
                 floor: np.ndarray | None = None) -> tuple[tuple[float, float, float], str]:
    """
    All three room dimensions for the legacy room box: the typed ones as they
    are, the rest from RoomGeometry's estimate for this photo. Raises
    ValueError (a 422) when a dimension is neither typed nor estimable.
    """
    if all(value is not None for value in room_mm):
        return tuple(float(v) for v in room_mm), "USER_INPUT"

    frame = (geometry or {}).get("room_frame")

    if not frame:
        raise ValueError(
            "Please enter one known measurement: this room has no stored geometry to "
            "estimate its size from (re-running Clean Room also works)."
        )

    from tiles_backend.perspective_engine.room import geometry as room_geometry

    solved = room_geometry.solve(frame, dict(zip(("width", "length", "height"), room_mm)),
                                 floor_mask=floor)
    filled = tuple(
        typed if typed is not None else solved.get(f"{name}_mm")
        for typed, name in zip(room_mm, ("width", "length", "height"))
    )

    missing = [name for value, name in zip(filled, ("width", "length", "height")) if not value]

    # A room with no measurable ceiling: the box's height only sets the box's
    # own camera and its own wall planes, and neither reaches the tiles (the
    # horizon comes from the detected floor, the scale from RoomGeometry). So it
    # gets the lowest height at which its camera sits at its own default --
    # live_scene's constants, no new guess -- and says so. Floor tiles still
    # render; a wall with no scale still asks for a measurement.
    height_note = None
    if missing == ["height"]:
        filled = (filled[0], filled[1], live_scene.DEFAULT_CAMERA_HEIGHT_MM / live_scene.MAX_CAMERA_HEIGHT_FRACTION)
        missing = []
        height_note = "height not estimated (room box only; tiles do not use it)"

    if missing:
        raise ValueError(
            "Please enter one known measurement: the room " + " and ".join(missing)
            + " could not be estimated from this photo."
        )

    source = "ESTIMATED" if all(value is None for value in room_mm) else "USER_INPUT+ESTIMATED"
    if height_note:
        source += f"; {height_note}"

    return tuple(float(v) for v in filled), source


def _detected_horizon_vp(geometry: dict | None, shape: tuple[int, int]):
    """
    ((x, y), pitch_deg) of the horizon straight ahead, from the room frame's
    DETECTED floor plane -- its up axis in camera coordinates (floor VPs /
    horizon, stored at Clean Room) -- or None without a room frame. No room
    size enters it.

    The horizon is where rays meet no floor: at the centre column, the row v
    with Y_y * (v - cy) / f + Y_z = 0 for the up axis Y.
    """
    fr = (geometry or {}).get("room_frame") or {}
    axes, K = fr.get("axes_camera"), fr.get("K")
    if not axes or not K:
        return None
    up = axes["Y"]
    if abs(up[1]) < 1e-9:
        return None
    height, width = shape
    cw, ch = fr.get("canvas") or [width, height]
    sx, sy = width / cw, height / ch
    f, cx, cy = K[0][0], K[0][2], K[1][2]
    v = cy - f * up[2] / up[1]
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, -up[2]))))
    return (cx * sx, v * sy), pitch


def _horizon_vp(estimate: dict, shape: tuple[int, int]) -> tuple[float, float]:
    """
    The estimated camera's vanishing point straight ahead, in image pixels.

    `live_scene` models a level camera pitched down by `camera_pitch_deg`; the
    floor's far edge approaches row `cy - f * tan(pitch)` as the room gets
    deeper, which is the horizon, and a level camera looks at it along the
    image's centre column.
    """
    height, width = shape

    pitch = math.radians(estimate["camera_pitch_deg"])

    return width / 2.0, height / 2.0 - estimate["focal_px"] * math.tan(pitch)


def _summary(info: dict) -> dict:
    """The engine's per-surface diagnostics, trimmed to what a response needs."""
    summary = {}

    floor = info["surfaces"].get("floor")

    if floor:
        summary["floor"] = {
            key: floor.get(key)
            for key in (
                "focal_px", "focal_source", "vp_source", "vp_confidence", "scale_source",
                "surface_width_mm", "surface_depth_mm", "tiles_across", "tiles_deep",
            )
        } | {"clip": floor["clip"]}

    wall = info["surfaces"].get("wall")

    if wall:
        summary["wall"] = {
            "focal_px": wall.get("focal_px"),
            "scale_source": wall.get("scale_source"),
            "walls": [
                {
                    key: entry.get(key)
                    for key in (
                        "index", "label", "plane_source", "wall_width_mm", "wall_height_mm",
                        "tiles_across", "tiles_up", "grid_rotation_deg",
                        "roll_applied_deg", "roll_rejected", "roll_gate",
                        "placement", "validated", "validation",
                    )
                }
                for entry in wall.get("walls", [])
            ],
            "failures": wall.get("failures"),
            "clip": wall["clip"],
        }

    if info.get("skipped"):
        summary["skipped"] = info["skipped"]

    # The canonical room this render used (tiles_backend/.../room/geometry.py).
    if info.get("room"):
        summary["room"] = info["room"]

    return summary


def render_tiled_room(
    room: np.ndarray,
    floor_mask: np.ndarray,
    wall_mask: np.ndarray,
    spec: TileSpec,
    surface: str,
    room_mm: tuple[float | None, float | None, float | None],
    *,
    clean: np.ndarray | None = None,
    props: np.ndarray | None = None,
    wall_labels: tuple[str, ...] | None = None,
    object_count: int | None = None,
    photo_bytes: bytes | None = None,
    wall_index: int | None = None,
    geometry: dict | None = None,
) -> Rendered:
    """
    Tile a room onto its floor and wall masks with the migrated tile engine.

    `room` is the photograph at render resolution and `clean` the same room
    with its objects removed; `props` is the union of those objects, restored
    over the tiles at the end. `floor_mask` and `wall_mask` are FLOOR_MASK.png
    and WALL_MASK.png — 0/255 or bool, at any resolution; they are brought to
    `room`'s. `photo_bytes` is the original upload, for its EXIF focal length.
    `wall_index` tiles that one wall on its own (surface "wall" only): the
    same wall `surfaces()` lists as "wall-<index>".

    Raises `engine.MissingGeometryError` when there is no floor to estimate a
    camera from, `masks.MaskError` for an unusable mask, and `ValueError` when
    nothing could be tiled.
    """
    shape = room.shape[:2]
    geometry_in = geometry

    floor, wall, contested = masks.prepare(floor_mask, wall_mask, shape)

    # `room_mm` is what the user typed (None = AUTO). The room box below only
    # supplies the fallback horizon, and needs all three: missing ones come from
    # the room geometry's own estimate, never from a default.
    box_mm, box_source = _room_box_mm(room_mm, geometry, floor)

    base = room if clean is None else clean

    # ---- existing geometry: the estimated camera, for its horizon ----
    scene, estimate = live_scene.build(
        room,
        [],
        [_instance("floor", floor), _instance("wall", wall)],
        room_mm=box_mm,
        clean=clean,
        props=props,
        object_count=object_count,
    )

    # The fallback horizon for the floor (used only when the floor's own VP is
    # rejected). AUTO: from the detected floor plane -- never from a room length
    # nobody measured. MANUAL: the room box fitted to the typed size, as before.
    detected = _detected_horizon_vp(geometry, shape) if all(v is None for v in room_mm) else None
    if detected is not None:
        fallback_vp, horizon_pitch = detected
        horizon_source = "detected floor plane (room frame)"
    else:
        fallback_vp = _horizon_vp(estimate, shape)
        horizon_pitch = estimate["camera_pitch_deg"]
        horizon_source = ("room box fitted to the typed size" if any(v is not None for v in room_mm)
                          else "room box (no room frame for this photo)")

    # ---- which walls, when only some were asked for ----
    #
    # The left / right / back the UI offers are this backend's own room-box
    # walls, so the choice is resolved the same way: the pixels `engine` assigns
    # to the chosen planes. The box is convex, so a ray meets at most one wall
    # inside its extent, and projecting only the chosen walls returns exactly
    # the pixels they own. The tile engine still renders every wall; this only
    # decides which pixels of them are kept.
    wall_region = None

    if wall_labels:
        chosen, _ = live_scene.build(
            room,
            [],
            [_instance("floor", floor), _instance("wall", wall)],
            room_mm=box_mm,
            wall_labels=tuple(wall_labels),
            clean=clean,
            props=props,
        )

        canvas = np.zeros((*shape, 3), dtype=np.float32)

        wall_region = engine._project_walls(replace(chosen, wall=wall, wall_coverage={}), spec, canvas)

    # ---- the tile engine ----
    request = TileRequest(
        tile_width_mm=spec.width_mm,
        tile_height_mm=spec.height_mm,
        rotation_deg=spec.rotation_deg,
        grout_mm=spec.grout_mm,
        room_width_mm=room_mm[0],
        room_length_mm=room_mm[1],
        room_height_mm=room_mm[2],
    )

    tiled = render_room(
        base,
        floor,
        wall,
        spec.artwork,
        surface,
        request,
        photo_bytes=photo_bytes,
        wall_region=wall_region,
        fallback_vp=fallback_vp,
        wall_index=wall_index,
        geometry=geometry,
    )

    # ---- objects back over the tiles, from the original photograph ----
    #
    # The engine's image is already lit — lighting is part of its composite —
    # so `raw` and `lit` are the same image here.
    composite = engine._restore_props(scene, tiled.image)

    result = Result(
        raw=tiled.image,
        lit=tiled.image.copy(),
        composite=composite,
        target=tiled.tiled.copy(),
        underlying=base.copy(),
        stats={
            "renderer": "tiles_backend.perspective_engine",
            "canvas": [int(shape[1]), int(shape[0])],
            "floor_pixels": int(tiled.floor_tiled.sum()),
            "wall_pixels": int(tiled.wall_tiled.sum()),
            "prop_pixels": int(scene.props.sum()),
            "tile_mm": [spec.width_mm, spec.height_mm],
            "grout_mm": spec.grout_mm,
            "rotation_deg": spec.rotation_deg,
            "surface": surface,
        },
    )

    # ---- the final guarantee ----
    clip = compositing.clip(result, scene, compositing.allowed(surface, floor, wall))

    clip["contested_pixels"] = contested
    clip["masks"] = "FLOOR_MASK.png + WALL_MASK.png"

    geometry = {
        "method": "tile engine",
        "renderer": "tiles_backend.perspective_engine",
        "surface_masks": clip["masks"],
        "fallback_vp": [round(fallback_vp[0], 1), round(fallback_vp[1], 1)],
        "fallback_horizon": {"source": horizon_source, "pitch_deg": round(float(horizon_pitch), 2)},
        "estimated_camera": {
            key: estimate[key]
            for key in ("focal_px", "horizontal_fov_deg", "camera_pitch_deg", "camera_height_mm")
        },
        "room_mode": "manual" if any(value is not None for value in room_mm) else "auto",
        # Typed by the user; None = not typed (AUTO).
        "room_mm": [None if value is None else round(value, 1) for value in room_mm],
        "room_box_mm": [round(value, 1) for value in box_mm],
        "room_box_source": box_source,
        "object_count": estimate["object_count"],
        "walls_selected": list(wall_labels) if wall_labels else "all",
        **_summary(tiled.info),
    }

    regions = {}

    if tiled.floor_tiled.any():
        regions["floor"] = tiled.floor_tiled & result.target

    for region in tiled.wall_regions:
        mask = region["mask"] & result.target

        if mask.any():
            regions[f"wall-{region['index']}"] = mask

    three = {}
    floor_info = tiled.info["surfaces"].get("floor") or {}
    if "floor" in regions and floor_info.get("three"):
        three["floor"] = floor_info["three"]
    for entry in (tiled.info["surfaces"].get("wall") or {}).get("walls", []):
        key = f"wall-{entry.get('index')}"
        if key in regions and entry.get("three"):
            three[key] = entry["three"]

    # A wall whose stored direction was not validated (weak or disagreeing
    # edges) is not validated either, whatever placed its plane.
    stored_dirs = {int(w.get("index")): (w.get("direction") or {}) for w in (geometry_in or {}).get("walls") or []}
    for entry in (geometry.get("wall") or {}).get("walls") or []:
        d = stored_dirs.get(entry.get("index"))
        if d is not None and d.get("validated") is False:
            entry["validated"] = False
            entry["validation"] = "; ".join(filter(None, [entry.get("validation"),
                                                          f"direction not validated: {d.get('validation')}"]))

    # Tile count and edge cuts from the room's mm and the tile's mm (reporting
    # only; computed after the render, from the same RoomGeometry it used).
    if tiled.info.get("room"):
        from tiles_backend.perspective_engine.room import tile_count

        geometry["tile_layout"] = tile_count.room_layout(
            tiled.info["room"], spec.width_mm, spec.height_mm, spec.rotation_deg)

    room_debug = None
    if tiled.info.get("room") and geometry_in and geometry_in.get("room_frame"):
        from tiles_backend.perspective_engine.room import debug as room_debug_mod

        room_debug = room_debug_mod.draw(
            cv2.cvtColor(np.ascontiguousarray(base), cv2.COLOR_RGB2BGR),
            geometry_in["room_frame"], tiled.info["room"],
            layout=geometry.get("tile_layout"),
        )

    return Rendered(result=result, scene=scene, geometry=geometry, clip=clip, regions=regions,
                    three=three, room_debug=room_debug)


def _dot(mask: np.ndarray) -> list[int]:
    """
    Where a surface's select dot goes: the pixel farthest from its edge.

    The tile engine's own `interior_point`, so the point a wall's dot sits on
    is exactly the point that wall is rendered from.
    """
    return list(interior_point(mask))


def describe(regions: dict[str, np.ndarray]) -> list[dict]:
    """
    The selectable surfaces, in the shape the UI reads.

    Floor first, then walls numbered left to right by where their dot sits —
    "Wall 1" is the leftmost. The id is the tile engine's own wall index, so
    it means the same wall in every response for this room.
    """
    listed = []

    if "floor" in regions and regions["floor"].any():
        listed.append(
            {
                "id": "floor",
                "kind": "floor",
                "label": "Floor",
                "dot": _dot(regions["floor"]),
                "pixels": int(regions["floor"].sum()),
            }
        )

    walls = [
        {"id": key, "kind": "wall", "dot": _dot(mask), "pixels": int(mask.sum())}
        for key, mask in regions.items()
        if key.startswith("wall-") and mask.any()
    ]

    for number, wall in enumerate(sorted(walls, key=lambda item: item["dot"][0]), start=1):
        listed.append({**wall, "label": f"Wall {number}"})

    return listed


def ensure_geometry(
    segments,
    room: np.ndarray,
    floor_mask: np.ndarray,
    wall_mask: np.ndarray,
    *,
    clean: np.ndarray | None = None,
    photo_bytes: bytes | None = None,
) -> dict:
    """
    This room's floor and wall geometry: stored in `segments` when it has
    been detected before, otherwise detected now — on the render-resolution
    clean room, the image the tiles go on — and stored for next time.
    """
    stored = geometry_store.load(segments)

    if stored is not None:
        return stored

    floor, wall, _ = masks.prepare(floor_mask, wall_mask, room.shape[:2])

    geometry, wall_masks = detect_room_geometry(
        room if clean is None else clean, floor, wall, photo_bytes=photo_bytes,
        objects_mask=_objects_mask(segments, room.shape[:2]),
    )

    geometry_store.write(segments, geometry, wall_masks)

    # In memory only: the stored wall split, which the tile engine renders.
    return {**geometry, "_wall_masks": wall_masks}


def _objects_mask(segments, shape):
    """Where objects were removed from the photo (ALL_OBJECTS.png alpha), or None."""
    path = Path(segments) / "ALL_OBJECTS.png"
    if not path.exists():
        return None
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    if image.ndim == 3 and image.shape[2] == 4:
        alpha = image[..., 3]
    elif image.ndim == 3:
        alpha = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        alpha = image
    if alpha.shape != tuple(shape):
        alpha = cv2.resize(alpha, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return alpha > 127


def surfaces(
    room: np.ndarray,
    floor_mask: np.ndarray,
    wall_mask: np.ndarray,
    *,
    clean: np.ndarray | None = None,
    photo_bytes: bytes | None = None,
    geometry: dict | None = None,
) -> list[dict]:
    """
    What a cleaned room offers for tiling, before any tile is chosen.

    The floor is FLOOR_MASK; the walls are WALL_MASK split exactly as the tile
    engine will split it when it tiles them, so each dot belongs to the wall
    whose tiles it will switch on and off.
    """
    shape = room.shape[:2]

    floor, wall, _ = masks.prepare(floor_mask, wall_mask, shape)

    if geometry is not None:
        # Every wall exactly as the room's geometry split it, each dot on the
        # point that geometry chose inside it.
        listed = []

        if floor.any():
            listed.append({"id": "floor", "kind": "floor", "label": "Floor",
                           "dot": _dot(floor), "pixels": int(floor.sum())})

        walls = sorted(geometry.get("walls", []), key=lambda w: w["select_point"][0])

        for number, w in enumerate(walls, start=1):
            listed.append({"id": w["id"], "kind": "wall", "label": f"Wall {number}",
                           "dot": list(w["select_point"]), "pixels": int(w["pixels"])})

        return listed

    found = detect_surfaces(
        room if clean is None else clean, floor, wall, photo_bytes=photo_bytes
    )

    regions = {"floor": found["floor"]}

    claimed = np.zeros(shape, dtype=bool)

    # Later walls are painted over earlier ones; give shared pixels to them,
    # exactly as the render does.
    for index, mask in reversed(found["walls"]):
        regions[f"wall-{index}"] = mask & ~claimed
        claimed |= mask

    return describe(regions)
