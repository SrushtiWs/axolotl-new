"""
The engine's entry point: a cleaned room and its two masks in, a tiled room out.

    CLEAN_ROOM.png + FLOOR_MASK.png + WALL_MASK.png + tile + dimensions
      -> per surface, the reference project's own pipeline, unchanged:
           floor  pipeline/floor_pipeline.render_floor_with_info
           wall   pipeline/wall_pipeline.render_wall_with_info
      -> strict clip to that surface's mask            compositing.clip
      -> floor and wall merged, each through its own mask
      -> final tiled room

Everything that decides where a tile goes — vanishing points, focal length,
plane, metric scale, basis, rotation, anchoring, the millimetre grid,
fractional and cut tiles, grout, lighting — is the migrated code, called with
the options the reference frontend sent. This module only builds those inputs
from the masks and puts the results together.

Masks are the source of truth. Detection is not done here: FLOOR_MASK and
WALL_MASK come from the backend's floor/wall stage and are handed to the
pipelines exactly as they are, as white-on-black surface masks, which is the
form the reference `/render-tile-full` endpoint accepted.
"""

from __future__ import annotations

import hashlib
import math
import threading
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from typing import Optional

import cv2
import numpy as np

from . import compositing, depth
from .camera.focal_estimate import estimate_focal_from_exif
from .camera.metric_scale import MetricScaleError
from .camera.focal_estimate import resolve_focal_length
from .core.composite import decode_mask, normalise_depth_within_mask, normalise_inputs
from .core.options import SurfaceRenderOptions
from .pipeline.floor_pipeline import render_floor_with_info
from .pipeline.wall_pipeline import (
    DEPTH_BAND_HI,
    DEPTH_BAND_LO,
    WallProfile,
    _select_instances,
    render_wall_with_info,
)

# What the reference frontend (ai-tile-mapper/src/App.tsx) actually sent, where
# it differs from SurfaceRenderOptions' own defaults. Those are the values the
# reference renders were made with, so they are the ones reproduced here.
#
# With one exception: the frontend also sent focal_length=1500 on every
# request. The engine's own ladder is documented, in four places, as
# "EXIF > two-VP calibration > FOV prior", with the caller's value only a
# fallback — but `resolve_focal_length` checks a plausible caller value BEFORE
# the prior, so in practice every photo without EXIF or a two-VP calibration
# got 1500 px whatever its width. On a 540 px room that is a 20 degree lens,
# and the floor reconstructed 13.4 m deep; the prior (26 mm equivalent)
# reconstructs the same floor 3.6 m deep. So no fixed focal length is sent and
# the engine's documented ladder decides. `TileRequest.focal_px` overrides.
REFERENCE_FRONTEND_OPTIONS = {
    "focal_length": None,
    "lighting_blend": 0.85,
    "ransac_threshold": 10.0,
    "ransac_iterations": 600,
}

# The reference frontend's grout colour.
DEFAULT_GROUT_COLOR = "#D0D0C8"

# The floor's metric anchor in the reference project: camera height above the
# floor. There is no control for it in this app; this is the reference default.
DEFAULT_CAMERA_HEIGHT_MM = 1500.0

SURFACES = ("floor", "wall", "both")

# Stored room geometry older than this is detected again: version 2 stores the
# refined wall split, which renders then use instead of splitting again;
# version 3 takes the floor's horizon from the room's structure when the
# floor-only detection is rejected; version 4 lays walls out from their
# floor junction when it is visible; version 5 adds the canonical room frame
# (room/geometry.py) that one metric scale is solved on.
GEOMETRY_VERSION = 5


@dataclass
class TileRequest:
    """One tile, as installed, and the room it goes into."""

    tile_width_mm: float
    tile_height_mm: float
    rotation_deg: float = 0.0
    grout_mm: float = 3.0
    grout_color: str = DEFAULT_GROUT_COLOR
    camera_height_mm: float = DEFAULT_CAMERA_HEIGHT_MM
    # A known focal length in pixels at the room's resolution. `None` lets the
    # engine decide: EXIF, then two-VP calibration, then its FOV prior.
    focal_px: Optional[float] = None
    # The wall's metric anchor. The floor ignores them, as it did in the
    # reference project — its scale comes from the camera height.
    room_width_mm: Optional[float] = None
    room_length_mm: Optional[float] = None
    room_height_mm: Optional[float] = None


@dataclass
class RoomRender:
    """A tiled room and what the render decided."""

    image: np.ndarray  # (H, W, 3) uint8 RGB — the clean room, tiled inside the masks
    floor_tiled: np.ndarray  # (H, W) bool — pixels carrying a floor tile
    wall_tiled: np.ndarray  # (H, W) bool — pixels carrying a wall tile
    info: dict = field(default_factory=dict)
    # One entry per wall the engine tiled: {"index": i, "mask": (H, W) bool},
    # the pixels of `wall_tiled` that belong to that wall. Lets a caller show
    # or hide each wall's tiles on its own.
    wall_regions: list = field(default_factory=list)

    @property
    def tiled(self) -> np.ndarray:
        return self.floor_tiled | self.wall_tiled


def _binary(mask: np.ndarray) -> np.ndarray:
    """A 0/255 or bool mask as bool. Anything else is refused, not thresholded."""
    if mask.dtype == bool:
        return mask

    values = set(np.unique(mask).tolist())

    if not values <= {0, 255}:
        raise ValueError(f"surface masks must be strictly 0/255; got values {sorted(values)[:8]}")

    return mask == 255


def _mask_bgr(mask: np.ndarray) -> np.ndarray:
    """A bool mask as the white-on-black BGR image the pipelines decode."""
    white = mask.astype(np.uint8) * 255

    return cv2.merge([white, white, white])


def _options(
    request: TileRequest,
    surface: str,
    exif_focal_px: Optional[float],
    fallback_vp: Optional[tuple[float, float]] = None,
    wall_point: Optional[tuple[float, float]] = None,
) -> SurfaceRenderOptions:
    wall = surface == "wall"

    options = dict(REFERENCE_FRONTEND_OPTIONS)

    if request.focal_px is not None:
        options["focal_length"] = float(request.focal_px)

    return SurfaceRenderOptions(
        **options,
        # The engine's own fallback for when its detector rejects the vanishing
        # point: used only then, and ignored when detection is accepted.
        vanishing_point_x=None if fallback_vp is None else float(fallback_vp[0]),
        vanishing_point_y=None if fallback_vp is None else float(fallback_vp[1]),
        tile_rotation=float(request.rotation_deg),
        tile_width_mm=float(request.tile_width_mm),
        tile_height_mm=float(request.tile_height_mm),
        enable_grout=request.grout_mm > 0,
        grout_width_mm=float(request.grout_mm),
        grout_color=request.grout_color,
        camera_height_mm=float(request.camera_height_mm),
        exif_focal_px=exif_focal_px,
        surface=surface,
        # Attached only for a wall, exactly as the reference frontend did.
        room_width_mm=request.room_width_mm if wall else None,
        room_length_mm=request.room_length_mm if wall else None,
        room_height_mm=request.room_height_mm if wall else None,
        # The reference engine's single-wall mode: render only the wall under
        # this point, with that wall as its own scale anchor.
        wall_point_x=None if wall_point is None else float(wall_point[0]),
        wall_point_y=None if wall_point is None else float(wall_point[1]),
    )


# ---------------------------------------------------------------------------
# The walls, as the wall pipeline splits them.
#
# `render_wall_with_info` reports each wall's plane, size and centroid but not
# its mask, and showing or hiding one wall's tiles needs its pixels. So the
# pipeline's own set-up is repeated here with its own functions — the same
# decode, depth normalisation, focal length, detector and selection — which is
# deterministic, and `render_room` then checks every wall's pixel count against
# the one the render itself reported before trusting the match.
#
# The split does not depend on the tile at all, so it is cached: a room's walls
# are found once, however many tiles are tried on them.

_INSTANCE_CACHE: "OrderedDict[str, list]" = OrderedDict()
_INSTANCE_CACHE_SIZE = 8
_instance_lock = threading.Lock()


def wall_instances(
    clean_rgb: np.ndarray,
    wall_mask: np.ndarray,
    request: Optional["TileRequest"] = None,
    *,
    exif_focal_px: Optional[float] = None,
    frame_depth: Optional[np.ndarray] = None,
    focal_px: Optional[float] = None,
) -> list[tuple[int, np.ndarray]]:
    """
    The walls the wall pipeline renders, as `[(index, mask), ...]`.

    `focal_px` is the camera's focal length when the room's geometry has
    already fixed it; the split depends on it, so a render handed the same
    focal splits the walls identically.

    Indices are the pipeline's own, the ones `render_room` reports under
    `info["surfaces"]["wall"]["walls"]`. Masks are (H, W) bool.
    """
    wall = _binary(wall_mask)

    request = request or TileRequest(tile_width_mm=1.0, tile_height_mm=1.0)

    key = hashlib.sha1(
        np.ascontiguousarray(clean_rgb).tobytes()
        + np.packbits(wall).tobytes()
        + repr((clean_rgb.shape, exif_focal_px, request.focal_px, focal_px)).encode()
    ).hexdigest()

    with _instance_lock:
        if key in _INSTANCE_CACHE:
            _INSTANCE_CACHE.move_to_end(key)
            return _INSTANCE_CACHE[key]

    opts = _options(request, "wall", exif_focal_px)

    room_bgr = cv2.cvtColor(np.ascontiguousarray(clean_rgb), cv2.COLOR_RGB2BGR)

    if frame_depth is None:
        frame_depth = depth.estimate_depth_heatmap(clean_rgb)

    depth_bgr = depth.surface_depth_heatmap(frame_depth, wall)

    _depth, mask_bgra, _tile, depth_val, had_alpha = normalise_inputs(
        room_bgr, depth_bgr, _mask_bgr(wall), np.zeros((1, 1, 4), np.uint8), opts.invert_depth
    )

    wall_bool, _factor = decode_mask(mask_bgra, had_alpha, surface_label="wall")

    if opts.depth_normalisation_enabled(default=True):
        depth_val = normalise_depth_within_mask(
            depth_val, wall_bool, out_lo=DEPTH_BAND_LO, out_hi=DEPTH_BAND_HI
        )

    height, width = room_bgr.shape[:2]
    cx, cy = width / 2.0, height / 2.0

    if focal_px is not None:
        f = float(focal_px)
    else:
        f, _info = resolve_focal_length(
            image_width=width,
            exif_focal_px=opts.exif_focal_px,
            vp1=None,
            vp2=None,
            principal_point=(cx, cy),
            fallback_focal=opts.focal_length,
            auto=opts.auto_focal_length,
        )

    found = WallProfile(opts).instances(wall_bool, room_bgr, depth_val, opts, cx, cy, f)

    found = _select_instances(found, opts, width, height)

    instances = [(int(item.index), item.mask.astype(bool)) for item in found]

    with _instance_lock:
        _INSTANCE_CACHE[key] = instances

        while len(_INSTANCE_CACHE) > _INSTANCE_CACHE_SIZE:
            _INSTANCE_CACHE.popitem(last=False)

    return instances


def interior_point(mask: np.ndarray) -> tuple[int, int]:
    """
    The pixel of `mask` farthest from its edge, as (x, y).

    Always inside the surface — a centroid can fall outside an L-shaped wall or
    into a window cut out of it — and the frame edge counts as an edge, so the
    point never lands on the border of a surface that runs off the photo.
    """
    padded = cv2.copyMakeBorder(mask.astype(np.uint8), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)

    distance = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]

    y, x = np.unravel_index(int(np.argmax(distance)), distance.shape)

    return int(x), int(y)


def detect_surfaces(
    clean_rgb: np.ndarray,
    floor_mask: np.ndarray,
    wall_mask: np.ndarray,
    request: Optional["TileRequest"] = None,
    *,
    photo_bytes: Optional[bytes] = None,
) -> dict:
    """
    The surfaces a room offers for tiling, before any tile is chosen.

    Returns `{"floor": mask, "walls": [(index, mask), ...]}`: the floor mask
    itself, and the walls exactly as the wall pipeline will split them.
    """
    floor = _binary(floor_mask)
    wall = _binary(wall_mask)

    exif_focal_px = None

    if photo_bytes:
        exif_focal_px, _exif = estimate_focal_from_exif(photo_bytes, clean_rgb.shape[1])

    walls = (
        wall_instances(clean_rgb, wall, request, exif_focal_px=exif_focal_px)
        if wall.any()
        else []
    )

    return {"floor": floor, "walls": walls}


# ---------------------------------------------------------------------------
# The room's geometry, detected once.

def detect_room_geometry(
    clean_rgb: np.ndarray,
    floor_mask: np.ndarray,
    wall_mask: np.ndarray,
    request: Optional[TileRequest] = None,
    *,
    photo_bytes: Optional[bytes] = None,
    objects_mask: Optional[np.ndarray] = None,
) -> tuple[dict, dict]:
    """
    The floor's and every wall's perspective geometry, from the image itself.

    Returns `(geometry, wall_masks)`. `geometry` is JSON-ready:

        camera   the one focal length floor and walls share (the floor's
                 detection calibrates it from two vanishing points when it can)
        floor    surface/floor/geometry.detect: vanishing points, horizon,
                 pitch, floor direction, plane, coverage, status
        walls    per wall: its region's size and select point, and its own
                 along-wall vanishing point and orientation where its lines
                 settle them (surface/wall/geometry.wall_direction)
        room_vps the room-level horizontal vanishing points the wall pipeline
                 falls back to for a wall without its own

    `wall_masks` maps each wall index to its (H, W) bool mask.
    """
    from .surface.floor import geometry as floor_geometry
    from .surface.wall import geometry as wall_geometry
    from .surface.wall import vp as wall_vp

    request = request or TileRequest(tile_width_mm=1.0, tile_height_mm=1.0)

    floor = _binary(floor_mask)
    wall = _binary(wall_mask)

    room_bgr = cv2.cvtColor(np.ascontiguousarray(clean_rgb), cv2.COLOR_RGB2BGR)
    height, width = floor.shape
    cx, cy = width / 2.0, height / 2.0

    exif_focal_px = None
    if photo_bytes:
        exif_focal_px, _exif = estimate_focal_from_exif(photo_bytes, width)

    opts = _options(request, "floor", exif_focal_px)

    floor_geo = floor_geometry.detect(room_bgr, floor, opts, request.camera_height_mm)
    f = floor_geo["focal_px"]

    frame_depth = depth.estimate_depth_heatmap(clean_rgb)

    instances = (
        wall_instances(clean_rgb, wall, request, exif_focal_px=exif_focal_px,
                       frame_depth=frame_depth, focal_px=f)
        if wall.any() else []
    )

    room_vps, room_vps_info = wall_vp.room_horizontal_vps(room_bgr, wall) if wall.any() else ([], {})

    # The floor's own lines were not enough: take its horizon from the room's
    # structure (the walls' horizontal vanishing points) when that passes the
    # checks, instead of leaving the render to a camera guessed from the typed
    # room size (surface/floor/geometry.from_room_structure).
    # Where the walls stand on the floor: their runs, and where the receding
    # runs meet -- the room's depth vanishing point (surface/wall/junction_layout).
    from .surface.wall import junction_layout

    removed = None if objects_mask is None else _binary(objects_mask)
    junction_runs, junction_share = (
        junction_layout.runs(floor, wall, removed) if wall.any() else (None, 0.0))
    junction_vp = junction_layout.depth_vp(junction_runs, width) if junction_runs else None

    if floor_geo.get("status") != "detected" and (room_vps or junction_vp):
        structural = floor_geometry.from_room_structure(
            room_bgr, floor, room_vps, floor_geo, request.camera_height_mm,
            opts.depth_perspective_gain, junction_vp=junction_vp,
        )
        if structural is not None:
            floor_geo = structural

    # The floor's depth VP judged on the room's LONG structural lines (image
    # lines on floor and walls plus the mask boundaries, camera/boundary_vp.py):
    # kept when they support it; replaced by the pool's own VP when that is
    # confident instead; rejected when neither is -- never used on faith.
    floor_geo, vp_report = _check_depth_vp(room_bgr, floor, wall, removed, floor_geo, request, opts)

    walls = []
    masks = {}

    # The split, settled once: pieces regrouped by direction, slivers and the
    # ghosts of removed objects folded into their wall (surface/wall/refine.py).
    from .surface.wall import refine as wall_refine

    objects = None if objects_mask is None else _binary(objects_mask)

    # 1. The junction layout, when the walls' feet are visible: exact corners,
    #    and each wall's direction from its own run against the floor's horizon.
    horizon_y, depth_point = None, None
    if floor_geo.get("status") == "detected":
        vps = floor_geo.get("vanishing_points") or {}
        horizon_y, depth_point = vps.get("horizon_y"), vps.get("depth_vp")
    laid, layout_info = (None, {"stage": "junction-hidden"})
    if junction_runs:
        laid, layout_info = junction_layout.layout(
            floor, wall, horizon_y, f, cx, cy, depth_point, junction_runs, junction_share)

    # 1b. Otherwise the edges the walls do show -- floor junction and ceiling
    #     line together -- when enough of them is in view (edge_layout.py):
    #     corners only where an edge visibly bends, never across a hidden gap.
    #     Used only when it sees at least one corner: with none seen (a corner
    #     hidden in a gap of both edges) it would merge walls it cannot tell
    #     apart, so the pipeline's own split is kept instead.
    edged, edge_info = (None, {})
    if not laid:
        from .surface.wall import edge_layout
        edged, edge_info = edge_layout.layout(floor, wall, horizon_y, f, cx, cy, depth_point, objects)
        if edged and not edge_info.get("corners"):
            edged, edge_info = None, {**edge_info, "stage": "no-corner-seen: pipeline split kept"}

    if laid:
        refined = [(i, m, d) for i, (m, d) in enumerate(laid)]
        split_report = {"method": "floor-junction", **layout_info, "walls": len(refined)}
    elif edged:
        # A wall neither edge gives a run for keeps the pipeline's own direction.
        refined = [(i, m, d if d is not None else wall_refine._direction(room_bgr, m, floor, wall, floor_geo, f, cx, cy))
                   for i, (m, d) in enumerate(edged)]
        split_report = {"method": "floor-junction+ceiling-line", "junction_layout": layout_info,
                        **edge_info, "walls": len(refined)}
    else:
        # 2. Otherwise the pipeline's split, regrouped by direction (refine.py).
        refined, split_report = (
            wall_refine.refine(instances, room_bgr, floor, wall, floor_geo, f, cx, cy, objects)
            if instances else ([], {"walls": 0})
        )
        split_report = {"method": "refined-split", "junction_layout": layout_info,
                        "edge_layout": edge_info, **split_report}

    # Every wall pixel in exactly one wall: a pixel two pieces of the split both
    # claim stays with the first (the refined split can overlap by a few pixels).
    claimed = np.zeros(wall.shape, dtype=bool)
    disjoint = []
    for index, mask, direction in refined:
        mask = mask & ~claimed
        if mask.any():
            claimed |= mask
            disjoint.append((index, mask, direction))
    overlap_px = int(sum(int(m.sum()) for _, m, _ in refined) - claimed.sum())
    if overlap_px:
        split_report = {**split_report, "overlap_pixels_resolved": overlap_px}
    refined = disjoint

    # Each wall's direction from its OWN junction and ceiling line, fitted
    # robustly (surface/wall/edge_direction.py). A wall with no usable line of
    # its own keeps the pipeline's direction only if every one of its pixels can
    # be tiled with it, and is then not validated; otherwise it has none.
    from .surface.wall import edge_direction
    from .surface.wall.geometry import _reach as _wall_reach
    directed = []
    for index, mask, direction in refined:
        fitted = edge_direction.fit(mask, floor, objects, horizon_y, f, cx, cy, depth_point)
        if fitted is not None:
            direction = fitted
        elif direction is not None:
            reach = _wall_reach(np.asarray(direction["normal"], float), mask, f, cx, cy)
            if reach < 1.0:
                direction = None
            else:
                direction = {**direction, "validated": False,
                             "validation": f"no usable junction or ceiling line; direction from "
                                           f"{direction.get('source') or 'wall lines'}, every pixel tileable"}
        directed.append((index, mask, direction))
    refined = directed
    split_report = {**split_report, "direction_check": _direction_check(refined, horizon_y, depth_point, f, cx)}

    for index, mask, direction in refined:
        ys, xs = np.nonzero(mask)

        walls.append({
            "id": f"wall-{index}",
            "index": int(index),
            "pixels": int(mask.sum()),
            "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
            "select_point": list(interior_point(mask)),
            "direction": direction,
            "orientation_source": (
                (direction.get("source") or "wall-lines") if direction else "room-vp (pipeline)"
            ),
        })
        masks[int(index)] = mask

    # The canonical room, from the image only (room/geometry.py): gravity, the
    # room's axes, each wall measured from its floor junction, the ceiling, the
    # detected corners -- in camera-height units. The typed room size is applied
    # per request (room_geometry.solve), never here.
    from .camera import vertical_vp
    from .room import geometry as room_geometry

    vertical = vertical_vp.detect(room_bgr, wall, f, cx, cy) if wall.any() else {"reliable": False}
    room_frame = room_geometry.frame(floor_geo, walls, masks, floor, wall, f, cx, cy, vertical, objects)

    geometry = {
        "version": GEOMETRY_VERSION,
        "room_frame": room_frame,
        "wall_split": split_report,
        "vp_report": vp_report,
        "canvas": [int(width), int(height)],
        "camera": {
            "focal_px": float(f),
            "focal_source": floor_geo.get("focal_source"),
            "principal_point": [cx, cy],
        },
        "floor": floor_geo,
        "walls": walls,
        "room_vps": {"points": [list(v) for v in room_vps], "info": {
            k: v for k, v in room_vps_info.items() if k != "vps"
        }},
    }

    return geometry, masks


def _check_depth_vp(room_bgr, floor, wall, objects, floor_geo: dict, request, opts):
    """(floor_geo, report): the depth VP scored on long structural lines (see the call site)."""
    from .camera import boundary_vp
    from .surface.floor import geometry as floor_geometry

    pool = boundary_vp.long_line_pool(room_bgr, floor, wall, objects)
    vps = (floor_geo.get("vanishing_points") or {})
    current = vps.get("depth_vp") if floor_geo.get("status") == "detected" else None
    report = {"rule": {"inliers": boundary_vp.ACCEPT_INLIERS, "support": boundary_vp.ACCEPT_SUPPORT,
                       "spread_deg": boundary_vp.ACCEPT_SPREAD_DEG, "inlier_deg": boundary_vp.INLIER_DEG},
              "pool_lines": len(pool),
              "floor_vp": {"point": current, "origin": vps.get("origin", "floor-lines"),
                           "detector_confidence": vps.get("confidence"), **boundary_vp.score_vp(pool, current)}}
    fitted = boundary_vp.fit_vp(pool, finite_only=True) if len(pool) >= 2 else None
    pool_point = fitted["image"] if fitted and fitted.get("image") else None
    report["pool_vp"] = {"point": pool_point, **boundary_vp.score_vp(pool, pool_point)}

    if current is not None and report["floor_vp"]["confident"]:
        report["decision"] = "floor VP kept: confident on the room's long lines"
        return floor_geo, report
    if pool_point is not None and report["pool_vp"]["confident"]:
        replaced = floor_geometry.from_room_structure(
            room_bgr, floor, [pool_point], floor_geo, request.camera_height_mm, opts.depth_perspective_gain)
        if replaced is not None:
            replaced["vanishing_points"]["origin"] = "long-structural-lines"
            report["decision"] = ("floor VP replaced by the long lines' VP" if current is not None
                                  else "long lines' VP used (the floor gave none)")
            return replaced, report
        report["pool_vp"]["structure_check"] = "failed (floor coverage or pitch)"
    if current is None:
        report["decision"] = "no confident depth VP"
        return floor_geo, report
    rejected = dict(floor_geo)
    rejected["status"] = "rejected"
    rejected["vanishing_points"] = {**vps, "rejected_reason": "low confidence on the room's long lines"}
    report["decision"] = "floor VP rejected: low confidence, and no confident alternative"
    return rejected, report


def _direction_check(walls, horizon_y, depth_vp, f, cx) -> dict:
    """
    The walls' directions checked against each other: side walls against the
    room's depth direction, the back wall facing the camera, one horizon.
    """
    az_depth = None if depth_vp is None else math.degrees(math.atan2(depth_vp[0] - cx, f)) % 180.0
    rows = []
    for index, mask, d in walls:
        if d is None:
            rows.append({"wall": f"wall-{index}", "direction": None, "validated": False})
            continue
        n = d["normal"]
        az = math.degrees(math.atan2(-n[2], n[0])) % 180.0       # along-wall direction (n rotated 90 deg)
        off = None if az_depth is None else min(abs(az - az_depth) % 180.0, 180.0 - abs(az - az_depth) % 180.0)
        rows.append({"wall": f"wall-{index}", "faces": d.get("faces"), "source": d.get("source"),
                     "snapped_to_depth_vp": (d.get("evidence") or {}).get("snapped_to_depth_vp"),
                     "deg_from_depth_direction": None if off is None else round(off, 2),
                     "horizon_y": (d.get("evidence") or {}).get("horizon_y"),
                     "validated": d.get("validated", True), "validation": d.get("validation")})
    horizons = {r.get("horizon_y") for r in rows if r.get("horizon_y") is not None}
    return {"horizon_y": None if horizon_y is None else round(float(horizon_y), 1),
            "one_horizon": len(horizons) <= 1, "depth_vp": None if depth_vp is None else [round(float(v), 1) for v in depth_vp],
            "walls": rows}


def _stored_floor(geometry: Optional[dict], shape, fallback_vp):
    """
    (GeometryEvidence or None, (f, info) or None) for a render of `shape`.

    A detected floor is used as detected. Otherwise the evidence is the
    fallback vanishing point -- the backend's estimated camera, which knows the
    room's dimensions -- or None, leaving the pipeline to decide as before.
    """
    if not geometry:
        return None, None

    from .surface.floor import geometry as floor_geometry
    from .surface.base import GeometryEvidence

    scale = shape[1] / float(geometry["canvas"][0])
    camera = geometry["camera"]
    f = camera["focal_px"] * scale
    focal = (f, {"source": f"stored:{camera.get('focal_source')}",
                 "hfov_deg": math.degrees(2 * math.atan(shape[1] / (2 * f)))})

    floor = geometry.get("floor") or {}

    if floor.get("status") == "detected":
        return floor_geometry.evidence_from(floor, scale), focal

    if fallback_vp is not None:
        ev = GeometryEvidence(
            vp_x=float(fallback_vp[0]), vp_y=float(fallback_vp[1]),
            vp_horizon_y=float(fallback_vp[1]), source=f"fallback-after-{floor.get('status', 'rejected')}",
        )
        return ev, focal

    return None, focal


def _stored_wall_masks(geometry: Optional[dict], shape, wall: np.ndarray):
    """
    {wall index: (H, W) bool} from the room's stored split, or None when the
    geometry carries none (older geometry, or none at all). Each mask is
    brought to the render's shape and kept inside the wall mask.
    """
    stored = (geometry or {}).get("_wall_masks")
    if not stored:
        return None
    out = {}
    for index, mask in stored.items():
        m = np.asarray(mask)
        if m.shape != tuple(shape):
            m = cv2.resize(m.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        m = m.astype(bool) & wall
        if m.any():
            out[int(index)] = m
    return out or None


def _stored_wall_normals(geometry: Optional[dict]):
    """{wall index: (a, b, c)} for every wall whose own lines fixed its orientation."""
    if not geometry:
        return None

    return {
        int(w["index"]): tuple(w["direction"]["normal"])
        for w in geometry.get("walls", [])
        if w.get("direction")
    }


def render_room(
    clean_rgb: np.ndarray,
    floor_mask: np.ndarray,
    wall_mask: np.ndarray,
    tile_rgb: np.ndarray,
    surface: str,
    request: TileRequest,
    *,
    photo_bytes: Optional[bytes] = None,
    wall_region: Optional[np.ndarray] = None,
    fallback_vp: Optional[tuple[float, float]] = None,
    wall_index: Optional[int] = None,
    geometry: Optional[dict] = None,
) -> RoomRender:
    """
    Tile `surface` of a cleaned room, strictly inside its mask.

    `clean_rgb` is the room with its objects removed, and `floor_mask` /
    `wall_mask` the floor/wall stage's masks for it — 0/255 or bool, the same
    size as the room. `surface` is "floor", "wall" or "both". `photo_bytes` is
    the original upload, whose EXIF focal length the reference project read
    first; without it the focal length comes from the vanishing points or the
    field-of-view prior, as it did there.

    `wall_region` limits wall tiling to part of the wall mask — the walls a
    user picked. Every wall is still rendered as the reference engine renders
    them, sharing one metric scale; the region only decides which of their
    pixels are kept. `None` keeps every wall, the reference default.

    `geometry` is the room's detected geometry (`detect_room_geometry`). With
    it, tiles are projected with that geometry rather than a fresh detection:
    the floor with its stored vanishing points and focal length, every wall
    with the same camera and, where its own lines settled it, its own
    orientation. It is the single source of truth for where tiles go.

    `wall_index` renders one wall on its own — the index `wall_instances`
    gives it — through the reference engine's single-wall mode. That wall is
    then its own metric anchor, sized from the room height. Rendering every wall
    together shares one anchor between them, and on rooms whose walls sit at
    very different depths that shrinks the others to a few centimetres: measured
    on an atrium, three of four walls reconstructed 11-32 mm wide, where alone
    each came out 2.0-2.6 m wide and 3.05 m tall.

    `fallback_vp` is an (x, y) vanishing point from existing geometry — the
    backend's estimated camera. The reference engine takes the caller's manual
    vanishing point as its fallback when its own detection is rejected; its
    frontend never sent one in automatic mode, so a rejected detection fell
    through to a plane fitted on depth. Supplying one here gives that fallback
    something to use. An accepted detection still wins.

    Raises `ValueError` when nothing could be tiled, with the reason.
    """
    if surface not in SURFACES:
        raise ValueError(f"surface must be one of {SURFACES}; got {surface!r}")

    floor = _binary(floor_mask)
    wall = _binary(wall_mask)

    shape = clean_rgb.shape[:2]

    if floor.shape != shape or wall.shape != shape:
        raise ValueError(
            f"masks must match the clean room's size {shape}; got floor {floor.shape}, "
            f"wall {wall.shape}"
        )

    if (floor & wall).any():
        raise ValueError("floor and wall masks overlap; they must be mutually exclusive")

    room_bgr = cv2.cvtColor(np.ascontiguousarray(clean_rgb), cv2.COLOR_RGB2BGR)
    tile_bgra = cv2.cvtColor(np.ascontiguousarray(tile_rgb), cv2.COLOR_RGB2BGRA)

    exif_focal_px = None

    if photo_bytes:
        exif_focal_px, _exif = estimate_focal_from_exif(photo_bytes, shape[1])

    # One MiDaS pass for the whole frame, shared by both surfaces. The
    # reference frontend then asked `/depth` for a per-surface map — the same
    # heatmap re-spread across that surface's mask — so each surface below
    # gets exactly that.
    frame_depth = depth.estimate_depth_heatmap(clean_rgb)

    out_bgr = room_bgr.copy()
    wall_regions: list = []
    floor_tiled = np.zeros(shape, dtype=bool)
    wall_tiled = np.zeros(shape, dtype=bool)
    info: dict = {"surfaces": {}, "skipped": {}}

    # ---- the canonical room: ONE metric scale for floor and walls ----
    #
    # The room frame was measured from the image when the room was cleaned;
    # the typed room size (already millimetres) is applied here. Its single
    # scale -- the camera height -- drives the floor, and its wall planes (in
    # millimetres, from each wall's floor junction) drive the walls.
    metric_planes = None
    if geometry and geometry.get("room_frame"):
        from dataclasses import replace as _replace
        from .room import geometry as room_geometry

        room = room_geometry.solve(geometry["room_frame"], {
            "width": request.room_width_mm, "length": request.room_length_mm,
            "height": request.room_height_mm,
        }, floor_mask=floor)
        info["room"] = room
        request = _replace(request, camera_height_mm=float(room["camera_height_mm"]))
        metric_planes = {
            int(wid.split("-")[1]): plane
            for wid, plane in (room.get("wall_planes_mm") or {}).items()
        } or None

    # ---------------------------------------------------------------- floor
    if surface in ("floor", "both"):
        if not floor.any():
            info["skipped"]["floor"] = "FLOOR_MASK.png has no floor pixels"
        else:
            opts = _options(request, "floor", exif_focal_px, fallback_vp)

            # Depth matters to the floor too. When its vanishing point is
            # accepted the plane comes from the horizon and depth cancels out;
            # when it is rejected (confidence under the 0.55 gate, which is
            # common) the reference pipeline fits the plane to depth instead.
            # Measured: a flat depth map turned 2 tiles across into 105 on
            # three of five rooms. So the floor gets the real depth, always.
            floor_depth = depth.surface_depth_heatmap(frame_depth, floor)

            floor_evidence, camera = _stored_floor(geometry, shape, fallback_vp)

            try:
                rendered, floor_info = render_floor_with_info(
                    room_bgr, floor_depth, _mask_bgr(floor), tile_bgra, opts,
                    evidence=floor_evidence, focal=camera,
                )

                rendered, floor_info = _retry_out_of_view_floor(
                    rendered, floor_info, room_bgr, floor_depth, floor, tile_bgra, opts, fallback_vp
                )
            except (ValueError, MetricScaleError) as error:
                # The reference app rendered one surface per request, so a
                # failure there cost only that surface. Keep it that way.
                if surface == "floor":
                    raise
                info["skipped"]["floor"] = str(error)
            else:
                clipped, report = compositing.clip(rendered, room_bgr, floor)

                out_bgr[floor] = clipped[floor]
                floor_tiled = np.any(clipped != room_bgr, axis=2) & floor

                if not floor_tiled.any():
                    info["skipped"]["floor"] = (
                        "the floor plane the engine settled on meets no floor pixel "
                        f"(vanishing point {floor_info.get('vp_source')}, confidence "
                        f"{floor_info.get('vp_confidence', 0):.2f})"
                    )

                info["surfaces"]["floor"] = {**floor_info, "clip": report}

    # ----------------------------------------------------------------- wall
    if surface in ("wall", "both"):
        if not wall.any():
            info["skipped"]["wall"] = "WALL_MASK.png has no wall pixels"
        else:
            # No fallback vanishing point: the reference frontend never sent
            # one, and the wall pipeline takes its planes from the room's own
            # horizontal vanishing directions.
            wall_point = None

            _floor_ev, camera = _stored_floor(geometry, shape, None)
            focal_px = None if camera is None else camera[0]
            normals = _stored_wall_normals(geometry)

            # The room's stored wall split, when its geometry carries one: the
            # render then tiles exactly those walls instead of splitting again.
            stored_walls = _stored_wall_masks(geometry, shape, wall)

            if wall_index is not None:
                owned = stored_walls if stored_walls is not None else dict(
                    wall_instances(
                        clean_rgb, wall, request, exif_focal_px=exif_focal_px,
                        frame_depth=frame_depth, focal_px=focal_px,
                    )
                )

                if wall_index not in owned:
                    raise ValueError(f"This room has no wall {wall_index}.")

                wall_point = interior_point(owned[wall_index])

            opts = _options(request, "wall", exif_focal_px, wall_point=wall_point)
            if geometry and geometry.get("room_frame"):
                from dataclasses import replace as _with
                opts = _with(opts, vertical_vp=geometry["room_frame"].get("vertical_vp"))

            # The reference frontend requested depth with the wall mask, so
            # `/depth` re-spread it across the wall before the renderer saw it.
            depth_bgr = depth.surface_depth_heatmap(frame_depth, wall)

            try:
                rendered, wall_info = render_wall_with_info(
                    room_bgr, depth_bgr, _mask_bgr(wall), tile_bgra, opts,
                    focal=camera, plane_normals=normals, metric_planes=metric_planes,
                    instance_masks=None if stored_walls is None else sorted(stored_walls.items()),
                )
            except (ValueError, MetricScaleError) as error:
                if surface == "wall":
                    raise
                info["skipped"]["wall"] = str(error)
            else:
                allowed = wall if wall_region is None else wall & _binary(wall_region)

                clipped, report = compositing.clip(rendered, room_bgr, allowed)

                out_bgr[allowed] = clipped[allowed]
                wall_tiled = np.any(clipped != room_bgr, axis=2) & allowed

                info["surfaces"]["wall"] = {**wall_info, "clip": report}

                wall_regions = _wall_regions(
                    clean_rgb, wall, request, exif_focal_px, frame_depth, wall_info, wall_tiled, info,
                    focal_px=focal_px, stored=stored_walls,
                )

    if not (floor_tiled.any() or wall_tiled.any()):
        reasons = "; ".join(f"{k}: {v}" for k, v in info["skipped"].items()) or "no tile landed"
        raise ValueError(f"Nothing could be tiled for surface={surface!r} ({reasons}).")

    info["outside_mask_pixels"] = int(
        (np.any(out_bgr != room_bgr, axis=2) & ~(floor | wall)).sum()
    )

    return RoomRender(
        image=cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB),
        floor_tiled=floor_tiled,
        wall_tiled=wall_tiled,
        info=info,
        wall_regions=wall_regions,
    )


# A floor render that tiles less than this share of FLOOR_MASK has a plane that
# the mask contradicts. See `_retry_out_of_view_floor`.
FLOOR_MIN_COVERAGE = 0.5


def _retry_out_of_view_floor(rendered, floor_info, room_bgr, floor_depth, floor, tile_bgra, opts, fallback_vp):
    """
    Re-render a floor whose accepted vanishing point put the plane out of view.

    The engine's confidence gate accepts a detection on how self-consistent its
    lines are, not on whether the floor they imply is the floor in the mask.
    Every visible floor pixel lies below the floor's horizon, so a correct
    horizon lets the floor plane reach essentially all of FLOOR_MASK. Measured
    on a staircase hall: the curved stair edges produced a vanishing point at
    (58, 1217) on a 1104-row image, confidence 0.74 over the 0.55 gate — a
    horizon under the whole floor, a camera pitched 51 degrees UP, and no floor
    pixel tiled.

    So when less than `FLOOR_MIN_COVERAGE` of the mask is tiled and there is a
    fallback vanishing point, the floor is rendered once more through the
    engine's own manual-VP path with that point, and whichever render tiles
    more of the floor is kept. A floor the detector got right never gets here.
    """
    if fallback_vp is None or not floor.any():
        return rendered, floor_info

    def coverage(image):
        return float((np.any(image != room_bgr, axis=2) & floor).sum()) / float(floor.sum())

    first = coverage(rendered)

    if first >= FLOOR_MIN_COVERAGE:
        return rendered, floor_info

    retry_opts = replace(
        opts,
        auto_detect_vp=False,
        vanishing_point_x=float(fallback_vp[0]),
        vanishing_point_y=float(fallback_vp[1]),
    )

    retried, retry_info = render_floor_with_info(
        room_bgr, floor_depth, _mask_bgr(floor), tile_bgra, retry_opts
    )

    second = coverage(retried)

    note = {
        "rejected_vp_source": floor_info.get("vp_source"),
        "rejected_vp_confidence": floor_info.get("vp_confidence"),
        "coverage_with_detected_vp": round(first, 4),
        "coverage_with_fallback_vp": round(second, 4),
    }

    if second <= first:
        floor_info["out_of_view_retry"] = {**note, "kept": "detected"}
        return rendered, floor_info

    retry_info["out_of_view_retry"] = {**note, "kept": "fallback"}
    retry_info["vp_source"] = "fallback-after-out-of-view"

    return retried, retry_info


def _wall_regions(clean_rgb, wall, request, exif_focal_px, frame_depth, wall_info, wall_tiled, info,
                  focal_px=None, stored=None):
    """
    Each rendered wall's tiled pixels, matched to the render by index.

    A wall the render skipped gets no region. If the repeated split ever
    disagreed with the render — a wall's pixel count not matching what the
    render reported — the walls are returned as one region rather than risk a
    toggle that shows or hides the wrong wall's tiles.
    """
    rendered = {entry["index"]: entry for entry in wall_info.get("walls", [])}

    regions = []
    mismatch = False

    split = sorted(stored.items()) if stored is not None else wall_instances(
        clean_rgb, wall, request, exif_focal_px=exif_focal_px, frame_depth=frame_depth,
        focal_px=focal_px,
    )

    for index, mask in split:
        entry = rendered.get(index)

        if entry is None:
            continue

        if int(entry.get("pixels", -1)) != int(mask.sum()):
            mismatch = True
            break

        region = mask & wall_tiled

        if region.any():
            regions.append({"index": index, "mask": region})

    if mismatch:
        info["wall_regions"] = "split did not match the render; walls returned as one region"

        return [{"index": 0, "mask": wall_tiled.copy()}] if wall_tiled.any() else []

    # The pipeline paints walls in this order, each over the last, so where two
    # walls' masks touch the later one's tile is what the render shows. Give
    # every pixel to exactly that wall, so the regions never overlap.
    claimed = np.zeros_like(wall_tiled)

    for region in reversed(regions):
        region["mask"] &= ~claimed
        claimed |= region["mask"]

    return [region for region in regions if region["mask"].any()]
