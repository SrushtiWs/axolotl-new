"""
Estimated floor geometry for an uploaded room.

`/generate` normally needs the calibrated scene: a floor mask and a camera the
GPU pipeline fitted to one specific photo. This module builds a *plausible*
version of both for any photo, so an upload can still be tiled:

  * the floor mask comes from the live SegFormer segmentation,
  * the camera is synthesised — a level, unrolled pinhole at a typical standing
    height, with its pitch solved so the far edge of the predicted floor lands
    where the photo actually shows the floor meeting the back wall.

That last step is what makes the room dimensions on the form matter: the
solved pitch depends on how long the user says the room is.

Nothing in engine.py changes. This produces the same `Scene` the calibrated path
produces, so the identical, verified metric projection runs on top of it.

The geometry is an estimate, not a measurement. A fitted camera reprojects the
real room box to within a few pixels; this one assumes the camera is level, the
lens is rectilinear with a typical field of view, and the photographer stood at
one end of the room.
"""

from __future__ import annotations

import math

import numpy as np

from engine import MissingGeometryError
from scene import Scene, WallPlane
from surfaces import Instance

MM_PER_FOOT = 304.8

# Horizontal field of view assumed for the lens, as focal length / image width.
# 0.75 is about 67 degrees, typical for an interior phone or real-estate shot.
FOCAL_RATIO = 0.75

# Where the camera is assumed to be, above the floor.
DEFAULT_CAMERA_HEIGHT_MM = 1500.0

# The camera never sits higher than this fraction of the stated room height.
MAX_CAMERA_HEIGHT_FRACTION = 0.6

# A floor smaller than this fraction of the frame is not enough to tile.
MIN_FLOOR_FRACTION = 0.01

# The far floor edge is taken as this percentile of floor rows, so a few stray
# pixels high in the frame cannot drag the horizon with them.
FLOOR_TOP_PERCENTILE = 2.0

# Pitch search bounds, in degrees. Negative means the camera tilts upward.
PITCH_MIN_DEG = -20.0
PITCH_MAX_DEG = 80.0

# The tile grid is clipped to this multiple of the stated room size, so a room
# whose dimensions are guessed low does not lose its tiles to clipping.
CLIP_MARGIN = 6.0

# Largest enclosed gap in a surface mask that may be sealed, as a fraction of
# that mask. Above it a gap is a real feature — a doorway, or an object standing
# against the surface — and sealing it would tile straight over that object.
SURFACE_HOLE_FRACTION = 0.02

# Detached specks below this fraction of a surface are class-map noise.
SURFACE_SPECK_FRACTION = 0.001

# Guided-filter window for snapping a surface boundary onto the photo's own
# edges, as a fraction of the longest edge. The same value the extraction
# pipeline uses on object masks.
SURFACE_EDGE_FRACTION = 0.0035

# Edge refinement that keeps less than this much of a surface has disagreed with
# the class map rather than tidied it, and is discarded.
MIN_SURFACE_AFTER_REFINE = 0.80


def _rotation(pitch_rad: float) -> np.ndarray:
    """
    World-to-camera rotation for a level camera pitched down by `pitch_rad`.

    Rows are the camera axes in world coordinates: x right, y down, z forward,
    matching the convention the fitted stage08g3 camera uses. Image x increases
    toward world -X, which is the direction `engine` calls U.
    """
    cos, sin = math.cos(pitch_rad), math.sin(pitch_rad)

    return np.array(
        [
            [-1.0, 0.0, 0.0],
            [0.0, -cos, -sin],
            [0.0, -sin, cos],
        ],
        dtype=np.float64,
    )


def _far_edge_y(pitch_rad: float, focal: float, cy: float, height_mm: float, length_mm: float) -> float:
    """
    Image row where the floor meets the back wall, for a given pitch.

    The junction point sits on the floor directly ahead, `length_mm` away. Its
    projected row falls as the camera pitches further down, which is what makes
    the solve below a simple bisection.
    """
    cos, sin = math.cos(pitch_rad), math.sin(pitch_rad)

    forward = height_mm * sin + length_mm * cos

    if forward < 1e-6:
        return -math.inf

    return cy + focal * (height_mm * cos - length_mm * sin) / forward


def _solve_pitch(
    target_y: float, focal: float, cy: float, height_mm: float, length_mm: float
) -> float:
    """Find the pitch that puts the floor/wall junction at `target_y`."""
    low = math.radians(PITCH_MIN_DEG)
    high = math.radians(PITCH_MAX_DEG)

    # _far_edge_y decreases monotonically in pitch, so clamp outside the range.
    if _far_edge_y(low, focal, cy, height_mm, length_mm) < target_y:
        return low

    if _far_edge_y(high, focal, cy, height_mm, length_mm) > target_y:
        return high

    for _ in range(60):
        middle = (low + high) / 2.0

        if _far_edge_y(middle, focal, cy, height_mm, length_mm) > target_y:
            low = middle
        else:
            high = middle

    return (low + high) / 2.0


def _wall_planes(
    clip_u: float, room_u_mm: float, room_v_mm: float, room_h_mm: float
) -> tuple[WallPlane, ...]:
    """
    The three walls a room photo can show, placed around the estimated camera.

    The camera sits at `u = clip_u / 2` — the middle of the clipped floor, at its
    near edge — so the real room is centred on it across `u` and runs forward to
    `room_v_mm`. That puts its walls at

        image-left    X = room_u/2 - clip_u/2      running back along Z
        image-right   X = -(clip_u/2 + room_u/2)   running back along Z
        back          Z = room_v_mm                running across X

    Note these use the *stated* room size, not the generously clipped floor
    extent. The clip margin exists so a room bigger than stated does not lose
    its floor tiles to cropping; applying it to the walls would push them six
    times too far away and tile them at a sixth of the correct scale.

    Each wall's horizontal direction is signed so that the tile grid runs the
    way the wall actually recedes in the image: on the left wall the far end
    lies toward the middle of the frame, on the right wall it lies the other
    way, and on the back wall across is simply left to right.
    """
    half_u = room_u_mm / 2.0

    left_x = half_u - clip_u / 2.0
    right_x = -(clip_u / 2.0 + half_u)

    return (
        WallPlane(
            label="left",
            axis=0,
            value=left_x,
            horizontal_axis=2,
            horizontal_sign=1.0,
            horizontal_origin=0.0,
            width_mm=room_v_mm,
            height_mm=room_h_mm,
        ),
        WallPlane(
            label="right",
            axis=0,
            value=right_x,
            horizontal_axis=2,
            horizontal_sign=-1.0,
            horizontal_origin=room_v_mm,
            width_mm=room_v_mm,
            height_mm=room_h_mm,
        ),
        WallPlane(
            label="back",
            axis=2,
            value=room_v_mm,
            horizontal_axis=0,
            horizontal_sign=-1.0,
            horizontal_origin=left_x,
            width_mm=room_u_mm,
            height_mm=room_h_mm,
        ),
    )


def _clean_surface(mask: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    """
    Tidy a surface mask before it decides where tiles land.

    Every *object* mask in this system is cleaned and edge-refined before it is
    used. The wall and floor masks never were: `surfaces.instances` returns
    SegFormer's raw argmax and `live_scene` handed it straight to `Scene`, so
    `engine._project_floor` painted tiles through whatever boundary a 512x512
    class map happened to produce. The result is visible in a render as a torn
    edge along the top of a wall and ragged notches in the floor — the tile
    stops in a shape no surface in the room actually has.

    Two passes, both reusing what the extraction pipeline already does.

    **Enclosed gaps are sealed.** A speckle hole inside a wall is a class-map
    artefact, not an opening, and it leaves an untiled blot mid-wall. Measured
    across three rooms the wall masks carried 1,743, 7,300 and 5,466 pixels of
    such holes. The cap is what keeps this safe: a hole larger than
    `SURFACE_HOLE_FRACTION` of the mask is a real feature — a doorway, or an
    object standing against the surface — and is left alone, so a piece of
    furniture the detector missed cannot be tiled over by this.

    **The boundary is snapped to the photograph's own edges.** `snap_to_edges`
    is the same guided filter used on object masks, with the same guarantees:
    it may only move the edge within a band, pixels well inside are kept, and
    nothing beyond the band can be added. So the wall stops where the photo
    shows it stopping, and the mask cannot grow across the room.
    """
    if not mask.any():
        return mask

    import cv2

    import matting
    from extraction import cleanup

    area = int(mask.sum())

    radius = int(np.clip(round(SURFACE_EDGE_FRACTION * max(rgb.shape[:2])), 3, 12))

    snapped = matting.snap_to_edges(mask.astype(np.float32), rgb, radius, max(2, radius))

    # Refinement that disagrees with the class map wholesale is not refinement.
    # Falling back to the raw mask keeps the tidying below and discards the move.
    if int(snapped.sum()) < area * MIN_SURFACE_AFTER_REFINE:
        snapped = mask

    # Sealing runs *after* the snap, not before it. The guided filter follows
    # the photo's gradients, and a floor's own veining and reflections are
    # gradients — so snapping a clean mask punches a scatter of pinholes into
    # it. Measured on two rooms whose floors had no enclosed holes at all, the
    # snap alone opened eight in each. Tidying last removes what the snap
    # introduced as well as what SegFormer produced, which is why this order is
    # the one that leaves a mask with zero holes rather than eight.
    sealed = cleanup.fill_small_holes(
        snapped, max(16, int(area * SURFACE_HOLE_FRACTION))
    )

    return cleanup.drop_small_components(
        sealed, max(16, int(area * SURFACE_SPECK_FRACTION))
    )


def _union(instances: list[Instance], shape: tuple[int, int]) -> np.ndarray:
    combined = np.zeros(shape, dtype=bool)

    for instance in instances:
        combined |= instance.mask

    return combined


def _props_alpha(mask: np.ndarray, rgb: np.ndarray) -> np.ndarray | None:
    """
    Soft coverage for the restored objects, from the photo's own edges.

    `matting.refine` is the extraction pipeline's matting pass, reused rather
    than reimplemented: it lifts the binary union into an alpha that follows the
    image's gradients, confined to a couple of pixels either side of the
    boundary. Well inside stays fully opaque and well outside fully
    transparent, so no part of an object is softened — only the edge it meets
    the new tiles along stops being a staircase.

    The masks are unioned *before* matting, never after. Matting each object
    separately and stacking the results feathers every internal boundary as
    though the object behind were background, leaving a seam wherever two
    objects touch.
    """
    if not mask.any():
        return None

    import matting

    return matting.refine(mask, rgb)


def build(
    rgb: np.ndarray,
    objects: list[Instance],
    surfaces: list[Instance],
    room_width_ft: float,
    room_length_ft: float,
    room_height_ft: float,
    wall_labels: tuple[str, ...] | None = None,
    clean: np.ndarray | None = None,
    props: np.ndarray | None = None,
    object_count: int | None = None,
) -> tuple[Scene, dict]:
    """
    Assemble a `Scene` for an uploaded photo.

    Returns the scene and the geometry estimate that produced it, so the
    response can say what was assumed.

    `wall_labels` restricts which of the room's walls are tiled. `None` means
    all of them, which is what every existing caller gets. This is a filter on
    the plane list and nothing more: the wall *detection* is unchanged, and so
    is the projection — `engine._project_walls` still intersects every plane it
    is given and keeps the nearest hit per pixel. Leaving a wall out simply
    means no plane claims those pixels, so they keep the original photograph.

    `clean` is the same room with its objects removed and the surfaces behind
    them rebuilt, from the inpainting stage. When it is given it becomes the
    scene's `empty_room` — the base the tiles are projected onto and the image
    the shading is read from — which is what that field has always meant and
    what this pipeline previously had no way to supply. `master_input` stays the
    original photograph regardless, because that is where the objects are copied
    back from, so they return with their own pixels at their own coordinates.

    Passing nothing gives exactly the previous behaviour: the photograph doubles
    as the empty room, objects and all.

    `props` is the union of the accepted object masks, when the caller already
    has it. `/segment` writes `ALL_OBJECTS.png` and `MIRRORS_ONLY.png` from a
    set of masks it accepted; handing that same union back here is what makes
    the objects restored over the tiles land on exactly the boundaries those
    files were cut along, instead of on a second, independently-computed
    opinion. Omitted, it is unioned from `objects` as before. `object_count`
    likewise only labels the estimate — it exists because a caller supplying
    `props` has no per-object list left to count.
    """
    height, width = rgb.shape[:2]

    # The surfaces are cleaned against the emptied room when there is one: a
    # floor boundary snapped to the photograph's edges snaps to the legs of the
    # chair standing on it, and there is no chair in the clean room.
    base = rgb if clean is None else clean

    floor = next((item.mask for item in surfaces if item.label == "floor"), None)

    if floor is not None:
        floor = _clean_surface(floor, base)

    if floor is None or floor.sum() < height * width * MIN_FLOOR_FRACTION:
        raise MissingGeometryError(
            "No floor was found in this photo. The segmentation needs to see a "
            "clear stretch of floor to project tiles onto it — try a photo that "
            "shows more of the ground."
        )

    room_u_mm = max(room_width_ft, 0.1) * MM_PER_FOOT
    room_v_mm = max(room_length_ft, 0.1) * MM_PER_FOOT
    room_h_mm = max(room_height_ft, 0.1) * MM_PER_FOOT

    camera_height = min(DEFAULT_CAMERA_HEIGHT_MM, room_h_mm * MAX_CAMERA_HEIGHT_FRACTION)

    focal = FOCAL_RATIO * width
    cx, cy = width / 2.0, height / 2.0

    rows = np.nonzero(floor)[0]

    far_edge_y = float(np.percentile(rows, FLOOR_TOP_PERCENTILE))

    pitch = _solve_pitch(far_edge_y, focal, cy, camera_height, room_v_mm)

    K = np.array([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    R = _rotation(pitch)

    # Clip generously: the stated room size drives the pitch solve and the tile
    # scale, but it must not crop tiles out of a room that is bigger than stated.
    clip_u = room_u_mm * CLIP_MARGIN
    clip_v = room_v_mm * CLIP_MARGIN

    # Sit the camera halfway across the clipped floor, at the near edge of it,
    # so the visible floor falls inside [0, clip_u] x [0, clip_v].
    centre = np.array([-clip_u / 2.0, camera_height, 0.0], dtype=np.float64)

    # Every visible wall pixel the segmentation found. Which plane each of them
    # belongs to is decided per pixel at render time, by which wall the ray
    # actually reaches first — so a corner needs no special case here.
    wall = next((item.mask for item in surfaces if item.label == "wall"), None)

    if wall is None:
        wall = np.zeros((height, width), dtype=bool)
    else:
        wall = _clean_surface(wall, base)

        # Surface classes are independently predicted, so a few boundary
        # pixels can be claimed by both wall and an adjacent structural
        # surface. The wall mask is the final authority for material placement;
        # keep those existing ceiling/floor exclusions explicit without
        # changing either the metric wall planes or their camera geometry.
        ceiling = next((item.mask for item in surfaces if item.label == "ceiling"), None)
        floor_surface = next((item.mask for item in surfaces if item.label == "floor"), None)

        if ceiling is not None:
            wall &= ~ceiling

        if floor_surface is not None:
            wall &= ~floor_surface

    detected = _wall_planes(clip_u, room_u_mm, room_v_mm, room_h_mm) if wall.any() else ()

    # Every wall the room box has is reported; only the chosen ones are tiled.
    walls = (
        detected
        if wall_labels is None
        else tuple(plane for plane in detected if plane.label in wall_labels)
    )

    props = _union(objects, (height, width)) if props is None else props

    scene = Scene(
        empty_room=base,
        master_input=rgb,
        floor=floor,
        # The estimated scene tiles whole walls rather than a skirting course,
        # so the screeding band stays empty and `walls` carries the geometry.
        screeding=np.zeros((height, width), dtype=bool),
        props=props,
        props_alpha=_props_alpha(props, rgb),
        K=K,
        R=R,
        t=-R @ centre,
        camera_center=centre,
        back_x_image=cx,
        room_u_mm=clip_u,
        room_v_mm=clip_v,
        room_height_mm=room_h_mm,
        wall=wall,
        walls=walls,
        # An ordinary photograph, not an even render: every original tile has
        # its own tone, and a blurred luminance cannot tell that from lighting.
        # See `engine.ILLUM_POLY_ORDER`.
        illumination="polynomial",
    )

    estimate = {
        "method": "estimated",
        "focal_px": round(focal, 2),
        "horizontal_fov_deg": round(math.degrees(2 * math.atan(0.5 / FOCAL_RATIO)), 1),
        "camera_pitch_deg": round(math.degrees(pitch), 2),
        "camera_height_mm": round(camera_height, 1),
        "far_floor_edge_y": round(far_edge_y, 1),
        "room_mm": [round(room_u_mm, 1), round(room_v_mm, 1), round(room_h_mm, 1)],
        "floor_pixels": int(floor.sum()),
        "wall_pixels": int(wall.sum()),
        # What is being tiled...
        "walls": [
            {
                "label": plane.label,
                "width_mm": round(plane.width_mm, 1),
                "height_mm": round(plane.height_mm, 1),
            }
            for plane in walls
        ],
        # ...and every wall the room has, so a caller can offer the choice.
        "walls_detected": [plane.label for plane in detected],
        "walls_selected": [plane.label for plane in walls],
        "object_count": len(objects) if object_count is None else int(object_count),
        # Whether the tiles were projected onto a room with its objects removed
        # and the surfaces behind them rebuilt, or onto the raw photograph.
        "empty_room": "inpainted" if clean is not None else "original photograph",
    }

    return scene, estimate
