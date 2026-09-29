"""
Wall pipeline.

Same shape as the floor pipeline and the same core/, with every
surface-specific decision coming from surface/wall/. The one structural
difference is the loop: a floor renders one plane, a wall renders one plane
PER DETECTED WALL, compositing each onto the previous result so a corner shot
comes back with both walls tiled correctly rather than one plane smeared
across both.

    Room image
      -> wall segmentation (openings removed)      surface/wall/segment.py
      -> instance split into separate planes       surface/wall/instances.py
      for each instance:
        -> wall VP: plumb + along-wall            surface/wall/vp.py
        -> plane constraint (plumb)               surface/wall/constraints.py
        -> gravity-aligned basis                  core/raycast.py
        -> metric scale                           surface/wall/scale.py
        -> junction / corner anchor               surface/wall/anchor.py
        -> rays, UV, orientation                  core/raycast.py, core/uv.py
        -> grout, lighting, composite             core/composite.py
"""

import math

import numpy as np

from ..camera.focal_estimate import resolve_focal_length
from ..camera.metric_scale import MetricScaleError
from ..core import VP_DEBUG_INFO, debug as _debug
from ..core import composite as _composite
from ..core import raycast as _raycast
from ..core import uv as _uv
from ..core.plane_fit import WallConstraint, backproject, fit_plane_ransac
from .. import three_layer
from ..surface.base import SurfaceInstance, SurfaceKind, SurfaceProfile
from ..surface.wall import anchor as wall_anchor
from ..surface.wall import constraints as wall_constraints
from ..surface.wall import instances as wall_instances
from ..surface.wall import scale as wall_scale
from ..surface.wall import vp as wall_vp

#: Largest reconstructed wall dimension treated as real. Bigger than any
#: interior wall, small enough to catch a grazing plane fit whose rays diverge.
MAX_PLAUSIBLE_WALL_MM = 30000.0

#: A caller-given wall orientation is used only when its plane reaches at least
#: this share of the wall's own mask pixels (see _geometry).
GIVEN_NORMAL_MIN_MASK_REACH = 0.98

#: Target band for the wall's depth re-normalisation.
#:
#: NOT 0..1. Back-projection is Z = 1000 / (d + 0.05), so a pixel pushed to
#: d = 0 lands at Z = 20000 while its neighbours sit near 1000, and the plane
#: fit is then dominated by that 21x spread. Measured on a real room, the
#: wall's native depth spanned 0.074..0.912 and normalising to a full 0..1
#: tripled the spread of magnitudes the fit had to cope with. This band keeps
#: clear of the pole while still expanding the contrast.
DEPTH_BAND_LO = 0.18
DEPTH_BAND_HI = 0.95


class WallProfile(SurfaceProfile):
    """The wall's answers to the SurfaceProfile contract."""

    kind = SurfaceKind.WALL
    # No empirical squash: the floor's 1.20 is a hand-tuned floor value and
    # applying it to a wall visibly compresses the course.
    tile_height_stretch = 1.0

    def __init__(self, opts=None):
        self.opts = opts

    def instances(self, surface_bool, room_bgr, depth_val, opts, cx=None, cy=None, f=None):
        h, w = surface_bool.shape[:2]
        cx = w / 2.0 if cx is None else cx
        cy = h / 2.0 if cy is None else cy
        f = max(h, w) * 1.2 if f is None else f
        return wall_instances.detect_instances(
            surface_bool, room_bgr, depth_val, cx, cy, f, opts
        )

    def detect_geometry(self, room_bgr, instance, opts, cx, cy):
        return wall_vp.resolve(room_bgr, instance.mask, opts, cx, cy)

    def plane_constraint(self, opts):
        return wall_constraints.plane_constraint(opts)

    def basis_up_hint(self, plane):
        return wall_constraints.gravity_up_hint()

    def rotation_baseline(self, evidence, opts, cx, cy, f, e_u, e_v) -> float:
        # NOT the VP angle. See surface/wall/constraints.py: a wall's grid
        # follows gravity unless plumb evidence is both confident and small.
        return wall_constraints.rotation_baseline(evidence, opts)

    def anchor_offsets(self, instance, u_rot, v_rot, valid_ray, mm_per_unit, opts):
        return wall_anchor.offsets(
            instance.mask, u_rot, v_rot, valid_ray, mm_per_unit, opts
        )

    def resolve_scale(self, plane, instance, opts, context):
        return wall_scale.resolve(
            instance, opts,
            u_units=context.get("u_units"),
            v_units=context.get("v_units"),
            visible=context.get("visible"),
            floor_mm_per_unit=context.get("floor_mm_per_unit"),
            door_height_units=context.get("door_height_units"),
        )


def render_wall(room_bgr, depth_bgr, mask_bgra, tile_bgra, opts,
                floor_mm_per_unit=None, openings_mask=None):
    """Returns the rendered BGR image."""
    result, _info = render_wall_with_info(
        room_bgr, depth_bgr, mask_bgra, tile_bgra, opts,
        floor_mm_per_unit=floor_mm_per_unit, openings_mask=openings_mask
    )
    return result


def render_wall_with_info(room_bgr, depth_bgr, mask_bgra, tile_bgra, opts,
                          floor_mm_per_unit=None, openings_mask=None,
                          focal=None, plane_normals=None, instance_masks=None,
                          metric_planes=None):
    """
    As render_wall, plus per-instance diagnostics.

    floor_mm_per_unit lets the caller hand over the scale the floor path
    resolved for the SAME photograph -- the strongest automatic anchor a wall
    has. openings_mask enables the door-height prior.

    `focal` and `plane_normals` let a caller hand over geometry it detected
    beforehand: the camera's focal length as (f, focal_info), and a wall
    orientation (a, b, c) per instance index. A wall listed there is projected
    on that orientation instead of one chosen from the room's vanishing points.
    Both default to None, which is exactly the behaviour before they existed.
    """
    profile = WallProfile(opts)

    h_img, w_img = room_bgr.shape[:2]
    cx, cy = w_img / 2.0, h_img / 2.0

    depth_bgr, mask_bgra, tile_bgra, depth_val, mask_had_alpha = _composite.normalise_inputs(
        room_bgr, depth_bgr, mask_bgra, tile_bgra, opts.invert_depth
    )
    wall_bool, mask_factor = _composite.decode_mask(
        mask_bgra, mask_had_alpha, surface_label="wall"
    )

    # ---- Per-surface depth ----
    # ON by default for a wall: the incoming depth map is normalised across the
    # whole frame, where the floor owns the range, so the wall's own variation
    # is compressed to near-nothing and every wall fits back nearly
    # fronto-parallel. Re-spreading the range inside the wall mask is what lets
    # a side wall recover its real angle.
    if opts.depth_normalisation_enabled(default=True):
        depth_val = _composite.normalise_depth_within_mask(
            depth_val, wall_bool, out_lo=DEPTH_BAND_LO, out_hi=DEPTH_BAND_HI
        )
        _debug(f"[depth] re-normalised within the wall mask "
               f"-> [{DEPTH_BAND_LO}, {DEPTH_BAND_HI}]")

    # ---- Focal length, once for the whole image ----
    # Resolved before instance splitting because back-projection needs it, and
    # a wall gives no two-VP calibration of its own -- EXIF or the FOV prior.
    if focal is not None:
        f, focal_info = focal
    else:
        f, focal_info = resolve_focal_length(
            image_width=w_img,
            exif_focal_px=opts.exif_focal_px,
            vp1=None,
            vp2=None,
            principal_point=(cx, cy),
            fallback_focal=opts.focal_length,
            auto=opts.auto_focal_length,
        )
    print(
        f"[focal] f={f:.1f}px ({focal_info['hfov_deg']:.1f} deg HFOV) "
        f"source={focal_info['source']}"
    )

    # ---- Split into independent walls ----
    # ---- Room horizontal directions, once for the whole image ----
    # A rectangular room has exactly two, every wall contains one of them, and
    # recovering them from all wall pixels together is far more reliable than
    # asking each fragment for its own (two of three walls carried 0 and 1
    # horizontal lines respectively).
    room_vps, vps_info = wall_vp.room_horizontal_vps(room_bgr, wall_bool)
    _debug(f"[wall] room horizontal VPs: {room_vps}  ({vps_info.get('stage')}, "
           f"{vps_info.get('horizontal_lines')} horizontal lines)")

    if instance_masks is not None:
        # The caller's own split -- the room's stored walls -- instead of
        # splitting again. None (the default) is the pipeline's own split.
        found = [SurfaceInstance(index=int(i), mask=np.asarray(m, bool) & wall_bool, label=f"wall[{int(i)}]")
                 for i, m in instance_masks]
        found = [inst for inst in found if inst.pixel_count]
    else:
        found = profile.instances(wall_bool, room_bgr, depth_val, opts, cx, cy, f)
    all_found = found
    found = _select_instances(found, opts, w_img, h_img)
    _debug(f"[wall] {len(found)} instance(s) to render "
           f"({', '.join(f'{i.label}:{i.pixel_count}px' for i in found)})")

    # ---- Metric scale, resolved ONCE for the whole image ----
    #
    # mm-per-plane-unit is a property of the CAMERA FRAME, not of an individual
    # plane: u and v come from camera-space points built off one depth map with
    # one global depth scale, so every plane in the photo converts to
    # millimetres by the same factor. Resolving it per instance is not merely
    # wasteful, it is wrong -- a clipped sliver of wall that shows only a
    # third of the room height divides the known height by a third of the
    # extent and lands on a factor several times too large. Measured on the
    # synthetic corner test that produced 2.0, 5.0, 2.4 and 36.9 mm/unit for
    # four fragments of the same two walls.
    #
    # So the anchor instance is the one spanning the most image ROWS -- the
    # best proxy for "this wall runs floor to ceiling", which is exactly the
    # assumption the room-height anchor makes -- and its factor is shared.
    geoms = {}
    metric = {int(k): tuple(float(x) for x in v) for k, v in (metric_planes or {}).items()}
    # Only planes that hold their wall count (pixels in front, >= 30 cm away).
    metric = {i: pl for i, pl in metric.items()
              if any(inst.index == i and _room_plane_ok(pl, inst.mask, cx, cy, f) for inst in all_found)}
    legacy_reason = None
    if metric and not any(inst.index in metric for inst in found):
        # The wall asked for cannot be placed by the room geometry (its floor
        # junction is hidden): it renders exactly as before, per-wall scale.
        legacy_reason = "room geometry could not place this wall; per-wall scale as before"
        metric = {}
    depth_to_mm = None
    if metric:
        # ONE metric scale, the room's: walls with a RoomGeometry plane are in
        # millimetres already (camera frame), so the wall frame is 1 mm/unit.
        # A wall the room frame could not place keeps its depth plane, brought
        # into millimetres by the depth->mm ratio of the placed walls -- MiDaS
        # supplies only RELATIVE depth, never a second scale.
        depth_to_mm = _depth_to_mm(all_found, metric, depth_val, cx, cy, f, opts)
        shared_mm_per_unit = 1.0
        scale_info = {"source": "room-geometry", "depth_to_mm": depth_to_mm}
    else:
        anchor_inst = _scale_anchor_instance(found)  # the pre-existing per-wall scale
        anchor_geom = _geometry(anchor_inst, room_bgr, depth_val, opts, profile, cx, cy, f, room_vps,
                                plane_normals)
        geoms[anchor_inst.index] = anchor_geom

        door_units = wall_scale.door_height_in_units(
            anchor_inst.mask, openings_mask, anchor_geom["v_raw"], anchor_geom["visible"]
        )
        shared_mm_per_unit, scale_info = profile.resolve_scale(
            anchor_geom["plane"], anchor_inst, opts,
            {"u_units": anchor_geom["u_raw"], "v_units": anchor_geom["v_raw"],
             "visible": anchor_geom["visible"], "floor_mm_per_unit": floor_mm_per_unit,
             "door_height_units": door_units},
        )
    if legacy_reason:
        scale_info = {**scale_info, "room_geometry": legacy_reason}
    _debug(f"[scale] wall frame: {shared_mm_per_unit:.4f} mm/unit "
           f"anchor={scale_info.get('source')}")

    out = room_bgr.copy()
    rendered = 0
    failures = []
    instance_info = []

    for instance in found:
        try:
            geom = geoms.get(instance.index) or _geometry(
                instance, room_bgr, depth_val, opts, profile, cx, cy, f, room_vps,
                plane_normals, metric=metric, depth_to_mm=depth_to_mm,
            )
            out = _render_one(
                out, room_bgr, tile_bgra, mask_factor, instance, geom, opts,
                profile, shared_mm_per_unit, scale_info, instance_info
            )
            rendered += 1
        except (MetricScaleError, ValueError) as e:
            # One wall failing must not kill the others.
            failures.append({"instance": instance.label, "error": str(e)})
            _debug(f"[wall] {instance.label} skipped: {e}")

    if rendered == 0:
        if failures:
            raise MetricScaleError(failures[0]["error"])
        raise ValueError("wall render: no wall instance could be rendered")

    info = {
        "surface": "wall",
        "instances": len(found),
        "rendered": rendered,
        "focal_px": float(f),
        "focal_source": focal_info.get("source"),
        "mm_per_unit": float(shared_mm_per_unit),
        "scale_source": scale_info.get("source"),
        "scale_caveat": scale_info.get("caveat"),
        "scale_anchor_instance": "room-geometry" if metric else anchor_inst.label,
        "room_vps": [list(v) for v in room_vps],
        "walls": instance_info,
        "failures": failures,
    }
    return out, info


def _room_plane_ok(plane, mask, cx, cy, f) -> bool:
    """A room-geometry plane holds its wall: the wall's pixels in front of the camera, >= 30 cm away."""
    return plane[3] >= 300.0 and _mask_reach(plane, mask, cx, cy, f) >= GIVEN_NORMAL_MIN_MASK_REACH


def _depth_to_mm(instances, metric, depth_val, cx, cy, f, opts):
    """Millimetres per depth unit, from walls placed by the room geometry (median)."""
    ratios = []
    for inst in instances:
        plane = metric.get(inst.index)
        if plane is None or not _room_plane_ok(plane, inst.mask, cx, cy, f):
            continue
        Xc, Yc, Zc, _xs, _ys, _ = backproject(depth_val, inst.mask, cx, cy, f, opts.depth_contrast)
        if not len(Xc):
            continue
        a, b, c, d_mm = plane
        d_depth = -(a * float(np.median(Xc)) + b * float(np.median(Yc)) + c * float(np.median(Zc)))
        if d_depth > 1e-9 and d_mm > 0:
            ratios.append(d_mm / d_depth)
    return float(np.median(ratios)) if ratios else None


def _mask_reach(plane, mask, cx, cy, f):
    """Share of `mask`'s pixels whose camera ray meets `plane` in front of the camera."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return 0.0
    a, b, c, d = plane
    den = a * (xs - cx) / f + b * (ys - cy) / f + c
    ok = (np.abs(den) >= 1e-6) & (-d / np.where(np.abs(den) < 1e-6, 1e-6, den) > 0)
    return float(ok.mean())


def _scale_anchor_instance(found):
    """
    Which wall the room-height anchor is measured on.

    Row count alone was the first attempt and it is wrong: a narrow sliver down
    the edge of the frame spans every row while being 90cm wide, and on a real
    room it won the vote, pinned ITS height to the room height, and left the
    main wall reconstructing 3729mm tall in a 2700mm room -- a 38% scale error
    on the wall the user is actually looking at.

    Score = rows spanned x sqrt(pixels). Row span still matters, because the
    anchor assumes the wall runs floor to ceiling, but it can no longer beat a
    wall with an order of magnitude more evidence behind its plane.
    """
    def score(inst):
        rows = int(np.count_nonzero(inst.mask.any(axis=1)))
        return rows * math.sqrt(max(inst.pixel_count, 1))

    return max(found, key=score)


def _select_instances(found, opts, w_img, h_img):
    """
    Narrow the detected walls down to the ones the caller asked for.

    Two ways to ask, in priority order:

      1. A POINT on the image -- what a click on the render becomes. Preferred,
         because it means "that wall, the one I am looking at", and stays
         correct when a re-render reorders instances by size. The point is
         matched to the instance whose mask contains it; if it landed on a gap
         (a window, the ceiling), the nearest instance by centroid wins so a
         near-miss still selects something sensible instead of silently
         reverting to all walls.
      2. An INDEX, for API callers that already know which one they want.

    Neither given renders every wall.
    """
    if not found:
        return found

    px, py = opts.wall_point_x, opts.wall_point_y
    if px is not None and py is not None:
        x = int(round(float(px)))
        y = int(round(float(py)))
        if 0 <= x < w_img and 0 <= y < h_img:
            for inst in found:
                if bool(inst.mask[y, x]):
                    _debug(f"[wall] point ({x}, {y}) selected {inst.label}")
                    return [inst]

            # Fell in a gap -- take the nearest wall rather than nothing.
            def _distance(inst):
                ys, xs = np.where(inst.mask)
                if len(xs) == 0:
                    return float("inf")
                return float(np.hypot(xs.mean() - x, ys.mean() - y))

            nearest = min(found, key=_distance)
            _debug(f"[wall] point ({x}, {y}) hit no wall; nearest is {nearest.label}")
            return [nearest]

    if opts.wall_instance is not None and opts.wall_instance >= 0:
        chosen = [i for i in found if i.index == opts.wall_instance]
        return chosen or found[:1]

    return found


def _geometry(instance, room_bgr, depth_val, opts, profile, cx, cy, f, room_vps=(),
              plane_normals=None, metric=None, depth_to_mm=None):
    """
    Everything about one wall instance that does not depend on metric scale:
    its plane, its direction evidence, its gravity-aligned basis, its grid
    rotation and its per-pixel plane coordinates.

    Split out from rendering so the scale pass can measure a wall before any
    wall is drawn.
    """
    h_img, w_img = room_bgr.shape[:2]
    inst_mask = instance.mask

    # Wall lines and vanishing points FIRST -- the plane is derived from them.
    evidence = profile.detect_geometry(room_bgr, instance, opts, cx, cy)
    if evidence.vp_x is not None:
        instance.vp = (evidence.vp_x, evidence.vp_y)

    Xc, Yc, Zc, xs_pts, ys_pts, _ = backproject(
        depth_val, inst_mask, cx, cy, f, opts.depth_contrast
    )
    points = (Xc, Yc, Zc)

    # ---- Wall plane ----
    #
    # Analytic from the room's horizontal vanishing directions, exactly as the
    # floor takes its plane from the horizon rather than from RANSAC. The depth
    # fit is the FALLBACK, not the primary source: decoding the Turbo-coloured
    # depth PNG warps it by ~13%, so a flat wall is not flat by the time RANSAC
    # sees it and every wall fits back near edge-on.
    given = (plane_normals or {}).get(instance.index)

    if given is not None:
        # This wall's own detected orientation. Only its distance comes from
        # the depth sample, and distance is absorbed by the metric anchor.
        a, b, c = (float(v) for v in given)
        d = -(a * float(np.median(Xc)) + b * float(np.median(Yc)) + c * float(np.median(Zc))) \
            if len(Xc) else 1000.0
        plane = WallConstraint().orient(a, b, c, d)
        if plane[3] <= 0:
            plane = (plane[0], plane[1], plane[2], abs(plane[3]) or 1000.0)
            # |d| alone moves the wall to the other side of the camera; the
            # normal that keeps the wall's own pixels in front is the right sign.
            flipped = (-plane[0], -plane[1], -plane[2], plane[3])
            if _mask_reach(flipped, inst_mask, cx, cy, f) > _mask_reach(plane, inst_mask, cx, cy, f):
                plane = flipped
        plane_source = "wall-lines"
        choose_info = {"stage": "given"}

        # The wall's mask is the truth about where it is. Every visible pixel
        # of a plane lies on the camera side of that plane's vanishing line, so
        # an orientation that leaves part of the wall's own white pixels behind
        # the camera contradicts the mask: those pixels could never be tiled.
        # Measured on a room's right wall: a given orientation reached 12,461 of
        # its 113,952 pixels. Such an orientation is dropped and the wall takes
        # the room-VP choice below, as it did before stored orientations existed.
        reach = _mask_reach(plane, inst_mask, cx, cy, f)
        choose_info["mask_reach"] = reach
        if reach < GIVEN_NORMAL_MIN_MASK_REACH:
            given = None
            rejected = {"stage": "given-rejected", "mask_reach": reach}
    if given is None:
        plane, choose_info = wall_constraints.choose_plane(room_vps, points, cx, cy, f)
        plane_source = "room-vp"
        if (plane_normals or {}).get(instance.index) is not None:
            choose_info = {**(choose_info or {}), "given_orientation": rejected}
    evidence.info["plane_choice"] = choose_info

    if plane is None:
        plane = instance.plane
        plane_source = "depth-ransac"
        if plane is None:
            plane = fit_plane_ransac(
                Xc, Yc, Zc, xs_pts, ys_pts,
                iterations=opts.ransac_iterations,
                threshold=opts.ransac_threshold,
                constraint=profile.plane_constraint(opts),
            )

    # Grazing planes reconstruct to tens of metres; face the camera instead.
    # A given orientation has already been checked against the wall's own mask
    # (every pixel in front of the camera), so a side wall running straight
    # away from the camera -- a small |c| that is real, not grazing -- is kept;
    # an implausible size is still refused in _render_one.
    if given is not None and plane[3] > 0:
        sane = plane
    else:
        sane = wall_constraints.sanitise_plane(plane, points=points)
    if sane != plane:
        plane_source += "+fronto-fallback"
    plane = sane

    # The room's single metric scale (render_wall_with_info(metric_planes=...)).
    if metric:
        placed = metric.get(instance.index)
        # A room plane is used only if it holds this wall: its pixels in front
        # of the camera and the wall no closer than 30 cm.
        if placed is not None and not _room_plane_ok(placed, inst_mask, cx, cy, f):
            evidence.info["room_plane_rejected"] = True
            placed = None
        if placed is not None:
            plane, plane_source = placed, "room-geometry"
        elif depth_to_mm:
            plane = (plane[0], plane[1], plane[2], plane[3] * depth_to_mm)
            plane_source += "+room-scale"

    instance.plane = plane
    plane_a, plane_b, plane_c, _plane_d = plane
    _debug(f"[wall] {instance.label}: plane from {plane_source} "
           f"n=({plane_a:+.3f},{plane_b:+.3f},{plane_c:+.3f}) d={plane[3]:.1f}")
    evidence.info["plane_source"] = plane_source

    e_u, e_v = _raycast.build_plane_basis(
        plane_a, plane_b, plane_c, up_hint=profile.basis_up_hint(plane)
    )
    base_rad = profile.rotation_baseline(evidence, opts, cx, cy, f, e_u, e_v)
    # Same composition rule as the floor: the surface's own baseline plus the
    # user's orientation preset (0/45/90/135), so a diagonal lay is a rotation
    # ON TOP of a plumb grid rather than a replacement for it.
    rad = base_rad + math.radians(opts.tile_rotation)

    X, Y, Z, valid_ray = _raycast.intersect_rays_with_plane(h_img, w_img, cx, cy, f, plane)
    u, v = _raycast.project_to_plane_uv(X, Y, Z, e_u, e_v)
    u_rot, v_rot = _uv.rotate(u, v, rad)

    visible = inst_mask & valid_ray
    if not np.any(visible):
        raise ValueError(f"{instance.label}: no pixel of this wall produced a valid ray")

    return {
        "plane": plane, "evidence": evidence, "e_u": e_u, "e_v": e_v,
        "base_rad": base_rad, "rad": rad,
        # Rotated: what the tile grid is laid out on.
        "u_rot": u_rot, "v_rot": v_rot,
        # UNROTATED: the wall's own level/plumb axes. Metric scale and the
        # reported wall size must both come from these -- the wall is 2.4 m
        # tall whether the tiles are laid straight or on a 45 degree diagonal,
        # and measuring the height along a rotated axis made mm-per-unit swing
        # from 5.41 to 2.04 purely because the user pressed a rotation button.
        "u_raw": u, "v_raw": v,
        "valid_ray": valid_ray, "visible": visible,
        "f": f,
    }


def _render_one(base_bgr, room_bgr, tile_bgra, mask_factor, instance, geom, opts,
                profile, mm_per_unit, scale_info, instance_info):
    """Render a single wall instance onto `base_bgr` and return the result."""
    inst_mask = instance.mask
    plane = geom["plane"]
    plane_a, plane_b, plane_c, plane_d = plane
    evidence = geom["evidence"]
    u_rot, v_rot = geom["u_rot"], geom["v_rot"]
    valid_ray, visible, rad, base_rad = (
        geom["valid_ray"], geom["visible"], geom["rad"], geom["base_rad"]
    )

    ys_i, xs_i = np.where(inst_mask)
    average_brightness = _composite.average_masked_brightness(room_bgr, ys_i, xs_i)
    instance.mm_per_unit = mm_per_unit

    # ---- Anchor, grid ----
    offset_u, offset_v = profile.anchor_offsets(
        instance, u_rot, v_rot, valid_ray, mm_per_unit, opts
    )

    u_mm = (u_rot + opts.tile_offset_x + offset_u) * mm_per_unit
    v_mm = (v_rot + opts.tile_offset_y + offset_v) * mm_per_unit

    tile_w_mm = max(opts.tile_width_mm, 1e-3)
    tile_h_mm = max(opts.tile_height_mm, 1e-3)

    frac_u, frac_v, _gu, _gv = _uv.to_grid_fractions(
        u_mm, v_mm, tile_w_mm, tile_h_mm, opts.flip_tile_x, opts.flip_tile_y
    )

    # Physical wall size measured on the UNROTATED axes, so it reports the
    # wall, not the grid: a 4.2 x 2.4 m wall stays 4.2 x 2.4 m at every
    # orientation. tiles-across/up then follow from the tile's rotated
    # footprint against that fixed surface -- the same rule the frontend's
    # Room Size panel uses.
    wall_w_mm, wall_h_mm = _uv.surface_extent_mm(
        geom["u_raw"] * mm_per_unit, geom["v_raw"] * mm_per_unit, visible
    )
    if scale_info.get("source") == "room-geometry" and np.any(visible):
        # On the room's true plane, a few mask pixels at the wall's vanishing
        # line (really another surface) lie arbitrarily far away; the wall's
        # size is its 1st-99th percentile extent, not its outliers.
        uu = geom["u_raw"][visible] * mm_per_unit
        vv = geom["v_raw"][visible] * mm_per_unit
        wall_w_mm = float(np.percentile(uu, 99) - np.percentile(uu, 1))
        wall_h_mm = float(np.percentile(vv, 99) - np.percentile(vv, 1))
    across_mm, deep_mm = _uv.effective_footprint_mm(tile_w_mm, tile_h_mm, opts.tile_rotation)

    # Plausibility guard.
    #
    # A wall fitted nearly edge-on to the camera -- a grazing sliver at the
    # frame edge, or a fragment whose depth was too flat to fit -- produces
    # rays that diverge across the plane and a reconstructed surface tens of
    # metres wide. Rendering it lays a few enormous tiles over the region and
    # looks like a bug in the tiling rather than a bad fit. Skipping the
    # instance leaves the other walls tiled and reports why in `failures`.
    if max(wall_w_mm, wall_h_mm) > MAX_PLAUSIBLE_WALL_MM:
        raise ValueError(
            f"{instance.label}: reconstructed {wall_w_mm:.0f} x {wall_h_mm:.0f} mm, "
            f"beyond the {MAX_PLAUSIBLE_WALL_MM:.0f} mm plausibility limit -- the "
            "plane is grazing or the depth was too flat to fit. Skipped."
        )

    if VP_DEBUG_INFO:
        _debug(f"=== WALL GRID [{instance.label}] ===")
        _debug(f"Plane n=({plane_a:.4f}, {plane_b:.4f}, {plane_c:.4f}) d={plane_d:.2f}")
        _debug(f"Wall width: {wall_w_mm:.0f} mm   height: {wall_h_mm:.0f} mm")
        _debug(f"Tile: {tile_w_mm:.0f} x {tile_h_mm:.0f} mm  "
               f"at {_uv.snap_orientation(opts.tile_rotation)}deg -> footprint "
               f"{across_mm:.0f} x {deep_mm:.0f} mm")
        _debug(f"Tiles across: {wall_w_mm / across_mm:.2f}   "
               f"Tiles up: {wall_h_mm / deep_mm:.2f}")
        _debug(f"Grid rotation: {math.degrees(rad):.2f}deg "
               f"(baseline {math.degrees(base_rad):.2f} + preset {opts.tile_rotation:.2f})")
        _debug("==============================")

    # ---- Sample, grout, composite ----
    tile_b, tile_g, tile_r, tile_a = _composite.sample_tile(
        tile_bgra, frac_u, frac_v, height_stretch=profile.tile_height_stretch
    )

    if opts.enable_grout and opts.grout_width_mm > 0:
        tile_b, tile_g, tile_r = _composite.apply_grout(
            tile_b, tile_g, tile_r, frac_u, frac_v, u_mm, v_mm,
            tile_w_mm, tile_h_mm, visible, opts.grout_width_mm, opts.grout_color
        )

    # Restrict coverage to THIS instance, and composite onto the running
    # result so each wall adds to the previous instead of erasing it.
    inst_factor = mask_factor * inst_mask.astype(mask_factor.dtype)

    # Centroid and bounding box so the UI can place a per-wall hotspot on the
    # render, the way a click selects a wall in the first place.
    centroid = (float(xs_i.mean()), float(ys_i.mean())) if len(xs_i) else (0.0, 0.0)
    bbox = ([int(xs_i.min()), int(ys_i.min()), int(xs_i.max()), int(ys_i.max())]
            if len(xs_i) else [0, 0, 0, 0])

    instance_info.append({
        "label": instance.label,
        "index": instance.index,
        "pixels": instance.pixel_count,
        "centroid": centroid,
        "bbox": bbox,
        "plane": [float(x) for x in plane],
        "mm_per_unit": float(mm_per_unit),
        "scale_source": scale_info.get("source"),
        "scale_caveat": scale_info.get("caveat"),
        "vp_source": evidence.source,
        "vp_confidence": float(evidence.confidence),
        "plane_source": evidence.info.get("plane_source"),
        "grid_rotation_deg": math.degrees(rad),
        "wall_width_mm": wall_w_mm,
        "wall_height_mm": wall_h_mm,
        "tile_footprint_mm": [across_mm, deep_mm],
        "tiles_across": wall_w_mm / across_mm if across_mm else 0.0,
        "tiles_up": wall_h_mm / deep_mm if deep_mm else 0.0,
        "detection": instance.info,
        # What this render decided, recorded for a 3D view to replay (read-only).
        "three": three_layer.record(
            kind="wall", f=geom["f"], cx=room_bgr.shape[1] / 2.0, cy=room_bgr.shape[0] / 2.0,
            image_size=(room_bgr.shape[1], room_bgr.shape[0]), plane=plane,
            e_u=geom["e_u"], e_v=geom["e_v"], rad=rad,
            offset_units=(opts.tile_offset_x + offset_u, opts.tile_offset_y + offset_v),
            mm_per_unit=mm_per_unit, opts=opts,
            tile_size_px=(tile_bgra.shape[1], tile_bgra.shape[0]),
            height_stretch=profile.tile_height_stretch, u_mm=u_mm, v_mm=v_mm,
            u_raw=geom["u_raw"], v_raw=geom["v_raw"], visible=visible,
            average_brightness=average_brightness,
        ),
    })

    return _composite.composite(
        room_bgr, tile_b, tile_g, tile_r, tile_a, inst_factor, valid_ray,
        average_brightness, opts.lighting_blend, opts.tile_opacity,
        base_bgr=base_bgr,
    )
