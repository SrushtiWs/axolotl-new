"""
Floor pipeline.

The original render_tile_full, stage for stage and formula for formula, with
each stage now calling either core/ (shared with the wall) or surface/floor/
(the floor's own decisions). No maths changed: the plane fit, the metric
anchor, the basis, the rotation composition, the anchoring, the grout and the
compositing are the same code paths, reached through one more function call.

Stage map -- and where each stage lives now:

    normalise + decode mask       core/composite.py
    floor boundary                surface/floor/vp.py
    vanishing point               surface/floor/vp.py
    focal length                  camera/focal_estimate.py   (unchanged)
    back-projection               core/plane_fit.py
    plane fit                     surface/floor/constraints.py -> core/plane_fit.py
    metric scale                  surface/floor/scale.py -> camera/metric_scale.py
    basis + rays                  core/raycast.py
    rotation                      surface/floor/  (VP) + core/raycast.py
    grid anchor                   surface/floor/anchor.py
    UV -> tile grid               core/uv.py
    sample + grout + composite    core/composite.py
"""

import math

import numpy as np

from ..camera.focal_estimate import resolve_focal_length
from ..core import VP_DEBUG_INFO, debug as _debug
from ..core import composite as _composite
from ..core import raycast as _raycast
from ..core import uv as _uv
from ..core.plane_fit import backproject
from .. import three_layer
from ..surface.base import SurfaceInstance, SurfaceKind, SurfaceProfile
from ..surface.floor import anchor as floor_anchor
from ..surface.floor import constraints as floor_constraints
from ..surface.floor import scale as floor_scale
from ..surface.floor import vp as floor_vp

#: Texture-space v squash inherited from the original renderer. Hand-tuned,
#: not derived; kept because removing it would change every existing floor
#: render. See core.composite.sample_tile.
FLOOR_TILE_HEIGHT_STRETCH = 1.20


class FloorProfile(SurfaceProfile):
    """The floor's answers to the SurfaceProfile contract."""

    kind = SurfaceKind.FLOOR
    tile_height_stretch = FLOOR_TILE_HEIGHT_STRETCH

    def __init__(self, opts=None):
        self.opts = opts

    def instances(self, surface_bool, room_bgr, depth_val, opts):
        # A floor is one plane. Always exactly one instance -- this is the
        # structural difference from a wall, expressed in one line.
        return [SurfaceInstance(index=0, mask=surface_bool, label="floor")]

    def detect_geometry(self, room_bgr, instance, opts, cx, cy):
        boundary = floor_vp.extract_boundary(
            instance.mask, room_bgr.shape, enabled=opts.use_floor_boundary
        )
        return floor_vp.resolve(room_bgr, instance.mask, opts, cx, cy, boundary=boundary)

    def plane_constraint(self, opts):
        return floor_constraints.plane_constraint(opts)

    def basis_up_hint(self, plane):
        # A floor has no natural in-plane up direction; the original
        # construction is used unchanged.
        return None

    def resolve_scale(self, plane, instance, opts, context):
        return floor_scale.resolve(plane, opts)

    def rotation_baseline(self, evidence, opts, cx, cy, f, e_u, e_v) -> float:
        if not opts.use_vanishing_point:
            return 0.0
        return _raycast.vanishing_point_angle(
            evidence.vp_x, evidence.vp_y, cx, cy, f, e_u, e_v
        )

    def anchor_offsets(self, instance, u_rot, v_rot, valid_ray, mm_per_unit, opts):
        return floor_anchor.offsets(
            instance.mask, u_rot, v_rot, valid_ray, mm_per_unit, opts
        )


def render_floor(room_bgr, depth_bgr, mask_bgra, tile_bgra, opts):
    """
    room_bgr:  HxWx3 uint8 (BGR, cv2 convention)
    depth_bgr: HxWx3 uint8 (BGR) -- same size as room
    mask_bgra: HxWx4 uint8 (BGRA) -- alpha/brightness define floor coverage
    tile_bgra: hxwx4 uint8 (BGRA) tile texture, alpha used for blending

    Returns the rendered BGR image.
    """
    result, _info = render_floor_with_info(room_bgr, depth_bgr, mask_bgra, tile_bgra, opts)
    return result


def render_floor_with_info(room_bgr, depth_bgr, mask_bgra, tile_bgra, opts,
                           evidence=None, focal=None):
    """
    As render_floor, but also returns the diagnostics the API can surface.

    `evidence` and `focal` let a caller hand over floor geometry it detected
    beforehand -- the vanishing points (a GeometryEvidence) and the resolved
    focal length as (f, focal_info) -- so that geometry, not a fresh
    detection, is what the tiles are projected with. Both default to None,
    which runs the detection exactly as before.
    """
    profile = FloorProfile(opts)

    h_img, w_img = room_bgr.shape[:2]
    cx, cy = w_img / 2.0, h_img / 2.0
    # f is NOT taken from opts here. It is resolved further down, after
    # vanishing-point detection, because two orthogonal VPs calibrate it.

    depth_bgr, mask_bgra, tile_bgra, depth_val, mask_had_alpha = _composite.normalise_inputs(
        room_bgr, depth_bgr, mask_bgra, tile_bgra, opts.invert_depth
    )
    floor_bool, mask_factor = _composite.decode_mask(
        mask_bgra, mask_had_alpha, surface_label="floor"
    )

    # Per-surface depth normalisation is OFF by default here. It would change
    # every existing floor render, and the floor is the surface that already
    # owns the frame's depth range, so it is the one that gains least. Opt in
    # per request when a floor occupies only a small part of the frame.
    if opts.depth_normalisation_enabled(default=False):
        depth_val = _composite.normalise_depth_within_mask(depth_val, floor_bool)
        _debug("[depth] re-normalised within the floor mask")

    instance = profile.instances(floor_bool, room_bgr, depth_val, opts)[0]

    # ---- Floor boundary + vanishing point ----
    if evidence is None:
        evidence = profile.detect_geometry(room_bgr, instance, opts, cx, cy)

    # ---- Camera intrinsics ----
    # Must be settled BEFORE back-projection: f appears directly in Xc and Yc,
    # so an f that is wrong by 30% shears the entire point cloud, and the plane
    # fit then tilts to absorb the shear.
    if focal is not None:
        f, focal_info = focal
    else:
        f, focal_info = resolve_focal_length(
            image_width=w_img,
            exif_focal_px=opts.exif_focal_px,
            vp1=evidence.vp1_raw,
            vp2=evidence.vp2_raw,
            principal_point=(cx, cy),
            fallback_focal=opts.focal_length,
            auto=opts.auto_focal_length,
        )
    print(
        f"[focal] f={f:.1f}px ({focal_info['hfov_deg']:.1f} deg HFOV) "
        f"source={focal_info['source']}"
        + (f" agreement={focal_info['agreement_ratio']:.3f}"
           if "agreement_ratio" in focal_info else "")
    )

    # ---- Back-projected point cloud ----
    Xc, Yc, Zc, xs, ys, _step = backproject(
        depth_val, floor_bool, cx, cy, f, opts.depth_contrast
    )
    average_brightness = _composite.average_masked_brightness(room_bgr, ys, xs)

    # ---- Plane ----
    plane = floor_constraints.fit_plane(Xc, Yc, Zc, xs, ys, opts, evidence, cy, f)
    plane_a, plane_b, plane_c, plane_d = plane
    instance.plane = plane

    # ---- Metric calibration ----
    mm_per_spatial_unit, scale_info = profile.resolve_scale(plane, instance, opts, None)
    instance.mm_per_unit = mm_per_spatial_unit
    _debug(
        f"[scale] {mm_per_spatial_unit:.4f} mm/unit  anchor={scale_info['source']}"
        f"  camera_height={scale_info.get('camera_height_mm')}mm"
        f"  plane_d={abs(plane_d):.4f} units"
    )

    # ---- Basis, rotation, rays ----
    e_u, e_v = _raycast.build_plane_basis(
        plane_a, plane_b, plane_c, up_hint=profile.basis_up_hint(plane)
    )
    vp_base_rad = profile.rotation_baseline(evidence, opts, cx, cy, f, e_u, e_v)
    # Two independent contributions, COMPOSED rather than one overwriting the
    # other: where "parallel to the walls" actually is, plus the user's preset.
    # Adding them is what makes a 45 degree diagonal lay possible on top of a
    # wall-aligned floor instead of the two settings fighting.
    rad = vp_base_rad + math.radians(opts.tile_rotation)

    if VP_DEBUG_INFO:
        _pitch = math.degrees(math.atan2(
            cy - (evidence.vp_horizon_y if evidence.vp_horizon_y is not None else cy), f))
        _debug("=== VP -> GRID CHAIN ===")
        _debug(f"vp_source: {evidence.source}   vp used for ROTATION: "
               f"({evidence.vp_x}, {evidence.vp_y})   horizon used for PITCH: "
               f"{evidence.vp_horizon_y}")
        _debug(f"focal: {f:.1f}px   pitch: {_pitch:.2f}deg   "
               f"plane n=({plane_a:.4f}, {plane_b:.4f}, {plane_c:.4f}) d={plane_d:.2f}")
        _debug(f"e_u=({e_u[0]:.4f}, {e_u[1]:.4f}, {e_u[2]:.4f})  "
               f"e_v=({e_v[0]:.4f}, {e_v[1]:.4f}, {e_v[2]:.4f})")
        _debug(f"vp_base_rad: {math.degrees(vp_base_rad):.2f}deg   "
               f"tile_rotation: {opts.tile_rotation:.2f}deg   "
               f"final rad: {math.degrees(rad):.2f}deg")
        _debug("========================")

    X, Y, Z, valid_ray = _raycast.intersect_rays_with_plane(h_img, w_img, cx, cy, f, plane)
    u, v = _raycast.project_to_plane_uv(X, Y, Z, e_u, e_v)
    u_rot, v_rot = _uv.rotate(u, v, rad)

    # ---- Anchor the grid to the floor-wall junction ----
    auto_offset_u, auto_offset_v = profile.anchor_offsets(
        instance, u_rot, v_rot, valid_ray, mm_per_spatial_unit, opts
    )

    u_mm = (u_rot + opts.tile_offset_x + auto_offset_u) * mm_per_spatial_unit
    v_mm = (v_rot + opts.tile_offset_y + auto_offset_v) * mm_per_spatial_unit

    tile_w_mm = max(opts.tile_width_mm, 1e-3)
    tile_h_mm = max(opts.tile_height_mm, 1e-3)

    frac_u, frac_v, _gu, _gv = _uv.to_grid_fractions(
        u_mm, v_mm, tile_w_mm, tile_h_mm, opts.flip_tile_x, opts.flip_tile_y
    )

    visible = floor_bool & valid_ray
    surface_w_mm, surface_d_mm = _uv.surface_extent_mm(u_mm, v_mm, visible)

    if VP_DEBUG_INFO and np.any(visible):
        # A readout of what the render already decided, so the metric claim can
        # be checked rather than trusted. Nothing here feeds the render.
        _debug("=== METRIC TILE GRID ===")
        _debug(f"Floor width: {surface_w_mm:.0f} mm")
        _debug(f"Floor depth: {surface_d_mm:.0f} mm")
        _debug(f"Tile width: {tile_w_mm:.0f} mm")
        _debug(f"Tile height: {tile_h_mm:.0f} mm")
        _debug(f"Tiles across: {surface_w_mm / tile_w_mm:.2f}")
        _debug(f"Tiles deep: {surface_d_mm / tile_h_mm:.2f}")
        _debug(f"World tile size: {tile_w_mm:.0f} x {tile_h_mm:.0f} mm")
        _debug(f"Scale override: NONE  (mm/unit {mm_per_spatial_unit:.4f} "
               f"from {scale_info['source']} anchor)")
        _debug("========================")

    # ---- Sample, grout, composite ----
    tile_b, tile_g, tile_r, tile_a = _composite.sample_tile(
        tile_bgra, frac_u, frac_v, height_stretch=profile.tile_height_stretch
    )

    if opts.enable_grout and opts.grout_width_mm > 0:
        tile_b, tile_g, tile_r = _composite.apply_grout(
            tile_b, tile_g, tile_r, frac_u, frac_v, u_mm, v_mm,
            tile_w_mm, tile_h_mm, visible, opts.grout_width_mm, opts.grout_color
        )

    out = _composite.composite(
        room_bgr, tile_b, tile_g, tile_r, tile_a, mask_factor, valid_ray,
        average_brightness, opts.lighting_blend, opts.tile_opacity
    )

    info = {
        "surface": "floor",
        "instances": 1,
        "focal_px": float(f),
        "focal_source": focal_info.get("source"),
        "vp_source": evidence.source,
        "vp_confidence": float(evidence.confidence),
        "plane": [float(x) for x in plane],
        "mm_per_unit": float(mm_per_spatial_unit),
        "scale_source": scale_info.get("source"),
        "surface_width_mm": surface_w_mm,
        "surface_depth_mm": surface_d_mm,
        "tiles_across": surface_w_mm / tile_w_mm if tile_w_mm else 0.0,
        "tiles_deep": surface_d_mm / tile_h_mm if tile_h_mm else 0.0,
    }

    # What this render decided, recorded for a 3D view to replay. Read-only:
    # nothing below changes `out`.
    info["three"] = three_layer.record(
        kind="floor", f=f, cx=cx, cy=cy, image_size=(w_img, h_img), plane=plane,
        e_u=e_u, e_v=e_v, rad=rad,
        offset_units=(opts.tile_offset_x + auto_offset_u, opts.tile_offset_y + auto_offset_v),
        mm_per_unit=mm_per_spatial_unit, opts=opts,
        tile_size_px=(tile_bgra.shape[1], tile_bgra.shape[0]),
        height_stretch=profile.tile_height_stretch, u_mm=u_mm, v_mm=v_mm,
        u_raw=u, v_raw=v, visible=visible, average_brightness=average_brightness,
    )
    return out, info
