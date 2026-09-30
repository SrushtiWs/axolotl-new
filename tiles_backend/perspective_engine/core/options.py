"""
Render options shared by every surface.

This is the ORIGINAL tile_engine.RenderOptions, moved here unchanged so the
floor and wall pipelines can share one options object, plus new fields that
only the wall path reads. Every new field has a default, so any existing
caller -- app.py, the React frontend, a notebook -- constructs exactly the
object it constructed before and gets exactly the floor behaviour it got
before. tile_engine.RenderOptions is now an alias of this class.
"""

from dataclasses import dataclass
from typing import Optional

from ..camera.metric_scale import DEFAULT_CAMERA_HEIGHT_MM


@dataclass
class SurfaceRenderOptions:
    # ---------------- shared: tile grid ----------------
    tile_rotation: float = 0.0
    tile_offset_x: float = 0.0
    tile_offset_y: float = 0.0
    focal_length: float = 800.0
    invert_depth: bool = False
    lighting_blend: float = 0.5
    ransac_threshold: float = 5.0
    ransac_iterations: int = 150
    depth_contrast: float = 1.0
    tile_opacity: float = 1.0
    use_vanishing_point: bool = True
    vanishing_point_x: Optional[float] = None
    vanishing_point_y: Optional[float] = None
    # Detect the VP from the room image + surface mask. vanishing_point_x/y are
    # the fallback when detection fails, or the override when this is False.
    auto_detect_vp: bool = True
    # Minimum detection confidence before an auto VP is USED. Detecting a
    # vanishing point is not the same as recovering the right camera angle:
    # furniture edges, a patterned rug, a half-hidden floor, an imperfect mask
    # or lens distortion each produce a geometrically self-consistent VP that
    # is simply wrong. Below this score the detection is discarded and the
    # manual vanishing_point_x/y take over. 0.0 disables the gate.
    vp_confidence_threshold: float = 0.55
    # Feed the floor-to-wall junction from the mask silhouette into VP
    # detection at extra weight. It is the one line evidence a carpet pattern
    # cannot imitate.
    use_floor_boundary: bool = True
    # Estimate focal length per image (EXIF > two-VP calibration > FOV prior)
    # instead of trusting focal_length. A phone's focal length in pixels is not
    # a constant -- it moves with the lens that fired and the delivered
    # resolution.
    auto_focal_length: bool = True
    # Focal length in pixels read from the photo's EXIF by the caller, if any.
    exif_focal_px: Optional[float] = None
    # The single physical anchor that fixes millimetre scale for the FLOOR.
    # Absolute scale is not observable from one photograph, so a conversion
    # from plane units to millimetres must be set by something real; this is
    # it. There is no tile_scale, density slider or tuning constant any more --
    # tile size is tile_width_mm x tile_height_mm and nothing else scales it.
    # See perspective_engine/camera/metric_scale.py.
    #
    # A wall cannot use this anchor: |plane_d| is the camera-to-WALL distance,
    # which has no prior. See surface/wall/scale.py.
    camera_height_mm: float = DEFAULT_CAMERA_HEIGHT_MM
    # Explicit override of the camera PITCH used to build the floor plane,
    # and of nothing else. 1.0 = unmodified pinhole projection (the default;
    # this option is inert unless you ask for it). Above 1.0 tilts the
    # reconstructed floor further from edge-on, so the depth axis foreshortens
    # less; the lateral axis is governed by focal_length and barely moves.
    #
    # This does NOT change tile_width_mm/tile_height_mm or the
    # world-space tile aspect: an 800x1600 tile stays exactly 800x1600 mm and
    # exactly 1:2 at every setting. It changes the camera the floor is viewed
    # with, not the floor or the tiles.
    #
    # Only takes effect on the align_plane_to_vanishing_point path, which is
    # where the pitch is derived analytically. With the RANSAC plane fit there
    # is no explicit pitch term to override and this is ignored.
    depth_perspective_gain: float = 1.0
    lock_horizontal_floor: bool = True
    align_plane_to_vanishing_point: bool = True
    tile_width_mm: float = 800.0
    tile_height_mm: float = 1600.0
    enable_grout: bool = True
    grout_width_mm: float = 3.0
    grout_color: str = "#D0D0C8"
    flip_tile_x: bool = False
    flip_tile_y: bool = False
    auto_anchor_to_wall: bool = True

    # ---------------- surface selection ----------------
    # "floor" keeps every existing caller on the exact path it used before.
    surface: str = "floor"

    # ---------------- wall-only ----------------
    # Which detected wall to tile. -1 renders every wall instance found, each
    # with its own plane, basis, scale and anchor. A 0-based index renders one.
    wall_instance: int = -1
    # Force the wall plane's normal horizontal, i.e. the wall plumb. The exact
    # counterpart of lock_horizontal_floor: it removes the one degree of
    # freedom that noisy depth most often gets wrong, and stops a wall leaning
    # a few degrees out of true and shearing the tile grid.
    lock_vertical_wall: bool = True
    # The room's VP_Y (camera/vertical_vp.detect), when known. The wall grid's
    # roll correction is applied only when it passes vertical_vp.roll_gate;
    # None (no room geometry) keeps the roll at 0.
    vertical_vp: Optional[dict] = None
    # Physical room dimensions, when the user has supplied them. These are the
    # preferred metric anchor for a wall -- see surface/wall/scale.py -- and
    # are ignored entirely by the floor path.
    room_length_mm: Optional[float] = None
    room_width_mm: Optional[float] = None
    room_height_mm: Optional[float] = None
    # Assumed door height for the architectural-prior fallback. 2100mm is the
    # common European/Indian internal door leaf.
    door_height_mm: float = 2100.0
    # Minimum share of the image a wall region must cover to be treated as a
    # separate wall rather than a segmentation speck.
    wall_min_area_fraction: float = 0.01
    # Pick the wall under this image point instead of by index. This is what a
    # click on the render maps to, and it is the only selection method that
    # stays meaningful when instance ordering shifts between renders.
    # Overrides wall_instance when both are supplied.
    wall_point_x: Optional[float] = None
    wall_point_y: Optional[float] = None

    # ---------------- per-surface depth ----------------
    # Re-normalise the depth map inside the surface mask before fitting.
    #
    # None = use the surface's own default: OFF for the floor, because it would
    # change existing floor output and the floor already owns the frame's depth
    # range so it gains least; ON for the wall, which otherwise occupies a
    # narrow slice of a whole-frame normalisation and gets fitted against
    # quantisation noise. See core.composite.normalise_depth_within_mask.
    surface_depth_normalisation: Optional[bool] = None

    def depth_normalisation_enabled(self, default: bool) -> bool:
        """Explicit setting when given, else the surface's own default."""
        if self.surface_depth_normalisation is None:
            return default
        return bool(self.surface_depth_normalisation)
