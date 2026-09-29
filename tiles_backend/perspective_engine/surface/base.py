"""
The contract between a surface and the shared rendering core.

A SurfaceProfile answers the handful of questions where a floor and a wall
genuinely disagree. Everything else -- back-projection, RANSAC, ray casting,
UV, grout, lighting, compositing -- is core's, and is identical for both.

    SurfaceProfile
      .segment_labels()     which ADE20K classes make this surface
      .detect_geometry()    VP / direction evidence from the room image
      .plane_constraint()   what the plane is allowed to be
      .basis_up_hint()      which in-plane direction is "v"
      .resolve_scale()      millimetres per plane unit, from a real anchor
      .rotation_baseline()  where "aligned with the room" is, in radians
      .anchor_offsets()     where a tile edge must land
      .instances()          how many independent planes this surface has

Adding a third surface (ceiling, splashback) means one more package under
surface/ implementing these. It does not mean touching core/, and it does not
mean touching floor/ or wall/.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

import numpy as np


class SurfaceKind(str, Enum):
    FLOOR = "floor"
    WALL = "wall"

    @classmethod
    def parse(cls, value) -> "SurfaceKind":
        """Accepts a SurfaceKind, or any casing of its name. Defaults to floor."""
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError:
            return cls.FLOOR


@dataclass
class SurfaceInstance:
    """
    One independently-tiled plane.

    A floor always yields exactly one of these. A wall yields one PER VISIBLE
    WALL, which is the whole reason this type exists: a photo of a room corner
    contains two walls at 90 degrees, and a single plane fitted across both
    passes through neither.
    """
    index: int
    mask: np.ndarray                      # HxW bool, this instance only
    label: str = "surface"
    #: Plane (a, b, c, d) in camera coordinates, filled in by the pipeline.
    plane: Optional[Tuple[float, float, float, float]] = None
    #: Per-instance vanishing point, in pixels.
    vp: Optional[Tuple[float, float]] = None
    #: Millimetres per plane unit for this instance.
    mm_per_unit: Optional[float] = None
    #: Free-form diagnostics, surfaced to the API response.
    info: dict = field(default_factory=dict)

    @property
    def pixel_count(self) -> int:
        return int(np.count_nonzero(self.mask))


@dataclass
class GeometryEvidence:
    """
    What the surface's own detector recovered from the image.

    vp_rotation is the point used to ORIENT the grid; vp_horizon_y is the
    height used to derive PITCH. They are separate because mixing them
    describes a direction that is not in the scene -- see the long note in the
    original renderer. A wall fills in rotation only; it has no horizon of its
    own to speak of.
    """
    vp_x: Optional[float] = None
    vp_y: Optional[float] = None
    vp_horizon_y: Optional[float] = None
    vp1_raw: Optional[Tuple[float, float]] = None
    vp2_raw: Optional[Tuple[float, float]] = None
    confidence: float = 0.0
    source: str = "manual"
    boundary: Optional[dict] = None
    info: dict = field(default_factory=dict)


class SurfaceProfile:
    """Base class; see the module docstring for the contract."""

    kind: SurfaceKind = SurfaceKind.FLOOR
    #: Texture-space v squash applied when sampling the tile. See
    #: core.composite.sample_tile -- the floor inherits a hand-tuned 1.20 from
    #: the original renderer, a wall has no such fudge.
    tile_height_stretch: float = 1.0

    def instances(self, surface_bool, room_bgr, depth_val, opts):
        raise NotImplementedError

    def detect_geometry(self, room_bgr, instance, opts, cx, cy) -> GeometryEvidence:
        raise NotImplementedError

    def plane_constraint(self, opts):
        raise NotImplementedError

    def basis_up_hint(self, plane):
        return None

    def resolve_scale(self, plane, instance, opts, context):
        raise NotImplementedError

    def rotation_baseline(self, evidence, opts, cx, cy, f, e_u, e_v) -> float:
        return 0.0

    def anchor_offsets(self, instance, u_rot, v_rot, valid_ray, mm_per_unit, opts):
        return 0.0, 0.0
