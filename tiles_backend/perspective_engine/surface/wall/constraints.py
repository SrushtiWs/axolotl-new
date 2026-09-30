"""
Keeping a wall plumb, and keeping noise from rotating it.

Two separate guards:

  1. The PLANE guard -- core.plane_fit.WallConstraint -- forces the fitted
     plane's normal horizontal, so the wall itself cannot lean.

  2. The GRID guard, below. Even with a perfect plane, the tile grid's
     rotation is a free parameter, and on a wall the correct value is almost
     always "square to gravity". The basis is already gravity-aligned (see
     core.raycast.build_plane_basis with an up hint), so the correct baseline
     rotation is 0. Letting a detected VP set it instead is how a single
     mis-detected diagonal -- a staircase, a sloped ceiling line, a picture
     hung askew -- tips an entire wall of tiles a few degrees off plumb.

     So a wall accepts only a small, confidence-gated roll correction, and
     hard-clamps it. This is the difference between "the grid follows the
     evidence" (right for a floor, whose orientation is genuinely unknown) and
     "the grid follows gravity unless the evidence is strong AND small" (right
     for a wall, whose orientation is known a priori).
"""

import math

import numpy as np

from ...core.plane_fit import WallConstraint

#: Largest grid rotation the image is ever allowed to introduce on a wall.
#: Beyond this the detection is not a rolled camera, it is a wrong line set.
MAX_ROLL_CORRECTION_DEG = 12.0

#: Below this confidence no correction is applied at all.
MIN_ROLL_CONFIDENCE = 0.45


def plane_constraint(opts) -> WallConstraint:
    return WallConstraint(lock_vertical=opts.lock_vertical_wall)


def rotation_baseline(evidence, opts) -> float:
    """
    Radians to rotate the tile grid before the user's tile_rotation is added.

    Returns 0.0 -- grid square to gravity -- unless plumb-line evidence is both
    confident and modest. Reported through evidence.info so a surprising
    result is explainable rather than mysterious.
    """
    from ...camera import vertical_vp

    roll = float(evidence.info.get("roll_rad", 0.0) or 0.0)
    conf = float(evidence.confidence or 0.0)

    # Camera roll is applied only on a confident VP_Y: one gate for the image.
    apply, why = vertical_vp.roll_gate(getattr(opts, "vertical_vp", None))
    evidence.info["roll_gate"] = why
    if not apply:
        evidence.info["roll_applied_deg"] = 0.0
        evidence.info["roll_rejected"] = why
        return 0.0

    if conf < MIN_ROLL_CONFIDENCE:
        evidence.info["roll_applied_deg"] = 0.0
        evidence.info["roll_rejected"] = f"confidence {conf:.2f} < {MIN_ROLL_CONFIDENCE}"
        return 0.0

    limit = math.radians(MAX_ROLL_CORRECTION_DEG)
    if abs(roll) > limit:
        evidence.info["roll_applied_deg"] = 0.0
        evidence.info["roll_rejected"] = (
            f"roll {math.degrees(roll):.1f}deg exceeds +/-{MAX_ROLL_CORRECTION_DEG}deg"
        )
        return 0.0

    evidence.info["roll_applied_deg"] = math.degrees(roll)
    return roll


#: Below this |c| a plane is so close to edge-on that its rays diverge across
#: the wall and the reconstruction runs to tens of metres. Deliberately low:
#: a genuine side wall at the edge of frame legitimately sits around |c| = 0.13,
#: and an earlier value of 0.15 was converting real side walls into
#: fronto-parallel ones -- a worse error than the one it was preventing.
MIN_FACING_COMPONENT = 0.05


def plane_from_along_wall_vp(vp_x, vp_y, cx, cy, f, points=None):
    """
    The wall plane, derived analytically from its along-wall vanishing point.

    This is the wall's counterpart to the floor's plane_from_horizon, and it
    exists for the same reason: the depth map cannot be trusted to give a plane.

    Why not fit the depth?
    ----------------------
    The depth map reaches the renderer as a Turbo-coloured PNG, and decoding it
    is monotonic but heavily warped -- measured at 13% mean absolute error
    against a known ramp. A flat surface is therefore NOT flat after the round
    trip, and RANSAC dutifully fits the warp: on a real room every wall came
    back with |c| between 0.05 and 0.13, i.e. reconstructed as almost exactly
    edge-on to the camera, which is impossible for a wall filling half the
    frame. The floor never hits this because it derives its plane from the
    horizon rather than from RANSAC.

    The geometry
    ------------
    A plumb wall has a horizontal normal, n = (a, 0, c). A vanishing point is
    the image of a direction at infinity, so the along-wall VP corresponds to a
    ray direction D = ((vx - cx)/f, (vy - cy)/f, 1). Horizontal lines on the
    wall lie IN the wall, so that direction is perpendicular to the normal:

        n . D = a*Dx + c*Dz = 0     (the Dy term drops out because b = 0)

    which fixes (a, c) up to sign as (Dz, -Dx), normalised. One VP, one exact
    wall orientation, no depth involved.

    `points` is an optional (X, Y, Z) sample used only to place the plane's
    DISTANCE. Distance is a uniform scaling of the reconstruction and is
    absorbed by the metric anchor later, so a rough value is fine; the shape
    comes entirely from the normal.
    """
    if vp_x is None or vp_y is None:
        return None

    dx = (float(vp_x) - cx) / f
    dz = 1.0
    norm = math.hypot(dz, dx)
    if norm < 1e-9:
        return None
    a, c = dz / norm, -dx / norm

    if abs(c) < MIN_FACING_COMPONENT:
        # The VP implies a wall nearly edge-on to the camera. For a wall we can
        # see across a real region that is a bad detection, not a bad room.
        return None

    d = 1000.0
    if points is not None:
        X, Y, Z = points
        if len(X):
            d = -(a * float(np.median(X)) + c * float(np.median(Z)))

    a, b, c, d = WallConstraint().orient(a, 0.0, c, d)
    if d <= 0:
        d = abs(d) if abs(d) > 1e-6 else 1000.0
    return (a, b, c, d)


def choose_plane(vps, points, cx, cy, f):
    """
    Pick which of the room's horizontal directions this wall actually contains.

    A rectangular room offers exactly two horizontal directions, so a wall's
    normal has only two candidates -- and telling them apart does not need
    accurate depth, only depth good enough to distinguish two hypotheses 90
    degrees apart. The decoded depth is warped by ~13% but it is MONOTONIC, so
    it discriminates that easily even though it cannot be fitted directly.

    Test: for a candidate normal (a, 0, c), the quantity a*X + c*Z is constant
    across a wall that really has that normal. So score each candidate by how
    little that quantity varies, normalised by the region's depth so the
    comparison is scale-free. Lowest spread wins.

    Returns (plane, info) or (None, info).
    """
    info = {"candidates": []}
    if not vps or points is None:
        info["stage"] = "no-candidates"
        return None, info

    X, Y, Z = points
    if len(X) < 16:
        info["stage"] = "too-few-points"
        return None, info

    depth_scale = max(float(np.median(np.abs(Z))), 1e-6)
    best = None

    for vx, vy in vps:
        plane = plane_from_along_wall_vp(vx, vy, cx, cy, f, points=points)
        if plane is None:
            info["candidates"].append({"vp": (vx, vy), "rejected": "degenerate-or-grazing"})
            continue
        a, _b, c, _d = plane
        proj = a * X + c * Z
        spread = float(np.std(proj)) / depth_scale
        info["candidates"].append({
            "vp": (vx, vy), "normal": (a, c), "relative_spread": spread,
        })
        if best is None or spread < best[0]:
            best = (spread, plane, (vx, vy))

    if best is None:
        info["stage"] = "all-candidates-rejected"
        return None, info

    info["stage"] = "ok"
    info["chosen_vp"] = best[2]
    info["chosen_spread"] = best[0]
    return best[1], info


def sanitise_plane(plane, points=None):
    """
    Replace a grazing plane with a fronto-parallel one.

    Last line of defence for instances with no usable VP: a wall reconstructed
    almost edge-on produces rays that diverge across it and surfaces tens of
    metres wide. Facing the camera squarely is both the commonest real case
    (it is the back wall) and the safest wrong answer, because it keeps the
    tile grid finite and the error is a yaw rather than a divergence.
    """
    if plane is None:
        a, b, c = 0.0, 0.0, -1.0
        d = 1000.0
        if points is not None and len(points[0]):
            d = float(np.median(points[2]))
        return (a, b, c, max(d, 1.0))

    a, b, c, d = plane
    if abs(c) >= MIN_FACING_COMPONENT and d > 0:
        return plane

    d_new = 1000.0
    if points is not None and len(points[0]):
        d_new = float(np.median(points[2]))
    return (0.0, 0.0, -1.0, max(d_new, 1.0))


def gravity_up_hint():
    """
    World "up" in camera coordinates.

    The camera convention here is +y DOWN, so up is (0, -1, 0). Handed to
    core.raycast.build_plane_basis, this makes e_v the plumb direction in the
    wall plane and e_u level along it -- which is what makes a wall course
    horizontal without any image evidence at all.
    """
    return (0.0, -1.0, 0.0)
