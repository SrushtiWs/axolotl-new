"""
What the floor plane is allowed to be.

Two paths, exactly as before: an analytic plane derived from the horizon when
align_plane_to_vanishing_point is on, and the constrained RANSAC fit
otherwise. Both are unchanged; they simply live here now instead of inline in
the renderer.
"""

import math

import numpy as np

from ...core.plane_fit import FloorConstraint, fit_plane_ransac


def plane_constraint(opts) -> FloorConstraint:
    return FloorConstraint(lock_horizontal=opts.lock_horizontal_floor)


def plane_from_horizon(vp_horizon_y, cy, f, Yc, Zc, depth_perspective_gain):
    """
    Floor plane derived analytically from the horizon height.

    Camera pitch, as the tangent of the angle between the optical axis and the
    floor plane. This single number decides how hard the DEPTH axis
    foreshortens; the lateral axis is set by f alone. Measured on a real
    render, moving the horizon from 341px to -400px (pitch 3.5 deg -> 29 deg)
    doubled a tile's projected height, 208px -> 415px, while its projected
    width moved 5%, 263px -> 278px.

    depth_perspective_gain is an EXPLICIT, opt-in override of that pitch and
    nothing else. At the default 1.0 this is exactly the unmodified pinhole
    model.

    It exists because (pitch, f) are jointly under-determined here: pitch =
    atan((cy - horizon_y) / f), so a hand-typed focal length and a hand-placed
    vanishing point together pin a camera angle that need not match the
    photograph. The principled alternative, which needs no override at all, is
    to let the two detected floor vanishing points calibrate f (see
    camera/focal_estimate.py) and take the horizon from the same fit.

    Applied to the pitch rather than to v afterwards on purpose: the plane
    stays a plane, so foreshortening still grows with distance and near/far
    perspective stays coherent. A multiplier on v_mm would instead be
    arithmetically identical to shrinking tile_height_mm.
    """
    inv_f = 1.0 / f
    vpy_cam = (vp_horizon_y - cy) * inv_f * depth_perspective_gain
    ny_val = -1.0 / math.sqrt(1.0 + vpy_cam * vpy_cam)
    nz_val = -vpy_cam * ny_val

    d_candidates = -(ny_val * Yc + nz_val * Zc)
    d_candidates = d_candidates[d_candidates > 0]
    best_d = float(np.median(d_candidates)) if len(d_candidates) else 200.0

    return 0.0, ny_val, nz_val, best_d


def fit_plane(Xc, Yc, Zc, xs, ys, opts, evidence, cy, f):
    """The renderer's plane-selection branch, unchanged."""
    if opts.align_plane_to_vanishing_point and evidence.vp_horizon_y is not None:
        return plane_from_horizon(
            evidence.vp_horizon_y, cy, f, Yc, Zc, opts.depth_perspective_gain
        )
    return fit_plane_ransac(
        Xc, Yc, Zc, xs, ys,
        iterations=opts.ransac_iterations,
        threshold=opts.ransac_threshold,
        constraint=plane_constraint(opts),
    )
