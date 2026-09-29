"""
Camera rays against a plane, and the 2D basis the tile grid lives in.

Both stages are pure geometry and identical for every surface. The one thing
that differs is which direction on the plane counts as "v", so the basis
builder takes an optional hint:

  * a floor has no natural up-direction in its own plane, so it uses the
    original arbitrary-but-stable construction;
  * a wall does -- gravity -- and a tile grid that ignores it produces
    skirting-to-ceiling courses that lean. build_plane_basis(up_hint=...)
    is how surface/wall asks for a plumb v axis.
"""

import math
from typing import Optional, Tuple

import numpy as np

Vec3 = Tuple[float, float, float]


def build_plane_basis(nx: float, ny: float, nz: float,
                      up_hint: Optional[Vec3] = None):
    """
    Orthonormal (e_u, e_v) spanning the plane with normal (nx, ny, nz).

    With up_hint=None this is the ORIGINAL floor construction, unchanged: take
    the world x axis, project out the normal, and fall back to the z axis when
    that degenerates (a plane whose normal is itself nearly x). Arbitrary but
    stable, which is all a floor needs -- its grid direction is set later by
    the vanishing point.

    With an up_hint, e_v is the hint projected into the plane, so the v axis
    follows gravity and e_u runs level along the surface.
    """
    if up_hint is None:
        eux = 1.0 - nx * nx
        euy = -nx * ny
        euz = -nx * nz
        eu_len = math.sqrt(eux * eux + euy * euy + euz * euz)
        if eu_len < 1e-4:
            eux = -nz * nx
            euy = -nz * ny
            euz = 1.0 - nz * nz
            eu_len = math.sqrt(eux * eux + euy * euy + euz * euz)

        eux, euy, euz = eux / eu_len, euy / eu_len, euz / eu_len
        evx = ny * euz - nz * euy
        evy = nz * eux - nx * euz
        evz = nx * euy - ny * eux
        return (eux, euy, euz), (evx, evy, evz)

    # ---- gravity-aligned basis (walls) ----
    ux, uy, uz = up_hint
    dot = ux * nx + uy * ny + uz * nz
    evx = ux - dot * nx
    evy = uy - dot * ny
    evz = uz - dot * nz
    ev_len = math.sqrt(evx * evx + evy * evy + evz * evz)
    if ev_len < 1e-6:
        # The plane is perpendicular to the hint (a floor handed an up hint).
        # Nothing sensible to align to, so fall back to the stable form.
        return build_plane_basis(nx, ny, nz, None)
    evx, evy, evz = evx / ev_len, evy / ev_len, evz / ev_len

    # e_u = e_v x n -- level, in-plane, right-handed with the normal.
    eux = evy * nz - evz * ny
    euy = evz * nx - evx * nz
    euz = evx * ny - evy * nx
    eu_len = math.sqrt(eux * eux + euy * euy + euz * euz)
    if eu_len < 1e-6:
        return build_plane_basis(nx, ny, nz, None)
    eux, euy, euz = eux / eu_len, euy / eu_len, euz / eu_len

    return (eux, euy, euz), (evx, evy, evz)


def intersect_rays_with_plane(h_img: int, w_img: int, cx: float, cy: float,
                              f: float, plane):
    """
    Intersect every pixel's camera ray with the plane.

    Returns (X, Y, Z, valid_ray), all HxW. valid_ray is False where the ray is
    parallel to the plane or hits it behind the camera -- for a wall that is a
    large part of the frame (everything past the corner), so it must be
    respected by the compositor rather than assumed away.
    """
    plane_a, plane_b, plane_c, plane_d = plane
    yy, xx = np.mgrid[0:h_img, 0:w_img].astype(np.float32)
    x_cam = (xx - cx) / f
    y_cam = (yy - cy) / f

    denom = plane_a * x_cam + plane_b * y_cam + plane_c
    denom_safe = np.where(np.abs(denom) < 1e-6, 1e-6, denom)
    Z = -plane_d / denom_safe
    valid_ray = (np.abs(denom) >= 1e-6) & (Z > 0)

    X = x_cam * Z
    Y = y_cam * Z
    return X, Y, Z, valid_ray


def project_to_plane_uv(X, Y, Z, e_u: Vec3, e_v: Vec3):
    """Camera-space points -> (u, v) coordinates in the plane's own basis."""
    eux, euy, euz = e_u
    evx, evy, evz = e_v
    u = X * eux + Y * euy + Z * euz
    v = X * evx + Y * evy + Z * evz
    return u, v


def vanishing_point_angle(vp_x, vp_y, cx, cy, f, e_u: Vec3, e_v: Vec3) -> float:
    """
    The in-plane direction, in radians, that a vanishing point corresponds to.

    A VP is the image of a direction at infinity, so its camera ray direction
    projected onto the plane basis IS that direction. This is what lets the
    tile grid line up with the room's own axes instead of the basis's
    arbitrary ones. Returns 0.0 when the VP carries no usable direction.
    """
    if vp_x is None or vp_y is None:
        return 0.0
    inv_f = 1.0 / f
    vpx_cam = (vp_x - cx) * inv_f
    vpy_cam = (vp_y - cy) * inv_f
    eux, euy, euz = e_u
    evx, evy, evz = e_v
    u_vp = vpx_cam * eux + vpy_cam * euy + 1.0 * euz
    v_vp = vpx_cam * evx + vpy_cam * evy + 1.0 * evz
    if abs(u_vp) > 1e-5 or abs(v_vp) > 1e-5:
        return math.atan2(u_vp, v_vp)
    return 0.0
