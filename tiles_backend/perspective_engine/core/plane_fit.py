"""
Plane fitting, shared by every surface.

The RANSAC search, the restart/polish consensus loop and the deterministic
tie-break are IDENTICAL to the original tile_engine._fit_plane_ransac. What
was hard-coded floor knowledge in that function -- "zero the roll", "make the
normal point up", "reject anything not nearly horizontal", "solve y as a
function of z" -- is now supplied by a PlaneConstraint. FloorConstraint
reproduces the original maths line for line; WallConstraint is its plumb
counterpart.

Camera convention throughout: +x right, +y DOWN, +z forward. So a floor
normal points up => b < 0, and a wall normal is nearly horizontal => |b| ~ 0.
"""

import math
from typing import Optional, Tuple

import numpy as np

# Plane RANSAC determinism. Fixed seed so repeated renders of the same image
# explore the same hypotheses; several restarts plus a few reassign-and-refit
# rounds so competing planes converge instead of the result depending on which
# mode the draw found first.
PLANE_FIT_SEED = 1337
PLANE_FIT_RESTARTS = 4
PLANE_FIT_POLISH_ROUNDS = 3

Plane = Tuple[float, float, float, float]


class PlaneConstraint:
    """
    What a surface believes about its own plane.

    Four hooks, each a place a floor and a wall genuinely disagree:

      lock_normal   remove a degree of freedom the depth map gets wrong
      orient        pick the sign convention for (a, b, c, d)
      is_plausible  reject hypotheses that cannot be this surface
      refine_lsq    least-squares refit that respects the lock
    """

    #: Returned when nothing survives the search.
    fallback: Plane = (0.0, -1.0, 0.0, 200.0)

    def lock_normal(self, a, b, c) -> Optional[Tuple[float, float, float]]:
        return a, b, c

    def orient(self, a, b, c, d) -> Plane:
        return a, b, c, d

    def is_plausible(self, a, b, c) -> bool:
        return True

    def refine_lsq(self, Xi, Yi, Zi, current: Plane) -> Plane:
        return current


class FloorConstraint(PlaneConstraint):
    """
    The original floor behaviour, unchanged.

    lock_horizontal removes sideways roll (a = 0), which is what stops the
    tile grid shearing; the |b| >= 0.45 gate rejects planes too steep to be a
    floor; the b > 0 flip makes the normal point up so |d| is the camera's
    height above it.
    """

    fallback: Plane = (0.0, -1.0, 0.0, 200.0)

    def __init__(self, lock_horizontal: bool = True, min_up: float = 0.45):
        self.lock_horizontal = lock_horizontal
        self.min_up = min_up

    def lock_normal(self, a, b, c):
        if not self.lock_horizontal:
            return a, b, c
        a = 0.0
        len_yz = math.hypot(b, c)
        if len_yz <= 1e-6:
            return None
        return a, b / len_yz, c / len_yz

    def orient(self, a, b, c, d):
        if b > 0:
            return -a, -b, -c, -d
        return a, b, c, d

    def is_plausible(self, a, b, c):
        return abs(b) >= self.min_up

    def refine_lsq(self, Xi, Yi, Zi, current):
        """
        Verbatim port of tile_engine._refine_plane_lsq. Kept as explicit
        normal-equation sums rather than rewritten as an SVD, because the
        floor path's output must not move by even a rounding step.
        """
        N = len(Xi)
        if N < 3:
            return current

        sXX = float(np.sum(Xi * Xi))
        sXZ = float(np.sum(Xi * Zi))
        sX = float(np.sum(Xi))
        sZZ = float(np.sum(Zi * Zi))
        sZ = float(np.sum(Zi))
        sXY = float(np.sum(Xi * Yi))
        sYZ = float(np.sum(Yi * Zi))
        sY = float(np.sum(Yi))

        best_plane = current

        if self.lock_horizontal:
            D = N * sZZ - sZ * sZ
            if abs(D) > 1e-6:
                q = (N * sYZ - sY * sZ) / D
                r = (sY * sZZ - sZ * sYZ) / D
                len_refined = math.sqrt(1 + q * q)
                refined_b = -1.0 / len_refined
                if abs(refined_b) >= self.min_up:
                    best_plane = (0.0, refined_b, q / len_refined, r / len_refined)
        else:
            detA = (
                sXX * (sZZ * N - sZ * sZ)
                - sXZ * (sXZ * N - sX * sZ)
                + sX * (sXZ * sZ - sX * sZZ)
            )
            if abs(detA) > 1e-6:
                detP = (
                    sXY * (sZZ * N - sZ * sZ)
                    - sXZ * (sYZ * N - sY * sZ)
                    + sX * (sYZ * sZ - sY * sZZ)
                )
                detQ = (
                    sXX * (sYZ * N - sY * sZ)
                    - sXY * (sXZ * N - sX * sZ)
                    + sX * (sXZ * sY - sX * sYZ)
                )
                detR = (
                    sXX * (sZZ * sY - sZ * sYZ)
                    - sXZ * (sXZ * sY - sX * sYZ)
                    + sXY * (sXZ * sZ - sX * sZZ)
                )
                p = detP / detA
                q = detQ / detA
                r = detR / detA
                len_refined = math.sqrt(p * p + 1 + q * q)
                refined_b = -1.0 / len_refined
                if abs(refined_b) >= self.min_up:
                    best_plane = (
                        p / len_refined,
                        refined_b,
                        q / len_refined,
                        r / len_refined,
                    )

        return best_plane


class WallConstraint(PlaneConstraint):
    """
    A wall is plumb and faces the camera.

    lock_vertical zeroes the normal's vertical component (b = 0), the exact
    counterpart of the floor's a = 0: it removes the axis noisy depth most
    often gets wrong, so a wall cannot lean a few degrees out of true and
    shear the tile grid. |b| <= max_tilt rejects hypotheses that are really
    the floor or ceiling leaking through an imperfect mask -- which is the
    single most common failure when a wall mask includes a strip of skirting
    or cornice.

    The refit is a total-least-squares (PCA) fit rather than the floor's
    normal equations. A wall is not a function of z: a back wall has z nearly
    constant, so "solve x = q*z + r" is exactly the degenerate case, and it is
    also the most common wall in a room photo. PCA has no preferred axis.
    """

    fallback: Plane = (0.0, 0.0, -1.0, 2000.0)

    def __init__(self, lock_vertical: bool = True, max_tilt: float = 0.45):
        self.lock_vertical = lock_vertical
        self.max_tilt = max_tilt

    def lock_normal(self, a, b, c):
        if not self.lock_vertical:
            return a, b, c
        b = 0.0
        len_xz = math.hypot(a, c)
        if len_xz <= 1e-6:
            return None
        return a / len_xz, b, c / len_xz

    def orient(self, a, b, c, d):
        # The camera looks down +z, so a wall it can see must have its normal
        # pointing back towards the origin: c < 0, hence d > 0 = the
        # perpendicular camera-to-wall distance.
        if c > 0:
            return -a, -b, -c, -d
        return a, b, c, d

    def is_plausible(self, a, b, c):
        # Nearly-horizontal normal (plumb wall) AND actually in view.
        return abs(b) <= self.max_tilt and math.hypot(a, c) > 1e-3

    def refine_lsq(self, Xi, Yi, Zi, current):
        N = len(Xi)
        if N < 3:
            return current

        if self.lock_vertical:
            # Plumb wall: the plane is a vertical line in the x-z plan view,
            # so fit that line by PCA and lift it back to 3D with b = 0.
            P = np.stack([Xi, Zi], axis=1).astype(np.float64)
            centre = P.mean(axis=0)
            centred = P - centre
            if not np.all(np.isfinite(centred)):
                return current
            try:
                _u, _s, vt = np.linalg.svd(centred, full_matrices=False)
            except np.linalg.LinAlgError:
                return current
            # Direction of least variance = the plan-view normal.
            nx, nz = vt[-1]
            norm = math.hypot(nx, nz)
            if norm < 1e-9:
                return current
            nx, nz = nx / norm, nz / norm
            d = -(nx * centre[0] + nz * centre[1])
            a, b, c, d = self.orient(float(nx), 0.0, float(nz), float(d))
        else:
            P = np.stack([Xi, Yi, Zi], axis=1).astype(np.float64)
            centre = P.mean(axis=0)
            centred = P - centre
            if not np.all(np.isfinite(centred)):
                return current
            try:
                _u, _s, vt = np.linalg.svd(centred, full_matrices=False)
            except np.linalg.LinAlgError:
                return current
            nx, ny, nz = vt[-1]
            norm = math.sqrt(nx * nx + ny * ny + nz * nz)
            if norm < 1e-9:
                return current
            nx, ny, nz = nx / norm, ny / norm, nz / norm
            d = -(nx * centre[0] + ny * centre[1] + nz * centre[2])
            a, b, c, d = self.orient(float(nx), float(ny), float(nz), float(d))

        if not self.is_plausible(a, b, c) or d <= 0:
            return current
        return (a, b, c, d)


def fit_plane_ransac(
    X,
    Y,
    Z,
    px,
    py,
    *,
    iterations: int,
    threshold: float,
    constraint: PlaneConstraint,
) -> Plane:
    """
    3-point RANSAC plane fit. X, Y, Z, px, py are 1D float arrays (already
    downsampled). Returns (a, b, c, d).

    Stability
    ---------
    Plain single-shot RANSAC was measured returning d = 456 or d = 944 for the
    SAME inputs across repeated runs -- two competing planes that both score
    well, so whichever the random draw happened to find first won. That is a 2x
    swing in the recovered geometry between identical requests.

    Two things fix it, and both are needed:

      1. A fixed seed, so a given input always explores the same hypotheses.
         Necessary but not sufficient -- it makes the coin flip repeatable
         rather than removing it.
      2. Consensus + polish. The hypothesis search is repeated over several
         seeds, and each candidate is then refined by re-assigning inliers and
         re-fitting a few times (the plane moves, so the inlier set that
         defined it is stale). Competing modes converge onto the same basin,
         and the winner is chosen by inlier count with a deterministic
         tie-break on d, so ties cannot flip between runs.
    """
    n = len(X)

    def _search(seed):
        """One seeded hypothesis search. Returns (plane, inlier_idx) or None."""
        best_plane = None
        max_inliers = -1
        best_inlier_idx = np.array([], dtype=np.int64)

        rng = np.random.default_rng(seed)
        pts3d = np.stack([X, Y, Z], axis=1)

        for _ in range(iterations):
            idx = rng.choice(n, size=3, replace=False)
            i1, i2, i3 = idx

            # Reject near-collinear image triplets: they define a plane only
            # weakly, and the normal swings wildly with depth noise.
            area = abs(
                (px[i2] - px[i1]) * (py[i3] - py[i1])
                - (py[i2] - py[i1]) * (px[i3] - px[i1])
            )
            if area < 500:
                continue

            p1, p2, p3 = pts3d[i1], pts3d[i2], pts3d[i3]
            u = p2 - p1
            v = p3 - p1
            normal = np.cross(u, v)
            length = np.linalg.norm(normal)
            if length < 1e-6:
                continue

            a, b, c = normal / length

            locked = constraint.lock_normal(a, b, c)
            if locked is None:
                continue
            a, b, c = locked

            d = -(a * p1[0] + b * p1[1] + c * p1[2])
            a, b, c, d = constraint.orient(a, b, c, d)

            if not constraint.is_plausible(a, b, c):
                continue

            dist = np.abs(a * X + b * Y + c * Z + d)
            inliers_mask = dist < threshold
            inliers_count = int(np.sum(inliers_mask))

            if inliers_count > max_inliers:
                max_inliers = inliers_count
                best_plane = (a, b, c, d)
                best_inlier_idx = np.where(inliers_mask)[0]

        if best_plane is None:
            return None
        return best_plane, best_inlier_idx

    def _score(plane):
        a, b, c, d = plane
        dist = np.abs(a * X + b * Y + c * Z + d)
        return int(np.sum(dist < threshold))

    candidates = []
    if n >= 3:
        for k in range(PLANE_FIT_RESTARTS):
            found = _search(PLANE_FIT_SEED + k * 7919)
            if found is None:
                continue
            plane, inlier_idx = found
            # Polish: the least-squares fit moves the plane, which makes the
            # inlier set that produced it stale. Re-assigning and re-fitting a
            # few times pulls competing modes into the same basin.
            for _ in range(PLANE_FIT_POLISH_ROUNDS):
                refined = refine_plane_lsq(X, Y, Z, inlier_idx, plane, constraint)
                if refined == plane:
                    break
                plane = refined
                a, b, c, d = plane
                dist = np.abs(a * X + b * Y + c * Z + d)
                inlier_idx = np.where(dist < threshold)[0]
                if len(inlier_idx) < 3:
                    break
            candidates.append(plane)

    if not candidates:
        return constraint.fallback

    # Deterministic winner: most inliers, tie-broken on d so equal-scoring
    # planes can never swap between runs.
    return max(candidates, key=lambda p: (_score(p), round(p[3], 6)))


def refine_plane_lsq(X, Y, Z, inlier_idx, current_plane: Plane,
                     constraint: PlaneConstraint) -> Plane:
    """
    Least-squares plane refit over `inlier_idx`, delegated to the constraint
    so each surface refits in the parametrisation that suits it. Returns
    `current_plane` unchanged when the system is degenerate.
    """
    if len(inlier_idx) < 3:
        return current_plane
    return constraint.refine_lsq(
        X[inlier_idx], Y[inlier_idx], Z[inlier_idx], current_plane
    )


def backproject(depth_val, mask_bool, cx, cy, f, depth_contrast, target_samples=3000):
    """
    Downsampled 3D point cloud for the masked region, in camera coordinates.

    Returns (X, Y, Z, px, py, sample_step). The 1000.0 / (d * contrast + 0.05)
    inversion is the original's: MiDaS output is inverse-depth-like and only
    determined up to scale, which is exactly why a metric anchor is needed
    later.
    """
    h_img, w_img = depth_val.shape[:2]
    mask_pixel_count = int(np.count_nonzero(mask_bool))
    step = max(1, int(math.floor(math.sqrt(max(mask_pixel_count, 1) / float(target_samples)))))

    y_full, x_full = np.mgrid[0:h_img:step, 0:w_img:step]
    y_full = y_full.ravel()
    x_full = x_full.ravel()
    valid = mask_bool[y_full, x_full]
    xs = x_full[valid].astype(np.float32)
    ys = y_full[valid].astype(np.float32)

    d_vals = depth_val[ys.astype(np.int64), xs.astype(np.int64)]
    Zc = 1000.0 / (d_vals * depth_contrast + 0.05)
    Xc = ((xs - cx) * Zc) / f
    Yc = ((ys - cy) * Zc) / f

    return Xc, Yc, Zc, xs, ys, step
