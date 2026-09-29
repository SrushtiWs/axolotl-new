# perspective_engine/camera/vp_from_mask.py
"""
Automatic floor vanishing-point detection from a room image + floor mask.

Used by the live /render-tile-full path (perspective_engine/tile/tile_engine.py)
to replace hand-typed vanishing point coordinates.

Pipeline
--------
  1. LSD line segments over the whole room image (Hough fallback).
  2. Keep ONLY lines whose midpoint sits on the segmented floor. This is the
     whole point of the exercise: run detection on the unmasked room and the
     pillar edges, door frame, ceiling line and wall corners all vote in the
     same RANSAC as the real floor-grout lines, which is what drags the tile
     grid off the room's axes.
  3. Mix in the floor-to-wall junction segments from floor_boundary.py at a
     higher weight than image-texture lines (see BOUNDARY_LINE_WEIGHT).
  4. Roll-invariant k=2 clustering on line angle (doubled-angle trick), so a
     tilted camera keeps both floor directions.
  5. RANSAC + SVD vanishing point per cluster, repeated over several seeds.
  6. Score the result's confidence and hand it to the caller (see below).

Two quantities come out, and they are NOT interchangeable
--------------------------------------------------------
  (vp_x, vp_y)  the depth-direction vanishing point's ACTUAL image position.
                Tile grid rotation is the direction from the camera towards
                this point, so it needs the real point.
  horizon_y     horizon height at the principal point, from the line through
                both floor VPs. Floor PITCH needs this; it is better
                conditioned than either VP's own y because both VPs lie on
                the horizon and fitting through the pair averages out noise.

A previous version returned horizon_y in the vp_y slot and the rotation maths
consumed it as though it were the VP's own y. Pairing one VP's x with a y
taken from elsewhere describes a direction that does not exist in the scene,
which tilted the grid off the room's axes -- the diagonal tile pattern. They
are reported separately here and must stay separate.

Confidence, and why a bare pass/fail is not enough
--------------------------------------------------
Detecting a vanishing point does not mean recovering the right camera angle.
On real room photos the failure modes are not exotic:

  * furniture edges lying on the floor pass the mask test while pointing
    nowhere useful;
  * a patterned rug supplies dozens of strong parallel lines that are not
    aligned with the room and can outvote the actual grout;
  * the floor is half-hidden, so the surviving lines span a small image region
    and the intersection is poorly constrained;
  * the mask itself is imperfect at the wall junction;
  * a wide-angle phone lens leaves barrel distortion that bends what should
    be straight lines, and some phones apply their own perspective correction
    before you ever see the file.

Each of those produces a geometrically self-consistent vanishing point. The
fit residual looks fine; the answer is wrong. So detection reports a score in
[0, 1] combining independent evidence -- RANSAC support, how much floor line
there was to work with, agreement across restarts, whether both directions
were populated, whether the wall junction agrees, and how much of the floor
the photo actually shows. The caller compares it against a threshold and falls
back to manual rather than rendering from a confident-looking guess. See
score_vp_confidence for what each term is doing.

Stability
---------
RANSAC is randomised. A fixed seed alone only makes the coin flip repeatable:
on real room photos the horizon estimate was measured swinging by up to 74% of
the image height between seeds, which as a floor pitch is not a small error
but a different room. So the RANSAC is re-run over several seeds against the
same (deterministic) clusters and the median is taken; if the restarts
disagree by more than `max_horizon_spread`, the estimate is reported as FAILED
so the caller falls back rather than rendering from a coin flip.
"""

import math

import numpy as np
import cv2

from .floor_boundary import extract_floor_boundary

# How many texture lines one floor-to-wall junction segment is worth in the
# vote. The junction is real room geometry that a rug pattern cannot imitate,
# so it should outweigh texture -- but not silently decide the answer alone,
# which is why this is 3 and not 50. Deliberately conservative: a mask whose
# wall boundary is a few degrees off would, at a high weight, drag a perfectly
# good texture-derived VP off the room's real axes.
BOUNDARY_LINE_WEIGHT = 3.0

# Points sampled along each candidate line, and the share of them that must
# land on the floor for the line to be kept. See filter_lines_by_floor_mask.
LINE_SAMPLE_POINTS = 9
LINE_INSIDE_RATIO = 0.8

# Morphology kernel for the VP-only mask clean-up. Small on purpose: the
# floor-to-wall junction must survive intact. See clean_floor_mask_for_vp.
MASK_CLEAN_KERNEL_PX = 5

# Iterative reweighted refinement rounds after RANSAC picks a winner. Each
# round re-fits over the current inlier set, so the final VP is determined by
# every strongly supporting line rather than by the two that seeded it.
VP_REFINE_ROUNDS = 3

# Maximum angular disagreement (degrees), across RANSAC restarts, in the
# direction from the principal point to the depth VP. This is the quantity the
# tile grid's rotation is computed from, so it is gated directly. A VP whose
# absolute position swings by thousands of pixels while staying on the same
# ray is harmless to rotation; one that shifts a few degrees is not, and the
# horizon-spread test cannot see the difference.
MAX_VP_DIRECTION_SPREAD_DEG = 6.0

# Relative weights for choosing WHICH of the two floor VPs is the
# depth direction. See _select_depth_vp -- picking wrong rotates the tile grid
# by 90 degrees, so this is scored from several independent signals rather
# than from line orientation alone.
DEPTH_SELECT_WEIGHTS = {
    "orientation": 1.0,
    "support": 0.5,
    "boundary": 0.7,
}

# Angular tolerance (degrees) for deciding that a boundary segment points at
# the detected VP. Angular rather than a pixel distance ON PURPOSE: the
# perpendicular distance from a VP to a line grows with the VP's distance from
# the image, so a fixed pixel tolerance is a tighter and tighter angular test
# the further out the VP sits. Measured across 65 rooms, a 6px version of this
# scored 0.02 on nearly every image with a distant VP and 0.7+ on the few with
# a close one -- it was reporting VP distance, not agreement.
BOUNDARY_AGREEMENT_DEG = 4.0

# Typical share of a floor boundary that is real geometry rather than image
# border. A floor running out of the bottom of the frame is normal, not a
# defect, so the visibility term is normalised against this and only penalises
# photos well below it.
TYPICAL_VISIBLE_FRACTION = 0.45

# Floor lines needed before line count stops penalising the score. Below this
# the VP is being fit to too little evidence to trust regardless of residual.
CONFIDENCE_FULL_LINE_COUNT = 40.0

# Total floor-line length, as a multiple of the image's long edge, at which
# the length half of the line-evidence term saturates. Scored alongside the
# count so that many short specks cannot masquerade as strong evidence.
CONFIDENCE_FULL_LINE_LENGTH_FRAC = 4.0

# Depth-vs-cross selection margin at which the "direction" confidence term
# reaches full marks. Below it the 90-degree choice was close, and confidence
# drops to reflect that.
DEPTH_MARGIN_FULL_SCORE = 0.25

# Steepest slope the line through the two floor VPs may have before the pair
# is treated as geometrically inconsistent (tan 30 degrees).
#
# Both floor vanishing points lie on the true horizon by construction, and for
# a camera with anything less than extreme roll the horizon is near-horizontal.
# A steep line through them therefore does not mean a tilted horizon, it means
# one of the two VPs is wrong. Extrapolating along it to the principal point
# then amplifies the error enormously: a measured pair at (430, 513) and
# (332, 987) gave slope -4.84 and a horizon of -1123px on a 1024px-tall image,
# which as a floor pitch is nonsense. Detected and handled rather than
# extrapolated -- see the restart loop.
MAX_HORIZON_SLOPE = 0.577

# Weights of the confidence terms. Applied as a weighted GEOMETRIC mean, so a
# single term near zero drags the whole score down instead of being averaged
# away by four healthy ones -- which is the entire point. A carpet-driven VP
# scores high on support and stability and near zero on boundary agreement,
# and must not come out at 0.8.
CONFIDENCE_WEIGHTS = {
    "support": 1.0,      # RANSAC inlier ratio on the depth cluster
    "lines": 0.8,        # how much floor line evidence existed at all
    "stability": 1.2,    # agreement of the horizon across RANSAC restarts
    "balance": 0.6,      # were both floor directions actually populated
    "boundary": 1.0,     # does the floor-wall junction agree with the VP
    "direction": 0.7,    # how decisively depth beat cross (90-degree risk)
    "visibility": 0.8,   # how much of the floor the frame actually shows
}


def detect_lines_lsd(image_gray: np.ndarray) -> np.ndarray:
    """
    Line segment detection, returned as an (N, 4) array of [x1, y1, x2, y2].

    Prefers LSD and falls back to probabilistic Hough on builds where LSD is
    unavailable (it was removed from OpenCV over a patent claim in 4.1.0 and
    only restored in 4.5.3), so this never hard-fails on a version difference.
    """
    try:
        lsd = cv2.createLineSegmentDetector(0)
        lines, _, _, _ = lsd.detect(image_gray)
        if lines is not None and len(lines):
            return lines.reshape(-1, 4).astype(np.float32)
    except cv2.error:
        pass

    edges = cv2.Canny(image_gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180.0, threshold=60,
        minLineLength=30, maxLineGap=10
    )
    if lines is None:
        return np.empty((0, 4), dtype=np.float32)
    return lines.reshape(-1, 4).astype(np.float32)


def clean_floor_mask_for_vp(floor_mask: np.ndarray,
                            kernel_px: int = MASK_CLEAN_KERNEL_PX) -> np.ndarray:
    """
    A cleaned COPY of the floor mask, for line filtering only.

    Opening first, then closing: the opening deletes the isolated specks the
    segmenter leaves on reflective floors (each one is a place a stray wall or
    furniture line can be certified as "on the floor"), and the closing seals
    the pinholes chair legs and table feet punch through an otherwise solid
    region, which would otherwise chop one long grout line into fragments too
    short to survive the length filter.

    The kernel is deliberately small. A large one would round off the
    floor-to-wall junction, and that junction is the single most valuable
    piece of evidence this module has.

    The input is never mutated. The renderer's floor_bool must keep selecting
    exactly the pixels it selects today -- this cleaning exists only so line
    filtering asks a better question, and must not move a single rendered
    pixel.
    """
    if floor_mask is None:
        return None

    mask = floor_mask
    if mask.dtype == bool:
        mask = mask.astype(np.uint8) * 255
    elif mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8) * 255
    else:
        mask = mask.copy()

    if mask.ndim != 2 or kernel_px < 3:
        return mask

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_px, kernel_px))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k)


def filter_lines_by_floor_mask(lines: np.ndarray, floor_mask: np.ndarray,
                               min_length: float = 20.0,
                               n_samples: int = LINE_SAMPLE_POINTS,
                               inside_ratio: float = LINE_INSIDE_RATIO) -> np.ndarray:
    """
    Keeps only the lines that lie ALONG the floor, not merely across it.

    The previous version tested a single point: the segment's midpoint. That
    admits any line whose middle happens to overlap the floor region while
    both of its halves are somewhere else entirely -- the vertical edge of a
    door frame, the leading edge of a sofa, a skirting board, the corner where
    two walls meet above a floor pixel. Those lines are not floor geometry and
    they do not converge to the floor's vanishing points, but they vote in the
    same RANSAC, and they are exactly what pulls the tile grid off the room's
    axes.

    So the segment is sampled at `n_samples` points spread evenly end to end,
    and kept only when at least `inside_ratio` of them land on the floor. A
    door frame crossing the floor at its midpoint scores about one sample in
    nine; a real grout line scores nine in nine.

    Endpoints are included in the sample. That is intentional and slightly
    strict: a genuine floor line running under a rug edge or out of the mask
    at one end loses a sample or two, and the 0.8 default leaves room for
    exactly that while still rejecting anything merely passing through.
    """
    if lines is None or len(lines) == 0:
        return np.empty((0, 4), dtype=np.float32)

    mask = floor_mask
    if mask.dtype == bool:
        mask = mask.astype(np.uint8) * 255
    h, w = mask.shape[:2]

    x1, y1, x2, y2 = lines[:, 0], lines[:, 1], lines[:, 2], lines[:, 3]
    long_enough = np.hypot(x2 - x1, y2 - y1) >= min_length
    if not np.any(long_enough):
        return np.empty((0, 4), dtype=np.float32)

    n_samples = max(3, int(n_samples))
    t = np.linspace(0.0, 1.0, n_samples)[None, :]          # (1, S)

    # (N, S) sample grid: every segment sampled at the same parametric offsets.
    sx = x1[:, None] + (x2 - x1)[:, None] * t
    sy = y1[:, None] + (y2 - y1)[:, None] * t

    ix = np.rint(sx).astype(np.int64)
    iy = np.rint(sy).astype(np.int64)

    in_bounds = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    # Clip before indexing so out-of-frame samples cannot raise; they are
    # excluded by `in_bounds` regardless of what they happen to read.
    np.clip(ix, 0, w - 1, out=ix)
    np.clip(iy, 0, h - 1, out=iy)

    on_floor = (mask[iy, ix] > 0) & in_bounds
    ratio = on_floor.mean(axis=1)

    keep = long_enough & (ratio >= inside_ratio)
    if not np.any(keep):
        return np.empty((0, 4), dtype=np.float32)

    return lines[keep].astype(np.float32)


def _lines_to_homo(lines):
    """
    Line segments -> homogeneous line vectors L = P1 x P2, each scaled so that
    a^2 + b^2 = 1.

    That normalisation lets the RANSAC scoring below use a plain dot product:
    for a unit-normalised line, |l . [x, y, 1]| IS the perpendicular distance
    from (x, y) to the line, with no per-candidate division.
    """
    ones = np.ones(len(lines), dtype=np.float64)
    p1 = np.column_stack([lines[:, 0], lines[:, 1], ones])
    p2 = np.column_stack([lines[:, 2], lines[:, 3], ones])
    l = np.cross(p1, p2)
    norm = np.linalg.norm(l[:, :2], axis=1, keepdims=True)
    return np.where(norm > 1e-5, l / np.maximum(norm, 1e-12), l)


def _estimate_vp_ransac(lines, iterations=300, threshold=3.0, rng=None,
                        weights=None):
    """
    RANSAC vanishing point from one direction cluster.

    Sample 2 lines, intersect, count inliers by perpendicular distance, refine
    the winner by SVD over its inliers -- but scored as one (iterations x N)
    matrix rather than a Python loop. The looped form cost 1-3 seconds per
    render on real room photos; this runs on every live-render request.

    `weights` lets a floor-to-wall junction segment count for more than a
    texture line. Weighting the VOTE is the correct place for that preference:
    duplicating the boundary rows instead would work, but it would also inflate
    the inlier ratio that the confidence score reads, making thin evidence look
    strong. Support is therefore returned as a weighted ratio.

    Returns (vp, support) with support in [0, 1], or None.
    """
    n = len(lines)
    if n < 2:
        return None

    rng = rng if rng is not None else np.random.default_rng(1337)
    homo_all = _lines_to_homo(lines)

    w = np.ones(n, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    total_w = float(w.sum())
    if total_w <= 0:
        return None

    # `iterations` random line PAIRS at once. The offset trick guarantees
    # i2 != i1 without rejection sampling.
    i1 = rng.integers(0, n, size=iterations)
    i2 = (i1 + rng.integers(1, n, size=iterations)) % n if n > 1 else i1

    cand = np.cross(homo_all[i1], homo_all[i2])
    usable = np.abs(cand[:, 2]) >= 1e-5
    if not np.any(usable):
        return None
    cand = cand[usable] / cand[usable, 2:3]

    dist = np.abs(cand @ homo_all.T)          # (candidates, N)
    inlier_mask = dist < threshold
    scores = inlier_mask @ w                  # weighted vote
    best = int(np.argmax(scores))

    best_vp = cand[best]
    best_inliers = inlier_mask[best]
    support = float(scores[best] / total_w)

    # Iteratively re-fit over the supporting lines. A single SVD pass is still
    # anchored to whichever inliers the seed pair happened to select;
    # re-deciding membership between fits lets the estimate settle on the
    # consensus of every line that genuinely supports it, rather than on one
    # lucky pair.
    #
    # Membership is re-decided WITHIN the RANSAC winner's own inlier pool, and
    # never reopened to the whole set. Reopening it is a degeneracy: a VP
    # pushed toward infinity makes every near-parallel line an inlier, so
    # growing the set monotonically rewards drifting outward, and the fit
    # walks off to infinity while its inlier count keeps improving. Measured on
    # real rooms, that put the horizon at -1131px on a 1024px-tall image. The
    # pool can shrink or stabilise here, never grow, so the iteration
    # converges instead of escaping.
    pool = best_inliers.copy()
    if pool.sum() >= 2:
        active = pool.copy()
        for _ in range(VP_REFINE_ROUNDS):
            # Weighted least squares: scaling a row by sqrt(w) makes the SVD
            # minimise sum(w * residual^2).
            rows = homo_all[active] * np.sqrt(w[active])[:, None]
            _, _, vt = np.linalg.svd(rows)
            refined = vt[-1]
            if abs(refined[2]) <= 1e-5 or not np.all(np.isfinite(refined)):
                break
            refined = refined / refined[2]

            next_active = pool & (np.abs(homo_all @ refined) < threshold)
            if next_active.sum() < 2:
                break
            best_vp = refined
            converged = np.array_equal(next_active, active)
            active = next_active
            if converged:
                break
        best_inliers = active

    support = float((best_inliers @ w) / total_w)
    return best_vp, support


def _cluster_lines_by_orientation(lines, iterations=25, weights=None):
    """
    Roll-invariant k=2 clustering on line angle (doubled-angle trick), so a
    tilted camera doesn't lose either floor direction. Deterministic.

    Returns a list of (cluster_lines, cluster_weights).
    """
    if len(lines) < 4:
        return []

    w = np.ones(len(lines), dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)

    angles = np.arctan2(lines[:, 3] - lines[:, 1], lines[:, 2] - lines[:, 0])
    theta2 = angles * 2.0
    pts = np.column_stack([np.cos(theta2), np.sin(theta2)])

    c0 = pts[0]
    c1 = pts[np.argmin(pts @ c0)]
    assign = np.zeros(len(pts), dtype=bool)

    for _ in range(iterations):
        d0 = np.linalg.norm(pts - c0, axis=1)
        d1 = np.linalg.norm(pts - c1, axis=1)
        new_assign = d1 < d0
        if new_assign.sum() == 0 or (~new_assign).sum() == 0:
            break
        # Weighted centroids, so a boundary segment pulls the cluster centre
        # as hard as it votes in the RANSAC.
        w0, w1 = w[~new_assign], w[new_assign]
        new_c0 = (pts[~new_assign] * w0[:, None]).sum(axis=0) / max(w0.sum(), 1e-9)
        new_c1 = (pts[new_assign] * w1[:, None]).sum(axis=0) / max(w1.sum(), 1e-9)
        converged = np.array_equal(new_assign, assign)
        c0, c1, assign = new_c0, new_c1, new_assign
        if converged:
            break

    clusters = []
    if (~assign).sum() >= 2:
        clusters.append((lines[~assign], w[~assign]))
    if assign.sum() >= 2:
        clusters.append((lines[assign], w[assign]))
    return clusters


def compute_vanishing_points(lines, iterations=300, threshold=3.0, rng=None):
    """
    TWO orthogonal vanishing points, one per dominant floor-line direction.

    A floor plane's orientation has 2 degrees of freedom in the image, so it
    needs 2 real vanishing points. Blending every pairwise intersection into
    ONE point via a coordinate-wise median is not a vanishing point of
    anything, and forces the caller to invent the missing second direction.
    """
    clusters = _cluster_lines_by_orientation(lines)
    if len(clusters) < 2:
        return None, None
    r1 = _estimate_vp_ransac(clusters[0][0], iterations, threshold, rng, clusters[0][1])
    r2 = _estimate_vp_ransac(clusters[1][0], iterations, threshold, rng, clusters[1][1])
    vp1 = r1[0] if r1 else None
    vp2 = r2[0] if r2 else None
    return (vp1, clusters[0][0]), (vp2, clusters[1][0])


def _is_sane_vp(vp, w, h, max_factor=50.0):
    """
    Rejects vanishing points at (or numerically near) infinity. A frontal view
    legitimately puts one of the two floor VPs at infinity; its coordinates are
    then meaningless as a position.
    """
    if vp is None:
        return False
    if not np.all(np.isfinite(vp[:2])):
        return False
    return abs(vp[0]) < max_factor * w and abs(vp[1]) < max_factor * h


def _mean_abs_vertical_fraction(cluster):
    """
    How vertical a cluster's lines are in image space, in [0, 1].

    The floor direction running away from the camera projects to near-vertical
    segments; the cross direction to near-horizontal ones. This decides which
    of the two VPs drives the tile grid rotation -- choosing wrong rotates the
    grid by 90 degrees.
    """
    dx = cluster[:, 2] - cluster[:, 0]
    dy = cluster[:, 3] - cluster[:, 1]
    length = np.hypot(dx, dy) + 1e-6
    return float(np.mean(np.abs(dy) / length))


def _fit_vp_on_horizon(homo, weights, horizon_y):
    """
    Best vanishing point for a set of lines, CONSTRAINED to lie on the
    horizontal line y = horizon_y. Closed form.

    For a unit-normalised line l = (a, b, c), the distance from (vx, hy, 1) is
    |a*vx + b*hy + c|. Holding hy fixed leaves a 1-D weighted least squares in
    vx, minimised by

        vx = -sum(w * a * k) / sum(w * a^2),    k = b*hy + c

    Returns (vx, residual). residual is the weighted sum of squared distances,
    which is what the horizon search below compares.
    """
    a = homo[:, 0]
    k = homo[:, 1] * horizon_y + homo[:, 2]
    denom = float(np.sum(weights * a * a))
    if denom < 1e-12:
        # Every line is horizontal: it has no opinion about vx at all.
        return None, float("inf")
    vx = -float(np.sum(weights * a * k)) / denom
    resid = float(np.sum(weights * (a * vx + k) ** 2))
    return vx, resid


def _joint_horizon_vp_fit(homo_a, wa, homo_b, wb, image_h, init_hy,
                          rounds=3, span=None, samples=161):
    """
    Fit ONE horizon and BOTH vanishing points together, instead of fitting the
    two VPs independently and hoping they agree.

    Why this is the right constraint
    --------------------------------
    Both floor vanishing points lie on the horizon by construction -- that is
    what makes them floor VPs. Fitting them independently throws that away, and
    measured across 65 rooms it showed: only 15% of accepted detections put the
    depth VP within 2% of the image height of their own reported horizon, and
    the median miss was 30% of image height. Those results were not two views
    of one camera, they were two unrelated estimates.

    Enforcing a shared horizon turns two under-determined 2-D fits into one
    1-D search plus two 1-D closed forms. Every line in both clusters
    constrains the single horizon, so the estimate uses roughly twice the
    evidence for the quantity the floor pitch depends on.

    The horizon is taken as HORIZONTAL. That is not an extra approximation:
    the renderer already forces plane_a = 0.0 (see tile_engine's analytic plane)
    and keeps only horizon_y, discarding roll. A tilted horizon here would be
    silently flattened downstream anyway, so fitting one would add freedom the
    consumer cannot use, at the cost of a worse-conditioned fit.

    Robustified by IRLS with Cauchy weights so a handful of rug or furniture
    lines that survived the floor filter cannot drag the horizon.

    Returns (horizon_y, vx_a, vx_b, info).
    """
    lo = max(-image_h, init_hy - (span if span else image_h))
    hi = min(2.0 * image_h, init_hy + (span if span else image_h))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = -image_h, 2.0 * image_h

    ra = np.ones(len(homo_a), dtype=np.float64)
    rb = np.ones(len(homo_b), dtype=np.float64)

    best = (init_hy, None, None, float("inf"))
    for _ in range(max(1, rounds)):
        grid = np.linspace(lo, hi, samples)
        best = (init_hy, None, None, float("inf"))
        for hy in grid:
            va, resa = _fit_vp_on_horizon(homo_a, wa * ra, hy)
            vb, resb = _fit_vp_on_horizon(homo_b, wb * rb, hy)
            if va is None and vb is None:
                continue
            total = (0.0 if not np.isfinite(resa) else resa) + \
                    (0.0 if not np.isfinite(resb) else resb)
            if total < best[3]:
                best = (float(hy), va, vb, total)

        hy, va, vb, _ = best
        if va is None and vb is None:
            break

        # Cauchy reweighting on the current residuals. Scale is the median
        # absolute residual, so it adapts to how noisy this particular image's
        # lines are rather than assuming a pixel tolerance.
        def _reweight(homo, vx):
            if vx is None:
                return np.ones(len(homo), dtype=np.float64)
            d = np.abs(homo @ np.array([vx, hy, 1.0]))
            s = max(float(np.median(d)), 1e-3)
            return 1.0 / (1.0 + (d / (2.0 * s)) ** 2)

        ra = _reweight(homo_a, va)
        rb = _reweight(homo_b, vb)

        # Tighten the bracket around the current best for the next round.
        step = (hi - lo) / (samples - 1)
        lo, hi = hy - 4.0 * step, hy + 4.0 * step

    hy, va, vb, resid = best
    return hy, va, vb, {"residual": resid}


def _vp_direction_rad(vp_x, vp_y, cx, cy):
    """
    Direction from the principal point to a VP, folded to [0, pi).

    Folded because a VP and its antipode describe the SAME family of parallel
    world lines and produce the same tile grid orientation; treating them as
    2*pi apart would report a perfect agreement as a total disagreement.
    """
    return float(np.arctan2(vp_y - cy, vp_x - cx) % np.pi)


def _circular_spread_deg(angles_rad):
    """
    Peak-to-peak spread of angles that live on a half-circle, in degrees.

    Computed via the doubled-angle trick so the wrap at pi is handled: the
    angles are mapped onto the full circle, the mean direction is taken as a
    vector sum, and each angle's deviation is measured against it. A plain
    max-minus-min would report two readings straddling pi -- 179 degrees and 1
    degree, which are 2 degrees apart -- as a 178 degree disagreement and
    reject a perfectly stable detection.
    """
    if angles_rad is None or len(angles_rad) < 2:
        return 0.0
    a = np.asarray(angles_rad, dtype=np.float64) * 2.0
    mean = np.arctan2(np.sin(a).mean(), np.cos(a).mean())
    dev = np.abs(np.angle(np.exp(1j * (a - mean))))
    # Halved on the way out to undo the doubling.
    return float(np.degrees(dev.max()) / 2.0)


def _circular_median_rad(angles_rad):
    """Mean direction on the doubled circle, folded back to [0, pi)."""
    a = np.asarray(angles_rad, dtype=np.float64) * 2.0
    mean = np.arctan2(np.sin(a).mean(), np.cos(a).mean())
    return float((mean / 2.0) % np.pi)


def _select_depth_vp(cand_a, cand_b, boundary_lines):
    """
    Decide which of the two floor VPs is the DEPTH direction.

    Getting this wrong rotates the tile grid by exactly 90 degrees, which is
    the most visible failure this module can produce, so it is not left to a
    single heuristic. The old test was line orientation alone -- the fraction
    of a cluster's segments that run near-vertically in the image. That is a
    good signal and it is still the strongest term, but it degrades badly in
    two common cases: a camera rolled toward 45 degrees makes both clusters
    similarly "vertical", and a floor shot nearly along one tile axis leaves
    the cross direction almost as steep as the depth direction.

    So three independent signals are scored and summed:

      orientation  fraction of the cluster's segments running near-vertical.
      support      RANSAC inlier strength; the better-supported VP is the more
                   trustworthy candidate when orientation is ambiguous.
      boundary     agreement with the floor-to-wall junction. Side-wall
                   junctions run in the depth direction, so this breaks ties
                   using real room geometry rather than texture statistics.

    Returns (depth, cross, margin). `margin` is the normalised score gap; a
    small margin means the two directions were nearly indistinguishable and
    the 90-degree flip was close to a coin toss, so the caller folds it into
    the confidence score rather than discarding it.
    """
    def _score(c):
        orient = c["vertical_fraction"]
        support = c["support"]
        boundary = _boundary_agreement(boundary_lines, c["vp"][0], c["vp"][1])
        boundary = 0.0 if boundary is None else boundary
        w = DEPTH_SELECT_WEIGHTS
        total_w = w["orientation"] + w["support"] + w["boundary"]
        total = (w["orientation"] * orient
                 + w["support"] * support
                 + w["boundary"] * boundary) / total_w
        # The per-term breakdown is retained, not just the total, so a wrong
        # depth/cross pick can be attributed to a specific signal instead of
        # being observed as a bare score. Diagnostic only -- nothing reads it
        # to make a decision.
        c["score_terms"] = {"orientation": float(orient),
                            "support": float(support),
                            "boundary": float(boundary),
                            "total": float(total)}
        return total

    sa, sb = _score(cand_a), _score(cand_b)
    margin = abs(sa - sb) / max(sa, sb, 1e-6)
    if sa >= sb:
        return cand_a, cand_b, margin
    return cand_b, cand_a, margin


def _angular_deviation_to_point(lines, px, py):
    """
    Per-segment angle (radians) between the segment's own direction and the
    direction from its midpoint to (px, py). Zero means the segment points
    exactly at that point.

    Uses |cos| so a segment's arbitrary endpoint order cannot turn a perfect
    alignment into a 180-degree error.
    """
    mx = 0.5 * (lines[:, 0] + lines[:, 2])
    my = 0.5 * (lines[:, 1] + lines[:, 3])

    dx = lines[:, 2] - lines[:, 0]
    dy = lines[:, 3] - lines[:, 1]
    dn = np.hypot(dx, dy) + 1e-9

    tx = px - mx
    ty = py - my
    tn = np.hypot(tx, ty) + 1e-9

    cos = np.abs((dx * tx + dy * ty) / (dn * tn))
    return np.arccos(np.clip(cos, 0.0, 1.0))


def _boundary_agreement(boundary_lines, vp_x, vp_y, vp2=None,
                        tolerance_deg=BOUNDARY_AGREEMENT_DEG):
    """
    Length-weighted fraction of floor-to-wall junction segments that actually
    point at one of the detected vanishing points.

    This is the term that separates a real room VP from a rug-pattern VP. The
    wall junction is a physical line of the room, so it must converge with the
    floor grid; a VP fitted to carpet weave has no reason to align with it.

    Measured as an ANGLE, not as the perpendicular distance from the VP to the
    line. Those are equivalent only for a VP near the image: at 3000px out, 6px
    of perpendicular distance is a tenth of a degree, so a distance test scores
    ~0 for every distant VP regardless of how well it agrees. That made the
    term a near-constant penalty rather than a discriminator.

    Length-weighted, because a 400px wall junction is far better evidence than
    a 30px fragment. Returns None when there are no boundary lines to check,
    which the score treats as "no information" rather than as a failure.
    """
    if boundary_lines is None or len(boundary_lines) == 0:
        return None

    lines = np.asarray(boundary_lines, dtype=np.float64)
    lengths = np.hypot(lines[:, 2] - lines[:, 0], lines[:, 3] - lines[:, 1])
    tol = np.radians(tolerance_deg)

    hits = _angular_deviation_to_point(lines, vp_x, vp_y) < tol
    if vp2 is not None and np.all(np.isfinite(np.asarray(vp2, dtype=np.float64)[:2])):
        v2 = np.asarray(vp2, dtype=np.float64)
        hits = hits | (_angular_deviation_to_point(lines, v2[0], v2[1]) < tol)

    total = float(lengths.sum())
    if total <= 1e-6:
        return None
    return float(lengths[hits].sum() / total)


def score_vp_confidence(terms):
    """
    Weighted geometric mean of the confidence terms present in `terms`.

    Geometric rather than arithmetic on purpose. The failure modes this guards
    against are single-term collapses: a patterned rug yields high RANSAC
    support, high restart stability and a healthy line count while agreeing
    with nothing in the room's actual geometry. Averaged, that scores well
    above threshold and gets used. Multiplied, the near-zero boundary term
    pulls it under and the caller falls back -- which is the whole reason the
    score exists.

    Terms that could not be measured are omitted by the caller rather than
    filled with a default, so absent evidence neither rewards nor punishes.
    """
    num = 0.0
    den = 0.0
    for name, value in terms.items():
        if value is None:
            continue
        w = CONFIDENCE_WEIGHTS.get(name, 1.0)
        # Floor at a small epsilon: a genuine zero would annihilate the product
        # and erase the difference between "bad" and "catastrophic".
        v = float(np.clip(value, 1e-3, 1.0))
        num += w * np.log(v)
        den += w
    if den <= 0:
        return 0.0
    return float(np.exp(num / den))


def detect_floor_vanishing_points(
    room_bgr: np.ndarray,
    floor_mask: np.ndarray,
    principal_point=None,
    min_line_length: float = 20.0,
    min_floor_lines: int = 8,
    ransac_iterations: int = 300,
    ransac_threshold: float = 3.0,
    seed: int = 1337,
    n_restarts: int = 5,
    max_horizon_spread: float = 0.15,
    boundary=None,
    use_boundary_lines: bool = True,
    line_sample_points: int = LINE_SAMPLE_POINTS,
    line_inside_ratio: float = LINE_INSIDE_RATIO,
    clean_mask: bool = True,
    constrain_to_horizon: bool = False,
    max_vp_direction_spread_deg: float = MAX_VP_DIRECTION_SPREAD_DEG,
    min_confidence=None,
):
    """
    Returns (vp_x, vp_y, horizon_y, info) on success, or
    (None, None, None, info) when there is not enough usable floor-line
    evidence -- in which case the caller must fall back rather than render
    from a guess.

    (vp_x, vp_y) is the depth-direction VP's own position, for ROTATION.
    horizon_y is the horizon at the principal point, for PITCH.
    See the module docstring for why those must not be swapped.

    `info` always carries `stage` (why it failed, or "ok") and `confidence`,
    and on success additionally:
      vp_x, vp_y          the selected depth VP
      vp2                 the second (cross-direction) VP, also used for
                          two-VP focal calibration in focal_estimate.py
      vp1                 alias of the selected depth VP, kept for callers
                          written against the previous field name
      horizon_y           horizon height at the principal point, for PITCH
      confidence_terms    the individual components, so a low score can be
                          explained rather than merely observed
      floor_lines         lines surviving the floor filter
      boundary_lines      floor-to-wall junction segments mixed in
      floor_line_length   total length of the surviving floor lines, px
      horizon_spread      restart disagreement on the horizon, as a fraction
                          of image height
      vp_direction_spread restart disagreement on the depth VP's direction,
                          in degrees -- the quantity grid rotation depends on
      depth_select_margin how decisively the depth direction beat the cross
                          direction; near zero means the 90-degree choice was
                          nearly a coin toss

    `boundary` is the dict from floor_boundary.extract_floor_boundary. Pass it
    when the caller has already computed it (the renderer has); it is derived
    here otherwise.

    `min_confidence` optionally applies the accept/reject gate here instead of
    in the caller: below it the function returns (None, None, None, info)
    rather than a VP the caller is expected to know not to trust. It defaults
    to None because tile_engine.py already applies exactly this gate via
    RenderOptions.vp_confidence_threshold; setting both is harmless but
    redundant. Either way no VP is ever invented or approximated -- a rejected
    detection returns None, it does not degrade into a guess.
    """
    info = {"stage": "start", "total_lines": 0, "floor_lines": 0,
            "boundary_lines": 0, "confidence": 0.0}

    if room_bgr is None or floor_mask is None:
        info["stage"] = "no-input"
        return None, None, None, info

    h, w = room_bgr.shape[:2]
    cx, cy = principal_point if principal_point is not None else (w / 2.0, h / 2.0)

    gray = cv2.cvtColor(room_bgr, cv2.COLOR_BGR2GRAY) if room_bgr.ndim == 3 else room_bgr

    # 1. Lines over the whole room.
    lines = detect_lines_lsd(gray)
    info["total_lines"] = int(len(lines))

    # 2. Keep only those running ALONG the floor. The mask is cleaned first,
    #    for this test only -- the renderer's own floor_bool is untouched.
    filter_mask = clean_floor_mask_for_vp(floor_mask) if clean_mask else floor_mask
    floor_lines = filter_lines_by_floor_mask(
        lines, filter_mask, min_line_length,
        n_samples=line_sample_points, inside_ratio=line_inside_ratio,
    )
    info["floor_lines"] = int(len(floor_lines))
    info["floor_line_length"] = (
        float(np.hypot(floor_lines[:, 2] - floor_lines[:, 0],
                       floor_lines[:, 3] - floor_lines[:, 1]).sum())
        if len(floor_lines) else 0.0
    )

    # 3. Floor-to-wall junction segments, weighted above texture lines.
    if boundary is None and use_boundary_lines:
        boundary, _ = extract_floor_boundary(floor_mask, room_bgr.shape)

    boundary_lines = None
    visibility = None
    if boundary is not None:
        visibility = boundary.get("visible_fraction")
        if use_boundary_lines:
            wall = boundary.get("wall_lines")
            if wall is not None and len(wall):
                boundary_lines = np.asarray(wall, dtype=np.float32)

    if boundary_lines is not None and len(boundary_lines):
        combined = np.vstack([floor_lines, boundary_lines]).astype(np.float32)
        weights = np.concatenate([
            np.ones(len(floor_lines), dtype=np.float64),
            np.full(len(boundary_lines), BOUNDARY_LINE_WEIGHT, dtype=np.float64),
        ])
        info["boundary_lines"] = int(len(boundary_lines))
    else:
        combined = floor_lines
        weights = np.ones(len(floor_lines), dtype=np.float64)
        info["boundary_lines"] = 0

    if len(combined) < min_floor_lines:
        info["stage"] = "too-few-floor-lines"
        return None, None, None, info

    # 4. Two direction clusters. Clustering is deterministic, so it is done
    #    once and only the RANSAC below is restarted.
    clusters = _cluster_lines_by_orientation(combined, weights=weights)
    if len(clusters) < 2:
        info["stage"] = "no-two-directions"
        return None, None, None, info

    (lines_a, wa), (lines_b, wb) = clusters[0], clusters[1]
    vert_a = _mean_abs_vertical_fraction(lines_a)
    vert_b = _mean_abs_vertical_fraction(lines_b)

    # 5. Restart the RANSAC and require the results to agree.
    #
    # Which cluster is the DEPTH direction is decided per restart from the
    # fitted VPs themselves (orientation + support + boundary agreement),
    # not once up front from orientation alone -- see _select_depth_vp.
    horizons, vp_xs, vp_ys, supports, margins, directions = [], [], [], [], [], []
    cross_xs, cross_ys, consistencies = [], [], []
    select_detail = None
    for k in range(max(1, n_restarts)):
        rng = np.random.default_rng(seed + k * 7919)
        ra = _estimate_vp_ransac(lines_a, ransac_iterations, ransac_threshold, rng, wa)
        rb = _estimate_vp_ransac(lines_b, ransac_iterations, ransac_threshold, rng, wb)
        vp_a, sup_a = ra if ra else (None, 0.0)
        vp_b, sup_b = rb if rb else (None, 0.0)

        sane_a = _is_sane_vp(vp_a, w, h)
        sane_b = _is_sane_vp(vp_b, w, h)
        if not sane_a and not sane_b:
            continue

        if sane_a and sane_b:
            cand_a = {"vp": vp_a, "support": sup_a, "vertical_fraction": vert_a}
            cand_b = {"vp": vp_b, "support": sup_b, "vertical_fraction": vert_b}
            depth, cross, margin = _select_depth_vp(cand_a, cand_b, boundary_lines)
            depth_vp, cross_vp = depth["vp"], cross["vp"]
            depth_sup = depth["support"]
            # Diagnostic record of the depth/cross decision, kept from the
            # LAST restart. Lets a wrong pick be attributed to orientation,
            # support or boundary agreement rather than guessed at.
            select_detail = {
                "vp_a": (float(vp_a[0]), float(vp_a[1])),
                "vp_b": (float(vp_b[0]), float(vp_b[1])),
                "a_terms": cand_a.get("score_terms"),
                "b_terms": cand_b.get("score_terms"),
                "picked": "A" if depth is cand_a else "B",
                "margin": float(margin),
            }
        else:
            # Only one VP is usable; a frontal view legitimately sends the
            # other to infinity. There is nothing to choose between.
            depth_vp = vp_a if sane_a else vp_b
            cross_vp = None
            depth_sup = sup_a if sane_a else sup_b
            margin = 0.0

        # Horizon at the principal point, from the line through both VPs --
        # but only when that line is plausibly a horizon at all.
        #
        # Fitting through the pair averages out noise and is the better
        # estimator whenever the pair agrees. When the line through them comes
        # out steep the pair does NOT agree: both VPs lie on the true horizon
        # by construction, so a steep connector means one of them is wrong,
        # and extrapolating along it to the principal point turns a bad VP into
        # a wild horizon. In that case the estimate falls back to the depth
        # VP's own y -- itself a point on the horizon, and merely noisy rather
        # than extrapolated -- and the restart is recorded as inconsistent so
        # the confidence score can account for it.
        # Scored on a graded scale rather than pass/fail. Measured across 65
        # rooms the connector's tilt is spread right across the range -- 40%
        # within 30 degrees, 60% within 45, a tail out to 84 -- so a binary
        # flag would collapse to zero on most images and, in a geometric mean,
        # crush every score indiscriminately instead of ranking them. The
        # 30-degree threshold below still decides the horizon FALLBACK, which
        # is a correctness question with a real answer; the grade only feeds
        # confidence.
        pair_tilt = None
        if cross_vp is not None and abs(cross_vp[0] - depth_vp[0]) > 1e-3:
            slope = (cross_vp[1] - depth_vp[1]) / (cross_vp[0] - depth_vp[0])
            pair_tilt = abs(math.atan(slope))
            if abs(slope) <= MAX_HORIZON_SLOPE:
                hy = float(depth_vp[1] + (cx - depth_vp[0]) * slope)
            else:
                hy = float(depth_vp[1])
        else:
            hy = float(depth_vp[1])

        # The depth VP's own position, kept intact for the rotation maths.
        vx, vy = float(depth_vp[0]), float(depth_vp[1])
        if not (np.isfinite(hy) and np.isfinite(vx) and np.isfinite(vy)):
            continue

        horizons.append(hy)
        vp_xs.append(vx)
        vp_ys.append(vy)
        supports.append(depth_sup)
        margins.append(margin)
        # 1.0 for a perfectly horizontal connector, 0.0 for a vertical one.
        consistencies.append(1.0 if pair_tilt is None
                             else max(0.0, 1.0 - pair_tilt / (math.pi / 2.0)))
        directions.append(_vp_direction_rad(vx, vy, cx, cy))
        if cross_vp is not None:
            cross_xs.append(float(cross_vp[0]))
            cross_ys.append(float(cross_vp[1]))

    info["restarts_ok"] = len(horizons)

    if len(horizons) < max(1, min(3, n_restarts)):
        info["stage"] = "too-few-stable-restarts"
        return None, None, None, info

    spread = (max(horizons) - min(horizons)) / max(h, 1)
    info["horizon_spread"] = spread
    if spread > max_horizon_spread:
        info["stage"] = "unstable-horizon"
        return None, None, None, info

    # Rotation stability, which the horizon test cannot see. The grid's angle
    # comes from the DIRECTION to the depth VP, so a VP sliding along its own
    # ray between restarts is harmless while a few degrees of swing is not --
    # and both can leave the horizon perfectly steady. Gated separately.
    direction_spread = _circular_spread_deg(directions)
    info["vp_direction_spread"] = direction_spread
    if direction_spread > max_vp_direction_spread_deg:
        info["stage"] = "unstable-vp-direction"
        return None, None, None, info

    horizon_y = float(np.median(horizons))
    vp_x = float(np.median(vp_xs))
    vp_y = float(np.median(vp_ys))

    cross_vp = None
    if cross_xs:
        cross_vp = (float(np.median(cross_xs)), float(np.median(cross_ys)))

    # ---- Joint horizon-constrained refit (OFF by default -- measured worse) ----
    # The idea: both floor VPs lie on the horizon by construction, so fit them
    # and the horizon together rather than as three independent estimates.
    # Geometrically motivated, and it does what it says -- 100% of results land
    # exactly on their own horizon, and horizon-out-of-range failures go to
    # zero, with detection up 52 -> 54 of 65.
    #
    # It is off anyway, because it makes the vanishing points WORSE by the one
    # independent measure available. Across the corpus, holding everything else
    # fixed, the confidence terms moved:
    #
    #     support 0.446 -> 0.446      lines      0.869 -> 0.869
    #     stability 0.928 -> 0.928    balance    0.500 -> 0.500
    #     direction 0.926 -> 0.926    visibility 0.954 -> 0.954
    #     boundary  0.252 -> 0.158                        <-- only mover, -37%
    #
    # boundary agreement is how well the floor-to-wall junction points at the
    # VP. It is the one term grounded in real room geometry rather than in the
    # fit's own residuals, so a fit that improves its self-consistency while
    # agreeing LESS with the walls has moved away from the room, not toward it.
    # The likely cause is the horizontal-horizon constraint: the renderer
    # discards roll downstream, but the IMAGE may still have some, and forcing
    # y_a == y_b then bends both VPs to absorb it.
    #
    # Kept, not deleted, because the constraint is sound and the failure is
    # probably in this implementation of it -- a roll-aware horizon (fit a
    # tilted line, then report its height at the principal point) is the
    # obvious next attempt. Enable with constrain_to_horizon=True to compare.
    #
    # The RANSAC medians above are kept as the starting point, so this inherits
    # their outlier rejection rather than replacing it.
    if constrain_to_horizon:
        depth_is_a = (info.get("depth_select_detail") or {}).get("picked", "A") == "A"
        homo_a, homo_b = _lines_to_homo(lines_a), _lines_to_homo(lines_b)
        j_hy, j_va, j_vb, j_info = _joint_horizon_vp_fit(
            homo_a, wa, homo_b, wb, h, horizon_y)

        jd, jc = (j_va, j_vb) if depth_is_a else (j_vb, j_va)
        if jd is not None and np.isfinite(j_hy):
            info["horizon_shift"] = float(j_hy - horizon_y)
            info["vp_shift"] = float(abs(jd - vp_x))
            horizon_y = float(j_hy)
            vp_x, vp_y = float(jd), float(j_hy)     # ON the horizon, by construction
            cross_vp = (float(jc), float(j_hy)) if jc is not None else None
            info["joint_fit"] = {"residual": j_info["residual"]}
        else:
            info["joint_fit"] = {"residual": None, "note": "fell back to medians"}

    # A horizon far outside the frame means the fit collapsed; a real indoor
    # floor shot puts it in or just beyond the image.
    if horizon_y < -h or horizon_y > 2.0 * h:
        info["stage"] = "horizon-out-of-range"
        info["horizon_y"] = horizon_y
        return None, None, None, info

    # 6. Confidence. Every term is in [0, 1] and measures something the others
    #    do not; see score_vp_confidence for why they are combined
    #    multiplicatively.
    twa, twb = float(wa.sum()), float(wb.sum())
    depth_margin = float(np.median(margins)) if margins else 0.0

    # Line evidence scored on COUNT and total LENGTH together. Count alone
    # rates forty 25px specks off a marble vein as richer evidence than four
    # 500px grout lines spanning the room, when the opposite is true: long
    # lines pin down an intersection far outside the frame, short ones barely
    # constrain it. The geometric mean of the two demands both.
    line_count_term = min(1.0, len(combined) / CONFIDENCE_FULL_LINE_COUNT)
    line_length_term = min(1.0, info["floor_line_length"]
                           / (CONFIDENCE_FULL_LINE_LENGTH_FRAC * max(w, h)))
    terms = {
        "support": float(np.median(supports)) if supports else None,
        "lines": float(np.sqrt(max(line_count_term, 1e-6) * max(line_length_term, 1e-6))),
        # Worst of the two instabilities, not the horizon's alone. A detection
        # is only as stable as its least stable output.
        "stability": min(
            1.0 - min(1.0, spread / max(max_horizon_spread, 1e-6)),
            1.0 - min(1.0, direction_spread / max(max_vp_direction_spread_deg, 1e-6)),
        ),
        "balance": min(twa, twb) / max(max(twa, twb), 1e-6),
        "boundary": _boundary_agreement(boundary_lines, vp_x, vp_y, cross_vp),
        # How decisively the depth direction beat the cross direction. A near
        # tie means the 90-degree choice was close to a coin toss, and a grid
        # rotated 90 degrees from the room is the most visible failure this
        # module can produce -- so an ambiguous pick lowers confidence rather
        # than passing silently.
        "direction": min(1.0, depth_margin / DEPTH_MARGIN_FULL_SCORE),
        # Normalised against what a normal room photo looks like. Raw
        # visible_fraction sits near 0.45 on almost every image (the floor runs
        # out of the bottom of the frame), so scoring it directly applied the
        # same penalty to every photo and discriminated nothing.
        "visibility": (None if visibility is None
                       else min(1.0, visibility / TYPICAL_VISIBLE_FRACTION)),
    }
    confidence = score_vp_confidence(terms)

    info["horizon_y"] = horizon_y
    info["vp_x"] = vp_x
    info["vp_y"] = vp_y
    # vp1 is the SELECTED depth VP and vp2 the cross direction. focal_estimate
    # only needs the pair to be mutually orthogonal, which they are either way.
    info["vp1"] = (vp_x, vp_y)
    info["vp2"] = cross_vp
    info["depth_select_margin"] = depth_margin
    info["depth_select_detail"] = select_detail
    info["cluster_sizes"] = (int(len(lines_a)), int(len(lines_b)))
    info["cluster_vertical_fraction"] = (float(vert_a), float(vert_b))
    # How horizontal the line through the two VPs came out, averaged over
    # restarts: 1.0 for a level connector, 0.0 for a vertical one. Reported as
    # DEBUG ONLY and deliberately not a confidence term. It was tried as one
    # and measured anti-correlated with the final score -- 0.38 on accepted
    # detections against 0.52 on rejected ones -- because a well-conditioned
    # pair is not what makes a depth VP trustworthy. It is still worth
    # surfacing: a low value tells you the horizon came from the depth VP's own
    # y rather than from fitting through the pair.
    info["pair_consistency"] = float(np.mean(consistencies)) if consistencies else None
    info["confidence"] = confidence
    info["confidence_terms"] = {k: (None if v is None else round(float(v), 4))
                                for k, v in terms.items()}

    # 7. Optional gate. Off by default because tile_engine.py already applies
    #    it; when set, a low-confidence detection is reported as a failure
    #    rather than handed back for the caller to second-guess.
    if min_confidence is not None and confidence < min_confidence:
        info["stage"] = "low-confidence"
        return None, None, None, info

    info["stage"] = "ok"
    return vp_x, vp_y, horizon_y, info
