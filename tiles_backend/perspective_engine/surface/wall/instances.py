"""
Splitting the wall mask into one independent plane per visible wall.

This is the problem a floor does not have. ADE20K is SEMANTIC, not instance:
a photo of a room corner returns one connected "wall" region covering two
walls that meet at 90 degrees. Fitting a single plane to that region fits a
plane through neither of them -- it slices the corner diagonally, and the tile
grid then slides along one wall and compresses on the other. Every visible
wall needs its own plane, basis, scale and anchor.

The split runs in three stages, each correcting the previous one's weakness:

  1. Sequential plane RANSAC over the wall's 3D points. Extract the largest
     plane, remove its inliers, repeat. Depth-driven, so it finds real
     geometric breaks rather than colour edges -- but its per-pixel labels are
     noisy where depth is soft, which is exactly at the corner.

  2. Column-majority regularisation. A plumb wall meeting another plumb wall
     produces a VERTICAL seam in the image. So the per-pixel labels are voted
     per column and the result is smoothed, which turns a ragged depth-driven
     boundary into a clean vertical cut.

  3. Seam snapping to detected vertical lines. The corner itself is usually a
     strong line segment; if LSD found one near the voted seam, the seam moves
     onto it. This is what makes the boundary land on the actual corner rather
     than a few pixels off it.
"""

import math

import numpy as np
import cv2

from ...core.plane_fit import WallConstraint, fit_plane_ransac
from ..base import SurfaceInstance

#: Never look for more than this many walls in one photo. Three is already
#: generous -- a rectangular room shows at most three walls from inside.
MAX_WALL_INSTANCES = 3

#: A plane must claim at least this share of the wall pixels to be kept.
MIN_PLANE_SHARE = 0.08

#: Half-width of the window used to smooth per-column labels, as a fraction of
#: image width.
COLUMN_SMOOTH_FRACTION = 0.02

#: A voted seam snaps to a detected vertical line within this fraction of the
#: image width.
SEAM_SNAP_FRACTION = 0.035

#: Plane-inlier tolerance, as a FRACTION OF THE REGION'S OWN DEPTH.
#:
#: opts.ransac_threshold is an absolute distance, and the units it is measured
#: in are whatever Z = 1000 / (d * contrast + 0.05) happens to produce -- on a
#: real room that was 952 to 20000, median 3279. Comparing a threshold of 5 or
#: 10 against those magnitudes is a tolerance of 0.15%, and it showed: 88% of
#: wall pixels were inliers of NO plane, every pixel was then assigned to its
#: nearest plane anyway, and the column vote that follows was driven by noise.
#: Scaling the tolerance to the median depth of the region being fitted makes
#: it mean the same thing whatever the depth map's arbitrary scale.
DEPTH_RELATIVE_TOLERANCE = 0.06

#: A plane must claim at least this share of the remaining points to be taken
#: as a real wall rather than a slice through noise.
MIN_REMAINING_SHARE = 0.10


#: Two fitted planes closer than this in normal direction AND in distance are
#: the same wall found twice.
#:
#: Tuned against measurements at full resolution, not guessed:
#:
#:   image_ne1  back wall split in two   24.7 deg, 38.3% apart  -> must MERGE
#:              back wall vs side wall   40.8 deg              -> must NOT
#:   room3      two genuinely different walls  13.7 deg, 68.1% -> must NOT
#:              (kept apart by the DISTANCE test, not the angle)
#:
#: 26 deg sits between 24.7 and 40.8 with room either side. The distance test
#: is what separates walls whose normals are similar, including parallel walls
#: on opposite sides of a room -- those share a normal exactly and differ by
#: the room's width, far beyond 40%.
PLANE_MERGE_ANGLE_DEG = 26.0
PLANE_MERGE_DISTANCE_FRACTION = 0.40

#: A run of columns shorter than this fraction of the image width is not a
#: wall, it is noise in the column vote. A wall a person would want to tile
#: occupies a good slice of the frame; the noise stripes measured 4-6%.
MIN_COLUMN_RUN_FRACTION = 0.10


def _dedupe_planes(planes, depth_scale):
    """
    Merge planes that are really the same wall found twice.

    Sequential RANSAC has no idea it has already seen a surface: on a large
    flat wall it happily extracts one plane, removes those inliers, and then
    fits a SECOND plane through the leftovers that is within a couple of
    degrees of the first. Both survive, every pixel is then assigned to
    whichever of the two is marginally nearer, and that choice flips on noise
    -- which is exactly how one flat back wall came back as four interleaved
    vertical stripes, red/blue/red/blue.

    Two planes are the same wall when their normals point the same way AND
    they sit at the same distance. Both tests are needed: parallel walls on
    opposite sides of a room share a normal and differ only in distance.
    """
    if len(planes) <= 1:
        return planes

    cos_tol = math.cos(math.radians(PLANE_MERGE_ANGLE_DEG))
    dist_tol = PLANE_MERGE_DISTANCE_FRACTION * max(depth_scale, 1e-6)

    kept = []
    for plane in planes:
        a, b, c, d = plane
        duplicate = False
        for ka, kb, kc, kd in kept:
            if (a * ka + b * kb + c * kc) >= cos_tol and abs(d - kd) <= dist_tol:
                duplicate = True
                break
        if not duplicate:
            kept.append(plane)
    return kept


def _enforce_min_run(column_labels, min_run):
    """
    Absorb column runs too short to be a wall into their neighbours.

    A seam is a corner, and corners are metres apart, not tens of pixels. A
    handful of columns disagreeing with everything either side of them is
    depth noise, and left alone it produces a sliver "wall" with its own plane
    and its own tile grid.
    """
    labels = np.asarray(column_labels).copy()
    n = len(labels)
    if n == 0:
        return labels

    changed = True
    while changed:
        changed = False
        start = 0
        while start < n:
            if labels[start] < 0:
                start += 1
                continue
            end = start
            while end + 1 < n and labels[end + 1] == labels[start]:
                end += 1
            length = end - start + 1

            if length < min_run:
                # Take the longer neighbouring run's label.
                left = labels[start - 1] if start > 0 else -1
                right = labels[end + 1] if end + 1 < n else -1
                replacement = left if left >= 0 else right
                if left >= 0 and right >= 0 and left != right:
                    # Whichever neighbour runs longer wins.
                    li = start - 1
                    while li > 0 and labels[li - 1] == left:
                        li -= 1
                    ri = end + 1
                    while ri + 1 < n and labels[ri + 1] == right:
                        ri += 1
                    replacement = left if (start - li) >= (ri - end) else right
                if replacement >= 0:
                    labels[start:end + 1] = replacement
                    changed = True
            start = end + 1

    return labels


def _adaptive_threshold(Z, opts) -> float:
    """
    Inlier tolerance in the depth map's own units.

    Never smaller than the caller's ransac_threshold, so an explicitly loose
    setting is still honoured; scaled up to DEPTH_RELATIVE_TOLERANCE of the
    median depth when that is larger, which is the case for every real scene.
    """
    if len(Z) == 0:
        return float(opts.ransac_threshold)
    median_z = float(np.median(np.abs(Z)))
    return max(float(opts.ransac_threshold), DEPTH_RELATIVE_TOLERANCE * median_z)


def _backproject(mask_bool, depth_val, cx, cy, f, depth_contrast):
    """3D camera-space points for every pixel in the mask, plus their indices."""
    ys, xs = np.where(mask_bool)
    d = depth_val[ys, xs]
    Z = 1000.0 / (d * depth_contrast + 0.05)
    X = ((xs.astype(np.float32) - cx) * Z) / f
    Y = ((ys.astype(np.float32) - cy) * Z) / f
    return X, Y, Z, xs, ys


def _sequential_planes(X, Y, Z, xs, ys, opts, max_planes=MAX_WALL_INSTANCES):
    """
    Extract up to `max_planes` wall planes, largest first.

    Reuses core.plane_fit with a WallConstraint, so the plumb lock, the
    orientation convention and the determinism guarantees are the same ones
    the floor gets -- there is no second RANSAC implementation here.
    """
    constraint = WallConstraint(lock_vertical=opts.lock_vertical_wall)
    remaining = np.ones(len(X), dtype=bool)
    total = len(X)
    planes = []

    threshold = _adaptive_threshold(Z, opts)

    for _ in range(max_planes):
        n_left = int(np.count_nonzero(remaining))
        if n_left < max(50, int(MIN_PLANE_SHARE * total)):
            break

        idx = np.where(remaining)[0]
        plane = fit_plane_ransac(
            X[idx], Y[idx], Z[idx], xs[idx].astype(np.float32), ys[idx].astype(np.float32),
            iterations=opts.ransac_iterations,
            threshold=threshold,
            constraint=constraint,
        )
        if plane == constraint.fallback:
            break

        a, b, c, d = plane
        dist = np.abs(a * X + b * Y + c * Z + d)
        inliers = remaining & (dist < threshold)
        n_inliers = int(np.count_nonzero(inliers))

        # Judged against what is LEFT, not against the original total: after
        # two walls are taken out a genuine third wall is only a small share of
        # the whole, and testing it against the whole is what stopped the back
        # wall of a three-wall room ever being found.
        if n_inliers < MIN_REMAINING_SHARE * n_left or n_inliers < 50:
            break

        planes.append(plane)
        remaining &= ~inliers

    # A large flat wall yields the same plane twice; keeping both makes every
    # pixel's label a coin flip between them.
    depth_scale = float(np.median(np.abs(Z))) if len(Z) else 1.0
    planes = _dedupe_planes(planes, depth_scale)

    return planes, threshold


def _assign_to_planes(X, Y, Z, planes):
    """Label each point with its nearest plane."""
    if not planes:
        return np.zeros(len(X), dtype=np.int32)
    dists = np.stack(
        [np.abs(a * X + b * Y + c * Z + d) for (a, b, c, d) in planes], axis=0
    )
    return np.argmin(dists, axis=0).astype(np.int32)


def _column_vote(labels, xs, width, n_planes):
    """
    Dominant plane per image column, smoothed.

    Two plumb walls meet along a vertical image line, so a per-column decision
    is not a simplification -- it is the correct model. Voting also repairs the
    depth noise that makes per-pixel labels unusable near the corner itself.
    """
    votes = np.zeros((n_planes, width), dtype=np.float32)
    np.add.at(votes, (labels, xs), 1.0)

    half = max(1, int(round(COLUMN_SMOOTH_FRACTION * width)))
    kernel = np.ones(2 * half + 1, dtype=np.float32)
    for p in range(n_planes):
        votes[p] = np.convolve(votes[p], kernel, mode="same")

    occupied = votes.sum(axis=0) > 0
    winner = np.argmax(votes, axis=0).astype(np.int32)
    winner[~occupied] = -1
    return winner


def _seams_from_columns(column_labels):
    """Column indices where the winning plane changes."""
    seams = []
    prev = None
    for x, lab in enumerate(column_labels):
        if lab < 0:
            continue
        if prev is not None and lab != prev:
            seams.append(x)
        prev = lab
    return seams


def _vertical_line_columns(room_bgr, wall_bool):
    """
    Columns carrying a strong vertical line inside the wall region.

    Reuses the shared LSD detector -- the same one floor VP detection uses --
    rather than introducing a second line detector.
    """
    from ...camera.vp_from_mask import detect_lines_lsd

    gray = cv2.cvtColor(room_bgr, cv2.COLOR_BGR2GRAY)
    lines = detect_lines_lsd(gray)
    if lines is None or len(lines) == 0:
        return np.array([], dtype=np.int32)

    h, w = wall_bool.shape[:2]
    cols = []
    for x1, y1, x2, y2 in lines:
        dx, dy = x2 - x1, y2 - y1
        length = float(np.hypot(dx, dy))
        if length < 0.08 * h:
            continue
        # Near-vertical only: a corner seam is plumb.
        if abs(dx) > 0.18 * abs(dy):
            continue
        xm = int(round((x1 + x2) * 0.5))
        ym = int(round((y1 + y2) * 0.5))
        if 0 <= xm < w and 0 <= ym < h and wall_bool[ym, xm]:
            cols.append(xm)
    return np.array(sorted(cols), dtype=np.int32)


def _snap_seams(seams, vertical_cols, width):
    """Move each voted seam onto a nearby real vertical line, when there is one."""
    if len(vertical_cols) == 0:
        return seams
    tol = max(2, int(round(SEAM_SNAP_FRACTION * width)))
    snapped = []
    for s in seams:
        d = np.abs(vertical_cols - s)
        j = int(np.argmin(d))
        snapped.append(int(vertical_cols[j]) if d[j] <= tol else int(s))
    return sorted(set(snapped))


#: Distinct colours for the per-wall region map, BGR. Chosen to stay legible
#: against a room photo and against each other.
INSTANCE_COLOURS = (
    (60, 90, 220),    # red
    (70, 190, 90),    # green
    (230, 150, 60),   # blue
    (60, 200, 230),   # amber
    (200, 90, 200),   # magenta
)


def render_instance_map(instances, shape, room_bgr=None, alpha=0.55):
    """
    A picture of WHICH AREA BELONGS TO WHICH WALL.

    The binary mask answers "is this wall?"; it cannot answer "which wall?",
    and that second question is the one that matters once a room shows three
    of them. Each instance gets its own colour, optionally blended over the
    room so the regions can be checked against what is actually in the photo.
    """
    h, w = shape[:2]
    canvas = (room_bgr.copy() if room_bgr is not None
              else np.zeros((h, w, 3), dtype=np.uint8))

    for inst in instances:
        colour = np.array(INSTANCE_COLOURS[inst.index % len(INSTANCE_COLOURS)],
                          dtype=np.float32)
        sel = inst.mask
        if not np.any(sel):
            continue
        if room_bgr is not None:
            canvas[sel] = (canvas[sel].astype(np.float32) * (1.0 - alpha)
                           + colour * alpha).astype(np.uint8)
        else:
            canvas[sel] = colour.astype(np.uint8)

    return canvas


def instance_white_mask(instance, shape):
    """
    One wall on its own: that wall white, everything else black.

    Same white-on-black convention as the whole-surface mask, so the UI can
    show a single wall wherever it would show all of them without needing a
    second code path for it.
    """
    h, w = shape[:2]
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[instance.mask] = (255, 255, 255)
    return out


def describe_instances(instances, shape, include_masks: bool = False):
    """
    Per-wall geometry, for the API and the UI: area, centroid, bounding box.

    include_masks adds each wall's own white-on-black mask, which is what lets
    the caller show ONLY the selected wall rather than the whole surface.
    """
    h, w = shape[:2]
    total = float(h * w)
    out = []
    for inst in instances:
        ys, xs = np.where(inst.mask)
        if len(xs) == 0:
            continue
        entry = {
            "index": inst.index,
            "label": inst.label,
            "pixels": int(len(xs)),
            "coverage": float(len(xs) / total),
            "centroid": [float(xs.mean()), float(ys.mean())],
            "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
            "colour_bgr": list(INSTANCE_COLOURS[inst.index % len(INSTANCE_COLOURS)]),
            "detection": inst.info,
        }
        if include_masks:
            entry["_mask_bgr"] = instance_white_mask(inst, shape)
        out.append(entry)
    return out


#: Two wall directions closer than this are the same wall seen with noise.
#:
#: This only has to be fine enough not to MISS a wall. Whether two directions
#: are really two walls is decided afterwards by comparing their fitted PLANES,
#: because an angle threshold alone cannot do it: on a real room the left wall
#: appeared at both -57 and -102 (45 apart, one wall) while the back wall sat
#: at -12 (also 45 from -57, a different wall). No single angle separates those
#: two cases -- but their planes do, immediately.
MIN_DIRECTION_SEPARATION_DEG = 30.0

#: A direction peak must hold at least this share of the wall pixels.
MIN_DIRECTION_SHARE = 0.06

#: A region must hold at least this share of the WALL to be its own wall.
#:
#: Measured against the wall, not the image, which is the difference between
#: 4 walls and 12: opts.wall_min_area_fraction is 1% of the FRAME, and on a
#: room whose walls fill half the frame that admitted every 1.2% speck along a
#: corner as its own wall, each with its own plane and tile grid. Anything
#: below this is merged into the nearest real wall rather than dropped, so
#: coverage stays complete.
MIN_INSTANCE_WALL_SHARE = 0.08

#: Upper bound on walls returned. A room can show several, but past this the
#: split is fragmenting rather than finding.
MAX_DETECTED_WALLS = 6

#: Azimuth is smoothed over this fraction of the diagonal before peaks are
#: found -- on sin/cos, because averaging angles directly is wrong across the
#: +/-180 wrap.
AZIMUTH_SMOOTH_FRACTION = 0.02


def _wall_azimuth(depth_val, cx, cy, f, depth_contrast):
    """
    The compass direction each pixel's surface faces, in degrees.

    atan2(nx, nz) of the surface normal: this IS a wall's direction, and it is
    what makes two walls different walls. A left wall faces +x, a back wall
    faces -z, and the corner between them is a 90-degree step in this field --
    visible without any assumption about where the corner sits in the image.

    Smoothed on sin/cos rather than on the angle, because averaging -179 and
    +179 the naive way gives 0 when the answer is 180.
    """
    from .depth_mask import surface_normals

    nx, _ny, nz = surface_normals(depth_val, cx, cy, f, depth_contrast)
    az = np.arctan2(nx, nz).astype(np.float32)

    h, w = depth_val.shape[:2]
    k = max(3, int(round(AZIMUTH_SMOOTH_FRACTION * np.hypot(h, w))) | 1)
    s = cv2.blur(np.sin(az), (k, k))
    c = cv2.blur(np.cos(az), (k, k))
    return np.degrees(np.arctan2(s, c))


def _direction_peaks(azimuth_deg, wall_bool, max_peaks=4):
    """
    The distinct directions the walls in this room face.

    A circular histogram of the azimuth over wall pixels, then its peaks. Each
    peak is one wall direction; peaks closer than MIN_DIRECTION_SEPARATION_DEG
    are the same direction found twice and the weaker is dropped.
    """
    vals = azimuth_deg[wall_bool]
    if vals.size == 0:
        return []

    bins = 72                                    # 5-degree resolution
    hist, edges = np.histogram(vals, bins=bins, range=(-180.0, 180.0))

    # Circular smoothing, so a direction straddling the wrap is not split in two.
    hist = hist.astype(np.float32)
    hist = (np.roll(hist, 1) + 2.0 * hist + np.roll(hist, -1)) / 4.0

    centres = (edges[:-1] + edges[1:]) / 2.0
    total = float(vals.size)
    half_window = MIN_DIRECTION_SEPARATION_DEG / 2.0

    # A direction's weight is the mass in a WINDOW around it, not the height of
    # one 5-degree bin.
    #
    # Measured: a real wall's normals spread over 20-30 degrees, so its tallest
    # single bin held only 4-13% of the wall while the window around it held
    # 11-43%. Testing the single bin against a 6% share therefore rejected
    # genuine walls -- and because candidates were walked tallest-first and the
    # test used `break`, one rejection discarded every remaining direction. A
    # three-wall room came back as one wall.
    window_mass = np.empty(bins, dtype=np.float32)
    for i in range(bins):
        gap = np.abs(((vals - centres[i] + 180.0) % 360.0) - 180.0)
        window_mass[i] = float(np.count_nonzero(gap <= half_window)) / max(total, 1.0)

    peaks = []
    for idx in np.argsort(hist)[::-1]:
        if len(peaks) >= max_peaks:
            break
        if window_mass[idx] < MIN_DIRECTION_SHARE:
            continue                              # not `break` -- see above
        angle = float(centres[idx])
        if any(_angular_gap(angle, p) < MIN_DIRECTION_SEPARATION_DEG for p in peaks):
            continue
        peaks.append(angle)

    return peaks


def _angular_gap(a, b):
    """Smallest absolute difference between two bearings, in degrees."""
    d = abs(float(a) - float(b)) % 360.0
    return d if d <= 180.0 else 360.0 - d


def _assign_to_directions(azimuth_deg, wall_bool, peaks):
    """Label every wall pixel with the direction peak it is nearest."""
    labels = np.full(azimuth_deg.shape, -1, dtype=np.int32)
    if not peaks:
        return labels
    gaps = np.stack([
        np.minimum(np.abs(azimuth_deg - p) % 360.0,
                   360.0 - (np.abs(azimuth_deg - p) % 360.0))
        for p in peaks
    ], axis=0)
    nearest = np.argmin(gaps, axis=0).astype(np.int32)
    labels[wall_bool] = nearest[wall_bool]
    return labels


def boundary_map(instances, shape, line_thickness=None):
    """
    Wall regions as BLACK with WHITE boundary lines.

    A filled colour overlay shows where the walls are; an outline shows where
    each one ENDS, which is the thing to check when the question is whether
    two walls were separated correctly. Every region is drawn as a closed
    contour, so a wall whose boundary is broken or leaking into its neighbour
    is obvious at a glance rather than buried under fill.
    """
    h, w = shape[:2]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    t = line_thickness or max(1, int(round(0.0018 * np.hypot(h, w))))

    for inst in instances:
        m = inst.mask.astype(np.uint8)
        if not m.any():
            continue
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, (255, 255, 255), t, cv2.LINE_AA)

    return canvas


def _vertical_architectural_lines(room_bgr):
    """Long near-vertical lines -- the candidates for a wall-wall corner."""
    from .layout import architectural_lines
    try:
        return architectural_lines(room_bgr)
    except Exception:                     # noqa: BLE001 - detection is best-effort
        return np.empty((0, 4), np.float32)


def _separated_by_corner_line(mask_a, mask_b, lines, shape):
    """
    Is there a real architectural line standing between these two regions?

    The merge test above compares planes, and a plane comparison alone can
    still swallow a shallow corner: two walls meeting at 30 degrees have
    similar normals and, if the room is small, similar distances. But a corner
    is a LINE in the photograph -- the one place the two walls' brightness
    differs sharply -- and if a long line runs between the regions then they
    are two walls whatever their planes say.

    Uses the columns each region occupies: a corner between them shows up as a
    line crossing the gap that separates their column ranges.
    """
    if lines is None or len(lines) == 0:
        return False

    cols_a = np.where(mask_a.any(axis=0))[0]
    cols_b = np.where(mask_b.any(axis=0))[0]
    if len(cols_a) == 0 or len(cols_b) == 0:
        return False

    # The band between them, if they are side by side rather than interleaved.
    if cols_a.max() < cols_b.min():
        lo, hi = int(cols_a.max()), int(cols_b.min())
    elif cols_b.max() < cols_a.min():
        lo, hi = int(cols_b.max()), int(cols_a.min())
    else:
        return False                     # overlapping columns: one wall's spread

    h, w = shape[:2]
    tol = max(4, int(round(0.02 * w)))
    lo, hi = lo - tol, hi + tol

    rows_a = np.where(mask_a.any(axis=1))[0]
    rows_b = np.where(mask_b.any(axis=1))[0]
    y_lo = min(int(rows_a.min()), int(rows_b.min()))
    y_hi = max(int(rows_a.max()), int(rows_b.max()))
    min_len = 0.25 * (y_hi - y_lo)

    for x1, y1, x2, y2 in lines:
        dx, dy = float(x2 - x1), float(y2 - y1)
        if abs(dy) < 1e-6 or abs(dx) > 0.35 * abs(dy):
            continue                     # not near-vertical: not a wall corner
        if math.hypot(dx, dy) < min_len:
            continue
        xm = 0.5 * (x1 + x2)
        if lo <= xm <= hi:
            return True
    return False


def _same_plane(p, q, depth_scale):
    """Do two fitted planes describe the same wall? Normal AND distance."""
    if p is None or q is None:
        return False
    cos_tol = math.cos(math.radians(PLANE_MERGE_ANGLE_DEG))
    dist_tol = PLANE_MERGE_DISTANCE_FRACTION * max(depth_scale, 1e-6)
    dot = p[0] * q[0] + p[1] * q[1] + p[2] * q[2]
    return dot >= cos_tol and abs(p[3] - q[3]) <= dist_tol


#: Two planes this close are the same wall beyond argument, and a line running
#: between them is a door jamb rather than a corner.
STRICT_MERGE_ANGLE_DEG = 8.0
STRICT_MERGE_DISTANCE_FRACTION = 0.10


def _same_plane_strongly(p, q, depth_scale):
    """
    Same wall beyond argument -- close enough to overrule the corner-line test.

    The corner-line test exists because a shallow corner can fool the plane
    comparison. It has an opposite failure of its own: a wall interrupted by a
    DOORWAY is two regions with a strong vertical line between them, and that
    line is a jamb, not a corner. Refusing the merge there splits one wall in
    two, each with its own scale and anchor, and the tile grid then steps
    across the doorway. When the two planes agree this closely there is nothing
    for the line to overrule.
    """
    if p is None or q is None:
        return False
    cos_tol = math.cos(math.radians(STRICT_MERGE_ANGLE_DEG))
    dist_tol = STRICT_MERGE_DISTANCE_FRACTION * max(depth_scale, 1e-6)
    dot = p[0] * q[0] + p[1] * q[1] + p[2] * q[2]
    return dot >= cos_tol and abs(p[3] - q[3]) <= dist_tol


def _drop_specks(mask_bool, min_px):
    """Remove islands too small to be part of a wall, keep everything else."""
    num, comp, stats, _ = cv2.connectedComponentsWithStats(
        mask_bool.astype(np.uint8), connectivity=8)
    if num <= 1:
        return mask_bool
    speck_limit = max(64, min_px // 8)
    keep = np.zeros(num, dtype=bool)
    for i in range(1, num):
        keep[i] = stats[i, cv2.CC_STAT_AREA] >= speck_limit
    return keep[comp]


def _connected_pieces(mask_bool, min_px):
    """
    The region's connected components, largest first, specks left behind.

    Pieces below `min_px` are not returned as their own region -- they are
    picked up by the leftover pass and given to whichever real wall is nearest,
    so nothing goes untiled and nothing becomes a sliver wall of its own.
    """
    num, comp, stats, _ = cv2.connectedComponentsWithStats(
        mask_bool.astype(np.uint8), connectivity=8)
    if num <= 2:
        return [mask_bool] if np.any(mask_bool) else []

    order = sorted(range(1, num), key=lambda i: -stats[i, cv2.CC_STAT_AREA])
    pieces = [comp == i for i in order if stats[i, cv2.CC_STAT_AREA] >= min_px]
    if not pieces and np.any(mask_bool):
        pieces = [comp == order[0]]
    return pieces


def detect_instances(wall_bool, room_bgr, depth_val, cx, cy, f, opts):
    """
    Split a wall mask into one instance per WALL DIRECTION.

    Separation is by the direction each surface faces, not by where a seam
    falls in the image. That is the definition of a different wall: a left wall
    faces one way, a back wall another, and the corner between them is a step
    in the azimuth field wherever it happens to sit. The previous version voted
    a label per image COLUMN, which assumed every boundary was a full-height
    vertical cut -- true of a plain room corner, false of a wall that stops at
    a soffit, a return above a doorway, or any boundary that is not vertical.

    Two walls facing the same way but physically apart -- a wall and the return
    of an alcove behind it -- are then separated spatially, by connected
    components within the direction group.

    Returns a list ordered largest-first, always at least one instance.
    """
    h, w = wall_bool.shape[:2]
    total_px = int(np.count_nonzero(wall_bool))
    min_px = max(
        64,
        int(opts.wall_min_area_fraction * h * w),
        int(MIN_INSTANCE_WALL_SHARE * total_px),
    )

    if total_px < min_px:
        return [SurfaceInstance(index=0, mask=wall_bool.copy(), label="wall",
                                info={"stage": "too-small-to-split"})]

    # ---- 1. Which way does each pixel's surface face? ----
    azimuth = _wall_azimuth(depth_val, cx, cy, f, opts.depth_contrast)
    peaks = _direction_peaks(azimuth, wall_bool)

    if len(peaks) <= 1:
        X, Y, Z, xs, ys = _backproject(wall_bool, depth_val, cx, cy, f, opts.depth_contrast)
        planes, _thr = _sequential_planes(X, Y, Z, xs, ys, opts)
        return [SurfaceInstance(
            index=0, mask=wall_bool.copy(), label="wall",
            plane=planes[0] if planes else None,
            info={"stage": "single-direction", "directions": peaks},
        )]

    labels = _assign_to_directions(azimuth, wall_bool, peaks)

    # ---- 2. Clean each direction region, then split it spatially ----
    radius = max(1, int(round(0.006 * np.hypot(h, w))))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))

    regions = []
    for di, peak in enumerate(peaks):
        region = (labels == di) & wall_bool
        if not np.any(region):
            continue

        # Close pinholes and shave ragged edges, so each wall ends up with one
        # clean boundary instead of a fringe of islands along the corner.
        m8 = cv2.morphologyEx(region.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        m8 = cv2.morphologyEx(m8, cv2.MORPH_OPEN, kernel)

        cleaned = (m8 > 0) & wall_bool
        cleaned = _drop_specks(cleaned, min_px)
        if not np.any(cleaned):
            continue

        # ONE DIRECTION IS NOT NECESSARILY ONE WALL.
        #
        # Two walls can face the same way from different places -- a near wall
        # and the return of an alcove behind it, the two sides of a doorway,
        # parallel walls on opposite sides of a room. Kept as one region they
        # produce a single instance whose convex hull spans the gap between
        # them, and on a real room that hull came out as a TRIANGLE reaching
        # from the left wall across to a wall six metres behind it, with its
        # apex in mid-air in the middle of the floor.
        #
        # So the direction group is broken into its connected pieces here and
        # the pieces that really are one wall are put back together at stage 4
        # by comparing their fitted PLANES -- which is the test that can tell a
        # doorway (same plane, merge) from an alcove (different distance, keep
        # apart). Splitting on connectivity and rejoining on geometry gets both
        # cases right; doing neither gets the alcove wrong, and splitting
        # without rejoining gets the doorway wrong.
        for piece in _connected_pieces(cleaned, min_px):
            regions.append({
                "mask": piece,
                "direction_deg": float(peak),
                "direction_index": di,
            })

    if not regions:
        return [SurfaceInstance(index=0, mask=wall_bool.copy(), label="wall",
                                info={"stage": "direction-split-produced-nothing",
                                      "directions": peaks})]

    # ---- 3. Nothing may go untiled ----
    covered = np.zeros((h, w), dtype=bool)
    for r in regions:
        covered |= r["mask"]
    leftover = wall_bool & ~covered
    if np.any(leftover):
        # Hand each stray pixel to the nearest region, so trimming a boundary
        # never leaves a hole that silently goes unrendered.
        num, comp, _stats, _c = cv2.connectedComponentsWithStats(
            leftover.astype(np.uint8), connectivity=8)
        centres = [np.argwhere(r["mask"]).mean(axis=0) for r in regions]
        for ci in range(1, num):
            piece = comp == ci
            centre = np.argwhere(piece).mean(axis=0)
            nearest = int(np.argmin([np.hypot(*(centre - c)) for c in centres]))
            regions[nearest]["mask"] |= piece

    # ---- 4. One plane per wall: merge regions that lie on the same plane ----
    #
    # This is what finally decides how many walls there are. Direction alone
    # cannot: measured on a real room, one wall's normals spread across two
    # peaks 45 degrees apart while a genuinely different wall sat 45 degrees
    # from one of them. Both pairs look identical to an angle threshold, and
    # the wall was drawn twice as two overlapping quads. Their planes are not
    # ambiguous at all -- same normal and same distance means the same wall,
    # whatever the azimuth histogram happened to do.
    for r in regions:
        X, Y, Z, xs, ys = _backproject(
            r["mask"], depth_val, cx, cy, f, opts.depth_contrast)
        planes, threshold = _sequential_planes(X, Y, Z, xs, ys, opts, max_planes=1)
        r["plane"] = planes[0] if planes else None
        r["threshold"] = threshold
        r["depth_scale"] = float(np.median(np.abs(Z))) if len(Z) else 1.0

    corner_lines = _vertical_architectural_lines(room_bgr)

    merged = []
    for r in sorted(regions, key=lambda x: int(np.count_nonzero(x["mask"])),
                    reverse=True):
        for m in merged:
            if not _same_plane(r["plane"], m["plane"], m["depth_scale"]):
                continue
            # Planes agree -- but a corner line between them overrules that.
            # Two walls meeting at a shallow angle have similar normals and,
            # in a small room, similar distances; the line where they meet is
            # unambiguous where the planes are not. Unless the planes agree
            # so closely that the line can only be a door jamb.
            if (not _same_plane_strongly(r["plane"], m["plane"], m["depth_scale"])
                    and _separated_by_corner_line(r["mask"], m["mask"],
                                                  corner_lines, wall_bool.shape)):
                continue
            m["mask"] = m["mask"] | r["mask"]
            m.setdefault("merged_directions", []).append(r["direction_deg"])
            break
        else:
            merged.append(r)

    # Past the cap the split is fragmenting rather than finding -- but the
    # pixels are still wall and still have to be tiled, so the excess is folded
    # into the nearest kept wall instead of being thrown away.
    if len(merged) > MAX_DETECTED_WALLS:
        kept, excess = merged[:MAX_DETECTED_WALLS], merged[MAX_DETECTED_WALLS:]
        centres = [np.argwhere(k["mask"]).mean(axis=0) for k in kept]
        for e in excess:
            centre = np.argwhere(e["mask"]).mean(axis=0)
            nearest = int(np.argmin([np.hypot(*(centre - c)) for c in centres]))
            kept[nearest]["mask"] |= e["mask"]
        merged = kept

    regions = merged
    instances = []
    for i, r in enumerate(regions):
        threshold = r.get("threshold")
        planes = [r["plane"]] if r.get("plane") else []
        instances.append(SurfaceInstance(
            index=i,
            mask=r["mask"],
            label=f"wall[{i}]",
            plane=planes[0] if planes else None,
            info={
                "stage": "direction-split",
                "direction_deg": r["direction_deg"],
                "direction_index": r["direction_index"],
                "directions_found": [round(p, 1) for p in peaks],
                "merged_directions": [round(x, 1)
                                      for x in r.get("merged_directions", [])],
                "threshold": threshold,
            },
        ))

    return instances