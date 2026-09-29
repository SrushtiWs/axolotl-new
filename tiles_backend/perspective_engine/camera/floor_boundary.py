# perspective_engine/camera/floor_boundary.py
"""
Floor boundary + corner extraction from a segmentation mask.

Stage 3 of the pipeline (Mask2Former -> MiDaS -> HERE -> VP -> focal -> plane
-> metric scale -> tile grid). Runs on the live /render-tile-full path.

Why this sits in front of vanishing-point detection
---------------------------------------------------
VP detection on raw image lines is only as good as the lines it is fed, and on
real room photos most detected lines are not floor geometry:

  * a rug or carpet with a woven/printed pattern produces dozens of strong,
    mutually parallel segments that are NOT aligned with the room, and being
    numerous they can outvote the real grout lines outright;
  * furniture edges sitting on the floor pass a naive "midpoint is on the
    floor mask" test while pointing in completely unrelated directions;
  * floorboard specular highlights fragment into short segments with noisy
    angles.

The floor-to-wall junction does not have any of those problems. It is a real
horizontal line of the room, it lies in the floor plane, and it converges to
the same vanishing points as the tile grid. Extracting it from the mask
silhouette gives VP detection a small set of high-quality lines to weight
above the texture lines, and it is exactly the evidence a carpet cannot fake.

The same polygon also yields the floor corners, and -- as a by-product -- an
honest measure of how much of the floor is actually visible, which feeds the
VP confidence score. A floor whose boundary is 80% image border or occluding
furniture cannot support a confident VP no matter how clean the fit looks.

Nothing here needs the focal length, so it can and must run before calibration.
"""

from typing import Optional

import cv2
import numpy as np

# Boundary edges shorter than this (in pixels) are dropped. Short polygon
# edges are overwhelmingly approxPolyDP staircase artefacts on a diagonal, and
# their angle is quantisation noise rather than room geometry.
MIN_BOUNDARY_EDGE_PX = 25.0

# How close to the image edge a segment must sit to count as frame clipping
# rather than real geometry. The floor almost always runs out of the bottom of
# the frame; that border is a property of the crop, not of the room, and its
# direction means nothing.
BORDER_TOLERANCE_PX = 3.0

# approxPolyDP epsilon as a fraction of contour perimeter. Large enough to
# collapse the ragged per-pixel mask edge into straight runs, small enough to
# keep genuine corners where two walls meet.
POLY_EPSILON_FRAC = 0.008

# Kernel size for the morphological close that bridges gaps punched in the
# mask by furniture legs, so the silhouette stays a single region.
CLOSE_KERNEL_PX = 7


def _as_uint8_mask(floor_mask: np.ndarray) -> np.ndarray:
    if floor_mask.dtype == bool:
        return floor_mask.astype(np.uint8) * 255
    if floor_mask.dtype != np.uint8:
        return (floor_mask > 0).astype(np.uint8) * 255
    return floor_mask


def _touches_border(x: float, y: float, w: int, h: int,
                    tol: float = BORDER_TOLERANCE_PX) -> bool:
    return x <= tol or y <= tol or x >= (w - 1 - tol) or y >= (h - 1 - tol)


def extract_floor_polygon(floor_mask: np.ndarray,
                          epsilon_frac: float = POLY_EPSILON_FRAC):
    """
    Largest floor region, simplified to a polygon.

    Returns (polygon, info). `polygon` is an (N, 2) float array of vertices in
    contour order, or None when there is no usable region. Holes are filled
    first: a coffee table punched out of the middle of the floor would
    otherwise contribute an inner contour whose edges are furniture outlines,
    not room geometry.
    """
    info = {"stage": "start", "area_px": 0, "vertices": 0}

    mask = _as_uint8_mask(floor_mask)
    if mask.ndim != 2:
        info["stage"] = "bad-mask-shape"
        return None, info

    h, w = mask.shape[:2]

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (CLOSE_KERNEL_PX, CLOSE_KERNEL_PX)
    )
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # RETR_EXTERNAL discards inner contours, which is the hole fill.
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        info["stage"] = "no-contour"
        return None, info

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    info["area_px"] = area

    # A sliver of floor cannot constrain anything downstream.
    if area < 0.005 * w * h:
        info["stage"] = "region-too-small"
        return None, info

    perimeter = float(cv2.arcLength(contour, True))
    approx = cv2.approxPolyDP(contour, epsilon_frac * perimeter, True)
    polygon = approx.reshape(-1, 2).astype(np.float64)

    if len(polygon) < 3:
        info["stage"] = "degenerate-polygon"
        return None, info

    info["stage"] = "ok"
    info["vertices"] = int(len(polygon))
    info["perimeter_px"] = perimeter
    info["floor_coverage"] = area / float(w * h)
    return polygon, info


def classify_boundary_edges(polygon: np.ndarray, image_shape,
                            min_length: float = MIN_BOUNDARY_EDGE_PX):
    """
    Split the polygon's edges into the three kinds that matter downstream.

    Returns a dict with:
      wall_lines   (M, 4) floor-to-wall junction segments. These are the VP
                   evidence -- real room geometry, immune to carpet pattern.
      border_lines (M, 4) segments running along the image frame. Recorded
                   only so the occlusion measure can account for them; their
                   direction is an artefact of the crop and must never vote.
      near_lines   (M, 4) real boundary below the floor centroid, i.e. the
                   near-camera edge. Weaker evidence: usually a furniture
                   silhouette rather than a wall junction.

    Classification is by the edge midpoint's height against the floor
    centroid. The floor-wall junction of the far wall is the upper boundary of
    the floor region in essentially every indoor photo taken by a standing
    person, which is the case this pipeline serves.
    """
    h, w = image_shape[:2]
    out = {
        "wall_lines": np.empty((0, 4), dtype=np.float64),
        "border_lines": np.empty((0, 4), dtype=np.float64),
        "near_lines": np.empty((0, 4), dtype=np.float64),
    }
    if polygon is None or len(polygon) < 3:
        return out

    centroid_y = float(np.mean(polygon[:, 1]))

    p1 = polygon
    p2 = np.roll(polygon, -1, axis=0)
    segs = np.column_stack([p1, p2])

    lengths = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    long_enough = lengths >= min_length

    mid_x = 0.5 * (segs[:, 0] + segs[:, 2])
    mid_y = 0.5 * (segs[:, 1] + segs[:, 3])

    # An edge is frame clipping only when BOTH endpoints ride the border --
    # a single endpoint on the border is a real wall line running out of frame.
    e1_border = np.array([_touches_border(x, y, w, h) for x, y in segs[:, 0:2]])
    e2_border = np.array([_touches_border(x, y, w, h) for x, y in segs[:, 2:4]])
    is_border = e1_border & e2_border

    wall = long_enough & ~is_border & (mid_y <= centroid_y)
    near = long_enough & ~is_border & (mid_y > centroid_y)

    out["wall_lines"] = segs[wall]
    out["border_lines"] = segs[is_border]
    out["near_lines"] = segs[near]
    out["centroid_y"] = centroid_y
    return out


def extract_floor_corners(polygon: np.ndarray, image_shape) -> np.ndarray:
    """
    Polygon vertices that are genuine room corners.

    A vertex sitting on the image frame is where the floor leaves the picture,
    not where two walls meet, so it is excluded -- feeding those to any
    downstream 4-point solve would be fitting to the crop.
    """
    if polygon is None or len(polygon) < 3:
        return np.empty((0, 2), dtype=np.float64)

    h, w = image_shape[:2]
    keep = np.array([not _touches_border(x, y, w, h) for x, y in polygon])
    return polygon[keep]


def estimate_vp_from_silhouette(polygon: np.ndarray, image_shape,
                                min_length: float = MIN_BOUNDARY_EDGE_PX):
    """
    Depth vanishing point from the floor SILHOUETTE alone.

    For a room whose floor is roughly rectangular, the left and right floor
    edges are the two side walls' junctions with the floor. Both run in the
    depth direction, so where they intersect IS the depth vanishing point.
    Two lines, one intersection, no line detector involved.

    Why this exists alongside the line-based detector
    -------------------------------------------------
    The line-based VP is better when it works: it uses dozens of grout lines
    rather than two mask edges. But it is fed by texture, and texture fails --
    measured over 65 rooms it produced no usable VP on 17% of them and was
    rejected for low confidence on roughly another 38%. On a bare floor with
    no visible grout, as in a freshly-rendered empty room, there is nothing for
    it to find at all.

    The silhouette always exists whenever the mask does. It is the geometric
    anchor of last resort, and it is derived per room rather than assumed, so
    it never degenerates into a hardcoded screen position.

    Accuracy caveat, stated plainly: this assumes the two side edges are
    parallel in the world. A room with non-parallel walls, a floor cut off by
    furniture, or a curved boundary breaks that assumption and the
    intersection means less. Returns None rather than guessing when the edges
    are too close to parallel in the image for the intersection to be stable.

    NOT WIRED INTO THE RENDERER -- measured and rejected
    ----------------------------------------------------
    Built as an always-available fallback anchor, measured over 65 rooms, and
    left disconnected because it did not survive the measurement:

        available on            46/65 (71%)
        vs accepted line-based VP, grid-rotation angle:
            median 37.2 deg, p25 16.4 deg, worst 79.7 deg
            within 15 deg: 0%

    A 37-degree median disagreement means at least one of the two is badly
    wrong on most images, and a fallback that can rotate the floor by 37
    degrees is worse than having no fallback. The likely cause is edge
    selection: the longest near-vertical mask edges are often furniture
    silhouettes or approxPolyDP staircase artefacts rather than the actual
    wall-floor junction, and this picks them anyway.

    Kept because the idea is sound and the failure is in this implementation
    of it -- fitting the side edges with RANSAC over the raw contour, rather
    than taking approxPolyDP's coarse polygon edges, is the obvious next
    attempt. Do not call it as an anchor until it measures better.

    Returns (vp_x, vp_y, info).
    """
    info = {"stage": "start"}
    if polygon is None or len(polygon) < 3:
        info["stage"] = "no-polygon"
        return None, None, info

    h, w = image_shape[:2]
    cx_poly = float(np.mean(polygon[:, 0]))

    p1 = polygon
    p2 = np.roll(polygon, -1, axis=0)
    segs = np.column_stack([p1, p2])

    dx = segs[:, 2] - segs[:, 0]
    dy = segs[:, 3] - segs[:, 1]
    lengths = np.hypot(dx, dy)

    e1_border = np.array([_touches_border(x, y, w, h) for x, y in segs[:, 0:2]])
    e2_border = np.array([_touches_border(x, y, w, h) for x, y in segs[:, 2:4]])
    is_border = e1_border & e2_border

    # Side edges run INTO the room, so they are the more vertical ones. A
    # near-horizontal edge is the far wall or the frame, and carries no depth
    # information.
    verticality = np.abs(dy) / (lengths + 1e-9)
    usable = (lengths >= min_length) & ~is_border & (verticality > 0.5)
    if usable.sum() < 2:
        info["stage"] = "no-side-edges"
        return None, None, info

    mid_x = 0.5 * (segs[:, 0] + segs[:, 2])
    left = usable & (mid_x < cx_poly)
    right = usable & (mid_x >= cx_poly)
    if not left.any() or not right.any():
        info["stage"] = "edges-on-one-side-only"
        return None, None, info

    # Longest edge on each side: the longest run of the floor-wall junction is
    # the least likely to be a furniture silhouette or an approxPolyDP artefact.
    li = int(np.argmax(np.where(left, lengths, -1)))
    ri = int(np.argmax(np.where(right, lengths, -1)))

    def _line(s):
        a = np.array([s[0], s[1], 1.0])
        b = np.array([s[2], s[3], 1.0])
        return np.cross(a, b)

    vp = np.cross(_line(segs[li]), _line(segs[ri]))
    if abs(vp[2]) < 1e-6:
        info["stage"] = "edges-parallel"
        return None, None, info

    vp_x, vp_y = float(vp[0] / vp[2]), float(vp[1] / vp[2])
    if not (np.isfinite(vp_x) and np.isfinite(vp_y)):
        info["stage"] = "non-finite"
        return None, None, info

    # A depth VP that lands far outside any plausible frame means the two
    # edges were nearly parallel and the intersection is noise.
    if abs(vp_x) > 20.0 * w or abs(vp_y) > 20.0 * h:
        info["stage"] = "vp-too-far"
        return None, None, info

    info["stage"] = "ok"
    info["left_edge"] = segs[li].tolist()
    info["right_edge"] = segs[ri].tolist()
    info["left_len"] = float(lengths[li])
    info["right_len"] = float(lengths[ri])
    return vp_x, vp_y, info


def extract_floor_boundary(floor_mask: np.ndarray,
                           image_shape: Optional[tuple] = None):
    """
    Full boundary stage: polygon, corners, classified edges, visibility.

    Returns (result, info). `result` is None when the mask has no usable floor
    region, in which case the caller carries on without boundary evidence
    rather than failing -- every consumer of this treats it as optional.

    `visible_fraction` is the share of the boundary that is real room geometry
    rather than image border. It is the honest answer to "how much of this
    floor did the photo actually capture", and the VP confidence score uses it:
    a floor cropped by the frame on three sides gives the VP almost nothing to
    work with even when the fit residuals look excellent.
    """
    shape = image_shape if image_shape is not None else floor_mask.shape

    polygon, info = extract_floor_polygon(floor_mask)
    if polygon is None:
        return None, info

    edges = classify_boundary_edges(polygon, shape)
    corners = extract_floor_corners(polygon, shape)

    def _total_len(lines):
        if len(lines) == 0:
            return 0.0
        return float(np.sum(np.hypot(lines[:, 2] - lines[:, 0],
                                     lines[:, 3] - lines[:, 1])))

    wall_len = _total_len(edges["wall_lines"])
    near_len = _total_len(edges["near_lines"])
    border_len = _total_len(edges["border_lines"])
    real_len = wall_len + near_len
    total_len = real_len + border_len

    result = {
        "polygon": polygon,
        "corners": corners,
        "wall_lines": edges["wall_lines"],
        "near_lines": edges["near_lines"],
        "border_lines": edges["border_lines"],
        "visible_fraction": (real_len / total_len) if total_len > 1e-6 else 0.0,
        "wall_line_fraction": (wall_len / real_len) if real_len > 1e-6 else 0.0,
        "floor_coverage": info.get("floor_coverage", 0.0),
    }

    info["stage"] = "ok"
    info["corners"] = int(len(corners))
    info["wall_lines"] = int(len(edges["wall_lines"]))
    info["visible_fraction"] = result["visible_fraction"]
    return result, info
