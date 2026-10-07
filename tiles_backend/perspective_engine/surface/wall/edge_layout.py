"""
The room's walls from the two edges a wall shows: where it stands on the floor
and where it meets the ceiling.

junction_layout.py splits the walls from the floor junction alone, and only
when that junction is visible under most of the wall. In a furnished room the
sofa, the bed and the cabinets hide it, but the ceiling line usually stays in
view. Both edges are horizontal lines in the wall's own plane, so both bend
exactly where the room turns a corner:

  * each unbroken stretch of either edge is simplified into straight runs;
  * a bend BETWEEN runs of one stretch is a corner that was seen;
  * nothing is bridged across a stretch that is hidden: a corner that neither
    edge shows is not placed (the walls either side stay one wall rather than
    being split at a guessed column);
  * a wall narrower than a jog (junction_layout.MIN_RUN_FRACTION) is merged
    into its wider neighbour, so a split always has room-scale evidence.

Edge pixels next to a removed object are not evidence: the clean-room fill
painted them (as junction_layout.junction_points already does for the floor).

Each wall takes its direction from its own longest run -- the floor junction
first, the ceiling line otherwise -- through junction_layout._direction: a
horizontal line in the wall's plane meets the horizon at the wall's along-wall
vanishing point either way.
"""

from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np

from . import junction_layout as jl

#: The layout is used only when the floor junction and/or the ceiling line are
#: seen under this share of the wall's columns.
MIN_EDGE_SHARE = 0.25

#: Seen corners from the two edges this close (share of the image width) are
#: one corner: a plumb corner edge meets floor and ceiling in nearly one column.
CORNER_MERGE_FRACTION = 0.02


def ceiling_points(wall: np.ndarray, objects: Optional[np.ndarray] = None) -> np.ndarray:
    """
    (N, 2) float (x, y): per column, the wall's highest pixel where the pixel
    above is not wall (the ceiling, or anything the wall stops at), never the
    image's top row, and not next to a removed object.
    """
    h, w = wall.shape
    cols = np.where(wall.any(axis=0))[0]
    if not len(cols):
        return np.empty((0, 2))
    top = np.argmax(wall[:, cols], axis=0)
    keep = (top > 2) & ~wall[np.clip(top - 1, 0, h - 1), cols]
    if objects is not None:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * jl.JUNCTION_REACH_PX + 1,) * 2)
        grown = cv2.dilate(objects.astype(np.uint8), k).astype(bool)
        keep &= ~grown[np.clip(top - 1, 0, h - 1), cols] & ~grown[top, cols]
    return np.column_stack([cols[keep], top[keep]]).astype(np.float64)


def _stretch_runs(pts: np.ndarray, w: int, h: int):
    """
    Straight runs per unbroken stretch: [[(x0, y0, x1, y1), ...] per stretch].
    Within a stretch, jogs join their longer neighbour and collinear runs join;
    stretches are never joined to each other.
    """
    if len(pts) < 2:
        return []
    pts = pts[np.argsort(pts[:, 0])]
    eps = jl.SIMPLIFY_FRACTION * float(np.hypot(h, w))
    breaks = np.where(np.diff(pts[:, 0]) > max(8.0, jl.GAP_FRACTION * w))[0] + 1
    min_run = jl.MIN_RUN_FRACTION * w
    stretches = []
    for chain in np.split(pts, breaks):
        if len(chain) < 2 or chain[-1, 0] - chain[0, 0] < min_run:
            continue
        poly = cv2.approxPolyDP(chain.astype(np.float32).reshape(-1, 1, 2), eps, False)[:, 0, :].astype(float)
        segs = [list(poly[i]) + list(poly[i + 1]) for i in range(len(poly) - 1)]
        segs = [s for s in segs if s[2] > s[0]]
        changed = True
        while changed and len(segs) > 1:                     # jogs -> longer neighbour
            changed = False
            for i, s in enumerate(segs):
                if s[2] - s[0] >= min_run:
                    continue
                if i == 0:
                    j = 1
                elif i == len(segs) - 1:
                    j = i - 1
                else:
                    j = i - 1 if (segs[i - 1][2] - segs[i - 1][0]) >= (segs[i + 1][2] - segs[i + 1][0]) else i + 1
                a, b = (segs[j], s) if j < i else (s, segs[j])
                segs[min(i, j)] = [a[0], a[1], b[2], b[3]]
                del segs[max(i, j)]
                changed = True
                break
        runs = []
        for s in segs:                                        # collinear -> one run
            if runs and abs(jl._angle_deg(runs[-1]) - jl._angle_deg(s)) < jl.COLLINEAR_DEG:
                runs[-1] = [runs[-1][0], runs[-1][1], s[2], s[3]]
            else:
                runs.append(list(s))
        stretches.append(runs)
    return stretches


def _seen_corners(stretches) -> list[float]:
    """Columns where one run of a stretch ends and the next begins."""
    return [(runs[i][2] + runs[i + 1][0]) / 2.0 for runs in stretches for i in range(len(runs) - 1)]


def layout(floor: np.ndarray, wall: np.ndarray, horizon_y: Optional[float], f: float, cx: float, cy: float,
           depth_vp=None, objects: Optional[np.ndarray] = None, room_bgr=None):
    """
    [(mask, direction or None)] -- one per wall, left to right -- or (None, info).
    Every WALL_MASK pixel goes to exactly one wall, by column, between seen
    corners. `direction` is None for a wall neither edge shows a run of.
    """
    h, w = wall.shape
    wall_cols = np.where(wall.any(axis=0))[0]
    if not len(wall_cols):
        return None, {"stage": "no-wall"}
    J = jl.junction_points(floor, wall, objects)
    T = ceiling_points(wall, objects)
    seen_cols = np.union1d(J[:, 0], T[:, 0]) if len(J) or len(T) else np.empty(0)
    share = len(seen_cols) / len(wall_cols)
    info = {"junction_share": round(len(J) / len(wall_cols), 3), "ceiling_share": round(len(T) / len(wall_cols), 3),
            "edge_share": round(share, 3)}
    if share < MIN_EDGE_SHARE:
        return None, {"stage": "edges-hidden", **info}

    j_runs, t_runs = _stretch_runs(J, w, h), _stretch_runs(T, w, h)
    seen = sorted([(c, "floor-junction") for c in _seen_corners(j_runs)] +
                  [(c, "ceiling-line") for c in _seen_corners(t_runs)])
    merge = CORNER_MERGE_FRACTION * w
    corners: list[list] = []                                  # [x, {sources}]
    for x, src in seen:
        if corners and x - corners[-1][0] <= merge:
            corners[-1][0] = (corners[-1][0] + x) / 2.0
            corners[-1][1].add(src)
        else:
            corners.append([x, {src}])

    # A wall narrower than a jog is a sliver: drop the corner on its narrow side
    # so it joins its wider neighbour.
    x_lo, x_hi = float(wall_cols.min()), float(wall_cols.max())
    min_width = jl.MIN_RUN_FRACTION * w
    changed = True
    while changed and corners:
        changed = False
        edges = [x_lo - 1.0] + [c[0] for c in corners] + [x_hi]
        for i in range(len(edges) - 1):
            if edges[i + 1] - edges[i] >= min_width:
                continue
            if i == 0:
                k = 0
            elif i == len(edges) - 2:
                k = len(corners) - 1
            else:
                left, right = edges[i] - edges[i - 1], edges[i + 2] - edges[i + 1]
                k = i if right >= left else i - 1          # drop the corner toward the wider neighbour
            del corners[k]
            changed = True
            break

    cols = np.arange(w)[None, :]

    # One wall is one plane. A cut that is not a confirmed corner (corner_cut:
    # both edges bend there, or a bend plus a long vertical photo edge) is
    # removed when the pieces on its two sides lie on ONE plane -- their floor
    # junction (else ceiling line) points fall on one straight image line
    # (corner_cut._coplanar). Two walls meeting at a corner are never on one
    # line, so they stay apart. Decided on the masks only, before any
    # direction is fitted. Without the photo the old behaviour is kept.
    one_plane_cuts = []
    if room_bgr is not None and corners:
        from . import corner_cut
        lines, _, _, _ = corner_cut.corners(floor, wall, objects, room_bgr)
        confirmed_x = [(t[0] + b[0]) / 2.0 for t, b, _ in lines]
        diag = float(np.hypot(h, w))
        changed = True
        while changed:
            changed = False
            edges = [-1.0] + [c[0] for c in corners] + [float(w)]
            for k, c in enumerate(corners):
                if any(abs(c[0] - x) <= corner_cut.MATCH * w for x in confirmed_x):
                    continue
                left = wall & np.broadcast_to((cols > edges[k]) & (cols <= edges[k + 1]), wall.shape)
                right = wall & np.broadcast_to((cols > edges[k + 1]) & (cols <= edges[k + 2]), wall.shape)
                evidence = corner_cut._coplanar(left, right, floor, objects, w, diag)
                if evidence:
                    one_plane_cuts.append({"x": round(c[0], 1), "evidence": evidence})
                    del corners[k]
                    changed = True
                    break

    edges = [-1.0] + [c[0] for c in corners] + [float(w)]
    all_runs = [(s, "floor-junction") for r in j_runs for s in r] + [(s, "ceiling-line") for r in t_runs for s in r]
    walls = []
    for i in range(len(edges) - 1):
        in_span = (cols > edges[i]) & (cols <= edges[i + 1])
        mask = wall & np.broadcast_to(in_span, wall.shape)
        if not mask.any():
            continue
        direction = None
        if horizon_y is not None:
            # This wall's own longest run inside its span: junction first.
            best = None
            for s, src in all_runs:
                a, b = max(s[0], edges[i]), min(s[2], edges[i + 1])
                if b - a < min_width:
                    continue
                t0, t1 = (a - s[0]) / (s[2] - s[0]), (b - s[0]) / (s[2] - s[0])
                clip = [a, s[1] + t0 * (s[3] - s[1]), b, s[1] + t1 * (s[3] - s[1])]
                key = (src == "floor-junction", b - a)
                if best is None or key > best[0]:
                    best = (key, clip, src)
            if best is not None:
                direction = jl._direction(best[1], horizon_y, f, cx, cy, depth_vp)
                if direction is not None:
                    direction["source"] = best[2]
                    direction["evidence"] = {**direction.get("evidence", {}), "run_source": best[2]}
        walls.append((mask, direction))

    return walls, {"stage": "ok", **info, "one_plane_cuts_removed": one_plane_cuts,
                   "corners": [{"x": round(c[0], 1), "seen_on": sorted(c[1])} for c in corners],
                   "junction_runs": [[round(float(v), 1) for v in s] for r in j_runs for s in r],
                   "ceiling_runs": [[round(float(v), 1) for v in s] for r in t_runs for s in r]}
