"""
One pixel = one plane: the settled wall split, cut at the room's real corners.

The split before this step can let one piece run past a corner onto the next
wall (that strip then gets the wrong wall's tile direction), and can leave one
plane in several pieces. Here:

  1. Corners are found from the WHOLE wall mask, never from the split: the
     floor-wall junction and the wall-ceiling line are each cut into straight
     runs (tolerances relative to the image); a bend is where a line turns.
     A corner is CONFIRMED when a bend has a second, independent witness:
       - the other line bends at the same x (the near-vertical line joining
         the two bends is the corner), or
       - a long near-vertical edge in the photo starts or ends at the bend
         (that edge, extended, is the corner).
     A bend with no witness is reported as uncertain and never cut along
     (not 95% sure). Bends at the image border are not corners.
  2. The confirmed corner lines divide the frame into plane slots. Every wall
     pixel lies in exactly one slot.
  3. Each piece keeps the pixels in its own main slot. Its pixels in other
     slots (the spill) go to the wall of that slot they touch.
  4. Pieces are one plane -- merged into one wall, one dot -- only when they
     are in the same slot, face the same way, have normals within MERGE_DEG,
     touch, and no bend lies between them (left and right walls share a
     normal but face opposite ways, so they are never merged).
  5. Spill with no wall of its own slot becomes a wall when it is at least
     MIN_WALL_SHARE of the image (its direction is then fitted from its own
     lines); smaller, it is dropped and counted.

Wall ids keep the split's order. Directions are refitted per wall
afterwards by the caller (edge_direction.fit). `ENABLED = False` restores the
previous split exactly (for before/after measurement).
"""

from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np

ENABLED = True

FIT_TOL = 0.006           # x diagonal: max deviation of a straight run from its line
GAP = 0.02                # x width: a hole this wide in a line ends a run
MIN_RUN = 0.04            # x width: shorter runs are not a wall's edge
BEND_DEG = 6.0            # runs meeting at more than this = a bend
MATCH = 0.03              # x width: floor and ceiling bends this close = one corner
MAX_LEAN_DEG = 12.0       # a corner line leaning more than this from vertical is not a corner
MERGE_DEG = 5.0           # pieces of one slot this close in direction = one plane
MIN_WALL_SHARE = 0.03     # x image area: an orphan region smaller than this is not a wall
BORDER = 0.02             # x width: bends this close to the image edge are the frame, not a corner
EDGE_MIN_LEN = 0.15       # x height: a corner's photo edge is at least this long
EDGE_MATCH = 0.01         # x width: the edge passes this close to the bend
EDGE_END = 0.20           # x height: ...and starts / ends this close to the bend's row (objects can break it)
TOUCH = 0.01              # x diagonal: pieces this close touch
STRAIGHT_SHARE = 0.5      # a bend-free run across this share of a slot makes it one plane
SEAM_MIN_LEN = 0.10       # x height: a seam shorter than this is too short to judge its lean
SEAM_VERTICAL_TOL_DEG = 3.0   # a seam this close to the photo's own vertical could be a corner
NEAR = 0.10               # x width: photo edges this close give the local vertical


def _split(pts: np.ndarray, tol: float) -> list[np.ndarray]:
    if len(pts) < 3:
        return [pts]
    a, b = pts[0], pts[-1]
    d = b - a
    n = np.array([-d[1], d[0]]) / (np.hypot(*d) or 1.0)
    dist = np.abs((pts - a) @ n)
    i = int(np.argmax(dist))
    if dist[i] <= tol:
        return [pts]
    return _split(pts[: i + 1], tol) + _split(pts[i:], tol)


def _fit(pts: np.ndarray) -> dict:
    vx, vy, x0, y0 = cv2.fitLine(pts.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    return {"x0": float(pts[0, 0]), "x1": float(pts[-1, 0]), "angle": math.degrees(math.atan2(vy, vx)),
            "line": (float(vx), float(vy), float(x0), float(y0)), "pts": pts}


def runs(points: np.ndarray, width: int, diag: float) -> list[dict]:
    """Straight runs of an edge's (x, y) points, x-sorted, never bridging a gap."""
    if len(points) < 2:
        return []
    pts = points[np.argsort(points[:, 0])]
    breaks = np.where(np.diff(pts[:, 0]) > GAP * width)[0]
    out = []
    for part in np.split(pts, breaks + 1):
        for seg in _split(part, FIT_TOL * diag):
            if len(seg) >= 2 and seg[-1, 0] - seg[0, 0] >= MIN_RUN * width:
                out.append(_fit(seg))
    merged = []
    for r in out:
        if merged and abs(merged[-1]["angle"] - r["angle"]) < BEND_DEG / 2 and r["x0"] - merged[-1]["x1"] <= GAP * width:
            merged[-1] = _fit(np.vstack([merged[-1]["pts"], r["pts"]]))
        else:
            merged.append(r)
    return merged


def bends(edge_runs: list[dict], width: int) -> list[dict]:
    """Where two consecutive touching runs meet at more than BEND_DEG: (x, y, turn)."""
    out = []
    for a, b in zip(edge_runs, edge_runs[1:]):
        if b["x0"] - a["x1"] > GAP * width:
            continue
        turn = abs(((a["angle"] - b["angle"]) + 90) % 180 - 90)
        if turn < BEND_DEG:
            continue
        # where the two runs' lines cross (falls back to the gap's middle)
        (ax, ay, ax0, ay0), (bx, by, bx0, by0) = a["line"], b["line"]
        den = ax * by - ay * bx
        if abs(den) > 1e-9:
            t = ((bx0 - ax0) * by - (by0 - ay0) * bx) / den
            x, y = ax0 + t * ax, ay0 + t * ay
        else:
            x = (a["x1"] + b["x0"]) / 2.0
            y = ay0 + (x - ax0) * ay / (ax or 1e-9)
        if not (a["x0"] - GAP * width <= x <= b["x1"] + GAP * width):
            x = (a["x1"] + b["x0"]) / 2.0
            y = ay0 + (x - ax0) * ay / (ax or 1e-9)
        out.append({"x": float(x), "y": float(y), "turn_deg": round(float(turn), 1)})
    return out


def gap_ends(edge_runs: list[dict], width: int) -> list[dict]:
    """
    An edge that stops at one angle and, after a hidden stretch, resumes at a
    clearly different one turned a corner somewhere in between: the two ends
    facing the gap are corner candidates (each still needs a witness).
    """
    out = []
    for a, b in zip(edge_runs, edge_runs[1:]):
        if b["x0"] - a["x1"] <= GAP * width:
            continue
        turn = abs(((a["angle"] - b["angle"]) + 90) % 180 - 90)
        if turn < BEND_DEG:
            continue
        # A run seen up to its end stops at or before the hidden corner, so
        # the corner lies on the gap side of the end ("into": +1 right, -1 left).
        for run, x, into in ((a, a["x1"], 1), (b, b["x0"], -1)):
            vx, vy, x0, y0 = run["line"]
            out.append({"x": float(x), "y": float(y0 + (x - x0) * vy / (vx or 1e-9)), "turn_deg": round(float(turn), 1),
                        "gap": [round(a["x1"], 1), round(b["x0"], 1)], "into": into})
    return out


def _vertical_edges(room_bgr, h: int, w: int, untrusted: Optional[np.ndarray] = None) -> list:
    """
    Long near-vertical photo edges as (top_xy, bottom_xy). An edge lying mostly
    in `untrusted` (the inpainted area of removed objects, where the fill only
    guesses texture) is not evidence and is left out.
    """
    if room_bgr is None:
        return []
    from ...camera.vp_from_mask import detect_lines_lsd

    gray = cv2.cvtColor(room_bgr, cv2.COLOR_BGR2GRAY) if room_bgr.ndim == 3 else room_bgr
    out = []
    for x1, y1, x2, y2 in np.asarray(detect_lines_lsd(gray) if gray is not None else [], float).reshape(-1, 4):
        if math.hypot(x2 - x1, y2 - y1) < EDGE_MIN_LEN * h:
            continue
        if math.degrees(math.atan2(abs(x2 - x1), abs(y2 - y1))) > MAX_LEAN_DEG:
            continue
        top, bottom = ((x1, y1), (x2, y2)) if y1 <= y2 else ((x2, y2), (x1, y1))
        if untrusted is not None:
            t = np.linspace(0.0, 1.0, 17)
            xs = np.clip(np.round(top[0] + t * (bottom[0] - top[0])).astype(int), 0, w - 1)
            ys = np.clip(np.round(top[1] + t * (bottom[1] - top[1])).astype(int), 0, h - 1)
            if untrusted[ys, xs].mean() > 0.5:
                continue
        out.append((top, bottom))
    return out


def _edge_at(bend: dict, edges: list, h: int, w: int, at: str):
    """
    A photo edge starting (ceiling) or ending (floor) at this bend, or None.
    For a run end facing a hidden gap (`into`), the edge may also lie up to
    GAP of the width further into the gap -- the corner is past the last seen point.
    """
    into = bend.get("into", 0)
    best = None
    for top, bottom in edges:
        end = top if at == "ceiling" else bottom
        if abs(end[1] - bend["y"]) > EDGE_END * h:
            continue
        dy = (bottom[1] - top[1]) or 1e-9
        x_at = top[0] + (bend["y"] - top[1]) * (bottom[0] - top[0]) / dy
        offset = (x_at - bend["x"]) * into
        miss = abs(x_at - bend["x"])
        ok = miss <= EDGE_MATCH * w or (into and 0 <= offset <= GAP * w)
        if ok and (best is None or miss < best[0]):
            best = (miss, top, bottom)
    return best


def corners(floor: np.ndarray, wall: np.ndarray, objects: Optional[np.ndarray], room_bgr=None) -> tuple[list, list, list]:
    """
    (confirmed corner lines [(top_xy, bottom_xy, evidence)], uncertain bends [evidence],
    the straight runs, the trusted long vertical photo edges).
    """
    from . import edge_layout
    from . import junction_layout as jl

    h, w = wall.shape
    diag = math.hypot(h, w)
    inside = lambda b: BORDER * w <= b["x"] <= (1 - BORDER) * w  # noqa: E731
    floor_runs = runs(jl.junction_points(floor, wall, objects), w, diag)
    ceiling_runs = runs(edge_layout.ceiling_points(wall, objects), w, diag)
    fb = [b for b in bends(floor_runs, w) if inside(b)]
    cb = [b for b in bends(ceiling_runs, w) if inside(b)]
    untrusted = None
    if objects is not None:
        r = max(2, int(round(TOUCH * diag)))
        untrusted = cv2.dilate(objects.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)) > 0
    edges = _vertical_edges(room_bgr, h, w, untrusted)
    confirmed, uncertain, used = [], [], set()

    def add(top, bottom, ev):
        x_mid = (top[0] + bottom[0]) / 2
        if any(abs(x_mid - (t[0] + b[0]) / 2) <= MATCH * w for t, b, _ in confirmed):
            return                                              # the same corner, already found
        confirmed.append((tuple(map(float, top)), tuple(map(float, bottom)), ev))

    for f in fb:
        near = [c for c in cb if abs(c["x"] - f["x"]) <= MATCH * w and id(c) not in used]
        c = min(near, key=lambda c: abs(c["x"] - f["x"])) if near else None
        if c is not None and f["y"] > c["y"]:
            lean = math.degrees(math.atan2(abs(f["x"] - c["x"]), f["y"] - c["y"]))
            if lean <= MAX_LEAN_DEG:
                used.add(id(c))
                add((c["x"], c["y"]), (f["x"], f["y"]),
                    {"x": round(f["x"], 1), "witness": "floor + ceiling bend", "turn_deg": [f["turn_deg"], c["turn_deg"]],
                     "lean_deg": round(lean, 1)})
                continue
        edge = _edge_at(f, edges, h, w, "floor")
        if edge is not None:
            add(edge[1], edge[2], {"x": round(f["x"], 1), "witness": "floor bend + photo edge",
                                   "turn_deg": [f["turn_deg"]], "edge_miss_px": round(edge[0], 1)})
            continue
        uncertain.append({"x": round(f["x"], 1), "seen_in": "floor line only", "turn_deg": f["turn_deg"]})
    for c in cb:
        if id(c) in used:
            continue
        edge = _edge_at(c, edges, h, w, "ceiling")
        if edge is not None:
            add(edge[1], edge[2], {"x": round(c["x"], 1), "witness": "ceiling bend + photo edge",
                                   "turn_deg": [c["turn_deg"]], "edge_miss_px": round(edge[0], 1)})
            continue
        uncertain.append({"x": round(c["x"], 1), "seen_in": "ceiling line only", "turn_deg": c["turn_deg"]})
    # Corners hidden in a gap of an edge (curtain, window, furniture): a run end
    # facing the gap counts only with a photo edge starting / ending there.
    for at, edge_runs in (("floor", floor_runs), ("ceiling", ceiling_runs)):
        for g in gap_ends(edge_runs, w):
            if not inside(g):
                continue
            edge = _edge_at(g, edges, h, w, at)
            if edge is not None:
                add(edge[1], edge[2], {"x": round(g["x"], 1), "witness": f"{at} line turns across a hidden gap + photo edge",
                                       "turn_deg": [g["turn_deg"]], "gap": g["gap"], "edge_miss_px": round(edge[0], 1)})
    confirmed.sort(key=lambda item: (item[0][0] + item[1][0]) / 2)
    return confirmed, sorted(uncertain, key=lambda u: u["x"]), floor_runs + ceiling_runs, edges


def slots(shape, lines) -> np.ndarray:
    """Per pixel: how many corner lines lie to its left (0 .. len(lines))."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    lab = np.zeros(shape, np.int32)
    for top, bottom, _ in lines:
        dy = (bottom[1] - top[1]) or 1e-9
        x_line = top[0] + (yy - top[1]) * (bottom[0] - top[0]) / dy
        lab += (xx > x_line).astype(np.int32)
    return lab


def _normal(direction):
    n = (direction or {}).get("normal")
    if n is None:
        return None
    n = np.asarray(n, float)
    return n / (np.linalg.norm(n) or 1.0)


def _same_plane(a, b) -> bool:
    if a["normal"] is None or b["normal"] is None:
        return False
    if (a["direction"] or {}).get("faces") != (b["direction"] or {}).get("faces"):
        return False                       # parallel opposite walls share a normal, not a plane
    return math.degrees(math.acos(min(1.0, abs(float(a["normal"] @ b["normal"]))))) <= MERGE_DEG


def _touch(a: np.ndarray, b: np.ndarray, radius: int) -> bool:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
    return bool((cv2.dilate(a.astype(np.uint8), k).astype(bool) & b).any())


def _signed_lean(top, bottom) -> float:
    """Degrees from image-vertical, signed (positive leans right going down)."""
    return math.degrees(math.atan2(bottom[0] - top[0], bottom[1] - top[1]))


def _local_vertical(x: float, edges: list, w: int):
    """The photo's own vertical near column x: median lean of the long vertical edges within NEAR of it (all, if none)."""
    if not edges:
        return None
    near = [e for e in edges if abs((e[0][0] + e[1][0]) / 2 - x) <= NEAR * w] or edges
    return float(np.median([_signed_lean(t, b) for t, b in near]))


def _seam_lean(a: np.ndarray, b: np.ndarray, radius: int, min_len: float):
    """
    (signed degrees from image-vertical, mean x) of the seam where two pieces
    touch (principal axis of the touching pixels), or None when they share too
    little seam to tell.
    """
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
    seam = cv2.dilate(a.astype(np.uint8), k).astype(bool) & b
    ys, xs = np.nonzero(seam)
    if len(xs) < 2:
        return None
    pts = np.column_stack([xs, ys]).astype(np.float64)
    pts -= pts.mean(axis=0)
    _, sv, vt = np.linalg.svd(pts, full_matrices=False)
    span = 2.0 * sv[0] / math.sqrt(len(pts)) * math.sqrt(3.0)    # ~ length of a uniform segment
    if span < min_len:
        return None
    dx, dy = vt[0]
    if dy < 0:
        dx, dy = -dx, -dy
    return math.degrees(math.atan2(dx, dy)), float(xs.mean())


def _bend_between(a: np.ndarray, b: np.ndarray, bend_xs: list) -> bool:
    xa, xb = float(np.nonzero(a)[1].mean()), float(np.nonzero(b)[1].mean())
    lo, hi = min(xa, xb), max(xa, xb)
    return any(lo < x < hi for x in bend_xs)


def cut(refined: list, floor: np.ndarray, wall: np.ndarray, objects: Optional[np.ndarray], room_bgr=None):
    """
    refined: [(index, mask, direction)] -> (new refined, report). Every pixel of
    the input masks ends in at most one output mask.
    """
    before = len(refined)
    if not ENABLED or not refined:
        return refined, {"enabled": ENABLED, "walls_before": before, "walls_after": before}
    h, w = wall.shape
    lines, uncertain, edge_runs, edges = corners(floor, wall, objects, room_bgr)
    lab = slots((h, w), lines)
    nslots = len(lines) + 1
    radius = max(2, int(round(TOUCH * math.hypot(h, w))))
    bend_xs = [u["x"] for u in uncertain]

    # 1. every piece keeps its pixels in its own main slot; the rest is spill
    groups, spill = [], [np.zeros((h, w), bool) for _ in range(nslots)]
    for index, mask, direction in refined:
        counts = np.bincount(lab[mask], minlength=nslots)
        home = int(np.argmax(counts))
        for s in range(nslots):
            if s != home:
                spill[s] |= mask & (lab == s)
        groups.append({"slot": home, "mask": mask & (lab == home), "direction": direction,
                       "normal": _normal(direction), "sources": [int(index)]})

    # 2. fragments of one plane: same slot, same facing, same normal, touching,
    #    and no bend of either line between them
    merged = True
    while merged:
        merged = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                a, b = groups[i], groups[j]
                if (a["slot"] == b["slot"] and _same_plane(a, b) and _touch(a["mask"], b["mask"], radius)
                        and not _bend_between(a["mask"], b["mask"], bend_xs)):
                    keep, gone = (i, j) if a["mask"].sum() >= b["mask"].sum() else (j, i)
                    groups[keep]["mask"] = groups[keep]["mask"] | groups[gone]["mask"]
                    groups[keep]["sources"] += groups[gone]["sources"]
                    del groups[gone]
                    merged = True
                    break
            if merged:
                break

    # 2b. one straight edge across a slot = one plane: when a floor or ceiling
    #     run lies inside one slot, spans at least STRAIGHT_SHARE of the slot's
    #     width at that height and no bend is seen inside the slot, every piece
    #     of the slot is that plane (pieces split off along a line that is no
    #     corner, e.g. a slanted cut on a plain wall).
    # 2b and 2c only where the room's corners are known: with no confirmed
    # corner, nothing says where one plane ends, and a merge would be a guess.
    straight = []
    for s in (range(nslots) if lines else []):
        members = [g for g in groups if g["slot"] == s]
        if len(members) < 2:
            continue
        if any(lab[h // 2, int(min(max(u["x"], 0), w - 1))] == s for u in uncertain):
            continue
        for r in edge_runs:
            p = r["pts"].astype(int)
            on = lab[np.clip(p[:, 1], 0, h - 1), np.clip(p[:, 0], 0, w - 1)] == s
            if on.mean() < 0.9:
                continue
            row = int(np.clip(np.median(p[:, 1]), 0, h - 1))
            width_here = int((lab[row] == s).sum())
            if width_here and (r["x1"] - r["x0"]) >= STRAIGHT_SHARE * width_here:
                # only the pieces the run lies over: most of their columns inside its x-range
                under = []
                for g in members:
                    cols = np.nonzero(g["mask"].any(axis=0))[0]
                    if len(cols) and ((cols >= r["x0"]) & (cols <= r["x1"])).mean() >= STRAIGHT_SHARE:
                        under.append(g)
                if len(under) < 2:
                    continue
                members = under
                keep = max(members, key=lambda g: (g["direction"] is not None, int(g["mask"].sum())))
                for g in members:
                    if g is not keep:
                        keep["mask"] = keep["mask"] | g["mask"]
                        keep["sources"] += g["sources"]
                groups = [g for g in groups if g is keep or not any(g is m for m in members)]
                straight.append({"slot": s, "run": [round(r["x0"], 1), round(r["x1"], 1)], "angle_deg": round(r["angle"], 1),
                                 "slot_width_px": width_here, "merged": list(keep["sources"])})
                break

    # 2c. a seam that does not follow the room's verticals is no corner (wall
    #     corners are vertical lines, as the photo's own long vertical edges
    #     show): touching pieces of one slot meeting along such a seam, with no
    #     bend between them, are one plane.
    seams = []
    merged = bool(lines)
    while merged:
        merged = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                a, b = groups[i], groups[j]
                if a["slot"] != b["slot"] or _bend_between(a["mask"], b["mask"], bend_xs):
                    continue
                seam = _seam_lean(a["mask"], b["mask"], radius, SEAM_MIN_LEN * h)
                if seam is None:
                    continue
                vertical = _local_vertical(seam[1], edges, w)
                off = abs(seam[0] - (vertical if vertical is not None else 0.0))
                if off <= SEAM_VERTICAL_TOL_DEG:
                    continue                      # follows the room's verticals: could be a corner
                lean = off
                keep, gone = (i, j) if (a["direction"] is not None, a["mask"].sum()) >= (b["direction"] is not None, b["mask"].sum()) else (j, i)
                seams.append({"merged": [list(groups[keep]["sources"]), list(groups[gone]["sources"])],
                              "seam_off_vertical_deg": round(lean, 1),
                              "local_vertical_deg": None if vertical is None else round(vertical, 1)})
                groups[keep]["mask"] = groups[keep]["mask"] | groups[gone]["mask"]
                groups[keep]["sources"] += groups[gone]["sources"]
                del groups[gone]
                merged = True
                break
            if merged:
                break

    # 3. spill goes to the wall of its own slot that it touches (else that slot's largest)
    reassigned, dropped, orphans = 0, 0, 0
    for s in range(nslots):
        if not spill[s].any():
            continue
        px = int(spill[s].sum())
        home = [g for g in groups if g["slot"] == s and g["mask"].any()]
        if home:
            touching = [g for g in home if _touch(spill[s], g["mask"], radius)]
            target = max(touching or home, key=lambda g: int(g["mask"].sum()))
            target["mask"] = target["mask"] | spill[s]
            reassigned += px
        elif px >= MIN_WALL_SHARE * h * w:
            groups.append({"slot": s, "mask": spill[s].copy(), "direction": None, "normal": None, "sources": []})
            orphans += 1
            reassigned += px
        else:
            dropped += px

    groups = [g for g in groups if g["mask"].any()]
    # Ids keep the split's order (a merged wall takes its first piece's place;
    # a new wall from orphan spill comes last).
    groups.sort(key=lambda g: (min(g["sources"]) if g["sources"] else len(refined) + g["slot"]))
    out = [(i, g["mask"], g["direction"]) for i, g in enumerate(groups)]
    two_directions = sorted({g["slot"] for g in groups
                             if sum(1 for o in groups if o["slot"] == g["slot"]) > 1})
    report = {
        "enabled": True,
        "walls_before": before,
        "walls_after": len(out),
        "corners": [ev for _, _, ev in lines],
        "uncertain_corners": uncertain,
        "spill_pixels_reassigned": reassigned,
        "spill_pixels_dropped": dropped,
        "orphan_walls_created": orphans,
        "merged": [g["sources"] for g in groups if len(g["sources"]) > 1],
        "one_straight_edge_slots": straight,
        "slanted_seams_merged": seams,
        "slots_with_two_directions": two_directions,
        # Not 95% sure: a bend with no witness, or two walls in one slot with no corner between.
        "uncertain": bool(uncertain or two_directions),
    }
    return out, report


def dot(mask: np.ndarray, objects: Optional[np.ndarray]) -> tuple[int, int]:
    """
    A wall's dot: the visible flat wall nearest the centroid of the wall with
    objects (curtains, furniture, shelves) taken out, kept a margin inside it.
    Falls back to the deepest wall pixel when nothing visible is left.
    """
    from ...engine import interior_point

    h, w = mask.shape
    visible = mask & ~objects if objects is not None else mask
    if visible.sum() < 0.05 * mask.sum():
        visible = mask
    margin = max(2, int(round(0.01 * math.hypot(h, w))))
    padded = cv2.copyMakeBorder(visible.astype(np.uint8), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    depth = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]
    inner = depth >= margin
    if not inner.any():
        return interior_point(visible)
    ys, xs = np.nonzero(visible)
    cy, cx = ys.mean(), xs.mean()
    iy, ix = np.nonzero(inner)
    k = int(np.argmin((iy - cy) ** 2 + (ix - cx) ** 2))
    return int(ix[k]), int(iy[k])


# ----------------------------------------------------------------------------
# One wall = one plane = one grid: pieces of one wall split by a pillar.

PROJECTION_MAX_WIDTH = 0.15   # x width: a piece narrower than this can be a pillar / projection face
PROJECTION_MAX_PIECES = 2     # at most this many pieces between two pieces of one wall
COPLANAR_TOL = 0.004          # x diagonal: two pieces' edge points this close to one line = one plane
COPLANAR_MIN_SHARE = 0.03     # x width: each piece needs this many edge columns to be judged


def _faces(direction) -> Optional[str]:
    return (direction or {}).get("faces")


def _edge_pts(piece: np.ndarray, floor: np.ndarray, objects) -> tuple:
    from . import edge_layout
    from . import junction_layout as jl

    return jl.junction_points(floor, piece, objects), edge_layout.ceiling_points(piece, objects)


def _coplanar(a: np.ndarray, b: np.ndarray, floor: np.ndarray, objects, w: int, diag: float):
    """
    Evidence that two pieces lie on ONE plane, not merely parallel ones: their
    floor-wall junction points (else their ceiling-line points) fall on one
    straight image line. Returns the evidence string, or None (no evidence:
    parallel walls at different depths give two different lines).
    """
    ja, ca = _edge_pts(a, floor, objects)
    jb, cb = _edge_pts(b, floor, objects)
    for name, pa, pb in (("floor junction", ja, jb), ("ceiling line", ca, cb)):
        if len(pa) < COPLANAR_MIN_SHARE * w or len(pb) < COPLANAR_MIN_SHARE * w:
            continue
        pts = np.vstack([pa, pb]).astype(np.float32)
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        n = np.array([-vy, vx])
        da = np.abs((pa - [x0, y0]) @ n)
        db = np.abs((pb - [x0, y0]) @ n)
        if max(np.median(da), np.median(db)) <= COPLANAR_TOL * diag:
            return f"{name} on one line (median off {max(np.median(da), np.median(db)):.1f} px)"
    return None


def merge_planes(refined: list, floor: np.ndarray, wall: np.ndarray, objects: Optional[np.ndarray], room_bgr=None):
    """
    refined: [(index, mask, direction)] with directions fitted -> (new refined, report).

    Two pieces facing the same way with normals within MERGE_DEG, ON ONE PLANE
    (their floor junctions, else ceiling lines, fall on one image line -- a
    parallel wall at another depth does not), joined by a touching chain of at
    most PROJECTION_MAX_PIECES narrow pieces (a pillar or a projection), are one wall: merged into one mask with one direction, so
    the renderer gives it one plane, one grid and one scale. The pieces between
    them (the pillar's faces) stay walls of their own only when every edge
    between the chain's pieces is a confirmed corner (two witnesses);
    otherwise they are merged into the wall too and the room is marked
    uncertain (needs_fix): a flattened pillar is not 95% certain.
    Side walls (left / right) are handled before camera-facing ones, so a
    pillar's face never pulls the back wall into a side wall.
    """
    if not ENABLED or len(refined) < 3:
        return refined, {"enabled": ENABLED, "merged": []}
    h, w = wall.shape
    diag = math.hypot(h, w)
    radius = max(2, int(round(TOUCH * diag)))
    order = sorted(refined, key=lambda item: float(np.nonzero(item[1])[1].mean()))
    xs_mid = [float(np.nonzero(m)[1].mean()) for _, m, _ in order]
    widths = [int(m.any(axis=0).sum()) for _, m, _ in order]
    lines, doubtful, _, _ = corners(floor, wall, objects, room_bgr)
    confirmed_x = [(t[0] + b[0]) / 2 for t, b, _ in lines]
    # every bend of the floor or ceiling line, witnessed or not: a corner is there
    bend_x = confirmed_x + [u["x"] for u in doubtful]

    used, merges = set(), []
    for side_first in (True, False):
        for i in range(len(order)):
            for k in range(PROJECTION_MAX_PIECES + 1, 1, -1):     # longest chain first: the wall on both sides of the pillar
                j = i + k
                if j >= len(order) or i in used or j in used:
                    continue
                (ia, ma, da), (ib, mb, db) = order[i], order[j]
                fa = _faces(da)
                if fa is None or fa != _faces(db) or (fa == "front") == side_first:
                    continue
                na, nb = _normal(da), _normal(db)
                if na is None or nb is None or math.degrees(math.acos(min(1.0, abs(float(na @ nb))))) > MERGE_DEG:
                    continue
                between = list(range(i + 1, j))
                if any(b in used or widths[b] > PROJECTION_MAX_WIDTH * w for b in between):
                    continue
                chain = [i] + between + [j]
                if not all(_touch(order[a][1], order[b][1], radius) for a, b in zip(chain, chain[1:])):
                    continue
                coplanar = _coplanar(ma, mb, floor, objects, w, diag)
                if coplanar is None:
                    continue                      # parallel, not one plane: never merged
                # the edges between the chain's pieces, and whether each is a confirmed corner
                edges = []
                for a, b in zip(chain, chain[1:]):
                    k_ = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
                    seam = cv2.dilate(order[a][1].astype(np.uint8), k_).astype(bool) & order[b][1]
                    edges.append(float(np.nonzero(seam)[1].mean()))
                witnessed = [any(abs(x - c) <= MATCH * w for c in confirmed_x) for x in edges]
                pillar_kept = all(witnessed)
                members = [i, j] if pillar_kept else chain
                used.update(members)
                merges.append({
                    "wall_pieces": [int(order[i][0]), int(order[j][0])],
                    "between": [int(order[b][0]) for b in between],
                    "edges_x": [round(x, 1) for x in edges],
                    "edges_with_two_witnesses": witnessed,
                    "pillar": "kept as its own wall(s)" if pillar_kept else "merged into the wall (not 95% sure)",
                    "one_plane_evidence": coplanar,
                    "members": [int(order[m][0]) for m in members],
                })
    # directly touching neighbours of one wall (same facing, normals within MERGE_DEG)
    touching = []
    for i in range(len(order) - 1):
        (ia, ma, da), (ib, mb, db) = order[i], order[i + 1]
        if _faces(da) is None or _faces(da) != _faces(db):
            continue
        na, nb = _normal(da), _normal(db)
        if na is None or nb is None or math.degrees(math.acos(min(1.0, abs(float(na @ nb))))) > MERGE_DEG:
            continue
        if int(ia) in [x for m in merges for x in m["members"]] and int(ib) in [x for m in merges for x in m["members"]]:
            continue
        if not _touch(ma, mb, radius):
            continue
        k_ = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
        seam = cv2.dilate(ma.astype(np.uint8), k_).astype(bool) & mb
        seam_x = float(np.nonzero(seam)[1].mean())
        if any(abs(seam_x - x) <= MATCH * w for x in bend_x):
            continue                              # the edge bends at the seam: a corner, not one plane
        if _coplanar(ma, mb, floor, objects, w, diag) is not None:
            touching.append([int(ia), int(ib)])
    if not merges and not touching:
        return refined, {"enabled": True, "merged": []}

    parent = {int(i): int(i) for i, _, _ in refined}

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for group in [m["members"] for m in merges] + touching:
        for x in group[1:]:
            ra, rb = root(group[0]), root(x)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    by_index = {int(i): (i, mk, d) for i, mk, d in refined}
    out = []
    for index, mask, direction in refined:
        if root(int(index)) != int(index):
            continue
        members = [i for i in by_index if root(i) == int(index)]
        if len(members) > 1:
            union = np.zeros_like(mask)
            for i in members:
                union |= by_index[i][1]
            # one direction for the whole wall: its largest piece that faces the wall's way
            # (a pillar's front face never decides a side wall's plane)
            counts = {}
            for i in members:
                counts[_faces(by_index[i][2])] = counts.get(_faces(by_index[i][2]), 0) + int(by_index[i][1].sum())
            facing = max(counts, key=counts.get)
            same = [by_index[i] for i in members if _faces(by_index[i][2]) == facing]
            direction = max(same, key=lambda t: int(t[1].sum()))[2]
            out.append((index, union, direction))
        else:
            out.append((index, mask, direction))
    uncertain = any(m["pillar"].startswith("merged") for m in merges)
    return out, {"enabled": True, "merged": merges, "touching_merged": touching,
                 "walls_before": len(refined), "walls_after": len(out), "uncertain": uncertain}
