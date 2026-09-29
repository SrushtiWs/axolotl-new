"""
The room's walls, settled once: which pixels are one wall, and which way it runs.

The wall pipeline's split (`instances.detect_instances`) groups WALL_MASK by
depth direction. On real rooms it makes three kinds of mistake that the
direction of each piece can correct:

  * several walls in one instance -- measured on a living room, the back wall
    between the curtains, the partition wall beside it (a side wall) and the
    kitchen seen through the glass came out as ONE wall with one plane;
  * one wall in several instances -- a coplanar patch cut out of a back wall
    got its own plane and its own, mismatched tile grid;
  * a "wall" that is not one -- a thin sliver along the frame edge, or the
    inpainted footprint of a removed sofa, offered as a wall of its own.

So, after the split:

    1. every instance is broken into its connected pieces;
    2. each piece gets a direction -- its own lines (`geometry.wall_direction`),
       else the room frame the floor fixes (`geometry.manhattan_direction`);
    3. pieces are merged back where they belong together:
         same original wall + same direction           -> one wall again
         touching + same direction                      -> one wall
         sliver or removed-object ghost                 -> the wall it touches most
    4. walls are re-indexed, and each gets its final direction.

Pixels are never dropped: every WALL_MASK pixel ends in exactly one wall.
"""

from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np

from . import geometry as wall_geometry

#: A connected piece smaller than this share of the whole wall mask is not
#: considered on its own; it stays with the largest piece of its instance.
MIN_PIECE_SHARE = 0.004

#: A piece is a sliver when it is narrower, at its widest, than this share of
#: the image diagonal (a strip along the frame edge or a corner seam), or holds
#: less than SLIVER_SHARE of the wall mask.
SLIVER_WIDTH_FRACTION = 0.03
SLIVER_SHARE = 0.02

#: A strip up to twice that wide is still a sliver when it is this many times
#: taller than it is wide (a band down the frame edge, not a wall face).
SLIVER_ELONGATION = 8.0

#: A piece is the ghost of a removed object when at least this share of it
#: lies where an object was taken out of the photograph.
GHOST_SHARE = 0.6

#: Two directions are the same wall direction within this many degrees.
SAME_DIRECTION_DEG = 10.0

#: Pixels of shared border for two pieces to count as touching.
MIN_CONTACT_PX = 12


def _pieces(mask: np.ndarray, min_px: int):
    n, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    parts = [labels == k for k in range(1, n)]
    parts.sort(key=lambda m: -int(m.sum()))
    big = [p for p in parts if p.sum() >= min_px]
    if not big:
        return [mask.copy()]
    for p in parts:
        if p.sum() < min_px:
            big[0] |= p                    # tiny bits stay with the main piece
    return big


def _direction(room_bgr, mask, floor, wall, floor_geo, f, cx, cy) -> Optional[dict]:
    """
    The wall's own lines first; the room frame when they give nothing, or give
    a plane that cannot hold the wall (part of it behind the camera).
    """
    d = wall_geometry.wall_direction(room_bgr, mask, floor, wall, f, cx, cy)
    if d is not None and wall_geometry._reach(np.asarray(d["normal"], float), mask, f, cx, cy) \
            < wall_geometry.MANHATTAN_MIN_REACH:
        d = None
    if d is None:
        d = wall_geometry.manhattan_direction(room_bgr, mask, floor, wall, floor_geo, f, cx, cy)
    return d


def _normal_angle(a: Optional[dict], b: Optional[dict]) -> float:
    if not a or not b:
        return 180.0
    na, nb = np.asarray(a["normal"], float), np.asarray(b["normal"], float)
    c = abs(float(na @ nb)) / ((np.linalg.norm(na) * np.linalg.norm(nb)) or 1.0)
    return math.degrees(math.acos(min(1.0, c)))


def _contact(a: np.ndarray, b: np.ndarray) -> int:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return int((cv2.dilate(a.astype(np.uint8), k).astype(bool) & b).sum())


def _width(mask: np.ndarray) -> float:
    """Twice the largest inscribed radius: the piece's width at its widest."""
    dist = cv2.distanceTransform(np.pad(mask.astype(np.uint8), 1), cv2.DIST_L2, 5)
    return 2.0 * float(dist.max())


def refine(instances, room_bgr, floor: np.ndarray, wall: np.ndarray, floor_geo: dict,
           f: float, cx: float, cy: float, objects: Optional[np.ndarray] = None):
    """
    instances: [(index, mask)] from the wall split.
    Returns ([(index, mask, direction)], report). Indices are 0..n-1, ordered
    by size, so the largest wall is wall-0.
    """
    total = int(wall.sum()) or 1
    diag = float(np.hypot(*wall.shape))
    min_px = max(200, int(MIN_PIECE_SHARE * total))

    pieces = []                            # dicts: mask, origin, direction, ghost, sliver
    for origin, mask in instances:
        for part in _pieces(mask & wall, min_px):
            pieces.append({"mask": part, "origin": int(origin)})

    for p in pieces:
        m = p["mask"]
        px = int(m.sum())
        p["px"] = px
        p["ghost"] = bool(objects is not None and (m & objects).sum() >= GHOST_SHARE * px)
        width = _width(m)
        rows = np.nonzero(m.any(axis=1))[0]
        height = float(rows.max() - rows.min() + 1) if len(rows) else 0.0
        p["sliver"] = bool(
            width < SLIVER_WIDTH_FRACTION * diag
            or px < SLIVER_SHARE * total
            or (width < 2 * SLIVER_WIDTH_FRACTION * diag and height > SLIVER_ELONGATION * width)
        )
        # Ghosts get a direction too: a wall behind removed curtains is still a
        # wall. Only slivers are too thin to say which way they run.
        p["direction"] = None if p["sliver"] else _direction(
            room_bgr, m, floor, wall, floor_geo, f, cx, cy)

    report = {"pieces": len(pieces), "merges": []}

    def merge(into, other, why):
        into["mask"] = into["mask"] | other["mask"]
        into["px"] += other["px"]
        into.setdefault("members", []).append(other.get("origin"))
        report["merges"].append(why)
        pieces[:] = [q for q in pieces if q is not other]   # by identity: pieces hold arrays

    # (0) A piece with no direction of its own that touches a wall whose
    #     direction is known continues that wall (the far end of a side wall
    #     beside a curtain, the part of a wall below a shelf).
    for p in sorted([p for p in pieces if p["direction"] is None and not p["sliver"]],
                    key=lambda p: p["px"]):
        if not any(p is q for q in pieces):
            continue
        hosts = [(q, _contact(p["mask"], q["mask"])) for q in pieces
                 if q is not p and q["direction"] is not None and not q["sliver"]]
        hosts = [(q, c) for q, c in hosts if c >= MIN_CONTACT_PX]
        if not hosts:
            continue
        # Touching is not enough -- two walls touch at every corner. The
        # piece's own edges must lean toward the neighbour's axis.
        cands, _segs = wall_geometry.manhattan_candidates(
            room_bgr, p["mask"], floor, wall, floor_geo, f, cx, cy)
        agreeing = []
        for q, c in hosts:
            axis = wall_geometry.axis_of(q["direction"]["normal"], cands)
            if axis is None:
                continue
            other = "back" if axis == "side" else "side"
            mine, theirs = cands[axis]["share"], cands[other]["share"]
            # The same bar the room-frame choice itself uses: clear evidence,
            # not a lean from a couple of short lines near a corner.
            if (mine >= wall_geometry.MANHATTAN_MIN_SHARE
                    and mine - theirs >= wall_geometry.MANHATTAN_MARGIN):
                agreeing.append((q, c))
        if agreeing:
            merge(max(agreeing, key=lambda t: t[1])[0], p, "continues the wall it touches")

    # (a) One original wall is split only on evidence: its pieces go back
    #     together unless BOTH have a direction and the directions differ.
    #     Slivers and removed-object ghosts of that wall rejoin it too.
    changed = True
    while changed:
        changed = False
        for i, a in enumerate(pieces):
            for b in pieces[i + 1:]:
                if a["origin"] != b["origin"]:
                    continue
                known = a["direction"] is not None and b["direction"] is not None
                if known and _normal_angle(a["direction"], b["direction"]) > SAME_DIRECTION_DEG:
                    continue
                keep, drop = (a, b) if a["px"] >= b["px"] else (b, a)
                if keep["direction"] is None:
                    keep["direction"] = drop["direction"]
                keep["ghost"] = keep["ghost"] and drop["ghost"]
                keep["sliver"] = False
                merge(keep, drop, "same wall")
                changed = True
                break
            if changed:
                break

    # (b) Two walls that touch and run the same way are one wall (a coplanar
    #     patch). Only walls whose direction is known, and not ghosts.
    changed = True
    while changed:
        changed = False
        solid = [p for p in pieces if p["direction"] is not None and not p["ghost"] and not p["sliver"]]
        for i, a in enumerate(solid):
            for b in solid[i + 1:]:
                if _normal_angle(a["direction"], b["direction"]) > SAME_DIRECTION_DEG:
                    continue
                if _contact(a["mask"], b["mask"]) >= MIN_CONTACT_PX:
                    keep, drop = (a, b) if a["px"] >= b["px"] else (b, a)
                    merge(keep, drop, "same direction (touching)")
                    changed = True
                    break
            if changed:
                break

    # (c) slivers and removed-object ghosts go to the wall they share the
    #     longest border with. A real wall with no direction evidence is NOT
    #     merged: it stays its own wall and the pipeline orients it as before.
    for p in sorted([p for p in pieces if p["ghost"] or p["sliver"]],
                    key=lambda p: p["px"]):
        if not any(p is q for q in pieces):
            continue
        hosts = [(q, _contact(p["mask"], q["mask"])) for q in pieces
                 if q is not p and not q["ghost"] and not q["sliver"]]
        hosts = [(q, c) for q, c in hosts if c >= MIN_CONTACT_PX]
        if not hosts:
            continue                       # nothing to join: stays its own wall
        host = max(hosts, key=lambda t: t[1])[0]
        why = "ghost of a removed object" if p["ghost"] else "sliver"
        merge(host, p, why)

    # Final walls, largest first; a merged wall keeps the direction of the
    # piece it was built on (its lines or the room frame), and one without
    # gets a fresh look now that it is whole.
    pieces.sort(key=lambda p: -p["px"])
    out = []
    for index, p in enumerate(pieces):
        d = p["direction"]
        if d is None:
            d = _direction(room_bgr, p["mask"], floor, wall, floor_geo, f, cx, cy)
        out.append((index, p["mask"], d))

    report["walls"] = len(out)
    return out, report
