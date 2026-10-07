"""
Mask v2: one solid mask per wall, the floor, pillars and an "excluded" class.

    backend/.venv/bin/python tools/mask_v2.py <room>

Standalone: reads a room's existing Clean Room data (photo, object masks,
FLOOR_MASK / WALL_MASK) and the L1 strong lines, and writes ONLY to
outputs/mask_v2/<room>/ (wall_1.png ..., floor.png, pillar_1.png ...,
excluded.png, overlay.png) and backups/mask_v2-report/<room>__mask.json.
No engine file and no existing output is read-modified-written.

1 excluded   curtain, blind, window(pane), door, glass, mirror -- SegFormer on
             the ORIGINAL photo (the clean room has them inpainted into "wall")
             -- plus every detected window / door / curtain object. Never part
             of a wall or of the floor.
2 lines      L1 strong lines (backend/perspective_engine/strong_lines.py) kept
             only when good: strong gradient along >= 90 % of the length and
             not within 10 px of furniture.
3 pillars    two near-parallel vertical lines, ceiling -> floor, 1.5-25 % of the
             width apart, around a strip whose floor junction stands forward of
             the wall beside it (a pillar protrudes). Searched with relaxed line
             thresholds (any L1 line, 15 % of the height, 5 deg) -- reported.
4 walls      the engine's own wall pieces (minus excluded, pillars, objects),
             merged across every seam that has no real corner. A corner is:
             a good vertical line (>= 20 % of the height) where the wall's own
             horizontal lines change family across it (different vanishing
             point each side: within 10 % of the width, >= 2 lines and a 70 %
             majority per side, lines on window / door frames ignored).
             Second witness pair, for rooms whose corner edges are too faint to
             detect: the ceiling junction AND the floor junction both bend at the
             same place (or one of them bends where a good vertical line is); the
             corner line then runs through the two bend points. A bend counts only
             where the junction is continuous on both sides (a gap behind a pillar
             or furniture is not a bend). Other corner lines run toward the
             vertical vanishing point. Wall pieces under 2 % of the image merge
             into the neighbour they share the longest border with.
5 clean      per mask: fill holes (not where something is excluded / an object /
             another surface), close then open, snap the boundary onto good
             lines within 4 px, drop components below 0.2 % of the image.
All thresholds are shares of the image size or of the photo's own lines.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "backend"), str(ROOT / "tests/regression")]

import perspective_engine as pe  # noqa: E402
import reuse  # noqa: E402
import surfaces  # noqa: E402
from baseline_rooms import ROOMS  # noqa: E402
from l1_lines import families  # noqa: E402
from perspective_engine import masks, strong_lines as sl  # noqa: E402

OUT = ROOT / "outputs" / "mask_v2"
REPORT = ROOT / "backups" / "mask_v2-report"
EXCLUDED_CLASSES = ("curtain", "blind", "windowpane", "window", "door", "double door", "screen door", "glass", "mirror")
EXCLUDED_OBJECTS = ("window", "door", "curtain", "blind", "mirror")
MIN_AREA = 0.002            # share of the image: smaller components are removed
GOOD_SUPPORT = 0.9
FURNITURE_PX = 10
CORNER_MIN_LEN = 0.20       # share of the image height
BEND_LINE_MIN_LEN = 0.10    # a vertical line paired with a junction bend: share of the height
SIDE_WINDOW = 0.10          # share of the width looked at on each side of a corner
SIDE_DOMINANCE = 0.7        # the dominant horizontal family must hold this share of a side's line length
MIN_WALL = 0.02             # wall pieces smaller than this share of the image merge into a neighbour
BEND_EPS = 0.004            # Douglas-Peucker tolerance, share of the diagonal
BEND_DEG = 5.0
BEND_SEG = 0.04             # both segments around a bend at least this share of the width
SKIRTING = 0.02             # floor within this share of the height below the wall's bottom still meets it
BEND_PAIR = 0.03            # ceiling and floor bends of one corner within this share of the width
PILLAR_MIN_LEN = 0.15       # relaxed (pillar search only)
PILLAR_ANGLE_DEG = 5.0      # relaxed (pillar search only)
PILLAR_WIDTH = (0.015, 0.25)
PILLAR_STEP = 0.01          # floor junction forward by this share of the height
PILLAR_SPAN = 0.6           # both pillar edges span this share of the local wall height
PILLAR_CLEAR = 0.1          # at most this share of the strip may be objects / excluded
SNAP_PX = 4


def save_png(path: Path, mask: np.ndarray):
    tmp = path.with_suffix(".tmp.png")
    Image.fromarray((mask.astype(np.uint8) * 255)).save(tmp)
    os.replace(tmp, path)


def line_mask_samples(s, h, w):
    L = max(2, int(np.hypot(s[2] - s[0], s[3] - s[1])))
    t = np.linspace(0, 1, L)
    x = np.clip(np.round(s[0] + t * (s[2] - s[0])).astype(int), 0, w - 1)
    y = np.clip(np.round(s[1] + t * (s[3] - s[1])).astype(int), 0, h - 1)
    return x, y


def support(s, gx, gy, strong, h, w):
    d = s[2:] - s[:2]
    L = float(np.hypot(*d))
    n = np.array([-d[1], d[0]]) / max(L, 1e-9)
    t = np.linspace(0, 1, max(2, int(L)))
    p = s[:2][None] + t[:, None] * d[None]
    best = np.zeros(len(p))
    for off in (-1, 0, 1):
        q = p + off * n
        x = np.clip(np.round(q[:, 0]).astype(int), 0, w - 1)
        y = np.clip(np.round(q[:, 1]).astype(int), 0, h - 1)
        best = np.maximum(best, np.abs(gx[y, x] * n[0] + gy[y, x] * n[1]))
    return float((best >= strong).mean())


def components_clean(m: np.ndarray, min_px: int) -> np.ndarray:
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= min_px
    return keep[lab]


def fill_holes(m: np.ndarray, protect: np.ndarray) -> np.ndarray:
    inv = (~m).astype(np.uint8)
    n, lab = cv2.connectedComponents(inv, connectivity=4)
    border = np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    hole = ~np.isin(lab, border) & ~m
    return m | (hole & ~protect)


def snap(m: np.ndarray, lines: np.ndarray) -> tuple[np.ndarray, int]:
    """Within SNAP_PX of a good line that the mask boundary follows, the boundary becomes the line."""
    h, w = m.shape
    edge = m & ~cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    ey, ex = np.nonzero(edge)
    if not len(ex):
        return m, 0
    pts = np.stack([ex, ey], 1).astype(float)
    out = m.copy()
    snapped = 0
    for s in lines:
        d = s[2:] - s[:2]
        L = float(np.hypot(*d))
        if L < 2:
            continue
        d = d / L
        nrm = np.array([-d[1], d[0]])
        u = (pts - s[:2]) @ d
        v = (pts - s[:2]) @ nrm
        near = (u >= 0) & (u <= L) & (np.abs(v) <= SNAP_PX)
        if near.sum() < 0.5 * L:
            continue
        x0, x1 = int(max(0, min(s[0], s[2]) - SNAP_PX)), int(min(w, max(s[0], s[2]) + SNAP_PX + 1))
        y0, y1 = int(max(0, min(s[1], s[3]) - SNAP_PX)), int(min(h, max(s[1], s[3]) + SNAP_PX + 1))
        yy, xx = np.mgrid[y0:y1, x0:x1]
        P = np.stack([xx, yy], -1).astype(float) - s[:2]
        uu, vv = P @ d, P @ nrm
        band = (uu >= 0) & (uu <= L) & (np.abs(vv) <= SNAP_PX)
        inside_side = np.sign(np.median(np.where(m[y0:y1, x0:x1] & band, vv, np.nan)[band & m[y0:y1, x0:x1]])) if (band & m[y0:y1, x0:x1]).any() else 0
        if inside_side == 0:
            continue
        new = (np.sign(vv) == inside_side) | (vv == 0)
        region = out[y0:y1, x0:x1]
        before = region.copy()
        region[band] = new[band]
        snapped += int((region != before).sum())
    return out, snapped


def clean(m, protect, lines, min_px, k):
    m = fill_holes(m, protect)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, ker).astype(bool)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, ker).astype(bool)
    m &= ~protect
    m, moved = snap(m, lines)
    m &= ~protect
    m = components_clean(m, min_px)
    return m, moved


def junction_bends(pts, w, diag):
    """Bends of a junction polyline [(x, y)]: Douglas-Peucker vertices where lines fitted
    over a window of BEND_SEG of the width on each side (each >= 70 % continuous) differ
    by >= BEND_DEG. The bend point is where the two fitted lines meet; bends closer than
    1 % of the width are one. A notch or bump (same direction on both sides) is no bend."""
    out = []
    if len(pts) < 10:
        return out
    pts = np.array(sorted(pts), float)
    win = BEND_SEG * w
    approx = cv2.approxPolyDP(pts.reshape(-1, 1, 2).astype(np.float32), BEND_EPS * diag, False).reshape(-1, 2)
    for m in approx[1:-1]:
        left = pts[(pts[:, 0] >= m[0] - win) & (pts[:, 0] <= m[0] - 2)]
        right = pts[(pts[:, 0] >= m[0] + 2) & (pts[:, 0] <= m[0] + win)]
        if len(left) < 0.7 * (win - 2) or len(right) < 0.7 * (win - 2):
            continue
        a1, b1 = np.polyfit(left[:, 0], left[:, 1], 1)
        a2, b2 = np.polyfit(right[:, 0], right[:, 1], 1)
        if abs(math.degrees(math.atan(a2)) - math.degrees(math.atan(a1))) < BEND_DEG:
            continue
        x = (b2 - b1) / (a1 - a2) if abs(a1 - a2) > 1e-9 else float(m[0])
        x = float(np.clip(x, m[0] - win, m[0] + win))
        q = np.array([x, a1 * x + b1])
        if not any(abs(q[0] - o[0]) <= 0.01 * w for o in out):
            out.append(q)
    return out


def side_of(line_pt, line_dir, xs, ys):
    nrm = np.array([-line_dir[1], line_dir[0]])
    return (np.stack([xs, ys], -1) - line_pt) @ nrm


def main() -> int:
    name = sys.argv[1]
    with tempfile.TemporaryDirectory() as tmp:
        job = Path(tmp) / ROOMS[name].name
        shutil.copytree(ROOMS[name], job)
        b = reuse.load(job)
        fl, wl = pe.load(job / "segments")
        objects = json.loads((job / "segments/objects.json").read_text()).get("objects", [])
        geo_doc = json.loads((job / "segments/wall/wall_geometry.json").read_text())
        pieces = [np.asarray(Image.open(job / f"segments/wall/walls/{wd['id']}.png").convert("L").resize(
            (b.room.shape[1], b.room.shape[0]), Image.NEAREST)) > 127 for wd in geo_doc.get("walls", [])]
    h, w = b.room.shape[:2]
    N, diag = h * w, math.hypot(h, w)
    min_px = int(MIN_AREA * N)
    k = max(3, int(round(0.003 * diag)) | 1)
    floor0, wall0, _ = masks.prepare(fl, wl, (h, w))
    props = np.asarray(b.props, bool)
    rep = {"room": name, "size": [w, h], "min_component_px": min_px, "morph_kernel_px": k}

    # 1 excluded
    lab, names = surfaces.label_map(b.room)
    excl = np.isin(lab, [i for i, v in names.items() if v in EXCLUDED_CLASSES])
    for o in objects:
        if o["label"] in EXCLUDED_OBJECTS:
            sw, sh = o.get("source_width") or w, o.get("source_height") or h
            x0, y0, x1, y1 = (np.array(o["bbox"], float) * [w / sw, h / sh, w / sw, h / sh]).astype(int)
            excl[max(0, y0):y1, max(0, x0):x1] |= props[max(0, y0):y1, max(0, x0):x1]
    excl = cv2.morphologyEx(excl.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((k, k), np.uint8)).astype(bool)
    excl = components_clean(excl, min_px)
    rep["excluded_px_share"] = round(float(excl.mean()), 4)

    # 2 lines
    res = sl.strong_lines(b.room, props=props, floor=floor0, labels=(lab, names))
    gx, gy = res["gradient"]
    lines = res["lines"]
    fam, vps = families(lines, w, h)
    near_obj = cv2.dilate(props.astype(np.uint8), np.ones((2 * FURNITURE_PX + 1,) * 2, np.uint8)).astype(bool)
    good = np.zeros(len(lines), bool)
    for i, s in enumerate(lines):
        x, y = line_mask_samples(s, h, w)
        good[i] = support(s, gx, gy, res["strong"], h, w) >= GOOD_SUPPORT and near_obj[y, x].mean() < 0.5
    rep["lines"] = {"l1_kept": int(len(lines)), "good": int(good.sum()),
                    "good_by_family": {f: int((good & (fam == f)).sum()) for f in ("vertical", "horizontal-A", "horizontal-B")}}
    V = vps.get("vertical")

    def vdir(p):
        if V is None:
            return np.array([0.0, 1.0])
        kind, v = V
        d = (v - p) if kind == "vp" else v
        d = d / np.linalg.norm(d)
        return d if d[1] >= 0 else -d

    # 3 pillars (relaxed vertical search)
    all_l = res["all_merged"]
    cand = []
    for s in all_l:
        L = float(np.hypot(s[2] - s[0], s[3] - s[1]))
        if L < PILLAR_MIN_LEN * h:
            continue
        mid = (s[:2] + s[2:]) / 2
        d = (s[2:] - s[:2]) / L
        if math.degrees(math.acos(min(1.0, abs(float(d @ vdir(mid)))))) > PILLAR_ANGLE_DEG:
            continue
        x, y = line_mask_samples(s, h, w)
        if (wall0 | excl)[y, x].mean() < 0.3 and not wall0[np.clip(y, 0, h - 1), np.clip(x + 3, 0, w - 1)].any():
            continue
        cand.append(s)
    cand = sorted(cand, key=lambda s: (s[0] + s[2]) / 2)
    bottom = np.full(w, -1)
    fjoin = np.zeros(w, bool)                 # the wall's bottom meets the floor here (not an object)
    gap = max(4, int(SKIRTING * h))
    for x in range(w):
        ys = np.nonzero(wall0[:, x])[0]
        if ys.size:
            bottom[x] = ys.max()
            fjoin[x] = bool(floor0[bottom[x] + 1:min(h, bottom[x] + 1 + gap), x].any()) and not props[bottom[x]:min(h, bottom[x] + gap), x].any()
    pillars, pillar_lines = [], []
    used = set()
    for i in range(len(cand)):
        for j in range(len(cand) - 1, i, -1):
            if i in used or j in used:
                continue
            a, c = cand[i], cand[j]
            xa, xc = (a[0] + a[2]) / 2, (c[0] + c[2]) / 2
            if not PILLAR_WIDTH[0] * w <= xc - xa <= PILLAR_WIDTH[1] * w:
                continue
            ya, yc = sorted([a[1], a[3]]), sorted([c[1], c[3]])
            ov = min(ya[1], yc[1]) - max(ya[0], yc[0])
            if ov < 0.5 * min(ya[1] - ya[0], yc[1] - yc[0]):
                continue
            band = max(3, int(0.03 * w))
            cols_in = np.arange(int(xa) + 2, int(xc) - 1)
            cols_l = np.arange(max(0, int(xa) - band), int(xa) - 1)
            cols_r = np.arange(int(xc) + 2, min(w, int(xc) + band))
            if not len(cols_in) or not fjoin[cols_in].mean() >= 0.7:
                continue
            inside = float(np.median(bottom[cols_in][fjoin[cols_in]]))
            # forward of the floor junction the wall itself would have here: a line through
            # the junction on both sides (a receding wall's junction slopes; that is no step)
            out_c = np.r_[cols_l[fjoin[cols_l]], cols_r[fjoin[cols_r]]] if len(cols_l) and len(cols_r) else np.zeros(0, int)
            if len(out_c) < 0.7 * (len(cols_l) + len(cols_r)) or not (fjoin[cols_l].any() and fjoin[cols_r].any()):
                continue
            a_, b_ = np.polyfit(out_c, bottom[out_c], 1)
            expected = a_ * cols_in[fjoin[cols_in]] + b_
            if float(np.median(bottom[cols_in][fjoin[cols_in]] - expected)) < PILLAR_STEP * h:
                continue
            strip = np.zeros((h, w), bool)
            strip[:, int(xa):int(xc) + 1] = True
            strip &= np.arange(h)[:, None] <= inside
            if not strip.any() or (strip & (props | excl)).mean() > PILLAR_CLEAR:
                continue
            col_h = [len(np.nonzero(wall0[:, int(x)] | props[:, int(x)])[0]) for x in (xa, xc)]
            if min(ya[1] - ya[0], yc[1] - yc[0]) < PILLAR_SPAN * max(1, min(col_h)):
                continue
            pm = np.zeros((h, w), np.uint8)
            poly = np.array([[a[0], a[1]], [a[2], a[3]], [c[2], c[3]], [c[0], c[1]]], np.float32)
            top_y = min(ya[0], yc[0])
            pa, pc = (a[:2] + a[2:]) / 2, (c[:2] + c[2:]) / 2
            da, dc = vdir(pa), vdir(pc)
            ta, tc = (top_y - pa[1]) / da[1], (top_y - pc[1]) / dc[1]
            ba, bc = (h - 1 - pa[1]) / da[1], (h - 1 - pc[1]) / dc[1]
            poly = np.array([pa + ta * da, pc + tc * dc, pc + bc * dc, pa + ba * da], np.int32)
            cv2.fillPoly(pm, [poly], 1)
            pillar = pm.astype(bool) & (wall0 | floor0) & ~excl
            pillar &= np.arange(h)[:, None] <= np.maximum(bottom[None, :], 0)
            if pillar.sum() < min_px:
                continue
            pillars.append(pillar)
            pillar_lines += [a, c]
            used.update({i, j})
    pillar_all = np.zeros((h, w), bool)
    for p in pillars:
        pillar_all |= p
    rep["pillar_search"] = {"relaxed": True, "min_length_share_of_height": PILLAR_MIN_LEN, "angle_tolerance_deg": PILLAR_ANGLE_DEG,
                            "vertical_candidates": len(cand), "pillars": len(pillars)}

    # 4 walls: corners
    base = wall0 & ~excl & ~pillar_all & ~props
    near_excl = cv2.dilate(excl.astype(np.uint8), np.ones((21, 21), np.uint8)).astype(bool)
    pil_x = [(s[0] + s[2]) / 2 for s in pillar_lines]
    corners, rejected = [], []
    vert_good = [s for s, g, f in zip(lines, good, fam) if g and f == "vertical" and np.hypot(s[2] - s[0], s[3] - s[1]) >= CORNER_MIN_LEN * h]
    hor = [(s, f) for s, g, f in zip(lines, good, fam) if g and f in ("horizontal-A", "horizontal-B")]
    seen_x = []
    for s in sorted(vert_good, key=lambda s: -np.hypot(s[2] - s[0], s[3] - s[1])):
        p = (s[:2] + s[2:]) / 2
        if any(abs(p[0] - x) <= 0.015 * w for x in seen_x + pil_x):
            continue
        x, y = line_mask_samples(s, h, w)
        if cv2.dilate(base.astype(np.uint8), np.ones((13, 13), np.uint8)).astype(bool)[y, x].mean() < 0.5:
            continue
        d = vdir(p)
        sides = {-1: {}, 1: {}}
        for hs, hf in hor:
            m = (hs[:2] + hs[2:]) / 2
            sd = float((m - p) @ np.array([-d[1], d[0]]))
            if abs(sd) > SIDE_WINDOW * w or abs(sd) < 3:
                continue
            xs, ys = line_mask_samples(hs, h, w)
            if not base[ys, xs].mean() >= 0.3 and not cv2.dilate(base.astype(np.uint8), np.ones((13, 13), np.uint8)).astype(bool)[ys, xs].mean() >= 0.5:
                continue
            if near_excl[ys, xs].mean() >= 0.3:
                continue                                  # a window / door frame, not the wall
            key = 1 if sd > 0 else -1
            sides[key].setdefault(hf, []).append(float(np.hypot(*(hs[2:] - hs[:2]))))
        dom = {}
        for k_, v in sides.items():
            tot = sum(sum(x) for x in v.values())
            best = max(v, key=lambda f_: sum(v[f_])) if v else None
            ok = best is not None and len(v[best]) >= 2 and sum(v[best]) >= SIDE_DOMINANCE * tot
            dom[k_] = best if ok else None
        why = None
        if dom[-1] and dom[1] and dom[-1] != dom[1]:
            why = f"horizontal lines change family ({dom[-1]} | {dom[1]})"
        if why:
            corners.append((p, d, why))
            seen_x.append(p[0])
        else:
            rejected.append({"x": round(float(p[0]), 1), "families": dom})
    # second witness pair: ceiling and floor junction bends
    ceil_pts, floor_pts = [], []
    ceil_like = ~wall0 & ~floor0 & ~props & ~excl
    for xx in range(w):
        col = np.nonzero(base[:, xx])[0]
        if not col.size:
            continue
        yt, yb = col.min(), col.max()
        if yt >= 3 and ceil_like[yt - 3:yt, xx].all():
            ceil_pts.append((xx, yt))
        gap = max(4, int(SKIRTING * h))          # a skirting board may sit between wall and floor
        if yb + 4 <= h and floor0[yb + 1:min(h, yb + 1 + gap), xx].any():
            floor_pts.append((xx, yb))
    cb, fb = junction_bends(ceil_pts, w, diag), junction_bends(floor_pts, w, diag)
    rep["junction_bends"] = {"ceiling": [[round(float(v), 1) for v in q] for q in cb],
                             "floor": [[round(float(v), 1) for v in q] for q in fb]}
    vg_x = [((s_[0] + s_[2]) / 2, s_) for s_, g_, f_ in zip(lines, good, fam)
            if g_ and f_ == "vertical" and np.hypot(s_[2] - s_[0], s_[3] - s_[1]) >= BEND_LINE_MIN_LEN * h]
    for q in cb + fb:
        if any(abs(q[0] - x) <= 0.02 * w for x in seen_x + pil_x):
            continue
        partner = [r for r in (fb if any(q is c_ for c_ in cb) else cb) if abs(r[0] - q[0]) <= BEND_PAIR * w]
        if partner:
            r = min(partner, key=lambda r: abs(r[0] - q[0]))
            top, bot = (q, r) if q[1] < r[1] else (r, q)
            d = (bot - top) / np.linalg.norm(bot - top)
            corners.append((top, d, f"ceiling and floor junction both bend (x {top[0]:.0f} / {bot[0]:.0f})"))
            seen_x.append(q[0])
            continue
        near_v = [s_ for x, s_ in vg_x if abs(x - q[0]) <= 0.015 * w]
        if near_v:
            corners.append((q, vdir(q), f"junction bend at x {q[0]:.0f} with a vertical photo line"))
            seen_x.append(q[0])
    corners.sort(key=lambda c: c[0][0])
    # walls: the engine's own wall pieces, merged across every seam with no corner evidence
    walls = [m & base for m in pieces if (m & base).any()]
    walls.sort(key=lambda m: np.nonzero(m)[1].mean())
    seams_kept, seams_merged = [], []
    i = 0
    while i < len(walls) - 1:
        ga = cv2.dilate(walls[i].astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
        seam = ga & walls[i + 1]
        if not seam.any():
            i += 1
            continue
        sx = float(np.nonzero(seam)[1].mean())
        ev = [why for p, d, why in corners if abs(float(p[0]) - sx) <= BEND_PAIR * w]
        if ev:
            seams_kept.append({"x": round(sx, 1), "evidence": ev[0]})
            i += 1
        else:
            walls[i] |= walls.pop(i + 1)
            seams_merged.append({"x": round(sx, 1), "evidence": "none -> merged"})
    # split a wall along a two-witness corner lying inside it (not at a kept seam)
    split_at = []
    for p, d, why in corners:
        if any(abs(float(p[0]) - k_["x"]) <= BEND_PAIR * w for k_ in seams_kept):
            continue
        for i, m in enumerate(walls):
            yy, xx = np.nonzero(m)
            sd = side_of(p, d, xx, yy) * np.sign(d[1])
            left, right = (sd >= 0), (sd < 0)
            if left.sum() >= MIN_WALL * N and right.sum() >= MIN_WALL * N:
                a_ = np.zeros((h, w), bool)
                a_[yy[right], xx[right]] = True
                b_ = np.zeros((h, w), bool)
                b_[yy[left], xx[left]] = True
                walls[i:i + 1] = [a_, b_]
                split_at.append({"x": round(float(p[0]), 1), "evidence": why})
                break
    walls.sort(key=lambda m: np.nonzero(m)[1].mean())
    rep["split_inside_pieces"] = split_at
    rep["engine_pieces"] = len(pieces)
    rep["seams_kept"] = seams_kept
    rep["seams_merged"] = seams_merged
    merged_small = 0
    changed = True
    while changed and len(walls) > 1:
        changed = False
        areas = [int(m.sum()) for m in walls]
        i = int(np.argmin(areas))
        if areas[i] < MIN_WALL * N:
            grown = cv2.dilate(walls[i].astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
            shared = [int((grown & walls[j]).sum()) if j != i else -1 for j in range(len(walls))]
            j = int(np.argmax(shared))
            if shared[j] > 0:
                walls[j] |= walls[i]
            walls.pop(i)
            merged_small += 1
            changed = True
    rep["small_wall_pieces_merged"] = merged_small
    rep["corners"] = [{"x": round(float(p[0]), 1), "evidence": why} for p, d, why in corners]
    rep["corner_candidates_rejected"] = rejected

    # 5 clean
    glines = lines[good]
    floor = floor0 & ~excl & ~pillar_all
    walls_c, moved = [], {}
    others_union = lambda i: np.logical_or.reduce([walls[j] for j in range(len(walls)) if j != i]) if len(walls) > 1 else np.zeros((h, w), bool)
    for i, m in enumerate(walls):
        protect = excl | pillar_all | floor | props | others_union(i)
        mc, mv = clean(m, protect, glines, min_px, k)
        walls_c.append(mc)
        moved[f"wall_{i + 1}"] = mv
    walls_c = [m for m in walls_c if m.any()]
    taken = np.zeros((h, w), bool)
    for i in range(len(walls_c)):
        walls_c[i] &= ~taken
        walls_c[i] = components_clean(walls_c[i], min_px)
        taken |= walls_c[i]
    walls_c = [m for m in walls_c if m.any()]
    floor_c, mvf = clean(floor, excl | pillar_all | taken, glines, min_px, k)
    moved["floor"] = mvf
    pillars_c = [components_clean(p & ~excl, min_px) for p in pillars]
    pillars_c = [p for p in pillars_c if p.any()]
    excl_c = components_clean(excl, min_px)

    # outputs
    od = OUT / name
    od.mkdir(parents=True, exist_ok=True)
    for f in od.glob("*.png"):
        f.unlink()
    for i, m in enumerate(walls_c):
        save_png(od / f"wall_{i + 1}.png", m)
    save_png(od / "floor.png", floor_c)
    for i, m in enumerate(pillars_c):
        save_png(od / f"pillar_{i + 1}.png", m)
    save_png(od / "excluded.png", excl_c)
    pal = [(230, 80, 80), (80, 200, 90), (80, 130, 240), (240, 180, 40), (190, 90, 230), (40, 210, 210)]
    ov = (b.room.astype(np.float32) * 0.45)
    for i, m in enumerate(walls_c):
        ov[m] = 0.5 * ov[m] + 0.5 * np.array(pal[i % len(pal)]) * 0.9
    ov[floor_c] = 0.5 * ov[floor_c] + 0.5 * np.array([120, 120, 120])
    for m in pillars_c:
        ov[m] = 0.4 * ov[m] + 0.6 * np.array([255, 140, 0])
    hatch = excl_c & ((np.add.outer(np.arange(h), np.arange(w)) // 6) % 2 == 0)
    ov[hatch] = (255, 255, 255)
    ov = ov.astype(np.uint8)
    for m in walls_c + [floor_c] + pillars_c:
        e = m & ~cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        ov[e] = (255, 255, 255)
    for p, d, _ in corners:
        a1, a2 = p - 4000 * d, p + 4000 * d
        ok, q1, q2 = cv2.clipLine((0, 0, w, h), tuple(int(v) for v in a1), tuple(int(v) for v in a2))
        if ok:
            cv2.line(ov, q1, q2, (255, 255, 0), 2)
    for i, m in enumerate(walls_c):
        yy, xx = np.nonzero(m)
        cv2.putText(ov, f"wall_{i + 1}", (int(np.median(xx)) - 30, int(np.median(yy))), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    for i, m in enumerate(pillars_c):
        yy, xx = np.nonzero(m)
        cv2.putText(ov, f"pillar_{i + 1}", (int(np.median(xx)) - 30, int(np.median(yy))), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    Image.fromarray(np.hstack([b.room, ov])).save(od / "overlay.png")

    # report
    def isolated(m):
        nb = cv2.filter2D(m.astype(np.uint8), -1, np.ones((3, 3), np.float32))
        return int((m & (nb <= 1)).sum())

    def small(m):
        n, lab_, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
        return int((st[1:, cv2.CC_STAT_AREA] < min_px).sum())
    allm = {f"wall_{i + 1}": m for i, m in enumerate(walls_c)} | {"floor": floor_c} | {f"pillar_{i + 1}": m for i, m in enumerate(pillars_c)}
    overlap = 0
    for i in range(len(walls_c)):
        for j in range(i + 1, len(walls_c)):
            overlap += int((walls_c[i] & walls_c[j]).sum())
    wall_union = np.logical_or.reduce(walls_c) if walls_c else np.zeros((h, w), bool)
    rep.update({
        "walls_found": len(walls_c), "pillars_found": len(pillars_c),
        "per_mask": {k_: {"area_share": round(float(m.mean()), 4), "speckle_ratio": round(isolated(m) / max(m.sum(), 1), 6),
                          "components_below_min": small(m), "excluded_overlap_share": round(float((m & excl_c).sum() / max(m.sum(), 1)), 4)}
                     for k_, m in allm.items()},
        "wall_vs_curtain_window_overlap_pct": round(100 * float((wall_union & excl_c).sum() / max(wall_union.sum(), 1)), 4),
        "floor_vs_curtain_window_overlap_pct": round(100 * float((floor_c & excl_c).sum() / max(floor_c.sum(), 1)), 4),
        "floor_area_pct": round(100 * float(floor_c.mean()), 2),
        "wall_wall_overlap_pct": round(100 * overlap / max(wall_union.sum(), 1), 4),
        "boundary_px_snapped": moved,
        "old_masks": {"WALL_MASK_share": round(float(wall0.mean()), 4), "FLOOR_MASK_share": round(float(floor0.mean()), 4)},
    })
    REPORT.mkdir(parents=True, exist_ok=True)
    tmpj = REPORT / f"{name}__mask.tmp"
    tmpj.write_text(json.dumps(rep, indent=1))
    os.replace(tmpj, REPORT / f"{name}__mask.json")
    print(name, json.dumps({k_: rep[k_] for k_ in ("walls_found", "pillars_found", "wall_vs_curtain_window_overlap_pct", "floor_area_pct", "wall_wall_overlap_pct")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
