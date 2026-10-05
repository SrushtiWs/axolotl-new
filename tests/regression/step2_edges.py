"""
Step 2 diagnostic (no pixel changes): straight ceiling / floor line per wall.

    backend/.venv/bin/python tests/regression/step2_edges.py <baseline_dir> <out_dir> [room ...]

For every wall region of the baseline render (<baseline_dir>/<room>.npz):
  junction points   the region's top pixel per column whose 3 px above are
                    neither WALL_MASK nor FLOOR_MASK (nor an object), and its
                    bottom pixel per column whose 3 px below are FLOOR_MASK --
                    the masks render_tiled_room has (no ceiling label there)
  line              RANSAC through those points (threshold relative to the
                    image diagonal), refitted by least squares on the inliers
  residual          RMS distance of the INLIER points to the line, px and % diagonal
                    (outliers are the bulge being diagnosed; all-point RMS is reported too)
  sagitta           how far a quadratic through the inliers departs from the
                    line over their span (curved edges), px and % diagonal
  edge support      along the line, the photo's gradient across the line (max
                    within EDGE_BAND_PX) compared to its median at the wall's own
                    inlier junction points; >= SUPPORT_SHARE of it = supported.
                    Gaps up to GAP_SHARE of the diagonal between supported
                    stretches are bridged (a fan blade crossing the line).
  removable         each connected area of tiled pixels past the line is judged
                    as a whole: removed only when >= BLOB_SUPPORT of its pixels
                    have a supported foot on the line and it is at least
                    MIN_BLOB_SHARE of the image, else kept whole. Feet
                    beyond the image border take the support of the line's
                    last in-image sample.
  past labels       share of the pixels past the line that Clean Room labelled
                    ceiling (or floor) / wall / object
  verdict           straight / curved (needs Step 1) / too few points
  trigger           tiled wall pixels more than TOL above the ceiling line
                    (or below the floor line) on a straight line
Writes <out_dir>/<room>__step2_lines.png (lines drawn on the photo) and
<out_dir>/step2_diagnose.json.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "backend"), str(ROOT / "tests/regression")]

import perspective_engine as pe  # noqa: E402
import reuse  # noqa: E402
from perspective_engine import masks  # noqa: E402
from baseline_rooms import ROOMS  # noqa: E402

LOOK_PX = 3                 # rows above / below a region edge that must be ceiling / floor
RANSAC_SHARE = 0.003        # inlier threshold, share of the image diagonal (min 2 px)
TOL_SHARE = 0.002           # pixels this far past the line count for the trigger (min 2 px)
CURVED_RESIDUAL = 0.015     # RMS residual above this share of the diagonal = curved
CURVED_SAGITTA = 0.005      # quadratic sagitta above this share of the diagonal = curved
MIN_SPAN_SHARE = 0.3        # junction points must span this share of the wall's width
MIN_POINTS = 20
MIN_INLIER_SHARE = 0.6      # below: the line rests on a minority of the junction -> needs_fix
MIN_INLIER_SPAN = 0.25      # inliers must span this share of the wall's width
EDGE_BAND_PX = 3
SUPPORT_SHARE = 0.4
GAP_SHARE = 0.01
BLOB_SUPPORT = 0.9
MIN_BLOB_SHARE = 0.0001     # areas smaller than this share of the image are specks, kept


def junction_points(region, other, props, top):
    h, w = region.shape
    pts = []
    for x in np.nonzero(region.any(axis=0))[0]:
        ys = np.nonzero(region[:, x])[0]
        y = ys.min() if top else ys.max()
        lo, hi = (y - LOOK_PX, y) if top else (y + 1, y + 1 + LOOK_PX)
        if lo < 0 or hi > h:
            continue
        if other[lo:hi, x].all() and not props[lo:hi, x].any():
            pts.append((float(x), float(y)))
    return np.array(pts)


def fit_line(pts, thr, rng):
    best = None
    n = len(pts)
    for _ in range(400):
        i, j = rng.choice(n, 2, replace=False)
        d = pts[j] - pts[i]
        if np.hypot(*d) < 1e-6:
            continue
        nrm = np.array([-d[1], d[0]]) / np.hypot(*d)
        inl = np.abs((pts - pts[i]) @ nrm) < thr
        if best is None or inl.sum() > best.sum():
            best = inl
    q = pts[best]
    c = q.mean(axis=0)
    _, _, vt = np.linalg.svd(q - c)
    nrm = vt[1]
    if nrm[1] < 0:                       # normal points down (+y): "above" = negative distance
        nrm = -nrm
    return nrm, c, best


def edge_support(grad, nrm, c, inl_pts, region, diag):
    """Per tiled pixel: is the line supported by an image edge at its foot point?"""
    gx, gy = grad
    h, w = gx.shape
    nrm, c = np.asarray(nrm), np.asarray(c)
    tang = np.array([nrm[1], -nrm[0]])
    ys, xs = np.nonzero(region)
    u_px = (np.stack([xs, ys], 1) - c) @ tang
    u0, u1 = int(np.floor(u_px.min())), int(np.ceil(u_px.max()))
    us = np.arange(u0, u1 + 1)

    def strength(u):
        best = np.zeros(len(u))
        for k in range(-EDGE_BAND_PX, EDGE_BAND_PX + 1):
            p = c[None] + u[:, None] * tang[None] + k * nrm[None]
            x = np.clip(np.round(p[:, 0]).astype(int), 0, w - 1)
            y = np.clip(np.round(p[:, 1]).astype(int), 0, h - 1)
            inside = (p[:, 0] >= 0) & (p[:, 0] < w) & (p[:, 1] >= 0) & (p[:, 1] < h)
            g = np.abs(gx[y, x] * nrm[0] + gy[y, x] * nrm[1]) * inside
            best = np.maximum(best, g)
        return best

    def in_image(u):
        p = c[None] + u[:, None] * tang[None]
        return (p[:, 0] >= 0) & (p[:, 0] <= w - 1) & (p[:, 1] >= 0) & (p[:, 1] <= h - 1)

    ref = float(np.median(strength((inl_pts - c) @ tang)))
    sup = strength(us.astype(float)) >= SUPPORT_SHARE * max(ref, 1e-6)
    inside = in_image(us.astype(float))
    if inside.any():                        # off-image feet: the nearest in-image sample's support
        first, last = np.nonzero(inside)[0][[0, -1]]
        sup[:first] = sup[first]
        sup[last + 1:] = sup[last]
    # bridge short gaps between supported stretches
    gap = int(GAP_SHARE * diag)
    idx = np.nonzero(sup)[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= gap + 1:
            sup[a:b] = True
    foot = np.clip(np.round(u_px).astype(int) - u0, 0, len(us) - 1)
    return sup[foot], float(sup.mean()), ref


def analyse(pts, region, diag, top, rng, labels, grad):
    span = np.ptp(np.nonzero(region.any(axis=0))[0]) + 1
    if len(pts) < MIN_POINTS or np.ptp(pts[:, 0]) < MIN_SPAN_SHARE * span:
        return {"verdict": "too few junction points", "points": int(len(pts)),
                "point_span_px": float(np.ptp(pts[:, 0])) if len(pts) else 0.0, "wall_span_px": int(span)}
    thr = max(2.0, RANSAC_SHARE * diag)
    nrm, c, inl = fit_line(pts, thr, rng)
    dist = (pts - c) @ nrm
    rms_all = float(np.sqrt((dist ** 2).mean()))
    rms = float(np.sqrt((dist[inl] ** 2).mean()))
    # sagitta: quadratic through the inliers, in line coordinates (u along, v across)
    tang = np.array([nrm[1], -nrm[0]])
    u = (pts[inl] - c) @ tang
    coef = np.polyfit(u, dist[inl], 2)
    uu = np.linspace(u.min(), u.max(), 50)
    sag = float(np.abs(np.polyval(coef, uu)).max())
    curved = rms > CURVED_RESIDUAL * diag or sag > CURVED_SAGITTA * diag
    inl_span = float(np.ptp(pts[inl][:, 0]))
    weak = inl.mean() < MIN_INLIER_SHARE or inl_span < MIN_INLIER_SPAN * span
    tol = max(2.0, TOL_SHARE * diag)
    ys, xs = np.nonzero(region)
    d_px = (np.stack([xs, ys], 1) - c) @ nrm
    past = d_px < -tol if top else d_px > tol
    supported, support_share, ref = edge_support(grad, nrm, c, pts[inl], region, diag)
    blob = np.zeros(region.shape, np.uint8)
    blob[ys[past], xs[past]] = 1
    n, lab = cv2.connectedComponents(blob, connectivity=8)
    lab_px = lab[ys, xs]
    removable = np.zeros(len(ys), bool)
    for i in range(1, n):
        m = past & (lab_px == i)
        if m.sum() >= MIN_BLOB_SHARE * region.size and supported[m].mean() >= BLOB_SUPPORT:
            removable |= m
    verdict = "curved edge, needs Step 1" if curved else ("uncertain fit, needs_fix" if weak else "straight")
    return {"verdict": verdict,
            "points": int(len(pts)), "inliers": int(inl.sum()), "inlier_share": round(float(inl.mean()), 3),
            "point_span_px": float(np.ptp(pts[:, 0])), "wall_span_px": int(span),
            "inlier_span_px": inl_span,
            "residual_all_points_px": round(rms_all, 2),
            "residual_rms_px": round(rms, 2), "residual_pct_diag": round(100 * rms / diag, 3),
            "sagitta_px": round(sag, 2), "sagitta_pct_diag": round(100 * sag / diag, 3),
            "angle_deg": round(float(np.degrees(np.arctan2(tang[1], tang[0]))), 2),
            "tol_px": round(tol, 2),
            "tiled_past_line_px": int(past.sum()),
            "tiled_past_line_any_px": int((d_px < 0).sum() if top else (d_px > 0).sum()),
            "max_past_px": round(float(max(0.0, (-d_px).max() if top else d_px.max())), 1),
            "edge_ref_gradient": round(ref, 1), "line_supported_share": round(support_share, 3),
            "removable_px": int(removable.sum()) if verdict == "straight" else 0,
            "kept_unsupported_px": int((past & ~supported).sum()),
            "trigger": bool(verdict == "straight" and removable.sum() > 0),
            "past_labels": {k: round(float(m[ys[past], xs[past]].mean()), 3) if past.any() else None
                            for k, m in labels.items()},
            "_line": (nrm.tolist(), c.tolist()), "_inl": inl, "_past": (ys[past], xs[past]),
            "_remove": (ys[removable], xs[removable]) if verdict == "straight" else (ys[:0], xs[:0])}


def draw_line(img, nrm, c, color, h, w):
    tang = np.array([nrm[1], -nrm[0]])
    p0, p1 = np.array(c) - 4000 * tang, np.array(c) + 4000 * tang
    ok, a, b = cv2.clipLine((0, 0, w, h), tuple(int(v) for v in p0), tuple(int(v) for v in p1))
    if ok:
        cv2.line(img, a, b, color, 2, cv2.LINE_AA)


def main() -> int:
    base, out = Path(sys.argv[1]), Path(sys.argv[2])
    names = sys.argv[3:] or list(ROOMS)
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    for name in names:
        with tempfile.TemporaryDirectory() as tmp:
            job = Path(tmp) / ROOMS[name].name
            shutil.copytree(ROOMS[name], job)
            b = reuse.load(job)
            fl, wl = pe.load(job / "segments")
        h, w = b.room.shape[:2]
        floor_m, wall_m, _ = masks.prepare(fl, wl, (h, w))
        diag = float(np.hypot(h, w))
        surf = {s.label: np.asarray(s.mask, bool) for s in b.surfaces}
        ceiling, floor = ~wall_m & ~floor_m, floor_m
        props = np.asarray(b.props, bool)
        regions = dict(np.load(base / f"{name}.npz"))
        rng = np.random.default_rng(0)
        gray = cv2.GaussianBlur(cv2.cvtColor(b.room, cv2.COLOR_RGB2GRAY).astype(np.float32), (0, 0), 1.0)
        grad = (cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
        allowed = np.zeros((h, w), bool)
        labels = {"ceiling": surf.get("ceiling", np.zeros((h, w), bool)), "floor": surf.get("floor", np.zeros((h, w), bool)), "wall": surf.get("wall", np.zeros((h, w), bool)), "object": props}
        img = b.room.copy()
        tint = img.copy()
        room = {}
        for key in sorted(k for k in regions if k.startswith("wall-")):
            region = regions[key]
            tint[region] = (0.5 * tint[region] + 0.5 * np.array([60, 160, 255])).astype(np.uint8)
            row = {}
            for edge, other, top, color in (("ceiling", ceiling, True, (255, 40, 40)), ("floor", floor, False, (40, 120, 255))):
                pts = junction_points(region, other, props, top)
                a = analyse(pts, region, diag, top, rng, labels, grad)
                if "_line" in a:
                    nrm, c = a.pop("_line")
                    inl = a.pop("_inl")
                    py, px = a.pop("_past")
                    tint[py, px] = (255, 150, 0)          # past the line, kept
                    ry, rx = a.pop("_remove")
                    tint[ry, rx] = (255, 0, 255)          # past the line, removable
                    if a["verdict"] == "straight":        # allowed area: ALL pixels past a straight line
                        allowed[py, px] = True
                    draw_line(tint, nrm, c, color if a["verdict"] == "straight" else (255, 200, 0), h, w)
                    for (x, y), good in zip(pts.astype(int), inl):
                        cv2.circle(tint, (x, y), 2, (0, 255, 0) if good else (255, 255, 0), -1)
                    a["line_normal_center"] = [[round(v, 4) for v in nrm], [round(v, 1) for v in c]]
                row[edge] = a
            room[key] = row
        report[name] = {"size": [w, h], "diag_px": round(diag, 1), "walls": room,
                        "trigger_fires": any(e.get("trigger") for r in room.values() for e in r.values()),
                        "tiled_above_ceiling_lines_px": sum(r["ceiling"].get("tiled_past_line_px", 0) for r in room.values()),
                        "tiled_below_floor_lines_px": sum(r["floor"].get("tiled_past_line_px", 0) for r in room.values()),
                        "allowed_px": int(allowed.sum()),
                        "removable_px": sum(e.get("removable_px", 0) for r in room.values() for e in r.values())}
        report[name]["trigger_fires"] = report[name]["removable_px"] > 0
        Image.fromarray((allowed * 255).astype(np.uint8)).save(out / f"{name}__allowed_step2.png")
        Image.fromarray(tint).save(out / f"{name}__step2_lines.png")
        print(name, json.dumps(report[name]["walls"]), flush=True)
    (out / "step2_diagnose.json").write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
