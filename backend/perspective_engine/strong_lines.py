"""
L1 strong lines: the room's own straight lines, detected, fused and filtered.

Nothing in the render path calls this module yet; it only measures. Later
parts (vanishing points, camera, wall edges) read its output.

    detect   DeepLSD at full scale and at 0.5x (run in ~/geocalib-env as a
             subprocess, backend/deeplsd_runner.py; lines cached per photo in
             backups/deeplsd-cache/) plus OpenCV LSD
    fuse     a segment is kept when a second detector (or the other DeepLSD
             scale) saw it too, or when DeepLSD saw it and the photo has a
             strong gradient across it
    merge    collinear pieces (within MERGE_DEG, end points within MERGE_OFFSET
             of each other's line, gap up to MERGE_GAP of the diagonal) become one
    drop     shorter than MIN_LEN of the diagonal; weak gradient; on an object;
             on a curtain, window glass or mirror (SegFormer labels); inside the
             floor mask and near-vertical in the image (a floor reflection).
             Leaning lines are never dropped for not being vertical.
    snap     wall boundary pixels within SNAP_PX of a fused line that the photo
             supports: reported only (which boundaries would snap), no pixel moves

Every length is a share of the image diagonal; the gradient threshold is a
multiple of the photo's own median gradient.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "backups" / "deeplsd-cache"
GEO_PYTHON = Path.home() / "geocalib-env" / "bin" / "python"
RUNNER = PROJECT_ROOT / "backend" / "deeplsd_runner.py"

MATCH_DEG = 2.0          # two detections of one line: within this angle ...
MATCH_PX = 2.0           # ... and this perpendicular distance, overlapping
MERGE_DEG = 1.0
MERGE_OFFSET = 2.0       # px
MERGE_GAP = 0.03         # share of the diagonal
MIN_LEN = 0.03           # share of the diagonal
STRONG = 2.0             # x the photo's median gradient (over textured pixels)
ON_SHARE = 0.5           # share of a line's samples inside an object / curtain / glass
OBJECT_GROW = 0.002      # objects grown by this share of the diagonal (min 2 px): a line along an outline is on it
FLOOR_SHARE = 0.8        # share inside the floor mask ...
REFLECTION_DEG = 15.0    # ... and within this angle of image vertical: a floor reflection
SNAP_PX = 4.0
GLASS_LABELS = ("curtain", "blind", "windowpane", "glass", "mirror", "screen door")


def _deeplsd(img: np.ndarray, scale: float) -> tuple[np.ndarray, bool]:
    """DeepLSD lines of `img` at `scale`, in full-image pixels; (lines, from_cache)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.md5(np.ascontiguousarray(img).tobytes()).hexdigest()
    path = CACHE_DIR / f"{key}_s{scale:g}.json"
    if path.exists():
        return np.array(json.loads(path.read_text()), float).reshape(-1, 4), True
    work = img if scale == 1 else cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    with tempfile.TemporaryDirectory() as tmp:
        src, out = Path(tmp) / "room.png", Path(tmp) / "lines.json"
        cv2.imwrite(str(src), cv2.cvtColor(work, cv2.COLOR_RGB2BGR))
        subprocess.run([str(GEO_PYTHON), str(RUNNER), str(src), str(out)], check=True, capture_output=True)
        lines = np.array(json.loads(out.read_text())["lines"], float).reshape(-1, 4) / scale
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(lines.tolist()))
    tmp_path.replace(path)
    return lines, False


def _lsd(gray: np.ndarray) -> np.ndarray:
    out = cv2.createLineSegmentDetector(0).detect(gray)[0]
    return np.zeros((0, 4)) if out is None else out.reshape(-1, 4).astype(float)


def _unit(s):
    d = s[2:] - s[:2]
    n = float(np.hypot(*d))
    return d / n if n > 1e-9 else np.array([1.0, 0.0]), n


def _angle_diff(a, b):
    return math.degrees(math.acos(min(1.0, abs(float(a @ b)))))


def _samples(s, n=None):
    d, L = _unit(s)
    k = int(max(2, n or L))
    t = np.linspace(0, 1, k)
    return s[:2][None] + t[:, None] * (s[2:] - s[:2])[None]


def _gradient_across(s, gx, gy):
    h, w = gx.shape
    d, _ = _unit(s)
    nrm = np.array([-d[1], d[0]])
    p = _samples(s)
    best = np.zeros(len(p))
    for k in (-1, 0, 1):
        q = p + k * nrm
        x = np.clip(np.round(q[:, 0]).astype(int), 0, w - 1)
        y = np.clip(np.round(q[:, 1]).astype(int), 0, h - 1)
        best = np.maximum(best, np.abs(gx[y, x] * nrm[0] + gy[y, x] * nrm[1]))
    return float(best.mean())


def _same_line(a, b, deg, px):
    """b lies on a's line (angle, both end points within px) and their spans overlap."""
    da, La = _unit(a)
    db, _ = _unit(b)
    if La < 1 or _angle_diff(da, db) > deg:
        return False
    nrm = np.array([-da[1], da[0]])
    if max(abs(float((b[:2] - a[:2]) @ nrm)), abs(float((b[2:] - a[:2]) @ nrm))) > px:
        return False
    u = sorted([float((b[:2] - a[:2]) @ da), float((b[2:] - a[:2]) @ da)])
    return u[1] >= 0 and u[0] <= La


def _merge(segs, diag):
    """Collinear pieces -> one segment (union-find, extreme end points on their joint line)."""
    n = len(segs)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    gap = MERGE_GAP * diag
    for i in range(n):
        di, Li = _unit(segs[i])
        nrm = np.array([-di[1], di[0]])
        for j in range(i + 1, n):
            dj, _ = _unit(segs[j])
            if _angle_diff(di, dj) > MERGE_DEG:
                continue
            if max(abs(float((segs[j][:2] - segs[i][:2]) @ nrm)), abs(float((segs[j][2:] - segs[i][:2]) @ nrm))) > MERGE_OFFSET:
                continue
            u = sorted([float((segs[j][:2] - segs[i][:2]) @ di), float((segs[j][2:] - segs[i][:2]) @ di)])
            if u[0] > Li + gap or u[1] < -gap:
                continue
            parent[find(j)] = find(i)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    out, members = [], []
    for g in groups.values():
        pts = np.vstack([segs[g][:, :2], segs[g][:, 2:]])
        c = pts.mean(0)
        d = np.linalg.svd(pts - c)[2][0]
        u = (pts - c) @ d
        out.append(np.r_[c + u.min() * d, c + u.max() * d])
        members.append(g)
    return np.array(out).reshape(-1, 4), members


def strong_lines(room: np.ndarray, *, props: np.ndarray | None = None, floor: np.ndarray | None = None,
                 labels: tuple[np.ndarray, dict] | None = None) -> dict:
    """
    The photo's strong straight lines and a report of every line dropped.

    `room` RGB at render resolution; `props` the object mask, `floor` FLOOR_MASK
    at that resolution; `labels` (class map, id -> name) from surfaces.label_map
    for curtains / glass. Returns {"lines": (N, 4) array, "report": {...}}.
    """
    h, w = room.shape[:2]
    diag = math.hypot(h, w)
    gray = cv2.cvtColor(room, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.0)
    gx, gy = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    strong = STRONG * float(np.median(mag[mag > 1e-3])) if (mag > 1e-3).any() else 0.0

    d_full, c1 = _deeplsd(room, 1.0)
    d_half, c2 = _deeplsd(room, 0.5)
    l_cv = _lsd(gray)
    pools = {"deeplsd_full": d_full, "deeplsd_half": d_half, "lsd": l_cv}
    segs = np.vstack([p for p in pools.values() if len(p)]) if any(len(p) for p in pools.values()) else np.zeros((0, 4))
    src = np.concatenate([[k] * len(p) for k, p in pools.items()]) if len(segs) else np.array([])

    # fuse: seen by a second detector / scale, or DeepLSD with a strong gradient
    grad = np.array([_gradient_across(s, gx, gy) for s in segs])
    keep = np.zeros(len(segs), bool)
    seen_by_two = np.zeros(len(segs), bool)
    for i, s in enumerate(segs):
        for k, pool in pools.items():
            if k == src[i] or not len(pool):
                continue
            if any(_same_line(s, t, MATCH_DEG, MATCH_PX) or _same_line(t, s, MATCH_DEG, MATCH_PX) for t in pool):
                seen_by_two[i] = True
                break
        keep[i] = seen_by_two[i] or (src[i].startswith("deeplsd") and grad[i] >= strong)
    fused_in = segs[keep]
    merged, members = _merge(fused_in, diag)

    # drop
    lab, names = labels if labels is not None else (None, {})
    glass_ids = [k for k, v in names.items() if v in GLASS_LABELS]
    if props is not None:
        r = max(2, int(round(OBJECT_GROW * diag)))
        props = cv2.dilate(props.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))).astype(bool)
    reasons = []
    for s in merged:
        d, L = _unit(s)
        p = _samples(s)
        x = np.clip(np.round(p[:, 0]).astype(int), 0, w - 1)
        y = np.clip(np.round(p[:, 1]).astype(int), 0, h - 1)
        if L < MIN_LEN * diag:
            reasons.append("shorter than 3% of the diagonal")
        elif _gradient_across(s, gx, gy) < strong:
            reasons.append("weak gradient")
        elif props is not None and props[y, x].mean() >= ON_SHARE:
            reasons.append("on an object")
        elif lab is not None and np.isin(lab[y, x], glass_ids).mean() >= ON_SHARE:
            reasons.append("on curtain / glass / mirror")
        elif floor is not None and floor[y, x].mean() >= FLOOR_SHARE and _angle_diff(d, np.array([0.0, 1.0])) <= REFLECTION_DEG:
            reasons.append("floor reflection (near-vertical inside the floor)")
        else:
            reasons.append("")
    reasons = np.array(reasons)
    kept = merged[reasons == ""]
    report = {
        "detected": {k: int(len(p)) for k, p in pools.items()},
        "deeplsd_from_cache": bool(c1 and c2),
        "strong_gradient_threshold": round(strong, 2),
        "fused": {"seen_by_two_detectors": int(seen_by_two.sum()),
                  "deeplsd_strong_gradient_only": int((keep & ~seen_by_two).sum()),
                  "not_kept": int((~keep).sum())},
        "after_collinear_merge": int(len(merged)),
        "dropped": {r: int((reasons == r).sum()) for r in sorted(set(reasons)) if r},
        "kept": int(len(kept)),
    }
    return {"lines": kept, "all_merged": merged, "drop_reason": reasons, "report": report,
            "gradient": (gx, gy), "strong": strong}


def snap_candidates(regions: dict, lines: np.ndarray, gradient, strong: float) -> dict:
    """
    Wall boundary stretches lying within SNAP_PX of a fused line that the photo
    supports (strong gradient along it). Reported only: no pixel is changed.
    """
    gx, gy = gradient
    out = {}
    for key, region in regions.items():
        if not key.startswith("wall-"):
            continue
        edge = region & ~cv2.erode(region.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        ey, ex = np.nonzero(edge)
        if not len(ex):
            continue
        pts = np.stack([ex, ey], 1).astype(float)
        rows = []
        for s in lines:
            d, L = _unit(s)
            if L < 1:
                continue
            nrm = np.array([-d[1], d[0]])
            u = (pts - s[:2]) @ d
            v = np.abs((pts - s[:2]) @ nrm)
            near = (u >= 0) & (u <= L) & (v <= SNAP_PX)
            if near.sum() >= 0.5 * L and _gradient_across(s, gx, gy) >= strong:
                rows.append({"line": [round(float(x), 1) for x in s], "boundary_px": int(near.sum()),
                             "mean_offset_px": round(float(v[near].mean()), 2),
                             "px_off_line_gt_0_5": int((v[near] > 0.5).sum())})
        out[key] = {"snappable_stretches": len(rows), "boundary_px_near_lines": int(sum(r["boundary_px"] for r in rows)),
                    "stretches": rows}
    return out
