"""
L1 report: strong lines per room (backend/perspective_engine/strong_lines.py). No pixel changes.

    backend/.venv/bin/python tests/regression/l1_lines.py <out_dir> <baseline_dir> <room>

Per room (job copied first): the module's fused / merged / kept lines and the
drop reasons; lines per family BEFORE (DeepLSD full + LSD, 2 % length, as the
diagnosis did) and AFTER (the L1 kept lines), each clustered the same way
(vertical: within 35 deg of vertical meeting at one VP outside the image rows,
leaning allowed; then horizontal-A and horizontal-B by RANSAC); the wall
boundaries that would snap (regions from the baseline render's .npz).
Writes <room>__l1.png (kept lines by family, dropped lines grey) and <room>__l1.json.
"""

from __future__ import annotations

import json
import math
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
import surfaces  # noqa: E402
from baseline_rooms import ROOMS  # noqa: E402
from camera_evidence import lsd, seg_angle  # noqa: E402
from perspective_engine import masks, strong_lines as sl  # noqa: E402
from wall_direction_diagnose import fit_vp, vp_image  # noqa: E402

COL = {"vertical": (0, 200, 255), "horizontal-A": (255, 70, 70), "horizontal-B": (60, 220, 60), "unassigned": (255, 255, 255)}


def families(segs, w, h):
    rng = np.random.default_rng(0)
    fam = np.array(["unassigned"] * len(segs), object)
    if not len(segs):
        return fam, {}
    vps = {}
    cand = np.nonzero(np.abs(np.abs([seg_angle(s) for s in segs]) - 90) <= 35)[0]
    for _ in range(3):
        vvp, vinl = fit_vp(segs[cand], w, h, rng) if len(cand) >= 2 else (None, None)
        if vvp is None:
            break
        V = vp_image(vvp)
        if V[0] == "vp" and 0 <= V[1][1] <= h:
            cand = cand[~vinl]
            continue
        vps["vertical"] = V
        fam[cand[vinl]] = "vertical"
        break
    for fname in ("horizontal-A", "horizontal-B"):
        rest = np.nonzero(fam == "unassigned")[0]
        if len(rest) < 2:
            break
        hvp, hinl = fit_vp(segs[rest], w, h, rng)
        if hvp is None or hinl.sum() < 2:
            break
        vps[fname] = vp_image(hvp)
        fam[rest[hinl]] = fname
    return fam, vps


SUPPORT_SHARE = 0.9       # a line is gradient-supported when this share of its length is strong
FURNITURE_PX = 10         # a kept line this close to an object is "on furniture"
BAND = 6                  # px band for boundary / label tests
OPENINGS = ("door", "windowpane", "window", "double door", "screen door")


def line_quality(lines, fam, gradient, strong, props, floor, wall, labels, regions, w, h):
    """Share of kept lines with strong gradient along their whole length; where they lie
    (room structure vs furniture); families with < 3 lines or < 30 % width coverage."""
    gx, gy = gradient
    lab, names = labels
    open_ids = [k for k, v in names.items() if v in OPENINGS]
    k = np.ones((2 * BAND + 1, 2 * BAND + 1), np.uint8)
    near_obj = cv2.dilate(props.astype(np.uint8), np.ones((2 * FURNITURE_PX + 1,) * 2, np.uint8)).astype(bool)
    ceil = ~wall & ~floor & ~props
    ceil_line = cv2.dilate(ceil.astype(np.uint8), k).astype(bool) & cv2.dilate(wall.astype(np.uint8), k).astype(bool)
    floor_line = cv2.dilate(floor.astype(np.uint8), k).astype(bool) & cv2.dilate(wall.astype(np.uint8), k).astype(bool)
    walls = [m for key, m in regions.items() if key.startswith("wall-")]
    corner = np.zeros((h, w), bool)
    for i in range(len(walls)):
        for j in range(i + 1, len(walls)):
            corner |= cv2.dilate(walls[i].astype(np.uint8), k).astype(bool) & cv2.dilate(walls[j].astype(np.uint8), k).astype(bool)
    opening = cv2.dilate(np.isin(lab, open_ids).astype(np.uint8), k).astype(bool)
    supported, where = [], []
    for s in lines:
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
        supported.append(float((best >= strong).mean()) >= SUPPORT_SHARE)
        x = np.clip(np.round(p[:, 0]).astype(int), 0, w - 1)
        y = np.clip(np.round(p[:, 1]).astype(int), 0, h - 1)
        share = lambda m: float(m[y, x].mean())
        if share(near_obj) >= 0.5:
            where.append("furniture")
        elif share(corner) >= 0.5:
            where.append("corner")
        elif share(ceil_line) >= 0.5:
            where.append("ceiling line")
        elif share(floor_line) >= 0.5:
            where.append("floor line")
        elif share(opening) >= 0.5:
            where.append("window / door")
        elif share(floor) >= 0.5:
            where.append("floor surface (seams, pattern)")
        elif share(wall) >= 0.5:
            where.append("wall surface (tile joints, frames)")
        else:
            where.append("other (ceiling surface, outside masks)")
    where = np.array(where)
    supported = np.array(supported, bool)
    n = max(len(lines), 1)
    structure = {"ceiling line", "floor line", "corner", "window / door"}
    fams = {}
    insufficient = []
    for f in ("vertical", "horizontal-A", "horizontal-B"):
        sel = fam == f
        cnt = int(sel.sum())
        if cnt:
            xs = np.sort(np.stack([np.minimum(lines[sel, 0], lines[sel, 2]), np.maximum(lines[sel, 0], lines[sel, 2])], 1), axis=0)
            cover = np.zeros(w, bool)
            for x0, x1 in np.stack([np.minimum(lines[sel, 0], lines[sel, 2]), np.maximum(lines[sel, 0], lines[sel, 2])], 1):
                cover[int(max(0, x0)):int(min(w, x1)) + 1] = True
            span = float(np.ptp(np.r_[lines[sel, 0], lines[sel, 2]])) / w
            coverage = float(cover.mean()) if f != "vertical" else span
        else:
            coverage = 0.0
        fams[f] = {"count": cnt, "width_coverage": round(coverage, 3),
                   "supported_share": round(float(supported[sel].mean()), 3) if cnt else None}
        if cnt < 3 or coverage < 0.3:
            insufficient.append(f"{f}: {cnt} lines, {100 * coverage:.0f}% of the width")
    return {"kept": int(len(lines)),
            "gradient_supported_whole_length_share": round(float(supported.mean()), 3) if len(lines) else None,
            "where": {k_: round(float((where == k_).sum()) / n, 3) for k_ in sorted(set(where))},
            "room_structure_share": round(float(np.isin(where, list(structure)).sum()) / n, 3),
            "furniture_share": round(float((where == "furniture").sum()) / n, 3),
            "families": fams,
            "verdict": "insufficient lines -> needs_fix at L2" if insufficient else "sufficient",
            "insufficient": insufficient}


def main() -> int:
    out, base, name = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        job = Path(tmp) / ROOMS[name].name
        shutil.copytree(ROOMS[name], job)
        b = reuse.load(job)
        fl, wl = pe.load(job / "segments")
    h, w = b.room.shape[:2]
    diag = math.hypot(h, w)
    floor, wall, _ = masks.prepare(fl, wl, (h, w))
    props = np.asarray(b.props, bool)
    labels = surfaces.label_map(b.room)
    res = sl.strong_lines(b.room, props=props, floor=floor, labels=labels)
    # before: DeepLSD full + LSD (duplicates kept out), 2 % of the diagonal -- the diagnosis' set
    d_full, _ = sl._deeplsd(b.room, 1.0)
    before = np.vstack([d_full, lsd(cv2.cvtColor(b.room, cv2.COLOR_RGB2GRAY))])
    before = before[np.hypot(before[:, 2] - before[:, 0], before[:, 3] - before[:, 1]) >= 0.02 * diag]
    fb, _ = families(before, w, h)
    fa, vps_a = families(res["lines"], w, h)
    regions = dict(np.load(base / f"{name}.npz")) if (base / f"{name}.npz").exists() else {}
    quality = line_quality(res["lines"], fa, res["gradient"], res["strong"], props, floor, wall, labels, regions, w, h)
    snaps = sl.snap_candidates(regions, res["lines"], res["gradient"], res["strong"])
    rep = {"room": name, "size": [w, h], "module_report": res["report"],
           "families_before": {k: int((fb == k).sum()) for k in COL},
           "families_after": {k: int((fa == k).sum()) for k in COL},
           "vps_after": {k: ([round(float(x), 1) for x in v[1]] if v[0] == "vp" else "at infinity") for k, v in vps_a.items()},
           "quality": quality,
           "snap_report": {k: {kk: vv for kk, vv in v.items() if kk != "stretches"} for k, v in snaps.items()},
           "snap_details": snaps}
    img = (b.room.astype(np.float32) * 0.5).astype(np.uint8)
    for s, r in zip(res["all_merged"], res["drop_reason"]):
        if r:
            cv2.line(img, (int(s[0]), int(s[1])), (int(s[2]), int(s[3])), (110, 110, 110), 1, cv2.LINE_AA)
    for s, f in zip(res["lines"], fa):
        cv2.line(img, (int(s[0]), int(s[1])), (int(s[2]), int(s[3])), COL[f], 2, cv2.LINE_AA)
    for k, v in snaps.items():
        for st in v["stretches"]:
            l_ = st["line"]
            cv2.line(img, (int(l_[0]), int(l_[1])), (int(l_[2]), int(l_[3])), (255, 0, 255), 3, cv2.LINE_AA)
    Image.fromarray(img).save(out / f"{name}__l1.png")
    (out / f"{name}__l1.json").write_text(json.dumps(rep, indent=1))
    print(name, json.dumps({k: rep[k] for k in ("module_report", "families_before", "families_after", "quality", "snap_report")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
