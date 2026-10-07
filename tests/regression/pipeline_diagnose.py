"""
Lines -> VPs -> camera -> scale -> edges -> walls, DIAGNOSIS ONLY (no code or pixel change).

    backend/.venv/bin/python tests/regression/pipeline_diagnose.py <out_dir> <room>

One room per run (never in parallel). The job is copied first; nothing is written
back. DeepLSD (~/geocalib-env, backend/deeplsd_runner.py) runs as a subprocess and
its lines are cached per photo in backups/deeplsd-cache/<md5>.json.

A lines   DeepLSD (primary) + OpenCV LSD (secondary; an LSD segment repeating a
          DeepLSD one is a duplicate). Dropped: shorter than 2 % of the diagonal,
          midpoint on an object, not within 2 deg of a family. Families: vertical
          (segments within 35 deg of vertical meeting at one VP, finite allowed --
          leaning lines of a pitched / wide-angle photo are kept), horizontal-A,
          horizontal-B (RANSAC). The current pipeline's own line evidence is read
          from the stored geometry (room frame vertical lines, floor VP lines, each
          wall's ceiling-line / floor-junction fits).
B VPs     per family: VP, residual median in px (end-point offset) and degrees;
          horizon; principal point (image centre); vertical VP finite or not
C camera  focal from well-conditioned VP pairs (both within 3 diagonals),
          pitch / roll from the vertical VP, pitch from the horizon, yaw,
          orthogonality error; EXIF focal; the stored prior / joint camera
D scale   per rendered surface (its own 3D record): mm per pixel (sqrt of the
          pixel's footprint) at the 10th / 50th / 90th depth percentile of its
          pixels; the wall's scale / plane source; and at each wall's floor
          junction (pixels where wall and floor regions meet -- the same physical
          points on both) the ratio of the wall's mm per pixel to the floor's along
          the junction. 1.0 = one real tile size.
E edges   per wall: top and bottom boundary -- fitted line (Step 2 removed pixels
          past it) or raw mask contour; RMS to the best line and 1-px step count;
          pixels of the wall past a photo corner line (a long vertical segment near
          the cut, extended to the vertical VP) on the neighbour's side
F walls   planes and dots (dot inside the photo and inside its own region); per
          cut: vertical photo edge length near it (evidence of a corner); pairs of
          pieces with normals within 3 deg -- same plane (offsets within 5 %) or
          parallel faces (a pillar, a recess)
Writes <out_dir>/<room>__lines.png and <room>__camera.json.
"""

from __future__ import annotations

import hashlib
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
from baseline_rooms import ROOMS, TILE, TILE_W_MM, TILE_H_MM, GROUT_MM  # noqa: E402
from camera_evidence import ang, dir_at, label, lsd, proj, ray, residual_px, seg_angle  # noqa: E402
from engine import TileSpec  # noqa: E402
from perspective_engine import masks  # noqa: E402
from wall_direction_diagnose import deeplsd, fit_vp, vp_image  # noqa: E402

ROOMS = dict(ROOMS, kitchen=ROOT / "backend/jobs/cf6f7831d6de", pink_frontal=ROOT / "backend/jobs/63705139a5ee")
CACHE = ROOT / "backups/deeplsd-cache"
MIN_LEN, VERT_CAND_DEG, COND = 0.02, 35.0, 3.0
COL = {"vertical": (0, 200, 255), "horizontal-A": (255, 70, 70), "horizontal-B": (60, 220, 60),
       "dropped: not within 2 deg of a family": (130, 130, 130), "dropped: on an object": (200, 80, 255),
       "dropped: shorter than 2% of the diagonal": (90, 90, 90)}


def cached_deeplsd(img):
    CACHE.mkdir(parents=True, exist_ok=True)
    key = hashlib.md5(np.ascontiguousarray(img).tobytes()).hexdigest()
    path = CACHE / f"{key}.json"
    if path.exists():
        return np.array(json.loads(path.read_text()), float).reshape(-1, 4), True
    lines = deeplsd(img)
    path.write_text(json.dumps(lines.tolist()))
    return lines, False


def backproject(th, xs, ys):
    """3D points (units) on the surface's plane for pixels (xs, ys), via its own camera."""
    c, pl = th["camera"], th["plane"]
    f, cx, cy = c["focal_px"], c["cx"], c["cy"]
    n, d = np.array(pl["normal"]), pl["d_units"]
    rays = np.stack([(xs - cx) / f, (ys - cy) / f, np.ones_like(xs, float)], -1)
    t = -d / (rays @ n)
    return rays * t[..., None]


def mm_per_px(th, xs, ys):
    X = backproject(th, xs.astype(float), ys.astype(float))
    Xu = backproject(th, xs + 1.0, ys.astype(float))
    Xv = backproject(th, xs.astype(float), ys + 1.0)
    area = np.linalg.norm(np.cross(Xu - X, Xv - X), axis=-1)
    return np.sqrt(area) * th["grid"]["mm_per_unit"], X[..., 2]


def mm_along(th, xs, ys, t):
    X = backproject(th, xs.astype(float), ys.astype(float))
    X2 = backproject(th, xs + t[0], ys + t[1])
    return np.linalg.norm(X2 - X, axis=-1) * th["grid"]["mm_per_unit"]


def boundary_fit(points, diag=2000.0):
    """RANSAC line through the boundary points; RMS / steps on its inliers (pieces of
    the boundary interrupted by furniture are other lines, reported as the outlier share)."""
    if len(points) < 10:
        return None
    allp = np.array(points, float)
    rng = np.random.default_rng(0)
    thr = max(2.0, 0.003 * diag)
    best = None
    for _ in range(300):
        i, j = rng.choice(len(allp), 2, replace=False)
        dd = allp[j] - allp[i]
        if np.hypot(*dd) < 1e-6:
            continue
        nn = np.array([-dd[1], dd[0]]) / np.hypot(*dd)
        inl = np.abs((allp - allp[i]) @ nn) < thr
        if best is None or inl.sum() > best.sum():
            best = inl
    pts = allp[best]
    c = pts.mean(0)
    nrm = np.linalg.svd(pts - c)[2][1]
    tang = np.array([nrm[1], -nrm[0]])
    d = (pts - c) @ nrm
    u = np.round((pts - c) @ tang).astype(int)
    order = np.unique(u)
    prof = np.array([d[u == k].mean() for k in order])
    jumps = np.abs(np.diff(prof))
    gaps = np.diff(order) == 1
    d_all = (allp - c) @ nrm
    return {"inlier_share": round(float(best.mean()), 3), "rms_all_points_px": round(float(np.sqrt((d_all ** 2).mean())), 2),
            "rms_px": round(float(np.sqrt((d ** 2).mean())), 2), "steps_1px": int(((jumps >= 1) & gaps).sum()),
            "steps_2px": int(((jumps >= 2) & gaps).sum()), "length_px": int(np.ptp(order) + 1)}


def main() -> int:
    out, name = Path(sys.argv[1]), sys.argv[2]
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    spec = TileSpec(artwork=np.asarray(Image.open(TILE).convert("RGB")), width_mm=TILE_W_MM,
                    height_mm=TILE_H_MM, rotation_deg=0, grout_mm=GROUT_MM)
    src_job = ROOMS[name]
    with tempfile.TemporaryDirectory() as tmp:
        job = Path(tmp) / src_job.name
        shutil.copytree(src_job, job)
        b = reuse.load(job)
        fl, wl = pe.load(job / "segments")
        geo = pe.ensure_geometry(job / "segments", b.room, fl, wl, clean=b.clean)
        stored = json.loads((job / "segments/wall/wall_geometry.json").read_text())
        frame = json.loads((job / "segments/room/room_frame.json").read_text()) if (job / "segments/room/room_frame.json").exists() else {}
        floor_doc = json.loads((job / "segments/floor/floor_geometry.json").read_text()) if (job / "segments/floor/floor_geometry.json").exists() else {}
        surf, regions = {}, {}
        for surface, idx in [("floor", None)] + [("wall", int(w["index"])) for w in geo["walls"]]:
            key = "floor" if idx is None else f"wall-{idx}"
            try:
                r = pe.render_tiled_room(b.room, fl, wl, spec, surface, (None, None, None), clean=b.clean,
                                         props=b.props, wall_index=idx, geometry=geo)
            except Exception as e:  # noqa: BLE001
                surf[key] = {"refused": str(e)[:120]}
                continue
            if key in r.three and key in r.regions:
                entry = next((e for e in (r.geometry.get("wall") or {}).get("walls", []) if e.get("index") == idx), {})
                surf[key] = {"three": r.three[key], "geometry": r.geometry, "entry": entry}
                regions[key] = r.regions[key]
            else:
                surf[key] = {"refused": "no 3D record"}
        dots = pe.describe(regions)
    h, w = b.room.shape[:2]
    diag = math.hypot(h, w)
    cx, cy = w / 2.0, h / 2.0
    props = np.asarray(b.props, bool)
    floor_m, wall_m, _ = masks.prepare(fl, wl, (h, w))

    # ---- A lines
    deep, cached = cached_deeplsd(b.room)
    lsd_raw = lsd(cv2.cvtColor(b.room, cv2.COLOR_RGB2GRAY))
    keep = []
    for s in lsd_raw:
        mid = (s[:2] + s[2:]) / 2
        dup = False
        for t in deep:
            d = t[2:] - t[:2]
            L = float(np.hypot(*d))
            if L < 1:
                continue
            nrm = np.array([-d[1], d[0]]) / L
            u = float((mid - t[:2]) @ (d / L))
            if abs(float((mid - t[:2]) @ nrm)) <= 3 and abs((seg_angle(s) - seg_angle(t) + 90) % 180 - 90) <= 2 and -3 <= u <= L + 3:
                dup = True
                break
        if not dup:
            keep.append(s)
    segs = np.vstack([deep, np.array(keep).reshape(-1, 4)]) if keep else deep
    src = np.array(["deeplsd"] * len(deep) + ["lsd"] * len(keep))
    lens = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    mids = ((segs[:, :2] + segs[:, 2:]) / 2).astype(int)
    on_obj = props[np.clip(mids[:, 1], 0, h - 1), np.clip(mids[:, 0], 0, w - 1)]
    status = np.array(["candidate"] * len(segs), object)
    status[lens < MIN_LEN * diag] = "dropped: shorter than 2% of the diagonal"
    status[(status == "candidate") & on_obj] = "dropped: on an object"
    vps = {}
    cand = np.nonzero((status == "candidate") & (np.abs(np.abs([seg_angle(s) for s in segs]) - 90) <= VERT_CAND_DEG))[0]
    for _ in range(3):
        vvp, vinl = fit_vp(segs[cand], w, h, rng) if len(cand) >= 2 else (None, None)
        if vvp is None:
            break
        V = vp_image(vvp)
        if V[0] == "vp" and 0 <= V[1][1] <= h:
            cand = cand[~vinl]
            continue
        vps["vertical"] = V
        status[cand[vinl]] = "vertical"
        break
    for fname in ("horizontal-A", "horizontal-B"):
        rest = np.nonzero(status == "candidate")[0]
        if len(rest) < 2:
            break
        hvp, hinl = fit_vp(segs[rest], w, h, rng)
        if hvp is None or hinl.sum() < 2:
            break
        vps[fname] = vp_image(hvp)
        status[rest[hinl]] = fname
    status[status == "candidate"] = "dropped: not within 2 deg of a family"
    fams = {}
    for fname in ("vertical", "horizontal-A", "horizontal-B"):
        sel = status == fname
        row = {"count": int(sel.sum()), "deeplsd": int((sel & (src == "deeplsd")).sum()), "lsd": int((sel & (src == "lsd")).sum())}
        if fname in vps:
            res_px = residual_px(segs[sel], vps[fname])
            res_deg = [math.degrees(math.acos(min(1.0, abs(float(dir_at(vps[fname], (s[:2] + s[2:]) / 2) @ ((s[2:] - s[:2]) / np.hypot(*(s[2:] - s[:2])))))))) for s in segs[sel]]
            row.update({"vp": label(vps[fname]), "finite": vps[fname][0] == "vp" and math.hypot(*(vps[fname][1] - [cx, cy])) <= 10 * diag,
                        "residual_px_median": round(float(np.median(res_px)), 2), "residual_deg_median": round(float(np.median(res_deg)), 3)})
        fams[fname] = row
    lines_rep = {"deeplsd_raw": int(len(deep)), "deeplsd_cached": cached, "lsd_raw": int(len(lsd_raw)),
                 "lsd_duplicates_removed": int(len(lsd_raw) - len(keep)), "fused": int(len(segs)),
                 "dropped": {k: int((status == k).sum()) for k in COL if k.startswith("dropped")},
                 "kept": int(sum(fams[k]["count"] for k in fams)), "families": fams,
                 "current_pipeline_uses": {
                     "room_frame_vertical_lines": {k: (frame.get("vertical_vp") or {}).get(k) for k in ("lines", "inliers", "reliable")},
                     "floor_vp": {k: floor_doc.get(k) for k in ("status", "vp_source", "lines_used", "lines_detected") if k in floor_doc},
                     "walls": {wd["id"]: {k: {kk: f.get(kk) for kk in ("inliers", "points", "inlier_share")} for k, f in ((wd.get("direction") or {}).get("fit") or {}).items()}
                               for wd in stored.get("walls", [])}}}

    # ---- C camera
    pairs = {}
    for a, bb in (("horizontal-A", "horizontal-B"), ("horizontal-A", "vertical"), ("horizontal-B", "vertical")):
        if a in vps and bb in vps and vps[a][0] == "vp" and vps[bb][0] == "vp":
            if max(np.hypot(*(vps[a][1] - [cx, cy])), np.hypot(*(vps[bb][1] - [cx, cy]))) > COND * diag:
                pairs[f"{a}|{bb}"] = "undetermined (a VP is near infinity)"
                continue
            f2 = -float(np.dot(vps[a][1] - [cx, cy], vps[bb][1] - [cx, cy]))
            pairs[f"{a}|{bb}"] = round(math.sqrt(f2), 1) if f2 > 0 else "none (same side)"
    fvals = [v for v in pairs.values() if isinstance(v, float)]
    f_store = float(stored.get("focal_px"))
    f = float(np.mean(fvals)) if fvals else f_store
    cam = {"principal_point": [cx, cy], "focal_from_vp_pairs": pairs, "focal_px": round(f, 1),
           "focal_source": "VP pairs" if fvals else "not determinable from lines -> stored prior",
           "current_prior": {"focal_px": round(f_store, 1), "focal_source": stored.get("focal_source"),
                             "confidence": stored.get("confidence"), "joint": stored.get("joint")},
           "exif_focal_35mm": None,
           "orthogonality_error_deg": {f"{a}|{bb}": round(abs(90 - ang(ray(vps[a], f, cx, cy), ray(vps[bb], f, cx, cy))), 2)
                                       for a, bb in (("horizontal-A", "horizontal-B"), ("horizontal-A", "vertical"), ("horizontal-B", "vertical"))
                                       if a in vps and bb in vps}}
    try:
        ex = Image.open(src_job / "original.png").getexif()
        cam["exif_focal_35mm"] = ex.get_ifd(0x8769).get(0xA405) if ex else None
    except Exception:  # noqa: BLE001
        pass
    if "vertical" in vps:
        rv = ray(vps["vertical"], f, cx, cy)
        rv = -rv if rv[1] < 0 else rv
        cam["pitch_from_vertical_deg"] = round(math.degrees(math.asin(max(-1, min(1, rv[2])))), 2)
        cam["roll_deg"] = round(math.degrees(math.atan2(rv[0], rv[1])), 2)
    hz = None
    hf = [k for k in ("horizontal-A", "horizontal-B") if k in vps and vps[k][0] == "vp"]
    if len(hf) == 2:
        hz = (vps[hf[0]][1], vps[hf[1]][1])
    elif hf:
        A = vps[hf[0]][1]
        t = np.array([1.0, 0.0])
        if "vertical" in vps and vps["vertical"][0] == "vp":
            dv = vps["vertical"][1] - [cx, cy]
            t = np.array([-dv[1], dv[0]])
        hz = (A, A + t)
    if hz is not None:
        p, q = hz
        y_c = p[1] + (q[1] - p[1]) * (cx - p[0]) / (q[0] - p[0]) if abs(q[0] - p[0]) > 1e-6 else p[1]
        cam["horizon_y_at_centre"] = round(float(y_c), 1)
        cam["pitch_from_horizon_deg"] = round(math.degrees(math.atan2(cy - y_c, f)), 2)
    depth = min(hf, key=lambda k: np.hypot(*(vps[k][1] - [cx, cy]))) if hf else None
    if depth:
        cam["yaw_deg"] = round(math.degrees(math.atan2(vps[depth][1][0] - cx, f)), 2)
    if "pitch_from_vertical_deg" in cam and "pitch_from_horizon_deg" in cam:
        cam["pitch_agreement_deg"] = round(abs(cam["pitch_from_vertical_deg"] - cam["pitch_from_horizon_deg"]), 2)

    # ---- D scale
    scale = {}
    for key, sd in surf.items():
        if "refused" in sd:
            scale[key] = {"refused": sd["refused"]}
            continue
        ys, xs = np.nonzero(regions[key])
        pick = rng.choice(len(xs), min(4000, len(xs)), replace=False)
        mpp, Z = mm_per_px(sd["three"], xs[pick], ys[pick])
        ok = np.isfinite(mpp) & np.isfinite(Z) & (Z > 0)
        qs = np.percentile(Z[ok], [10, 50, 90]) if ok.any() else [np.nan] * 3
        near_mid_far = [round(float(np.median(mpp[ok][np.abs(Z[ok] - q) <= 0.05 * abs(q) + 1e-9])), 2) if ok.any() else None for q in qs]
        g = sd["geometry"]
        row = {"mm_per_px_near_mid_far": near_mid_far,
               "scale_source": (g.get("floor") or {}).get("scale_source") if key == "floor" else (g.get("wall") or {}).get("scale_source"),
               "plane_source": None if key == "floor" else sd["entry"].get("plane_source"),
               "mm_per_unit": round(sd["three"]["grid"]["mm_per_unit"], 4),
               "camera": sd["three"]["camera"]["focal_px"]}
        if key != "floor" and "floor" in regions and "refused" not in surf["floor"]:
            wr, frr = regions[key], regions["floor"]
            below = np.zeros_like(wr)
            below[:-2] = frr[2:]
            jy, jx = np.nonzero(wr & below)
            if len(jx) >= 10:
                c0 = np.stack([jx, jy], 1).astype(float)
                tdir = np.linalg.svd(c0 - c0.mean(0))[2][0]
                wmm = mm_along(sd["three"], jx, jy, tdir)
                fmm = mm_along(surf["floor"]["three"], jx, jy + 2, tdir)
                rr = wmm / fmm
                rr = rr[np.isfinite(rr)]
                row["junction_points"] = int(len(jx))
                row["tile_ratio_wall_to_floor_at_junction"] = round(float(np.median(rr)), 3) if len(rr) else None
            else:
                row["junction_points"] = int(len(jx))
                row["tile_ratio_wall_to_floor_at_junction"] = None
        scale[key] = row

    # ---- E edges + F walls
    vert = segs[status == "vertical"]
    cuts = []
    order = sorted([k for k in regions if k.startswith("wall-")], key=lambda k: np.nonzero(regions[k])[1].mean())
    edges, walls_f = {}, {}
    for a_k, b_k in zip(order, order[1:]):
        ra, rb = regions[a_k], regions[b_k]
        seam = cv2.dilate(ra.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & rb
        if not seam.any():
            cuts.append({"between": [a_k, b_k], "touching": False})
            continue
        x_cut = float(np.nonzero(seam)[1].mean())
        near = [s for s in vert if abs((s[0] + s[2]) / 2 - x_cut) <= 0.012 * w]
        longest = max(near, key=lambda s: abs(s[3] - s[1]), default=None)
        ev = float(abs(longest[3] - longest[1])) if longest is not None else 0.0
        row = {"between": [a_k, b_k], "x": round(x_cut, 1), "vertical_photo_edge_px": round(ev, 1),
               "edge_x": round(float((longest[0] + longest[2]) / 2), 1) if longest is not None else None,
               "corner_evidence": ev >= 0.1 * h}
        if longest is not None and ev >= 0.1 * h:
            p0 = (longest[:2] + longest[2:]) / 2
            dline = dir_at(vps["vertical"], p0) if "vertical" in vps else np.array([0.0, 1.0])
            nrm = np.array([-dline[1], dline[0]])
            ya, xa = np.nonzero(ra)
            yb, xb = np.nonzero(rb)
            sa = (np.stack([xa, ya], 1) - p0) @ nrm
            sb = (np.stack([xb, yb], 1) - p0) @ nrm
            side_b = np.sign(np.median(sb)) if len(sb) else 1.0
            row[f"{a_k}_px_past_corner_line"] = int((np.sign(sa) == side_b).sum() if len(sa) else 0)
            row[f"{b_k}_px_past_corner_line"] = int((np.sign(sb) == -side_b).sum() if len(sb) else 0)
        cuts.append(row)
    ceil_like = ~wall_m & ~floor_m & ~props
    for key in order:
        reg = regions[key]
        top, bot = [], []
        for x in np.nonzero(reg.any(axis=0))[0]:
            ys = np.nonzero(reg[:, x])[0]
            yt, yb = ys.min(), ys.max()
            if yt >= 3 and ceil_like[yt - 3:yt, x].all():
                top.append((x, yt))
            if yb + 3 < h and floor_m[yb + 1:yb + 4, x].all():
                bot.append((x, yb))
        se = (surf[key]["geometry"].get("straight_edges") or {}).get("removed", {})
        edges[key] = {"top": {"kind": "fitted line (Step 2 clipped)" if f"{key} ceiling" in se else "raw mask contour",
                              **(boundary_fit(top, diag) or {"points": len(top)})},
                      "bottom": {"kind": "fitted line (Step 2 clipped)" if f"{key} floor" in se else "raw mask contour",
                                 **(boundary_fit(bot, diag) or {"points": len(bot)})}}
    sw = {wd["id"]: wd for wd in stored.get("walls", [])}
    planes = []
    for key in order:
        th = surf[key]["three"]
        planes.append((key, np.array(th["plane"]["normal"]), th["plane"]["d_units"] * th["grid"]["mm_per_unit"]))
    same = []
    for i in range(len(planes)):
        for j in range(i + 1, len(planes)):
            (ka, na, da), (kb, nb, db) = planes[i], planes[j]
            dn = math.degrees(math.acos(min(1.0, abs(float(na @ nb)))))
            if dn <= 3:
                rel = abs(abs(da) - abs(db)) / max(abs(da), abs(db), 1e-6)
                same.append({"pieces": [ka, kb], "normal_diff_deg": round(dn, 2), "offset_diff_share": round(rel, 3),
                             "verdict": "same plane (split)" if rel <= 0.05 else "parallel faces (pillar / recess)"})
    for dd in dots:
        if dd["kind"] != "wall":
            continue
        x, y = dd["dot"]
        walls_f[dd["id"]] = {"label": dd["label"], "dot": dd["dot"], "dot_inside_photo": 0 <= x < w and 0 <= y < h,
                             "dot_inside_own_region": bool(regions[dd["id"]][min(h - 1, y), min(w - 1, x)]),
                             "faces": (sw.get(dd["id"], {}).get("direction") or {}).get("faces")}

    # ---- drawing
    img = (b.room.astype(np.float32) * 0.5).astype(np.uint8)
    for s, st in zip(segs, status):
        cv2.line(img, (int(s[0]), int(s[1])), (int(s[2]), int(s[3])), COL.get(st, (128, 128, 128)), 2 if not st.startswith("dropped") else 1, cv2.LINE_AA)
    for k, pv in vps.items():
        if pv[0] == "vp" and 0 <= pv[1][0] < w and 0 <= pv[1][1] < h:
            cv2.drawMarker(img, tuple(int(v) for v in pv[1]), COL[k], cv2.MARKER_CROSS, 40, 3)
        cv2.putText(img, f"{k} {label(pv)}", (10, 28 + 24 * list(vps).index(k)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL[k], 2)
    if hz is not None:
        p, q = hz
        dd_ = (q - p) / np.linalg.norm(q - p)
        ok, a1, a2 = cv2.clipLine((0, 0, w, h), tuple(int(v) for v in p - 6000 * dd_), tuple(int(v) for v in p + 6000 * dd_))
        if ok:
            cv2.line(img, a1, a2, (255, 0, 255), 2)
    cv2.drawMarker(img, (int(cx), int(cy)), (255, 255, 0), cv2.MARKER_TILTED_CROSS, 24, 2)
    for c in cuts:
        if "x" in c:
            col = (0, 255, 0) if c["corner_evidence"] else (255, 0, 0)
            cv2.line(img, (int(c["x"]), 0), (int(c["x"]), h), col, 1)
    for k, wf in walls_f.items():
        cv2.circle(img, tuple(int(v) for v in wf["dot"]), 9, (255, 255, 255), -1)
        cv2.putText(img, k, (int(wf["dot"][0]) + 10, int(wf["dot"][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # ---- checks
    checks = []

    def add(s, c, v, ok):
        checks.append({"surface": s, "check": c, "value": v, "result": "pass" if ok else ("fail" if ok is False else "n/a")})
    for k, v in fams.items():
        add("room", f"A lines in {k} >= 3", v["count"], v["count"] >= 3)
    for k, v in cam["orthogonality_error_deg"].items():
        add("room", f"C orthogonality {k} <= 3 deg", v, v <= 3)
    if "pitch_agreement_deg" in cam:
        add("room", "C pitch vertical vs horizon <= 2 deg", cam["pitch_agreement_deg"], cam["pitch_agreement_deg"] <= 2)
    if fvals:
        add("room", "C render focal within 3% of the VP focal", [round(f_store, 1), round(f, 1)], abs(f_store - f) <= 0.03 * f)
    for k, v in scale.items():
        r = v.get("tile_ratio_wall_to_floor_at_junction")
        if r is not None:
            add(k, "D tile size wall/floor at the junction within 5% of 1", r, abs(r - 1) <= 0.05)
        elif k != "floor" and "refused" not in v:
            add(k, "D tile size wall/floor at the junction", "no junction", None)
    for k, e in edges.items():
        for side in ("top", "bottom"):
            if "rms_px" in e[side]:
                add(k, f"E {side} edge RMS <= 1 px", e[side]["rms_px"], e[side]["rms_px"] <= 1)
    for c in cuts:
        for kk, vv in c.items():
            if kk.endswith("_px_past_corner_line"):
                add(kk.split("_px")[0], f"E px past the corner line with {c['between']}", vv, vv == 0)
        if "x" in c:
            add("room", f"F corner evidence at x={c['x']} ({c['between']})", c["vertical_photo_edge_px"], c["corner_evidence"])
    for s in same:
        add("room", f"F {s['pieces']} {s['verdict']}", s["offset_diff_share"], s["verdict"] != "same plane (split)")
    for k, wf in walls_f.items():
        add(k, "F dot inside the photo and its own wall", [wf["dot_inside_photo"], wf["dot_inside_own_region"]],
            wf["dot_inside_photo"] and wf["dot_inside_own_region"])
    rep = {"room": name, "job": str(src_job.relative_to(ROOT)), "size": [w, h], "A_lines": lines_rep, "B_vps": {
        k: v for k, v in fams.items()}, "B_horizon_y_at_centre": cam.get("horizon_y_at_centre"), "C_camera": cam,
        "D_scale": scale, "E_edges": edges, "E_F_cuts": cuts, "F_walls": {"planes": len(planes), "dots": len(walls_f),
                                                                          "walls": walls_f, "parallel_or_same": same},
        "checks": checks}
    Image.fromarray(img).save(out / f"{name}__lines.png")
    (out / f"{name}__camera.json").write_text(json.dumps(rep, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    print(name, "done", sum(c["result"] == "fail" for c in checks), "fails")
    return 0


if __name__ == "__main__":
    sys.exit(main())
