"""
Wall tile direction, DIAGNOSIS ONLY (no code or pixel change).

    backend/.venv/bin/python tests/regression/wall_direction_diagnose.py <out_dir> [room ...]

Per wall of each baseline room (tests/regression/baseline_rooms.ROOMS; the job
is copied first, nothing is written back):

  stored      direction source, stored VP, validated (segments/wall/wall_geometry.json)
  lines       DeepLSD segments on the photo at render resolution (backend/deeplsd_runner.py,
              run in ~/geocalib-env), assigned to a wall when >= 80 % of the segment lies in
              that wall's tiled region grown by 0.4 % of the diagonal, longer than 1.5 % of
              the diagonal, and NOT pointing at the PHOTO's vertical VP (within 15 deg; the
              VP of all segments steeper than 55 deg, RANSAC, rejected when it falls inside
              the image rows (that is receding floor seams); the render's when none).
              Type by position: ceiling junction / skirting junction (within 1 % of the
              diagonal of the region's top / bottom edge), inside object <label>, or
              interior wall edge (door / window top, frame, beam -- not told apart).
  common VP   RANSAC over line pairs (inlier: <= 2 deg), refined by weighted least
              squares on the inliers (also for structural lines alone: not inside an object); per line the residual in deg (line vs direction
              to the VP from its midpoint) and in px (end-point offset = half length x sin).
  angle error per horizontal line, deg between the line and the RENDERED horizontal seam
              through its midpoint; the seam direction is cos(r) e_u - sin(r) e_v of the
              render's own 3D record (plane basis + grid rotation), projected with its K.
  ortho       neighbouring walls (regions touching): the angle between their horizontal
              directions (stored VPs with the stored focal; photo VPs with the render's
              focal; rendered seam directions in 3D), and its error from 90 deg (or 0 deg
              for walls closer to parallel), and the focal that a right angle between the
              two photo VPs implies (f^2 = -(v1-p).(v2-p), p = principal point).
  distortion  wide_angle: residual vs distance from the image centre.
Writes <out_dir>/<room>__directions.png and <out_dir>/wall_direction_diagnose.json.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
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
from engine import TileSpec  # noqa: E402

GEO_PY = Path.home() / "geocalib-env/bin/python"
MIN_LEN, GROW, IN_SHARE, VERT_DEG, EDGE_BAND, INLIER_DEG = 0.015, 0.004, 0.8, 15.0, 0.01, 2.0
COLORS = [(255, 60, 60), (60, 200, 60), (60, 120, 255), (255, 170, 0), (200, 60, 255), (0, 220, 220), (255, 255, 0)]


def deeplsd(img: np.ndarray) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp:
        p, o = Path(tmp) / "room.png", Path(tmp) / "lines.json"
        Image.fromarray(img).save(p)
        subprocess.run([str(GEO_PY), str(ROOT / "backend/deeplsd_runner.py"), str(p), str(o)],
                       check=True, capture_output=True)
        return np.array(json.loads(o.read_text())["lines"], float).reshape(-1, 4)


def project(D, f, cx, cy):
    """Image point of direction D (camera x right, y down, z forward); None when at infinity."""
    if abs(D[2]) < 1e-9:
        return None
    return np.array([cx + f * D[0] / D[2], cy + f * D[1] / D[2]])


def dir_at(vp_or_dir, p):
    """Unit image direction at p toward a VP (finite) or along a 2D direction (infinite)."""
    kind, v = vp_or_dir
    d = (v - p) if kind == "vp" else v
    n = np.hypot(*d)
    return d / n if n > 1e-9 else np.array([1.0, 0.0])


def angle_between(d1, d2):
    c = abs(float(np.dot(d1, d2)) / (np.hypot(*d1) * np.hypot(*d2) + 1e-12))
    return math.degrees(math.acos(min(1.0, c)))


def fit_vp(segs, w, h, rng):
    """RANSAC + weighted least squares VP of segments [[x1,y1,x2,y2]]; (vp_h, inliers)."""
    if len(segs) < 2:
        return None, np.zeros(len(segs), bool)
    s = max(w, h)
    P1 = np.c_[segs[:, :2] / s, np.ones(len(segs))]
    P2 = np.c_[segs[:, 2:] / s, np.ones(len(segs))]
    L = np.cross(P1, P2)
    L /= np.linalg.norm(L[:, :2], axis=1, keepdims=True)
    mids = (segs[:, :2] + segs[:, 2:]) / 2
    dirs = segs[:, 2:] - segs[:, :2]
    lens = np.hypot(dirs[:, 0], dirs[:, 1])

    def residual_deg(vh):
        if abs(vh[2]) < 1e-9:
            to = np.tile(vh[:2], (len(segs), 1))
        else:
            to = vh[:2] / vh[2] * s - mids
        cos = np.abs((to * dirs).sum(1)) / (np.hypot(*to.T) * lens + 1e-12)
        return np.degrees(np.arccos(np.clip(cos, 0, 1)))

    best, best_score = None, -1
    n = len(segs)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if len(pairs) > 3000:
        pairs = [pairs[k] for k in rng.choice(len(pairs), 3000, replace=False)]
    for i, j in pairs:
        vh = np.cross(L[i], L[j])
        if np.linalg.norm(vh) < 1e-12:
            continue
        inl = residual_deg(vh) <= INLIER_DEG
        score = lens[inl].sum()
        if score > best_score:
            best, best_score = inl, score
    if best is None or best.sum() < 2:
        return None, np.zeros(n, bool)
    A = L[best] * np.sqrt(lens[best])[:, None]
    vh = np.linalg.svd(A)[2][-1]
    return (vh, s), residual_deg(vh) <= INLIER_DEG


def vp_image(vp):
    vh, s = vp
    if abs(vh[2]) < 1e-9 * np.linalg.norm(vh):
        return ("dir", np.array(vh[:2]))
    return ("vp", vh[:2] / vh[2] * s)


def main() -> int:
    out = Path(sys.argv[1])
    names = sys.argv[2:] or list(ROOMS)
    out.mkdir(parents=True, exist_ok=True)
    spec = TileSpec(artwork=np.asarray(Image.open(TILE).convert("RGB")), width_mm=TILE_W_MM,
                    height_mm=TILE_H_MM, rotation_deg=0, grout_mm=GROUT_MM)
    report = {}
    for name in names:
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as tmp:
            job = Path(tmp) / ROOMS[name].name
            shutil.copytree(ROOMS[name], job)
            b = reuse.load(job)
            fl, wl = pe.load(job / "segments")
            geo = pe.ensure_geometry(job / "segments", b.room, fl, wl, clean=b.clean)
            stored_doc = json.loads((job / "segments/wall/wall_geometry.json").read_text())
            objects = json.loads((job / "segments/objects.json").read_text()).get("objects", [])
            renders = {}
            for wdoc in geo["walls"]:
                idx = int(wdoc["index"])
                try:
                    r = pe.render_tiled_room(b.room, fl, wl, spec, "wall", (None, None, None), clean=b.clean,
                                             props=b.props, wall_index=idx, geometry=geo)
                except Exception as e:  # noqa: BLE001
                    renders[idx] = {"refused": str(e)[:120]}
                    continue
                key = f"wall-{idx}"
                entry = next((e for e in (r.geometry.get("wall") or {}).get("walls", []) if e.get("index") == idx), {})
                renders[idx] = {"region": r.regions.get(key), "three": r.three.get(key), "entry": entry}
        h, w = b.room.shape[:2]
        diag = float(np.hypot(h, w))
        segs = deeplsd(b.room)
        segs = segs[np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]) >= MIN_LEN * diag]
        sx = w / float((stored_doc.get("canvas") or [w, h])[0])
        ang_all = np.degrees(np.arctan2(segs[:, 3] - segs[:, 1], segs[:, 2] - segs[:, 0]))
        steep = segs[np.abs((ang_all + 90) % 180 - 90) > 55]
        vvp, vinl = fit_vp(steep, w, h, rng)
        photo_vert = vp_image(vvp) if vvp is not None else None
        if photo_vert is not None and photo_vert[0] == "vp" and 0 <= photo_vert[1][1] <= h:
            photo_vert = None            # inside the image: receding floor seams, not verticals
        stored_by_idx = {int(x["index"]): x for x in stored_doc.get("walls", [])}
        obj_boxes = []
        for o in objects:
            bx = o.get("bbox")
            sw, sh = o.get("source_width") or w, o.get("source_height") or h
            if bx:
                obj_boxes.append((o.get("label"), np.array(bx, float) * [w / sw, h / sh, w / sw, h / sh]))
        img = (b.room.astype(np.float32) * 0.55).astype(np.uint8)
        walls_out, horiz_dirs3d, photo_vps, regions = {}, {}, {}, {}
        grow = max(3, int(GROW * diag))
        for n_w, (idx, rd) in enumerate(sorted(renders.items())):
            sd = stored_by_idx.get(idx, {})
            dr = sd.get("direction") or {}
            vpd = dr.get("vanishing_point") or {}
            row = {"stored": {"source": dr.get("source"), "faces": dr.get("faces"),
                              "vanishing_point": [round(v * sx, 1) for v in vpd["image"]] if vpd.get("image") else None,
                              "vp_at_infinity": vpd.get("at_infinity"),
                              "snapped_to_depth_vp": (dr.get("evidence") or {}).get("snapped_to_depth_vp"),
                              "edges_disagree_deg": (dr.get("evidence") or {}).get("edges_disagree_deg"),
                              "validated": dr.get("validated"), "validation": dr.get("validation")}}
            if "refused" in rd or rd.get("three") is None or rd.get("region") is None:
                row["render"] = rd.get("refused", "no 3D record")
                walls_out[f"wall-{idx}"] = row
                continue
            region, three = rd["region"], rd["three"]
            regions[idx] = region
            cam, pl, gr = three["camera"], three["plane"], three["grid"]
            f, cx, cy = cam["focal_px"], cam["cx"], cam["cy"]
            eu, ev, rad = np.array(pl["e_u"]), np.array(pl["e_v"]), gr["rotation_rad"]
            D_h = math.cos(rad) * eu - math.sin(rad) * ev
            D_v = math.sin(rad) * eu + math.cos(rad) * ev
            horiz_dirs3d[idx] = D_h
            seam_vp = project(D_h, f, cx, cy)
            seam = ("vp", seam_vp) if seam_vp is not None else ("dir", D_h[:2])
            vert_vp = project(D_v, f, cx, cy)
            vert = ("vp", vert_vp) if vert_vp is not None else ("dir", D_v[:2])
            render_vert = vert
            if photo_vert is not None:
                vert = photo_vert
            row["render"] = {"plane_source": rd["entry"].get("plane_source"), "focal_px": round(f, 1),
                             "principal_point": [round(cx, 1), round(cy, 1)],
                             "grid_rotation_deg": round(math.degrees(rad), 2),
                             "seam_vp": [round(v, 1) for v in seam_vp] if seam_vp is not None else "at infinity",
                             "render_vertical_vp": [round(v, 1) for v in render_vert[1]] if render_vert[0] == "vp" else "at infinity",
                             "validated": rd["entry"].get("validated"), "validation": rd["entry"].get("validation")}
            grown = cv2.dilate(region.astype(np.uint8), np.ones((2 * grow + 1,) * 2, np.uint8)).astype(bool)
            cols = np.nonzero(region.any(axis=0))[0]
            top = np.full(w, -1)
            bot = np.full(w, -1)
            for x in cols:
                ys = np.nonzero(region[:, x])[0]
                top[x], bot[x] = ys.min(), ys.max()
            mine, types = [], []
            for sgm in segs:
                t = np.linspace(0, 1, 25)
                px = np.clip(np.round(sgm[0] + t * (sgm[2] - sgm[0])).astype(int), 0, w - 1)
                py = np.clip(np.round(sgm[1] + t * (sgm[3] - sgm[1])).astype(int), 0, h - 1)
                if grown[py, px].mean() < IN_SHARE:
                    continue
                mid = np.array([(sgm[0] + sgm[2]) / 2, (sgm[1] + sgm[3]) / 2])
                d = sgm[2:] - sgm[:2]
                if angle_between(d, dir_at(vert, mid)) <= VERT_DEG:
                    continue
                mx = int(np.clip(round(mid[0]), 0, w - 1))
                band = EDGE_BAND * diag
                typ = "interior wall edge (door/window top, frame, beam)"
                if top[mx] >= 0 and abs(mid[1] - top[mx]) <= band:
                    typ = "ceiling junction"
                elif bot[mx] >= 0 and abs(mid[1] - bot[mx]) <= band:
                    typ = "skirting junction"
                else:
                    for lab, bx in obj_boxes:
                        if bx[0] <= mid[0] <= bx[2] and bx[1] <= mid[1] <= bx[3]:
                            typ = f"inside object '{lab}'"
                            break
                mine.append(sgm)
                types.append(typ)
            mine = np.array(mine).reshape(-1, 4)
            vp, inl = fit_vp(mine, w, h, rng)
            lines = []
            common = vp_image(vp) if vp is not None else None
            for sgm, typ, good in zip(mine, types, inl):
                mid = (sgm[:2] + sgm[2:]) / 2
                d = sgm[2:] - sgm[:2]
                length = float(np.hypot(*d))
                img_angle = math.degrees(math.atan2(d[1], d[0]))
                img_angle = (img_angle + 90) % 180 - 90
                res_deg = angle_between(d, dir_at(common, mid)) if common else None
                err = angle_between(d, dir_at(seam, mid))
                lines.append({"type": typ, "length_px": round(length, 1), "image_angle_deg": round(img_angle, 2),
                              "mid": [round(v, 1) for v in mid], "inlier": bool(good),
                              "residual_deg": round(res_deg, 2) if res_deg is not None else None,
                              "residual_px": round(length / 2 * math.sin(math.radians(res_deg)), 2) if res_deg is not None else None,
                              "seam_angle_error_deg": round(err, 2),
                              "r_from_center": round(float(np.hypot(mid[0] - w / 2, mid[1] - h / 2)) / (diag / 2), 3)})
            if common:
                photo_vps[idx] = common
            struct = np.array([t_ for t_ in types]) if types else np.zeros(0)
            smask = np.array([not t_.startswith("inside object") for t_ in types], bool)
            svp, sinl = fit_vp(mine[smask], w, h, rng) if smask.sum() >= 2 else (None, None)
            scommon = vp_image(svp) if svp is not None else None
            L = np.array([x["length_px"] for x in lines]) if lines else np.zeros(0)
            E = np.array([x["seam_angle_error_deg"] for x in lines]) if lines else np.zeros(0)
            I = np.array([x["inlier"] for x in lines], bool) if lines else np.zeros(0, bool)
            R = np.array([x["residual_deg"] or 0 for x in lines]) if lines else np.zeros(0)
            row["lines"] = {"count": len(lines), "inliers": int(I.sum()),
                            "by_type": {t: types.count(t) for t in sorted(set(types))},
                            "common_vp": ([round(v, 1) for v in common[1]] if common and common[0] == "vp" else
                                          ("at infinity, direction " + str([round(v, 3) for v in common[1]]) if common else None)),
                            "structural_lines": int(smask.sum()) if len(types) else 0,
                            "structural_vp": ([round(v, 1) for v in scommon[1]] if scommon and scommon[0] == "vp" else
                                              ("at infinity" if scommon else None)),
                            "structural_inliers": int(sinl.sum()) if sinl is not None else 0,
                            "spread_inliers_deg": {"median": round(float(np.median(R[I])), 2), "max": round(float(R[I].max()), 2)} if I.any() else None,
                            "spread_all_deg": {"median": round(float(np.median(R)), 2), "max": round(float(R.max()), 2)} if len(R) else None,
                            "spread_inliers_px": {"median": round(float(np.median([x["residual_px"] for x in lines if x["inlier"]])), 2),
                                                  "max": round(float(max(x["residual_px"] for x in lines if x["inlier"])), 2)} if I.any() else None,
                            "list": lines}
            S = np.zeros(len(lines), bool)
            if sinl is not None:
                S[np.nonzero(smask)[0][sinl]] = True
            row["angle_error_today_deg"] = ({"structural_inliers_weighted_mean":
                                             round(float((E[S] * L[S]).sum() / L[S].sum()), 2) if S.any() else None,
                                             "structural_inliers": int(S.sum()),"length_weighted_mean": round(float((E * L).sum() / L.sum()), 2),
                                             "median": round(float(np.median(E)), 2), "max": round(float(E.max()), 2),
                                             "inliers_weighted_mean": round(float((E[I] * L[I]).sum() / L[I].sum()), 2) if I.any() else None}
                                            if len(E) else None)
            walls_out[f"wall-{idx}"] = row
            # ---- drawing ----
            col = COLORS[n_w % len(COLORS)]
            tint = img[region].astype(np.float32)
            img[region] = (0.75 * tint + 0.25 * np.array(col)).astype(np.uint8)
            # rendered horizontal seams (exact): v_mm level lines of the render's own grid
            yy, xx = np.nonzero(region)
            rays = np.stack([(xx - cx) / f, (yy - cy) / f, np.ones(len(xx))], 1)
            nrm, dd = np.array(pl["normal"]), pl["d_units"]
            Z = -dd / (rays @ nrm)
            X = rays * Z[:, None]
            u, v = X @ eu, X @ ev
            vr = u * math.sin(rad) + v * math.cos(rad)
            vmm = (vr + gr["offset_units"][1]) * gr["mm_per_unit"]
            k = np.floor(vmm / three["tile"]["height_mm"])
            field = np.full((h, w), np.nan)
            field[yy, xx] = k
            edge = np.zeros((h, w), bool)
            edge[1:, :] |= (field[1:, :] != field[:-1, :]) & ~np.isnan(field[1:, :]) & ~np.isnan(field[:-1, :])
            edge[:, 1:] |= (field[:, 1:] != field[:, :-1]) & ~np.isnan(field[:, 1:]) & ~np.isnan(field[:, :-1])
            img[cv2.dilate(edge.astype(np.uint8), np.ones((2, 2), np.uint8)).astype(bool)] = (255, 255, 255)
            for sgm, good in zip(mine, inl):
                cv2.line(img, tuple(int(v) for v in sgm[:2]), tuple(int(v) for v in sgm[2:]), col, 3 if good else 1, cv2.LINE_AA)
            if common and common[0] == "vp":
                p = common[1]
                if 0 <= p[0] < w and 0 <= p[1] < h:
                    cv2.drawMarker(img, (int(p[0]), int(p[1])), col, cv2.MARKER_CROSS, 40, 3)
                else:                                    # off-image: arrow at the border toward it
                    c0 = np.array([w / 2, h / 2])
                    d = (p - c0) / np.hypot(*(p - c0))
                    tmax = min(((w - 1 - c0[0]) / d[0]) if d[0] > 0 else (-c0[0] / d[0]) if d[0] < 0 else 1e9,
                               ((h - 1 - c0[1]) / d[1]) if d[1] > 0 else (-c0[1] / d[1]) if d[1] < 0 else 1e9)
                    q = c0 + d * (tmax - 10)
                    cv2.arrowedLine(img, tuple(int(v) for v in (q - 60 * d)), tuple(int(v) for v in q), col, 4, tipLength=0.4)
                    cv2.putText(img, f"VP w{idx} ({p[0]:.0f},{p[1]:.0f})", (int(np.clip(q[0] - 200, 5, w - 330)), int(np.clip(q[1] - 20, 30, h - 10))),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2, cv2.LINE_AA)
            if len(cols):
                ys_, xs_ = np.nonzero(region)
                cv2.putText(img, f"wall-{idx}", (int(xs_.mean()) - 40, int(ys_.mean())), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 3, cv2.LINE_AA)
        # ---- orthogonality between neighbouring walls ----
        ortho = []
        keys = sorted(regions)
        K_f = None
        for a_i, a in enumerate(keys):
            for bb in keys[a_i + 1:]:
                ga = cv2.dilate(regions[a].astype(np.uint8), np.ones((2 * grow + 1,) * 2, np.uint8)).astype(bool)
                if not (ga & regions[bb]).any():
                    continue
                cam = walls_out[f"wall-{a}"]["render"]
                f, (cx, cy) = cam["focal_px"], cam["principal_point"]
                pair = {"walls": [f"wall-{a}", f"wall-{bb}"]}

                def ray(vp):
                    kind, v = vp
                    r = np.array([v[0] - cx, v[1] - cy, f]) if kind == "vp" else np.array([v[0], v[1], 0.0])
                    return r / np.linalg.norm(r)

                def judge(r1, r2):
                    ang = math.degrees(math.acos(min(1.0, abs(float(np.dot(r1, r2))))))
                    return {"angle_deg": round(ang, 2), "error_deg": round(min(abs(90 - ang), ang), 2),
                            "expected": "perpendicular" if ang > 45 else "parallel"}

                pair["rendered_seams"] = judge(horiz_dirs3d[a], horiz_dirs3d[bb])
                if a in photo_vps and bb in photo_vps:
                    pair["photo_lines_vps"] = judge(ray(photo_vps[a]), ray(photo_vps[bb]))
                    if photo_vps[a][0] == "vp" and photo_vps[bb][0] == "vp":
                        f2 = -float(np.dot(photo_vps[a][1] - [cx, cy], photo_vps[bb][1] - [cx, cy]))
                        pair["photo_vps_focal_for_right_angle_px"] = round(math.sqrt(f2), 1) if f2 > 0 else "none (VPs on the same side)"
                sa, sb = stored_by_idx.get(a, {}), stored_by_idx.get(bb, {})
                va = ((sa.get("direction") or {}).get("vanishing_point") or {}).get("homogeneous")
                vb = ((sb.get("direction") or {}).get("vanishing_point") or {}).get("homogeneous")
                fs = stored_doc.get("focal_px")
                pp = stored_doc.get("principal_point") or [x / 2 for x in stored_doc.get("canvas", [w, h])]
                if va and vb and fs:
                    def sray(vh):
                        vh = np.array(vh, float)
                        r = np.array([vh[0] - pp[0] * vh[2], vh[1] - pp[1] * vh[2], fs * vh[2]])
                        return r / np.linalg.norm(r)
                    pair["stored_vps"] = judge(sray(va), sray(vb)) | {"focal_px": round(fs, 1)}
                ortho.append(pair)
        rep = {"size": [w, h], "segments_total": int(len(segs)), "walls": walls_out, "orthogonality": ortho,
               "photo_vertical_vp": ([round(v, 1) for v in photo_vert[1]] if photo_vert and photo_vert[0] == "vp" else
                                     ("at infinity" if photo_vert else None)),
               "photo_vertical_lines": int(vinl.sum()) if vvp is not None else 0}
        if name == "wide_angle":
            rows = [(x["r_from_center"], x["residual_deg"]) for wr in walls_out.values()
                    for x in (wr.get("lines") or {}).get("list", []) if x["inlier"] and x["residual_deg"] is not None]
            if rows:
                r_, e_ = np.array(rows).T
                inner, outer = e_[r_ < 0.5], e_[r_ >= 0.5]
                rep["distortion"] = {"inlier_lines": len(rows),
                                     "residual_deg_inner_half_median": round(float(np.median(inner)), 3) if len(inner) else None,
                                     "residual_deg_outer_half_median": round(float(np.median(outer)), 3) if len(outer) else None,
                                     "corr_residual_vs_radius": round(float(np.corrcoef(r_, e_)[0, 1]), 3) if len(rows) > 2 else None}
        report[name] = rep
        Image.fromarray(img).save(out / f"{name}__directions.png")
        print(name, "done", flush=True)
    (out / "wall_direction_diagnose.json").write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
