"""
Camera / vanishing-point / seam EVIDENCE (no code or pixel change).

    backend/.venv/bin/python tests/regression/camera_evidence.py <out_dir> <room>

One room per run (tests are never run in parallel). The job is copied first.

1 lines      LSD (cv2) + DeepLSD (backend/deeplsd_runner.py); an LSD segment that
             repeats a DeepLSD one (midpoint within 3 px of its line, within 2 deg)
             is dropped. Segments shorter than 1.5 % of the diagonal are ignored.
             vertical:      RANSAC VP of segments within 35 deg of vertical (leaning
                            lines of a pitched camera are kept: they meet at a
                            finite vertical VP); the VP must lie outside the image
                            rows or at infinity, else it is receding floor lines
             horizontal-A:  RANSAC VP of the remaining segments (most length)
             horizontal-B:  RANSAC VP of what is left
             rejected:      the rest
             inlier = line within 2 deg of the direction to its VP; residual in px =
             the end-point distance from the line through the midpoint and the VP
2 points     the three VPs, principal point (image centre), horizon (through A and
             B when both are finite, else the vertical VP's horizon for focal f)
3 camera     focal from every orthogonal VP pair, f^2 = -(v1 - p).(v2 - p), used only
             when both VPs lie within COND diagonals of the centre (a VP near infinity
             leaves f undetermined); none usable -> the stored focal, said so; pitch
             from the vertical VP and from the horizon; roll from the vertical VP;
             yaw from VP-A; orthogonality error of each pair at the mean focal;
             EXIF focal when the photo has one
4 seams      floor + every wall rendered as the Studio does (tests/fixtures/tile.png);
             each surface's two seam families come from its own 3D record
             (plane basis + grid rotation), so their common intersection is exact:
             VP = K (cos r e_u - sin r e_v) and K (sin r e_u + cos r e_v). Each
             family is matched to the nearest photo VP (by ray angle). Reported:
             distance seam-VP to photo VP (px, and ray angle), and the angle error
             between the seam and the line through the photo VP at 200 points of
             the surface (max and mean)
5 bedroom    floor seam VP, the VP of the lines along the ceiling, and the far end
Checks a-f as specified; every value and pass/fail is in <room>__camera.json,
the drawing in <room>__lines.png.
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
from baseline_rooms import ROOMS, TILE, TILE_W_MM, TILE_H_MM, GROUT_MM  # noqa: E402
from engine import TileSpec  # noqa: E402
from wall_direction_diagnose import deeplsd, fit_vp, vp_image  # noqa: E402

MIN_LEN, VERT_CAND_DEG = 0.015, 35.0
COND = 3.0                     # a VP pair fixes the focal only when both lie within this many diagonals
FAR = 10.0                     # a VP farther than this many diagonals is judged by angle only
FAMILY_COL = {"vertical": (0, 200, 255), "horizontal-A": (255, 70, 70), "horizontal-B": (60, 220, 60),
              "rejected": (120, 120, 120)}


def lsd(gray):
    det = cv2.createLineSegmentDetector(0)
    out = det.detect(gray)[0]
    return np.zeros((0, 4)) if out is None else out.reshape(-1, 4).astype(float)


def seg_angle(s):
    return math.degrees(math.atan2(s[3] - s[1], s[2] - s[0]))


def ray(pt, f, cx, cy):
    kind, v = pt
    r = np.array([v[0] - cx, v[1] - cy, f], float) if kind == "vp" else np.array([v[0], v[1], 0.0])
    return r / np.linalg.norm(r)


def ang(r1, r2):
    return math.degrees(math.acos(min(1.0, abs(float(np.dot(r1, r2))))))


def proj(D, f, cx, cy):
    if abs(D[2]) < 1e-9:
        return ("dir", np.array(D[:2], float))
    return ("vp", np.array([cx + f * D[0] / D[2], cy + f * D[1] / D[2]]))


def dir_at(pt, p):
    kind, v = pt
    d = (v - p) if kind == "vp" else v
    n = np.hypot(*d)
    return d / n if n > 1e-9 else np.array([1.0, 0.0])


def residual_px(segs, pt):
    out = []
    for s in segs:
        mid = (s[:2] + s[2:]) / 2
        d = dir_at(pt, mid)
        nrm = np.array([-d[1], d[0]])
        out.append(max(abs(float((s[:2] - mid) @ nrm)), abs(float((s[2:] - mid) @ nrm))))
    return out


def label(pt):
    if pt is None:
        return None
    return [round(float(x), 1) for x in pt[1]] if pt[0] == "vp" else {"at_infinity_direction": [round(float(x), 3) for x in pt[1]]}


def main() -> int:
    out, name = Path(sys.argv[1]), sys.argv[2]
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    spec = TileSpec(artwork=np.asarray(Image.open(TILE).convert("RGB")), width_mm=TILE_W_MM,
                    height_mm=TILE_H_MM, rotation_deg=0, grout_mm=GROUT_MM)
    with tempfile.TemporaryDirectory() as tmp:
        job = Path(tmp) / ROOMS[name].name
        shutil.copytree(ROOMS[name], job)
        b = reuse.load(job)
        fl, wl = pe.load(job / "segments")
        geo = pe.ensure_geometry(job / "segments", b.room, fl, wl, clean=b.clean)
        objects = json.loads((job / "segments/objects.json").read_text()).get("objects", [])
        exif_f = None
        try:
            ex = Image.open(job / "original.png").getexif()
            exif_f = ex.get_ifd(0x8769).get(0xA405) if ex else None
        except Exception:  # noqa: BLE001
            pass
        surfaces = {}
        for surface, idx in [("floor", None)] + [("wall", int(w["index"])) for w in geo["walls"]]:
            key = "floor" if idx is None else f"wall-{idx}"
            try:
                r = pe.render_tiled_room(b.room, fl, wl, spec, surface, (None, None, None), clean=b.clean,
                                         props=b.props, wall_index=idx, geometry=geo)
            except Exception as e:  # noqa: BLE001
                surfaces[key] = {"refused": str(e)[:120]}
                continue
            if key in r.three and key in r.regions:
                surfaces[key] = {"three": r.three[key], "region": r.regions[key]}
            else:
                surfaces[key] = {"refused": "no 3D record"}
    h, w = b.room.shape[:2]
    diag = math.hypot(h, w)
    cx, cy = w / 2.0, h / 2.0
    gray = cv2.cvtColor(b.room, cv2.COLOR_RGB2GRAY)

    # ---- 1 lines
    deep = deeplsd(b.room)
    lsd_s = lsd(gray)
    keep = []
    for s in lsd_s:
        mid = (s[:2] + s[2:]) / 2
        dup = False
        for t in deep:
            d = t[2:] - t[:2]
            L = np.hypot(*d)
            if L < 1:
                continue
            nrm = np.array([-d[1], d[0]]) / L
            if abs(float((mid - t[:2]) @ nrm)) <= 3 and abs((seg_angle(s) - seg_angle(t) + 90) % 180 - 90) <= 2:
                u = float((mid - t[:2]) @ (d / L))
                if -3 <= u <= L + 3:
                    dup = True
                    break
        if not dup:
            keep.append(s)
    segs = np.vstack([deep, np.array(keep).reshape(-1, 4)]) if len(keep) else deep
    src = np.array(["deeplsd"] * len(deep) + ["lsd"] * len(keep))
    lens = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    m = lens >= MIN_LEN * diag
    segs, src = segs[m], src[m]
    fam = np.array(["rejected"] * len(segs), object)
    from_vert = np.abs(np.abs(np.array([seg_angle(s) for s in segs])) - 90) <= VERT_CAND_DEG
    vps = {}
    cand = np.nonzero(from_vert)[0]
    for _ in range(3):                                 # steep receding floor lines can win first: set them aside
        vvp, vinl = fit_vp(segs[cand], w, h, rng) if len(cand) >= 2 else (None, None)
        if vvp is None:
            break
        V = vp_image(vvp)
        if V[0] == "vp" and 0 <= V[1][1] <= h:
            cand = cand[~vinl]                         # receding floor lines, not verticals
            continue
        vps["vertical"] = V
        fam[cand[vinl]] = "vertical"
        break
    for fname in ("horizontal-A", "horizontal-B"):
        rest = np.nonzero(fam == "rejected")[0]
        if len(rest) < 2:
            break
        hvp, hinl = fit_vp(segs[rest], w, h, rng)
        if hvp is None or hinl.sum() < 2:
            break
        vps[fname] = vp_image(hvp)
        fam[rest[hinl]] = fname
    families = {}
    for fname in ("vertical", "horizontal-A", "horizontal-B", "rejected"):
        sel = fam == fname
        row = {"count": int(sel.sum()), "lsd": int((sel & (src == "lsd")).sum()), "deeplsd": int((sel & (src == "deeplsd")).sum())}
        if fname in vps:
            res = residual_px(segs[sel], vps[fname])
            row.update({"vp": label(vps[fname]), "residual_px_median": round(float(np.median(res)), 2),
                        "residual_px_max": round(float(np.max(res)), 2),
                        "insufficient": bool(sel.sum() < 3)})
        families[fname] = row

    # ---- 3 camera
    pairs = {}
    for a, bb in (("horizontal-A", "horizontal-B"), ("horizontal-A", "vertical"), ("horizontal-B", "vertical")):
        if a in vps and bb in vps and vps[a][0] == "vp" and vps[bb][0] == "vp":
            if max(np.hypot(*(vps[a][1] - [cx, cy])), np.hypot(*(vps[bb][1] - [cx, cy]))) > COND * diag:
                pairs[f"{a}|{bb}"] = "undetermined (a VP is near infinity)"
                continue
            f2 = -float(np.dot(vps[a][1] - [cx, cy], vps[bb][1] - [cx, cy]))
            pairs[f"{a}|{bb}"] = round(math.sqrt(f2), 1) if f2 > 0 else "none (VPs on the same side)"
    fvals = [v for v in pairs.values() if isinstance(v, float)]
    stored_f = (geo.get("camera") or {}).get("focal_px") or json.loads((ROOMS[name] / "segments/wall/wall_geometry.json").read_text()).get("focal_px")
    f = float(np.mean(fvals)) if fvals else float(stored_f)
    cam = {"principal_point": [cx, cy], "focal_from_pairs_px": pairs,
           "focal_px": round(f, 1), "focal_source": "VP pairs (mean)" if fvals else "stored -- not determinable from this room's lines",
           "focal_stored_px": round(float(stored_f), 1),
           "hfov_deg": round(math.degrees(2 * math.atan(w / (2 * f))), 1), "exif_focal_35mm": exif_f}
    ortho = {}
    for a, bb in (("horizontal-A", "horizontal-B"), ("horizontal-A", "vertical"), ("horizontal-B", "vertical")):
        if a in vps and bb in vps:
            ortho[f"{a}|{bb}"] = round(abs(90 - ang(ray(vps[a], f, cx, cy), ray(vps[bb], f, cx, cy))), 2)
    cam["orthogonality_error_deg"] = ortho
    if "vertical" in vps:
        rv = ray(vps["vertical"], f, cx, cy)
        if rv[1] < 0:
            rv = -rv                                      # world down
        cam["pitch_from_vertical_deg"] = round(math.degrees(math.asin(max(-1, min(1, rv[2])))), 2)
        cam["roll_deg"] = round(math.degrees(math.atan2(rv[0], rv[1])), 2)
    hz = None
    if "horizontal-A" in vps and "horizontal-B" in vps and vps["horizontal-A"][0] == "vp" and vps["horizontal-B"][0] == "vp":
        A, B = vps["horizontal-A"][1], vps["horizontal-B"][1]
        hz = (A, B)
    elif "horizontal-A" in vps and vps["horizontal-A"][0] == "vp":
        A = vps["horizontal-A"][1]
        if "vertical" in vps and vps["vertical"][0] == "vp":
            dv = vps["vertical"][1] - [cx, cy]
            t = np.array([-dv[1], dv[0]])
        else:
            t = np.array([1.0, 0.0])
        hz = (A, A + t)
    if hz is not None:
        p, q = hz
        y_c = p[1] + (q[1] - p[1]) * (cx - p[0]) / (q[0] - p[0]) if abs(q[0] - p[0]) > 1e-6 else p[1]
        cam["horizon_y_at_centre"] = round(float(y_c), 1)
        cam["pitch_from_horizon_deg"] = round(math.degrees(math.atan2(cy - y_c, f)), 2)
    hfin = [k for k in ("horizontal-A", "horizontal-B") if k in vps and vps[k][0] == "vp"]
    depth_key = min(hfin, key=lambda k: np.hypot(*(vps[k][1] - [cx, cy]))) if hfin else None
    cam["depth_vp_family"] = depth_key
    if "horizontal-A" in vps and vps["horizontal-A"][0] == "vp":
        cam["yaw_deg"] = round(math.degrees(math.atan2(vps["horizontal-A"][1][0] - cx, f)), 2)
    if "pitch_from_vertical_deg" in cam and "pitch_from_horizon_deg" in cam:
        cam["pitch_agreement_deg"] = round(abs(cam["pitch_from_vertical_deg"] - cam["pitch_from_horizon_deg"]), 2)

    # ---- 4 seams
    seam_rows = {}
    rows_tbl = []
    cams_used = set()
    img = (b.room.astype(np.float32) * 0.55).astype(np.uint8)
    for key, sd in surfaces.items():
        if "refused" in sd:
            seam_rows[key] = {"refused": sd["refused"]}
            continue
        th, region = sd["three"], sd["region"]
        c3, pl, gr = th["camera"], th["plane"], th["grid"]
        fs, sx, sy = c3["focal_px"], c3["cx"], c3["cy"]
        cams_used.add((round(fs, 1), round(sx, 1), round(sy, 1)))
        eu, ev, rad = np.array(pl["e_u"]), np.array(pl["e_v"]), gr["rotation_rad"]
        fams3d = {"seam family 1 (const v)": math.cos(rad) * eu - math.sin(rad) * ev,
                  "seam family 2 (const u)": math.sin(rad) * eu + math.cos(rad) * ev}
        ys, xs = np.nonzero(region)
        pick = rng.choice(len(xs), min(200, len(xs)), replace=False)
        pts = np.stack([xs[pick], ys[pick]], 1).astype(float)
        srow = {"render_camera": {"focal_px": round(fs, 1), "cx": round(sx, 1), "cy": round(sy, 1)}}
        for fname3, D in fams3d.items():
            sv = proj(D, fs, sx, sy)
            rs = ray(sv, fs, sx, sy)
            cands = [(k, ang(rs, ray(vps[k], f, cx, cy))) for k in vps]
            if not cands:
                srow[fname3] = {"seam_vp": label(sv), "matched": None}
                continue
            k_best, a_best = min(cands, key=lambda t: t[1])
            pv = vps[k_best]
            errs = [math.degrees(math.acos(min(1.0, abs(float(dir_at(sv, p) @ dir_at(pv, p)))))) for p in pts]
            far = (sv[0] != "vp" or pv[0] != "vp" or np.hypot(*(sv[1] - [cx, cy])) > FAR * diag
                   or np.hypot(*(pv[1] - [cx, cy])) > FAR * diag)
            dist = None if far else float(np.hypot(*(sv[1] - pv[1])))
            row = {"seam_vp": label(sv), "matched_photo_vp": k_best, "ray_angle_deg": round(a_best, 2),
                   "seam_vp_to_photo_vp_px": None if dist is None else round(dist, 1),
                   "seam_vp_to_photo_vp_pct_diag": None if dist is None else round(100 * dist / diag, 2),
                   "seam_angle_error_deg_max": round(float(np.max(errs)), 2),
                   "seam_angle_error_deg_mean": round(float(np.mean(errs)), 2)}
            if key == "floor" and sv[0] == "vp" and hz is not None and np.hypot(*(sv[1] - [cx, cy])) <= FAR * diag:
                p, q = hz
                y_h = p[1] + (q[1] - p[1]) * (sv[1][0] - p[0]) / (q[0] - p[0]) if abs(q[0] - p[0]) > 1e-6 else p[1]
                row["seam_vp_to_horizon_px"] = round(abs(float(sv[1][1] - y_h)), 1)
            srow[fname3] = row
            # draw: lines through sample points toward the seam VP, extended
            for p in pts[:: max(1, len(pts) // 8)]:
                d = dir_at(sv, p)
                a1, a2 = p - 3000 * d, p + 3000 * d
                ok, q1, q2 = cv2.clipLine((0, 0, w, h), tuple(int(v) for v in a1), tuple(int(v) for v in a2))
                if ok:
                    cv2.line(img, q1, q2, (255, 255, 255), 1, cv2.LINE_AA)
        seam_rows[key] = srow
    for s, fn in zip(segs, fam):
        cv2.line(img, (int(s[0]), int(s[1])), (int(s[2]), int(s[3])), FAMILY_COL[fn], 2 if fn != "rejected" else 1, cv2.LINE_AA)
    for k, pv in vps.items():
        col = FAMILY_COL[k]
        if pv[0] == "vp" and 0 <= pv[1][0] < w and 0 <= pv[1][1] < h:
            cv2.drawMarker(img, tuple(int(v) for v in pv[1]), col, cv2.MARKER_CROSS, 40, 3)
        else:
            d = dir_at(pv, np.array([cx, cy])) if pv[0] == "vp" else pv[1] / np.linalg.norm(pv[1])
            tx = min(((w - 1 - cx) / d[0]) if d[0] > 0 else ((-cx) / d[0]) if d[0] < 0 else 1e9,
                     ((h - 1 - cy) / d[1]) if d[1] > 0 else ((-cy) / d[1]) if d[1] < 0 else 1e9)
            qq = np.array([cx, cy]) + d * (tx - 12)
            cv2.arrowedLine(img, tuple(int(v) for v in qq - 70 * d), tuple(int(v) for v in qq), col, 4, tipLength=0.4)
        cv2.putText(img, f"{k} {label(pv)}", (10, 30 + 26 * list(vps).index(k)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    if hz is not None:
        p, q = hz
        d = (q - p) / np.linalg.norm(q - p)
        ok, q1, q2 = cv2.clipLine((0, 0, w, h), tuple(int(v) for v in p - 5000 * d), tuple(int(v) for v in p + 5000 * d))
        if ok:
            cv2.line(img, q1, q2, (255, 0, 255), 2)
    cv2.drawMarker(img, (int(cx), int(cy)), (255, 255, 0), cv2.MARKER_TILTED_CROSS, 24, 2)

    # ---- 5 bedroom: floor seams vs ceiling lines vs far end
    special = None
    if name == "bedroom":
        ceil = next((np.asarray(s.mask, bool) for s in b.surfaces if s.label == "ceiling"), None)
        cl = []
        if ceil is not None:
            band = cv2.dilate(ceil.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
            for s, fn in zip(segs, fam):
                mid = ((s[:2] + s[2:]) / 2).astype(int)
                if band[min(h - 1, mid[1]), min(w - 1, mid[0])] and fn != "vertical":
                    cl.append(s)
        cvp, _ = fit_vp(np.array(cl).reshape(-1, 4), w, h, rng) if len(cl) >= 2 else (None, None)
        far_obj = next((o for o in objects if o["id"] == "object_010"), None)
        far = None
        if far_obj:
            x0, y0, x1, y1 = far_obj["bbox"]
            far = [round((x0 + x1) / 2, 1), round((y0 + y1) / 2, 1)]
        floor_vp = None
        if "floor" in seam_rows and "refused" not in seam_rows["floor"]:
            for k3 in ("seam family 1 (const v)", "seam family 2 (const u)"):
                v = seam_rows["floor"][k3]["seam_vp"]
                if isinstance(v, list) and (floor_vp is None or abs(v[0] - cx) + abs(v[1] - cy) < abs(floor_vp[0] - cx) + abs(floor_vp[1] - cy)):
                    floor_vp = v
        special = {"floor_seams_converge_at": floor_vp,
                   "ceiling_lines": len(cl), "ceiling_lines_converge_at": label(vp_image(cvp)) if cvp is not None else None,
                   "photo_depth_vp": label(vps.get(depth_key)) if depth_key else None,
                   "far_end_window_centre (object_010 box)": far}
        for nm, pnt, col in (("floor seams", floor_vp, (255, 255, 255)), ("far end", far, (255, 255, 0))):
            if pnt and 0 <= pnt[0] < w and 0 <= pnt[1] < h:
                cv2.circle(img, (int(pnt[0]), int(pnt[1])), 10, col, 2)
                cv2.putText(img, nm, (int(pnt[0]) + 12, int(pnt[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)

    # ---- checks
    H = h
    checks = []
    def add(surface, check, value, ok):
        checks.append({"surface": surface, "check": check, "value": value, "result": "pass" if ok else ("fail" if ok is False else "n/a")})
    for fname in ("vertical", "horizontal-A", "horizontal-B"):
        cnt = families.get(fname, {}).get("count", 0)
        add("room", f"f: {fname} lines >= 3", cnt, cnt >= 3)
    for pr, e in ortho.items():
        add("room", f"d: orthogonality {pr} <= 3 deg", e, e <= 3)
    if "pitch_agreement_deg" in cam:
        add("room", "d: pitch vertical vs floor <= 2 deg", cam["pitch_agreement_deg"], cam["pitch_agreement_deg"] <= 2)
    add("room", "e: one camera for floor and every wall", [list(c) for c in sorted(cams_used)], len(cams_used) <= 1)
    if cams_used:
        fr = list(cams_used)[0][0]
        add("room", "e: render focal vs VP-solved focal (px)", [fr, round(f, 1)], abs(fr - f) <= 0.03 * f if fvals else None)
    for key, srow in seam_rows.items():
        if "refused" in srow:
            continue
        for k3 in ("seam family 1 (const v)", "seam family 2 (const u)"):
            r = srow.get(k3) or {}
            if not r.get("matched_photo_vp"):
                add(key, f"{k3}: matched VP", None, None)
                continue
            if key == "floor" and "seam_vp_to_horizon_px" in r:
                add(key, f"a: {k3} VP to horizon <= 2% H ({0.02 * H:.0f} px)", r["seam_vp_to_horizon_px"], r["seam_vp_to_horizon_px"] <= 0.02 * H)
            lim = "a" if key == "floor" else "b"
            if r["seam_vp_to_photo_vp_px"] is not None:
                add(key, f"{lim}: {k3} VP to {r['matched_photo_vp']} <= 3% diag ({0.03 * diag:.0f} px)", r["seam_vp_to_photo_vp_px"], r["seam_vp_to_photo_vp_px"] <= 0.03 * diag)
            else:
                add(key, f"{lim}: {k3} far VP, ray angle to {r['matched_photo_vp']} <= 1.5 deg", r["ray_angle_deg"], r["ray_angle_deg"] <= 1.5)
            add(key, f"c: {k3} seam angle error max <= 1.5 deg", r["seam_angle_error_deg_max"], r["seam_angle_error_deg_max"] <= 1.5)
    rep = {"room": name, "size": [w, h], "families": families, "camera": cam, "surfaces": seam_rows,
           "bedroom_convergence": special, "checks": checks}
    Image.fromarray(img).save(out / f"{name}__lines.png")
    (out / f"{name}__camera.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({"room": name, "families": {k: v["count"] for k, v in families.items()}, "camera": cam,
                      "special": special,
                      "fails": [c for c in checks if c["result"] == "fail"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
