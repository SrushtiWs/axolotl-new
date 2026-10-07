"""
Camera regression: the camera and room geometry DETECTED, never loaded.

    backend/.venv/bin/python tests/regression/camera_regression.py check    <out_dir> [<baseline_dir>]
    backend/.venv/bin/python tests/regression/camera_regression.py selftest <run_dir>
    backend/.venv/bin/python tests/regression/camera_regression.py save     <run_dir> <baseline_dir>

check     For every room in baseline_rooms.ROOMS: the job is copied to a temp
          folder, its stored floor/ wall/ room/ geometry (and provenance) is
          removed, and ensure_geometry detects it again -- so the camera code
          itself runs. One process per room. Writes <room>.json (camera record
          + independent evidence + status) and <room>__camera.png (overlay).
          With a baseline: compares every field and exits 1 on any FAIL.
selftest  Negative controls: each perturbed record must FAIL against the
          unperturbed one, the unperturbed must PASS, and a moved vanishing
          point must lower the evidence status. Exit 0 only if all hold.
save      Copies a reviewed run's <room>.json files to a NEW baseline folder
          (refuses an existing one). Run it only after the overlays are approved.

Evidence is independent of the detector: the DeepLSD kept lines (the L1 set,
backend/perspective_engine/strong_lines.py, cached per photo) that point within
DEPTH_FAMILY_DEG of the floor's depth VP, and the floor-wall junction runs.
Tolerances and status limits are relative to the image (diagonal, height) or in
degrees; none is per room.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "backend"), str(ROOT / "tests/regression")]

from baseline_rooms import ROOMS  # noqa: E402

# ---- comparison tolerances (relative to the image, or degrees) -------------
VP_TOL = 0.005            # depth VP / principal point move, share of the diagonal
HORIZON_TOL = 0.005       # horizon row move, share of the height
FOCAL_TOL = 0.01          # relative focal change
ANGLE_TOL_DEG = 0.5       # pitch, roll, wall yaw
WALL_SHARE_TOL = 0.005    # a wall's pixel share of the frame
CATEGORICAL = ("floor_status", "vp_origin", "focal_source", "joint_confidence", "level", "walls")

# ---- evidence status limits --------------------------------------------------
DEPTH_FAMILY_DEG = 20.0   # a line points at the depth VP within this angle
RECEDING_MIN_DEG = 8.0    # flatter than this is image-horizontal, not a depth line
PLUMB_DEG = 10.0          # this close to vertical is a vertical line
CONFIRMED_DEG = 2.0       # mean deviation of the depth lines for "confirmed"
APPROX_DEG = 3.0          # ... and above this "needs_review"
MIN_DEPTH_LINES = 3       # fewer independent depth lines: "needs_review"
MEASURED_FOCAL = ("exif", "vp", "vp-pairs")
SAME_PLANE_DEG = 5.0      # two walls facing the same way with yaw this close: one plane split in pieces


# ---------------------------------------------------------------------------
# Geometry helpers

def _angle_to(seg, vp) -> float:
    x1, y1, x2, y2 = seg
    dx, dy = x2 - x1, y2 - y1
    tx, ty = vp[0] - (x1 + x2) / 2, vp[1] - (y1 + y2) / 2
    n = math.hypot(dx, dy) * math.hypot(tx, ty)
    return math.degrees(math.acos(min(1.0, abs(dx * tx + dy * ty) / n))) if n > 0 else 90.0


def _length(s) -> float:
    return math.hypot(s[2] - s[0], s[3] - s[1])


def _receding(s) -> bool:
    a = abs(math.degrees(math.atan2(s[3] - s[1], s[2] - s[0]))) % 180
    return min(a, 180 - a) >= RECEDING_MIN_DEG and abs(a - 90) >= PLUMB_DEG


def evidence(vp, lines, junction) -> dict:
    """Deviation of the independent depth lines from `vp`, and the status it supports."""
    if vp is None:
        return {"depth_lines": 0, "mean_dev_deg": None, "junction_runs": 0, "junction_dev_deg": None,
                "vp_status": "needs_review", "vp_reason": "no floor depth VP"}

    def mean(segs):
        d = [(_angle_to(s, vp), _length(s)) for s in segs]
        d = [(a, l) for a, l in d if a <= DEPTH_FAMILY_DEG]
        if not d:
            return 0, None
        return len(d), round(sum(a * l for a, l in d) / sum(l for _, l in d), 2)

    n, dev = mean([s for s in lines if _receding(s)])
    nj, devj = mean(junction)
    if n < MIN_DEPTH_LINES:
        status, reason = "needs_review", f"only {n} independent depth lines (< {MIN_DEPTH_LINES})"
    elif dev > APPROX_DEG:
        status, reason = "needs_review", f"depth lines deviate {dev} deg (> {APPROX_DEG})"
    elif dev > CONFIRMED_DEG:
        status, reason = "approximate", f"depth lines deviate {dev} deg (> {CONFIRMED_DEG})"
    else:
        status, reason = "confirmed", f"{n} depth lines within {dev} deg"
    return {"depth_lines": n, "mean_dev_deg": dev, "junction_runs": nj, "junction_dev_deg": devj,
            "vp_status": status, "vp_reason": reason}


def statuses(rec: dict) -> dict:
    """Focal and level status, and the overall (worst) one, with reasons."""
    order = ("confirmed", "approximate", "needs_review")
    focal = (("confirmed", f"focal from {rec['focal_source']}") if rec["focal_source"] in MEASURED_FOCAL
             else ("approximate", f"focal from {rec['focal_source']} (not measured)"))
    v = rec.get("vertical_lines") or {}
    if rec.get("level"):
        level = ("confirmed", "vertical lines say level")
    elif not v.get("reliable"):
        level = ("approximate", "vertical lines unreliable: level not verified")
    else:
        level = ("needs_review", f"pitched camera (pitch {v.get('pitch_deg')}, roll {v.get('roll_deg')}): walls built level")
    walls = rec.get("wall_records") or []
    unvalidated = [w["id"] for w in walls if not w.get("validated", True)]
    # One plane split in pieces: same faces, same normal, touching AND coplanar
    # (parallel walls at different depths, e.g. a pillar face and the wall
    # behind it, share faces and normal but are not one plane).
    split = [tuple(p) for p in rec.get("split_pairs", [])]
    if split or unvalidated:
        why = []
        if split:
            why.append("split into several walls with the same faces and normal: " + ", ".join(f"{a}+{b}" for a, b in split))
        if unvalidated:
            why.append("validated=False: " + ", ".join(unvalidated))
        wall_part = ("needs_review", "; ".join(why))
    else:
        wall_part = ("confirmed", "one wall per plane, every direction validated")
    parts = {"vp": (rec["evidence"]["vp_status"], rec["evidence"]["vp_reason"]), "focal": focal, "level": level,
             "walls": wall_part}
    worst = max(parts.values(), key=lambda p: order.index(p[0]))[0]
    return {"status": worst, "reasons": {k: f"{s}: {r}" for k, (s, r) in parts.items()}}


# ---------------------------------------------------------------------------
# One room, fresh detection (runs in its own process)

def one(name: str, out: Path) -> None:
    import cv2
    import numpy as np
    from PIL import Image

    import perspective_engine as pe
    import reuse
    import surfaces
    from perspective_engine import masks, strong_lines as sl
    from perspective_engine.render import _objects_mask
    from tiles_backend.perspective_engine.camera import boundary_vp

    with tempfile.TemporaryDirectory() as tmp:
        job = Path(tmp) / ROOMS[name].name
        shutil.copytree(ROOMS[name], job)
        seg = job / "segments"
        for stored in ("floor", "wall", "room", "geometry_provenance.json", "geometry_superseded"):
            p = seg / stored
            if p.is_dir():
                shutil.rmtree(p)
            elif p.exists():
                p.unlink()
        b = reuse.load(job)
        fl, wl = pe.load(seg)
        geo = pe.ensure_geometry(seg, b.room, fl, wl, clean=b.clean)
        h, w = b.room.shape[:2]
        floor, wall, _ = masks.prepare(fl, wl, (h, w))
        objects = _objects_mask(seg, (h, w))

    room = b.room
    diag = math.hypot(h, w)
    props = np.asarray(b.props, bool)
    res = sl.strong_lines(room, props=props, floor=floor, labels=surfaces.label_map(room))
    lines = [[round(float(v), 1) for v in s] for s in np.asarray(res["lines"], float).reshape(-1, 4)]
    junction = [[round(float(v), 1) for v in s] for s, src in boundary_vp.edge_runs(floor, wall, objects)
                if src == "floor-junction"]

    cam = geo["camera"]
    joint = cam.get("joint") or {}
    fvps = geo["floor"].get("vanishing_points") or {}
    vp = fvps.get("depth_vp")
    walls = []
    for wl_ in geo["walls"]:
        d = wl_.get("direction") or {}
        walls.append({"id": wl_["id"], "index": int(wl_["index"]), "faces": d.get("faces"),
                      "source": d.get("source"), "validated": d.get("validated", True),
                      "yaw_deg": None if d.get("yaw_deg") is None else round(float(d["yaw_deg"]), 2),
                      "share": round(float(wl_["pixels"]) / (h * w), 4)})
    # Pairs that are one plane in pieces, judged by the wall code's own tests.
    from tiles_backend.perspective_engine.surface.wall import corner_cut as cc
    wall_masks = geo.get("_wall_masks") or {}
    radius = max(2, int(round(cc.TOUCH * diag)))
    split_pairs = []
    for i, a in enumerate(walls):
        for b in walls[i + 1:]:
            if not a["faces"] or a["faces"] != b["faces"] or a["yaw_deg"] is None or b["yaw_deg"] is None:
                continue
            if abs(a["yaw_deg"] - b["yaw_deg"]) > SAME_PLANE_DEG:
                continue
            ma, mb = (np.asarray(wall_masks.get(x["index"]), bool) for x in (a, b))
            if ma.shape != (h, w) or mb.shape != (h, w):
                continue
            if cc._touch(ma, mb, radius) and cc._coplanar(ma, mb, floor, objects, w, diag) is not None:
                split_pairs.append([a["id"], b["id"]])
    rec = {
        "room": name, "size": [w, h], "diag": round(diag, 1),
        "split_pairs": split_pairs,
        "floor_status": geo["floor"].get("status"),
        "vp_origin": fvps.get("origin", "floor-lines"),
        "depth_vp": None if vp is None else [round(float(v), 1) for v in vp],
        "horizon_y": None if fvps.get("horizon_y") is None else round(float(fvps["horizon_y"]), 1),
        "focal_px": round(float(cam["focal_px"]), 1),
        "focal_source": cam.get("focal_source"),
        "principal_point": [round(float(v), 1) for v in cam["principal_point"]],
        "joint_confidence": joint.get("confidence"),
        "joint_decision": joint.get("decision"),
        "level": joint.get("level"),
        "vertical_lines": joint.get("vertical_lines"),
        "walls": len(walls),
        "wall_records": walls,
        "evidence": evidence(vp, lines, junction),
        "lines": lines,
        "junction_runs": junction,
    }
    rec.update(statuses(rec))

    # Overlay: wall split coloured, floor tint, depth rays from the VP clipped to
    # the floor, DeepLSD depth lines green (others grey), junction runs white,
    # horizon dashed, VP red, status text.
    img = (room.astype(np.float32) * 0.45)
    palette = [(255, 140, 40), (200, 80, 255), (255, 220, 60), (80, 200, 255), (255, 90, 120), (120, 255, 140)]
    wall_masks = geo.get("_wall_masks") or {}
    for k, (idx, m) in enumerate(sorted(wall_masks.items())):
        m = np.asarray(m, bool)
        if m.shape != (h, w):
            m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        img[m] = img[m] * 0.6 + np.array(palette[k % len(palette)]) * 0.4
    img[floor] = img[floor] * 0.7 + np.array([46, 168, 255]) * 0.3
    img = img.astype(np.uint8)
    t = max(1, round(diag / 900))
    if vp is not None:
        rays = np.zeros((h, w), np.uint8)
        for x in np.linspace(-w, 2 * w, 25):
            cv2.line(rays, (int(vp[0]), int(vp[1])), (int(x), int(3 * h)), 1, t, cv2.LINE_AA)
        img[(rays > 0) & floor] = (255, 255, 0)
    for s in lines:
        dep = vp is not None and _receding(s) and _angle_to(s, vp) <= DEPTH_FAMILY_DEG
        cv2.line(img, (int(s[0]), int(s[1])), (int(s[2]), int(s[3])), (0, 255, 0) if dep else (150, 150, 150),
                 2 * t if dep else t, cv2.LINE_AA)
    for s in junction:
        cv2.line(img, (int(s[0]), int(s[1])), (int(s[2]), int(s[3])), (255, 255, 255), 3 * t, cv2.LINE_AA)
    if rec["horizon_y"] is not None and 0 <= rec["horizon_y"] < h:
        y = int(rec["horizon_y"])
        for x in range(0, w, 24 * t):
            cv2.line(img, (x, y), (min(w - 1, x + 12 * t), y), (255, 60, 60), t)
    pad = 0
    if vp is not None and not (0 <= vp[0] < w and 0 <= vp[1] < h):
        pad = int(min(diag, max(-vp[0], vp[0] - w, -vp[1], vp[1] - h, 0))) + 20
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(30, 30, 30))
    if vp is not None and -pad <= vp[0] < w + pad and -pad <= vp[1] < h + pad:
        cv2.circle(img, (int(vp[0]) + pad, int(vp[1]) + pad), 6 * t, (255, 0, 0), -1)
    for i, wr in enumerate(walls):
        m = wall_masks.get(wr["index"])
        if m is None:
            continue
        ys, xs = np.nonzero(np.asarray(m, bool))
        if len(xs):
            cv2.putText(img, f"{wr['id']} {wr['faces']} yaw {wr['yaw_deg']}", (int(xs.mean()) + pad, int(ys.mean()) + pad),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5 * t, (255, 255, 255), t, cv2.LINE_AA)
    txt = [f"status: {rec['status']}"] + [f"{k}: {v}" for k, v in rec["reasons"].items()]
    for i, s in enumerate(txt):
        cv2.putText(img, s[:110], (10 + pad, 22 * t * (i + 1) + pad), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * t,
                    (255, 255, 255), t, cv2.LINE_AA)
    Image.fromarray(img).save(out / f"{name}__camera.png")
    (out / f"{name}.json").write_text(json.dumps(rec, indent=1))


# ---------------------------------------------------------------------------
# Comparison

def compare(rec: dict, base: dict) -> list[str]:
    """Every field outside its tolerance, as a readable failure."""
    fails = []
    diag, h = base["diag"], base["size"][1]
    for k in CATEGORICAL:
        if rec.get(k) != base.get(k):
            fails.append(f"{k}: {base.get(k)!r} -> {rec.get(k)!r}")

    def moved(a, b, tol, label):
        if (a is None) != (b is None):
            fails.append(f"{label}: {b} -> {a}")
        elif a is not None and math.dist(a, b) > tol:
            fails.append(f"{label}: {b} -> {a} ({math.dist(a, b):.1f} px > {tol:.1f})")

    moved(rec["depth_vp"], base["depth_vp"], VP_TOL * diag, "depth_vp")
    moved(rec["principal_point"], base["principal_point"], VP_TOL * diag, "principal_point")
    if (rec["horizon_y"] is None) != (base["horizon_y"] is None) or (
            rec["horizon_y"] is not None and abs(rec["horizon_y"] - base["horizon_y"]) > HORIZON_TOL * h):
        fails.append(f"horizon_y: {base['horizon_y']} -> {rec['horizon_y']}")
    if abs(rec["focal_px"] / base["focal_px"] - 1) > FOCAL_TOL:
        fails.append(f"focal_px: {base['focal_px']} -> {rec['focal_px']}")
    for k in ("pitch_deg", "roll_deg"):
        a, b = (rec.get("vertical_lines") or {}).get(k), (base.get("vertical_lines") or {}).get(k)
        if (a is None) != (b is None) or (a is not None and abs(a - b) > ANGLE_TOL_DEG):
            fails.append(f"vertical {k}: {b} -> {a}")
    by_index = {w["index"]: w for w in rec["wall_records"]}
    for bw in base["wall_records"]:
        rw = by_index.get(bw["index"])
        if rw is None:
            fails.append(f"{bw['id']}: missing")
            continue
        for k in ("faces", "source", "validated"):
            if rw.get(k) != bw.get(k):
                fails.append(f"{bw['id']} {k}: {bw.get(k)!r} -> {rw.get(k)!r}")
        if (rw["yaw_deg"] is None) != (bw["yaw_deg"] is None) or (
                rw["yaw_deg"] is not None and abs(rw["yaw_deg"] - bw["yaw_deg"]) > ANGLE_TOL_DEG):
            fails.append(f"{bw['id']} yaw: {bw['yaw_deg']} -> {rw['yaw_deg']}")
        if abs(rw["share"] - bw["share"]) > WALL_SHARE_TOL:
            fails.append(f"{bw['id']} share: {bw['share']} -> {rw['share']}")
    if rec["status"] != base["status"]:
        fails.append(f"status: {base['status']} -> {rec['status']}")
    return fails


# ---------------------------------------------------------------------------
# Modes

def check(out: Path, baseline: Path | None) -> int:
    out.mkdir(parents=True, exist_ok=True)
    for name in ROOMS:
        if (out / f"{name}.json").exists():
            continue
        r = subprocess.run([sys.executable, __file__, "_one", name, str(out)], capture_output=True, text=True)
        if r.returncode:
            print(f"{name}: detection FAILED\n{r.stderr[-1500:]}")
            return 2
    print(f"{'room':14s} {'status':13s} {'VP':>16s} {'dev':>6s} {'lines':>5s} {'focal':>8s} {'source':9s} walls  baseline")
    failed = False
    for name in ROOMS:
        rec = json.loads((out / f"{name}.json").read_text())
        verdict = "-"
        if baseline is not None:
            fails = compare(rec, json.loads((baseline / f"{name}.json").read_text()))
            verdict = "PASS" if not fails else "FAIL: " + "; ".join(fails)
            failed |= bool(fails)
        e = rec["evidence"]
        print(f"{name:14s} {rec['status']:13s} {str(rec['depth_vp']):>16s} {str(e['mean_dev_deg']):>6s} "
              f"{e['depth_lines']:>5d} {rec['focal_px']:>8.1f} {str(rec['focal_source']):9s} {rec['walls']:>5d}  {verdict}")
    review = [(n, json.loads((out / f"{n}.json").read_text())) for n in ROOMS]
    review = [(n, r) for n, r in review if r["status"] == "needs_review"]
    if review:
        print("\nneeds_review:")
        for n, r in review:
            print(f"  {n}: " + "; ".join(v for v in r["reasons"].values() if v.startswith("needs_review")))
    return 1 if failed else 0


def selftest(run: Path) -> int:
    import copy
    ok = True
    for name in ROOMS:
        base = json.loads((run / f"{name}.json").read_text())
        diag = base["diag"]
        cases = [("unchanged", lambda r: None, False)]
        if base["depth_vp"] is not None:
            cases.append(("depth VP moved 2x tol", lambda r: r.__setitem__(
                "depth_vp", [r["depth_vp"][0] + 2 * VP_TOL * diag, r["depth_vp"][1]]), True))
        cases += [
            ("focal +2x tol", lambda r: r.__setitem__("focal_px", r["focal_px"] * (1 + 2 * FOCAL_TOL)), True),
            ("focal source changed", lambda r: r.__setitem__("focal_source", "changed"), True),
            ("floor status changed", lambda r: r.__setitem__("floor_status", "changed"), True),
            ("one wall more", lambda r: r.__setitem__("walls", r["walls"] + 1), True),
        ]
        if base["wall_records"] and base["wall_records"][0]["yaw_deg"] is not None:
            cases.append(("wall yaw +2x tol", lambda r: r["wall_records"][0].__setitem__(
                "yaw_deg", r["wall_records"][0]["yaw_deg"] + 2 * ANGLE_TOL_DEG), True))
        for label, mutate, should_fail in cases:
            rec = copy.deepcopy(base)
            mutate(rec)
            failed = bool(compare(rec, base))
            good = failed == should_fail
            ok &= good
            if not good:
                print(f"{name}: '{label}' expected {'FAIL' if should_fail else 'PASS'}, got {'FAIL' if failed else 'PASS'}")
        # The evidence must notice a wrong vanishing point: moving it well off the
        # depth lines may not keep or improve the status.
        if base["depth_vp"] is not None and base["evidence"]["depth_lines"] >= MIN_DEPTH_LINES:
            off = [base["depth_vp"][0] + 0.1 * diag, base["depth_vp"][1]]
            e = evidence(off, base["lines"], base["junction_runs"])
            order = ("confirmed", "approximate", "needs_review")
            worse = e["mean_dev_deg"] is None or e["mean_dev_deg"] > (base["evidence"]["mean_dev_deg"] or 0)
            not_better = order.index(e["vp_status"]) >= order.index(base["evidence"]["vp_status"])
            if not (worse and not_better):
                ok = False
                print(f"{name}: VP moved 10% of diagonal did not worsen the evidence ({base['evidence']} -> {e})")
    print("selftest:", "ALL CONTROLS HOLD" if ok else "FAILED")
    return 0 if ok else 1


def save(run: Path, baseline: Path) -> int:
    if baseline.exists():
        print(f"{baseline} exists; a baseline is never overwritten")
        return 1
    baseline.mkdir(parents=True)
    for name in ROOMS:
        shutil.copyfile(run / f"{name}.json", baseline / f"{name}.json")
    print(f"saved {len(ROOMS)} rooms to {baseline}")
    return 0


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "_one":
        one(sys.argv[2], Path(sys.argv[3]))
        sys.exit(0)
    if mode == "check":
        sys.exit(check(Path(sys.argv[2]), Path(sys.argv[3]) if len(sys.argv) > 3 else None))
    if mode == "selftest":
        sys.exit(selftest(Path(sys.argv[2])))
    if mode == "save":
        sys.exit(save(Path(sys.argv[2]), Path(sys.argv[3])))
    sys.exit(f"unknown mode {mode}")
