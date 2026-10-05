"""
Camera / vanishing point / wall pipeline diagnosis for one room (read only).

    backend/.venv/bin/python tests/regression/camera_diagnose.py <name> <job_dir> <photo> <out_dir>

Geometry is recomputed with the CURRENT code in a scratch copy (the room's own
stored files are not touched). Writes <out_dir>/<name>/:
  lines.png       LSD lines coloured by the VP they point at (within 2 deg):
                  vertical / floor depth / floor second / each wall's own VP;
                  grey = no VP explains it; dark = too short; magenta = on a
                  removed object. VPs drawn (or arrowed at the border), horizon row.
  walls.png       every wall piece tinted, with id, faces, distance, and the
                  floor-junction it was fitted from vs where its plane puts it
  render_*.png    checker renders (floor, each wall) for the grid checks
  report.json     the numbers below
Numbers: camera (focal source, px, hfov, orientation, principal point, joint
decision); line groups and rejections; floor plane, camera height, grid angle
vs the depth axis; per wall piece: normal, angle to the floor's Manhattan
axes, distance (mm), split count per real wall, junction reprojection error
(px), ceiling-line angle error vs its VP, rendered grid direction vs its VP;
and whether a second render with a different tile changes anything outside
the tiles (cache / previous render used as base).
"""

from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "backend"), str(ROOT / "tests" / "regression")]

import perspective_engine as pe  # noqa: E402
import wall_direction as wd  # noqa: E402
from perspective_engine import masks as pe_masks  # noqa: E402
from reuse import load as reuse_load  # noqa: E402
from tiles_backend.perspective_engine import TileRequest, render_room  # noqa: E402
from tiles_backend.perspective_engine.camera.focal_estimate import focal_to_hfov_degrees  # noqa: E402
from tiles_backend.perspective_engine.camera.vp_from_mask import detect_lines_lsd  # noqa: E402
from tiles_backend.perspective_engine.room import geometry as room_geometry  # noqa: E402
from tiles_backend.perspective_engine.surface.wall import corner_cut, edge_layout  # noqa: E402
from tiles_backend.perspective_engine.surface.wall import junction_layout as jl  # noqa: E402

REQ = TileRequest(tile_width_mm=600, tile_height_mm=600, grout_mm=0,
                  room_width_mm=3658, room_length_mm=3658, room_height_mm=3048)   # scale only, for test renders
COLOURS = [(0, 200, 255), (0, 255, 0), (255, 128, 0), (255, 0, 255), (0, 128, 255), (255, 255, 0),
           (128, 0, 255), (0, 255, 255), (128, 255, 0), (255, 0, 128)]


def angle_to(seg, vp_h) -> float:
    x1, y1, x2, y2 = seg
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    if abs(vp_h[2]) < 1e-9:
        tx, ty = vp_h[0], vp_h[1]
    else:
        tx, ty = vp_h[0] / vp_h[2] - mx, vp_h[1] / vp_h[2] - my
    a = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180
    b = math.degrees(math.atan2(ty, tx)) % 180
    d = abs(a - b) % 180
    return min(d, 180 - d)


def draw_vp(img, vp_h, colour, label):
    h, w = img.shape[:2]
    if abs(vp_h[2]) < 1e-9:
        cv2.putText(img, f"{label}: at infinity", (10, 20 + 18 * COLOURS.index(colour) if colour in COLOURS else 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
        return
    x, y = vp_h[0] / vp_h[2], vp_h[1] / vp_h[2]
    if 0 <= x < w and 0 <= y < h:
        cv2.drawMarker(img, (int(x), int(y)), colour, cv2.MARKER_CROSS, 30, 3)
        cv2.putText(img, label, (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA)
    else:
        cxp, cyp = w / 2, h / 2
        dx, dy = x - cxp, y - cyp
        t = min((w / 2 - 15) / abs(dx) if dx else 1e9, (h / 2 - 15) / abs(dy) if dy else 1e9)
        px, py = int(cxp + dx * t), int(cyp + dy * t)
        cv2.arrowedLine(img, (int(cxp + dx * t * 0.85), int(cyp + dy * t * 0.85)), (px, py), colour, 3, tipLength=0.4)
        cv2.putText(img, f"{label} ({x:.0f},{y:.0f})", (max(5, min(px - 120, w - 260)), max(20, min(py + 20, h - 10))),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2, cv2.LINE_AA)


def main() -> int:
    name, job, photo, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
    out = out / name
    out.mkdir(parents=True, exist_ok=True)
    seg = out / "scratch" / "segments"
    shutil.rmtree(seg.parent, ignore_errors=True)
    shutil.copytree(job / "segments", seg)
    for stored in ("floor", "wall", "room"):
        shutil.rmtree(seg / stored, ignore_errors=True)
    bundle = reuse_load(job)
    fm, wm = pe.load(seg)
    geo = pe.ensure_geometry(seg, bundle.room, fm, wm, clean=bundle.clean, photo_bytes=photo.read_bytes())
    floor, wall, _ = pe_masks.prepare(fm, wm, bundle.clean.shape[:2])
    h, w = floor.shape
    diag = math.hypot(h, w)
    cam = geo["camera"]
    f = float(cam["focal_px"])
    cx, cy = (float(v) for v in cam["principal_point"])
    joint = cam.get("joint") or {}
    frame = geo.get("room_frame") or {}
    fvps = geo["floor"].get("vanishing_points") or {}
    objects = cv2.imread(str(seg / "ALL_OBJECTS.png"), cv2.IMREAD_UNCHANGED)
    objects = cv2.resize((objects[..., 3] > 127).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0

    rep = {"room": name, "size": [w, h], "orientation": "portrait" if h > w else "landscape",
           "camera": {"focal_source": cam.get("focal_source"), "focal_px": round(f, 1),
                      "hfov_deg": round(focal_to_hfov_degrees(f, w), 1), "principal_point": [round(cx, 1), round(cy, 1)],
                      "confidence": cam.get("confidence"), "horizon_y": joint.get("horizon_y", fvps.get("horizon_y")),
                      "vertical_lines": joint.get("vertical_lines"), "decision": joint.get("decision")}}

    # ---- 1. lines by VP group
    groups = []
    vert = frame.get("vertical_vp") or {}
    if vert.get("homogeneous"):
        groups.append(("vertical", np.array(vert["homogeneous"], float)))
    if fvps.get("vp1") is not None:
        groups.append(("floor depth", np.array([*fvps["vp1"], 1.0])))
    if fvps.get("vp2") is not None:
        groups.append(("floor 2nd", np.array([*fvps["vp2"], 1.0])))
    for wl in geo["walls"]:
        vp = ((wl.get("direction") or {}).get("vanishing_point") or {}).get("homogeneous")
        if vp is not None:
            groups.append((wl["id"], np.array(vp, float)))
    clean_bgr = cv2.cvtColor(bundle.clean, cv2.COLOR_RGB2BGR)
    lines = np.asarray(detect_lines_lsd(cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2GRAY))).reshape(-1, 4)
    canvas = (clean_bgr * 0.45).astype(np.uint8)
    counts = {g[0]: 0 for g in groups}
    rejected = {"too short (< 2% of diagonal)": 0, "on a removed object": 0, "no VP within 2 deg": 0}
    for seg_ in lines:
        x1, y1, x2, y2 = seg_
        length = math.hypot(x2 - x1, y2 - y1)
        mx, my = int((x1 + x2) / 2), int((y1 + y2) / 2)
        if length < 0.02 * diag:
            rejected["too short (< 2% of diagonal)"] += 1
            cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), (60, 60, 60), 1)
            continue
        if objects[min(my, h - 1), min(mx, w - 1)]:
            rejected["on a removed object"] += 1
            cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), (180, 0, 180), 1)
            continue
        res = [angle_to(seg_, v) for _, v in groups]
        k = int(np.argmin(res)) if res else -1
        if k < 0 or res[k] > 2.0:
            rejected["no VP within 2 deg"] += 1
            cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), (150, 150, 150), 1)
            continue
        counts[groups[k][0]] += 1
        cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), COLOURS[k % len(COLOURS)], 2)
    for k, (gname, v) in enumerate(groups):
        draw_vp(canvas, v, COLOURS[k % len(COLOURS)], gname)
    hz = rep["camera"]["horizon_y"]
    if hz is not None:
        cv2.line(canvas, (0, int(hz)), (w, int(hz)), (255, 255, 255), 1)
        cv2.putText(canvas, f"horizon y={hz}", (10, max(15, int(hz) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imwrite(str(out / "lines.png"), canvas)
    rep["lines"] = {"total": int(len(lines)), "by_vp": counts, "rejected": rejected,
                    "vps": {g: ([round(x, 1) for x in (v[:2] / v[2])] if abs(v[2]) > 1e-9 else "infinity") for g, v in groups}}

    # ---- 3. floor
    room = room_geometry.solve(frame, {"width": None, "length": None, "height": None}, floor_mask=floor) if frame else {}
    rep["floor"] = {"status": geo["floor"].get("status"), "plane": geo["floor"].get("plane"),
                    "camera_height_mm": room.get("camera_height_mm"), "camera_height_source": room.get("camera_height_source"),
                    "floor_direction_deg": (geo["floor"].get("camera") or {}).get("floor_direction_deg"),
                    "room_mm": [room.get("width_mm"), room.get("length_mm"), room.get("height_mm")],
                    "room_confidence": room.get("confidence")}
    tile = wd._checker()
    try:
        rf = render_room(bundle.clean, floor, wall, tile, "floor", REQ, geometry=geo)
        cv2.imwrite(str(out / "render_floor.png"), cv2.cvtColor(rf.image, cv2.COLOR_RGB2BGR))
        segs = wd._render_lines(rf.image, rf.floor_tiled)
        if fvps.get("vp1") is not None and len(segs) >= 5:
            dv = np.array([*fvps["vp1"], 1.0])
            res = wd._residuals(segs, dv)
            along = res[res <= 15]
            rep["floor"]["grid_vs_depth_axis_deg"] = {"median": round(float(np.median(along)), 2) if len(along) else None,
                                                     "lines": int(len(along))}
    except Exception as e:  # noqa: BLE001
        rep["floor"]["render"] = f"refused: {str(e)[:100]}"

    # ---- 4/5/7. walls
    X = np.array((frame.get("axes_camera") or {}).get("X", [1, 0, 0]), float)
    Z = np.array((frame.get("axes_camera") or {}).get("Z", [0, 0, 1]), float)
    Yup = -np.array((frame.get("axes_camera") or {}).get("Y", [0, -1, 0]), float)   # camera y points down
    planes = room.get("wall_planes_mm") or {}
    fplane = room.get("floor_plane_mm") or [0, -1, 0, room.get("camera_height_mm") or 1500]
    masks = geo["_wall_masks"]
    wcanvas = (clean_bgr * 0.5).astype(np.uint8)
    rep["walls"] = {}
    ray = lambda px, py: np.array([(px - cx) / f, (py - cy) / f, 1.0])  # noqa: E731
    for k, wl in enumerate(geo["walls"]):
        m = masks[int(wl["index"])]
        d = wl.get("direction") or {}
        row = {"pixels": int(m.sum()), "faces": d.get("faces"), "source": d.get("source"), "validated": d.get("validated")}
        colour = COLOURS[k % len(COLOURS)]
        wcanvas[m] = (wcanvas[m] * 0.4 + np.array(colour) * 0.6).astype(np.uint8)
        if d.get("normal"):
            n = np.array(d["normal"], float)
            n = n / np.linalg.norm(n)
            row["normal"] = [round(v, 3) for v in n]
            ax = {"X": math.degrees(math.acos(min(1, abs(float(n @ X))))),
                  "Z": math.degrees(math.acos(min(1, abs(float(n @ Z))))),
                  "up": math.degrees(math.acos(min(1, abs(float(n @ Yup)))))}
            row["angle_to_axes_deg"] = {kk: round(v, 2) for kk, v in ax.items()}
        pl = planes.get(wl["id"])
        J = jl.junction_points(floor, m, objects)
        C = edge_layout.ceiling_points(m, objects)
        if pl is not None:
            pn, pd = np.array(pl[:3], float), float(pl[3])
            row["distance_mm"] = round(pd, 1)
            fn, fd = np.array(fplane[:3], float), float(fplane[3])
            errs = []
            for px, py in J:
                r = ray(px, py)
                t = -pd / float(pn @ r) if abs(float(pn @ r)) > 1e-9 else None
                if t is None or t <= 0:
                    continue
                Q = r * t                                       # on the wall plane
                # drop Q to the floor along the up axis, and project back
                s = (-(fn @ Q) - fd) / float(fn @ Yup) if abs(float(fn @ Yup)) > 1e-9 else 0.0
                Qf = Q + s * Yup
                if Qf[2] <= 0:
                    continue
                u, v = f * Qf[0] / Qf[2] + cx, f * Qf[1] / Qf[2] + cy
                errs.append(math.hypot(u - px, v - py))
            if errs:
                row["junction_reprojection_px"] = {"median": round(float(np.median(errs)), 2),
                                                   "p90": round(float(np.percentile(errs, 90)), 2), "points": len(errs)}
        else:
            row["distance_mm"] = None
            row["plane"] = "none placed (no usable floor junction)"
        vp = (d.get("vanishing_point") or {}).get("homogeneous")
        if vp is not None and len(C) >= 0.03 * w:
            runs = corner_cut.runs(C, w, diag)
            if runs:
                errs = [angle_to((r["pts"][0, 0], r["pts"][0, 1], r["pts"][-1, 0], r["pts"][-1, 1]), np.array(vp, float)) for r in runs]
                row["ceiling_line_vs_vp_deg"] = round(float(np.median(errs)), 2)
        row["junction_points"], row["ceiling_points"] = int(len(J)), int(len(C))
        if d:
            try:
                rw = render_room(bundle.clean, floor, wall, tile, "wall", REQ, wall_index=wl["index"], geometry=geo)
                segs = wd._render_lines(rw.image, rw.wall_tiled)
                cv2.imwrite(str(out / f"render_{wl['id']}.png"), cv2.cvtColor(rw.image, cv2.COLOR_RGB2BGR))
                if vp is not None and len(segs) >= 5:
                    row["grid_vs_own_vp_deg"] = round(float(np.median(wd._residuals(segs, np.array(vp, float)))), 2)
                else:
                    row["grid_vs_own_vp_deg"] = f"{len(segs)} rendered lines (too few)"
            except Exception as e:  # noqa: BLE001
                row["grid_vs_own_vp_deg"] = f"render refused: {str(e)[:80]}"
        ys, xs = np.nonzero(m)
        cv2.putText(wcanvas, f"{wl['id']} {d.get('faces')} d={row.get('distance_mm')}", (int(xs.mean()) - 60, int(ys.mean())),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        for px, py in J[:: max(1, len(J) // 200)]:
            cv2.circle(wcanvas, (int(px), int(py)), 2, (0, 255, 255), -1)
        rep["walls"][wl["id"]] = row
    cv2.imwrite(str(out / "walls.png"), wcanvas)

    # split count: pieces per real wall (same facing, normals within 5 deg, one floor/ceiling line)
    ids = [wl["id"] for wl in geo["walls"]]
    parent = {i: i for i in ids}

    def root(x):
        while parent[x] != x:
            x = parent[x]
        return x
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = geo["walls"][i], geo["walls"][j]
            na, nb = (a.get("direction") or {}).get("normal"), (b.get("direction") or {}).get("normal")
            if not na or not nb or (a["direction"].get("faces") != b["direction"].get("faces")):
                continue
            na, nb = np.array(na) / np.linalg.norm(na), np.array(nb) / np.linalg.norm(nb)
            if math.degrees(math.acos(min(1, abs(float(na @ nb))))) > 5:
                continue
            if corner_cut._coplanar(masks[int(a["index"])], masks[int(b["index"])], floor, objects, w, diag):
                parent[root(ids[j])] = root(ids[i])
    real = {}
    for i in ids:
        real.setdefault(root(i), []).append(i)
    rep["real_walls"] = {"pieces": len(ids), "real_walls_by_same_plane_evidence": len(real),
                         "groups": [g for g in real.values()]}

    # ---- 6. does a previous render leak into the next one?
    try:
        teal = np.zeros((256, 256, 3), np.uint8)
        teal[:] = (0, 160, 160)
        grey = np.full((256, 256, 3), 128, np.uint8)
        render_room(bundle.clean, floor, wall, teal, "floor", REQ, geometry=geo)
        after_teal = render_room(bundle.clean, floor, wall, grey, "floor", REQ, geometry=geo).image
        fresh = render_room(bundle.clean, floor, wall, grey, "floor", REQ, geometry=geo).image
        teal_px = int((np.abs(after_teal.astype(int) - np.array([0, 160, 160])).sum(axis=2) < 40).sum())
        rep["previous_render_leak"] = {"grey render after a teal render == a fresh grey render": bool(np.array_equal(after_teal, fresh)),
                                       "teal-coloured pixels in the grey render": teal_px,
                                       "base": "the clean room (reuse/clean.png) on every render; never a previous result"}
    except Exception as e:  # noqa: BLE001
        rep["previous_render_leak"] = f"not tested: {str(e)[:80]}"

    (out / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps(rep, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
