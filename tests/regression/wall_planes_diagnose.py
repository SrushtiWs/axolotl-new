"""
Wall planes of one room, diagnosed (read only).

    backend/.venv/bin/python tests/regression/wall_planes_diagnose.py <name> <job_dir> <out_dir> [deeplsd.json]

Geometry is recomputed with the current code in a scratch copy, with the
pillar merge switched OFF so the split's raw pieces are seen. Reports and images:

  1 lines        OpenCV LSD and DeepLSD (if given): total, per VP family
                 (vertical / depth / lateral), rejected with the reason
  2 corners      every corner candidate: x, and its witnesses (floor bend,
                 ceiling bend, run end at a hidden gap, LSD vertical edge,
                 DeepLSD vertical edge); count vs the room's planes
  3 pieces       the wall pieces' column ranges and which corners have no
                 piece boundary (planes missed / merged); a column range
                 can be checked with --cols x0-x1
  4 planes       per piece: facing, normal, distance (mm), placed or not;
                 per shared corner: horizontal seam height mismatch (px)
                 between the two walls rendered alone
  5 old pixels   wall-mask / floor-mask gap (skirting left untiled) and the
                 old surface's fine detail carried by the shading
  6 floor grids  the floor rendered with a flat grey tile and no grout: any
                 long dark lines left are the OLD floor's joints
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
from camera_diagnose import angle_to  # noqa: E402
from perspective_engine import masks as pe_masks  # noqa: E402
from reuse import load as reuse_load  # noqa: E402
from tiles_backend.perspective_engine import TileRequest, render_room  # noqa: E402
from tiles_backend.perspective_engine.camera.vp_from_mask import detect_lines_lsd  # noqa: E402
from tiles_backend.perspective_engine.room import geometry as room_geometry  # noqa: E402
from tiles_backend.perspective_engine.surface.wall import corner_cut, edge_layout  # noqa: E402
from tiles_backend.perspective_engine.surface.wall import junction_layout as jl  # noqa: E402

REQ = TileRequest(tile_width_mm=600, tile_height_mm=600, grout_mm=0,
                  room_width_mm=3658, room_length_mm=3658, room_height_mm=3048)


def families(geo, frame, w, h):
    fv = geo["floor"].get("vanishing_points") or {}
    vert = (frame.get("vertical_vp") or {}).get("homogeneous")
    fam = {"vertical": np.array(vert, float) if vert is not None else np.array([0.0, 1.0, 0.0])}
    if fv.get("vp1") is not None:
        fam["depth"] = np.array([*fv["vp1"], 1.0])
    if fv.get("vp2") is not None:
        fam["lateral (floor vp2)"] = np.array([*fv["vp2"], 1.0])
    fam["lateral (image-horizontal)"] = np.array([1.0, 0.0, 0.0])
    return fam


def classify(lines, fam, objects, diag, w, h):
    counts = {k: 0 for k in fam}
    rej = {"too short (< 2% of diagonal)": 0, "on a removed object": 0, "no family within 2 deg": 0}
    keep = []
    for seg in lines:
        x1, y1, x2, y2 = seg
        if math.hypot(x2 - x1, y2 - y1) < 0.02 * diag:
            rej["too short (< 2% of diagonal)"] += 1
            continue
        mx, my = int((x1 + x2) / 2), int((y1 + y2) / 2)
        if objects[min(max(my, 0), h - 1), min(max(mx, 0), w - 1)]:
            rej["on a removed object"] += 1
            continue
        res = {k: angle_to(seg, v) for k, v in fam.items()}
        k = min(res, key=res.get)
        if res[k] > 2.0:
            rej["no family within 2 deg"] += 1
            keep.append((seg, None))
            continue
        counts[k] += 1
        keep.append((seg, k))
    return counts, rej, keep


def vertical_edges(lines, wall, h, w, min_len=0.10):
    xs = []
    for x1, y1, x2, y2 in lines:
        if math.hypot(x2 - x1, y2 - y1) < min_len * h:
            continue
        if math.degrees(math.atan2(abs(x2 - x1), abs(y2 - y1))) > 3.0:
            continue
        mx, my = int((x1 + x2) / 2), int((y1 + y2) / 2)
        band = wall[max(0, my - 3):my + 4, max(0, mx - 12):mx + 13]
        if band.any():
            xs.append((x1 + x2) / 2)
    return xs


def seam_mismatch(img_a, tiled_a, img_b, tiled_b, seam_x, side_a):
    """Median |dy| between the horizontal seams of two walls at their shared corner."""
    def rows(img, tiled, x):
        x = int(np.clip(x, 0, img.shape[1] - 1))
        col = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[:, x].astype(float)
        ok = tiled[:, x]
        d = np.abs(np.diff(col))
        return [y for y in range(1, len(d) - 1) if ok[y] and ok[y + 1] and d[y] > 60 and d[y] >= d[y - 1] and d[y] >= d[y + 1]]
    off = 6
    ra = rows(img_a, tiled_a, seam_x - off if side_a == "left" else seam_x + off)
    rb = rows(img_b, tiled_b, seam_x + off if side_a == "left" else seam_x - off)
    if not ra or not rb:
        return None, len(ra), len(rb)
    diffs = [min(abs(y - z) for z in rb) for y in ra]
    diffs = [d for d in diffs if d <= 60]
    return (round(float(np.median(diffs)), 1) if diffs else None), len(ra), len(rb)


def main() -> int:
    name, job, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]) / sys.argv[1]
    deeplsd = json.loads(Path(sys.argv[4]).read_text())["lines"] if len(sys.argv) > 4 else None
    out.mkdir(parents=True, exist_ok=True)
    seg = out / "scratch" / "segments"
    shutil.rmtree(seg.parent, ignore_errors=True)
    shutil.copytree(job / "segments", seg)
    for stored in ("floor", "wall", "room"):
        shutil.rmtree(seg / stored, ignore_errors=True)
    original_merge = corner_cut.merge_planes
    corner_cut.merge_planes = lambda refined, *a, **k: (refined, {"enabled": False, "merged": []})   # raw pieces
    bundle = reuse_load(job)
    fm, wm = pe.load(seg)
    geo = pe.ensure_geometry(seg, bundle.room, fm, wm, clean=bundle.clean)
    corner_cut.merge_planes = original_merge
    floor, wall, _ = pe_masks.prepare(fm, wm, bundle.clean.shape[:2])
    h, w = floor.shape
    diag = math.hypot(h, w)
    frame = geo.get("room_frame") or {}
    objects = cv2.imread(str(seg / "ALL_OBJECTS.png"), cv2.IMREAD_UNCHANGED)
    objects = cv2.resize((objects[..., 3] > 127).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    clean_bgr = cv2.cvtColor(bundle.clean, cv2.COLOR_RGB2BGR)
    rep = {"room": name, "size": [w, h], "camera": {k: geo["camera"].get(k) for k in ("focal_px", "focal_source", "principal_point", "confidence")},
           "joint_decision": (geo["camera"].get("joint") or {}).get("decision")}

    # ---- 1. lines
    fam = families(geo, frame, w, h)
    lsd = np.asarray(detect_lines_lsd(cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2GRAY))).reshape(-1, 4)
    sets = {"opencv_lsd": lsd}
    if deeplsd is not None:
        sets["deeplsd"] = np.asarray(deeplsd, float).reshape(-1, 4)
    colours = {"vertical": (0, 200, 255), "depth": (0, 255, 0), "lateral (floor vp2)": (255, 128, 0),
               "lateral (image-horizontal)": (255, 0, 255), None: (140, 140, 140)}
    rep["lines"] = {}
    for key, ls in sets.items():
        counts, rej, keep = classify(ls, fam, objects, diag, w, h)
        rep["lines"][key] = {"total": int(len(ls)), "by_family": counts, "rejected": rej}
        img = (clean_bgr * 0.45).astype(np.uint8)
        for (x1, y1, x2, y2), k in keep:
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), colours[k], 2 if k else 1)
        cv2.imwrite(str(out / f"lines_{key}.png"), img)
    rep["lines"]["families"] = {k: ([round(x, 1) for x in (v[:2] / v[2])] if abs(v[2]) > 1e-9 else "infinity") for k, v in fam.items()}

    # ---- 2. corners
    confirmed, uncertain, runs, edges = corner_cut.corners(floor, wall, objects, clean_bgr)
    fr = corner_cut.runs(jl.junction_points(floor, wall, objects), w, diag)
    cr = corner_cut.runs(edge_layout.ceiling_points(wall, objects), w, diag)
    cands = []
    for b in corner_cut.bends(fr, w):
        cands.append((b["x"], "floor bend"))
    for b in corner_cut.bends(cr, w):
        cands.append((b["x"], "ceiling bend"))
    for g in corner_cut.gap_ends(fr, w):
        cands.append((g["x"], "floor run end at gap"))
    for g in corner_cut.gap_ends(cr, w):
        cands.append((g["x"], "ceiling run end at gap"))
    for x in vertical_edges(lsd, wall, h, w):
        cands.append((x, "LSD vertical edge"))
    if deeplsd is not None:
        for x in vertical_edges(sets["deeplsd"], wall, h, w):
            cands.append((x, "DeepLSD vertical edge"))
    cands = [c for c in cands if 0.02 * w <= c[0] <= 0.98 * w]
    cands.sort()
    clusters = []
    for x, why in cands:
        if clusters and x - clusters[-1]["xs"][-1] <= 0.012 * w:
            clusters[-1]["xs"].append(x)
            clusters[-1]["witnesses"].add(why)
        else:
            clusters.append({"xs": [x], "witnesses": {why}})
    corner_list = []
    for c in clusters:
        geom = {"floor bend", "ceiling bend", "floor run end at gap", "ceiling run end at gap"} & c["witnesses"]
        photo = {"LSD vertical edge", "DeepLSD vertical edge"} & c["witnesses"]
        conf = "HIGH" if geom and photo else ("MEDIUM" if (len(geom) >= 2 or len(photo) >= 2) else "LOW")
        corner_list.append({"x": round(float(np.median(c["xs"])), 1), "witnesses": sorted(c["witnesses"]), "confidence": conf})
    rep["corners"] = {"candidates": corner_list,
                      "high": sum(1 for c in corner_list if c["confidence"] == "HIGH"),
                      "medium": sum(1 for c in corner_list if c["confidence"] == "MEDIUM"),
                      "low": sum(1 for c in corner_list if c["confidence"] == "LOW"),
                      "corner_cut_confirmed": [e["x"] for _, _, e in confirmed]}
    img = (clean_bgr * 0.5).astype(np.uint8)
    for c in corner_list:
        col = {"HIGH": (0, 255, 0), "MEDIUM": (0, 200, 255), "LOW": (0, 0, 255)}[c["confidence"]]
        cv2.line(img, (int(c["x"]), 0), (int(c["x"]), h), col, 2)
        cv2.putText(img, f"{c['x']:.0f} {c['confidence']}", (int(c["x"]) + 4, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
    cv2.imwrite(str(out / "corners.png"), img)

    # ---- 3/4. pieces and planes
    room = room_geometry.solve(frame, {"width": None, "length": None, "height": None}, floor_mask=floor) if frame else {}
    planes = room.get("wall_planes_mm") or {}
    masks = geo["_wall_masks"]
    pieces = []
    pimg = (clean_bgr * 0.5).astype(np.uint8)
    pal = [(0, 200, 255), (0, 255, 0), (255, 128, 0), (255, 0, 255), (0, 128, 255), (255, 255, 0), (128, 0, 255), (0, 255, 255)]
    for k, wl in enumerate(sorted(geo["walls"], key=lambda x: float(np.nonzero(masks[int(x["index"])])[1].mean()))):
        m = masks[int(wl["index"])]
        cols = np.nonzero(m.any(axis=0))[0]
        d = wl.get("direction") or {}
        pl = planes.get(wl["id"])
        pieces.append({"id": wl["id"], "x_range": [int(cols.min()), int(cols.max())], "pixels": int(m.sum()),
                       "faces": d.get("faces"), "normal": [round(v, 3) for v in d["normal"]] if d.get("normal") else None,
                       "distance_mm": None if pl is None else round(float(pl[3]), 1), "placed": pl is not None,
                       "dot": wl["select_point"]})
        pimg[m] = (pimg[m] * 0.4 + np.array(pal[k % len(pal)]) * 0.6).astype(np.uint8)
        cv2.putText(pimg, f"{wl['id']} {d.get('faces')}", (int(cols.mean()) - 40, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.circle(pimg, tuple(int(v) for v in wl["select_point"]), 8, (255, 255, 255), -1)
    cv2.imwrite(str(out / "pieces.png"), pimg)
    boundaries = [p["x_range"][1] for p in pieces[:-1]]
    rep["pieces"] = pieces
    rep["corners_without_a_piece_boundary"] = [c["x"] for c in corner_list if c["confidence"] != "LOW"
                                               and not any(abs(c["x"] - b) <= 0.012 * w for b in boundaries)]
    rep["piece_boundaries_without_a_corner"] = [b for b in boundaries
                                                if not any(abs(c["x"] - b) <= 0.012 * w and c["confidence"] != "LOW" for c in corner_list)]

    renders = {}
    tile = wd._checker()
    for p in pieces:
        idx = int(p["id"].split("-")[1])
        try:
            rw = render_room(bundle.clean, floor, wall, tile, "wall", REQ, wall_index=idx, geometry=geo)
            renders[p["id"]] = (rw.image, rw.wall_tiled)
        except Exception as e:  # noqa: BLE001
            p["render"] = f"refused: {str(e)[:60]}"
    seams = []
    for a, b in zip(pieces, pieces[1:]):
        if a["id"] in renders and b["id"] in renders:
            x = (a["x_range"][1] + b["x_range"][0]) / 2
            mm, na, nb = seam_mismatch(*renders[a["id"]], *renders[b["id"]], x, "left")
            seams.append({"between": [a["id"], b["id"]], "x": round(x, 1), "seam_height_mismatch_px": mm,
                          "seams_found": [na, nb]})
    rep["corner_seams"] = seams
    if renders:
        comp = bundle.clean.copy()
        for img_, t_ in renders.values():
            comp[t_] = img_[t_]
        cv2.imwrite(str(out / "walls_checker.png"), cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))

    # ---- 5. old pixels
    gap = np.zeros_like(floor)
    for x in range(w):
        fy = np.nonzero(floor[:, x])[0]
        wy = np.nonzero(wall[:, x])[0]
        if len(fy) and len(wy):
            top = fy.min()
            below = wy[wy < top]
            if len(below):
                gap[below.max() + 1:top, x] = True
    gap &= ~objects
    rep["old_pixels"] = {"skirting_gap_px (between wall and floor masks, never tiled)": int(gap.sum()),
                         "columns_with_a_gap": int(gap.any(axis=0).sum()),
                         "max_gap_px": int(gap.sum(axis=0).max()) if gap.any() else 0}
    oimg = clean_bgr.copy()
    oimg[gap] = (0, 0, 255)
    cv2.imwrite(str(out / "old_pixels_gap.png"), oimg)

    # ---- 6. floor grids
    try:
        grey = np.full((256, 256, 3), 150, np.uint8)
        rf = render_room(bundle.clean, floor, wall, grey, "floor",
                         TileRequest(tile_width_mm=600, tile_height_mm=600, grout_mm=0,
                                     room_width_mm=3658, room_length_mm=3658, room_height_mm=3048), geometry=geo)
        cv2.imwrite(str(out / "floor_flat_grey.png"), cv2.cvtColor(rf.image, cv2.COLOR_RGB2BGR))
        inner = cv2.erode(rf.floor_tiled.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        g = cv2.cvtColor(rf.image, cv2.COLOR_RGB2GRAY)
        long_lines = [l for l in np.asarray(detect_lines_lsd(g)).reshape(-1, 4)
                      if math.hypot(l[2] - l[0], l[3] - l[1]) >= 0.05 * diag
                      and inner[int((l[1] + l[3]) / 2), int((l[0] + l[2]) / 2)]]
        gc = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2GRAY)
        old_lines = [l for l in np.asarray(detect_lines_lsd(gc)).reshape(-1, 4)
                     if math.hypot(l[2] - l[0], l[3] - l[1]) >= 0.05 * diag
                     and inner[int((l[1] + l[3]) / 2), int((l[0] + l[2]) / 2)]]
        rep["floor_grids"] = {"long lines in a FLAT grey floor render (no grout)": len(long_lines),
                              "long lines on the old floor in the clean room": len(old_lines),
                              "meaning": "lines in a flat, grout-free render can only come from the old floor through the shading"}
    except Exception as e:  # noqa: BLE001
        rep["floor_grids"] = f"render refused: {str(e)[:80]}"
    (out / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps(rep, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
