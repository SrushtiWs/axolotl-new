"""
Kitchen (L-shaped platform), DIAGNOSIS ONLY: where the faults start, stage by stage.

    backend/.venv/bin/python tests/regression/kitchen_diagnose.py <job_dir> <out_dir>

The platform outline is traced BY HAND on this photo (render pixels, ESTIMATED
+/- 5 px) and used only to measure; nothing here feeds the engine.
  A  SegFormer label shares on each platform part (backend/surfaces.label_map)
  B  ALL_OBJECTS (props) coverage of each part
  C  FLOOR_MASK / WALL_MASK shares of each part
  D  floor-wall junction points (top floor pixel per column with WALL_MASK just
     above) and ceiling-wall points (top wall pixel per column with neither
     mask above): how many lie on the platform instead of the wall/floor seam,
     and which columns have no junction at all
  E  each stored wall cut: what the photo has there (nearest platform part,
     vertical DeepLSD edge above the counter on both sides) -- real corner or not
Writes <out_dir>/kitchen__stages.png and kitchen__diagnose.json.
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
from perspective_engine import masks  # noqa: E402
from wall_direction_diagnose import deeplsd  # noqa: E402

PARTS = {   # hand-traced, render pixels of job cf6f7831d6de (1456 x 1088)
    "left counter top": [(0, 562), (300, 520), (590, 458), (680, 462), (680, 482), (320, 612), (300, 616), (0, 595)],
    "left counter front edge": [(320, 612), (680, 482), (682, 502), (330, 632)],
    "end slab (front face)": [(0, 595), (300, 616), (320, 628), (318, 1032), (300, 1032), (35, 978), (0, 975)],
    "sink bowl": [(350, 580), (520, 548), (522, 605), (480, 622), (352, 628)],
    "cabinet panel": [(425, 612), (570, 570), (586, 575), (586, 735), (575, 738), (428, 708)],
    "back counter top + edge": [(680, 462), (1336, 476), (1338, 505), (1325, 515), (680, 503)],
    "support 1": [(655, 495), (682, 495), (682, 662), (655, 660)],
    "support 2": [(765, 500), (797, 500), (797, 668), (765, 666)],
    "support 3": [(1055, 505), (1088, 505), (1088, 698), (1055, 695)],
}
REAL_CORNERS = {"left|back": 590, "back|right": 1320}   # hand-read: ceiling-line bends + vertical tile edges


def poly(shape, pts):
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.array(pts, np.int32)], 1)
    return m.astype(bool)


def main() -> int:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        job = Path(tmp) / src.name
        shutil.copytree(src, job)
        b = reuse.load(job)
        fl, wl = pe.load(job / "segments")
        geo = json.loads((job / "segments/wall/wall_geometry.json").read_text())
    h, w = b.room.shape[:2]
    floor, wall, _ = masks.prepare(fl, wl, (h, w))
    props = np.asarray(b.props, bool)
    lab, names = surfaces.label_map(b.room)
    rep = {"size": [w, h], "parts": {}}
    gt = np.zeros((h, w), bool)
    for name, pts in PARTS.items():
        m = poly((h, w), pts)
        gt |= m
        v, c = np.unique(lab[m], return_counts=True)
        top = sorted(zip(c, v), reverse=True)[:4]
        rep["parts"][name] = {
            "px": int(m.sum()),
            "segformer": {names[int(k)]: round(n / m.sum(), 3) for n, k in top},
            "in_ALL_OBJECTS": round(float(props[m].mean()), 3),
            "in_FLOOR_MASK": round(float(floor[m].mean()), 3),
            "in_WALL_MASK": round(float(wall[m].mean()), 3),
            "missing_from_objects_px": int((m & ~props).sum()),
        }
    v, c = np.unique(lab[gt], return_counts=True)
    rep["platform_total"] = {"px": int(gt.sum()), "in_ALL_OBJECTS": round(float(props[gt].mean()), 3),
                             "missing_from_objects_px": int((gt & ~props).sum()),
                             "in_FLOOR_MASK": round(float(floor[gt].mean()), 3),
                             "in_WALL_MASK": round(float(wall[gt].mean()), 3),
                             "segformer": {names[int(k)]: round(n / gt.sum(), 3) for n, k in sorted(zip(c, v), reverse=True)[:6]}}
    # objects: what the accepted ones cover
    objs = json.loads((src / "segments/objects.json").read_text())
    rep["accepted_objects"] = [{"id": o["id"], "label": o["label"], "score": o["confidence"], "bbox": o["bbox"],
                                "sam_mask_px": o["metrics"].get("mask_area"), "final_px": o["mask_area"],
                                "components": o["metrics"].get("connected_component_count"),
                                "surface_overlap": o["metrics"].get("surface_overlap_ratio"),
                                "also_detected_as": o["metrics"].get("also_detected_as")} for o in objs["objects"]]
    rep["props_on_platform_px"] = int((props & gt).sum())
    rep["props_off_platform_px"] = int((props & ~gt).sum())
    # D junction points
    fj, cj = [], []
    for x in range(w):
        fy = np.nonzero(floor[:, x])[0]
        if fy.size:
            y = fy.min()
            if y > 2 and wall[y - 3:y, x].all():
                fj.append((x, y))
        wy = np.nonzero(wall[:, x])[0]
        if wy.size:
            y = wy.min()
            if y > 3 and not (wall[y - 3:y, x] | floor[y - 3:y, x] | props[y - 3:y, x]).any():
                cj.append((x, y))
    fj, cj = np.array(fj), np.array(cj)
    on_plat = gt[np.clip(fj[:, 1] - 2, 0, h - 1), fj[:, 0]] | gt[np.clip(fj[:, 1] + 2, 0, h - 1), fj[:, 0]] if len(fj) else np.zeros(0, bool)
    has_fj = np.zeros(w, bool)
    if len(fj):
        has_fj[fj[:, 0]] = True
    gaps = []
    run = None
    for x in range(w):
        if not has_fj[x] and run is None:
            run = x
        if has_fj[x] and run is not None:
            if x - run >= 10:
                gaps.append([run, x - 1])
            run = None
    if run is not None and w - run >= 10:
        gaps.append([run, w - 1])
    rep["junctions"] = {"floor_wall_points": int(len(fj)), "floor_wall_points_on_platform": int(on_plat.sum()),
                        "floor_wall_column_gaps_ge10px": gaps,
                        "ceiling_wall_points": int(len(cj)),
                        "ceiling_points_x_range": [int(cj[:, 0].min()), int(cj[:, 0].max())] if len(cj) else None}
    # E cuts
    segs = deeplsd(b.room)
    vert = []
    for s in segs:
        d = s[2:] - s[:2]
        L = float(np.hypot(*d))
        if L >= 0.03 * h and abs(d[0]) / (L + 1e-9) < math.sin(math.radians(12)):
            vert.append(s)
    vert = np.array(vert).reshape(-1, 4)
    cuts = geo["wall_split"].get("corners") or []
    rows = []
    for x in cuts:
        near = [list(map(int, s)) for s in vert if abs((s[0] + s[2]) / 2 - x) <= 12 and min(s[1], s[3]) < 455]
        long_above = max([abs(s[3] - s[1]) for s in near], default=0)
        part = min(PARTS, key=lambda n: min(abs(px - x) for px, _ in PARTS[n]))
        part_dx = min(abs(px - x) for px, _ in PARTS[part])
        real = min(REAL_CORNERS.items(), key=lambda kv: abs(kv[1] - x))
        rows.append({"cut_x": x, "nearest_platform_part": part, "part_dx_px": int(part_dx),
                     "vertical_edge_above_counter_px": round(float(long_above), 1),
                     "nearest_real_corner": real[0], "real_corner_dx_px": int(abs(real[1] - x)),
                     "verdict": "real corner" if abs(real[1] - x) <= 12 and long_above >= 0.1 * h else "not a corner"})
    rep["cuts"] = rows
    rep["real_corners_vertical_edges"] = {k: round(float(max([abs(s[3] - s[1]) for s in vert if abs((s[0] + s[2]) / 2 - x) <= 12 and min(s[1], s[3]) < 455], default=0)), 1)
                                          for k, x in REAL_CORNERS.items()}
    rep["wall_pieces"] = [{"id": wd["id"], "bbox": wd["bbox"], "faces": wd["direction"].get("faces"),
                           "normal": [round(v, 3) for v in wd["direction"].get("normal") or []],
                           "source": wd["direction"].get("source"), "validated": wd["direction"].get("validated"),
                           "validation": wd["direction"].get("validation")} for wd in geo["walls"]]
    rep["merge"] = geo["wall_split"].get("plane_merge")
    # overlay
    img = (b.room.astype(np.float32) * 0.55).astype(np.uint8)
    img[gt & props] = (0.5 * img[gt & props] + 0.5 * np.array([255, 0, 255])).astype(np.uint8)
    img[gt & ~props & wall] = (0.4 * img[gt & ~props & wall] + 0.6 * np.array([0, 220, 80])).astype(np.uint8)
    img[gt & ~props & floor] = (0.4 * img[gt & ~props & floor] + 0.6 * np.array([0, 140, 255])).astype(np.uint8)
    for name, pts in PARTS.items():
        cv2.polylines(img, [np.array(pts, np.int32)], True, (255, 255, 255), 1)
    for (x, y), bad in zip(fj, on_plat):
        cv2.circle(img, (int(x), int(y)), 2, (255, 60, 60) if bad else (255, 255, 0), -1)
    for x, y in cj:
        cv2.circle(img, (int(x), int(y)), 2, (0, 255, 255), -1)
    for s in vert:
        cv2.line(img, (int(s[0]), int(s[1])), (int(s[2]), int(s[3])), (255, 160, 0), 1)
    for r in rows:
        col = (0, 255, 0) if r["verdict"] == "real corner" else (255, 0, 0)
        cv2.line(img, (int(r["cut_x"]), 0), (int(r["cut_x"]), h), col, 2)
        cv2.putText(img, f"cut {r['cut_x']:.0f}", (int(r["cut_x"]) + 4, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    for k, x in REAL_CORNERS.items():
        cv2.line(img, (x, 0), (x, 450), (255, 255, 255), 1)
        cv2.putText(img, f"real {k}", (x + 4, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    Image.fromarray(np.hstack([b.room, img])).save(out / "kitchen__stages.png")
    (out / "kitchen__diagnose.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k not in ("wall_pieces",)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
