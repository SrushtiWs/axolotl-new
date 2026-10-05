"""
Missing thin object parts (legs, rails) and sheer curtains, DIAGNOSIS ONLY.

    backend/.venv/bin/python tests/regression/thin_parts_diagnose.py <out_dir> [room ...]

Per room of tests/regression/baseline_rooms.ROOMS (job copied, nothing written back):

Objects standing on the floor (the accepted mask's bottom within 1 % of the
diagonal of FLOOR_MASK), from segments/objects.json boxes and the union object
mask (props):
  mask_bottom        lowest object-mask row inside the box's columns
  thin candidates    in a search window around the box (x +/-10 % of its width,
                     y from 30 % of its height down to 60 % of its height below
                     it): pixels NOT in the object mask whose luminance differs
                     from the local floor estimate -- a grey closing (dark parts)
                     / opening (bright parts) with a horizontal line as long as
                     1.2 % of the image width, which removes thin structures --
                     by >= max(25, 15 % of the estimate); a component is kept when
                     it is thin (widest point <= 0.5 % of the diagonal, >= 2 px),
                     touches the object mask (3 px) or a kept component, and has
                     sharp edges (median Sobel at its outline >= 2x the floor's);
                     "leg" when its main axis is within 30 deg of vertical, else
                     "rail" (cross bars, stretchers); a component that runs out through
                     the window's left, right or bottom side is floor pattern (grout
                     lines, reflections) and is rejected
  true_bottom        lowest candidate row (the photo's bottom of the object)
  in_band_8pct       how many candidate pixels lie within 8 % of the object's
                     height below the mask's bottom (the Phase 2 search band)
  tiled              share of candidate pixels inside FLOOR_MASK and changed by
                     the baseline render (tiles drawn over them)
Curtains: SegFormer (backend/surfaces.label_map) on the photo; per "curtain"
component: its share in the object mask / window objects / WALL_MASK, its mean
object alpha (ALL_OBJECTS.png), and the tiled share in the baseline render.
Thin-part erosion: object pixels at most 2 px from the mask's own edge on both
sides (a part <= 4 px wide) whose alpha render._tight_alpha takes below 0.5.
Writes <out_dir>/<room>__thin_<object>.png crops (3x: photo | overlay | render;
magenta = object mask outline, green = missing legs, cyan = missing rails) and
<out_dir>/thin_parts_diagnose.json.
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
from perspective_engine import masks  # noqa: E402
from perspective_engine import render as pe_render  # noqa: E402

BASELINE = ROOT / "backups/baselines-20261005_141642"
LINE_SHARE, MIN_DIFF, DIFF_SHARE, THIN_SHARE, EDGE_RATIO, VERT_DEG = 0.012, 25.0, 0.15, 0.005, 2.0, 30.0


def main() -> int:
    out = Path(sys.argv[1])
    names = sys.argv[2:] or list(ROOMS)
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    for name in names:
        with tempfile.TemporaryDirectory() as tmp:
            job = Path(tmp) / ROOMS[name].name
            shutil.copytree(ROOMS[name], job)
            b = reuse.load(job)
            fl, wl = pe.load(job / "segments")
            objects = json.loads((job / "segments/objects.json").read_text()).get("objects", [])
            rgba = cv2.imread(str(job / "segments/ALL_OBJECTS.png"), cv2.IMREAD_UNCHANGED)
        h, w = b.room.shape[:2]
        diag = math.hypot(h, w)
        floor, wall, _ = masks.prepare(fl, wl, (h, w))
        props = np.asarray(b.props, bool)
        alpha = cv2.resize(rgba[..., 3], (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0 \
            if rgba is not None and rgba.ndim == 3 and rgba.shape[2] == 4 else props.astype(np.float32)
        base = np.asarray(Image.open(BASELINE / f"{name}.png").convert("RGB"))
        tiled = np.abs(base.astype(int) - b.room.astype(int)).max(axis=2) > 0
        L = cv2.cvtColor(b.room, cv2.COLOR_RGB2GRAY).astype(np.float32)
        k = max(9, int(LINE_SHARE * w)) | 1
        line = np.ones((1, k), np.uint8)
        est_dark, est_bright = cv2.morphologyEx(L, cv2.MORPH_CLOSE, line), cv2.morphologyEx(L, cv2.MORPH_OPEN, line)
        tophat = np.maximum(est_dark - L, L - est_bright)
        estimate = np.where(est_dark - L >= L - est_bright, est_dark, est_bright)
        sob = np.hypot(cv2.Sobel(L, cv2.CV_32F, 1, 0), cv2.Sobel(L, cv2.CV_32F, 0, 1))
        floor_sob = float(np.median(sob[floor & ~props])) if (floor & ~props).any() else 1.0
        near_obj = cv2.dilate(props.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
        thin_max = max(2.0, THIN_SHARE * diag)
        objs_out = {}
        for o in objects:
            sw, sh = o.get("source_width") or w, o.get("source_height") or h
            x0, y0, x1, y1 = (np.array(o["bbox"], float) * [w / sw, h / sh, w / sw, h / sh]).astype(int)
            bw, bh = max(1, x1 - x0), max(1, y1 - y0)
            cols = slice(max(0, x0), min(w, x1))
            in_box = np.zeros((h, w), bool)
            in_box[max(0, y0):min(h, y1 + 1), cols] = True
            obj = props & in_box
            if not obj.any():
                continue
            mask_bottom = int(np.nonzero(obj.any(axis=1))[0].max())
            band = max(3, int(0.01 * diag))
            below = floor[mask_bottom + 1:min(h, mask_bottom + 1 + band), cols]
            if not below.any():
                continue                                  # not standing on the floor
            wx0, wx1 = max(0, int(x0 - 0.1 * bw)), min(w, int(x1 + 0.1 * bw))
            wy0, wy1 = max(0, int(y0 + 0.3 * bh)), min(h, int(y1 + 0.6 * bh))
            win = np.zeros((h, w), bool)
            win[wy0:wy1, wx0:wx1] = True
            cand = win & ~props & ~wall & (tophat >= np.maximum(MIN_DIFF, DIFF_SHARE * estimate))
            n, lab, stats, _ = cv2.connectedComponentsWithStats(cand.astype(np.uint8), connectivity=8)
            dist = cv2.distanceTransform(cand.astype(np.uint8), cv2.DIST_L2, 3)
            touch = cv2.dilate(obj.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
            kept = np.zeros(n, bool)
            kinds = {}
            rejected = {"thick": 0, "soft_edge": 0, "not_connected": 0, "floor_pattern": 0}
            info = []
            for i in range(1, n):
                comp = lab == i
                width = 2 * float(dist[comp].max())
                if width > 2 * thin_max or stats[i, cv2.CC_STAT_AREA] < 4:
                    rejected["thick"] += int(comp.sum())
                    continue
                outline = comp & ~cv2.erode(comp.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
                if float(np.median(sob[outline])) < EDGE_RATIO * max(floor_sob, 1.0):
                    rejected["soft_edge"] += int(comp.sum())
                    continue
                ys, xs = np.nonzero(comp)
                if xs.min() <= wx0 or xs.max() >= wx1 - 1 or ys.max() >= wy1 - 1:
                    rejected["floor_pattern"] += int(comp.sum())
                    continue
                if len(ys) > 2 and (np.ptp(ys) > 0 or np.ptp(xs) > 0):
                    cov = np.cov(np.stack([xs, ys]))
                    ev, evec = np.linalg.eigh(cov)
                    ax = evec[:, -1]
                    ang = math.degrees(math.atan2(abs(ax[0]), abs(ax[1])))   # 0 = vertical
                else:
                    ang = 90.0
                info.append((i, comp, ang))
            # connected to the object, directly or through kept components
            changed = True
            while changed:
                changed = False
                for i, comp, ang in info:
                    if kept[i]:
                        continue
                    if (comp & touch).any():
                        kept[i] = True
                        changed = True
                        touch |= cv2.dilate(comp.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
            legs = np.zeros((h, w), bool)
            rails = np.zeros((h, w), bool)
            for i, comp, ang in info:
                if kept[i]:
                    (legs if ang <= VERT_DEG else rails)[comp] = True
                else:
                    rejected["not_connected"] += int(comp.sum())
            missing = legs | rails
            if not missing.any():
                objs_out[f"{o['id']} {o['label']}"] = {"bbox": [int(x0), int(y0), int(x1), int(y1)], "mask_bottom": mask_bottom,
                                                       "missing_px": 0, "rejected_px": rejected}
                continue
            true_bottom = int(np.nonzero(missing.any(axis=1))[0].max())
            band8 = np.zeros((h, w), bool)
            band8[mask_bottom + 1:min(h, mask_bottom + 1 + max(1, int(0.08 * bh))), :] = True
            objs_out[f"{o['id']} {o['label']}"] = {
                "bbox": [int(x0), int(y0), int(x1), int(y1)], "height_px": int(bh),
                "mask_bottom": mask_bottom, "true_bottom": true_bottom,
                "bottom_gap_px": true_bottom - mask_bottom,
                "bottom_gap_share_of_height": round((true_bottom - mask_bottom) / bh, 3),
                "missing_px": int(missing.sum()), "missing_leg_px": int(legs.sum()), "missing_rail_px": int(rails.sum()),
                "missing_in_8pct_band_px": int((missing & band8).sum()),
                "missing_in_floor_mask_share": round(float(floor[missing].mean()), 3),
                "missing_tiled_share": round(float(tiled[missing].mean()), 3),
                "rejected_px": rejected,
            }
            # crop
            cx0, cy0, cx1, cy1 = max(0, wx0 - 10), max(0, y0 - 10), min(w, wx1 + 10), min(h, wy1 + 5)
            ph = b.room[cy0:cy1, cx0:cx1].copy()
            ov = ph.copy()
            e = obj & ~cv2.erode(obj.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            ov[e[cy0:cy1, cx0:cx1]] = (255, 0, 255)
            ov[legs[cy0:cy1, cx0:cx1]] = (0, 255, 0)
            ov[rails[cy0:cy1, cx0:cx1]] = (0, 230, 255)
            strip = np.hstack([ph, ov, base[cy0:cy1, cx0:cx1]])
            Image.fromarray(cv2.resize(strip, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)).save(
                out / f"{name}__thin_{o['id']}_{o['label'].replace(' ', '_')}.png")
        # curtains
        lab_map, names_ = surfaces.label_map(b.room)
        curtain_ids = [k_ for k_, v in names_.items() if v in ("curtain", "blind")]
        curtain = np.isin(lab_map, curtain_ids)
        window_obj = np.zeros((h, w), bool)
        for o in objects:
            if o["label"] in ("window", "curtain", "blind"):
                sw, sh = o.get("source_width") or w, o.get("source_height") or h
                x0, y0, x1, y1 = (np.array(o["bbox"], float) * [w / sw, h / sh, w / sw, h / sh]).astype(int)
                window_obj[max(0, y0):y1, max(0, x0):x1] |= props[max(0, y0):y1, max(0, x0):x1]
        curt_out = []
        n, lab, stats, _ = cv2.connectedComponentsWithStats(curtain.astype(np.uint8), connectivity=8)
        for i in range(1, n):
            comp = lab == i
            if comp.sum() < 0.0005 * h * w:
                continue
            x, y, cw, ch = stats[i, :4]
            curt_out.append({"bbox": [int(x), int(y), int(x + cw), int(y + ch)], "pixels": int(comp.sum()),
                             "in_object_mask": round(float(props[comp].mean()), 3),
                             "in_window_object": round(float(window_obj[comp].mean()), 3),
                             "in_WALL_MASK": round(float(wall[comp].mean()), 3),
                             "in_FLOOR_MASK": round(float(floor[comp].mean()), 3),
                             "mean_object_alpha": round(float(alpha[comp].mean()), 3),
                             "tiled_share": round(float(tiled[comp].mean()), 3),
                             "tiled_px": int((tiled & comp).sum()),
                             "photo_luminance_mean": round(float(L[comp].mean()), 1)})
        if curtain.any():
            ov = b.room.copy()
            ov[curtain & tiled] = (0.4 * ov[curtain & tiled] + 0.6 * np.array([255, 120, 0])).astype(np.uint8)
            ov[curtain & ~tiled] = (0.6 * ov[curtain & ~tiled] + 0.4 * np.array([0, 200, 255])).astype(np.uint8)
            Image.fromarray(np.hstack([ov, base])).save(out / f"{name}__curtains.png")
        # thin parts that _tight_alpha would take away
        a = props.astype(np.float32) if rgba is None else np.clip(alpha, 0, 1)
        tight = pe_render._tight_alpha(a)
        inside = cv2.distanceTransform((a > 0.5).astype(np.uint8), cv2.DIST_L2, 3)
        thin_obj = (a > 0.5) & (cv2.dilate(inside, np.ones((5, 5), np.uint8)) <= 2.0)
        report[name] = {"size": [w, h], "objects_on_floor": objs_out, "curtains": curt_out,
                        "curtain_labels": [names_[k_] for k_ in curtain_ids],
                        "tight_alpha_thin_parts": {"thin_object_px_(<=4px_wide)": int(thin_obj.sum()),
                                                   "of_which_alpha_below_0.5_after": int((thin_obj & (tight < 0.5)).sum())}}
        print(name, "done", flush=True)
    (out / "thin_parts_diagnose.json").write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
