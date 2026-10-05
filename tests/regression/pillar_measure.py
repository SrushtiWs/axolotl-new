"""
Floor seams vs the back wall's bottom edge, and per-wall tile size (read only).

    backend/.venv/bin/python tests/regression/pillar_measure.py <job_dir> [out_dir]

floor seams   the floor is rendered with a checker; its seams across the room
              (LSD lines within 30 deg of image-horizontal, in the tiled floor,
              not pointing at the floor's depth VP -- those run into the room)
              are compared with the back wall's bottom edge (the floor-wall
              junction of the largest camera-facing wall, fitted as a line):
              median / 90% angle difference in degrees
tile size     per wall, the rendered tile width in mm on that wall's own plane
              (from the engine's own grid report)
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "backend")]

import perspective_engine as pe  # noqa: E402
import reuse  # noqa: E402
from perspective_engine import masks as pe_masks  # noqa: E402
from tiles_backend.perspective_engine import TileRequest, render_room  # noqa: E402
from tiles_backend.perspective_engine.camera.vp_from_mask import detect_lines_lsd  # noqa: E402
from tiles_backend.perspective_engine.surface.wall import junction_layout as jl  # noqa: E402


def checker(px=256, n=4):
    idx = np.arange(px) * n // px
    return np.dstack([((idx[:, None] + idx[None, :]) % 2 * 255).astype(np.uint8)] * 3)


def main() -> int:
    job = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else job
    seg = job / "segments"
    b = reuse.load(job)
    fl, wl = pe.load(seg)
    geo = pe.ensure_geometry(seg, b.room, fl, wl, clean=b.clean)
    floor, wall, _ = pe_masks.prepare(fl, wl, b.clean.shape[:2])
    masks = geo["_wall_masks"]
    fronts = [w for w in geo["walls"] if (w.get("direction") or {}).get("faces") == "front"]
    back = max(fronts, key=lambda w: w["pixels"]) if fronts else max(geo["walls"], key=lambda w: w["pixels"])
    J = jl.junction_points(floor, masks[int(back["index"])], None)
    vx, vy, _, _ = cv2.fitLine(J.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    edge_deg = math.degrees(math.atan2(vy, vx))
    edge_deg = ((edge_deg + 90) % 180) - 90
    req = TileRequest(tile_width_mm=600, tile_height_mm=600, grout_mm=0)
    rf = render_room(b.clean, floor, wall, checker(), "floor", req, geometry=geo)
    gray = cv2.cvtColor(rf.image, cv2.COLOR_RGB2GRAY)
    inner = cv2.erode(rf.floor_tiled.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    depth = (geo["floor"].get("vanishing_points") or {}).get("depth_vp")
    diffs = []
    for x1, y1, x2, y2 in np.asarray(detect_lines_lsd(gray)).reshape(-1, 4):
        if math.hypot(x2 - x1, y2 - y1) < 20:
            continue
        a = math.degrees(math.atan2(y2 - y1, x2 - x1))
        a = ((a + 90) % 180) - 90
        if abs(a) > 30:
            continue
        mx, my = int((x1 + x2) / 2), int((y1 + y2) / 2)
        if depth is not None:
            to_vp = math.degrees(math.atan2(depth[1] - my, depth[0] - mx))
            off = abs(((a - to_vp) + 90) % 180 - 90)
            if off <= 8.0:
                continue                      # a seam running into the room, not across it
        if inner[min(my, inner.shape[0] - 1), min(mx, inner.shape[1] - 1)]:
            diffs.append(abs(a - edge_deg))
    cv2.imwrite(str(out / "floor_checker.png"), cv2.cvtColor(rf.image, cv2.COLOR_RGB2BGR))
    res = {"back_wall": back["id"], "back_bottom_edge_deg": round(edge_deg, 2),
           "floor_seams": len(diffs),
           "floor_seam_vs_back_edge_deg": {"median": round(float(np.median(diffs)), 2) if diffs else None,
                                           "p90": round(float(np.percentile(diffs, 90)), 2) if diffs else None},
           "floor_direction": (geo["floor"].get("floor_direction") or "floor vanishing points")}
    print(json.dumps(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
