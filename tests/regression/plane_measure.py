"""
Wall plane separation, measured (read only: nothing in the pipeline changes).

    backend/.venv/bin/python tests/regression/plane_measure.py [name=segments_dir ...]

Default rooms: the frozen test rooms plus every room-data room that has a job.

Real corner lines are found WITHOUT the app's own wall split: the floor-wall
junction and the wall-ceiling line of the whole wall mask are each cut into
straight runs (split-and-merge, tolerance relative to the image diagonal);
a corner is where a line bends. A corner seen in both lines at the same x is
"confirmed" (confidence HIGH); seen in one line only it is "single" (MEDIUM).
Each corner line runs from that point toward the vertical vanishing point.

Per wall piece (wall/walls/wall-<i>.png):
  a. its stored normal, yaw, faces and vanishing point
  b. spill: its pixels outside the corner-bounded region holding most of it
     (count, and % of the piece)
  c. two directions: a second region holds >= 5% of the piece
  d. camera pitch: vertical lines vs floor plane
  e. its VP vs the nearest floor VP (distance / image diagonal; for VPs at
     infinity the angle between directions)

Overlays: tests/.work/plane_measure/<room>__<wall>.png and <room>__all.png
(piece green, spill red, corner lines yellow, junction/ceiling runs cyan).
Numbers: tests/.work/plane_measure/report.json
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

from tiles_backend.perspective_engine.surface.wall import edge_layout  # noqa: E402
from tiles_backend.perspective_engine.surface.wall import junction_layout as jl  # noqa: E402

WORK = ROOT / "tests" / ".work" / "plane_measure"
# Straight-run fitting, relative to the image.
FIT_TOL = 0.006          # x diagonal: max deviation of a run from its line
GAP = 0.02               # x width: a hole in the line this wide ends a run
MIN_RUN = 0.04           # x width: shorter runs are noise, not a wall
BEND_DEG = 6.0           # two runs meeting at more than this = a corner
MATCH = 0.03             # x width: floor and ceiling bends this close = one corner
TWO_DIR_SHARE = 0.05     # a second region holding this share of a piece = two directions


def _read(path: Path) -> np.ndarray | None:
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        return None
    if m.ndim == 3:
        m = m[..., 3] if m.shape[2] == 4 else cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    return m > 127


def _split(pts: np.ndarray, tol: float) -> list[np.ndarray]:
    """Douglas-Peucker style recursive split of an x-sorted polyline into straight runs."""
    if len(pts) < 3:
        return [pts]
    a, b = pts[0], pts[-1]
    d = b - a
    n = np.array([-d[1], d[0]]) / (np.hypot(*d) or 1.0)
    dist = np.abs((pts - a) @ n)
    i = int(np.argmax(dist))
    if dist[i] <= tol:
        return [pts]
    return _split(pts[: i + 1], tol) + _split(pts[i:], tol)


def _runs(points: np.ndarray, w: int, diag: float) -> list[dict]:
    """Straight runs of a junction / ceiling point set (x-sorted, broken at gaps)."""
    if len(points) < 2:
        return []
    pts = points[np.argsort(points[:, 0])]
    breaks = np.where(np.diff(pts[:, 0]) > GAP * w)[0]
    runs = []
    for part in np.split(pts, breaks + 1):
        for seg in _split(part, FIT_TOL * diag):
            if len(seg) < 2 or seg[-1, 0] - seg[0, 0] < MIN_RUN * w:
                continue
            vx, vy, x0, y0 = cv2.fitLine(seg.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
            runs.append({"x0": float(seg[0, 0]), "x1": float(seg[-1, 0]),
                         "angle": math.degrees(math.atan2(vy, vx)), "line": (vx, vy, x0, y0),
                         "pts": seg})
    # merge neighbours that continue in the same direction
    merged = []
    for r in runs:
        if merged and abs(merged[-1]["angle"] - r["angle"]) < BEND_DEG / 2 and r["x0"] - merged[-1]["x1"] <= GAP * w:
            m = merged[-1]
            pts2 = np.vstack([m["pts"], r["pts"]])
            vx, vy, x0, y0 = cv2.fitLine(pts2.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
            merged[-1] = {"x0": m["x0"], "x1": r["x1"], "angle": math.degrees(math.atan2(vy, vx)),
                          "line": (vx, vy, x0, y0), "pts": pts2}
        else:
            merged.append(r)
    return merged


def _bends(runs: list[dict], w: int) -> list[dict]:
    """Where two consecutive, touching runs meet at more than BEND_DEG."""
    out = []
    for a, b in zip(runs, runs[1:]):
        if b["x0"] - a["x1"] > GAP * w:
            continue
        turn = abs(((a["angle"] - b["angle"]) + 90) % 180 - 90)
        if turn < BEND_DEG:
            continue
        x = (a["x1"] + b["x0"]) / 2.0
        # the meeting point on run a's line
        vx, vy, x0, y0 = a["line"]
        y = y0 + (x - x0) * vy / (vx or 1e-9)
        out.append({"x": float(x), "y": float(y), "turn_deg": round(float(turn), 1)})
    return out


def _corners(floor_b, ceil_b, w) -> list[dict]:
    corners, used = [], set()
    for f in floor_b:
        match = [c for c in ceil_b if abs(c["x"] - f["x"]) <= MATCH * w]
        if match:
            c = min(match, key=lambda c: abs(c["x"] - f["x"]))
            used.add(id(c))
            corners.append({"x": f["x"], "floor": (f["x"], f["y"]), "ceiling": (c["x"], c["y"]),
                            "status": "confirmed", "confidence": "HIGH", "turn_deg": [f["turn_deg"], c["turn_deg"]]})
        else:
            corners.append({"x": f["x"], "floor": (f["x"], f["y"]), "ceiling": None,
                            "status": "single (floor line)", "confidence": "MEDIUM", "turn_deg": [f["turn_deg"]]})
    for c in ceil_b:
        if id(c) not in used:
            corners.append({"x": c["x"], "floor": None, "ceiling": (c["x"], c["y"]),
                            "status": "single (ceiling line)", "confidence": "MEDIUM", "turn_deg": [c["turn_deg"]]})
    return sorted(corners, key=lambda c: c["x"])


def _corner_line(c, vvp, h):
    """Two image points of a corner's vertical line: through its point(s), toward the vertical VP."""
    if c["floor"] and c["ceiling"]:
        return np.array(c["ceiling"], float), np.array(c["floor"], float)
    p = np.array(c["floor"] or c["ceiling"], float)
    if vvp is not None:
        d = np.array(vvp, float) - p
        d = d / (np.hypot(*d) or 1.0)
        if d[1] < 0:
            d = -d
    else:
        d = np.array([0.0, 1.0])
    return p - d * h, p + d * h


def _regions(shape, lines) -> np.ndarray:
    """Label every pixel by how many corner lines lie to its left (0 .. len(lines))."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    lab = np.zeros(shape, np.int32)
    for top, bottom in lines:
        d = bottom - top
        # x of the line at each row
        t = (yy - top[1]) / (d[1] or 1e-9)
        x_line = top[0] + t * d[0]
        lab += (xx > x_line).astype(np.int32)
    return lab


def _vp_point(vp: dict | None):
    if not vp:
        return None, None
    if vp.get("at_infinity") or vp.get("image") is None:
        hmg = vp.get("homogeneous")
        return None, (None if hmg is None else math.degrees(math.atan2(hmg[1], hmg[0])))
    return tuple(vp["image"]), None


def measure(name: str, seg: Path) -> dict:
    floor = _read(seg / "floor" / "floor_mask.png")
    wall = _read(seg / "wall" / "wall_mask.png")
    h, w = wall.shape
    diag = math.hypot(h, w)
    objects = _read(seg / "ALL_OBJECTS.png")
    if objects is not None and objects.shape != wall.shape:
        objects = cv2.resize(objects.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    geo = json.loads((seg / "wall" / "wall_geometry.json").read_text())
    fgeo = json.loads((seg / "floor" / "floor_geometry.json").read_text())
    fvps = json.loads((seg / "floor" / "floor_vanishing_points.json").read_text())
    frame_p = seg / "room" / "room_frame.json"
    frame = json.loads(frame_p.read_text()) if frame_p.exists() else {}
    vert = frame.get("vertical_vp") or {}
    vvp = tuple(vert["image"]) if vert.get("image") else None

    floor_runs = _runs(jl.junction_points(floor, wall, objects), w, diag)
    ceil_runs = _runs(edge_layout.ceiling_points(wall, objects), w, diag)
    corners = _corners(_bends(floor_runs, w), _bends(ceil_runs, w), w)
    lines = [_corner_line(c, vvp, h) for c in corners]
    regions = _regions((h, w), lines)

    floor_vps = [p for p in (fvps.get("vp1"), fvps.get("vp2")) if p is not None]
    report = {
        "room": name, "size": [w, h], "walls_detected": len(geo["walls"]),
        "camera": {
            "pitch_vertical_lines_deg": None if not vert.get("reliable") else round(vert["pitch_deg"], 2),
            "roll_vertical_lines_deg": None if not vert.get("reliable") else round(vert["roll_deg"], 2),
            "pitch_floor_plane_deg": round((fgeo.get("camera") or {}).get("pitch_deg"), 2)
            if (fgeo.get("camera") or {}).get("pitch_deg") is not None else None,
            "floor_status": fgeo.get("status"),
        },
        "floor_vps": [[round(v, 1) for v in p] for p in floor_vps],
        "corners": [{"x": round(c["x"], 1), "status": c["status"], "confidence": c["confidence"],
                     "turn_deg": c["turn_deg"]} for c in corners],
        "walls": [],
    }
    base = cv2.imread(str(seg / "CLEAN_ROOM.png")) if (seg / "CLEAN_ROOM.png").exists() else None
    base = cv2.resize(base, (w, h)) if base is not None else np.full((h, w, 3), 128, np.uint8)
    dim = (base * 0.45).astype(np.uint8)

    def draw_common(img):
        for top, bottom in lines:
            cv2.line(img, tuple(np.round(top).astype(int)), tuple(np.round(bottom).astype(int)), (0, 255, 255), 2)
        for r in floor_runs + ceil_runs:
            p = r["pts"]
            cv2.line(img, (int(p[0, 0]), int(p[0, 1])), (int(p[-1, 0]), int(p[-1, 1])), (255, 255, 0), 2)

    sheet = dim.copy()
    for wd in sorted(geo["walls"], key=lambda x: x["index"]):
        piece = _read(seg / "wall" / "walls" / f"{wd['id']}.png")
        if piece is None or piece.shape != wall.shape:
            continue
        n = int(piece.sum())
        counts = np.bincount(regions[piece], minlength=len(lines) + 1) if n else np.zeros(len(lines) + 1, int)
        main = int(np.argmax(counts))
        spill = n - int(counts[main])
        shares = sorted((counts / max(n, 1)).tolist(), reverse=True)
        d = wd.get("direction") or {}
        vp_img, vp_dir = _vp_point(d.get("vanishing_point"))
        if vp_img is not None and floor_vps:
            dist = min(math.hypot(vp_img[0] - p[0], vp_img[1] - p[1]) for p in floor_vps)
            vp_vs_floor = {"nearest_floor_vp_px": round(dist, 1), "relative_to_diagonal": round(dist / diag, 3)}
        elif vp_dir is not None and floor_vps:
            # a wall VP at infinity: compare its direction with the floor VPs' directions from the image centre
            angs = [math.degrees(math.atan2(p[1] - h / 2, p[0] - w / 2)) for p in floor_vps]
            diff = min(abs(((vp_dir - a) + 90) % 180 - 90) for a in angs)
            vp_vs_floor = {"wall_vp": "at infinity", "angle_to_nearest_floor_vp_dir_deg": round(diff, 1)}
        else:
            vp_vs_floor = None
        row = {
            "id": wd["id"], "pixels": n, "dot": wd.get("select_point"),
            "normal": [round(v, 3) for v in d["normal"]] if d.get("normal") else None,
            "yaw_deg": round(d["yaw_deg"], 1) if d.get("yaw_deg") is not None else None,
            "faces": d.get("faces"), "direction_source": d.get("source"),
            "vp": [round(v, 1) for v in vp_img] if vp_img else ("infinity" if vp_dir is not None else None),
            "spill_px": spill, "spill_pct": round(100.0 * spill / max(n, 1), 1),
            "two_directions": bool(len(shares) > 1 and shares[1] >= TWO_DIR_SHARE),
            "region_shares": [round(s, 3) for s in shares if s > 0],
            "vp_vs_floor": vp_vs_floor,
        }
        report["walls"].append(row)
        img = dim.copy()
        img[piece] = (img[piece] * 0.3 + np.array([60, 200, 60]) * 0.7).astype(np.uint8)
        wrong = piece & (regions != main)
        img[wrong] = (0, 0, 255)
        sheet[piece & ~wrong] = (sheet[piece & ~wrong] * 0.5 + np.array([60, 200, 60]) * 0.5).astype(np.uint8)
        sheet[wrong] = (0, 0, 255)
        draw_common(img)
        if wd.get("select_point"):
            cv2.circle(img, tuple(int(v) for v in wd["select_point"]), 7, (255, 255, 255), -1)
        cv2.putText(img, f"{wd['id']} spill {row['spill_pct']}% faces {row['faces']}", (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(WORK / f"{name}__{wd['id']}.png"), img)
    draw_common(sheet)
    for wd in geo["walls"]:
        if wd.get("select_point"):
            cv2.circle(sheet, tuple(int(v) for v in wd["select_point"]), 7, (255, 255, 255), -1)
            cv2.putText(sheet, wd["id"], (int(wd["select_point"][0]) + 9, int(wd["select_point"][1]) + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(WORK / f"{name}__all.png"), sheet)
    return report


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    rooms = {}
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            k, v = arg.split("=", 1)
            rooms[k] = Path(v)
    else:
        for base in (ROOT / "tests" / "fixtures" / "rooms", ROOT / "tests" / "validation" / "rooms"):
            for p in sorted(base.iterdir()):
                if (p / "segments" / "wall" / "wall_geometry.json").exists():
                    rooms[p.name] = p / "segments"
        for p in sorted((ROOT / "room-data" / "rooms").glob("room_*")):
            if (p / "job" / "segments" / "wall" / "wall_geometry.json").exists():
                rooms[p.name] = p / "job" / "segments"
    out = []
    for name, seg in rooms.items():
        r = measure(name, seg)
        out.append(r)
        c = r["camera"]
        print(f"== {name} {r['size'][0]}x{r['size'][1]}: {r['walls_detected']} wall pieces | pitch lines "
              f"{c['pitch_vertical_lines_deg']} vs floor {c['pitch_floor_plane_deg']} (roll {c['roll_vertical_lines_deg']}) "
              f"| floor VPs {r['floor_vps']}")
        print("   corners: " + ("; ".join(f"x={k['x']} {k['status']} ({k['confidence']}, turn {k['turn_deg']})"
                                       for k in r["corners"]) or "none found"))
        for wl in r["walls"]:
            print(f"   {wl['id']}: {wl['pixels']:>7} px  faces {str(wl['faces']):<10} yaw {wl['yaw_deg']}  vp {wl['vp']}  "
                  f"spill {wl['spill_px']} px ({wl['spill_pct']}%)  two directions: {wl['two_directions']}  "
                  f"vs floor VP: {wl['vp_vs_floor']}")
    (WORK / "report.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
