"""
Floor / wall mask audit (read only). Nothing in the pipeline is changed.

    backend/.venv/bin/python tests/regression/mask_audit.py [room ...]

For each frozen room, from its clean room (the image floor_wall.py runs on):

  stages     the class map alone (tidy=False), after the guided-filter snap
             (tidy=True, refine=False), and the final stored masks
             (FLOOR_MASK.png / WALL_MASK.png, after the GrabCut refine)
  boundary   every wall-boundary pixel classified by what it touches:
             floor, ceiling, another wall piece (the split), a removed object,
             an opening (window / door / curtain / mirror, by the segmenter's
             own label), the image border, or other
  per type   pixels, edge offset (distance along the boundary's normal to the
             strongest image edge within the refine band: median / p90 px),
             and roughness (boundary length / length after light smoothing;
             1.000 = clean)
  openings   wall-mask pixels the segmenter labels window / door / curtain /
             mirror (should be 0)
  masks      holes, islands, floor/wall overlap; wall pieces and tiny pieces

Overlays: tests/.work/mask_audit/<room>__masks.png and __boundaries.png.
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

import floor_wall  # noqa: E402
import surfaces  # noqa: E402
from reuse import load as reuse_load  # noqa: E402

WORK = ROOT / "tests" / ".work" / "mask_audit"
ALL_ROOMS = {p.name: p for base in (ROOT / "tests" / "fixtures" / "rooms", ROOT / "tests" / "validation" / "rooms")
             for p in sorted(base.iterdir()) if (p / "fixture.json").is_file()}
OPENINGS = {"windowpane", "door", "curtain", "mirror", "screen door", "double door"}
TYPES = ["wall/floor", "wall/ceiling", "wall/wall", "wall/object", "wall/opening", "wall/border", "wall/other"]
COLOURS = {"wall/floor": (0, 255, 0), "wall/ceiling": (0, 200, 255), "wall/wall": (255, 0, 255),
           "wall/object": (0, 0, 255), "wall/opening": (255, 255, 0), "wall/border": (128, 128, 128),
           "wall/other": (255, 255, 255)}


def _objects(seg: Path, shape) -> np.ndarray:
    o = cv2.imread(str(seg / "ALL_OBJECTS.png"), cv2.IMREAD_UNCHANGED)
    if o is None:
        return np.zeros(shape, bool)
    m = (o[..., 3] > 0) if o.ndim == 3 and o.shape[2] == 4 else (o > 127)
    if m.shape != shape:
        m = cv2.resize(m.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    return m


def _pieces(seg: Path, shape) -> np.ndarray:
    lab = np.zeros(shape, np.int32)
    for k, p in enumerate(sorted((seg / "wall" / "walls").glob("wall-*.png")), start=1):
        m = cv2.imread(str(p), 0) > 127
        lab[m] = k
    return lab


def _classify(wall, floor, pieces, objects, labels, names):
    """Per wall-boundary pixel: its type, and the (dy, dx) normal pointing out of the wall."""
    h, w = wall.shape
    out = {t: [] for t in TYPES}
    ys, xs = np.nonzero(wall)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        ny, nx = ys + dy, xs + dx
        border = (ny < 0) | (ny >= h) | (nx < 0) | (nx >= w)
        nyc, nxc = np.clip(ny, 0, h - 1), np.clip(nx, 0, w - 1)
        other_piece = wall[nyc, nxc] & (pieces[nyc, nxc] != pieces[ys, xs]) & (pieces[ys, xs] > 0) & ~border
        outside = ~wall[nyc, nxc] | border
        for i in np.nonzero(outside | other_piece)[0]:
            y, x = ys[i], xs[i]
            if border[i]:
                t = "wall/border"
            elif other_piece[i]:
                t = "wall/wall"
            elif objects[nyc[i], nxc[i]]:
                t = "wall/object"
            elif floor[nyc[i], nxc[i]]:
                t = "wall/floor"
            else:
                name = names.get(int(labels[nyc[i], nxc[i]]), "?")
                t = "wall/ceiling" if name == "ceiling" else "wall/opening" if name in OPENINGS else "wall/other"
            out[t].append((y, x, dy, dx))
    return {t: np.array(v, int).reshape(-1, 4) for t, v in out.items()}


def _edge_offset(pts, grad, band):
    """Median / p90 distance along each pixel's normal to the strongest image edge within `band`."""
    if not len(pts):
        return None, None
    h, w = grad.shape
    k = np.arange(-band, band + 1)
    ys = np.clip(pts[:, 0:1] + pts[:, 2:3] * k[None, :], 0, h - 1)
    xs = np.clip(pts[:, 1:2] + pts[:, 3:4] * k[None, :], 0, w - 1)
    off = np.abs(k[np.argmax(grad[ys, xs], axis=1)])
    return float(np.median(off)), float(np.percentile(off, 90))


def _roughness(mask) -> float:
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    raw = sum(cv2.arcLength(c, True) for c in cs)
    sm = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 2.0) > 0.5
    cs2, _ = cv2.findContours(sm.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    smooth = sum(cv2.arcLength(c, True) for c in cs2)
    return raw / max(smooth, 1.0)


def _holes_islands(mask, min_frac=0.001):
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    area = max(int(mask.sum()), 1)
    islands = int(sum(1 for i in range(1, n) if st[i, cv2.CC_STAT_AREA] < min_frac * area))
    inv = (~mask).astype(np.uint8)
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(inv, 4)
    h, w = mask.shape
    holes = 0
    for i in range(1, n2):
        x, y, bw, bh, a = st2[i]
        if x > 0 and y > 0 and x + bw < w and y + bh < h:
            holes += 1
    return int(n - 1), islands, holes


def audit(name: str, room: Path) -> dict:
    seg = room / "segments"
    bundle = reuse_load(room)
    clean = bundle.clean
    h, w = clean.shape[:2]
    band = max(4, min(24, int(round(0.012 * max(h, w)))))
    gray = cv2.cvtColor(clean, cv2.COLOR_RGB2GRAY).astype(np.float32)
    grad = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    labels, names = surfaces.label_map(clean)
    objects = _objects(seg, (h, w))
    pieces = _pieces(seg, (h, w))

    stages = {
        "class map": floor_wall.detect_floor_wall(clean, tidy=False),
        "snap": floor_wall.detect_floor_wall(clean, tidy=True, refine=False),
    }
    final_floor = cv2.imread(str(seg / "FLOOR_MASK.png"), 0) > 127
    final_wall = cv2.imread(str(seg / "WALL_MASK.png"), 0) > 127
    rerun = floor_wall.detect_floor_wall(clean)
    result = {"name": name, "size": [w, h], "band_px": band,
              "final_equals_rerun": bool(np.array_equal(rerun.floor, final_floor) and np.array_equal(rerun.wall, final_wall)),
              "stages": {}}
    for stage, fw in list(stages.items()) + [("final", None)]:
        floor, wall = (final_floor, final_wall) if fw is None else (fw.floor, fw.wall)
        pc = pieces if fw is None else np.zeros_like(pieces)
        b = _classify(wall, floor, pc, objects, labels, names)
        per = {}
        for t in TYPES:
            med, p90 = _edge_offset(b[t], grad, band)
            per[t] = {"px": int(len(b[t])), "edge_offset_med": med, "edge_offset_p90": p90}
        comps, islands, holes = _holes_islands(wall)
        fcomps, fislands, fholes = _holes_islands(floor)
        result["stages"][stage] = {
            "wall_px": int(wall.sum()), "floor_px": int(floor.sum()), "overlap_px": int((floor & wall).sum()),
            "wall_roughness": round(_roughness(wall), 4), "floor_roughness": round(_roughness(floor), 4),
            "wall_components": comps, "wall_islands": islands, "wall_holes": holes,
            "floor_components": fcomps, "floor_islands": fislands, "floor_holes": fholes,
            "wall_px_labelled_opening": int((wall & np.isin(labels, [k for k, v in names.items() if v in OPENINGS])).sum()),
            "wall_px_on_objects": int((wall & objects).sum()),
            "boundary": per,
        }
        if fw is None:
            fin_b = b
    # pieces
    areas = [int((pieces == k).sum()) for k in range(1, pieces.max() + 1)]
    result["pieces"] = {"count": len(areas), "areas": areas,
                        "tiny(<1% of wall)": sum(1 for a in areas if a < 0.01 * max(final_wall.sum(), 1))}
    result["split"] = (json.loads((seg / "wall" / "wall_geometry.json").read_text()).get("wall_split") or {}).get("method")

    # overlays
    WORK.mkdir(parents=True, exist_ok=True)
    img = cv2.cvtColor(clean, cv2.COLOR_RGB2BGR).astype(np.float32)
    ov = img.copy()
    ov[final_floor] = ov[final_floor] * 0.5 + np.array([255, 168, 46]) * 0.5
    ov[final_wall] = ov[final_wall] * 0.5 + np.array([46, 138, 255]) * 0.5
    ov[objects] = ov[objects] * 0.4 + np.array([0, 0, 255]) * 0.6
    cv2.imwrite(str(WORK / f"{name}__masks.png"), ov.astype(np.uint8))
    bd = (img * 0.45).astype(np.uint8)
    for t, pts in fin_b.items():
        bd[pts[:, 0], pts[:, 1]] = COLOURS[t]
    y0 = 18
    for t in TYPES:
        cv2.putText(bd, t, (8, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOURS[t], 1, cv2.LINE_AA); y0 += 18
    cv2.imwrite(str(WORK / f"{name}__boundaries.png"), bd)
    return result


def main() -> int:
    names = sys.argv[1:] or ["empty", "living", "bedroom", "room01"]
    out = []
    for n in names:
        r = audit(n, ALL_ROOMS[n])
        out.append(r)
        print(f"== {n} ({r['size'][0]}x{r['size'][1]}, refine band {r['band_px']} px, split {r['split']}, "
              f"final masks = fresh re-run: {r['final_equals_rerun']})")
        for stage, s in r["stages"].items():
            print(f"   {stage:<9} wall {s['wall_px']:>7} px rough {s['wall_roughness']:.3f} comps {s['wall_components']} "
                  f"islands {s['wall_islands']} holes {s['wall_holes']} | floor {s['floor_px']:>7} rough {s['floor_roughness']:.3f} "
                  f"islands {s['floor_islands']} holes {s['floor_holes']} | overlap {s['overlap_px']} | wall px labelled opening "
                  f"{s['wall_px_labelled_opening']} | wall px on objects {s['wall_px_on_objects']}")
            parts = []
            for t, v in s["boundary"].items():
                if not v["px"]:
                    continue
                med, p90 = v["edge_offset_med"], v["edge_offset_p90"]
                off = "-" if med is None else f"{med:.0f}/{p90:.0f}"
                parts.append(f"{t.split('/')[1]} {v['px']}px off {off}")
            print("             " + "; ".join(parts))
        print(f"   pieces {r['pieces']}")
    (WORK / "audit.json").write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
