"""
Room validation (Step 11): what the engine estimates for each room, and -- when
a real measured size is on file -- how far off it is.

    backend/.venv/bin/python tests/validation/validate_rooms.py

Rooms and their measured sizes come from tests/validation/measured_rooms.json
(test data only). For each room, from its frozen Clean Room job:

  AUTO (nothing typed)   estimated W / L / H with sources, lower-bound flags,
                         status, confidence; error % vs the measured size
                         (a lower bound is checked as "<= measured" instead)
  VP                     floor VP source + confidence, VP_Y confidence, the
                         room axes' source, VP_X / VP_Z in the image
  camera                 focal (px, source), pitch / yaw / roll (+ why), camera
                         height (mm, source)
  seam                   floor/wall seam gap and tile-scale ratio per wall
                         (tests/regression/seam_scale.py)
  reprojection           room corners, mean / max px
  MANUAL (measured typed) "Consistent" / "Conflict with photo" + residuals,
                         only for rooms with a measured size

Writes tests/validation/report.md and report.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "backend"), str(ROOT / "tests" / "regression")]

import perspective_engine as pe  # noqa: E402
from seam_scale import seam  # noqa: E402
from tiles_backend.perspective_engine import TileRequest, render_room  # noqa: E402
from tiles_backend.perspective_engine.surface.wall.junction_layout import junction_points  # noqa: E402

HERE = Path(__file__).resolve().parent
DIMS = ("width", "length", "height")


def _vp(v):
    if not v:
        return "-"
    return "at infinity" if v.get("at_infinity") else f"({v['image'][0]:.0f}, {v['image'][1]:.0f})"


def _render(rgb, F, W, tile, surface, dims, geo, **kw):
    req = TileRequest(tile_width_mm=1200, tile_height_mm=1800, grout_mm=5,
                      room_width_mm=dims.get("width"), room_length_mm=dims.get("length"),
                      room_height_mm=dims.get("height"))
    return render_room(rgb, F, W, tile, surface, req, geometry=geo, **kw)


def validate(entry: dict, tile) -> dict:
    seg = ROOT / entry["fixture"] / "segments"
    clean = cv2.imread(str(seg / "CLEAN_ROOM.png"))
    rgb = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)
    Fm, Wm = cv2.imread(str(seg / "FLOOR_MASK.png"), 0), cv2.imread(str(seg / "WALL_MASK.png"), 0)
    F, W = Fm > 127, Wm > 127
    geo = pe.ensure_geometry(seg, rgb, Fm, Wm, clean=rgb)
    h, w = F.shape
    fr = geo.get("room_frame") or {}
    out = {"name": entry["name"], "image": [w, h], "notes": entry.get("notes")}

    try:
        rf = _render(rgb, F, W, tile, "floor", {}, geo)
    except Exception as ex:                                  # noqa: BLE001 (reported)
        out["error"] = f"floor render failed: {ex}"
        return out
    fi = rf.info["surfaces"]["floor"]
    room = rf.info.get("room") or {}
    cam = room.get("camera") or {}
    vps = room.get("vanishing_points") or {}
    out["auto"] = {
        "status": room.get("status"), "confidence": room.get("confidence"),
        **{f"{d}_mm": room.get(f"{d}_mm") for d in DIMS},
        **{f"{d}_source": room.get(f"{d}_source") for d in DIMS},
        "width_is_lower_bound": room.get("width_is_lower_bound"),
        "length_is_lower_bound": room.get("length_is_lower_bound"),
        "room_check": room.get("room_check"),
    }
    out["vp"] = {
        "floor_vp_source": fi.get("vp_source"), "floor_vp_confidence": fi.get("vp_confidence"),
        "vp_y_confidence": ((vps.get("VP_Y_detected") or {}).get("confidence")),
        "axes_source": fr.get("axes_source"), "VP_X": _vp(vps.get("VP_X")), "VP_Z": _vp(vps.get("VP_Z")),
    }
    out["camera"] = {
        "focal_px": fi.get("focal_px"), "focal_source": fi.get("focal_source"),
        "pitch_deg": cam.get("pitch_deg"), "yaw_deg": cam.get("yaw_deg"), "roll_deg": cam.get("roll_deg"),
        "roll_source": cam.get("roll_source"),
        "height_mm": room.get("camera_height_mm"), "height_source": room.get("camera_height_source"),
    }
    out["reprojection_px"] = {"mean": room.get("reprojection_error_mean_px"), "max": room.get("reprojection_error_max_px"),
                              "corners": len(room.get("reprojection_error_px") or {})}

    fplane = [*fi["plane"][:3], fi["plane"][3] * fi["mm_per_unit"]]
    cx, cy = w / 2.0, h / 2.0
    walls = []
    for wd in geo.get("walls") or []:
        i = wd["index"]
        try:
            rw = _render(rgb, F, W, tile, "wall", {}, geo, wall_index=i)
        except Exception as ex:                              # noqa: BLE001 (reported)
            walls.append({"wall": f"wall-{i}", "error": str(ex)[:100]})
            continue
        winfo = rw.info["surfaces"]["wall"]
        e = winfo["walls"][0]
        wplane = [*e["plane"][:3], e["plane"][3] * e["mm_per_unit"]]
        s = seam(fplane, fi["focal_px"], wplane, winfo["focal_px"], junction_points(F, geo["_wall_masks"][i]), cx, cy)
        walls.append({"wall": f"wall-{i}", "placement": e.get("placement"), "validated": e.get("validated"),
                      "seam_gap": None if s is None else s[0], "scale_ratio": None if s is None else s[1]})
    out["walls"] = walls

    measured = {d: v for d, v in (entry.get("measured_mm") or {}).items() if v}
    out["measured_mm"] = measured or None
    if measured:
        errs = {}
        for d, m in measured.items():
            est = room.get(f"{d}_mm")
            if est is None:
                errs[d] = {"estimate": None, "note": "not estimated"}
            elif room.get(f"{d}_is_lower_bound"):
                errs[d] = {"estimate": est, "lower_bound": True, "holds": est <= m * 1.0,
                           "gap_pct": 100.0 * (est - m) / m}
            else:
                errs[d] = {"estimate": est, "error_pct": 100.0 * (est - m) / m}
        out["error_vs_measured"] = errs
        try:
            rm = _render(rgb, F, W, tile, "floor", measured, geo).info.get("room") or {}
            out["manual"] = {"status": rm.get("status"), "room_check": rm.get("room_check"),
                             "residuals": rm.get("residuals"), "camera_height_mm": rm.get("camera_height_mm")}
        except Exception as ex:                              # noqa: BLE001 (reported)
            out["manual"] = {"error": str(ex)[:120]}
    return out


def _f(v, fmt="{:.0f}", none="-"):
    return none if v is None else fmt.format(v)


def report_md(results: list[dict]) -> str:
    lines = ["# Room validation", "",
             "Generated by `tests/validation/validate_rooms.py`. Estimates are AUTO (nothing typed): "
             "the scale is the 1.5 m camera-height prior, so every size below is ESTIMATED. "
             "Error % needs a real measured size in `measured_rooms.json`; rooms without one show none.", ""]
    measured = [r for r in results if r.get("measured_mm")]
    lines += [f"Rooms: {len(results)}. With a measured size: {len(measured)}.", ""]
    lines += ["## Size (AUTO)", "", "| Room | Image | W mm | L mm | H mm | Status | Confidence | Error vs measured |",
              "|---|---|---|---|---|---|---|---|"]
    for r in results:
        a = r.get("auto") or {}
        lb = lambda d: ("≥" if a.get(f"{d}_is_lower_bound") else "")  # noqa: E731
        err = "no measurement"
        if r.get("error_vs_measured"):
            parts = []
            for d, e in r["error_vs_measured"].items():
                if e.get("lower_bound"):
                    parts.append(f"{d[0].upper()} lower bound {'holds' if e['holds'] else 'VIOLATED'} ({e['gap_pct']:+.1f}%)")
                elif e.get("error_pct") is not None:
                    parts.append(f"{d[0].upper()} {e['error_pct']:+.1f}%")
                else:
                    parts.append(f"{d[0].upper()} not estimated")
            err = ", ".join(parts)
        lines.append(f"| {r['name']} | {r['image'][0]}×{r['image'][1]} | {lb('width')}{_f(a.get('width_mm'))} | "
                     f"{lb('length')}{_f(a.get('length_mm'))} | {_f(a.get('height_mm'))} | {a.get('status', r.get('error', '-'))} | "
                     f"{_f(a.get('confidence'), '{:.3f}')} | {err} |")
    lines += ["", "## Vanishing points and camera", "",
              "| Room | Floor VP (conf) | VP_Y conf | Axes from | Focal px (source) | Pitch / yaw / roll ° | Roll | Camera height |",
              "|---|---|---|---|---|---|---|---|"]
    for r in results:
        v, c = r.get("vp") or {}, r.get("camera") or {}
        lines.append(f"| {r['name']} | {v.get('floor_vp_source', '-')} ({_f(v.get('floor_vp_confidence'), '{:.2f}')}) | "
                     f"{_f(v.get('vp_y_confidence'), '{:.2f}')} | {v.get('axes_source', '-')} | "
                     f"{_f(c.get('focal_px'))} ({c.get('focal_source', '-')}) | {_f(c.get('pitch_deg'), '{:.1f}')} / "
                     f"{_f(c.get('yaw_deg'), '{:.1f}')} / {_f(c.get('roll_deg'), '{:.2f}')} | {c.get('roll_source', '-')} | "
                     f"{_f(c.get('height_mm'))} mm ({c.get('height_source', '-')}) |")
    lines += ["", "## Seam and reprojection", "",
              "| Room | Reprojection mean / max px (corners) | Walls: seam gap % / scale ratio |", "|---|---|---|"]
    for r in results:
        rp = r.get("reprojection_px") or {}
        ws = "; ".join(
            f"{w['wall']} " + (f"n/a ({w['error'][:40]})" if w.get("error") else
                               f"{_f(None if w['seam_gap'] is None else 100 * w['seam_gap'], '{:.1f}')}% / "
                               f"{_f(w['scale_ratio'], '{:.2f}')}{'' if w.get('validated') else ' (not validated)'}")
            for w in r.get("walls") or [])
        lines.append(f"| {r['name']} | {_f(rp.get('mean'), '{:.1f}')} / {_f(rp.get('max'), '{:.1f}')} ({rp.get('corners', 0)}) | {ws or '-'} |")
    if measured:
        lines += ["", "## Measured size typed (MANUAL)", "", "| Room | Status line | Residuals |", "|---|---|---|"]
        for r in measured:
            m = r.get("manual") or {}
            lines.append(f"| {r['name']} | {m.get('room_check', m.get('error', '-'))} | {m.get('residuals')} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    spec = json.loads((HERE / "measured_rooms.json").read_text())
    tile = cv2.cvtColor(cv2.imread(str(ROOT / "tests" / "fixtures" / "tile.png")), cv2.COLOR_BGR2RGB)
    results = []
    for entry in spec["rooms"]:
        print(f"== {entry['name']}", flush=True)
        results.append(validate(entry, tile))
    (HERE / "report.json").write_text(json.dumps(results, indent=2, default=float))
    md = report_md(results)
    (HERE / "report.md").write_text(md)
    print("\n" + md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
