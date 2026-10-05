"""
Step 2B (room_03 / room_04 / room_07): corners across hidden gaps, measured
against reference corners marked by eye.

    backend/.venv/bin/python tests/regression/plane_step2b.py

REFERENCE corners are test data, estimated by eye on each photo (ESTIMATED,
about +-10 px; the camera is level, so each corner is a vertical line). A band
of TOL px around each is excluded from every count below, so the estimate's
own error cannot be counted as a wrong pixel. Production code never reads them.

Geometry is recomputed in scratch copies with corner_cut.ENABLED False
(before) and True (after). Reported per room:
  lines        floor-wall / ceiling-wall straight-run length (px) on the wall mask
  corners      what corner_cut confirmed, with its two witnesses
  spill        per wall piece: pixels outside the reference slot holding most of it
  claimed>1    wall pixels in more than one piece
  back only    the back wall alone rendered (real render_room, room 12x12x10 ft
               for scale only -- scale does not change which pixels are tiled): tiled pixels
               outside the reference back-wall slot (target 0)
Overlays: tests/.work/plane_step2b/<room>__{before,after}.png
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "backend"), str(ROOT / "tests" / "regression")]

import corner_cut_check as ccc  # noqa: E402
import perspective_engine as pe  # noqa: E402
from perspective_engine import masks as pe_masks  # noqa: E402
from reuse import load as reuse_load  # noqa: E402
from tiles_backend.perspective_engine import TileRequest, render_room  # noqa: E402
from tiles_backend.perspective_engine.surface.wall import corner_cut, edge_layout  # noqa: E402
from tiles_backend.perspective_engine.surface.wall import junction_layout as jl  # noqa: E402

WORK = ROOT / "tests" / ".work" / "plane_step2b"
REFERENCE = {"room_03": [300, 830], "room_04": [328, 855], "room_07": [305, 965]}   # ESTIMATED by eye
TOL = 10


def ref_slots(w: int, xs: list) -> np.ndarray:
    """Per column: reference slot index; -1 inside the +-TOL band of a reference corner."""
    col = np.arange(w)
    lab = np.searchsorted(np.asarray(xs), col, side="right")
    for x in xs:
        lab[np.abs(col - x) <= TOL] = -1
    return lab


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    tile = np.dstack([((np.arange(256)[:, None] // 64 + np.arange(256)[None, :] // 64) % 2 * 255).astype(np.uint8)] * 3)
    rooms = ccc.rooms()
    out = []
    for name, xs in REFERENCE.items():
        job, seg, photo = rooms[name]
        row = {"room": name, "reference_corners_x": xs}
        for label, enabled in (("before", False), ("after", True)):
            target = ccc.recompute(name, job, seg, photo, enabled, WORK / "scratch" / label)
            bundle = reuse_load(job)
            floor_mask, wall_mask = pe.load(target)
            geo = pe.ensure_geometry(target, bundle.room, floor_mask, wall_mask, clean=bundle.clean)
            floor, wall, _ = pe_masks.prepare(floor_mask, wall_mask, bundle.clean.shape[:2])
            h, w = wall.shape
            diag = math.hypot(h, w)
            colslot = ref_slots(w, xs)
            lab = np.broadcast_to(colslot, (h, w))
            objects = cv2.imread(str(target / "ALL_OBJECTS.png"), cv2.IMREAD_UNCHANGED)
            objects = cv2.resize((objects[..., 3] > 127).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
            fr = corner_cut.runs(jl.junction_points(floor, wall, objects), w, diag)
            cr = corner_cut.runs(edge_layout.ceiling_points(wall, objects), w, diag)
            masks = geo["_wall_masks"]
            count = sum(m.astype(np.int32) for m in masks.values())
            walls = []
            for i, m in sorted(masks.items()):
                counts = np.array([int((m & (lab == s)).sum()) for s in range(len(xs) + 1)])
                home = int(np.argmax(counts))
                walls.append({"id": f"wall-{i}", "pixels": int(m.sum()), "ref_slot": home,
                              "spill_px": int(counts.sum() - counts[home])})
            # the back wall = the piece with most pixels in the middle reference slot
            mid = 1
            back = max(masks, key=lambda i: int((masks[i] & (lab == mid)).sum()))
            # A back-wall slot with no wall piece in it (that wall is all window /
            # curtain): nothing can be selected there, so nothing lands on a wrong wall.
            has_back = bool((masks[back] & (lab == mid)).any())
            if has_back:
                rw = render_room(bundle.clean, floor, wall, tile, "wall",
                                 TileRequest(tile_width_mm=600, tile_height_mm=600, grout_mm=0,
                                             room_width_mm=3658, room_length_mm=3658, room_height_mm=3048),
                                 wall_index=back, geometry=geo)
                tiled = rw.wall_tiled
            else:
                tiled = np.zeros((h, w), bool)
            wrong = tiled & (lab != mid) & (lab != -1)
            split = json.loads((target / "wall" / "wall_geometry.json").read_text()).get("wall_split") or {}
            cut = split.get("corner_cut") or {}
            row[label] = {
                "split_method": split.get("method"),
                "line_px": {"floor_wall": round(sum(r["x1"] - r["x0"] for r in fr), 1),
                            "ceiling_wall": round(sum(r["x1"] - r["x0"] for r in cr), 1)},
                "corners": cut.get("corners"), "uncertain": cut.get("uncertain_corners"),
                "one_straight_edge_slots": cut.get("one_straight_edge_slots"),
                "seams": cut.get("slanted_seams_merged"),
                "walls": walls, "spill_total": sum(x["spill_px"] for x in walls),
                "claimed_more_than_once": int((count > 1).sum()),
                "back_wall": f"wall-{back}" if has_back else "none (no wall piece in the back-wall slot)",
                "back_tiled_px": int(tiled.sum()),
                "back_tiled_wrong_wall_px": int(wrong.sum()),
            }
            img = (cv2.cvtColor(bundle.clean, cv2.COLOR_RGB2BGR) * 0.5).astype(np.uint8)
            img[tiled] = (60, 200, 60)
            img[wrong] = (0, 0, 255)
            for x in xs:
                cv2.line(img, (x, 0), (x, h), (0, 255, 255), 1)
            for c in cut.get("corners") or []:
                cv2.line(img, (int(c["x"]), 0), (int(c["x"]), h), (255, 0, 255), 2)
            cv2.putText(img, f"{name} {label}: back wall {row[label]['back_wall']} wrong {int(wrong.sum())} px",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imwrite(str(WORK / f"{name}__{label}.png"), img)
        out.append(row)
        b, a = row["before"], row["after"]
        print(f"== {name} (reference corners x={xs}, ESTIMATED +-{TOL}px)")
        for label in ("before", "after"):
            r = row[label]
            print(f"   {label:<6} lines floor {r['line_px']['floor_wall']} / ceiling {r['line_px']['ceiling_wall']} px | "
                  f"walls {len(r['walls'])} | spill {r['spill_total']} px | claimed>1 {r['claimed_more_than_once']} | "
                  f"back only ({r['back_wall']}): {r['back_tiled_px']} tiled, {r['back_tiled_wrong_wall_px']} on the wrong wall")
        for c in a["corners"] or []:
            print(f"      corner x={c['x']}: {c['witness']}")
        for s in a["one_straight_edge_slots"] or []:
            print(f"      one plane: slot {s['slot']} straight run {s['run']} ({s['angle_deg']} deg) merged {s['merged']}")
        for s in (row["after"].get("seams") or []):
            print(f"      one plane: seam {s['seam_off_vertical_deg']} deg off the photo's vertical "
                  f"({s['local_vertical_deg']} deg) merged {s['merged']}")
        for u in a["uncertain"] or []:
            print(f"      uncertain x={u['x']} ({u['seen_in']})")
    (WORK / "report.json").write_text(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
