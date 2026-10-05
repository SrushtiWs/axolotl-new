"""
Step 2 visual check and after-counts.

    backend/.venv/bin/python tests/regression/step2_visual.py <out_dir>

1. wide_angle with the user's orange marble tile (backend/jobs/80dfd3734b69/tile.png,
   1200 x 1800 mm, 5 mm grout), render.STRAIGHT_EDGES off (before) and on (after):
   Before | After crops of the top-left (ceiling and fan) and of the left wall's
   top edge along its whole length.
2. wide_angle and living with the baseline tile: wall-tile pixels above / below a
   straight ceiling / floor line, before (baseline regions) and after (regions
   rendered now), with the lines refitted by tests/regression/step2_edges.py.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "backend"), str(ROOT / "tests/regression")]

import baseline_rooms  # noqa: E402
from perspective_engine import render as pe_render  # noqa: E402

BASELINE = ROOT / "backups/baselines-20261005_141642"
ORANGE = ROOT / "backend/jobs/80dfd3734b69/tile.png"


def pair(before, after, box, label):
    x0, y0, x1, y1 = box
    a, b = Image.fromarray(before[y0:y1, x0:x1]), Image.fromarray(after[y0:y1, x0:x1])
    out = Image.new("RGB", (a.width * 2 + 8, a.height), (255, 255, 255))
    out.paste(a, (0, 0))
    out.paste(b, (a.width + 8, 0))
    d = ImageDraw.Draw(out)
    d.text((6, 6), f"Before  {label}", fill=(255, 255, 0))
    d.text((a.width + 14, 6), "After", fill=(255, 255, 0))
    return out


def main() -> int:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    report = {}

    # 1. orange marble, before / after
    baseline_rooms.TILE, baseline_rooms.TILE_W_MM, baseline_rooms.TILE_H_MM = ORANGE, 1200.0, 1800.0
    src = baseline_rooms.ROOMS["wide_angle"]
    pe_render.STRAIGHT_EDGES = False
    photo, before, _, _ = baseline_rooms.render(src)
    pe_render.STRAIGHT_EDGES = True
    _, after, _, _ = baseline_rooms.render(src)
    changed = np.abs(after.astype(int) - before.astype(int)).max(axis=2) > 0
    allowed = np.asarray(Image.open(ROOT / "backups/step2-check/wide_angle__allowed_step2.png")) > 0
    report["orange_marble_wide_angle"] = {"changed_px": int(changed.sum()),
                                          "changed_outside_allowed_px": int((changed & ~allowed).sum())}
    Image.fromarray(before).save(out / "wide_angle_orange__before.png")
    Image.fromarray(after).save(out / "wide_angle_orange__after.png")
    pair(before, after, (0, 0, 700, 560), "top-left: ceiling and fan").save(out / "wide_angle_orange__topleft_before_after.png")
    # the left wall's top edge: x 0..wall corner (~484), around its ceiling line (y ~230 at x=0 .. ~450 at x=484)
    pair(before, after, (0, 120, 540, 520), "left wall top edge, full length").save(out / "wide_angle_orange__leftwall_top_before_after.png")

    # 2. after-counts with the baseline tile
    baseline_rooms.TILE, baseline_rooms.TILE_W_MM, baseline_rooms.TILE_H_MM = ROOT / "tests/fixtures/tile.png", 600.0, 600.0
    now = out / "regions_now"
    now.mkdir(exist_ok=True)
    for name in ("wide_angle", "living"):
        _, _, regions, _ = baseline_rooms.render(baseline_rooms.ROOMS[name])
        np.savez_compressed(now / f"{name}.npz", **regions)
        removed_before = np.asarray(Image.open(ROOT / f"backups/step2-check/{name}__allowed_step2.png")) > 0
        wall_now = np.zeros(removed_before.shape, bool)
        for k, m in regions.items():
            if k.startswith("wall-"):
                wall_now |= m
        report[f"{name}_wall_tile_in_old_allowed_area_now"] = int((wall_now & removed_before).sum())
    for label, folder in (("before", BASELINE), ("after", now)):
        subprocess.run([sys.executable, str(ROOT / "tests/regression/step2_edges.py"), str(folder),
                        str(out / f"edges_{label}"), "wide_angle", "living"], check=True, capture_output=True)
        d = json.loads((out / f"edges_{label}" / "step2_diagnose.json").read_text())
        for name in ("wide_angle", "living"):
            walls = d[name]["walls"]
            report.setdefault(name, {})[label] = {
                "removable_above_ceiling_lines_px": sum(r["ceiling"].get("removable_px", 0) for r in walls.values()),
                "removable_below_floor_lines_px": sum(r["floor"].get("removable_px", 0) for r in walls.values()),
                "all_past_ceiling_lines_px": d[name]["tiled_above_ceiling_lines_px"],
            }
    (out / "step2_visual.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
