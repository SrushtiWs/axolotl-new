"""
Wall split from the floor junction and the ceiling line (fix Step 2).

    backend/.venv/bin/python tests/regression/wall_split.py

For every frozen room (tests/fixtures/rooms and tests/validation/rooms), the
wall split is recomputed exactly as a new Clean Room would do it -- the real
perspective_engine.ensure_geometry on a scratch copy of the room's segments
with its stored geometry removed (the fixtures are not touched) -- and checked:

  partition     every WALL_MASK pixel is in exactly one wall, none outside it
  corners       for the junction / edge layouts: walls = seen corners + 1, and
                every boundary between two walls is a corner an edge was seen
                to bend at (none guessed)
  compared      against the split the room was frozen with

Overlays go to tests/.work/wall_split/<room>.png.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "backend")]

import perspective_engine as pe  # noqa: E402
from perspective_engine import masks as pe_masks  # noqa: E402
from reuse import load as reuse_load  # noqa: E402

WORK = ROOT / "tests" / ".work" / "wall_split"
ROOMS = [p for base in (ROOT / "tests" / "fixtures" / "rooms", ROOT / "tests" / "validation" / "rooms")
         for p in sorted(base.iterdir()) if (p / "fixture.json").is_file()]


def _boundaries(masks_by_index: dict) -> list[float]:
    """Column boundaries between consecutive walls (by their column spans)."""
    spans = sorted((np.nonzero(m.any(axis=0))[0].min(), np.nonzero(m.any(axis=0))[0].max())
                   for m in masks_by_index.values() if m.any())
    return [(a[1] + b[0]) / 2.0 for a, b in zip(spans, spans[1:])]


def main() -> int:
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    failed = 0
    for room in ROOMS:
        seg = WORK / room.name / "segments"
        shutil.copytree(room / "segments", seg)
        for stored in ("floor", "wall", "room"):
            shutil.rmtree(seg / stored, ignore_errors=True)
        bundle = reuse_load(room)
        photo = next(room.glob("photo.*")).read_bytes()
        floor_mask, wall_mask = pe.load(seg)
        geo = pe.ensure_geometry(seg, bundle.room, floor_mask, wall_mask, clean=bundle.clean, photo_bytes=photo)
        wall_masks = geo["_wall_masks"]
        split = geo.get("wall_split") or {}
        floor, wall, _ = pe_masks.prepare(floor_mask, wall_mask, bundle.clean.shape[:2])

        count = sum(m.astype(np.int32) for m in wall_masks.values()) if wall_masks else np.zeros(wall.shape, np.int32)
        unassigned, double, outside = int((wall & (count == 0)).sum()), int((count > 1).sum()), int(((count > 0) & ~wall).sum())
        problems = []
        if unassigned or double or outside:
            problems.append(f"partition: {unassigned} unassigned, {double} in >1 wall, {outside} outside WALL_MASK")

        method = split.get("method")
        corners = [c["x"] for c in split.get("corners", [])] if method == "floor-junction+ceiling-line" else \
            (split.get("corners") or []) if method == "floor-junction" else None
        bounds = _boundaries(wall_masks)
        if corners is not None:
            if len(wall_masks) != len(corners) + 1:
                problems.append(f"{len(wall_masks)} walls for {len(corners)} seen corners")
            tol = 0.5 + 1.0
            unseen = [round(b) for b in bounds if not any(abs(b - c) <= tol for c in corners)]
            if unseen:
                problems.append(f"boundaries with no seen corner at x={unseen}")

        old = json.loads((room / "segments" / "wall" / "wall_geometry.json").read_text())
        old_method = (old.get("wall_split") or {}).get("method")
        ok = not problems
        failed += not ok
        sources = [str((w.get("direction") or {}).get("source") or w.get("orientation_source")) for w in geo["walls"]]
        print(f"{'PASS' if ok else 'FAIL'}  {room.name:<8} {method:<27} walls {len(wall_masks)} (was {len(old['walls'])}, {old_method})"
              f" | boundaries x={[round(b) for b in bounds]} | corners seen on: "
              f"{[c['seen_on'] for c in split.get('corners', [])] if method == 'floor-junction+ceiling-line' else '-'}"
              f" | edges seen: junction {split.get('junction_share', (split.get('edge_layout') or {}).get('junction_share'))},"
              f" ceiling {split.get('ceiling_share', (split.get('edge_layout') or {}).get('ceiling_share'))}"
              f" | directions {sources}")
        for p in problems:
            print(f"        - {p}")

        img = (cv2.cvtColor(bundle.clean, cv2.COLOR_RGB2BGR) * 0.55).astype(np.uint8)
        palette = [(255, 80, 80), (80, 160, 255), (255, 0, 255), (0, 255, 255), (255, 200, 0), (120, 255, 120)]
        for k, (i, m) in enumerate(sorted(wall_masks.items())):
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(img, cs, -1, palette[k % 6], 2)
            ys, xs = np.nonzero(m)
            cv2.putText(img, f"wall-{i} {sources[k] if k < len(sources) else ''}", (int(xs.mean()) - 60, int(ys.mean())),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, palette[k % 6], 1, cv2.LINE_AA)
        for key, col in (("junction_runs", (0, 0, 255)), ("ceiling_runs", (0, 200, 255))):
            for s in split.get(key, []):
                cv2.line(img, (int(s[0]), int(s[1])), (int(s[2]), int(s[3])), col, 2)
        cv2.imwrite(str(WORK / f"{room.name}.png"), img)

    print(f"\n{len(ROOMS) - failed}/{len(ROOMS)} passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
