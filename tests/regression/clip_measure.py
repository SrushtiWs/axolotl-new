"""
Tile pixels outside the final white mask, by layer (read only).

    backend/.venv/bin/python tests/regression/clip_measure.py <segment_job> <render_job> [<render_job> ...]

The final white mask = (FLOOR_MASK | WALL_MASK) minus objects (the render's own
object alpha, three/objects.png, > 0), at the render's resolution. A pixel
"differs" when any channel is more than 2 levels from the photo. Per render:

  tile layer     raw.png (engine output: tiles + the room's lighting) differs
                 from CLEAN_ROOM.png outside FLOOR_MASK | WALL_MASK
  feathered      result.png differs from the photo where the object alpha is
                 between 0 and 1 (tile showing through an object's soft edge)
  put-back       result.png differs from the photo where the object alpha is 1,
                 or outside every mask with no tile under it (the clean room
                 showing where an object was removed)
  composed       the composed image the app shows (composed_*.png) vs the photo,
                 outside the white mask
  band           the floor-wall line band (1% of the height each side of the
                 floor-wall boundary): share of its mask pixels left untiled
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


def rgb(path):
    return np.asarray(Image.open(path).convert("RGB")).astype(np.int16)


def mask(path, shape):
    m = np.asarray(Image.open(path).convert("L")) > 127
    if m.shape != shape:
        m = cv2.resize(m.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR) >= 0.5
    return m


def differs(a, b):
    return np.abs(a - b).max(axis=2) > 2


def main() -> int:
    seg = Path(sys.argv[1]) / "segments"
    out = []
    for rj in sys.argv[2:]:
        job = Path(rj)
        photo = rgb(job / "original.png")
        shape = photo.shape[:2]
        h = shape[0]
        floor = mask(seg / "FLOOR_MASK.png", shape)
        wall = mask(seg / "WALL_MASK.png", shape)
        alpha = np.asarray(Image.open(job / "three" / "objects.png"))[..., 3].astype(np.float32) / 255.0
        objects = alpha > 0
        white = (floor | wall) & ~objects
        raw, result, clean = rgb(job / "raw.png"), rgb(job / "result.png"), rgb(job / "CLEAN_ROOM.png")
        tiles = differs(raw, clean)
        changed = differs(result, photo)
        outside = changed & ~white
        row = {
            "render": job.name,
            "tiled_px": int((tiles & (floor | wall)).sum()),
            "changed_outside_white_mask": int(outside.sum()),
            "by_layer": {
                "tile layer outside floor|wall": int((tiles & ~(floor | wall)).sum()),
                "feathered object edge": int((outside & (alpha > 0) & (alpha < 1)).sum()),
                "object put-back (alpha 1)": int((outside & (alpha >= 1)).sum()),
                "clean room outside masks": int((outside & ~objects & ~(floor | wall) & ~tiles).sum()),
            },
        }
        # the floor-wall band
        k = max(1, int(round(0.01 * h)))
        kern = np.ones((2 * k + 1, 1), np.uint8)
        boundary = cv2.dilate(floor.astype(np.uint8), kern).astype(bool) & cv2.dilate(wall.astype(np.uint8), kern).astype(bool)
        band = boundary & (floor | wall) & ~objects
        row["band_px"] = int(band.sum())
        row["band_untiled_pct"] = round(100.0 * (band & ~tiles).sum() / max(band.sum(), 1), 2)
        for comp in sorted(glob.glob(str(job / "composed_*.png"))):
            c = rgb(comp)
            row[Path(comp).name + " outside white mask"] = int((differs(c, photo) & ~white).sum())
        out.append(row)
        print(json.dumps(row))
    return 0


if __name__ == "__main__":
    sys.exit(main())
