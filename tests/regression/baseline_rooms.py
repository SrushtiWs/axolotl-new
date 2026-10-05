"""
Pixel baselines for the surgical-change steps.

    backend/.venv/bin/python tests/regression/baseline_rooms.py save  <out_dir>
    backend/.venv/bin/python tests/regression/baseline_rooms.py check <baseline_dir> <out_dir> [allowed_key]

Each room's segment job is COPIED to a scratch folder first and rendered from the
copy, so saved geometry in backend/jobs and tests/fixtures is never written.
The render is the floor plus every wall, each alone, composed region by region
onto the photo (as /compose does), with one fixed tile, tile size and grout.

save   renders every room twice; the two renders must be identical (else the
       baseline is useless and the room is reported NOT DETERMINISTIC).
       Writes <room>.png and <room>.npz (one mask per tiled region).
check  renders again and diffs against the baseline: changed pixel count, a
       diff image (changed pixels red over a dimmed photo), and when the step
       writes an allowed-area mask to <out_dir>/<room>__allowed_<key>.png,
       the changed pixels outside it (must be 0).
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "backend")]

import perspective_engine as pe  # noqa: E402
import reuse  # noqa: E402
from engine import TileSpec  # noqa: E402

ROOMS = {
    "wide_angle": ROOT / "backend/jobs/37a3d22ab0e9",
    "pillar": ROOT / "backend/jobs/47080a8bc81d",
    "sunset_living": ROOT / "backend/jobs/897dc4295929",
    "bedroom": ROOT / "tests/fixtures/rooms/bedroom",
    "living": ROOT / "tests/fixtures/rooms/living",
    "empty": ROOT / "tests/fixtures/rooms/empty",
}
TILE = ROOT / "tests/fixtures/tile.png"
TILE_W_MM, TILE_H_MM, GROUT_MM = 600.0, 600.0, 5.0


def render(src: Path):
    spec = TileSpec(artwork=np.asarray(Image.open(TILE).convert("RGB")), width_mm=TILE_W_MM,
                    height_mm=TILE_H_MM, rotation_deg=0, grout_mm=GROUT_MM)
    with tempfile.TemporaryDirectory() as tmp:
        job = Path(tmp) / src.name
        shutil.copytree(src, job)
        b = reuse.load(job)
        fl, wl = pe.load(job / "segments")
        geo = pe.ensure_geometry(job / "segments", b.room, fl, wl, clean=b.clean)
        out = b.room.copy()
        regions, refused = {}, {}
        for surface, idx in [("floor", None)] + [("wall", int(w["index"])) for w in geo["walls"]]:
            try:
                r = pe.render_tiled_room(b.room, fl, wl, spec, surface, (None, None, None), clean=b.clean,
                                         props=b.props, wall_index=idx, geometry=geo)
            except Exception as e:  # noqa: BLE001
                refused[f"{surface}{'' if idx is None else f'-{idx}'}"] = str(e)[:120]
                continue
            for key, m in r.regions.items():
                out[m] = r.result.composite[m]
                regions[key] = m
        return b.room, out, regions, refused


def diff_image(photo, changed):
    img = (photo.astype(np.float32) * 0.35).astype(np.uint8)
    img[changed] = (255, 0, 0)
    return img


def main() -> int:
    mode, a = sys.argv[1], Path(sys.argv[2])
    report = {}
    if mode == "save":
        a.mkdir(parents=True, exist_ok=True)
        for name, src in ROOMS.items():
            photo, first, regions, refused = render(src)
            _, second, _, _ = render(src)
            same = int((np.abs(first.astype(int) - second.astype(int)).max(axis=2) > 0).sum())
            Image.fromarray(first).save(a / f"{name}.png")
            np.savez_compressed(a / f"{name}.npz", **regions)
            report[name] = {"job": str(src.relative_to(ROOT)), "size": [int(photo.shape[1]), int(photo.shape[0])],
                            "regions": sorted(regions), "refused": refused,
                            "tiled_px": int(sum(m.sum() for m in regions.values())),
                            "rerun_changed_px": same, "deterministic": same == 0}
            print(name, json.dumps(report[name]), flush=True)
        report["_settings"] = {"tile": str(TILE.relative_to(ROOT)), "tile_mm": [TILE_W_MM, TILE_H_MM], "grout_mm": GROUT_MM}
        (a / "baseline.json").write_text(json.dumps(report, indent=1))
        return 0

    out = Path(sys.argv[3])
    key = sys.argv[4] if len(sys.argv) > 4 else None
    out.mkdir(parents=True, exist_ok=True)
    for name, src in ROOMS.items():
        base = np.asarray(Image.open(a / f"{name}.png").convert("RGB"))
        photo, now, _, refused = render(src)
        changed = np.abs(now.astype(int) - base.astype(int)).max(axis=2) > 0
        row = {"changed_px": int(changed.sum()), "identical": not changed.any(), "refused": refused}
        allowed_path = out / f"{name}__allowed_{key}.png" if key else None
        if allowed_path is not None and allowed_path.exists():
            allowed = np.asarray(Image.open(allowed_path).convert("L")) > 0
            row["allowed_px"] = int(allowed.sum())
            row["changed_outside_allowed_px"] = int((changed & ~allowed).sum())
        Image.fromarray(diff_image(photo, changed)).save(out / f"{name}__diff.png")
        Image.fromarray(now).save(out / f"{name}__now.png")
        report[name] = row
        print(name, json.dumps(row), flush=True)
    (out / "check.json").write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
