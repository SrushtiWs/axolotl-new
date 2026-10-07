"""
L1 on the ORIGINAL photo vs on the CLEAN ROOM (no pixel or code-path change).

    backend/.venv/bin/python tests/regression/l1_source_compare.py <out_dir> <room>

inpainted area  pixels where the clean room differs from the photo by more than
                INPAINT_DIFF levels (any channel), cleaned of 1-px noise
1  kept lines lying more than 50 % inside the inpainted area, for each run
2  the L1 counts for each run (same module, same object / floor / label masks)
3  per family (clustered as in l1_lines.py): lines found in one run and not the
   other (same line = within 2 deg and 2 px, overlapping)
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "backend"), str(ROOT / "tests/regression")]

import perspective_engine as pe  # noqa: E402
import reuse  # noqa: E402
import surfaces  # noqa: E402
from baseline_rooms import ROOMS  # noqa: E402
from l1_lines import families  # noqa: E402
from perspective_engine import masks, strong_lines as sl  # noqa: E402

INPAINT_DIFF = 8


def inside_share(s, m):
    h, w = m.shape
    L = max(2, int(np.hypot(s[2] - s[0], s[3] - s[1])))
    t = np.linspace(0, 1, L)
    x = np.clip(np.round(s[0] + t * (s[2] - s[0])).astype(int), 0, w - 1)
    y = np.clip(np.round(s[1] + t * (s[3] - s[1])).astype(int), 0, h - 1)
    return float(m[y, x].mean())


def unmatched(a, b):
    return int(sum(not any(sl._same_line(s, t, 2.0, 2.0) or sl._same_line(t, s, 2.0, 2.0) for t in b) for s in a))


def main() -> int:
    out, name = Path(sys.argv[1]), sys.argv[2]
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        job = Path(tmp) / ROOMS[name].name
        shutil.copytree(ROOMS[name], job)
        b = reuse.load(job)
        fl, wl = pe.load(job / "segments")
    h, w = b.room.shape[:2]
    floor, wall, _ = masks.prepare(fl, wl, (h, w))
    props = np.asarray(b.props, bool)
    diff = np.abs(b.clean.astype(int) - b.room.astype(int)).max(axis=2) > INPAINT_DIFF
    inpainted = cv2.morphologyEx(diff.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
    rep = {"room": name, "inpainted_px": int(inpainted.sum()), "inpainted_share": round(float(inpainted.mean()), 4),
           "inpainted_inside_objects_share": round(float(props[inpainted].mean()), 3) if inpainted.any() else None}
    runs = {}
    for src, img in (("original", b.room), ("clean_room", b.clean)):
        res = sl.strong_lines(img, props=props, floor=floor, labels=surfaces.label_map(img))
        fam, _ = families(res["lines"], w, h)
        ins = np.array([inside_share(s, inpainted) for s in res["lines"]]) if len(res["lines"]) else np.zeros(0)
        runs[src] = (res, fam)
        rep[src] = {"counts": res["report"],
                    "families": {k: int((fam == k).sum()) for k in ("vertical", "horizontal-A", "horizontal-B", "unassigned")},
                    "kept_lines_mostly_inside_inpainted": int((ins > 0.5).sum()),
                    "kept_lines_partly_inside_inpainted_(>10%)": int((ins > 0.1).sum())}
    diffs = {}
    for k in ("vertical", "horizontal-A", "horizontal-B"):
        a = runs["original"][0]["lines"][runs["original"][1] == k]
        c = runs["clean_room"][0]["lines"][runs["clean_room"][1] == k]
        diffs[k] = {"only_in_original": unmatched(a, c), "only_in_clean_room": unmatched(c, a)}
    rep["family_differences"] = diffs
    (out / f"{name}__l1_source.json").write_text(json.dumps(rep, indent=1))
    print(name, "done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
