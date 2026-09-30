"""
9c: room height comes from the room box's walls only (room/geometry._box_ceiling).

    backend/.venv/bin/python tests/unit/test_box_ceiling.py
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tiles_backend.perspective_engine.room.geometry import _box_ceiling  # noqa: E402

BOX = {"left_wall": "wall-3", "right_wall": "wall-5", "back_wall": "wall-4"}


def _all_walls_median(walls):
    """The rule before 9c: every wall with a top votes, weighted by its columns."""
    tops = [(e["top_height"], e["top_columns"]) for e in walls.values() if e.get("top_height")]
    hs = np.array([t[0] for t in tops]); ws = np.array([t[1] for t in tops], float)
    order = np.argsort(hs); cum = np.cumsum(ws[order])
    return float(hs[order][np.searchsorted(cum, cum[-1] / 2)])


def test_pillar_and_partial_wall_cannot_set_the_height():
    walls = {
        "wall-3": {"top_height": 2.36, "top_columns": 36},
        "wall-4": {"top_height": 2.32, "top_columns": 709},
        "wall-5": {"top_height": 2.31, "top_columns": 101},
        # A wide half-height partition and a pillar top: more columns than the box.
        "wall-6": {"top_height": 1.10, "top_columns": 1500},
        "wall-7": {"top_height": 1.90, "top_columns": 400},
    }
    assert _all_walls_median(walls) == 1.10               # the old rule: the partition wins
    h, used, excluded = _box_ceiling({"walls": walls}, BOX)
    assert h == 2.32 and used == ["wall-3", "wall-4", "wall-5"] and excluded == ["wall-6", "wall-7"]


def test_no_box_wall_top_means_unknown_not_a_pillar():
    walls = {"wall-6": {"top_height": 1.10, "top_columns": 1500}}
    h, used, excluded = _box_ceiling({"walls": walls}, BOX)
    assert h is None and used == [] and excluded == ["wall-6"]


def test_fixtures_unchanged():
    for name in ("empty", "living", "bedroom"):
        fr = json.loads((ROOT / f"tests/fixtures/rooms/{name}/segments/room/room_frame.json").read_text())
        h, used, excluded = _box_ceiling(fr, fr["box_units"])
        assert abs(h - fr["box_units"]["ceiling_y"]) < 1e-12, name
        assert excluded == [], name


if __name__ == "__main__":
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn(); print(f"PASS  {name}")
        except AssertionError as err:
            failed += 1; print(f"FAIL  {name}  {err!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
