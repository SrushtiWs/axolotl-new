"""
Tile count and edge cuts (tiles_backend/perspective_engine/room/tile_count.py).

    backend/.venv/bin/python tests/unit/test_tile_count.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tiles_backend.perspective_engine.room.geometry import feet_to_mm  # noqa: E402
from tiles_backend.perspective_engine.room.tile_count import room_layout, surface_layout  # noqa: E402

U = "USER_INPUT"


def test_spec_example_12ft_600mm():
    # 12 ft = 3657.6 mm; 3657.6 / 600 = 6 full + 57.6 mm cut, nothing rounded first.
    s = surface_layout({"x": (feet_to_mm(12), U), "z": (3000.0, U)}, 600, 600, 0)
    assert s["span_x_mm"] == 3657.6
    assert (s["full_x"], s["edge_cut_x_mm"], s["tile_count_x"]) == (6, 57.6, 7)
    assert (s["full_z"], s["edge_cut_z_mm"], s["tile_count_z"]) == (5, 0.0, 5)
    assert (s["total_tiles"], s["full_tiles"], s["partial_tiles"]) == (35, 30, 5)
    assert s["status"] == "USER_INPUT"


def test_exact_fit_has_no_cut():
    s = surface_layout({"x": (3600.0, U), "z": (1800.0, U)}, 600, 1200, 0)
    assert (s["tile_count_x"], s["edge_cut_x_mm"]) == (6, 0.0)
    assert (s["full_z"], s["edge_cut_z_mm"], s["tile_count_z"]) == (1, 600.0, 2)
    assert s["partial_tiles"] == 6


def test_tile_never_stretched_and_90_swaps():
    # 4000 x 5000 mm room, 1200 x 1800 mm tile (the architecture note's example).
    s0 = surface_layout({"x": (4000.0, U), "z": (5000.0, U)}, 1200, 1800, 0)
    assert s0["footprint_mm"] == [1200, 1800]
    assert (s0["full_x"], s0["edge_cut_x_mm"], s0["tile_count_x"]) == (3, 400.0, 4)
    assert (s0["full_z"], s0["edge_cut_z_mm"], s0["tile_count_z"]) == (2, 1400.0, 3)
    s90 = surface_layout({"x": (4000.0, U), "z": (5000.0, U)}, 1200, 1800, 90)
    assert s90["footprint_mm"] == [1800, 1200]
    assert (s90["full_x"], s90["edge_cut_x_mm"]) == (2, 400.0)
    assert (s90["full_z"], s90["edge_cut_z_mm"]) == (4, 200.0)


def test_diagonal_not_reported():
    for rot in (45, 135):
        s = surface_layout({"x": (4000.0, U), "z": (5000.0, U)}, 600, 600, rot)
        assert s["status"] == "UNAVAILABLE" and "tile_count_x" not in s


def test_missing_size_is_unavailable_and_estimated_is_labelled():
    s = surface_layout({"x": (None, "UNKNOWN"), "z": (5000.0, "ESTIMATED")}, 600, 600, 0)
    assert s["status"] == "UNAVAILABLE" and s["tile_count_x"] is None and s["tile_count_z"] == 9
    e = surface_layout({"x": (4000.0, U), "z": (5000.0, "ESTIMATED")}, 600, 600, 0)
    assert e["status"] == "ESTIMATED"


def test_room_layout_walls_follow_their_axis():
    room = {"status": "OK", "confidence": 0.4,
            "width_mm": 4000.0, "length_mm": 5000.0, "height_mm": 2800.0,
            "width_source": U, "length_source": U, "height_source": U,
            "walls": {"wall-0": {"class": "side"}, "wall-1": {"class": "back"},
                      "wall-2": {"class": "oblique"}}}
    out = room_layout(room, 1200, 1800, 0)["surfaces"]
    assert out["floor"]["tile_count_x"] == 4 and out["floor"]["tile_count_z"] == 3
    assert out["wall-0"]["axes"] == ["z", "y"] and out["wall-0"]["span_z_mm"] == 5000.0
    assert out["wall-1"]["axes"] == ["x", "y"] and out["wall-1"]["span_x_mm"] == 4000.0
    assert (out["wall-1"]["full_y"], out["wall-1"]["edge_cut_y_mm"]) == (1, 1000.0)
    assert out["wall-2"]["status"] == "UNAVAILABLE"


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
