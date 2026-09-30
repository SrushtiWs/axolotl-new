# Regression harness

Every tile render of three frozen rooms: floor + each wall, at rotations 0/45/90/135.

```bash
backend/.venv/bin/python tests/regression/harness.py                    # compare to the baseline
backend/.venv/bin/python tests/regression/harness.py --update-baseline  # accept the current output
backend/.venv/bin/python tests/regression/selftest.py                   # prove each check can fail
backend/.venv/bin/python tests/unit/test_tile_count.py                  # tile count + edge cuts
backend/.venv/bin/python tests/regression/manual_sizes.py             # manual W/L/H: scale, 12% conflict, status line
backend/.venv/bin/python tests/regression/auto_size.py                # nothing typed: prior scale, lower bounds, UI line
backend/.venv/bin/python tests/regression/seam_scale.py               # floor/wall seam gap + tile-scale ratio per wall
backend/.venv/bin/python tests/unit/test_box_ceiling.py                 # room height from box walls only
backend/.venv/bin/python tests/regression/three_parity.py             # 3D (real Three.js renderer) vs 2D, pixel by pixel
backend/.venv/bin/python tests/regression/tile_change.py              # a tile change re-runs no detection / camera
backend/.venv/bin/python tests/unit/test_server_import.py              # the app imports as uvicorn starts it
backend/.venv/bin/python tests/regression/wall_split.py               # wall split re-run per room: partition + seen corners
backend/.venv/bin/python tests/regression/wall_direction.py           # wall directions: evidence, cross-check, rendered lines vs each wall's VP
backend/.venv/bin/python tests/regression/weak_texture.py             # depth VP judged on long structural lines: keep / replace / reject
backend/.venv/bin/python tests/validation/validate_rooms.py           # per-room estimates vs measured sizes -> tests/validation/report.md
node tests/threejs/full_flow.mjs <app url> <api port> <shots dir> <photo>...  # full user flow in Chrome
```

Options: `--rooms empty living`, `--rotations 0 90`, `--manifest PATH` (compare to or save another baseline, with its images beside it), and `--set module.ATTR=VALUE` (test-only: override a constant for one run, e.g. `--set tiles_backend.perspective_engine.camera.vertical_vp.HIGH_CONFIDENCE=2` forces camera roll off). Parity also compares the `result.png` file bytes whenever the baseline image is on disk. The exit status is 0 only when every case passes. The full run takes about 100 s and needs no running server. It renders through the real `POST /generate` in-process, into `tests/.work/` (`backend/jobs/` is never touched).

## Per case

| Check | Passes when |
|---|---|
| Pixel parity | `result.png` and the tile coverage (`three/<surface>.png`) are identical to the baseline, pixel for pixel. A failed case reports how many pixels differ and the largest channel difference. |
| Mask clip | 0 tile pixels outside `FLOOR_MASK.png` / `WALL_MASK.png`. 0 pixels changed outside the tiles: every other pixel must equal `original·a + clean·(1−a)`, where `a` is the render's own object alpha (`three/objects.png`), within 2 levels of 8-bit rounding. The engine's own `outside_mask_after_clip` is 0. |
| three.json | The JSON round-trips unchanged. The quad corners lie on the plane. `plane -> corners -> rotate + offset -> x mm_per_unit` reproduces `quad_grid_mm`. Every tiled pixel's camera ray meets the plane inside the quad. `meters_per_unit = mm_per_unit / 1000`. |

A case that returns an HTTP error is recorded with its message. Parity then means the same status and the same message.

## Files

| Path | What |
|---|---|
| `fixtures/rooms/<name>/` | A frozen Clean Room job (`original.png`, `reuse/`, `segments/`), plus `photo.*` (the exact uploaded bytes; the job is matched to the photo by SHA-256) and `fixture.json` |
| `fixtures/tile.png` | The tile artwork. Settings: 1200 x 1800 mm, 5 mm grout, room size AUTO |
| `baseline/manifest.json` | The per-case status, pixel hashes, tile pixel count and room status |
| `baseline/images/` | The baseline `result.png` per case, for diffing (git-ignored because of size; the hashes alone still decide pass/fail) |
| `regression/make_fixtures.py` | Re-freezes rooms from a running backend: `make_fixtures.py name=path/to/photo.png` |

Re-freezing a room reruns Clean Room (the ML models), so its masks can change. Update the baseline afterwards, and say why.

`three_parity.py` needs Node, Google Chrome, and `npm install` once in `tests/threejs/` (playwright-core).

Measured room sizes live only in `tests/validation/measured_rooms.json` (test data; production code never reads it). Leave a value null unless it was measured in the real room.
