"""Write current and controlled-material tile compositing audit layers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

import live_scene  # noqa: E402
import reuse  # noqa: E402
from engine import TileSpec, render  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("job", help="render job id, for example 50575948a7c5")
    args = parser.parse_args()

    job_dir = PROJECT_ROOT / "backend" / "jobs" / args.job
    state = json.loads((job_dir / "state.json").read_text(encoding="utf-8"))
    source_job = state["timings"].get("source_job") or args.job
    bundle = reuse.load(PROJECT_ROOT / "backend" / "jobs" / source_job)

    if bundle is None:
        raise RuntimeError(f"No reusable scene bundle found for {source_job}")

    submitted = state["submitted"]
    spec = TileSpec(
        artwork=_image(job_dir / "tile.png"),
        width_mm=float(submitted["tile_mm"][0]),
        height_mm=float(submitted["tile_mm"][1]),
        rotation_deg=float(submitted["rotation"]),
        grout_mm=float(submitted.get("grout_mm", 5.0)),
    )

    walls = tuple(submitted.get("walls") or []) or None
    scene, _ = live_scene.build(
        bundle.room,
        [],
        bundle.surfaces,
        *submitted["room_ft"],
        wall_labels=walls,
        clean=bundle.clean,
        props=bundle.props,
        object_count=int(state["geometry"].get("object_count") or 0),
    )

    current = render(scene, spec, submitted["surface"], controlled_material=False)
    controlled = render(
        scene, spec, submitted["surface"], controlled_material=True
    )

    audit_dir = job_dir / "tile_audit"
    audit_dir.mkdir(exist_ok=True)

    _save(audit_dir / "A_projected_raw_tile.png", current.raw)
    _save(audit_dir / "B_current_lit_before_composite.png", current.lit)
    _save(audit_dir / "C_current_final_composite.png", current.composite)
    _save(audit_dir / "D_underlying_clean_room_wall_source.png", _masked(current.underlying, current.target))
    _save(audit_dir / "E_final_wall_composite_mask.png", current.target.astype("uint8") * 255)
    _save(audit_dir / "B_controlled_lit_before_composite.png", controlled.lit)
    _save(audit_dir / "C_controlled_final_composite.png", controlled.composite)

    report = {
        "job": args.job,
        "source_job": source_job,
        "current": current.stats,
        "controlled": controlled.stats,
        "target_pixels": int(current.target.sum()),
    }
    (audit_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"audit_dir={audit_dir}")


def _image(path: Path):
    return __import__("numpy").array(Image.open(path).convert("RGB"))


def _masked(image, mask):
    result = image.copy()
    result[~mask] = 0
    return result


def _save(path: Path, image) -> None:
    Image.fromarray(image.clip(0, 255).astype("uint8")).save(path)


if __name__ == "__main__":
    main()