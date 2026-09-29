#!/usr/bin/env python3
"""Run the room-retiling pipeline end to end with one command.

    python3 run_pipeline.py --list              # show the full plan, run nothing
    python3 run_pipeline.py --dry-run           # same, but says what WOULD run
    python3 run_pipeline.py                     # run everything this machine can
    python3 run_pipeline.py --phase 08          # run one stage group
    python3 run_pipeline.py --from 06d1         # resume from a stage id
    python3 run_pipeline.py --only h1           # run a single stage
    python3 run_pipeline.py --lineage test08    # the locked-parameter rerun

Stage scripts are discovered from scripts/, ordered by their stage id
(01 -> 02 -> ... -> 08h2), and launched through run_local.py, which maps
the hardcoded /workspace root onto ~/workspace without editing them.

Each script is classified automatically:
  cpu   runs anywhere
  gpu   imports torch/transformers/diffusers/sam2 -> needs an NVIDIA box
  note  a comment-only execution record from the notebook, nothing to run
GPU stages are skipped unless CUDA is present (or --include-gpu is passed);
their outputs are already committed in the tree.
"""

import argparse
import ast
import re
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
RUNNER = PROJECT / "run_local.py"
SCRIPTS = PROJECT / "scripts"
WORKSPACE = Path.home() / "workspace"

GPU_IMPORTS = {"torch", "transformers", "diffusers", "sam2", "qwen_vl_utils"}

FINAL_IMAGE = (
    PROJECT
    / "test07/production_pipeline/stage08_tile_application"
    / "08h2_props_over_metric_floor"
    / "02_metric_floor_plus_physical_props.png"
)

PHASE_NAMES = {
    "01": "master input",
    "02": "room structure",
    "03": "mirror layer",
    "04": "glass layer",
    "05": "clean room with props",
    "06": "prop layer",
    "07": "empty room + shadows",
    "08": "tile application",
}


class Stage:
    def __init__(self, path, lineage, stage_id, kind):
        self.path = path
        self.lineage = lineage
        self.stage_id = stage_id
        self.kind = kind
        self.phase = stage_id[:2]

    @property
    def description(self):
        name = self.path.stem
        name = re.sub(r"^test0[78]_prod_stage\w+?_", "", name)
        name = re.sub(r"_v\d+$", "", name)
        return name.replace("_", " ")


def sort_key(stage_id):
    """Order stage ids naturally: 06c1 < 06c1b < 06c2 < 06c10 < 06d1."""
    chunks = re.findall(r"\d+|[a-z]+", stage_id)
    return [(0, int(c), "") if c.isdigit() else (1, 0, c) for c in chunks]


def classify(path):
    source = path.read_text()

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return "cpu"

    if not tree.body:
        return "note"

    imports = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])

    return "gpu" if imports & GPU_IMPORTS else "cpu"


def discover(lineage):
    """Find the stage scripts, newest version of each, in pipeline order."""
    pattern = re.compile(rf"^{lineage}_prod_stage([0-9a-z]+)_(.+?)(?:_v(\d+))?$")

    best = {}

    for path in SCRIPTS.glob(f"{lineage}_prod_stage*.py"):
        match = pattern.match(path.stem)

        if not match:
            continue

        stage_id, name, version = match.groups()
        key = (stage_id, name)
        version = int(version or 0)

        if key not in best or version > best[key][0]:
            best[key] = (version, path)

    stages = [
        Stage(path, lineage, stage_id, classify(path))
        for (stage_id, _name), (_version, path) in best.items()
    ]

    stages.sort(key=lambda s: (sort_key(s.stage_id), s.path.stem))
    return stages


def has_cuda():
    try:
        import torch
    except ImportError:
        return False

    return torch.cuda.is_available()


def ensure_workspace():
    (WORKSPACE / "data" / "huggingface-cache").mkdir(parents=True, exist_ok=True)

    link = WORKSPACE / "axolotl"

    if not link.exists():
        link.symlink_to(PROJECT)


def check_deps():
    missing = [m for m in ("cv2", "numpy", "PIL", "matplotlib")
               if not _importable(m)]

    if missing:
        sys.exit(
            "missing packages: " + ", ".join(missing) + "\n"
            "install with: pip3 install opencv-python numpy pillow matplotlib"
        )


def _importable(name):
    try:
        __import__(name)
        return True
    except ImportError:
        return False


def interesting(output):
    skip_prefixes = (
        "AUDIT:", "STATE:", "GRID:", "RAW", "LIT/GLOSSY:", "COMPOSITE:",
        "PROP RGBA:", "MAIN AUDIT:", "POLYGON AUDIT:", "SCREEDING MASK:",
        "COMBINED TARGET:", "FLOOR + SCREEDING:", "PER PROP:",
    )

    keep = []

    for line in output.splitlines():
        line = line.rstrip()

        if not line or set(line) <= {"=", "-"}:
            continue
        if "FigureCanvasAgg" in line or "plt.show()" in line:
            continue
        if line.startswith(skip_prefixes):
            continue

        keep.append(line)

    return keep[-5:]


def select(stages, args, gpu_ok):
    chosen = []

    for stage in stages:
        if stage.kind == "note":
            continue
        if stage.kind == "gpu" and not gpu_ok:
            continue
        if args.phase and stage.phase != args.phase.zfill(2):
            continue

        chosen.append(stage)

    if args.only:
        chosen = [s for s in chosen if args.only in s.stage_id]
    elif args.start:
        matches = [i for i, s in enumerate(chosen) if s.stage_id.startswith(args.start)]

        if not matches:
            sys.exit(f"no stage id starting with {args.start!r}")

        chosen = chosen[matches[0]:]

    return chosen


def print_plan(stages, chosen, gpu_ok):
    chosen_paths = {s.path for s in chosen}
    phase = None

    for stage in stages:
        if stage.phase != phase:
            phase = stage.phase
            print(f"\n  stage {phase} — {PHASE_NAMES.get(phase, '')}")

        if stage.kind == "note":
            mark = "note  (notebook record, nothing to run)"
        elif stage.path in chosen_paths:
            mark = "RUN"
        elif stage.kind == "gpu" and not gpu_ok:
            mark = "skip  (needs GPU; output already on disk)"
        else:
            mark = "skip"

        print(f"    {stage.stage_id:<8} {stage.description[:44]:<46} {mark}")


def run_stage(stage, index, total):
    print(f"\n[{index}/{total}] {stage.stage_id:<8} {stage.description}")

    start = time.time()

    result = subprocess.run(
        [sys.executable, str(RUNNER), str(stage.path)],
        capture_output=True,
        text=True,
    )

    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"          FAIL   ({elapsed:.1f}s)")

        for line in (result.stderr or result.stdout).strip().splitlines()[-8:]:
            print(f"          | {line}")

        return False

    print(f"          OK     ({elapsed:.1f}s)")

    for line in interesting(result.stdout):
        print(f"          | {line}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--list", action="store_true", help="show the plan and exit")
    parser.add_argument("--dry-run", action="store_true", help="show what would run, run nothing")
    parser.add_argument("--lineage", default="test07", choices=["test07", "test08"])
    parser.add_argument("--phase", metavar="NN", help="run one stage group, e.g. 08")
    parser.add_argument("--from", dest="start", metavar="ID", help="resume from this stage id")
    parser.add_argument("--only", metavar="ID", help="run stages matching this id")
    parser.add_argument("--include-gpu", action="store_true", help="run GPU stages even without CUDA")
    parser.add_argument("--keep-going", action="store_true", help="continue after a failure")
    args = parser.parse_args()

    stages = discover(args.lineage)

    if not stages:
        sys.exit(f"no {args.lineage} stage scripts found in {SCRIPTS}")

    gpu_ok = args.include_gpu or has_cuda()
    chosen = select(stages, args, gpu_ok)

    counts = {
        "cpu": sum(1 for s in stages if s.kind == "cpu"),
        "gpu": sum(1 for s in stages if s.kind == "gpu"),
        "note": sum(1 for s in stages if s.kind == "note"),
    }

    print("=" * 74)
    print(f"{args.lineage.upper()} PIPELINE — {len(stages)} stage scripts")
    print("=" * 74)
    print(f"project   : {PROJECT}")
    print(f"cuda      : {'yes' if gpu_ok else 'no — GPU stages will be skipped'}")
    print(f"scripts   : {counts['cpu']} cpu, {counts['gpu']} gpu, {counts['note']} notes")
    print(f"selected  : {len(chosen)}")

    if args.list or args.dry_run:
        print_plan(stages, chosen, gpu_ok)
        print(f"\n{len(chosen)} stage(s) would run. Drop --list/--dry-run to execute.")
        return 0

    if not chosen:
        sys.exit("no stages selected")

    check_deps()
    ensure_workspace()

    started = time.time()
    ok = 0
    failed = []

    for index, stage in enumerate(chosen, start=1):
        if run_stage(stage, index, len(chosen)):
            ok += 1
        else:
            failed.append(stage.stage_id)

            if not args.keep_going:
                print(f"\nstopped at {stage.stage_id} — later stages read its output")
                print("pass --keep-going to continue anyway")
                break

    print("\n" + "=" * 74)
    print(f"DONE — {ok}/{len(chosen)} stages ok in {time.time() - started:.1f}s")

    if failed:
        print(f"FAILED: {', '.join(failed)}")

    if FINAL_IMAGE.exists():
        print(f"\nfinal composite:\n  open '{FINAL_IMAGE}'")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
