#!/usr/bin/env python3
# Run a pipeline stage script on this Mac without editing it.
#
# The stage scripts hardcode BASE = Path("/workspace/axolotl"), which cannot
# exist on macOS (read-only root volume). This maps /workspace -> ~/workspace,
# where ~/workspace/axolotl is a symlink back to this project.
#
#   python3 run_local.py scripts/test07_prod_stage08e1_floor_line_geometry_v2.py

import os
import sys
from pathlib import Path

REMOTE_ROOT = "/workspace"
LOCAL_ROOT = str(Path.home() / "workspace")

if len(sys.argv) < 2:
    sys.exit("usage: run_local.py <stage_script.py>")
  
stage = Path(sys.argv[1]).resolve()

if not stage.exists():
    sys.exit(f"no such stage script: {stage}")

os.environ.setdefault("MPLBACKEND", "Agg")

source = stage.read_text().replace(REMOTE_ROOT, LOCAL_ROOT)

sys.argv = sys.argv[1:]

exec(
    compile(source, str(stage), "exec"),
    {"__name__": "__main__", "__file__": str(stage)},
)