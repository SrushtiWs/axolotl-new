"""
The backend imports the way the server starts it: `uvicorn app:app --app-dir
backend` (./dev.sh), with only backend/ on the path -- not the project root the
other tests add first. Catches an import that only works inside the tests.

    backend/.venv/bin/python tests/unit/test_server_import.py
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_app_imports_like_uvicorn():
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    code = ("import sys; sys.path[:] = [p for p in sys.path if p not in ('', %r)]; "
            "sys.path.insert(0, %r); import app; print('ok')" % (str(ROOT), str(ROOT / "backend")))
    run = subprocess.run([sys.executable, "-c", code], cwd="/", env=env, capture_output=True, text=True, timeout=300)
    assert run.returncode == 0 and run.stdout.strip().endswith("ok"), run.stderr[-800:]


if __name__ == "__main__":
    try:
        test_app_imports_like_uvicorn(); print("PASS  test_app_imports_like_uvicorn\n\n1/1 passed")
    except AssertionError as err:
        print(f"FAIL  test_app_imports_like_uvicorn\n{err}\n\n0/1 passed"); sys.exit(1)
