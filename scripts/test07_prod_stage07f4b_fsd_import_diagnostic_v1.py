
from pathlib import Path
import sys
import json
import traceback
import importlib.util


BASE = Path(
    "/workspace/axolotl"
)

FSD_DIR = (
    BASE
    / "test07"
    / "models"
    / "FSD"
)

OUT = (
    BASE
    / "test07"
    / "production_pipeline"
    / "stage07_empty_room"
    / "07f4b_fsd_import_diagnostic"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


print("=" * 110)
print("07F4B — FSD IMPORT FAILURE DIAGNOSTIC")
print("=" * 110)

print()

print(
    "FSD DIR:",
    FSD_DIR
)

print(
    "Exists:",
    FSD_DIR.exists()
)


# ============================================================
# ADD REPOSITORY TO PYTHON PATH
# ============================================================

fsd_string = str(
    FSD_DIR
)

if fsd_string not in sys.path:
    sys.path.insert(
        0,
        fsd_string
    )


print()
print(
    "sys.path[0]:",
    sys.path[0]
)


# ============================================================
# BASIC PACKAGE PRESENCE
# ============================================================

packages = [
    "torch",
    "torchvision",
    "timm",
    "einops",
    "efficientnet_pytorch",
    "numpy",
    "cv2",
]


print()
print("=" * 110)
print("DEPENDENCY PRESENCE")
print("=" * 110)

package_status = {}


for name in packages:

    spec = importlib.util.find_spec(
        name
    )

    present = (
        spec is not None
    )

    package_status[
        name
    ] = present

    print(
        f"{name:24s}",
        "✅"
        if present
        else "❌"
    )


# ============================================================
# CHECK FSD SOURCE FILES
# ============================================================

source_files = [
    FSD_DIR / "networks" / "__init__.py",
    FSD_DIR / "networks" / "fdrnet.py",
]


print()
print("=" * 110)
print("SOURCE FILE STATUS")
print("=" * 110)


for path in source_files:

    print(
        str(path),
        "✅"
        if path.exists()
        else "❌"
    )


# ============================================================
# ATTEMPT IMPORT AND CAPTURE FULL TRACEBACK
# ============================================================

import_ok = False

error_type = None
error_message = None
traceback_text = None


print()
print("=" * 110)
print("FDRNET IMPORT ATTEMPT")
print("=" * 110)

print()


try:

    from networks.fdrnet import FDRNet

    import_ok = True

    print(
        "✅ IMPORT SUCCESS"
    )

    print(
        "FDRNet:",
        FDRNet
    )


except Exception as e:

    error_type = type(
        e
    ).__name__

    error_message = str(
        e
    )

    traceback_text = traceback.format_exc()

    print(
        "❌ IMPORT FAILED"
    )

    print()
    print(
        "ERROR TYPE:",
        error_type
    )

    print(
        "ERROR MESSAGE:",
        error_message
    )

    print()
    print("=" * 110)
    print("FULL TRACEBACK")
    print("=" * 110)

    print(
        traceback_text
    )


# ============================================================
# SAVE DIAGNOSTIC
# ============================================================

STATE = {

    "stage":
        "07F4B",

    "purpose":
        "FSD_IMPORT_FAILURE_DIAGNOSTIC",

    "fsd_directory":
        str(
            FSD_DIR
        ),

    "dependency_presence":
        package_status,

    "import_ok":
        import_ok,

    "error_type":
        error_type,

    "error_message":
        error_message,

    "traceback":
        traceback_text,

    "status":
        (
            "PASS"
            if import_ok
            else "DIAGNOSED_IMPORT_FAILURE"
        )
}


STATE_PATH = (
    OUT
    / "00_stage07f4b_result.json"
)


STATE_PATH.write_text(
    json.dumps(
        STATE,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


print()
print("=" * 110)
print("07F4B RESULT")
print("=" * 110)

print()

print(
    "FDRNet import:",
    "✅ PASS"
    if import_ok
    else "❌ FAIL"
)

if not import_ok:

    print(
        "ERROR TYPE:",
        error_type
    )

    print(
        "ERROR MESSAGE:",
        error_message
    )

print()

print(
    "STATE:",
    STATE_PATH
)
