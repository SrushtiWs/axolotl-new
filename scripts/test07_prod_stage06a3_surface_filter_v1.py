
from pathlib import Path
import importlib.util
import json

INPUT_JSON = Path('/workspace/axolotl/test07/production_pipeline/stage06_prop_layer/06a2_pairwise_dedup/02_pairwise_verified_inventory.json')
OUT = Path('/workspace/axolotl/test07/production_pipeline/stage06_prop_layer/06a3_surface_filter')
LEGACY_SCRIPT = Path('/workspace/axolotl/scripts/test07_stage02_v34_surface_filter_only.py')

OUT.mkdir(
    parents=True,
    exist_ok=True
)

print("=" * 100)
print("PRODUCTION STAGE 06A3")
print("CONSERVATIVE STRUCTURAL SURFACE FILTER")
print("=" * 100)

required = {
    "PAIRWISE INVENTORY": INPUT_JSON,
    "VERIFIED V3.4 SCRIPT": LEGACY_SCRIPT,
}

for name, path in required.items():

    print(
        "✅" if path.exists() else "❌",
        name,
        path
    )

    if not path.exists():
        raise FileNotFoundError(path)


spec = importlib.util.spec_from_file_location(
    "test07_v34_surface_filter",
    LEGACY_SCRIPT
)

module = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    module
)

print()
print(
    "✅ VERIFIED V3.4 SURFACE FILTER LOADED"
)

module.run(
    input_json=INPUT_JSON,
    output_dir=OUT,
)


RESULT_PATH = (
    OUT
    / "00_conservative_inventory.json"
)

REMOVED_PATH = (
    OUT
    / "01_removed_surfaces.json"
)

if not RESULT_PATH.exists():
    raise RuntimeError(
        "Filtered inventory was not produced."
    )


result = json.loads(
    RESULT_PATH.read_text(
        encoding="utf-8"
    )
)

removed = json.loads(
    REMOVED_PATH.read_text(
        encoding="utf-8"
    )
)


print()
print("=" * 100)
print("PRODUCTION 06A3 WRAPPER COMPLETE")
print("=" * 100)

print(
    "INPUT OBJECTS:",
    result.get(
        "input_objects"
    )
)

print(
    "REMOVED STRUCTURAL SURFACES:",
    result.get(
        "removed_structural_surfaces"
    )
)

print(
    "FINAL CLEAN INVENTORY:",
    result.get(
        "final_inventory"
    )
)

print()
print("REMOVED:")

for row in removed:

    print(
        "-",
        row.get("name", ""),
        "|",
        row.get("grounding_phrase", "")
    )

print()
print("CLEAN OBJECT LIST:")

for row in result.get(
    "objects",
    []
):

    print(
        f'045. '
        f'toilet | '
        f'high | '
        f'white ceramic toilet | '
        f'sources=['bottom_right']'
    )
