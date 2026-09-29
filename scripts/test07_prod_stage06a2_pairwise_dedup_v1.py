
from pathlib import Path
import importlib.util
import json

MASTER_PATH = Path('/workspace/axolotl/test07/production_pipeline/stage05_clean_room_with_props/07_clean_room_with_exact_props.png')
RAW_CANDIDATES_PATH = Path('/workspace/axolotl/test07/production_pipeline/stage06_prop_layer/06a_inventory/01_raw_tiled_candidates.json')
OUT = Path('/workspace/axolotl/test07/production_pipeline/stage06_prop_layer/06a2_pairwise_dedup')
LEGACY_SCRIPT = Path('/workspace/axolotl/scripts/test07_stage02_v32_pairwise_image_dedup.py')
MODEL_CACHE = Path('/workspace/data/huggingface-cache')

OUT.mkdir(
    parents=True,
    exist_ok=True
)

print("=" * 100)
print("PRODUCTION STAGE 06A2")
print("PAIRWISE IMAGE-GROUNDED DEDUP")
print("=" * 100)

required = {
    "STAGE05 DISCOVERY IMAGE": MASTER_PATH,
    "RAW 06A CANDIDATES": RAW_CANDIDATES_PATH,
    "VERIFIED V3.2 SCRIPT": LEGACY_SCRIPT,
}

for name, path in required.items():

    print(
        "✅" if path.exists() else "❌",
        name,
        path
    )

    if not path.exists():
        raise FileNotFoundError(path)


# ------------------------------------------------------------
# Verify we are using RAW candidates, not bad consolidation.
# ------------------------------------------------------------

raw_candidates = json.loads(
    RAW_CANDIDATES_PATH.read_text(
        encoding="utf-8"
    )
)

if not isinstance(
    raw_candidates,
    list
):
    raise RuntimeError(
        "Raw candidate JSON must be a list."
    )

print()
print(
    "RAW INPUT CANDIDATES:",
    len(raw_candidates)
)

print(
    "EXPECTED FROM STAGE06A:",
    44
)

if len(raw_candidates) != 44:

    print(
        "⚠️ Candidate count differs from previous run; "
        "continuing with actual saved raw candidates."
    )


# ------------------------------------------------------------
# Dynamically load frozen verified implementation.
# ------------------------------------------------------------

spec = importlib.util.spec_from_file_location(
    "test07_v32_pairwise",
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
    "✅ VERIFIED V3.2 IMPLEMENTATION LOADED"
)

print(
    "Calling run(...) using production paths"
)


module.run(
    master_path=MASTER_PATH,
    candidate_json=RAW_CANDIDATES_PATH,
    output_dir=OUT,
    model_cache=MODEL_CACHE,
)


# ------------------------------------------------------------
# Validate expected output.
# ------------------------------------------------------------

RESULT_PATH = (
    OUT
    / "02_pairwise_verified_inventory.json"
)

if not RESULT_PATH.exists():

    raise RuntimeError(
        "Pairwise inventory was not produced."
    )


result = json.loads(
    RESULT_PATH.read_text(
        encoding="utf-8"
    )
)


print()
print("=" * 100)
print("PRODUCTION 06A2 WRAPPER COMPLETE")
print("=" * 100)

print(
    "INPUT:",
    result.get(
        "input_candidate_count"
    )
)

print(
    "PLAUSIBLE PAIRS:",
    result.get(
        "plausible_pair_count"
    )
)

print(
    "FINAL AFTER PAIRWISE DEDUP:",
    result.get(
        "final_inventory_count"
    )
)

print(
    "OUTPUT:",
    RESULT_PATH
)
