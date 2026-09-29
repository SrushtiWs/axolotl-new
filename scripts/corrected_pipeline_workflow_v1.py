#!/usr/bin/env python3
"""
Corrected workflow:
1. Original upload
2. AI empty room (preserve surfaces, remove objects only)
3. Object identification / extraction
4. Object data storage (masks + original positions)
5. Empty room (surfaces intact)
6. Tiles placed
7. Objects regain original positions
"""
STAGES = [
    ("01", "original_master_input", "runtime_assets/00_master_input.png"),
    ("02", "ai_empty_room_preserve_surfaces", "scripts/test07_prod_stage07a_qwen_empty_room_v1.py"),
    ("03", "object_identification", "scripts/test07_prod_stage06c13a_florence_wall_switch_sam2_v1.py"),
    ("04", "object_extraction", "backend/extraction/sam2.py"),
    ("05", "object_data_storage", "runtime_assets/01_final_six_prop_union_mask.png"),
    ("06", "empty_room_confirmed", "runtime_assets/00_stage07_empty_room_candidate.png"),
    ("07", "tiles_placed", "backend/engine.py"),
    ("08", "objects_regain_position", "backend/engine.py::_restore_props"),
]
print("Corrected workflow stages:")
for s in STAGES:
    print(f"  {s[0]} | {s[1]} -> {s[2]}")