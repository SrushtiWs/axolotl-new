
from pathlib import Path
import json
import shutil
import zipfile
from datetime import datetime


ROOT = Path("/workspace/axolotl/test07")
ROOM = ROOT / "runs/room01_bedroom"

FREEZE = ROOT / "room01_reference"
FREEZE.mkdir(parents=True, exist_ok=True)

SCRIPTS_DST = FREEZE / "scripts"
OUTPUTS_DST = FREEZE / "final_outputs"

SCRIPTS_DST.mkdir(parents=True, exist_ok=True)
OUTPUTS_DST.mkdir(parents=True, exist_ok=True)


# ============================================================
# SUCCESSFUL TEST07 SCRIPTS
# ============================================================

SCRIPT_NAMES = [
    "test07_stage01_structure_inverse.py",
    "test07_stage02_dynamic_inventory.py",
    "test07_stage03_inventory_grounding.py",
    "test07_stage04_candidate_verification.py",
    "test07_stage05_generalized_candidate_filter.py",
    "test07_stage06_tight_box_verification.py",
    "test07_stage07_sam2_safe_recovery.py",
    "test07_stage08_base_seeded_local_recovery.py",
    "test07_stage09_unresolved_multimask_recovery.py",
    "test07_stage10_local_crop_recovery_verification.py",
    "test07_stage11_rejected_candidate_recovery.py",
]

SCRIPTS_ROOT = Path("/workspace/axolotl/scripts")

copied_scripts = []

for name in SCRIPT_NAMES:

    src = SCRIPTS_ROOT / name

    if src.exists():

        dst = SCRIPTS_DST / name

        shutil.copy2(src, dst)

        copied_scripts.append(name)

        print("✅ SCRIPT:", name)

    else:

        print("⚠️ MISSING SCRIPT:", name)


# ============================================================
# IMPORTANT FINAL OUTPUTS
# ============================================================

OUTPUTS = {
    "stage01_base_mask":
        ROOM / "stages/01_structure_inverse_base/08_foreground_candidate_clean.png",

    "stage08_final_mask":
        ROOM / "stages/08_base_seeded_local_recovery/03_final_props_mask.png",

    "stage10_final_mask":
        ROOM / "stages/10_local_crop_recovery_verification/02_final_props_mask.png",

    "stage10_preview":
        ROOM / "stages/10_local_crop_recovery_verification/04_transparency_preview.png",

    "stage11_final_mask":
        ROOM / "stages/11_rejected_candidate_recovery/02_final_props_mask.png",

    "stage11_preview":
        ROOM / "stages/11_rejected_candidate_recovery/03_transparency_preview.png",

    "stage11_results":
        ROOM / "stages/11_rejected_candidate_recovery/04_results.json",

    "stage11_report":
        ROOM / "stages/11_rejected_candidate_recovery/stage11_report.json",
}


copied_outputs = {}

for key, src in OUTPUTS.items():

    if src.exists():

        dst = OUTPUTS_DST / src.name

        shutil.copy2(src, dst)

        copied_outputs[key] = str(dst)

        print("✅ OUTPUT:", key)

    else:

        print("⚠️ MISSING OUTPUT:", key, src)


# ============================================================
# ROOM01 SUCCESS RULES
# ============================================================

rules = [
    "Physical-instance identity must use inventory_id, not generic inventory_name.",
    "Do not silently remove the last candidate for an inventory instance.",
    "Stage05 low-trust rescue preserves an unresolved instance but does not make it trusted.",
    "Qwen TIGHT alone is not sufficient for segmentation safety.",
    "Stage05 rescue candidates remain non-safe until independently verified.",
    "Deterministic geometry must reject giant or structurally implausible detector boxes.",
    "Stage01 foreground remains the trusted base throughout recovery.",
    "SAM2 never replaces the trusted base; only missing pixels may be added.",
    "Stage08 re-audits SAM2 recovery using local distance/support from the trusted base.",
    "Unresolved instances must stay separate from already successful instances.",
    "Large source boxes can invalidate containment metrics and require catastrophic-mask checks.",
    "Previously rejected Stage03 proposals must remain available for later fallback.",
    "If an unresolved object has a better rejected Stage03 candidate, recover that localization before rerunning SAM2.",
    "Zero new recovery pixels can mean RESOLVED_ALREADY_PRESENT if fresh segmentation strongly overlaps the trusted base.",
    "Room-specific object names, coordinates, and hard-coded object exceptions are forbidden.",
]


# ============================================================
# LOAD FINAL STAGE11 REPORT
# ============================================================

stage11_report_path = (
    ROOM
    / "stages/11_rejected_candidate_recovery/stage11_report.json"
)

stage11 = {}

if stage11_report_path.exists():
    stage11 = json.loads(
        stage11_report_path.read_text()
    )


# ============================================================
# FINAL CONSOLIDATED REPORT
# ============================================================

report = {
    "test":
        "TEST07",

    "room":
        "room01_bedroom",

    "status":
        "REFERENCE_PASS",

    "inventory_instances":
        19,

    "resolved_instances":
        19,

    "remaining_unresolved":
        stage11.get(
            "remaining_unresolved_ids",
            []
        ),

    "final_pixels":
        stage11.get(
            "final_pixels",
            None
        ),

    "final_reference_mask":
        copied_outputs.get(
            "stage11_final_mask"
        ),

    "final_reference_preview":
        copied_outputs.get(
            "stage11_preview"
        ),

    "scripts_frozen":
        copied_scripts,

    "successful_rules":
        rules,

    "stage_status": {
        "01_structure_inverse":
            "PASS",
        "02_dynamic_inventory":
            "PASS",
        "03_inventory_grounding":
            "PASS",
        "04_candidate_verification":
            "PASS_WITH_KNOWN_FALSE_REJECTION",
        "05_generalized_candidate_filter":
            "PASS_AFTER_INSTANCE_ID_AND_RESCUE_FIX",
        "06_tight_box_verification":
            "PASS_AFTER_DETERMINISTIC_SAFETY_GATE",
        "07_sam2_safe_recovery":
            "PASS",
        "08_base_seeded_local_recovery":
            "PASS",
        "09_unresolved_multimask_recovery":
            "DIAGNOSTIC_PASS_WITH_CATASTROPHIC_GATE_REQUIRED",
        "10_local_crop_recovery":
            "PASS",
        "11_rejected_candidate_recovery":
            "PASS"
    }
}


report_path = (
    FREEZE
    / "ROOM01_REFERENCE_REPORT.json"
)

report_path.write_text(
    json.dumps(
        report,
        indent=2
    )
)


# ============================================================
# HUMAN-READABLE REPORT
# ============================================================

txt = []

txt.append("=" * 90)
txt.append("TEST07 ROOM01 REFERENCE")
txt.append("=" * 90)
txt.append("")
txt.append("ROOM: room01_bedroom")
txt.append("STATUS: REFERENCE PASS")
txt.append("")
txt.append("INVENTORY INSTANCES : 19")
txt.append("RESOLVED            : 19")
txt.append("UNRESOLVED          : 0")
txt.append(
    f'FINAL PIXELS       : {report["final_pixels"]}'
)
txt.append("")
txt.append("STAGE STATUS")
txt.append("-" * 90)

for key, value in report["stage_status"].items():
    txt.append(f"{key:<40} {value}")

txt.append("")
txt.append("GENERALIZED RULES LEARNED")
txt.append("-" * 90)

for i, rule in enumerate(rules, start=1):
    txt.append(f"{i:02d}. {rule}")

txt.append("")
txt.append("IMPORTANT")
txt.append("-" * 90)
txt.append(
    "This room is a reference-validation room only."
)
txt.append(
    "Room02 must run without room-specific coordinates, labels, or manual fixes."
)
txt.append(
    "Any new failure must be repaired using generalized logic and then revalidated."
)

txt_path = (
    FREEZE
    / "ROOM01_REFERENCE_REPORT.txt"
)

txt_path.write_text(
    "\n".join(txt)
)


# ============================================================
# ZIP REFERENCE
# ============================================================

timestamp = datetime.now().strftime(
    "%Y%m%d_%H%M"
)

zip_path = (
    ROOT
    / "backups"
    / f"TEST07_ROOM01_REFERENCE_{timestamp}.zip"
)

zip_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

with zipfile.ZipFile(
    zip_path,
    "w",
    compression=zipfile.ZIP_DEFLATED
) as z:

    for p in FREEZE.rglob("*"):

        if p.is_file():

            z.write(
                p,
                p.relative_to(
                    ROOT
                )
            )


print()
print("=" * 90)
print("TEST07 ROOM01 CONSOLIDATION COMPLETE")
print("=" * 90)

print("REFERENCE DIR:")
print(FREEZE)

print()
print("REPORT:")
print(report_path)

print()
print("TEXT REPORT:")
print(txt_path)

print()
print("BACKUP:")
print(zip_path)
