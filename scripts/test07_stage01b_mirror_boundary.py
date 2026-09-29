
from pathlib import Path
import json
import cv2

ROOM_ROOT = Path("/workspace/axolotl/test07/runs/room03_bathroom")
MASTER = ROOM_ROOT / "stages/01_structure_inverse_base/00_input_resized.png"
OUT = ROOM_ROOT / "stages/01b_mirror_boundary"

OUT.mkdir(parents=True, exist_ok=True)

img = cv2.imread(str(MASTER))
if img is None:
    raise FileNotFoundError(MASTER)

h, w = img.shape[:2]

print("=" * 80)
print("TEST07 - STAGE01B MIRROR BOUNDARY")
print("=" * 80)
print("MASTER:", MASTER)
print("SIZE:", w, "x", h)
print("OUTPUT:", OUT)

cv2.imwrite(str(OUT / "00_input.png"), img)

meta = {
    "room_id": "room03_bathroom",
    "image_width": w,
    "image_height": h,
    "master": str(MASTER),
    "rule": {
        "mirror_is_single_prop": True,
        "ignore_reflected_objects_inside_mirror": True,
        "preserve_outer_boundary": True
    }
}

with open(OUT / "00_stage01b_setup.json", "w") as f:
    json.dump(meta, f, indent=2)

print()
print("CREATED:")
print(OUT / "00_input.png")
print(OUT / "00_stage01b_setup.json")
print()
print("STAGE01B SETUP READY")
