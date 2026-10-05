"""
Two repairs at the seam between object removal and floor/wall detection,
made once per Clean Room run, on the files that run has just written.

1. Missed furniture (unremoved_furniture)

   The object detector can miss part of an object -- measured on a living
   room: a sofa's front cushion. A missed part is not inpainted, so it is
   still in the clean room, and the floor/wall segmenter, which runs on the
   clean room, may call it wall (a white cushion against a white wall). Tiles
   are then drawn on the sofa and nothing is put back over them.

   The segmenter itself tells them apart when it looks at the ORIGINAL photo,
   where the whole sofa is in view: a pixel that is in the floor or wall mask,
   was not removed, was not inpainted, and that the original photo labels as a
   movable piece of furniture is furniture. It is taken out of the floor/wall
   masks and added to the object layer, so it is restored over the tiles
   exactly like the rest of its object.

2. Floor behind objects (floor_behind_objects)

   Thin objects (chair and stool legs) leave ghosts after inpainting, and the
   segmenter then calls the floor between the legs neither floor nor wall: no
   tiles go there and the old floor shows between the legs. Such a region
   lies inside the boxes of objects standing on the floor, never touches the
   wall, and meets the floor only through the object's own pixels. It is added
   to the floor: tiles go there and the object is restored on top.

Both only move pixels between the object layer and the surfaces they were
wrongly given to, never between floor and wall; both are proportional to the
image size; and both are reported (pixels, classes) in the Clean Room metadata.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

#: ADE20K classes that are movable furniture and décor: never a tile surface.
#: Structural and opening classes (curtain, window, door, cabinet, ceiling...)
#: are deliberately not here.
MOVABLE = frozenset({
    "sofa", "chair", "armchair", "armchair leg", "swivel chair", "bed", "table", "window","coffee table", "desk", "cushion",  "pillow", "stool", "bench", "ottoman", "box", "basket", "vase", "pot", "flowerpot", "plant","book", "bottle", "bag", "toy", "seat", "blanket", "towel", "apparel", "tray", "bowl", "glass", "furniture", "other"
})

#: Specks below this (share of the image diagonal, as an opening radius) are
#: label noise, not furniture.
SPECK_RADIUS_FRACTION = 0.002

#: The segmenter's furniture label stops a few pixels short of the object's
#: edge; gaps up to twice this (share of the diagonal) between the recovered
#: furniture and the object layer are closed -- next to that furniture only.
EDGE_GAP_FRACTION = 0.006

#: Floor behind objects: share of the region inside floor-standing object
#: boxes, and share of its border that may touch the wall.
INSIDE_BOXES_SHARE = 0.9
MAX_WALL_BORDER_SHARE = 0.02


def _disc(radius: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def unremoved_furniture(floor, wall, objects, inpainted, labels, names) -> tuple[np.ndarray, dict]:
    """(mask, pixels per class) of furniture still inside the floor/wall masks."""
    h, w = floor.shape
    movable_ids = [k for k, v in names.items() if v in MOVABLE]
    cand = (floor | wall) & ~objects & ~inpainted & np.isin(labels, movable_ids)
    r = max(1, int(round(SPECK_RADIUS_FRACTION * math.hypot(h, w))))
    cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_OPEN, _disc(r)).astype(bool)
    if cand.any():
        rc = max(2, int(round(EDGE_GAP_FRACTION * math.hypot(h, w))))
        closed = cv2.morphologyEx((cand | objects).astype(np.uint8), cv2.MORPH_CLOSE, _disc(rc)).astype(bool)
        near = cv2.dilate(cand.astype(np.uint8), _disc(rc)).astype(bool)
        # Includes the inpainting's margin ring there: enclosed by furniture on
        # both sides, it is the sofa's own edge, not floor or wall.
        cand |= closed & near & ~objects & (floor | wall)
    by_class = {names[int(k)]: int((cand & (labels == k)).sum()) for k in np.unique(labels[cand])} if cand.any() else {}
    return cand, by_class


def floor_behind_objects(floor, wall, objects, boxes) -> np.ndarray:
    """Regions to add to the floor (see the module docstring)."""
    h, w = floor.shape
    standing = np.zeros_like(floor)
    for x0, y0, x1, y1 in boxes:
        x0, y0 = int(max(0, x0)), int(max(0, y0))
        x1, y1 = int(min(w, x1)), int(min(h, y1))
        if x1 <= x0 or y1 <= y0:
            continue
        foot = floor[max(0, y1 - 3):min(h, y1 + 6), x0:x1]
        if foot.size and foot.mean() > 0.05:                  # stands on the floor
            standing[y0:y1, x0:x1] = True
    add = np.zeros_like(floor)
    unassigned = ~floor & ~wall & ~objects
    n, lab, _, _ = cv2.connectedComponentsWithStats(unassigned.astype(np.uint8), 8)
    through = objects & standing
    for i in range(1, n):
        comp = lab == i
        if (comp & standing).sum() < INSIDE_BOXES_SHARE * comp.sum():
            continue
        ring = cv2.dilate(comp.astype(np.uint8), _disc(2)).astype(bool) & ~comp
        if ring.any() and (ring & wall).mean() > MAX_WALL_BORDER_SHARE:
            continue
        # Reaches the floor directly, or through the standing objects' pixels.
        grown = comp.copy()
        for _ in range(64):
            if (cv2.dilate(grown.astype(np.uint8), _disc(2)).astype(bool) & floor).any():
                add |= comp
                break
            nxt = grown | (cv2.dilate(grown.astype(np.uint8), _disc(2)).astype(bool) & through)
            if (nxt == grown).all():
                break
            grown = nxt
    return add


def apply(segments: Path, original_rgb: np.ndarray) -> dict:
    """
    Repair one Clean Room job's `segments/` in place: FLOOR_MASK.png,
    WALL_MASK.png and ALL_OBJECTS.png. Returns what changed.
    """
    import floor_wall
    import surfaces
    from extraction import output

    floor_path = segments / floor_wall.FLOOR_MASK_FILENAME
    wall_path = segments / floor_wall.WALL_MASK_FILENAME
    objects_path = segments / output.ALL_OBJECTS_FILENAME
    clean_path = segments / output.CLEAN_ROOM_FILENAME
    if not (floor_path.exists() and wall_path.exists() and objects_path.exists() and clean_path.exists()):
        return {"applied": False, "reason": "masks, objects or clean room missing"}

    floor = np.asarray(Image.open(floor_path)) > 127
    wall = np.asarray(Image.open(wall_path)) > 127
    layer = np.array(Image.open(objects_path).convert("RGBA"))
    clean = np.asarray(Image.open(clean_path).convert("RGB"))
    rgb = np.ascontiguousarray(original_rgb[..., :3], dtype=np.uint8)
    if not (floor.shape == wall.shape == layer.shape[:2] == clean.shape[:2] == rgb.shape[:2]):
        return {"applied": False, "reason": "sizes differ"}

    objects = layer[..., 3] > 0
    mirrors_path = segments / output.MIRRORS_FILENAME
    if mirrors_path.exists():
        mirrors = np.asarray(Image.open(mirrors_path).convert("RGBA"))[..., 3] > 0
        if mirrors.shape == objects.shape:
            objects = objects | mirrors
    inpainted = np.any(rgb != clean, axis=2)

    labels, names = surfaces.label_map(rgb)
    furniture, by_class = unremoved_furniture(floor, wall, objects, inpainted, labels, names)
    if furniture.any():
        floor = floor & ~furniture
        wall = wall & ~furniture
        layer[furniture, :3] = rgb[furniture]
        layer[furniture, 3] = 255
        Image.fromarray(layer, mode="RGBA").save(objects_path)

    boxes = []
    objects_json = segments / "objects.json"
    if objects_json.exists():
        boxes = [o["bbox"] for o in json.loads(objects_json.read_text()).get("objects", []) if o.get("bbox")]
    added = floor_behind_objects(floor, wall, objects | furniture, boxes)
    floor = floor | added

    floor_wall.write(rgb, floor_wall.FloorWall(floor=floor, wall=wall, shape=floor.shape), segments)
    return {"applied": True, "furniture_px": int(furniture.sum()), "furniture_classes": by_class,
            "floor_behind_objects_px": int(added.sum())}
