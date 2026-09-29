
# ============================================================
# TEST07 - ROOM03
# STAGE01Z2
# QWEN GENERALIZED SURFACE-TEMPLATE RECONSTRUCTION
#
# SCRIPT:
# test07_stage01z2_qwen_surface_template_full_prompt.py
#
# MODEL:
# Qwen/Qwen-Image-Edit-2509
#
# PIPELINE:
# QwenImageEditPlusPipeline
#
# PURPOSE:
# Convert arbitrary architectural image into standardized
# surface template while preserving geometry and real props.
#
# WALLS   -> plain smooth matte white
# CEILING -> plain smooth matte white
# FLOOR   -> solid #00C800
#
# LARGE WALL MIRROR:
# remove only when covering editable wall.
#
# INTERNAL TRANSPARENT ARCHITECTURAL PARTITION:
# remove only when obstructing editable architecture.
#
# NORMAL WINDOWS / GLASS / SCREENS:
# preserve.
#
# IMPORTANT:
# - Original RGB is the Qwen input.
# - No destructive placeholder input.
# - Frozen prop pixels are restored EXACTLY after generation.
# - Previous TEST07 stages are NEVER overwritten.
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

from pathlib import Path
from PIL import Image, ImageDraw
from IPython.display import display

import os
import gc
import json
import time
import hashlib
import numpy as np
import torch

from diffusers import QwenImageEditPlusPipeline


# ============================================================
# 2. VERSION / PATHS
# ============================================================

STAGE_NAME = (
    "TEST07_STAGE01Z2_"
    "QWEN_SURFACE_TEMPLATE_FULL_PROMPT"
)

SCRIPT_NAME = (
    "test07_stage01z2_"
    "qwen_surface_template_full_prompt.py"
)

MODEL_ID = "Qwen/Qwen-Image-Edit-2509"


BASE = Path(
    "/workspace/axolotl"
)

SCRIPTS_DIR = (
    BASE / "scripts"
)

ROOT = (
    BASE
    / "test07"
    / "runs"
    / "room03_bathroom"
    / "stages"
)

OUT = (
    ROOT
    / "01z2_qwen_surface_template_full_prompt"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


ORIGINAL_PATH = (
    ROOT
    / "01_structure_inverse_base"
    / "00_input_resized.png"
)


PROP_MASK_PATH = (
    ROOT
    / "01y3_frozen_canonical_prop_protection"
    / "00_canonical_prop_protection_mask.png"
)


CACHE_DIR = Path(
    "/workspace/data/huggingface-cache"
)

CACHE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


os.environ["HF_HOME"] = str(
    CACHE_DIR
)

os.environ["HUGGINGFACE_HUB_CACHE"] = str(
    CACHE_DIR
)


# ============================================================
# 3. GENERATION SETTINGS
# ============================================================

SEED = 12345

NUM_INFERENCE_STEPS = 20

TRUE_CFG_SCALE = 4.0

DTYPE = torch.bfloat16


# ============================================================
# 4. FULL GENERALIZED PROMPT
# ============================================================

PROMPT = r"""
TASK:

Convert the input architectural/room image into a clean neutral surface-template image for a tile visualization pipeline.

The input may be ANY type of space, including:

living room, bedroom, kitchen, bathroom, dining room, office, showroom, corridor, lobby, commercial interior, balcony, terrace, outdoor area, or any other architectural environment.

The output must remain the SAME ORIGINAL SCENE.

This is a SURFACE PREPARATION task only.

DO NOT redesign, beautify, restage, reconstruct, recompose, or reinterpret the room.


==================================================
1. PRESERVE ORIGINAL SCENE
==================================================

Preserve the original as closely as possible:

- camera position
- camera angle
- perspective
- focal appearance
- image framing
- image aspect ratio
- room geometry
- room proportions
- architectural layout
- wall positions
- wall boundaries
- floor boundaries
- ceiling boundaries
- corners
- wall-floor intersections
- wall-ceiling intersections
- columns
- beams
- recesses
- niches
- stairs
- steps
- openings
- windows
- doors
- built-in structures
- depth relationships
- object positions
- object scale
- object orientation
- lighting direction
- realistic shadows
- occlusions

The output must look like the exact same room captured from the exact same viewpoint.


==================================================
2. WALL SURFACE PROCESSING
==================================================

Detect ALL visible editable wall surfaces.

Remove existing wall surface finishes such as:

- ceramic tiles
- porcelain tiles
- marble
- granite
- natural stone
- artificial stone
- mosaic
- wallpaper
- paint colors
- wall patterns
- decorative prints
- wall texture
- wood wall cladding
- laminate wall cladding
- decorative panels
- textured panels
- wall coverings
- brick texture
- concrete texture
- decorative surface finishes
- grout lines
- tile joints

Replace ONLY the editable wall surfaces with:

CLEAN
UNIFORM
SMOOTH
PLAIN
MATTE WHITE

The white wall must follow the exact original wall:

- perspective
- plane
- orientation
- boundaries
- shape
- corners
- geometry
- depth
- occlusions

Do not flatten or straighten walls.

Do not merge separate walls.

Do not alter architectural geometry.


==================================================
3. CEILING SURFACE PROCESSING
==================================================

Detect ALL visible ceiling surfaces.

Remove existing ceiling surface appearance such as:

- decorative colors
- patterns
- textures
- tiles
- wooden finishes
- decorative panels
- wallpaper
- surface designs

Replace editable ceiling surfaces with:

CLEAN
UNIFORM
SMOOTH
PLAIN
MATTE WHITE

Preserve exact ceiling geometry including:

- false ceiling
- dropped ceiling
- recessed ceiling
- coves
- beams
- level changes
- ceiling edges
- openings

Preserve ceiling-mounted objects such as:

- lights
- spotlights
- chandeliers
- fans
- vents
- AC outlets
- speakers
- sprinklers
- sensors
- fixtures

Do NOT remove these objects.


==================================================
4. FLOOR SURFACE PROCESSING
==================================================

Detect the COMPLETE visible floor surface.

Remove the existing floor appearance including:

- tiles
- marble
- granite
- stone
- terrazzo
- wood
- laminate
- vinyl
- carpet texture
- concrete texture
- decorative patterns
- floor designs
- grout
- tile joints
- surface markings

Replace ONLY the editable visible floor surface with one uniform solid green color.

TARGET GREEN:

HEX: #00C800

RGB: 0, 200, 0

The green floor must be:

- completely uniform
- solid
- texture-free
- pattern-free
- grout-free
- joint-free

The green must follow the EXACT original:

- floor perspective
- floor plane
- depth
- geometry
- boundaries
- shape
- edges
- corners
- occlusions
- floor-wall intersection

Do NOT paint vertical walls green.

Do NOT paint objects green.

Do NOT paint furniture green.

Do NOT paint stairs, walls, cabinets, fixtures or vertical surfaces green unless a surface is clearly part of the actual floor plane.


==================================================
5. OBJECT AND PROP PRESERVATION
==================================================

Preserve ALL real physical objects and props unless explicitly identified later as removable mirrors or removable transparent architectural partitions.

Examples of objects that MUST normally remain:

- sofa
- chair
- table
- bed
- mattress
- cabinet
- wardrobe
- kitchen cabinet
- kitchen island
- countertop
- sink
- basin
- toilet
- bathtub
- sanitaryware
- faucets
- taps
- shower head
- plumbing fixtures
- appliances
- refrigerator
- oven
- television
- monitor
- electronics
- shelves
- lights
- lamps
- fans
- switches
- sockets
- electrical panels
- handles
- doors
- windows
- curtains
- blinds
- plants
- artwork
- decoration
- accessories
- furniture
- equipment
- fixed fixtures
- movable objects
- visible structural elements

Preserve each object's:

- exact position
- approximate shape
- dimensions
- scale
- perspective
- orientation
- occlusion
- relationship with the room

Do NOT move objects.

Do NOT resize objects.

Do NOT replace objects.

Do NOT redesign objects.

Do NOT add new objects.

Do NOT hallucinate furniture.

Do NOT remove normal props just because they cover part of a wall or floor.


==================================================
6. OCCLUSION RULE
==================================================

Foreground objects must continue to correctly cover surfaces behind them.

If a sofa covers part of the floor:
KEEP the sofa.
Change only the visible floor around and beneath visible gaps to green.

If a cabinet covers part of a wall:
KEEP the cabinet.
Change only the visible wall around it to white.

If a fixture is mounted on a wall:
KEEP the fixture.
Change only the visible wall surface behind and around it.

NEVER paint white or green over foreground objects.

Maintain natural object boundaries.


==================================================
7. MIRROR / REFLECTIVE PANEL PROCESSING
==================================================

Detect large mirrors or large reflective decorative wall panels that cover an editable wall region.

Remove the MIRROR / REFLECTIVE SURFACE itself.

Remove the reflected scene content inside that mirror region.

Replace that mirror region with a:

CLEAN
PLAIN
MATTE WHITE

surface aligned with the corresponding wall plane.

Preserve all independent objects near or in front of the mirror, including:

- mirror lights
- lamps
- sinks
- faucets
- shelves
- cabinets
- fixtures
- accessories

Do not remove these objects.

IMPORTANT:

Do NOT blindly remove every reflective object.

KEEP:

- TV screens
- monitors
- appliance screens
- oven glass
- shiny metal
- polished furniture
- small decorative mirrors unless they clearly cover an editable wall area
- reflective furniture
- glossy fixtures
- polished surfaces

Remove only a mirror or reflective panel whose purpose in this task is to expose an editable wall surface.

When uncertain, PRESERVE the object.


==================================================
8. TRANSPARENT / GLASS PARTITION PROCESSING
==================================================

Detect transparent or semi-transparent ARCHITECTURAL PARTITIONS that obstruct editable wall or floor surfaces.

Examples that may be removed:

- shower glass partition
- glass shower enclosure
- transparent internal divider
- glass room separator
- transparent internal partition
- partition door that is part of a removable glass enclosure
- partition frames
- partition rails
- partition handles
- partition hinges
- associated partition hardware

Remove ONLY such obstructing transparent architectural partitions when necessary to reveal editable surfaces.

IMPORTANT BACKGROUND RULE:

Anything already visible THROUGH transparent glass must remain visually consistent with the original scene.

Do NOT regenerate or change a visible object merely because it was behind glass.

Preserve:

- visible wall geometry
- visible floor geometry
- fixtures
- objects
- shadows
- architectural structure

behind the transparent partition.

If an area was completely hidden by the removed partition and cannot be directly recovered:

Reconstruct ONLY the minimum required continuation of the surrounding architecture.

If clearly WALL:
use plain matte WHITE.

If clearly FLOOR:
use solid #00C800 GREEN.

Do NOT invent furniture or fixtures.


==================================================
9. GLASS CLASSIFICATION — VERY IMPORTANT
==================================================

DO NOT remove all glass.

KEEP unchanged:

- windows
- exterior glazing
- balcony glazing
- exterior glass doors
- ordinary windows
- glass table
- glass furniture
- display cabinet glass
- kitchen cabinet glass
- TV screen
- monitor screen
- oven glass
- appliance glass
- decorative glass
- glass shelves
- bottles
- small glass objects
- transparent furniture elements

REMOVE ONLY transparent architectural partitions that obstruct editable architectural surfaces.

When classification is uncertain:

PRESERVE THE GLASS OBJECT.


==================================================
10. DOORS AND WINDOWS
==================================================

Preserve normal doors and windows.

Do not fill them with white.

Do not treat windows as walls.

Do not treat doors as walls.

Preserve:

- door panels
- door frames
- handles
- window frames
- window openings
- outside views where visible
- glass windows
- exterior doors

unless a specific transparent panel is clearly part of a removable internal architectural partition.


==================================================
11. BUILT-IN ARCHITECTURE
==================================================

Preserve the geometry of:

- niches
- recessed shelves
- built-in shelves
- columns
- pillars
- beams
- steps
- staircases
- raised platforms
- recessed areas
- fireplace openings
- wall openings
- arches
- false ceilings
- skirting geometry
- ledges

Only neutralize their editable SURFACE MATERIAL where appropriate.

Do not alter their shape.


==================================================
12. LIGHTING AND SHADOW PRESERVATION
==================================================

Preserve the natural illumination of the original photograph.

White walls and green floor must still visually belong to the original room.

Maintain reasonable:

- shading
- depth cues
- ambient light
- soft shadows
- contact shadows
- room illumination

Do not make the image look like a flat segmentation map.

Do not remove shadows belonging to preserved objects unless absolutely necessary during surface replacement.

Do not dramatically brighten or darken the room.


==================================================
13. GEOMETRY PRESERVATION — ABSOLUTE PRIORITY
==================================================

Geometry preservation has higher priority than visual beautification.

DO NOT:

- move the camera
- rotate the camera
- change camera height
- change perspective
- change field of view
- zoom
- crop
- extend the image
- alter room size
- enlarge walls
- shrink walls
- move room corners
- move floor boundaries
- move ceiling boundaries
- straighten perspective
- warp architecture
- make walls more symmetrical
- change doorway size
- change window size
- change openings
- change ceiling height
- modify object positions
- change furniture layout
- generate a different version of the room

The output must remain spatially aligned with the original input image.


==================================================
14. DO NOT REDESIGN
==================================================

THIS IS NOT:

- an interior design task
- an image enhancement task
- a renovation proposal
- a room redesign
- a staging task
- a decluttering task
- a furniture replacement task
- an architecture generation task

Do not improve the room.

Do not make the room more modern.

Do not add decorative elements.

Do not change furniture style.

Do not change fixtures.

Do not create a new room.


==================================================
15. NEGATIVE INSTRUCTIONS
==================================================

AVOID ALL OF THE FOLLOWING:

- new furniture
- removed furniture
- moved furniture
- changed furniture
- different cabinet design
- different sanitaryware
- different appliances
- different doors
- different windows
- new lights
- changed lighting fixtures
- new decorations
- plants being removed
- objects being duplicated
- objects disappearing
- objects changing shape
- hallucinated objects
- geometry distortion
- warped walls
- warped floors
- incorrect perspective
- shifted corners
- changed camera angle
- zoom changes
- crop changes
- room redesign
- architectural redesign
- beautification
- excessive smoothing of objects
- replacing real objects with generated alternatives
- changing object colors unnecessarily
- green walls
- green furniture
- green cabinets
- green fixtures
- green doors
- green windows
- white furniture
- white objects caused by wall masking
- white-painted foreground props
- remaining floor textures
- remaining floor tiles
- visible grout on floor
- remaining wall tiles
- wall patterns
- decorative wall textures
- artificial tile patterns
- artificial grout lines
- mirror reflections after mirror removal
- unnecessary removal of windows
- unnecessary removal of normal glass
- removal of exterior glazing
- removal of TV screens
- removal of appliance glass
- transparent furniture being deleted
- changing exterior views
- changing objects visible through glass
- flattening room depth
- changing shadows dramatically
- overexposure
- unrealistic pure-white foreground objects
- blending object edges into wall
- painting over object boundaries
- invented architectural structures
- reconstructed furniture
- missing architectural fixtures


==================================================
16. DECISION PRIORITY
==================================================

When uncertain about whether something should be changed:

PRESERVE IT.

Only modify a region when it is confidently identified as:

1. editable wall surface
2. editable ceiling surface
3. editable floor surface
4. removable large wall mirror / reflective wall panel
5. removable transparent internal architectural partition

Preservation is more important than aggressive removal.


==================================================
17. FINAL TARGET
==================================================

Create the SAME ORIGINAL ROOM with the following standardized appearance:

WALLS:
plain smooth matte white

CEILING:
plain smooth matte white

FLOOR:
solid uniform #00C800 green

LARGE WALL MIRROR / REFLECTIVE WALL PANEL:
removed only when it covers an editable wall region, replaced with plain white wall surface

OBSTRUCTING TRANSPARENT INTERNAL ARCHITECTURAL PARTITION:
removed where necessary while preserving everything visible behind it

NORMAL WINDOWS / EXTERIOR GLASS / GLASS FURNITURE / SCREENS:
preserved

FURNITURE / FIXTURES / PROPS / APPLIANCES / DECOR:
preserved

ROOM GEOMETRY:
preserved

CAMERA:
preserved

PERSPECTIVE:
preserved

IMAGE COMPOSITION:
preserved

Do not add anything.

Do not redesign anything.

Do not intentionally alter anything outside the required editable architectural surfaces.

The final image must function as a clean architectural surface template for applying new floor and wall tiles later while maintaining maximum pixel-level similarity to the original scene everywhere outside the edited surfaces.
"""


# ============================================================
# 5. NEGATIVE PROMPT
# ============================================================

NEGATIVE_PROMPT = r"""
different room,
redesigned room,
changed camera,
changed camera angle,
changed perspective,
changed field of view,
zoomed image,
cropped image,
extended image,
warped architecture,
warped walls,
warped floor,
shifted corners,
changed room dimensions,
moved objects,
resized objects,
replaced furniture,
different furniture,
missing objects,
duplicated objects,
hallucinated objects,
new furniture,
new decoration,
changed sanitaryware,
changed cabinet,
changed countertop,
changed doors,
changed windows,
new lights,
green walls,
green furniture,
green cabinets,
green fixtures,
green doors,
green windows,
white furniture,
white foreground objects,
white-painted props,
remaining wall tiles,
remaining wall patterns,
remaining floor tiles,
remaining floor texture,
floor grout,
tile joints,
mirror reflections,
remaining removable shower glass partition,
removed normal windows,
removed exterior glazing,
removed appliance glass,
changed exterior views,
flat segmentation map,
artificial placeholder blocks,
painted polygons,
collage,
cutout appearance,
flattened depth,
overexposure
"""


# ============================================================
# 6. START REPORT
# ============================================================

print()
print("=" * 100)
print(STAGE_NAME)
print("=" * 100)

print()
print("SCRIPT NAME :", SCRIPT_NAME)
print("OUTPUT DIR  :", OUT)
print("MODEL       :", MODEL_ID)


# ============================================================
# 7. INPUT VALIDATION
# ============================================================

required = {
    "ORIGINAL":
        ORIGINAL_PATH,

    "FROZEN PROP MASK":
        PROP_MASK_PATH,
}


print()
print("INPUT CHECK")
print("-" * 100)


for name, path in required.items():

    exists = (
        path.exists()
    )

    print(
        f"{name:20s}",
        "✅ FOUND"
        if exists
        else "❌ MISSING",
        path
    )

    if not exists:

        raise FileNotFoundError(
            path
        )


# ============================================================
# 8. LOAD ORIGINAL
# ============================================================

original_pil = Image.open(
    ORIGINAL_PATH
).convert(
    "RGB"
)

original = np.array(
    original_pil
)

H, W = (
    original.shape[:2]
)


print()
print(
    "MASTER SIZE:",
    W,
    "x",
    H
)


# ============================================================
# 9. LOAD PROP MASK
# ============================================================

prop_mask_pil = Image.open(
    PROP_MASK_PATH
).convert(
    "L"
)


if prop_mask_pil.size != (
    W,
    H
):

    prop_mask_pil = (
        prop_mask_pil.resize(
            (
                W,
                H
            ),
            Image.Resampling.NEAREST
        )
    )


prop_mask = (
    np.array(
        prop_mask_pil
    )
    > 0
)


protected_pixels = int(
    prop_mask.sum()
)


print(
    "FROZEN PROP PIXELS:",
    protected_pixels
)


# ============================================================
# 10. SAVE EXACT INPUT SNAPSHOT
# ============================================================

input_snapshot = (
    OUT
    / "00_input_original.png"
)

original_pil.save(
    input_snapshot
)


prop_snapshot = (
    OUT
    / "01_input_frozen_prop_mask.png"
)

prop_mask_pil.save(
    prop_snapshot
)


# ============================================================
# 11. SAVE PROMPTS BEFORE GENERATION
# ============================================================

prompt_path = (
    OUT
    / "02_prompt.txt"
)

prompt_path.write_text(
    PROMPT,
    encoding="utf-8"
)


negative_prompt_path = (
    OUT
    / "03_negative_prompt.txt"
)

negative_prompt_path.write_text(
    NEGATIVE_PROMPT,
    encoding="utf-8"
)


# ============================================================
# 12. MODEL MEMORY CLEANUP
# ============================================================

gc.collect()


if torch.cuda.is_available():

    torch.cuda.empty_cache()

    torch.cuda.ipc_collect()


# ============================================================
# 13. MEMORY STATUS
# ============================================================

if torch.cuda.is_available():

    free_before, total_before = (
        torch.cuda.mem_get_info()
    )

    print()
    print(
        "GPU FREE BEFORE MODEL GB:",
        round(
            free_before
            / 1024**3,
            2
        )
    )

    print(
        "GPU TOTAL GB:",
        round(
            total_before
            / 1024**3,
            2
        )
    )


# ============================================================
# 14. REUSE EXISTING PIPE IF POSSIBLE
# ============================================================
#
# When executed from Jupyter using exec(), the already-loaded
# global "pipe" may exist.
#
# We reuse it instead of loading Qwen twice.
#
# If no compatible pipe exists, load it here.
# ============================================================

reuse_existing_pipe = False


if "pipe" in globals():

    try:

        if isinstance(
            globals()["pipe"],
            QwenImageEditPlusPipeline
        ):

            pipe = globals()["pipe"]

            reuse_existing_pipe = True

    except Exception:

        reuse_existing_pipe = False


if reuse_existing_pipe:

    print()
    print(
        "✅ Reusing existing QwenImageEditPlusPipeline"
    )

    model_load_seconds = 0.0


else:

    print()
    print("=" * 100)
    print("LOADING QWEN IMAGE EDIT")
    print("=" * 100)

    load_start = (
        time.time()
    )


    pipe = (
        QwenImageEditPlusPipeline
        .from_pretrained(
            MODEL_ID,
            torch_dtype=DTYPE,
            cache_dir=str(
                CACHE_DIR
            ),
        )
    )


    print()
    print(
        "✅ Pipeline loaded"
    )


    pipe.enable_sequential_cpu_offload()


    print(
        "✅ Sequential CPU offload enabled"
    )


    model_load_seconds = (
        time.time()
        - load_start
    )


    print(
        "MODEL LOAD TIME SEC:",
        round(
            model_load_seconds,
            2
        )
    )


# ============================================================
# 15. ENSURE LOW-VRAM MODE
# ============================================================

try:

    pipe.enable_sequential_cpu_offload()

    print(
        "✅ Sequential CPU offload active"
    )

except Exception as e:

    print(
        "CPU offload status:",
        str(e)
    )


# ============================================================
# 16. GENERATOR
# ============================================================

generator = (
    torch.Generator(
        device="cpu"
    )
    .manual_seed(
        SEED
    )
)


# ============================================================
# 17. RUN QWEN
# ============================================================

print()
print("=" * 100)
print("STARTING QWEN GENERATION")
print("=" * 100)

print(
    "SEED:",
    SEED
)

print(
    "STEPS:",
    NUM_INFERENCE_STEPS
)

print(
    "TRUE CFG SCALE:",
    TRUE_CFG_SCALE
)


generation_start = (
    time.time()
)


with torch.inference_mode():

    result = pipe(

        image=original_pil,

        prompt=PROMPT,

        negative_prompt=NEGATIVE_PROMPT,

        true_cfg_scale=TRUE_CFG_SCALE,

        num_inference_steps=NUM_INFERENCE_STEPS,

        generator=generator,

        output_type="pil",
    )


generation_seconds = (
    time.time()
    - generation_start
)


print()
print(
    "GENERATION TIME SEC:",
    round(
        generation_seconds,
        2
    )
)


# ============================================================
# 18. RAW QWEN OUTPUT
# ============================================================

raw_qwen = (
    result.images[0]
    .convert(
        "RGB"
    )
)


raw_original_size = (
    raw_qwen.size
)


print(
    "RAW QWEN SIZE:",
    raw_original_size
)


raw_qwen_path = (
    OUT
    / "04_raw_qwen_surface_template.png"
)


raw_qwen.save(
    raw_qwen_path
)


# ============================================================
# 19. NORMALIZE TO EXACT MASTER SIZE
# ============================================================

if raw_qwen.size != (
    W,
    H
):

    print(
        "Normalizing Qwen output to exact master dimensions..."
    )


    qwen_master_size = (
        raw_qwen.resize(
            (
                W,
                H
            ),
            Image.Resampling.LANCZOS
        )
    )


else:

    qwen_master_size = (
        raw_qwen.copy()
    )


qwen_master_path = (
    OUT
    / "05_qwen_surface_template_master_size.png"
)


qwen_master_size.save(
    qwen_master_path
)


generated = np.array(
    qwen_master_size
)


# ============================================================
# 20. RESTORE FROZEN PROPS EXACTLY
# ============================================================

safe = (
    generated.copy()
)


safe[
    prop_mask
] = original[
    prop_mask
]


safe_path = (
    OUT
    / "06_safe_surface_template_props_restored.png"
)


Image.fromarray(
    safe
).save(
    safe_path
)


# ============================================================
# 21. VERIFY EXACT RGB
# ============================================================

rgb_diff = np.abs(

    safe.astype(
        np.int16
    )

    -

    original.astype(
        np.int16
    )
)


protected_diff = (
    rgb_diff[
        prop_mask
    ]
)


if protected_diff.size:

    max_prop_rgb_error = int(
        protected_diff.max()
    )

    mean_prop_rgb_error = float(
        protected_diff.mean()
    )


else:

    max_prop_rgb_error = 0

    mean_prop_rgb_error = 0.0


print()
print("=" * 100)
print("FROZEN PROP RGB VERIFICATION")
print("=" * 100)

print(
    "PROTECTED PIXELS:",
    protected_pixels
)

print(
    "MAX RGB ERROR:",
    max_prop_rgb_error
)

print(
    "MEAN RGB ERROR:",
    mean_prop_rgb_error
)


if max_prop_rgb_error == 0:

    print(
        "✅ EXACT ORIGINAL RGB PRESERVED INSIDE FROZEN PROP MASK"
    )


else:

    print(
        "❌ PROP RGB VERIFICATION FAILED"
    )


# ============================================================
# 22. CREATE PROP OVERLAY
# ============================================================

prop_overlay = (
    original.copy()
)


green = np.array(
    [
        0,
        255,
        0
    ],
    dtype=np.float32
)


prop_overlay[
    prop_mask
] = (

    0.40
    *
    prop_overlay[
        prop_mask
    ].astype(
        np.float32
    )

    +

    0.60
    *
    green

).clip(
    0,
    255
).astype(
    np.uint8
)


prop_overlay_path = (
    OUT
    / "07_frozen_prop_overlay.png"
)


Image.fromarray(
    prop_overlay
).save(
    prop_overlay_path
)


# ============================================================
# 23. 3-PANEL COMPARISON
# ============================================================

LABEL_H = 36


comparison = Image.new(

    "RGB",

    (
        W * 3,
        H + LABEL_H
    ),

    (
        255,
        255,
        255
    )
)


comparison.paste(
    original_pil,
    (
        0,
        LABEL_H
    )
)


comparison.paste(
    qwen_master_size,
    (
        W,
        LABEL_H
    )
)


safe_pil = (
    Image.fromarray(
        safe
    )
)


comparison.paste(
    safe_pil,
    (
        W * 2,
        LABEL_H
    )
)


draw = ImageDraw.Draw(
    comparison
)


draw.text(
    (
        8,
        10
    ),
    "ORIGINAL",
    fill=(
        0,
        0,
        0
    )
)


draw.text(
    (
        W + 8,
        10
    ),
    "RAW QWEN",
    fill=(
        0,
        0,
        0
    )
)


draw.text(
    (
        W * 2 + 8,
        10
    ),
    "SAFE + ORIGINAL PROPS",
    fill=(
        0,
        0,
        0
    )
)


comparison_path = (
    OUT
    / "08_original_raw_safe_comparison.png"
)


comparison.save(
    comparison_path
)


# ============================================================
# 24. ORIGINAL VS FINAL
# ============================================================

compare_final = Image.new(

    "RGB",

    (
        W * 2,
        H + LABEL_H
    ),

    (
        255,
        255,
        255
    )
)


compare_final.paste(
    original_pil,
    (
        0,
        LABEL_H
    )
)


compare_final.paste(
    safe_pil,
    (
        W,
        LABEL_H
    )
)


draw_final = ImageDraw.Draw(
    compare_final
)


draw_final.text(
    (
        8,
        10
    ),
    "ORIGINAL",
    fill=(
        0,
        0,
        0
    )
)


draw_final.text(
    (
        W + 8,
        10
    ),
    "STAGE01Z2 SURFACE TEMPLATE",
    fill=(
        0,
        0,
        0
    )
)


compare_final_path = (
    OUT
    / "09_original_vs_final_surface_template.png"
)


compare_final.save(
    compare_final_path
)


# ============================================================
# 25. SAVE MASKED-RGB PROP LAYER
# ============================================================

rgba = np.zeros(
    (
        H,
        W,
        4
    ),
    dtype=np.uint8
)


rgba[
    ...,
    :3
] = original


rgba[
    ...,
    3
] = (
    prop_mask.astype(
        np.uint8
    )
    * 255
)


rgba_path = (
    OUT
    / "10_frozen_props_original_rgb_rgba.png"
)


Image.fromarray(
    rgba,
    mode="RGBA"
).save(
    rgba_path
)


# ============================================================
# 26. HASH IMPORTANT INPUTS
# ============================================================

def sha256_file(path):

    h = hashlib.sha256()

    with open(
        path,
        "rb"
    ) as f:

        while True:

            chunk = (
                f.read(
                    1024 * 1024
                )
            )

            if not chunk:

                break

            h.update(
                chunk
            )

    return (
        h.hexdigest()
    )


input_hash = (
    sha256_file(
        ORIGINAL_PATH
    )
)


mask_hash = (
    sha256_file(
        PROP_MASK_PATH
    )
)


# ============================================================
# 27. SAVE REPORT
# ============================================================

report = {

    "stage":
        STAGE_NAME,

    "status":
        "COMPLETED",

    "script_name":
        SCRIPT_NAME,

    "model":
        MODEL_ID,

    "pipeline":
        "QwenImageEditPlusPipeline",

    "input_original":
        str(
            ORIGINAL_PATH
        ),

    "input_original_sha256":
        input_hash,

    "frozen_prop_mask":
        str(
            PROP_MASK_PATH
        ),

    "frozen_prop_mask_sha256":
        mask_hash,

    "output_directory":
        str(
            OUT
        ),

    "method":
        (
            "original RGB -> generalized Qwen surface-template "
            "edit -> normalize to original dimensions -> restore "
            "frozen physical props exactly from original RGB"
        ),

    "generation": {

        "seed":
            SEED,

        "steps":
            NUM_INFERENCE_STEPS,

        "true_cfg_scale":
            TRUE_CFG_SCALE,

        "dtype":
            str(
                DTYPE
            ),

        "sequential_cpu_offload":
            True,

        "reused_existing_pipeline":
            reuse_existing_pipe,

        "model_load_seconds":
            round(
                model_load_seconds,
                3
            ),

        "generation_seconds":
            round(
                generation_seconds,
                3
            ),
    },

    "image": {

        "master_width":
            W,

        "master_height":
            H,

        "raw_qwen_size":
            list(
                raw_original_size
            ),

        "protected_pixels":
            protected_pixels,

        "max_rgb_error_inside_frozen_props":
            max_prop_rgb_error,

        "mean_rgb_error_inside_frozen_props":
            mean_prop_rgb_error,
    },

    "target": {

        "walls":
            "plain smooth matte white",

        "ceiling":
            "plain smooth matte white",

        "floor":
            "#00C800 / RGB 0,200,0",

        "mirror":
            (
                "remove only large wall mirror / reflective panel "
                "covering editable wall"
            ),

        "glass":
            (
                "remove only obstructing transparent internal "
                "architectural partition"
            ),

        "normal_glass":
            "preserve",

        "props":
            "preserve",
    },

    "outputs": {

        "input_snapshot":
            str(
                input_snapshot
            ),

        "prop_mask_snapshot":
            str(
                prop_snapshot
            ),

        "prompt":
            str(
                prompt_path
            ),

        "negative_prompt":
            str(
                negative_prompt_path
            ),

        "raw_qwen":
            str(
                raw_qwen_path
            ),

        "master_size_qwen":
            str(
                qwen_master_path
            ),

        "safe_surface_template":
            str(
                safe_path
            ),

        "prop_overlay":
            str(
                prop_overlay_path
            ),

        "three_panel_comparison":
            str(
                comparison_path
            ),

        "original_vs_final":
            str(
                compare_final_path
            ),

        "props_rgba":
            str(
                rgba_path
            ),
    },

    "rules": [

        "Never overwrite Stage01Z1",

        "Never overwrite previous TEST07 stages",

        "Use original photograph as Qwen input",

        "No destructive flat placeholder input",

        "Geometry preservation has highest priority",

        "Walls -> matte white",

        "Ceiling -> matte white",

        "Floor -> #00C800 green",

        "Preserve normal doors/windows/glass",

        "Remove qualifying wall mirrors",

        "Remove qualifying internal transparent partitions",

        "Frozen prop RGB restored exactly from original",
    ]
}


report_path = (
    OUT
    / "00_stage01z2_report.json"
)


report_path.write_text(
    json.dumps(
        report,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# ============================================================
# 28. MEMORY STATUS AFTER GENERATION
# ============================================================

if torch.cuda.is_available():

    free_end, total_end = (
        torch.cuda.mem_get_info()
    )

    print()
    print(
        "GPU FREE AFTER GENERATION GB:",
        round(
            free_end
            / 1024**3,
            2
        )
    )


# ============================================================
# 29. FINAL RESULTS
# ============================================================

print()
print("=" * 100)
print("STAGE01Z2 COMPLETE")
print("=" * 100)

print()
print(
    "RAW QWEN:"
)

print(
    raw_qwen_path
)


print()
print(
    "SAFE SURFACE TEMPLATE:"
)

print(
    safe_path
)


print()
print(
    "3-PANEL COMPARISON:"
)

print(
    comparison_path
)


print()
print(
    "ORIGINAL VS FINAL:"
)

print(
    compare_final_path
)


print()
print(
    "REPORT:"
)

print(
    report_path
)


print()
print(
    "MAX RGB ERROR INSIDE FROZEN PROPS:",
    max_prop_rgb_error
)


# ============================================================
# 30. DISPLAY RESULTS
# ============================================================

print()
print(
    "ORIGINAL / RAW QWEN / SAFE"
)

display(
    comparison
)


print()
print(
    "ORIGINAL VS STAGE01Z2"
)

display(
    compare_final
)
