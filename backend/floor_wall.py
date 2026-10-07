"""
Floor and wall detection for a cleaned room.

One stage, one question: which pixels of `CLEAN_ROOM.png` are floor, and which
are wall. It produces two binary masks at the source image's exact resolution —
`FLOOR_MASK.png` and `WALL_MASK.png`, white where the surface is, black
everywhere else — and nothing else in the pipeline changes because of it.

    CLEAN_ROOM.png -> Floor + Wall detection -> FLOOR_MASK.png + WALL_MASK.png

In full:

    CLEAN_ROOM.png
      -> ADE20K semantic segmentation        (surfaces.py, already loaded)
      -> floor and wall class probabilities   the only two output classes
      -> each must beat every other class     mutually exclusive by construction
      -> guided-filter edge snap             (surfaces.snap, already tuned)
      -> minimal cleanup                     specks out, enclosed pinholes in
      -> FLOOR_MASK.png + WALL_MASK.png

Two classes, two files. Ceiling, door, curtain and furniture are background:
they are never detected, never stored and never returned.

**No new model.** `surfaces.py` is already an ADE20K semantic segmenter —
SegFormer-B4, 150 classes, ONNX on the CPU — and it is already the module the
rest of the backend asks "which pixels are the room itself". This stage calls
its cached session rather than loading a second network, which is why adding it
costs no weights, no dependency and no extra download.

Mask2Former was the first choice and is not usable on this machine; `_MASK2FORMER_NOTE`
records exactly why, and `detect_floor_wall(..., backend="mask2former")` raises
it rather than silently substituting something else.

Why this is not simply `surfaces.detect(rgb)["floor"]`
------------------------------------------------------

`surfaces.analyse` takes a plain argmax over all 150 ADE20K classes and then
reads the surface labels out of the result. That is right for its job — deciding
which detections to reject — and it loses floor to a coin-flip wherever the
network splits its confidence. A pixel scoring `floor` 0.31, `rug` 0.22,
`wall` 0.34 is 53% floor by any honest reading, and a plain argmax calls it wall.

Grouping first fixes that: the classes that *are* floor are summed, the classes
that *are* wall are summed, and each is then held against every other class the
network offers. A pixel is floor when floor wins outright, wall when wall wins
outright, and background otherwise. The two masks cannot overlap, because the
comparison that separates them assigns each pixel at most once.

Holding them against the *strongest single* competitor rather than the sum of
all 149 others is what makes "detect the complete visible floor" true: a floor
at 0.40 against a long tail of ten classes at 0.06 is floor, not a tie. A class
that genuinely owns its pixels — ceiling on a ceiling, door on a door — still
wins there, which is how those stay out of both masks without being named.

Measured over the cleaned rooms in `backend/jobs/`, ADE20K spends 93% of a
cleaned room on three labels — wall 0.556, floor 0.268, ceiling 0.107 — so the
two groups below are short on purpose. `rug` and `earth` are carried in the
floor group for the rooms that keep a floor covering; neither appeared in the
sample, and including them cannot cost anything in the rooms where they do not.

What this stage deliberately does not do
----------------------------------------

No erosion, no blur, no morphological opening, no polygon fitting, no colour
thresholding, and no assumption about where in the frame a surface sits or
which way a wall runs. Every boundary comes from the network and the
photograph's own gradients.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import surfaces
from extraction import cleanup

# ---------------------------------------------------------------------------
# The two output classes. There are no others: every remaining ADE20K label —
# ceiling, door, curtain, furniture, all 147 of them — is background, and no
# mask, field or file is produced for any of them.
#
# Short by measurement, not by guesswork: see the module docstring for the
# coverage numbers these were chosen from. A label in neither group can still
# *win* a pixel, and a pixel it wins is left out of both masks, which is what
# "contain only their respective surface" requires. `ceiling` (0.107 mean
# coverage), `curtain` (0.031) and `door` (0.013) are the ones worth naming:
# all three are common in a cleaned room and none of them is floor or wall.

FLOOR_CLASSES: frozenset[str] = frozenset({"floor", "rug", "earth"})

WALL_CLASSES: frozenset[str] = frozenset({"wall"})

# Below this a winning surface is not trusted and the pixel goes to neither
# mask. Deliberately low: the job is to find the *complete* visible surface, and
# a pixel that is 26% floor against a field of 150 classes is still floor. It
# exists only to withhold the genuinely undecided.
MIN_SURFACE_PROBABILITY = 0.25

# Enclosed gaps up to this fraction of a mask are sealed; larger ones are real
# features and are left alone. Same value, for the same reason, as
# `live_scene.SURFACE_HOLE_FRACTION`.
HOLE_FRACTION = 0.02

# Detached components below this fraction of a mask are class-map noise.
SPECK_FRACTION = 0.001

# Colour an overlay paints its mask in, and how strongly. Debug output only —
# the source image is never modified.
FLOOR_OVERLAY_RGB = (46, 168, 255)
WALL_OVERLAY_RGB = (255, 138, 46)
OVERLAY_ALPHA = 0.45

FLOOR_MASK_FILENAME = "FLOOR_MASK.png"
WALL_MASK_FILENAME = "WALL_MASK.png"
FLOOR_OVERLAY_FILENAME = "FLOOR_OVERLAY.png"
WALL_OVERLAY_FILENAME = "WALL_OVERLAY.png"

_MASK2FORMER_NOTE = (
    "Mask2Former needs PyTorch, and PyTorch cannot run on this machine. "
    "2.2.2 is the last release with a macOS x86_64 wheel; it was compiled "
    "against NumPy 1.x and raises 'Numpy is not available' on every array "
    "conversion under the NumPy 2.x this backend runs on, while the installed "
    "opencv-python-headless 5.0 requires numpy>=2. Making Mask2Former work "
    "means downgrading NumPy and OpenCV underneath the whole extraction "
    "pipeline. The default backend is SegFormer-B4 ADE20K, which is the same "
    "ADE20K semantic segmentation this module needs and is already loaded."
)


class FloorWallUnavailable(RuntimeError):
    """The stage could not run. Carries a reason fit to show a user."""


@dataclass(frozen=True)
class FloorWall:
    """One cleaned room's two surfaces."""

    floor: np.ndarray  # (H, W) bool, True where floor
    wall: np.ndarray  # (H, W) bool, True where wall
    shape: tuple[int, int]  # (height, width) of the source image
    stats: dict = field(default_factory=dict)

    @property
    def floor_png(self) -> np.ndarray:
        """The floor mask as white-on-black uint8, ready to write."""
        return (self.floor.astype(np.uint8)) * 255

    @property
    def wall_png(self) -> np.ndarray:
        return (self.wall.astype(np.uint8)) * 255


def _surface_probabilities(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-pixel probability of floor, of wall, and of the best rival to both.

    Returns three (H, W) float32 planes: `floor`, `wall`, and `rival` — the
    highest probability held by any single ADE20K class that is neither. Only
    the first two ever become a mask; `rival` exists to be beaten, and is what
    keeps a ceiling out of the wall mask without the ceiling being an output
    class of this stage.

    The softmax and the grouping both happen at the network's own output
    resolution, and only these three planes are then resized to the photograph.
    Upsampling all 150 classes first — which is what `surfaces.label_map` does,
    correctly, for its own purpose — would cost 1.15 GB on a 1600x1200 room to
    produce planes worth 22 MB. Bilinear interpolation of a probability is the
    same operation as bilinear interpolation of a logit as far as the boundary
    is concerned, and the boundary is refined against the photograph itself
    immediately afterwards.
    """
    session, labels = surfaces._load()

    logits = session.run(None, {session.get_inputs()[0].name: surfaces._preprocess(rgb)})[0][0]

    logits = logits.astype(np.float32)

    # Softmax over the class axis, shifted for numerical stability.
    logits -= logits.max(axis=0, keepdims=True)

    np.exp(logits, out=logits)

    logits /= logits.sum(axis=0, keepdims=True)

    floor_ids = [index for index, label in labels.items() if label in FLOOR_CLASSES]
    wall_ids = [index for index, label in labels.items() if label in WALL_CLASSES]

    if not floor_ids and not wall_ids:
        raise FloorWallUnavailable(
            "The segmentation model's label map has neither a floor nor a wall class; "
            "backend/models/segformer_b4_ade_config.json is not an ADE20K config."
        )

    def grouped(indices: list[int]) -> np.ndarray:
        if not indices:
            return np.zeros(logits.shape[1:], dtype=np.float32)

        return logits[indices].sum(axis=0)

    native = [grouped(floor_ids), grouped(wall_ids)]

    # The strongest class that is neither floor nor wall. Not a group and not
    # an output: one number per pixel, saying how much competition the two
    # output classes actually face there.
    rest = np.delete(logits, floor_ids + wall_ids, axis=0)

    native.append(
        rest.max(axis=0) if rest.size else np.zeros(logits.shape[1:], dtype=np.float32)
    )

    height, width = rgb.shape[:2]

    resized = [cv2.resize(plane, (width, height), interpolation=cv2.INTER_LINEAR) for plane in native]

    return resized[0], resized[1], resized[2]


def _tidy(mask: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    """
    The minimum that removes obvious noise, and nothing beyond it.

    Sealing runs after the snap for the reason `live_scene._clean_surface`
    documents: the guided filter follows the photograph's gradients, and a
    floor's own veining is a gradient, so snapping a clean mask punches a
    scatter of pinholes into it. Tidying last removes what the snap introduced
    as well as what the class map produced.

    Both limits are proportional to the mask, so neither can eat a real
    feature: a gap bigger than `HOLE_FRACTION` of the surface is a doorway or
    an alcove and is left open, and a component bigger than `SPECK_FRACTION` is
    a second stretch of wall seen past a corner, not noise.
    """
    if not mask.any():
        return mask

    area = int(mask.sum())

    snapped = surfaces.snap(mask, rgb)

    sealed = cleanup.fill_small_holes(snapped, max(16, int(area * HOLE_FRACTION)))

    return cleanup.drop_small_components(sealed, max(16, int(area * SPECK_FRACTION)))


# ---------------------------------------------------------------------------
# Boundary refinement.
#
# The class map decides WHAT is floor and wall; this decides WHERE the line
# between them runs. `_tidy`'s guided-filter snap pulls a boundary onto the
# image's edges, but it does so pixel by pixel and follows JPEG noise and floor
# veining as readily as the real junction, so the edge it returns is on target
# and ragged. Measured across 21 rooms, mean roughness (perimeter over the
# perimeter of the same shape lightly smoothed; 1.000 is clean) was 1.043.
#
# A graph cut fixes both halves at once. Its data term is a colour model of
# each side, so brown floor separates from a white skirting board; its pairwise
# term is weighted by image contrast, so the cut settles on the real edge; and
# the same pairwise term charges for every unit of boundary length, so the line
# it chooses is continuous rather than dotted. OpenCV's GrabCut is exactly that
# construction and ships with the OpenCV already installed.
#
# It is only ever allowed to move the boundary, never to redraw the region:
# every pixel farther than `band` from the existing boundary is locked to the
# side it is already on, and the core of the other surface is locked out. So
# the shift is bounded by the band, and a surface cannot grow across into its
# neighbour.
#
# Measured on the same 21 rooms: roughness 1.043 -> 1.020, mean image-edge
# strength along the boundary 358 -> 460, no room rougher and no room's edge
# alignment worse, largest area change 2.7%.

# Half-width of the band a boundary may move within, as a fraction of the
# image's longest edge. Wide enough to reach a skirting line the class map
# missed by a few pixels — 5.5 px on a 540 px room — and narrow enough that the
# cut has no second surface's worth of edges to choose from.
REFINE_BAND_FRACTION = 0.012
REFINE_BAND_MIN_PX = 4
REFINE_BAND_MAX_PX = 24

# GrabCut iterations: colour-model refits. The locked pixels already describe
# each side well, so it settles in two — measured, four bought nothing.
REFINE_ITERATIONS = 2

# The cut runs on tiles of this core size, each with twice the band of context
# on every side. A tile's colour model is then local — the floor under a
# window's light and the floor in shadow are fitted separately — and tiles
# holding no boundary are skipped. Chosen by sweep on six rooms: 768 px tiles
# with 2 iterations gave the smoothest boundaries of every setting tried
# (roughness 1.0166, against 1.0215 for one cut over the whole image) at about
# a third of the whole-image time.
REFINE_TILE_PX = 768

# Widest gap the cut may leave between floor and wall, in pixels, that is closed
# afterwards. Only pixels that were floor or wall before refinement are filled,
# each back to the surface it came from, so this seals a seam without adding
# coverage anywhere.
REFINE_SEAM_PX = 2


def _signed_distance(mask: np.ndarray) -> np.ndarray:
    """Distance to the boundary, positive inside the mask and negative outside."""
    inside = mask.astype(np.uint8)

    return cv2.distanceTransform(inside, cv2.DIST_L2, 5) - cv2.distanceTransform(
        1 - inside, cv2.DIST_L2, 5
    )


def _cut(mask: np.ndarray, bgr: np.ndarray, locked_out: np.ndarray, band: int) -> np.ndarray:
    """
    Graph-cut one mask's boundary within `band` pixels of where it already is.

    Everything outside the band keeps its side; `locked_out` is forced outside.
    A tile whose cut cannot be run — nothing locked on one side, so no colour
    model for it — keeps the input unchanged, which is always a safe answer.
    """
    distance = _signed_distance(mask)

    moving = np.abs(distance) <= band

    labels = np.full(mask.shape, cv2.GC_BGD, dtype=np.uint8)
    labels[distance > band] = cv2.GC_FGD
    labels[moving & mask] = cv2.GC_PR_FGD
    labels[moving & ~mask] = cv2.GC_PR_BGD
    labels[locked_out] = cv2.GC_BGD

    refined = mask.copy()

    height, width = mask.shape
    pad = 2 * band

    def run(top: int, left: int, bottom: int, right: int) -> None:
        """Cut one window and write back its core, [top:bottom, left:right]."""
        if not moving[top:bottom, left:right].any():
            return

        y0, x0 = max(top - pad, 0), max(left - pad, 0)
        y1, x1 = min(bottom + pad, height), min(right + pad, width)

        tile = labels[y0:y1, x0:x1].copy()

        if not (tile == cv2.GC_FGD).any() or not (tile == cv2.GC_BGD).any():
            return

        # GrabCut seeds its colour model with k-means, which is random. A fixed
        # seed makes the same room give the same mask every time.
        cv2.setRNGSeed(0)

        cv2.grabCut(
            np.ascontiguousarray(bgr[y0:y1, x0:x1]),
            tile,
            None,
            np.zeros((1, 65), np.float64),
            np.zeros((1, 65), np.float64),
            REFINE_ITERATIONS,
            cv2.GC_INIT_WITH_MASK,
        )

        core = tile[top - y0 : bottom - y0, left - x0 : right - x0]

        refined[top:bottom, left:right] = (core == cv2.GC_FGD) | (core == cv2.GC_PR_FGD)

    step = REFINE_TILE_PX

    for top in range(0, height, step):
        for left in range(0, width, step):
            run(top, left, min(top + step, height), min(left + step, width))

    # Seams. Two neighbouring tiles cut the same boundary with different
    # context, and can disagree by a pixel or two where they meet — measured,
    # a 2-3 px detached fragment exactly on a tile edge. Each seam is cut again
    # in a strip centred on it, which sees both sides at once, and only the
    # band around the seam is taken from that strip.
    for seam in range(step, width, step):
        for top in range(0, height, step):
            run(top, max(seam - band, 0), min(top + step, height), min(seam + band, width))

    for seam in range(step, height, step):
        for left in range(0, width, step):
            run(max(seam - band, 0), left, min(seam + band, height), min(left + step, width))

    return refined


def _refine_boundaries(
    floor: np.ndarray, wall: np.ndarray, rgb: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Move each boundary onto the real image edge, and make it continuous.

    The floor is cut first with the wall's core locked out, then the wall with
    the refined floor locked out, so the two stay disjoint by construction. A
    3x3 majority vote inside the band then removes single-pixel spurs and
    pinholes — it cannot move a straight edge — and seams of up to
    `REFINE_SEAM_PX` that opened between the two surfaces are closed back to
    whichever surface each pixel belonged to before.
    """
    band = int(np.clip(round(REFINE_BAND_FRACTION * max(rgb.shape[:2])), REFINE_BAND_MIN_PX, REFINE_BAND_MAX_PX))

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    new_floor = _cut(floor, bgr, _signed_distance(wall) > band, band) if floor.any() else floor

    new_wall = _cut(wall, bgr, new_floor, band) if wall.any() else wall

    refined = []

    for new, old in ((new_floor, floor), (new_wall, wall)):
        near = np.abs(_signed_distance(old)) <= band

        majority = cv2.medianBlur(new.astype(np.uint8) * 255, 3) > 127

        refined.append(np.where(near, majority, new))

    new_floor, new_wall = refined

    new_wall &= ~new_floor

    # Seams: pixels that were a surface, are now neither, and sit within a
    # hair of both refined surfaces. Each goes back to what it was.
    seam = (floor | wall) & ~(new_floor | new_wall)

    if seam.any():
        near_floor = cv2.distanceTransform((~new_floor).astype(np.uint8), cv2.DIST_L2, 3) <= REFINE_SEAM_PX
        near_wall = cv2.distanceTransform((~new_wall).astype(np.uint8), cv2.DIST_L2, 3) <= REFINE_SEAM_PX

        seam &= near_floor & near_wall

        new_floor |= seam & floor
        new_wall |= seam & wall & ~new_floor

    # Isolated specks and pinholes the cut left behind, with the same
    # proportional limits `_tidy` uses — so a real opening or a real second
    # stretch of wall is never touched. Filling cannot reach into the other
    # surface: it is removed again below.
    tidied = []

    for mask in (new_floor, new_wall):
        area = int(mask.sum())

        if area:
            mask = cleanup.fill_small_holes(mask, max(16, int(area * HOLE_FRACTION)))
            mask = cleanup.drop_small_components(mask, max(16, int(area * SPECK_FRACTION)))

        tidied.append(mask)

    new_floor, new_wall = tidied

    return new_floor, new_wall & ~new_floor


def detect_floor_wall(
    image: np.ndarray,
    *,
    backend: str = "auto",
    tidy: bool = True,
    refine: bool = True,
) -> FloorWall:
    """
    Detect the floor and the wall in a cleaned room image.

    `image` is an (H, W, 3) uint8 RGB array — the clean room, ideally, since
    that is the image with nothing standing in front of the surfaces. Returns a
    `FloorWall` whose two boolean masks are exactly (H, W), pixel-aligned with
    the input, and disjoint.

    Floor and wall are the only classes this produces. Nothing else is
    detected, stored or returned: a ceiling, a door, a curtain and a chair are
    all simply absent from both masks.

    `backend` is `"auto"` or `"segformer"`, which are the same thing here, or
    `"mask2former"`, which raises `FloorWallUnavailable` explaining why it
    cannot run on this machine. `tidy=False` returns the raw comparison, which
    is what the cleanup is measured against. `refine=False` skips the boundary
    refinement and returns the masks exactly as they were before it existed.
    """
    if backend == "mask2former":
        raise FloorWallUnavailable(_MASK2FORMER_NOTE)

    if backend not in ("auto", "segformer"):
        raise ValueError(f"Unknown backend {backend!r}; expected auto, segformer or mask2former.")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected an (H, W, 3) RGB image; got shape {image.shape}.")

    rgb = np.ascontiguousarray(image, dtype=np.uint8)

    height, width = rgb.shape[:2]

    try:
        floor_p, wall_p, rival_p = _surface_probabilities(rgb)
    except surfaces.SurfacesUnavailable as error:
        raise FloorWallUnavailable(str(error)) from error

    # Two output classes, and a pixel belongs to one of them only by winning it
    # outright: against the other surface, against the strongest class that is
    # neither, and against the confidence floor. Anything else is background —
    # which is every ceiling, door, curtain and piece of furniture in the room,
    # none of which this stage detects, stores or returns.
    floor_wins = floor_p >= wall_p

    floor = floor_wins & (floor_p > rival_p) & (floor_p >= MIN_SURFACE_PROBABILITY)
    wall = ~floor_wins & (wall_p > rival_p) & (wall_p >= MIN_SURFACE_PROBABILITY)

    raw_floor = int(floor.sum())
    raw_wall = int(wall.sum())

    if tidy:
        floor = _tidy(floor, rgb)
        wall = _tidy(wall, rgb)

        # Snapping moves two boundaries independently and can walk them into
        # each other. The same comparison that separated them settles it, so a
        # contested pixel goes to exactly one mask and the two stay disjoint.
        contested = floor & wall

        if contested.any():
            floor &= ~(contested & ~floor_wins)
            wall &= ~(contested & floor_wins)

    refined = tidy and refine

    if refined:
        # Last, so it acts on the finished regions: the class map has decided
        # what is floor and wall, and this only decides where the line runs.
        floor, wall = _refine_boundaries(floor, wall, rgb)

    stats = {
        "backend": "segformer_b4_ade20k_onnx",
        "classes": ["floor", "wall"],
        "source_shape": [int(height), int(width)],
        "floor_pixels": int(floor.sum()),
        "wall_pixels": int(wall.sum()),
        "floor_coverage": round(float(floor.mean()), 5),
        "wall_coverage": round(float(wall.mean()), 5),
        "cleanup_delta": {
            "floor": int(floor.sum()) - raw_floor,
            "wall": int(wall.sum()) - raw_wall,
        },
        "overlap_pixels": int((floor & wall).sum()),
        "tidied": tidy,
        "boundary_refined": refined,
    }

    return FloorWall(floor=floor, wall=wall, shape=(height, width), stats=stats)


def _overlay(rgb: np.ndarray, mask: np.ndarray, colour: tuple[int, int, int]) -> np.ndarray:
    """Tint `mask` over a copy of `rgb`. The source array is never written to."""
    tinted = rgb.astype(np.float32).copy()

    paint = np.array(colour, dtype=np.float32)

    tinted[mask] = (1.0 - OVERLAY_ALPHA) * tinted[mask] + OVERLAY_ALPHA * paint

    return np.clip(tinted, 0, 255).astype(np.uint8)


def write(
    image: np.ndarray,
    result: FloorWall,
    directory: Path,
    *,
    overlays: bool = False,
) -> dict:
    """
    Write the two masks, and optionally the two debug overlays, into `directory`.

    Returns the filenames written and the run's stats, in the shape the HTTP
    layer hands to the UI. Saved through PIL in mode "L", so both files are
    single-channel 0/255 at the source image's own resolution — no palette, no
    alpha, nothing a downstream reader has to interpret.
    """
    directory.mkdir(parents=True, exist_ok=True)

    Image.fromarray(result.floor_png, mode="L").save(directory / FLOOR_MASK_FILENAME)
    Image.fromarray(result.wall_png, mode="L").save(directory / WALL_MASK_FILENAME)

    written = {
        "floor_mask": FLOOR_MASK_FILENAME,
        "wall_mask": WALL_MASK_FILENAME,
    }

    if overlays:
        Image.fromarray(_overlay(image, result.floor, FLOOR_OVERLAY_RGB)).save(
            directory / FLOOR_OVERLAY_FILENAME
        )
        Image.fromarray(_overlay(image, result.wall, WALL_OVERLAY_RGB)).save(
            directory / WALL_OVERLAY_FILENAME
        )

        written["floor_overlay"] = FLOOR_OVERLAY_FILENAME
        written["wall_overlay"] = WALL_OVERLAY_FILENAME

    return {"files": written, "stats": result.stats}






