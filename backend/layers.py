"""
Composing a combined mask into a full-canvas RGBA layer.

This is the shared back end for the two-layer output — ALL_OBJECTS.png and
MIRRORS_ONLY.png — used by `pipeline_assets` for the calibrated room's committed
masks. Live uploads go through `extraction.output`, which assembles its layers
from per-object alphas instead; both end up writing the same two filenames,
taken from `OUTPUT_FILENAMES` here so the two paths cannot drift apart.

A layer is the size of the original photo and holds its objects at their
original coordinates, so compositing one back over a retiled room is a straight
alpha blend with no placement maths.

Nothing here generates pixels. Every pixel in a layer is a pixel of the original
photograph; the only thing computed is how much of each one is object.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

import matting

# Connected components smaller than this fraction of the photo are specks the
# segmentation left behind, not objects.
MIN_COMPONENT_FRACTION = 2e-5

# Enclosed gaps smaller than this fraction are pinholes inside an object. Real
# gaps — between chair legs, under a table — are larger than this and usually
# open to the outside anyway, so they survive.
MAX_HOLE_FRACTION = 0.002

# Sealing radius for hairline breaks. Deliberately 1: closing only ever adds
# pixels, so a small radius cannot eat a leaf tip or a chair leg, and a large
# one would bridge genuinely separate objects.
CLOSE_RADIUS = 1

# The filename each layer is written as, and the whole set of files a room photo
# produces. Both writers — the live upload in app.py and the committed room in
# pipeline_assets.py — take their names from here, so there is one place that
# decides what lands on disk and no way for the two to drift apart.
OUTPUT_FILENAMES = {
    "all_objects": "ALL_OBJECTS.png",
    "mirrors": "MIRRORS_ONLY.png",
}


@dataclass(frozen=True)
class Layer:
    """One full-canvas RGBA layer, ready to composite."""

    name: str
    rgba: np.ndarray  # (H, W, 4) uint8
    pixels: int  # opaque-equivalent area, alpha summed
    members: list[str] = field(default_factory=list)

    @property
    def size(self) -> tuple[int, int]:
        return int(self.rgba.shape[1]), int(self.rgba.shape[0])

    @property
    def filename(self) -> str:
        return OUTPUT_FILENAMES[self.name]

    def to_image(self) -> Image.Image:
        return Image.fromarray(self.rgba, mode="RGBA")


def _disk(radius: int) -> np.ndarray:
    radius = max(1, int(radius))

    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def _drop_small_components(mask: np.ndarray, minimum: int) -> np.ndarray:
    """Remove isolated specks. Components are whole objects, so this is safe."""
    count, components, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )

    kept = np.zeros(mask.shape, dtype=bool)

    for index in range(1, count):
        if int(stats[index, cv2.CC_STAT_AREA]) >= minimum:
            kept |= components == index

    return kept


def _fill_small_holes(mask: np.ndarray, maximum: int) -> np.ndarray:
    """
    Close pinholes inside objects, leaving real gaps alone.

    A hole is a component of the background that does not reach the border of
    the photo. Filling only the small ones keeps the space between a chair's
    legs open while sealing the speckle a segmentation leaves inside a textured
    cabinet door.
    """
    count, components, stats, _ = cv2.connectedComponentsWithStats(
        (~mask).astype(np.uint8), connectivity=4
    )

    if count <= 1:
        return mask

    touching = set(
        np.unique(
            np.concatenate(
                [
                    components[0, :],
                    components[-1, :],
                    components[:, 0],
                    components[:, -1],
                ]
            )
        ).tolist()
    )

    filled = mask.copy()

    for index in range(1, count):
        if index in touching:
            continue

        if int(stats[index, cv2.CC_STAT_AREA]) <= maximum:
            filled |= components == index

    return filled


def refine_mask(mask: np.ndarray) -> np.ndarray:
    """
    Clean a combined object mask without shrinking anything.

    Closing, speck removal and pinhole filling only. There is no erosion and no
    opening anywhere in here on purpose: both take pixels away, and the pixels
    they take first are exactly the ones worth keeping — leaf tips, lamp flex,
    chair and table legs, the thin inner edge of a mirror frame.
    """
    if not mask.any():
        return mask

    frame = int(mask.shape[0] * mask.shape[1])

    sealed = cv2.morphologyEx(
        mask.astype(np.uint8), cv2.MORPH_CLOSE, _disk(CLOSE_RADIUS)
    ).astype(bool)

    cleaned = _drop_small_components(sealed, max(32, int(frame * MIN_COMPONENT_FRACTION)))

    return _fill_small_holes(cleaned, max(16, int(frame * MAX_HOLE_FRACTION)))


def compose(
    rgb: np.ndarray,
    mask: np.ndarray,
    name: str,
    members: list[str],
    limit: np.ndarray | None = None,
) -> Layer:
    """
    Turn one combined mask into a full-canvas RGBA layer.

    The alpha is the matting pass's: a soft transition confined to a couple of
    pixels either side of the boundary, which is what stops a diagonal edge
    reading as a staircase. Everything well inside is fully opaque and
    everything well outside is fully transparent, and the colour under the
    transparent part is cleared so no trace of the room travels with the layer.

    `limit` caps the result against another layer's alpha. The mirrors are a
    subset of the objects by construction, but the two masks are cleaned and
    matted separately, and a pixel the cleanup adds to one and not the other
    would break that — by a handful of pixels at the edge, which is still a
    mirror showing through where all_objects has nothing. Capping settles it in
    the bytes that actually reach the file.
    """
    height, width = rgb.shape[:2]

    rgba = np.zeros((height, width, 4), dtype=np.uint8)

    if not mask.any():
        return Layer(name=name, rgba=rgba, pixels=0, members=members)

    alpha = matting.refine(mask, rgb)

    colour = matting.decontaminate(rgb, alpha)

    # What ends up in the file is the 8-bit alpha, so that is what decides which
    # pixels are transparent. An alpha of 0.001 rounds to 0 in the PNG; keeping
    # its colour would leave a stray scrap of room behind a pixel nothing can
    # see, which is exactly what "no background in the layer" rules out.
    opacity = np.clip(np.rint(alpha * 255.0), 0, 255).astype(np.uint8)

    if limit is not None:
        opacity = np.minimum(opacity, limit)

    visible = opacity > 0

    rgba[..., :3] = np.where(visible[..., None], colour, 0)
    rgba[..., 3] = opacity

    return Layer(
        name=name,
        rgba=rgba,
        pixels=int(round(float(alpha.sum()))),
        members=members,
    )
