"""
The two clipping masks, read and made ready for the renderer.

`perspective_engine` does not detect anything. It consumes `FLOOR_MASK.png` and
`WALL_MASK.png` exactly as `floor_wall.py` wrote them, and this module is the
only place those files are interpreted:

    mask == 255  ->  tile allowed
    mask == 0    ->  tile forbidden

Anything that is not exactly 0 or 255 is refused rather than guessed at. The
stage writes strict binary masks, so a grey value means the file came from
somewhere else, and thresholding it would quietly redraw the boundary.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from floor_wall import FLOOR_MASK_FILENAME, WALL_MASK_FILENAME


class MaskError(ValueError):
    """A mask cannot be used as a clipping boundary. Carries a reason fit to show."""


def _binary(mask: np.ndarray, name: str) -> np.ndarray:
    """A 0/255 mask, or an already-boolean one, as bool. Anything else is refused."""
    if mask.ndim != 2:
        raise MaskError(f"{name} must be single-channel; got shape {mask.shape}.")

    if mask.dtype == bool:
        return mask

    values = np.unique(mask)

    if not set(values.tolist()) <= {0, 255}:
        raise MaskError(
            f"{name} is not a strict binary mask: it holds values other than 0 and 255 "
            f"({values[:8].tolist()}...)."
        )

    return mask == 255


def load(directory: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """
    `FLOOR_MASK.png` and `WALL_MASK.png` from a job folder, as bool arrays.

    `None` when either file is missing, so a caller can fall back to what it
    did before the masks existed. A file that exists but is not a usable mask
    raises instead: that is a broken stage, not an absent one.
    """
    floor_path = directory / FLOOR_MASK_FILENAME
    wall_path = directory / WALL_MASK_FILENAME

    if not floor_path.exists() or not wall_path.exists():
        return None

    floor = _binary(np.asarray(Image.open(floor_path)), FLOOR_MASK_FILENAME)
    wall = _binary(np.asarray(Image.open(wall_path)), WALL_MASK_FILENAME)

    if floor.shape != wall.shape:
        raise MaskError(
            f"{FLOOR_MASK_FILENAME} is {floor.shape} but {WALL_MASK_FILENAME} is "
            f"{wall.shape}; both must be the size of the same clean room."
        )

    return floor, wall


def _to_shape(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """
    A boolean mask at another resolution.

    Resized as a float and thresholded at 0.5, never nearest-neighbour — the
    same rule `app._to_shape` and `extraction.to_original` use, so the boundary
    lands where it would have if the mask had been decided at this size.
    """
    if mask.shape == shape:
        return mask

    resized = cv2.resize(
        mask.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR
    )

    return resized >= 0.5


def prepare(
    floor: np.ndarray,
    wall: np.ndarray,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Both masks at the render resolution, strictly disjoint.

    Returns `(floor, wall, contested)`. The two masks arrive disjoint, and stay
    so at their own size; resizing can in principle leave a pixel exactly on
    both boundaries at another size, and such a pixel is given to the floor
    alone. `contested` counts them so a caller can see it happened — it is 0
    whenever the render runs at the masks' own resolution.
    """
    floor = _to_shape(_binary(floor, "floor mask"), shape)
    wall = _to_shape(_binary(wall, "wall mask"), shape)

    contested = floor & wall

    return floor, wall & ~contested, int(contested.sum())
