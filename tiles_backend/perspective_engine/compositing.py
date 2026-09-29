"""
Strict mask clipping: the final say on where a tile pixel may exist.

    mask == 255  ->  tile allowed
    mask == 0    ->  tile forbidden, the room pixel is kept exactly

The migrated renderer already respects its mask: `core.composite.composite`
keeps the base pixel wherever `mask_factor < 0.1` or the camera ray misses the
plane, and a binary 0/255 mask gives a `mask_factor` of exactly 0 or 1. This
module is the check that says so, and the clip that makes it true regardless:
anything outside the allowed mask is put back to the base image before the
result leaves the engine, and the count of pixels that needed it is reported.
The count is 0 on every room tested; the point is that it stays 0 whatever
changes upstream.

The mask is used pixel for pixel. No bounding box, polygon or rectangle is
derived from it anywhere.
"""

from __future__ import annotations

import numpy as np


def clip(rendered: np.ndarray, base: np.ndarray, allowed: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Keep `rendered` inside `allowed`, `base` everywhere else.

    Returns `(image, report)`. `report["outside_before"]` is how many pixels
    the renderer had changed outside the mask — what the clip removed — and
    `report["outside_after"]` is always 0.
    """
    if rendered.shape != base.shape or allowed.shape != base.shape[:2]:
        raise ValueError(
            f"clip: shapes disagree (rendered {rendered.shape}, base {base.shape}, "
            f"mask {allowed.shape})"
        )

    changed = np.any(rendered != base, axis=2)

    outside_before = int((changed & ~allowed).sum())

    image = np.where(allowed[..., None], rendered, base)

    return image, {
        "tiled_pixels": int((changed & allowed).sum()),
        "outside_before": outside_before,
        "outside_after": int((np.any(image != base, axis=2) & ~allowed).sum()),
    }
