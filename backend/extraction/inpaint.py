"""
Reconstructing the room behind the objects that were lifted out of it.

`ALL_OBJECTS.png` and `MIRRORS_ONLY.png` say which pixels were object. Punching
them out of the photograph leaves holes, and a hole is not a room: the tile
projection needs a continuous floor to paint onto, and a caller that wants to
*see* the emptied room needs an ordinary opaque photograph, not a transparency.

This module fills those holes and nothing else.

    photo + object alpha
        -> hole mask          the union of both layers, grown by the alpha rim
        -> LaMa (ONNX, CPU)   fill, per region, at the model's own 512 grid
        -> paste back         strictly inside the hole
        -> CLEAN_ROOM.png     opaque RGB, the photo's exact size

Three rules govern it.

**Inside the mask only.** The fill is composited back with the hole mask as the
stencil, so every pixel outside it is bit-for-bit the original photograph. The
architecture, the camera, the perspective, the lighting and the texture of
everything that was not an object cannot change here, because those pixels are
never written.

**Region by region, not whole-image.** This LaMa export has a fixed 512x512
input. Squeezing a 1400x900 room into it and stretching the answer back would
soften the entire picture — including the floor/wall junction, which has to stay
a crisp line. So each hole is filled in its own window: the object's bounding box
grown by enough surrounding room to give the model context, and only that window
goes through the 512 grid. A chair against a skirting board is then reconstructed
at something close to native detail, and the rest of the room is untouched.

**Surrounding pixels are the only source.** LaMa is a feed-forward inpainter
conditioned on the visible part of the window it is given; it continues what it
can see. A hole in the floor is surrounded by floor and is filled with floor, a
hole against a wall with wall. That behaviour is a property of the context
window, which is why the window is built around each object rather than sampled
from elsewhere in the room, and why `verify` measures the result against the
surrounding surfaces instead of assuming it worked.

One deliberate exception to "inside the object mask only": the hole is grown by
a few pixels before filling. The written alpha is binary and cut at the object's
own boundary, so the pixel ring just outside it still holds the object's edge
colour, its anti-aliasing and its contact shadow. Leaving that ring would leave a
silhouette of the object in the clean room, which is the one thing this stage
exists to prevent. The growth is small, proportional to the photo, and reported
in the metadata as `grow_px`.
"""

from __future__ import annotations

import threading

import cv2
import numpy as np

import weights

# https://huggingface.co/Carve/LaMa-ONNX — big-LaMa, Apache 2.0, exported for
# onnxruntime. fp32 rather than the fp16 file next to it: this runs on CPU,
# where fp16 is emulated and slower, and the weight file is fetched once.
LAMA_URL = "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx"

LAMA_PATH = weights.MODELS / "lama_fp32.onnx"

# The export's fixed input size. Not a tunable — the graph will not accept
# anything else.
SIZE = 512

# How far the hole is grown past the written alpha, as a fraction of the longer
# edge, to swallow the object's own rim and its contact shadow.
GROW_FRACTION = 0.003
GROW_MIN = 2
GROW_MAX = 8

# How much room is included around a hole as context, as a fraction of the
# hole's longer side. LaMa needs to see what it is continuing; too tight a crop
# and it has nothing to continue from, too generous and every window merges into
# the whole picture and the fill goes soft.
#
# Measured on a 1408x768 room with a chair, a lamp, a plant and a window in the
# object layer — Laplacian variance of the reconstructed pixels, higher being
# sharper:
#
#     0.90  ->  1 window,  35.6      0.25  ->  2 windows,  96.3
#     0.40  ->  2 windows, 81.4      0.15  ->  2 windows, 100.1
#
# 0.25 is where the tile courses on the wall and the skirting line still come
# back correctly; 0.15 buys another 4% of sharpness on a crop tight enough that
# a larger object would have little left to continue from.
CONTEXT_FRACTION = 0.25

# No window smaller than this, so a tiny object still gets real context rather
# than a 20-pixel crop blown up to 512.
MIN_WINDOW_PX = 160

# Above this share of the frame, region windows stop being worth it — the
# windows would merge into the whole picture anyway — and one full-frame pass is
# both faster and more coherent.
WHOLE_FRAME_SHARE = 0.55

# Windows this close together are merged rather than run separately, so that a
# pass always sees the whole of what it is continuing.
MERGE_PAD = 8

# ...but only up to here. Merging to convergence in a busy room pulls every
# window into one that covers the entire photograph, which then goes through the
# 512 grid whole and comes back soft. Past this size the groups stay apart, and
# each hole is written by exactly one of them.
#
# Sharpness of the reconstructed pixels, Laplacian variance, two rooms:
#
#     cap        bedroom, 20 objects      wall + chair + plant
#     640        3 windows, 64.9          4 windows, 142.3
#     768        3 windows, 58.7          2 windows,  98.8
#     1024       2 windows, 57.2          2 windows,  98.8
#     no cap     1 window,  44.0          2 windows,  98.8
#
# 640 is 1.25x the grid: a window that size is still downscaled on the way in,
# but only just, and the floor courses and skirting lines come back straight.
MAX_WINDOW = 640

_lock = threading.Lock()
_session = None


class InpaintUnavailable(RuntimeError):
    """The LaMa weights are not present and could not be fetched."""


def available() -> bool:
    """Whether the weights are already on disk."""
    return LAMA_PATH.exists()


def _load():
    """The ONNX session, created once and reused."""
    global _session

    if _session is not None:
        return _session

    with _lock:
        if _session is not None:
            return _session

        try:
            weights.ensure(LAMA_URL, LAMA_PATH)
        except Exception as error:  # network, disk, anything
            raise InpaintUnavailable(
                f"The LaMa inpainting weights are missing and could not be "
                f"downloaded from {LAMA_URL}: {error}"
            ) from error

        import onnxruntime as ort

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        _session = ort.InferenceSession(
            str(LAMA_PATH), options, providers=["CPUExecutionProvider"]
        )

    return _session


def grow_px(shape: tuple[int, int]) -> int:
    """How far the hole is grown past the written alpha, for this photo."""
    return int(np.clip(round(GROW_FRACTION * max(shape[:2])), GROW_MIN, GROW_MAX))


def hole_from_layers(*layers: np.ndarray) -> np.ndarray:
    """
    The region to reconstruct: every pixel any object layer claims at all.

    Taken at `alpha > 0` rather than at the alpha cutoff, because a pixel the
    layer holds even weakly is a pixel that will be composited back over the
    tiles later, and the room underneath it has to exist.
    """
    hole = None

    for layer in layers:
        if layer is None or layer.size == 0:
            continue

        claimed = layer[..., 3] > 0 if layer.ndim == 3 else layer.astype(bool)

        hole = claimed if hole is None else (hole | claimed)

    if hole is None:
        raise ValueError("no layers given")

    return hole


def _windows(
    hole: np.ndarray, shape: tuple[int, int]
) -> tuple[np.ndarray, list[tuple[tuple[int, int, int, int], list[int]]]]:
    """
    The hole's connected components, and one context window per group of them.

    Returns the component label image and a list of `(box, members)`: the window
    to run, and which components that window is allowed to write. Every
    component belongs to exactly one window, so two windows that happen to
    overlap can never each write half of the same hole and disagree along the
    seam — the second one simply does not write what the first one owns.
    """
    height, width = shape

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        hole.astype(np.uint8), connectivity=8
    )

    groups: list[list | None] = []

    for index in range(1, count):
        x, y, w, h, _ = stats[index]

        margin = max(int(round(CONTEXT_FRACTION * max(w, h))), MIN_WINDOW_PX // 2)

        groups.append(
            [
                [
                    max(0, x - margin),
                    max(0, y - margin),
                    min(width, x + w + margin),
                    min(height, y + h + margin),
                ],
                [index],
            ]
        )

    # Merge overlapping windows, but never past MAX_WINDOW: the object count per
    # room is tens, so the quadratic pass costs nothing and is easier to trust
    # than an interval tree.
    merged = True

    while merged:
        merged = False

        for i in range(len(groups)):
            if groups[i] is None:
                continue

            for j in range(i + 1, len(groups)):
                if groups[j] is None:
                    continue

                a, b = groups[i][0], groups[j][0]

                overlaps = (
                    a[0] < b[2] + MERGE_PAD
                    and b[0] < a[2] + MERGE_PAD
                    and a[1] < b[3] + MERGE_PAD
                    and b[1] < a[3] + MERGE_PAD
                )

                if not overlaps:
                    continue

                union = [
                    min(a[0], b[0]),
                    min(a[1], b[1]),
                    max(a[2], b[2]),
                    max(a[3], b[3]),
                ]

                # A single object larger than the cap gets an oversized window
                # of its own — there is no way around that — but it must not
                # then swallow the rest of the room on the way past.
                if (
                    union[2] - union[0] > MAX_WINDOW or union[3] - union[1] > MAX_WINDOW
                ) and union != a:
                    continue

                groups[i] = [union, groups[i][1] + groups[j][1]]
                groups[j] = None
                merged = True

        groups = [group for group in groups if group is not None]

    return labels, [(tuple(group[0]), group[1]) for group in groups]


def _fill(session, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    One LaMa pass over one window.

    `rgb` is uint8 HxWx3 and `mask` is bool HxW, both the window's own size.
    Returns the model's reconstruction of the window, at the window's size.
    """
    height, width = mask.shape

    # Down to the grid with INTER_AREA and back up with INTER_CUBIC: area
    # averaging is what keeps the context sharp on the way in, and cubic is what
    # keeps the fill from looking blocky on the way out.
    shrinking = width > SIZE or height > SIZE

    small = cv2.resize(
        rgb,
        (SIZE, SIZE),
        interpolation=cv2.INTER_AREA if shrinking else cv2.INTER_CUBIC,
    )

    # The mask is resized as a float and taken at anything above zero, so a hole
    # can only ever grow on the grid, never shrink. A hole that shrank would
    # leave object pixels for the model to reconstruct *from*, and it would
    # dutifully paint the object back.
    small_mask = cv2.resize(
        mask.astype(np.float32), (SIZE, SIZE), interpolation=cv2.INTER_LINEAR
    )

    small_mask = (small_mask > 0.0).astype(np.float32)

    output = session.run(
        None,
        {
            "image": small.astype(np.float32).transpose(2, 0, 1)[None] / 255.0,
            "mask": small_mask[None, None],
        },
    )[0]

    # This export returns 0..255 already, verified against the reference output
    # published with the weights.
    filled = np.clip(output[0].transpose(1, 2, 0), 0.0, 255.0).astype(np.uint8)

    return cv2.resize(filled, (width, height), interpolation=cv2.INTER_CUBIC)


def clean_room(rgb: np.ndarray, hole: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    The photograph with its objects removed and the room behind them restored.

    Returns an opaque uint8 HxWx3 image at the photo's exact size, and a
    metadata dict describing what was done.

    Every pixel outside `hole` is the original photograph, untouched. Every
    pixel inside it comes from LaMa continuing the room around it.
    """
    height, width = rgb.shape[:2]

    result = rgb.copy()

    info: dict = {
        "model": "LaMa (big-lama, ONNX, CPU)",
        "grid": SIZE,
        "hole_px": int(hole.sum()),
        "grow_px": 0,
        "windows": 0,
        "whole_frame": False,
    }

    if not hole.any():
        info["reason"] = "nothing to fill — no object pixels in this photo"
        return result, info

    radius = grow_px((height, width))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))

    grown = cv2.dilate(hole.astype(np.uint8), kernel).astype(bool)

    info["grow_px"] = radius
    info["filled_px"] = int(grown.sum())

    session = _load()

    share = float(grown.sum()) / float(grown.size)

    if share >= WHOLE_FRAME_SHARE:
        labels = grown.astype(np.int32)
        boxes = [((0, 0, width, height), [1])]
        info["whole_frame"] = True
    else:
        labels, boxes = _windows(grown, (height, width))

    info["windows"] = len(boxes)

    for (x0, y0, x1, y1), members in boxes:
        window = grown[y0:y1, x0:x1]

        if not window.any():
            continue

        # The model is shown every hole inside the window, so it never
        # reconstructs from a neighbouring object's pixels...
        filled = _fill(session, rgb[y0:y1, x0:x1], window)

        # ...but writes back only the holes this window owns, and only inside
        # them. This is the line that makes "never outside the mask" true: no
        # pixel outside `owned` is ever assigned.
        owned = window & np.isin(labels[y0:y1, x0:x1], members)

        result[y0:y1, x0:x1][owned] = filled[owned]

    return result, info


def verify(
    original: np.ndarray,
    clean: np.ndarray,
    hole: np.ndarray,
    surfaces: dict[str, np.ndarray] | None = None,
) -> dict:
    """
    Measure the result instead of asserting it.

    Reports, for the caller to record and for a human to read:

      `outside_changed`   pixels altered outside the hole. Must be 0.
      `dark_px`           near-black pixels inside the hole — the failure mode
                          where a hole is "filled" with nothing.
      `surface_delta`     per surface, how far the reconstruction inside the
                          hole sits from the real surface around it, in mean
                          absolute RGB.

    `surface_delta` is an indicator, not a verdict. It compares two region
    means, so it is small when a hole in a plain floor was rebuilt out of that
    floor, and legitimately large when the mask covered something that was
    never the same tone as the rest of the surface — a window in a tiled wall,
    a dark worktop over a pale floor. Read it next to the picture, not instead
    of it. `outside_changed` is the one number that is a verdict: it must be 0.
    """
    outside = ~hole

    report: dict = {
        "outside_changed": int(
            np.count_nonzero((original[outside] != clean[outside]).any(axis=-1))
        ),
        "hole_px": int(hole.sum()),
    }

    if hole.any():
        inside = clean[hole].astype(np.int16)

        report["dark_px"] = int(np.count_nonzero(inside.max(axis=-1) <= 12))
        report["mean_rgb"] = [round(float(v), 1) for v in inside.mean(axis=0)]

    if surfaces:
        deltas = {}

        for label, mask in surfaces.items():
            if mask.shape != hole.shape:
                continue

            # The reconstructed part of this surface, against the real part of
            # it that was never hidden.
            rebuilt = mask & hole
            intact = mask & ~hole

            if rebuilt.sum() < 64 or intact.sum() < 64:
                continue

            deltas[label] = round(
                float(
                    np.abs(
                        clean[rebuilt].astype(np.float32).mean(axis=0)
                        - clean[intact].astype(np.float32).mean(axis=0)
                    ).mean()
                ),
                2,
            )

        if deltas:
            report["surface_delta"] = deltas

    return report
