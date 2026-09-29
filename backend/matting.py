"""
Turning a binary mask into an edge that survives compositing.

A mask straight out of a segmentation model is binary: every pixel is fully in
or fully out. Cut an object with it and drop it onto a different floor and two
things give the cut away immediately —

  * the boundary staircases, because a diagonal edge quantised to whole pixels
    *is* a staircase, and
  * a one-pixel rim of the old background rides along with the object, because
    the edge pixels of a photo are already a blend of object and background.
    Over a new floor that rim reads as a dark or bright outline.

So two passes here. `refine` re-draws the boundary as a soft alpha that follows
the photo's own edges, using a colour guided filter (He, Sun & Tang) — the same
filter the matting literature uses to lift a trimap into an alpha. `decontaminate`
then replaces the colour of every partially transparent pixel with foreground
colour pulled from inside the object, so what gets blended is the object's own
colour and not the room it was cut from.

Neither step invents detail: the alpha comes from the photo's gradients and the
replacement colour comes from pixels of the same object a few pixels away.
"""

from __future__ import annotations

import cv2
import numpy as np

# Guided-filter window. Larger follows softer edges but bleeds across thin
# features; 4 px is the usual matting default at photo resolution.
RADIUS = 4

# Regularisation. Smaller keeps the alpha closer to the image's own edges.
EPSILON = 1e-4

# How far either side of the binary boundary the alpha is allowed to move.
BAND = 2

# Above this the pixel is treated as pure object, and its colour is kept.
CORE_ALPHA = 0.9


def _box(src: np.ndarray, radius: int) -> np.ndarray:
    size = 2 * radius + 1

    return cv2.boxFilter(src, -1, (size, size), normalize=True, borderType=cv2.BORDER_REFLECT)


def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """
    Colour guided filter: smooth `src` while snapping it to `guide`'s edges.

    `guide` is (H, W, 3) float in [0, 1], `src` is (H, W) float. The colour
    variant is used rather than the grayscale one because an object and the
    surface behind it are often a similar brightness but a different hue, and
    the grayscale filter cannot see that edge at all.
    """
    mean_guide = _box(guide, radius)
    mean_src = _box(src, radius)

    mean_cross = _box(guide * src[..., None], radius)
    covariance = mean_cross - mean_guide * mean_src[..., None]

    products = _box(
        np.stack(
            [
                guide[..., 0] * guide[..., 0],
                guide[..., 0] * guide[..., 1],
                guide[..., 0] * guide[..., 2],
                guide[..., 1] * guide[..., 1],
                guide[..., 1] * guide[..., 2],
                guide[..., 2] * guide[..., 2],
            ],
            axis=-1,
        ),
        radius,
    )

    # Symmetric 3x3 covariance of the guide, per pixel, ridged by eps.
    rr = products[..., 0] - mean_guide[..., 0] * mean_guide[..., 0] + eps
    rg = products[..., 1] - mean_guide[..., 0] * mean_guide[..., 1]
    rb = products[..., 2] - mean_guide[..., 0] * mean_guide[..., 2]
    gg = products[..., 3] - mean_guide[..., 1] * mean_guide[..., 1] + eps
    gb = products[..., 4] - mean_guide[..., 1] * mean_guide[..., 2]
    bb = products[..., 5] - mean_guide[..., 2] * mean_guide[..., 2] + eps

    # Adjugate of that symmetric matrix, so the inverse is one division.
    adj_rr = gg * bb - gb * gb
    adj_rg = rb * gb - rg * bb
    adj_rb = rg * gb - gg * rb
    adj_gg = rr * bb - rb * rb
    adj_gb = rg * rb - rr * gb
    adj_bb = rr * gg - rg * rg

    determinant = rr * adj_rr + rg * adj_rg + rb * adj_rb
    determinant = np.where(np.abs(determinant) < 1e-12, 1e-12, determinant)

    cov_r, cov_g, cov_b = covariance[..., 0], covariance[..., 1], covariance[..., 2]

    a_r = (adj_rr * cov_r + adj_rg * cov_g + adj_rb * cov_b) / determinant
    a_g = (adj_rg * cov_r + adj_gg * cov_g + adj_gb * cov_b) / determinant
    a_b = (adj_rb * cov_r + adj_gb * cov_g + adj_bb * cov_b) / determinant

    b = (
        mean_src
        - a_r * mean_guide[..., 0]
        - a_g * mean_guide[..., 1]
        - a_b * mean_guide[..., 2]
    )

    return (
        _box(a_r, radius) * guide[..., 0]
        + _box(a_g, radius) * guide[..., 1]
        + _box(a_b, radius) * guide[..., 2]
        + _box(b, radius)
    )


def _disk(radius: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def refine(mask: np.ndarray, rgb: np.ndarray, band: int = BAND) -> np.ndarray:
    """
    Lift a binary mask into a soft alpha that follows the photo's edges.

    The filter is free to move the boundary within `band` pixels of where the
    mask put it and nowhere else: well inside stays fully opaque, well outside
    stays fully transparent. That keeps a smooth edge from eating thin parts of
    the object, and keeps the alpha honest about what the segmentation actually
    claimed.
    """
    binary = mask.astype(bool)

    if not binary.any():
        return np.zeros(binary.shape, dtype=np.float32)

    guide = (rgb.astype(np.float32) / 255.0).copy()

    filtered = guided_filter(guide, binary.astype(np.float32), RADIUS, EPSILON)

    alpha = np.clip(filtered, 0.0, 1.0).astype(np.float32)

    solid = binary.astype(np.uint8)

    core = cv2.erode(solid, _disk(band)).astype(bool)
    outer = cv2.dilate(solid, _disk(band)).astype(bool)

    alpha[core] = 1.0
    alpha[~outer] = 0.0

    return alpha


def snap_to_edges(
    coarse: np.ndarray,
    rgb: np.ndarray,
    radius: int,
    band: int,
    eps: float = EPSILON,
) -> np.ndarray:
    """
    Pull a soft, upscaled mask onto the photograph's own edges.

    A SAM mask is decided on a 256x256 logit grid and then interpolated up —
    for a 4000 px photo that is roughly a fifteen-fold enlargement by the time
    it reaches the output. Interpolation cannot invent the detail it never had,
    so the boundary arrives smooth in the wrong places and stepped in others,
    and the existing ±2 px matting pass can only polish whatever edge it is
    handed. This runs first and moves the edge itself.

    It is the same colour guided filter `refine` uses, with a wider window and
    a soft input: the filter's output follows `guide`'s edges wherever the input
    is ambiguous, so a boundary that fell a few pixels off lands back on the
    real one. Nothing is invented — the edge comes from the photo's gradients.

    Two limits keep it honest. Pixels well inside the coarse mask are kept
    whatever the filter says, so a thin chair leg or leaf stem cannot be
    dissolved; and nothing outside `band` pixels of the coarse mask can be
    added, so the mask cannot grow into the floor or wall. The erosion that
    defines "well inside" is deliberately much smaller than the dilation that
    defines the outer limit, because erosion is what destroys thin features.
    """
    binary = coarse >= 0.5

    if not binary.any():
        return binary

    guide = np.ascontiguousarray(rgb.astype(np.float32) / 255.0)

    filtered = guided_filter(guide, coarse.astype(np.float32), radius, eps)

    refined = np.clip(filtered, 0.0, 1.0) >= 0.5

    solid = binary.astype(np.uint8)

    # Asymmetric on purpose: a light erosion to find the confident interior, a
    # full-band dilation to cap outward movement.
    core = cv2.erode(solid, _disk(max(1, band // 3))).astype(bool)
    outer = cv2.dilate(solid, _disk(band)).astype(bool)

    return (refined | core) & outer


def decontaminate(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """
    Replace the colour of partially transparent pixels with foreground colour.

    Every pixel with 0 < alpha < 1 is, in the original photo, a mix of the
    object and whatever was behind it. Compositing it over something new keeps
    that old background in the blend and draws a visible rim, so those pixels
    take the colour of the nearest pixel that is wholly object. Alpha still
    decides how much of the result is shown; only the colour being blended
    changes.

    The fill has to come from inside the object specifically. A general inpaint
    cannot do this — it reads every neighbour it is not told to ignore, and the
    nearest neighbours of a boundary pixel are mostly background, so it
    reconstructs the very blend the rim is made of.
    """
    core = alpha >= CORE_ALPHA

    unknown = (alpha > 0.0) & ~core

    if not unknown.any() or np.count_nonzero(core) < 16:
        return rgb

    # distanceTransformWithLabels measures to the nearest *zero* pixel and, with
    # DIST_LABEL_PIXEL, reports which one that was — labelling the zeros 1..N in
    # raster order, the order np.nonzero returns them in.
    _, labels = cv2.distanceTransformWithLabels(
        (~core).astype(np.uint8),
        cv2.DIST_L2,
        5,
        labelType=cv2.DIST_LABEL_PIXEL,
    )

    ys, xs = np.nonzero(core)

    nearest = np.clip(labels - 1, 0, len(ys) - 1)

    out = rgb.copy()
    out[unknown] = rgb[ys[nearest[unknown]], xs[nearest[unknown]]]

    return out
