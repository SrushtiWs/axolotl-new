# perspective_engine/camera/metric_scale.py
"""
Metric scale for the floor plane: how many millimetres one spatial unit is.

Stage 7 of the pipeline (mask -> depth -> boundary -> VP -> focal -> plane ->
HERE -> tile grid). Runs on the live /render-tile-full path.

The problem this replaced
-------------------------
The renderer used to convert plane coordinates to millimetres with

    mm_per_spatial_unit = (tile_scale / 30.0) * 8.0     # REMOVED

`tile_scale` was a UI slider and 8/30 a tuning constant. That expression had
no connection to the room: MiDaS returns RELATIVE inverse depth and
renormalises per image, so one spatial unit meant a different physical
distance in every photograph. The slider value that made a 600x1200 tile look
right was measured across four rooms at 10.6, 20.6, 16.0 and 10.6 -- a 2x
spread with no way to predict which. No amount of retuning the constant fixed
that, because there was no single correct value to tune it to. It is gone.

What actually fixes it
----------------------
One real-world length somewhere in the scene. Given any single metric anchor,
the whole floor becomes metric, because the plane fit already knows every
distance up to one unknown factor.

The anchor used here is CAMERA HEIGHT, and it is nearly free. The fitted plane
(a, b, c, d) carries a unit normal, so |d| is exactly the perpendicular
distance from the camera centre to the floor -- in whatever arbitrary units
the depth map used. Setting that distance to a physical height gives

    mm_per_spatial_unit = camera_height_mm / |plane_d|

Every per-image arbitrariness in the depth map cancels in that ratio, which is
the property the old constant lacked.

Camera height is a prior, not a measurement -- a phone held by a standing adult
sits around 1.4-1.6 m, tilted down at the floor. It is an assumption, but a
bounded and physically meaningful one, whereas (tile_scale/30)*8 is an
assumption about nothing. When a stronger anchor is available (a detected door,
whose height is ~2030-2100mm by building convention), pass it as
`reference_*` and it supersedes the height prior.

Why this cannot be reduced to "no scaling factor at all"
--------------------------------------------------------
Absolute scale is not observable from one photograph. A room and a scale model
of that room project to identical images; no amount of vanishing-point or
plane-fitting work recovers the difference, because the information is not in
the pixels. So a conversion from plane units to millimetres MUST exist, and
something physical must set it.

What changed is what sets it. Previously: a slider times 8/30, which is an
assumption about nothing. Now: a stated distance from the camera to the floor,
which is an assumption about something real, is bounded, and is reported in
the render log so it is never silent. That is the honest limit of a monocular
pipeline without a known-size object in frame.
"""

import math
from typing import Optional

# Phone held by a standing adult, angled down at the floor. Not eye height --
# people lower the phone to frame a floor.
DEFAULT_CAMERA_HEIGHT_MM = 1500.0

# Outside this band the recovered geometry is not a person photographing a
# room, so the metric estimate is refused rather than used. Covers a low
# crouch through a raised arm.
MIN_CAMERA_HEIGHT_MM = 700.0
MAX_CAMERA_HEIGHT_MM = 2400.0

# Bounds on the resulting scale factor. Sanity guard only: |plane_d| collapsing
# toward zero (a degenerate fit) would otherwise produce an enormous scale and
# a single tile covering the whole floor.
MIN_MM_PER_UNIT = 1e-3
MAX_MM_PER_UNIT = 1e5

def camera_height_units(plane_d: float) -> float:
    """
    Camera-to-floor distance in depth units.

    For a plane ax+by+cz+d = 0 with a UNIT normal, the perpendicular distance
    from the origin (the camera centre) is |d| directly. The renderer's plane
    fit normalises its normal, so this needs no extra work -- but it does mean
    this function is silently wrong if handed an unnormalised plane, hence the
    guard in mm_per_unit_from_camera_height.
    """
    return abs(float(plane_d))


def mm_per_unit_from_camera_height(
    plane,
    camera_height_mm: float = DEFAULT_CAMERA_HEIGHT_MM,
):
    """
    Metric scale from the camera-height anchor. Returns (mm_per_unit, info),
    with mm_per_unit None when the plane cannot support the conversion.
    """
    info = {"stage": "start", "source": "camera_height"}

    if plane is None or len(plane) < 4:
        info["stage"] = "no-plane"
        return None, info

    a, b, c, d = (float(v) for v in plane[:4])

    normal_len = math.sqrt(a * a + b * b + c * c)
    if not math.isfinite(normal_len) or abs(normal_len - 1.0) > 1e-3:
        # |d| is only the camera-floor distance for a unit normal. Rescale
        # rather than refuse, since the relationship still holds after
        # normalising both sides.
        if normal_len < 1e-9:
            info["stage"] = "degenerate-normal"
            return None, info
        d = d / normal_len
        info["renormalised"] = True

    height_units = camera_height_units(d)
    if height_units < 1e-6:
        info["stage"] = "camera-on-the-floor"
        return None, info

    if not (MIN_CAMERA_HEIGHT_MM <= camera_height_mm <= MAX_CAMERA_HEIGHT_MM):
        info["stage"] = "implausible-camera-height"
        info["camera_height_mm"] = camera_height_mm
        return None, info

    mm_per_unit = camera_height_mm / height_units

    if not (MIN_MM_PER_UNIT <= mm_per_unit <= MAX_MM_PER_UNIT):
        info["stage"] = "scale-out-of-range"
        info["mm_per_unit"] = mm_per_unit
        return None, info

    info["stage"] = "ok"
    info["camera_height_units"] = height_units
    info["camera_height_mm"] = camera_height_mm
    info["mm_per_unit"] = mm_per_unit
    return mm_per_unit, info


def mm_per_unit_from_reference(reference_length_units: float,
                               reference_length_mm: float):
    """
    Metric scale from any measured object of known size, e.g. a detected door
    (~2100mm tall by building convention).

    Strictly better than the camera-height prior when available, because the
    reference is a real object in the scene rather than an assumption about
    the photographer. Wired here so door detection has somewhere to plug in.
    """
    info = {"stage": "start", "source": "reference"}

    if reference_length_units is None or reference_length_units < 1e-6:
        info["stage"] = "no-reference"
        return None, info
    if reference_length_mm is None or reference_length_mm <= 0:
        info["stage"] = "bad-reference-mm"
        return None, info

    mm_per_unit = float(reference_length_mm) / float(reference_length_units)
    if not (MIN_MM_PER_UNIT <= mm_per_unit <= MAX_MM_PER_UNIT):
        info["stage"] = "scale-out-of-range"
        info["mm_per_unit"] = mm_per_unit
        return None, info

    info["stage"] = "ok"
    info["mm_per_unit"] = mm_per_unit
    return mm_per_unit, info


class MetricScaleError(ValueError):
    """Raised when no physical anchor can fix the millimetre scale."""


def resolve_mm_per_unit(
    plane=None,
    camera_height_mm: float = DEFAULT_CAMERA_HEIGHT_MM,
    reference_length_units: Optional[float] = None,
    reference_length_mm: Optional[float] = None,
):
    """
    Millimetres per plane unit, from a physical anchor. No mode, no slider.

    Anchor order, strongest first:
      1. a measured object of known size (reference_length_*), when supplied
      2. the camera-height prior

    Raises MetricScaleError when neither can be applied, rather than falling
    back to a made-up number. There is no longer any arbitrary constant to
    fall back TO, and inventing one silently is precisely what this module
    exists to stop -- a render at a fabricated scale is worse than a refused
    one, because it looks authoritative.
    """
    info = {"attempts": {}}

    if reference_length_units is not None and reference_length_mm is not None:
        mm, ref_info = mm_per_unit_from_reference(reference_length_units,
                                                  reference_length_mm)
        info["attempts"]["reference"] = ref_info
        if mm is not None:
            info["source"] = "reference"
            info["mm_per_unit"] = mm
            return mm, info

    mm, ch_info = mm_per_unit_from_camera_height(plane, camera_height_mm)
    info["attempts"]["camera_height"] = ch_info
    if mm is not None:
        info["source"] = "camera_height"
        info["mm_per_unit"] = mm
        info["camera_height_mm"] = camera_height_mm
        info["camera_height_units"] = ch_info.get("camera_height_units")
        return mm, info

    raise MetricScaleError(
        "metric scale: no usable anchor. camera-height anchor failed with "
        f"stage='{ch_info.get('stage')}' (camera_height_mm={camera_height_mm}, "
        f"plane={plane}). Absolute scale cannot be recovered from a single "
        "image without one physical reference, so the render is refused "
        "rather than guessed."
    )
