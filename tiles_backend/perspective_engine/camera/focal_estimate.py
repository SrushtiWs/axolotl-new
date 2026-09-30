# perspective_engine/camera/focal_estimate.py
"""
Automatic focal-length estimation.

Stage 5 of the pipeline (mask -> depth -> boundary -> VP -> HERE -> plane ->
metric scale -> tile grid). Runs on the live /render-tile-full path.

Why a fixed focal length is wrong
---------------------------------
The renderer previously took focal length as a constant the UI happened to
send (1500 px). There is no such constant. A phone's focal length in pixels
depends on which physical lens fired and on the delivered resolution:

    f_px = f_35mm_equivalent * image_width / 36.0

so the same handset spans roughly 0.36*W (ultra-wide, ~13mm equiv) to 1.4*W
(tele, ~50mm equiv), and a 2x crop-zoom shot changes it again without changing
lenses. Feeding the wrong f distorts every downstream quantity: the
back-projected point cloud is sheared, the fitted floor plane tilts to
compensate, and the tile grid inherits both errors as a perspective that does
not match the photograph.

Estimation strategy, in falling order of trustworthiness
--------------------------------------------------------
1. EXIF. When the photo carries FocalLengthIn35mmFilm (most phone captures
   do), this is measured metadata from the device, not an inference. Nothing
   computed from image content beats it, so it wins outright.

2. Two-vanishing-point calibration. Two orthogonal floor directions with a
   principal point give f in closed form. Only usable when both VPs are
   finite and reasonably close in; see estimate_focal_from_vps for why that
   restriction is not negotiable.

3. Field-of-view prior. When neither is available, a phone-camera prior beats
   a hardcoded pixel count because it at least scales with image width.

Every path is gated on a plausible-FOV range, so a numerically valid but
physically absurd answer is rejected rather than propagated.
"""

import io
import math
from typing import Optional, Tuple

import numpy as np

from ..core.memo import room_memo

# Plausible focal range as a multiple of image width. 0.30*W is about a 118
# degree horizontal field of view (wider than any phone ultra-wide);
# 4.0*W is about 14 degrees (longer than any phone tele). Anything outside is
# an estimation failure, not an unusual camera.
MIN_FOCAL_WIDTH_RATIO = 0.30
MAX_FOCAL_WIDTH_RATIO = 4.00

# Default prior: 26mm-equivalent, the standard main camera on essentially
# every phone of the last decade. f = W * 26 / 36.
DEFAULT_FOCAL_WIDTH_RATIO = 26.0 / 36.0

# A vanishing point this far from the principal point (in image widths) makes
# the two-VP solve numerically hopeless -- see estimate_focal_from_vps.
MAX_VP_DISTANCE_WIDTHS = 8.0

# 35mm full-frame sensor width in mm, the reference for "35mm equivalent".
FULL_FRAME_WIDTH_MM = 36.0


def focal_range_for_width(image_width: int) -> Tuple[float, float]:
    return (MIN_FOCAL_WIDTH_RATIO * image_width,
            MAX_FOCAL_WIDTH_RATIO * image_width)


def is_plausible_focal(f: Optional[float], image_width: int) -> bool:
    if f is None or not np.isfinite(f) or f <= 0:
        return False
    lo, hi = focal_range_for_width(image_width)
    return lo <= f <= hi


def focal_to_hfov_degrees(f: float, image_width: int) -> float:
    return math.degrees(2.0 * math.atan(image_width / (2.0 * max(f, 1e-6))))


@room_memo  # per photo: a tile change must not re-read the camera
def estimate_focal_from_exif(image_bytes: bytes, image_width: int):
    """
    Focal length in pixels from the photo's own EXIF, or (None, info).

    Prefers FocalLengthIn35mmFilm because it already folds in the sensor size;
    the raw FocalLength tag is in millimetres on an unknown sensor and is
    useless on its own. Note the estimate is against the CURRENT image width,
    so a resized upload stays correct while a cropped one does not -- cropping
    changes the field of view without touching the tag.
    """
    info = {"stage": "start", "source": "exif"}

    if not image_bytes:
        info["stage"] = "no-bytes"
        return None, info

    try:
        from PIL import Image, ExifTags
    except ImportError:
        info["stage"] = "pillow-unavailable"
        return None, info

    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            exif = img.getexif()
            if exif is None or len(exif) == 0:
                info["stage"] = "no-exif"
                return None, info

            tag_ids = {name: tid for tid, name in ExifTags.TAGS.items()}
            f35 = exif.get(tag_ids.get("FocalLengthIn35mmFilm"))

            if f35 is None:
                # Some encoders park it in the Exif IFD instead of IFD0.
                try:
                    exif_ifd = exif.get_ifd(0x8769)
                    f35 = exif_ifd.get(tag_ids.get("FocalLengthIn35mmFilm"))
                except (KeyError, AttributeError):
                    f35 = None

            if f35 is None:
                info["stage"] = "no-focal-tag"
                return None, info

            f35 = float(f35)
            if f35 <= 0:
                info["stage"] = "bad-focal-tag"
                return None, info
    except Exception as exc:                      # noqa: BLE001 - EXIF is best-effort
        info["stage"] = f"exif-read-failed: {type(exc).__name__}"
        return None, info

    f_px = image_width * f35 / FULL_FRAME_WIDTH_MM
    info["focal_35mm"] = f35

    if not is_plausible_focal(f_px, image_width):
        info["stage"] = "implausible"
        info["focal_px"] = f_px
        return None, info

    info["stage"] = "ok"
    info["focal_px"] = f_px
    info["hfov_deg"] = focal_to_hfov_degrees(f_px, image_width)
    return f_px, info


def estimate_focal_from_vps(vp1, vp2, principal_point, image_width: int):
    """
    Closed-form focal length from two ORTHOGONAL vanishing points.

    Two perpendicular world directions project to VPs whose camera-frame rays
    must be perpendicular. With c the principal point, that is

        (vp1 - c) . (vp2 - c) + f^2 = 0
        f = sqrt( -[ (v1x-cx)(v2x-cx) + (v1y-cy)(v2y-cy) ] )

    Two conditions are enforced rather than assumed:

    Negative dot product. A positive dot means the two VPs sit on the same
    side of the principal point, which no pair of perpendicular directions can
    produce under a pinhole camera. It signals that the "orthogonal" pair is
    not orthogonal -- typically both clusters landed on the same floor
    direction, or one is a carpet pattern. There is no f that satisfies it.

    Bounded VP distance. The sensitivity of f to VP position grows without
    limit as the VPs recede: near a frontal view one floor direction runs to
    infinity, and a few pixels of line noise then swing f by a factor of
    several. The estimate stays formally valid and becomes worthless, so it is
    refused past MAX_VP_DISTANCE_WIDTHS.
    """
    info = {"stage": "start", "source": "vp"}

    if vp1 is None or vp2 is None:
        info["stage"] = "missing-vp"
        return None, info

    cx, cy = principal_point
    v1 = np.asarray(vp1, dtype=np.float64)[:2]
    v2 = np.asarray(vp2, dtype=np.float64)[:2]

    if not (np.all(np.isfinite(v1)) and np.all(np.isfinite(v2))):
        info["stage"] = "non-finite-vp"
        return None, info

    d1 = v1 - np.array([cx, cy])
    d2 = v2 - np.array([cx, cy])

    r1 = float(np.linalg.norm(d1)) / max(image_width, 1)
    r2 = float(np.linalg.norm(d2)) / max(image_width, 1)
    info["vp1_distance_widths"] = r1
    info["vp2_distance_widths"] = r2

    if max(r1, r2) > MAX_VP_DISTANCE_WIDTHS:
        info["stage"] = "vp-too-far-ill-conditioned"
        return None, info

    dot = float(np.dot(d1, d2))
    info["dot"] = dot

    if dot >= -1e-6:
        info["stage"] = "non-orthogonal-vps"
        return None, info

    f_px = math.sqrt(-dot)
    info["focal_px"] = f_px

    if not is_plausible_focal(f_px, image_width):
        info["stage"] = "implausible"
        return None, info

    info["stage"] = "ok"
    info["hfov_deg"] = focal_to_hfov_degrees(f_px, image_width)
    return f_px, info


def focal_from_fov_prior(image_width: int,
                         ratio: float = DEFAULT_FOCAL_WIDTH_RATIO):
    """Phone-camera prior. Scales with image width, unlike a fixed constant."""
    f_px = ratio * image_width
    return f_px, {
        "stage": "ok",
        "source": "prior",
        "focal_px": f_px,
        "hfov_deg": focal_to_hfov_degrees(f_px, image_width),
    }


def resolve_focal_length(
    image_width: int,
    exif_focal_px: Optional[float] = None,
    vp1=None,
    vp2=None,
    principal_point=None,
    fallback_focal: Optional[float] = None,
    auto: bool = True,
):
    """
    Pick the focal length, EXIF > two-VP > prior > caller's value.

    Returns (focal_px, info) and always yields a usable number. `info` carries
    `source` so the caller can log which rung of the ladder actually answered,
    and `agreement_ratio` when EXIF and the VP solve were both available --
    that ratio is the only independent check available on either method, so it
    is reported even though EXIF still wins.
    """
    info = {"source": "none", "candidates": {}}

    if not auto:
        f = fallback_focal if fallback_focal else DEFAULT_FOCAL_WIDTH_RATIO * image_width
        info["source"] = "manual"
        info["focal_px"] = float(f)
        info["hfov_deg"] = focal_to_hfov_degrees(f, image_width)
        return float(f), info

    f_vp = None
    if vp1 is not None and vp2 is not None and principal_point is not None:
        f_vp, vp_info = estimate_focal_from_vps(vp1, vp2, principal_point, image_width)
        info["candidates"]["vp"] = vp_info

    f_exif = exif_focal_px if is_plausible_focal(exif_focal_px, image_width) else None
    if exif_focal_px is not None:
        info["candidates"]["exif"] = {
            "focal_px": float(exif_focal_px),
            "accepted": f_exif is not None,
        }

    if f_exif is not None and f_vp is not None:
        info["agreement_ratio"] = float(max(f_exif, f_vp) / max(min(f_exif, f_vp), 1e-6))

    if f_exif is not None:
        chosen, source = f_exif, "exif"
    elif f_vp is not None:
        chosen, source = f_vp, "vp"
    elif is_plausible_focal(fallback_focal, image_width):
        chosen, source = float(fallback_focal), "caller"
    else:
        chosen, prior_info = focal_from_fov_prior(image_width)
        source = "prior"
        info["candidates"]["prior"] = prior_info

    info["source"] = source
    info["focal_px"] = float(chosen)
    info["hfov_deg"] = focal_to_hfov_degrees(chosen, image_width)
    return float(chosen), info
