"""
Input decoding, tile sampling, grout and final compositing.

Every stage here is surface-agnostic pixel work, extracted verbatim from the
original tile_engine so the floor path keeps producing the same pixels. The
wall path calls exactly the same functions -- this file is the reason adding
walls did not mean a second copy of the renderer.
"""

import numpy as np
import cv2

# On-screen grout width bounds, in pixels. The lower bound is what stops the
# far-field lines breaking into dots: below roughly one pixel a thin band is
# only caught on scattered samples. The upper bound stops a heavy mm setting
# on a small surface from turning the render into mostly grout.
GROUT_MIN_PX = 1.3
GROUT_MAX_PX = 14.0


def parse_hex_color(hex_str: str):
    c = (hex_str or "").replace("#", "").strip()
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    try:
        num = int(c, 16)
    except ValueError:
        return 208, 208, 200
    r = (num >> 16) & 255
    g = (num >> 8) & 255
    b = num & 255
    return r, g, b


def decode_depth_array(depth_rgb: np.ndarray) -> np.ndarray:
    """
    Vectorized port of decodeDepthValue(). depth_rgb: HxWx3 uint8, RGB order.
    Returns HxW float32 in [0,1], 1.0 = near, 0.0 = far.
    """
    r = depth_rgb[:, :, 0].astype(np.float32)
    g = depth_rgb[:, :, 1].astype(np.float32)
    b = depth_rgb[:, :, 2].astype(np.float32)

    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    d = mx - mn

    grayscale_mask = d < 15
    out = np.zeros_like(r)
    out[grayscale_mask] = mx[grayscale_mask] / 255.0

    color_mask = ~grayscale_mask
    d_safe = np.where(d == 0, 1e-6, d)

    is_r_max = (mx == r) & color_mask
    is_g_max = (mx == g) & color_mask & ~is_r_max
    is_b_max = color_mask & ~is_r_max & ~is_g_max

    h = np.zeros_like(r)
    h[is_r_max] = (
        (g[is_r_max] - b[is_r_max]) / d_safe[is_r_max]
        + np.where(g[is_r_max] < b[is_r_max], 6.0, 0.0)
    )
    h[is_g_max] = (b[is_g_max] - r[is_g_max]) / d_safe[is_g_max] + 2.0
    h[is_b_max] = (r[is_b_max] - g[is_b_max]) / d_safe[is_b_max] + 4.0
    h *= 60.0

    high = color_mask & (h > 300)
    mid = color_mask & ~high
    out[mid] = 1.0 - (h[mid] / 300.0)
    out[high] = np.where(r[high] > b[high], 1.0, 0.0)

    return out


def normalise_depth_within_mask(depth_val, mask_bool, lo_pct=2.0, hi_pct=98.0,
                                out_lo=0.0, out_hi=1.0):
    """
    Re-spread a depth map's range across ONE surface.

    MiDaS min-max normalises over the whole frame, so the returned range is
    owned by whatever varies most -- almost always the floor, which runs from
    right under the camera to the far wall. A wall occupying the top third of
    that frame lands inside a narrow band of the 0..1 range, and its real
    depth variation is a handful of quantisation steps wide. Fitting a plane
    to that is fitting a plane to rounding error, which is why a wall comes
    back nearly fronto-parallel no matter how it is actually angled.

    Re-normalising inside the surface mask restores the full range to that
    surface. Percentiles rather than min/max so one speckle of segmentation
    bleed onto a distant doorway cannot re-compress everything again.

    Absolute scale is NOT preserved by this, and does not need to be: the
    metric anchor is resolved afterwards from a real measurement, so any
    uniform rescaling of depth cancels out. What it preserves is the SHAPE,
    which is all the plane fit reads.

    out_lo / out_hi -- the TARGET band, and the reason this is a parameter
    rather than a fixed 0..1.
    -------------------------------------------------------------------
    Back-projection is Z = 1000 / (d * contrast + 0.05), a reciprocal with a
    pole just below d = 0. Stretching a surface to touch 0.0 therefore sends
    its far pixels to Z ~ 20000 while its near pixels sit near 1000, and a
    plane fitted through a 21x spread of magnitudes is dominated by the far
    outliers. Measured on a real room: the wall's own depth occupied
    0.074..0.912 (Z spread 7.8x), and normalising it to a full 0..1 made that
    spread 21x -- so the "improvement" made the plane fit materially worse.
    Mapping into a band that keeps clear of the pole gives the contrast
    without the conditioning loss.
    """
    if mask_bool is None or not np.any(mask_bool):
        return depth_val

    vals = depth_val[mask_bool]
    lo = float(np.percentile(vals, lo_pct))
    hi = float(np.percentile(vals, hi_pct))
    if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) < 1e-6:
        return depth_val

    unit = np.clip((depth_val - lo) / (hi - lo), 0.0, 1.0)
    return (out_lo + unit * (out_hi - out_lo)).astype(np.float32)


def normalise_inputs(room_bgr, depth_bgr, mask_bgra, tile_bgra, invert_depth: bool):
    """
    Resize, promote to BGRA, decode depth. Returns
    (depth_bgr, mask_bgra, tile_bgra, depth_val, mask_had_alpha).
    """
    h_img, w_img = room_bgr.shape[:2]

    if depth_bgr.shape[:2] != (h_img, w_img):
        depth_bgr = cv2.resize(depth_bgr, (w_img, h_img), interpolation=cv2.INTER_LINEAR)
    if mask_bgra.shape[:2] != (h_img, w_img):
        mask_bgra = cv2.resize(mask_bgra, (w_img, h_img), interpolation=cv2.INTER_NEAREST)

    # Whether the mask arrived WITH an alpha channel. Promoting GRAY/BGR to
    # BGRA fabricates alpha=255, which must not later be mistaken for a real
    # coverage signal -- an all-black BGR mask would otherwise read as
    # "alpha says every pixel is surface".
    mask_had_alpha = mask_bgra.ndim == 3 and mask_bgra.shape[2] == 4

    if mask_bgra.ndim == 2:
        mask_bgra = cv2.cvtColor(mask_bgra, cv2.COLOR_GRAY2BGRA)
    elif mask_bgra.shape[2] == 3:
        mask_bgra = cv2.cvtColor(mask_bgra, cv2.COLOR_BGR2BGRA)
    if tile_bgra.ndim == 2:
        tile_bgra = cv2.cvtColor(tile_bgra, cv2.COLOR_GRAY2BGRA)
    elif tile_bgra.shape[2] == 3:
        tile_bgra = cv2.cvtColor(tile_bgra, cv2.COLOR_BGR2BGRA)

    depth_rgb = cv2.cvtColor(depth_bgr, cv2.COLOR_BGR2RGB)
    depth_val = decode_depth_array(depth_rgb)  # HxW float32 [0,1]
    if invert_depth:
        depth_val = 1.0 - depth_val

    return depth_bgr, mask_bgra, tile_bgra, depth_val, mask_had_alpha


def decode_mask(mask_bgra, mask_had_alpha, surface_label: str = "floor"):
    """
    Mask image -> (surface_bool, mask_factor).

    Work out which channel actually carries the mask before combining them.
    This endpoint documents its mask as "alpha/brightness = coverage", i.e.
    EITHER channel may hold it, but requiring both (alpha AND brightness)
    rejects two shapes that are perfectly valid:

      * a fully transparent PNG whose RGB holds the mask -- alpha is 0
        everywhere, so the alpha test fails on every pixel;
      * an alpha-only mask with black RGB (the documented RGBA form) --
        brightness is 0 everywhere, so the brightness test fails.

    Both produced "mask is empty" even though the mask was fine. A channel
    that is uniformly zero carries no information, so it is ignored rather
    than allowed to veto the other one. A channel that is uniformly HIGH is
    still meaningful ("all surface"), so it is kept -- that is the normal case
    for a BGR white mask promoted to BGRA with alpha=255.
    """
    mask_alpha = mask_bgra[:, :, 3].astype(np.float32)
    mask_b = mask_bgra[:, :, 0].astype(np.float32)
    mask_g = mask_bgra[:, :, 1].astype(np.float32)
    mask_r = mask_bgra[:, :, 2].astype(np.float32)
    mask_brightness = 0.299 * mask_r + 0.587 * mask_g + 0.114 * mask_b
    mask_val_norm = mask_brightness / 255.0
    mask_alpha_norm = mask_alpha / 255.0

    alpha_usable = mask_had_alpha and float(mask_alpha.max()) > 0.0
    rgb_usable = float(mask_brightness.max()) > 0.0

    if alpha_usable and rgb_usable:
        surface_bool = (mask_alpha >= 50) & (mask_brightness > 50)
        mask_factor = mask_alpha_norm * mask_val_norm
    elif alpha_usable:
        surface_bool = mask_alpha >= 50
        mask_factor = mask_alpha_norm
    elif rgb_usable:
        surface_bool = mask_brightness > 50
        mask_factor = mask_val_norm
    else:
        surface_bool = np.zeros(mask_alpha.shape, dtype=bool)
        mask_factor = np.zeros_like(mask_alpha)

    if not np.any(surface_bool):
        # Say which test failed and on what -- "mask is empty" alone gives the
        # caller nothing to act on.
        raise ValueError(
            f"render: {surface_label} mask selects no pixels "
            f"(alpha max={mask_alpha.max():.0f}, brightness max={mask_brightness.max():.0f}; "
            "need alpha >= 50 or brightness > 50). "
            f"Either the segmenter found no {surface_label} in this image, or the mask is "
            "blank/inverted/too dark."
        )

    return surface_bool, mask_factor


def average_masked_brightness(room_bgr, ys, xs) -> float:
    """Mean brightness of the sampled surface pixels -- the lighting reference."""
    room_rgb_full = cv2.cvtColor(room_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    if len(xs) == 0:
        return 128.0
    sample = room_rgb_full[ys.astype(np.int64), xs.astype(np.int64)].mean(axis=1)
    return float(np.mean(sample)) if len(sample) else 128.0


def sample_tile(tile_bgra, frac_u, frac_v, height_stretch: float = 1.0):
    """
    Sample the tile texture at each pixel's within-tile position.

    height_stretch is a TEXTURE-SPACE squash of the v axis, not a geometric
    one: it changes which part of the tile image a pixel reads, and nothing
    about the tile's physical size or the grid's spacing. The floor path
    passes 1.20, which is a hand-tuned value inherited from the original
    renderer; the wall path passes 1.0 because a wall course has no equivalent
    empirical fudge and applying the floor's would visibly squash the texture.
    """
    tile_h_px, tile_w_px = tile_bgra.shape[:2]
    stretch = height_stretch if abs(height_stretch) > 1e-6 else 1.0
    tx = (frac_u * tile_w_px).astype(np.float32)
    ty = ((frac_v / stretch) * tile_h_px).astype(np.float32)

    sampled = cv2.remap(
        tile_bgra, tx, ty, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP
    )
    return (
        sampled[:, :, 0].astype(np.float32),
        sampled[:, :, 1].astype(np.float32),
        sampled[:, :, 2].astype(np.float32),
        sampled[:, :, 3].astype(np.float32),
    )


def apply_grout(tile_b, tile_g, tile_r, frac_u, frac_v, u_mm, v_mm,
                tile_w_mm: float, tile_h_mm: float, visible,
                grout_width_mm: float, grout_color: str):
    """
    Draw grout lines at a CONSTANT ON-SCREEN width rather than a constant
    physical width.

    The old code used a fixed band of `grout_width_mm` in world space,
    point-sampled once per pixel. Its width in pixels therefore tracked
    however many mm a pixel happened to span, which varies by orders of
    magnitude across a surface: near the camera one pixel is ~0.5 mm, so a
    3 mm band covered ~6 px and looked heavy; near the horizon one pixel is
    ~50 mm, so the same band was ~0.06 px wide and the test only caught it on
    scattered pixels -- the dotted, broken lines.

    u_mm / v_mm ARE the per-pixel world coordinates, so their gradient across
    the image is exactly "world mm per screen pixel" at each point. Scaling the
    band by that local factor cancels the perspective foreshortening out of the
    grout while leaving the tile shape and the grid's convergence fully
    perspective-correct.
    """
    gR, gG, gB = parse_hex_color(grout_color)
    dist_edge_u_mm = np.minimum(frac_u, 1.0 - frac_u) * tile_w_mm
    dist_edge_v_mm = np.minimum(frac_v, 1.0 - frac_v) * tile_h_mm

    du_dy, du_dx = np.gradient(u_mm)
    dv_dy, dv_dx = np.gradient(v_mm)
    mm_per_px_u = np.maximum(np.hypot(du_dx, du_dy), 1e-6)
    mm_per_px_v = np.maximum(np.hypot(dv_dx, dv_dy), 1e-6)

    # Convert the user's physical grout width into a pixel width once, using
    # the median scale over the visible surface, so the mm slider still
    # controls how heavy the grout looks.
    if np.any(visible):
        ref_u = float(np.median(mm_per_px_u[visible]))
        ref_v = float(np.median(mm_per_px_v[visible]))
    else:
        ref_u = ref_v = 1.0

    want_px_u = grout_width_mm / max(ref_u, 1e-6)
    want_px_v = grout_width_mm / max(ref_v, 1e-6)
    width_px_u = float(np.clip(want_px_u, GROUT_MIN_PX, GROUT_MAX_PX))
    width_px_v = float(np.clip(want_px_v, GROUT_MIN_PX, GROUT_MAX_PX))

    # A line thinner than about a pixel cannot be drawn continuously, so widths
    # below the floor are rendered at the floor width and FADED instead -- same
    # total ink, no dotting. Without this the slider is dead below the clamp:
    # 1 mm and 3 mm both pinned to GROUT_MIN_PX and produced pixel-identical
    # output.
    ink_u = min(1.0, want_px_u / width_px_u) if width_px_u > 0 else 1.0
    ink_v = min(1.0, want_px_v / width_px_v) if width_px_v > 0 else 1.0

    # Half-width expressed back in the LOCAL mm scale, so the band is width_px
    # wide on screen everywhere. Capped so that where tiles shrink towards the
    # horizon the grout can never swallow the tile whole.
    half_u_mm = np.minimum(0.5 * width_px_u * mm_per_px_u, 0.40 * tile_w_mm)
    half_v_mm = np.minimum(0.5 * width_px_v * mm_per_px_v, 0.40 * tile_h_mm)

    # Feather exactly one pixel, in local mm -- constant softness on screen
    # instead of a fixed 0.8 mm that was invisible far away and blurry near.
    factor_u = ink_u * (1.0 - np.clip((dist_edge_u_mm - half_u_mm) / mm_per_px_u, 0.0, 1.0))
    factor_v = ink_v * (1.0 - np.clip((dist_edge_v_mm - half_v_mm) / mm_per_px_v, 0.0, 1.0))
    grout_alpha = np.maximum(factor_u, factor_v)

    tile_r = tile_r * (1 - grout_alpha) + gR * grout_alpha
    tile_g = tile_g * (1 - grout_alpha) + gG * grout_alpha
    tile_b = tile_b * (1 - grout_alpha) + gB * grout_alpha
    return tile_b, tile_g, tile_r


# ---- "lowfreq" lighting: the room's light only, never its old surface ----
#: Shading blur, as a share of the image diagonal: removes joints, grout,
#: veins and glare of the old surface, keeps the light gradient and shadows.
SHADING_SIGMA_SHARE = 0.03
#: Shading range and softness: tile * clamp(shading, MIN, MAX) ** GAMMA.
SHADING_MIN, SHADING_MAX, SHADING_GAMMA = 0.55, 1.15, 0.8
#: Limited exposure match: the tile's mean follows this surface's brightness
#: relative to the room's, but only within these bounds (a light tile on a dark
#: wall stays light; a bright room does not blow a light tile out).
EXPOSURE_MIN, EXPOSURE_MAX = 0.6, 1.1
#: A highlight in the photo is real above this level (moderately blurred
#: luminance); no pixel above it = no gloss at all.
HIGHLIGHT_LEVEL = 225.0
HIGHLIGHT_SIGMA_SHARE = 0.01
#: Highlight roll-off: values above KNEE x the tile's peak ease smoothly
#: toward LIMIT x the peak (never past it), so a light tile in a bright room is
#: not blown out and its veins keep their contrast.
TONE_KNEE, TONE_LIMIT = 0.85, 0.97


def _luminance(bgr):
    bgr = bgr.astype(np.float32)
    return 0.114 * bgr[:, :, 0] + 0.587 * bgr[:, :, 1] + 0.299 * bgr[:, :, 2]


def _masked_blur(values, mask, sigma):
    m = mask.astype(np.float32)
    num = cv2.GaussianBlur(values * m, (0, 0), sigma)
    den = cv2.GaussianBlur(m, (0, 0), sigma)
    return num / np.maximum(den, 1e-3)


def lowfreq_light(room_bgr, tile_b, tile_g, tile_r, region, average_brightness: float,
                  exposure_reference=None, gloss_strength: float = 0.2):
    """
    (lit_r, lit_g, lit_b, info): the tile lit by the room's smooth light only.

      shading   the clean room's luminance blurred inside `region`
                (sigma SHADING_SIGMA_SHARE of the diagonal), / its median,
                clamped to SHADING_MIN..MAX, ** SHADING_GAMMA
      exposure  clamp(this surface's mean / the room's reference, EXPOSURE_MIN..MAX)
      gloss     + gloss_strength x highlight, tinted by the tile's own colour,
                only where the (blurred) photo exceeds HIGHLIGHT_LEVEL
      tone      above TONE_KNEE x the tile's peak, values ease toward TONE_LIMIT x the
                peak (smooth roll-off, never past it)
    """
    h, w = region.shape
    diag = float(np.hypot(h, w))
    if not region.any():
        return tile_r, tile_g, tile_b, {}
    lum = _luminance(room_bgr)
    low = _masked_blur(lum, region, SHADING_SIGMA_SHARE * diag)
    med = float(np.median(low[region]))
    shading = np.clip(low / max(med, 1e-3), SHADING_MIN, SHADING_MAX) ** SHADING_GAMMA
    reference = float(exposure_reference) if exposure_reference else float(np.median(lum))
    exposure = float(np.clip(average_brightness / max(reference, 1e-3), EXPOSURE_MIN, EXPOSURE_MAX))
    hl_src = cv2.GaussianBlur(lum, (0, 0), HIGHLIGHT_SIGMA_SHARE * diag)
    highlight = np.clip((hl_src - HIGHLIGHT_LEVEL) / (255.0 - HIGHLIGHT_LEVEL), 0.0, 1.0)
    highlight[~region] = 0.0
    gain = shading * exposure * (1.0 + gloss_strength * highlight)
    peak = float(np.percentile(np.maximum(np.maximum(tile_r, tile_g), tile_b)[region], 99.5))
    knee, limit = TONE_KNEE * peak, TONE_LIMIT * peak
    out = []
    for c in (tile_r, tile_g, tile_b):
        v = c * gain
        over = np.maximum(v - knee, 0.0)
        span = max(limit - knee, 1e-3)
        out.append(np.where(v > knee, knee + span * (1.0 - np.exp(-over / span)), v))
    info = {"lighting": "lowfreq", "exposure": round(exposure, 3), "reference_luminance": round(reference, 1),
            "surface_luminance": round(float(average_brightness), 1),
            "shading_range": [round(float(shading[region].min()), 3), round(float(shading[region].max()), 3)],
            "gloss_pixels": int((highlight > 0).sum())}
    return out[0], out[1], out[2], info


def composite(room_bgr, tile_b, tile_g, tile_r, tile_a, mask_factor, valid_ray,
              average_brightness: float, lighting_blend: float,
              tile_opacity: float, base_bgr=None, lighting_mode: str = "per-pixel",
              exposure_reference=None, gloss_strength: float = 0.2):
    """
    Light the sampled tiles from the room's own pixels and composite them in.

    base_bgr lets a caller composite onto something other than the original
    room image -- the wall pipeline passes the running result so each wall
    instance lays onto the previous one's output instead of erasing it.
    """
    base = room_bgr if base_bgr is None else base_bgr
    room_b = room_bgr[:, :, 0].astype(np.float32)
    room_g = room_bgr[:, :, 1].astype(np.float32)
    room_r = room_bgr[:, :, 2].astype(np.float32)
    room_brightness = (room_r + room_g + room_b) / 3.0

    base_b = base[:, :, 0].astype(np.float32)
    base_g = base[:, :, 1].astype(np.float32)
    base_r = base[:, :, 2].astype(np.float32)

    if lighting_mode == "lowfreq":
        region = (mask_factor >= 0.1) & valid_ray
        final_tile_r, final_tile_g, final_tile_b, _info = lowfreq_light(
            room_bgr, tile_b, tile_g, tile_r, region, average_brightness, exposure_reference, gloss_strength)
    else:
        light_factor = np.clip(room_brightness / (average_brightness + 0.1), 0.15, 2.2)
        blend = lighting_blend

        lit_r = tile_r * light_factor
        lit_g = tile_g * light_factor
        lit_b = tile_b * light_factor

        final_tile_r = tile_r * (1 - blend) + lit_r * blend
        final_tile_g = tile_g * (1 - blend) + lit_g * blend
        final_tile_b = tile_b * (1 - blend) + lit_b * blend

    tile_alpha = (tile_a / 255.0) * tile_opacity * mask_factor

    final_r = final_tile_r * tile_alpha + base_r * (1 - tile_alpha)
    final_g = final_tile_g * tile_alpha + base_g * (1 - tile_alpha)
    final_b = final_tile_b * tile_alpha + base_b * (1 - tile_alpha)

    # Outside the mask / invalid rays -> keep whatever was already there
    keep_original = (mask_factor < 0.1) | (~valid_ray)
    final_r = np.where(keep_original, base_r, final_r)
    final_g = np.where(keep_original, base_g, final_g)
    final_b = np.where(keep_original, base_b, final_b)

    return np.stack(
        [
            np.clip(final_b, 0, 255),
            np.clip(final_g, 0, 255),
            np.clip(final_r, 0, 255),
        ],
        axis=-1,
    ).astype(np.uint8)
