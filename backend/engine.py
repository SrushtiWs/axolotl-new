"""
Metric tile projection.

This is the production pipeline's stage 08h1 (true metric tile projection)
followed by 08h2 (props over metric floor), re-expressed as a parameterised,
vectorised function so a request can choose its own tile artwork, tile size,
grout, rotation and target surface.

The maths is unchanged from the stage scripts:

  * a pixel is back-projected through the fitted camera (K, R, t),
  * the ray is intersected with the floor plane Y=0, the left wall Z=0 or the
    right wall X=0,
  * the metric hit point indexes a tile grid with `tile + grout` pitch,
  * low-frequency illumination from the empty-room render is re-applied, plus
    the same restrained gloss response.

Nothing is inpainted, hallucinated or upscaled here — every output pixel is
either sampled tile artwork, grout colour, or original room RGB.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np

from scene import Scene

Surface = Literal["floor", "wall", "both"]

# Grout width in millimetres. Matches the locked pipeline spec.
GROUT_MM = 5.0

# Neutral grout colour, as frozen in stage 08h1.
GROUT_RGB = np.array([224, 224, 220], dtype=np.float32)

# How far up the wall the skirting/screeding course runs.
SCREEDING_MAX_HEIGHT_MM = 180.0

GLOSS_STRENGTH = 0.055

# Illumination transfer limits — keeps shading, refuses to invent contrast.
ILLUM_MIN = 0.82
ILLUM_MAX = 1.16
ILLUM_SIGMA = 24.0

# Illumination clip for the estimated path.
#
# ILLUM_MIN/MAX exist to stop the transfer inventing contrast, and on the
# calibrated bathroom — an even render, lit softly — +-16% is the whole range
# there is. An ordinary photograph is not lit like that: a room with a window
# down one side genuinely spans far more, and clipping at +-16% does not
# suppress an artefact, it deletes the shading.
#
# Measured on an uploaded bathroom wall, the fraction of tiled pixels pinned at
# a bound once the polynomial is supplying the field:
#
#     0.82 .. 1.16   30.3%
#     0.70 .. 1.30    8.1%
#     0.60 .. 1.45    4.8%
#
# 0.70..1.30 is where the wall stops banding. It is not opened further because
# the polynomial's tails are extrapolation, not measurement — the fit reaches
# 0.38 at the 1st percentile on that wall — and a wider clip starts trusting
# them.
ILLUM_MIN_ESTIMATED = 0.70
ILLUM_MAX_ESTIMATED = 1.30

# Order of the polynomial fitted to the empty room's luminance on the estimated
# path. 0 falls back to the Gaussian blur everywhere.
#
# A blur cannot separate illumination from albedo. It assumes the surface under
# the light is uniform, so whatever the surface's own tone does at a scale wider
# than the kernel is read as lighting and multiplied onto the new tiles. That
# assumption holds for the calibrated bathroom, whose empty room is an even
# render, and fails on an ordinary photograph of a tiled wall: each original
# tile has its own tone, the blocks are wider than the kernel, and the old wall
# prints straight through the new one.
#
# Measured on an uploaded bathroom, the fraction of tiled pixels whose
# illumination ratio is pinned at ILLUM_MIN or ILLUM_MAX — a pixel at a clip
# bound is not receiving a gradient, it is receiving a stencil of the old wall:
#
#     gaussian, sigma 24 (and 48, 80, 120)   62.0%
#     polynomial, order 1                    27.2%
#     polynomial, order 2                    30.3%
#     polynomial, order 3                    45.0%
#
# Raising the blur's sigma does not help: at sigma 200 the ratio is still
# saturated, because the variation is albedo and no amount of blurring turns
# albedo into light. A low-order polynomial fixes it structurally instead — a
# quadratic surface can represent a window's falloff or a lamp's pool of light,
# and *cannot* represent per-tile blocks, whatever their contrast.
#
# Order 3 begins to fit the blocks again, which is why this is 2 and not higher.
ILLUM_POLY_ORDER = 2

# Contact-shadow transfer. The reference blur is a fraction of the photo's
# longest edge so a shadow reads the same on a phone upload and a thumbnail;
# wide enough to average the surface rather than its texture, narrow enough to
# keep the shadow's own shape.
SHADOW_SIGMA_FRACTION = 0.035

# How dark a transferred shadow may go. A contact shadow deepens a floor; it
# never blacks it out, and a floor of pure black would look like a hole.
SHADOW_MIN = 0.45


class MissingGeometryError(RuntimeError):
    """
    Raised when a request cannot be served because the room has no calibrated
    geometry — no floor mask, no camera fit. Producing an image anyway would
    mean guessing where the floor is, which this backend does not do.
    """


@dataclass
class TileSpec:
    """A tile as installed: artwork, physical size, rotation of the layout."""

    artwork: np.ndarray  # (h, w, 3) uint8
    width_mm: float
    height_mm: float
    rotation_deg: float = 0.0
    grout_mm: float = GROUT_MM

    @property
    def pitch_u(self) -> float:
        return self.width_mm + self.grout_mm

    @property
    def pitch_v(self) -> float:
        return self.height_mm + self.grout_mm


@dataclass
class Result:
    """Rendered images plus the numbers needed to audit the render."""

    raw: np.ndarray  # tiled room, exact texture, no relighting
    lit: np.ndarray  # tiled room with illumination + gloss
    composite: np.ndarray  # lit room with the original props back on top
    target: np.ndarray  # pixels claimed by the tile projection
    underlying: np.ndarray  # exact base image available before projection
    stats: dict = field(default_factory=dict)


def _rays(scene: Scene, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """World-space unit direction for each pixel. Returns (3, N)."""
    pixels = np.stack([xs.astype(np.float64), ys.astype(np.float64), np.ones(xs.size)])

    directions = scene.R.T @ (np.linalg.inv(scene.K) @ pixels)

    norms = np.linalg.norm(directions, axis=0)
    norms[norms < 1e-12] = 1.0

    return directions / norms


def _intersect_plane(
    scene: Scene,
    directions: np.ndarray,
    axis: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Intersect rays with the plane `coordinate[axis] == 0`.

    Returns (points (3, N), valid mask). Rays parallel to the plane or hitting
    it behind the camera are marked invalid.
    """
    centre = scene.camera_center

    denominator = directions[axis]

    valid = np.abs(denominator) >= 1e-10

    lam = np.zeros(directions.shape[1], dtype=np.float64)
    lam[valid] = -centre[axis] / denominator[valid]

    valid &= lam > 0

    points = centre[:, None] + lam[None, :] * directions

    return points, valid


def _intersect_plane_at(
    scene: Scene,
    directions: np.ndarray,
    axis: int,
    value: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Intersect rays with the plane `coordinate[axis] == value`.

    The same maths as `_intersect_plane`, which is the `value == 0` case, but a
    wall of an uploaded room sits at whatever distance the stated room
    dimensions put it, not at the origin.

    Returns (points (3, N), valid mask, lambda). Lambda — the distance along the
    ray — is returned because with several walls in play the nearest valid hit
    is the one actually visible, and that cannot be decided plane by plane.
    """
    centre = scene.camera_center

    denominator = directions[axis]

    valid = np.abs(denominator) >= 1e-10

    lam = np.zeros(directions.shape[1], dtype=np.float64)
    lam[valid] = (value - centre[axis]) / denominator[valid]

    valid &= lam > 0

    points = centre[:, None] + lam[None, :] * directions

    return points, valid, lam


def _rotate_about(u: np.ndarray, v: np.ndarray, angle_deg: float, cu: float, cv: float):
    """Rotate a tile layout about a given centre. Identity at 0 degrees."""
    if not angle_deg:
        return u, v

    angle = math.radians(angle_deg)
    cos, sin = math.cos(angle), math.sin(angle)

    du, dv = u - cu, v - cv

    return cos * du + sin * dv + cu, -sin * du + cos * dv + cv


def _rotate_layout(u: np.ndarray, v: np.ndarray, spec: TileSpec, scene: Scene):
    """Rotate the tile grid about the floor centre. Identity at 0 degrees."""
    if not spec.rotation_deg:
        return u, v

    angle = math.radians(spec.rotation_deg)
    cos, sin = math.cos(angle), math.sin(angle)

    cu, cv = scene.room_u_mm / 2.0, scene.room_v_mm / 2.0

    du, dv = u - cu, v - cv

    return cos * du + sin * dv + cu, -sin * du + cos * dv + cv


def _tile_coordinate(coord: np.ndarray, size_mm: float, pitch_mm: float):
    """
    Split a metric coordinate into (is_grout, fraction across the tile face).

    Grout occupies the last `grout_mm` of every pitch. Negative coordinates are
    fine — a rotated layout produces them — because the modulo wraps them.
    """
    local = np.mod(coord, pitch_mm)

    is_grout = local >= size_mm

    fraction = np.clip(local / size_mm, 0.0, 0.999999)

    return is_grout, fraction


def _sample(spec: TileSpec, u_frac: np.ndarray, v_frac: np.ndarray) -> np.ndarray:
    """Nearest-neighbour sample of the tile artwork. Returns (N, 3) float32."""
    height, width = spec.artwork.shape[:2]

    tx = np.clip((u_frac * width).astype(np.int32), 0, width - 1)
    ty = np.clip((v_frac * height).astype(np.int32), 0, height - 1)

    return spec.artwork[ty, tx].astype(np.float32)


def _project_floor(scene: Scene, spec: TileSpec, canvas: np.ndarray) -> np.ndarray:
    """Paint tiles onto the floor mask. Returns the pixels actually covered."""
    ys, xs = np.nonzero(scene.floor)

    covered = np.zeros((scene.height, scene.width), dtype=bool)

    if ys.size == 0:
        return covered

    points, valid = _intersect_plane(scene, _rays(scene, xs, ys), axis=1)

    u = -points[0]
    v = points[2]

    valid &= (u >= 0.0) & (u <= scene.room_u_mm)
    valid &= (v >= 0.0) & (v <= scene.room_v_mm)

    if not valid.any():
        return covered

    ys, xs, u, v = ys[valid], xs[valid], u[valid], v[valid]

    ur, vr = _rotate_layout(u, v, spec, scene)

    grout_u, frac_u = _tile_coordinate(ur, spec.width_mm, spec.pitch_u)
    grout_v, frac_v = _tile_coordinate(vr, spec.height_mm, spec.pitch_v)

    is_grout = grout_u | grout_v

    colours = np.empty((ys.size, 3), dtype=np.float32)
    colours[is_grout] = GROUT_RGB
    colours[~is_grout] = _sample(spec, frac_u[~is_grout], frac_v[~is_grout])

    canvas[ys, xs] = colours
    covered[ys, xs] = True

    return covered


def _project_screeding(scene: Scene, spec: TileSpec, canvas: np.ndarray) -> np.ndarray:
    """
    Paint the wall skirting course, cut from the same tile.

    Pixels left of the back floor corner belong to the left wall (Z=0), the
    rest to the right wall (X=0) — the same split the stage script uses.
    """
    ys, xs = np.nonzero(scene.screeding)

    covered = np.zeros((scene.height, scene.width), dtype=bool)

    if ys.size == 0:
        return covered

    directions = _rays(scene, xs, ys)

    is_left = xs < scene.back_x_image

    horizontal = np.zeros(ys.size, dtype=np.float64)
    height_mm = np.zeros(ys.size, dtype=np.float64)
    valid = np.zeros(ys.size, dtype=bool)

    for side_mask, axis, extent in (
        (is_left, 2, scene.room_u_mm),
        (~is_left, 0, scene.room_v_mm),
    ):
        if not side_mask.any():
            continue

        points, hit = _intersect_plane(scene, directions[:, side_mask], axis=axis)

        # Left wall is parameterised by U=-X, right wall by V=+Z.
        along = -points[0] if axis == 2 else points[2]

        hit &= (along >= 0.0) & (along <= extent)

        horizontal[side_mask] = along
        height_mm[side_mask] = points[1]
        valid[side_mask] = hit

    valid &= (height_mm >= 0.0) & (height_mm <= SCREEDING_MAX_HEIGHT_MM)

    if not valid.any():
        return covered

    ys, xs = ys[valid], xs[valid]
    is_left = is_left[valid]
    horizontal = horizontal[valid]
    height_mm = height_mm[valid]

    # The skirting is a cut from the bottom of the tile face, so the vertical
    # fraction is driven by how far up the wall the pixel sits.
    vertical_fraction = np.clip(height_mm / spec.height_mm, 0.0, 1.0)
    v_frac = 1.0 - vertical_fraction

    colours = np.empty((ys.size, 3), dtype=np.float32)
    is_grout = np.zeros(ys.size, dtype=bool)

    if is_left.any():
        grout, frac = _tile_coordinate(horizontal[is_left], spec.width_mm, spec.pitch_u)
        is_grout[is_left] = grout
        colours[is_left] = _sample(spec, frac, v_frac[is_left])

    if (~is_left).any():
        grout, frac = _tile_coordinate(horizontal[~is_left], spec.height_mm, spec.pitch_v)
        is_grout[~is_left] = grout
        # The right wall runs along the tile's long installed direction, so the
        # horizontal cut indexes V while U stays on a stable mid-face strip.
        colours[~is_left] = _sample(spec, np.full(frac.shape, 0.5, dtype=np.float64), frac)

    colours[is_grout] = GROUT_RGB

    canvas[ys, xs] = colours
    covered[ys, xs] = True

    return covered


def _project_walls(scene: Scene, spec: TileSpec, canvas: np.ndarray) -> np.ndarray:
    """
    Paint tiles onto every visible wall, each on its own plane.

    This is the floor projection's exact counterpart, and deliberately so: a
    pixel is back-projected through the same camera, intersected with a plane,
    and the metric hit point indexes the same `tile + grout` grid. Only the
    plane and the two directions on it differ.

    What makes several walls work is choosing per pixel rather than per region.
    Every wall plane is intersected for every wall pixel, hits outside a wall's
    own extent are discarded, and of the hits that remain the **nearest** wins —
    which is simply what the camera sees. A wall meeting another in a corner
    therefore gets the right side of the corner without anyone having to say
    where the corner is, and a room showing three walls tiles all three, each
    running in its own direction.

    Tiles run across the wall horizontally and up it vertically, so a rotation
    turns the layout in the plane of that wall about its own centre — the same
    thing rotation means on the floor.
    """
    covered = np.zeros((scene.height, scene.width), dtype=bool)

    if scene.wall is None or not scene.walls:
        return covered

    ys, xs = np.nonzero(scene.wall)

    if ys.size == 0:
        return covered

    directions = _rays(scene, xs, ys)

    count = ys.size

    # Best hit so far, per pixel.
    best_lambda = np.full(count, np.inf, dtype=np.float64)
    across = np.zeros(count, dtype=np.float64)
    height_mm = np.zeros(count, dtype=np.float64)
    chosen = np.full(count, -1, dtype=np.int32)

    for index, wall in enumerate(scene.walls):
        points, hit, lam = _intersect_plane_at(scene, directions, wall.axis, wall.value)

        if not hit.any():
            continue

        # Where the hit sits across the wall, and how far up it.
        wall_across = wall.horizontal_sign * (points[wall.horizontal_axis] - wall.horizontal_origin)

        wall_height = points[1]

        hit &= (wall_across >= 0.0) & (wall_across <= wall.width_mm)
        hit &= (wall_height >= 0.0) & (wall_height <= wall.height_mm)

        # The nearest surface along the ray is the one in view; anything behind
        # it is hidden by it.
        closer = hit & (lam < best_lambda)

        best_lambda[closer] = lam[closer]
        across[closer] = wall_across[closer]
        height_mm[closer] = wall_height[closer]
        chosen[closer] = index

    painted = chosen >= 0

    if not painted.any():
        return covered

    # Recorded so a caller can see that each wall really was used, and with how
    # many pixels — the difference between "three planes were configured" and
    # "three planes were rendered".
    scene.wall_coverage.clear()

    for index, wall in enumerate(scene.walls):
        scene.wall_coverage[wall.label] = int(np.count_nonzero(chosen == index))

    ys, xs = ys[painted], xs[painted]
    across, height_mm, chosen = across[painted], height_mm[painted], chosen[painted]

    colours = np.empty((ys.size, 3), dtype=np.float32)

    for index, wall in enumerate(scene.walls):
        on_this_wall = chosen == index

        if not on_this_wall.any():
            continue

        u = across[on_this_wall]
        v = height_mm[on_this_wall]

        ur, vr = _rotate_about(
            u, v, spec.rotation_deg, wall.width_mm / 2.0, wall.height_mm / 2.0
        )

        grout_u, frac_u = _tile_coordinate(ur, spec.width_mm, spec.pitch_u)
        grout_v, frac_v = _tile_coordinate(vr, spec.height_mm, spec.pitch_v)

        is_grout = grout_u | grout_v

        # The tile's own vertical axis points up the wall, so a course laid from
        # the floor upwards reads the artwork bottom-to-top.
        sampled = _sample(spec, frac_u, 1.0 - frac_v)

        sampled[is_grout] = GROUT_RGB

        colours[on_this_wall] = sampled

    canvas[ys, xs] = colours
    covered[ys, xs] = True

    return covered


def _contact_shadow(scene: Scene, target: np.ndarray) -> np.ndarray:
    """
    The shading the room's own objects cast on the surfaces being retiled.

    An object's contact shadow lives on the *floor*, not on the object, so it is
    not part of the props mask and the tile projection paints straight over it.
    Restoring the object afterwards then puts a perfectly lit chair on a
    perfectly lit floor with nothing joining them, and it reads as pasted on.

    What is recovered here is the multiplicative part of the original photo's
    shading, and only that. The surface's own luma is divided by a local average
    of itself, computed over the surface pixels alone so that the dark object
    sitting in the middle of the neighbourhood cannot drag the average down. A
    pixel in shadow is darker than its surroundings and gives a ratio below one;
    a pixel in open light matches its surroundings and gives one exactly.

    The ratio is clamped at 1.0 on the bright side, so this can only darken. It
    transfers shadow and never invents highlights, and because the reference is
    a wide blur it carries the shadow's shape without carrying the old floor's
    texture or colour onto the new tile.
    """
    shading = np.ones((scene.height, scene.width), dtype=np.float32)

    if not target.any():
        return shading

    luma = cv2.cvtColor(scene.master_input, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # Only surface pixels contribute to the reference. Weighting by the mask and
    # dividing by the blurred mask is a normalised blur over an irregular
    # region: it answers "how bright is this part of the floor, ignoring
    # everything standing on it".
    valid = target.astype(np.float32)

    sigma = max(12.0, SHADOW_SIGMA_FRACTION * max(scene.height, scene.width))

    weighted = cv2.GaussianBlur(luma * valid, (0, 0), sigmaX=sigma, sigmaY=sigma)
    coverage = cv2.GaussianBlur(valid, (0, 0), sigmaX=sigma, sigmaY=sigma)

    reference = np.divide(
        weighted, coverage, out=np.zeros_like(weighted), where=coverage > 1e-3
    )

    usable = (coverage > 1e-3) & (reference > 1.0) & target

    ratio = np.ones_like(luma)

    ratio[usable] = luma[usable] / reference[usable]

    # Darkening only, and never to black: a contact shadow deepens a surface, it
    # does not replace it.
    shading[target] = np.clip(ratio[target], SHADOW_MIN, 1.0)

    return shading


def _restore_props(scene: Scene, lit: np.ndarray) -> np.ndarray:
    """
    Put the room's own objects back over the retiled surfaces.

    Every pixel of every object is the original photograph's pixel at the
    original coordinate, so position, scale, perspective, texture and lighting
    are the ones the camera recorded — nothing is regenerated or repainted.

    Where the scene carries a soft alpha the objects are blended with it rather
    than stamped through a binary mask. That matters at exactly one place, the
    boundary, and it is the difference between an object that meets the new
    floor along its own edge and one that meets it along a staircase. Because
    the blend is a convex combination of tiled surface and original object,
    every pixel is covered by one, the other, or a mix: there is no value of
    alpha that leaves a gap, a white rim or a transparent hole.
    """
    original = scene.master_input.astype(np.float32)

    if scene.props_alpha is None:
        composite = lit.copy()

        composite[scene.props] = scene.master_input[scene.props]

        return composite

    alpha = np.clip(scene.props_alpha, 0.0, 1.0).astype(np.float32)[..., None]

    blended = lit.astype(np.float32) * (1.0 - alpha) + original * alpha

    return np.clip(blended, 0, 255).astype(np.uint8)


def _polynomial_illumination(
    luma: np.ndarray, target: np.ndarray, order: int
) -> np.ndarray:
    """
    The smooth lighting field that best explains the surface being retiled.

    Fitted over `target` only — the pixels actually receiving tiles — because
    those are the ones whose illumination has to be reproduced, and fitting over
    the whole frame would let a bright window or a dark doorway drag the surface
    somewhere the wall never goes.

    Falls back to the blur if the fit is degenerate, which it is whenever the
    target is too small or too thin to constrain the coefficients.
    """
    height, width = luma.shape[:2]

    ys, xs = np.nonzero(target)

    terms = [(i, j) for i in range(order + 1) for j in range(order + 1 - i)]

    if ys.size < 4 * len(terms):
        return cv2.GaussianBlur(luma, (0, 0), sigmaX=ILLUM_SIGMA, sigmaY=ILLUM_SIGMA)

    # Centred and normalised so the basis stays well conditioned whatever the
    # photo's dimensions are.
    x = xs / float(width) - 0.5
    y = ys / float(height) - 0.5

    design = np.stack([x**i * y**j for i, j in terms], axis=1)

    try:
        coefficients, *_ = np.linalg.lstsq(
            design, luma[target].astype(np.float64), rcond=None
        )
    except np.linalg.LinAlgError:
        return cv2.GaussianBlur(luma, (0, 0), sigmaX=ILLUM_SIGMA, sigmaY=ILLUM_SIGMA)

    grid_y, grid_x = np.mgrid[0:height, 0:width]

    gx = grid_x / float(width) - 0.5
    gy = grid_y / float(height) - 0.5

    field = np.zeros((height, width), dtype=np.float64)

    for coefficient, (i, j) in zip(coefficients, terms):
        field += coefficient * (gx**i) * (gy**j)

    return field.astype(np.float32)


def _relight(
    scene: Scene,
    raw: np.ndarray,
    target: np.ndarray,
    controlled_material: bool = False,
) -> np.ndarray:
    """Re-apply the empty room's low-frequency illumination and gloss response."""
    lit = raw.copy()

    if not target.any():
        return lit

    luma = cv2.cvtColor(scene.empty_room, cv2.COLOR_RGB2GRAY).astype(np.float32)

    if controlled_material:
        # Keep broad room illumination, but prevent wall texture and object
        # remnants from becoming a tile-by-tile lighting signal.
        sigma = max(ILLUM_SIGMA, 0.08 * max(scene.height, scene.width))
        luma = cv2.GaussianBlur(luma, (0, 0), sigmaX=sigma, sigmaY=sigma)

    estimated = scene.illumination == "polynomial" and ILLUM_POLY_ORDER > 0

    if estimated:
        illum = _polynomial_illumination(luma, target, ILLUM_POLY_ORDER)
        low, high = ILLUM_MIN_ESTIMATED, ILLUM_MAX_ESTIMATED
    else:
        illum = cv2.GaussianBlur(luma, (0, 0), sigmaX=ILLUM_SIGMA, sigmaY=ILLUM_SIGMA)
        low, high = ILLUM_MIN, ILLUM_MAX

    reference = max(float(np.median(illum[target])), 1.0)

    ratio = np.clip(illum / reference, low, high)

    lit[target] *= ratio[target, None]

    highlight = cv2.GaussianBlur(
        np.clip((ratio - 1.015) / 0.145, 0.0, 1.0), (0, 0), sigmaX=8.0, sigmaY=8.0
    )

    lit[target] *= 1.0 - GLOSS_STRENGTH * highlight[target, None]

    return lit


def render(
    scene: Scene,
    spec: TileSpec,
    surface: Surface = "floor",
    controlled_material: bool = True,
) -> Result:
    """Run the metric tile projection and composite the props back on top."""
    if spec.width_mm <= 0 or spec.height_mm <= 0:
        raise ValueError("Tile dimensions must be greater than zero.")

    raw = scene.empty_room.astype(np.float32).copy()

    floor_covered = np.zeros((scene.height, scene.width), dtype=bool)
    screed_covered = np.zeros((scene.height, scene.width), dtype=bool)

    if surface in ("floor", "both"):
        floor_covered = _project_floor(scene, spec, raw)

    if surface in ("wall", "both"):
        # Whichever wall geometry the scene actually carries. The calibrated
        # room has a measured screeding band and keeps it; a room with fitted
        # wall planes tiles them in full. Neither path can affect the other.
        screed_covered = (
            _project_walls(scene, spec, raw)
            if scene.has_wall_planes
            else _project_screeding(scene, spec, raw)
        )

    target = floor_covered | screed_covered

    if not target.any():
        raise MissingGeometryError(
            f"No {surface} pixels could be projected for this request. Either the "
            "room photo shows none of that surface, or the tile is larger than the "
            "room it is being laid in."
        )

    lit = _relight(scene, raw, target, controlled_material=controlled_material)

    # The room's own shadows, back onto the surfaces that were just repainted.
    # Only for a scene whose master input still holds its objects: the
    # calibrated room's empty-room render already carries its lighting with the
    # props removed, and re-deriving shadows from it would double them.
    shadow_applied = False

    if scene.props_alpha is not None and not controlled_material:
        shading = _contact_shadow(scene, target)

        lit[target] *= shading[target, None]

        shadow_applied = True

    raw_u8 = np.clip(raw, 0, 255).astype(np.uint8)
    lit_u8 = np.clip(lit, 0, 255).astype(np.uint8)

    composite = _restore_props(scene, lit_u8)

    return Result(
        raw=raw_u8,
        lit=lit_u8,
        composite=composite,
        stats={
            "canvas": [scene.width, scene.height],
            "floor_pixels": int(floor_covered.sum()),
            "screeding_pixels": int(screed_covered.sum()),
            "wall_pixels_by_plane": dict(scene.wall_coverage),
            "prop_pixels": int(scene.props.sum()),
            "soft_prop_edges": scene.props_alpha is not None,
            "contact_shadow_applied": shadow_applied,
            "room_mm": [scene.room_u_mm, scene.room_v_mm, scene.room_height_mm],
            "tile_mm": [spec.width_mm, spec.height_mm],
            "grout_mm": spec.grout_mm,
            "rotation_deg": spec.rotation_deg,
            "surface": surface,
            "controlled_material": controlled_material,
        },
        target=target,
        underlying=scene.empty_room.copy(),
    )
