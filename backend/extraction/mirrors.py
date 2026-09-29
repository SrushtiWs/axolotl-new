"""
Deciding which objects are mirrors, and giving them their whole shape.

Two models get a say, and the split of authority between them is the point of
this module.

**Grounding DINO cannot decide this on its own.** It is prompted with text, and
its mirror phrases fire on anything flat and wall-mounted. Measured across two
rooms:

    room with a genuine wall mirror   "mirror" scored 0.212
    bedroom containing no mirror      "mirror" scored 0.266  (it was a shelf)

The false detection scored *higher* than the true one, so no confidence
threshold separates them. Nor does the margin over the best non-mirror phrase:
+0.132 for the real mirror against +0.153 for the shelf. Every signal DINO
offers ranks the shelf above the mirror.

**SegFormer can.** ADE20K carries a dedicated `mirror` class, distinct from
`glass` and `windowpane`, and it is pixel evidence rather than a text score. On
the same two rooms it put the mirror at `[31,91,133,242]` against a ground truth
of `[35,89,131,226]`, and found no mirror whatsoever in the bedroom.

So the rule is pixel evidence first:

  * a mask that is mostly ADE20K-mirror is a mirror, whatever DINO called it —
    which is what stops a mirror from reaching ALL_OBJECTS.png under some other
    name;
  * a mask DINO *also* called a mirror needs less pixel evidence, because two
    independent models agreeing is stronger than either alone;
  * anything else is an ordinary object, however reflective it looks. Glass and
    windowpanes are not mirrors, and neither is a shelf.

The same class map then repairs the mirror's shape. DINO tends to box only a
band of a large mirror — on the test room it found `[64,89,102,102]` of a mirror
that runs to `[35,89,131,226]` — and SAM faithfully segments the band it was
given, leaving the mirror in fragments. Unioning a confirmed mirror with the
mirror region it sits in restores the whole surface.
"""

from __future__ import annotations

import cv2
import numpy as np

from extraction.config import ExtractionConfig


def overlap_share(mask: np.ndarray, mirror_map: np.ndarray) -> float:
    """What fraction of `mask` the semantic mirror map also claims."""
    area = int(mask.sum())

    if area == 0 or not mirror_map.any():
        return 0.0

    return float((mask & mirror_map).sum()) / float(area)


def classify(
    mask: np.ndarray,
    names: tuple[str, ...],
    mirror_map: np.ndarray,
    config: ExtractionConfig,
    glazing_map: np.ndarray | None = None,
) -> tuple[str, dict]:
    """
    Decide what one detection is: `"mirror"`, `"object"` or `"reflection"`.

    `names` is every phrase that fired on it — the winning label plus the
    aliases merged into it — so a mirrored door still counts as corroborated
    when the plain word "door" won the label.

    The three-way answer matters. A mask lying on the mirror surface that the
    detector did *not* call a mirror is neither: it is the room seen in the
    mirror. Calling it a mirror duplicates the mirror; calling it an object
    paints a phantom copy of the furniture over the retiled floor. It belongs in
    neither layer, so it gets its own verdict.

    The evidence is returned in every case. When a cut-out lands in the wrong
    layer, the numbers that put it there are the first thing worth reading.
    """
    share = overlap_share(mask, mirror_map)

    mirror_names = [name for name in names if config.is_mirror(name)]

    corroborated = bool(mirror_names)

    rules = config.detection

    evidence: dict = {
        "mirror_pixel_share": round(share, 4),
        "detector_called_it_mirror": corroborated,
    }

    if mirror_names:
        evidence["mirror_phrases"] = mirror_names

    # Both models agree: this is the mirror itself.
    #
    # Glazing counts here and only here. A mirrored wall panel is often read as
    # "glass" or "windowpane" rather than "mirror" — the class map sees a flat
    # reflective sheet and takes the commoner label — so a real mirror can score
    # zero on the mirror class and be lost to ALL_OBJECTS.png. Allowing glazing
    # to support the mirror class recovers it.
    #
    # It is confined to this branch deliberately, and then confined again: a
    # plain window carries exactly the same glazing pixels, so glazing may only
    # *supplement* real mirror evidence, never stand in for it.
    #
    # Without that second guard the branch is simply a window detector. Measured
    # across four rooms once the detector's recall improved enough to find the
    # glazing, a kitchen window, a dining window and a pair of glass patio doors
    # were all called mirrors on 100%, 55% and 95% "reflective surface" with not
    # one mirror pixel among them, while the two real mirrors in the same set
    # measured 0.998 and 0.249 on the mirror class. Requiring a floor of genuine
    # mirror evidence separates them exactly.
    supported = share

    if (
        corroborated
        and share >= rules.mirror_glazing_needs_mirror_share
        and glazing_map is not None
        and glazing_map.any()
    ):
        supported = overlap_share(mask, mirror_map | glazing_map)

        if supported > share:
            evidence["mirror_or_glazing_share"] = round(supported, 4)

    if corroborated and supported >= rules.mirror_corroborated_share:
        evidence["mirror_threshold"] = rules.mirror_corroborated_share

        if supported > share:
            evidence["verdict"] = (
                f"detector called it a mirror and {supported:.0%} of the mask is "
                f"reflective surface ({share:.0%} of it read as mirror, the rest as "
                "glazing) — a mirrored panel"
            )

        return "mirror", evidence

    # Pixel evidence alone is enough when it is overwhelming and the detector
    # offered no opinion either way — this is what stops a mirror reaching
    # ALL_OBJECTS.png under some other name.
    if not corroborated and share >= rules.mirror_pixel_share:
        # ...unless it is so completely inside the mirror that it is far more
        # likely to be a reflection of something in the room.
        if share >= rules.mirror_reflection_share:
            evidence["verdict"] = (
                f"{share:.0%} of the mask is mirror surface and the detector "
                f"called it {names[0]!r} — the room seen in the mirror, not an "
                "object in it"
            )

            return "reflection", evidence

        evidence["mirror_threshold"] = rules.mirror_pixel_share

        evidence["verdict"] = (
            f"{share:.0%} of the mask is mirror pixels — a mirror despite the "
            f"detector calling it {names[0]!r}"
        )

        return "mirror", evidence

    if corroborated:
        # The single most useful line in the metadata: it names the exact case
        # that used to put a wall shelf in MIRRORS_ONLY.png.
        evidence["mirror_threshold"] = rules.mirror_corroborated_share

        evidence["verdict"] = (
            f"detector said mirror but only {supported:.0%} of the mask is mirror "
            f"or glazing (needs {rules.mirror_corroborated_share:.0%}) — treated "
            "as an ordinary object"
        )

    return "object", evidence


def unclaimed(
    mirror_map: np.ndarray, claimed: list[np.ndarray], minimum_pixels: int = 64
) -> list[np.ndarray]:
    """
    Mirror regions that no accepted mirror covers.

    Without this a mirror the detector never boxed at all would appear in
    neither layer — and since every detection lying on it is dropped as a
    reflection, it would vanish from the output entirely. Returning the bare
    region lets the caller add it, so the guarantee holds: a mirror is never
    silently lost, and never lands among the ordinary objects.
    """
    if not mirror_map.any():
        return []

    count, components = cv2.connectedComponents(mirror_map.astype(np.uint8), connectivity=8)

    covered = np.zeros(mirror_map.shape, dtype=bool)

    for mask in claimed:
        covered |= mask

    missing = []

    for index in range(1, count):
        region = components == index

        if int(region.sum()) < minimum_pixels:
            continue

        # A region is claimed if an accepted mirror already covers most of it.
        if float((region & covered).sum()) / float(region.sum()) >= 0.5:
            continue

        missing.append(region)

    return missing


def group(
    candidates: list[tuple[np.ndarray, str, float, dict]], mirror_map: np.ndarray
) -> list[dict]:
    """
    Collapse mirror detections into one entry per physical mirror.

    Box-level duplicate removal runs before SAM and cannot do this. The detector
    tends to box *bands* of a large mirror rather than the whole thing — on the
    test room it produced two boxes, `[64,89,102,102]` and `[61,134,84,165]`,
    neither overlapping the other, both belonging to one mirror running from
    y=89 to y=242. They survive box dedup honestly, and then each completes to
    the same surface, leaving the mirror written twice.

    Grouping by the mirror region a detection lands on is what settles it: one
    surface, one mirror, however many bands were boxed on it. The surviving
    label and confidence come from the strongest member.
    """
    if not candidates:
        return []

    count, components = cv2.connectedComponents(mirror_map.astype(np.uint8), connectivity=8)

    # component index -> the candidates sitting on it
    by_component: dict[int, list[int]] = {}

    loose: list[int] = []

    for position, (mask, _, _, _) in enumerate(candidates):
        touched = {
            index
            for index in range(1, count)
            if ((components == index) & mask).any()
        }

        if not touched:
            loose.append(position)
            continue

        # A detection spanning two regions merges them: it is one mirror whose
        # surface the class map happened to split.
        key = min(touched)

        by_component.setdefault(key, []).append(position)

        for index in touched:
            if index != key:
                by_component.setdefault(key, [])
                components[components == index] = key

    grouped: list[dict] = []

    for key, members in by_component.items():
        mask = components == key

        for position in members:
            mask = mask | candidates[position][0]

        best = max(members, key=lambda position: candidates[position][2])

        grouped.append(
            {
                "mask": mask,
                "label": candidates[best][1],
                "confidence": candidates[best][2],
                "evidence": {
                    **candidates[best][3],
                    "merged_detections": len(members),
                    "mirror_pixels": int(mask.sum()),
                },
            }
        )

    for position in loose:
        mask, label, confidence, evidence = candidates[position]

        grouped.append(
            {"mask": mask, "label": label, "confidence": confidence, "evidence": evidence}
        )

    return grouped


def complete(mask: np.ndarray, mirror_map: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Grow a confirmed mirror to the whole mirror surface it sits in.

    Only the connected mirror regions this mask actually touches are added, so
    a room with two mirrors does not merge them into one, and a mirror the mask
    never reaches is not dragged in.

    Returns the completed mask and how many pixels were added.
    """
    if not mirror_map.any() or not mask.any():
        return mask, 0

    count, components = cv2.connectedComponents(mirror_map.astype(np.uint8), connectivity=8)

    if count <= 1:
        return mask, 0

    grown = mask.copy()

    for index in range(1, count):
        region = components == index

        if (region & mask).any():
            grown |= region

    return grown, int(grown.sum()) - int(mask.sum())
