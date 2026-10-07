"""
Everything tunable about the extraction pipeline, in one place.

The prompt list and the thresholds live here rather than inside the detector or
the segmenter so that widening what the system looks for never means touching
inference code. `load()` merges an optional JSON override file on top of these
defaults, so a deployment can retune without a code change.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path

# ---------------------------------------------------------------------------
# What Grounding DINO is asked to look for.
# ---------------------------------------------------------------------------

# Order is not significant; duplicates across phrases are expected and are
# resolved downstream by the duplicate-detection pass, not by pruning this list.
# "sofa", "couch" and "large sofa" may all fire on one physical sofa — that is
# the point of a permissive prompt list, and filtering.deduplicate collapses them.
OBJECT_PROMPTS: tuple[str, ...] = (
    "chair",
    "sofa",
    "couch",
    "table",
    "curtain",
    "coffee table",
    "dining table",
    "dining chair",
    "bed",
    "bedside table",
    "cabinet",
    "cupboard",
    "wardrobe",
    "drawer",
    "shelf",
    "bookshelf",
    "tv",
    "television",
    "tv unit",
    "desk",
    "stool",
    "bench",
    "ottoman",
    "plant",
    "flower pot",
    "lamp",
    "floor lamp",
    "table lamp",
    "fan",
    "air conditioner",
    "refrigerator",
    "washing machine",
    "door",
    "window",
    "curtain",
    "blind",
    "mirror",
    "wall mirror",
    "bathroom mirror",
    "full length mirror",
    "mirror partition",
    "decorative mirror",
    "mirror door",
    "mirrored window",
    "mirror panel",
    "mirrored wardrobe",
    "clock",
    "wall clock",
    "chandelier",
    "pendant light",
    "ceiling fan",
    "vase",
    "side table",
    "toilet",
    "sink",
    "basin",
    "bathtub",
    "shower",
    "vanity",
    "rug",
    "carpet",
    "decor",
    "picture",
    "painting",
    "wall art",
    "artwork",
    "wall decoration",
    "oven",
    "microwave",
    "dishwasher",
    "kitchen cabinet",
    "rack",
    "partition",
    "vanity mirror",
    "framed mirror",
    "mirror cabinet",
    "mirrored partition",
    "mirrored wall panel",
    "mirrored door",
    "mirror window",
    "mirror wall",
    "matte window",
    "cross door",
    "cloth",
    "bottle",
    "plant",
    "potted plant",
    "kitchen plateform",
    "mirror stand"
    "wall panel",
    "chair legs",
    "table legs",

# --- soft furnishings -------------------------------------------------
    #
    # These were the largest measured gap in the list. A sofa's cushions are
    # not part of "sofa" as SAM segments it — it cuts them out — so with no
    # phrase of their own nothing ever proposes them, no mask is produced, and
    # they reach ALL_OBJECTS.png as holes in the sofa. Measured on a living
    # room: zero `pillow`/`cushion` proposals out of 71, because neither word
    # was ever asked for.
    "pillow",
    "cushion",
    "throw pillow",
    "seat cushion",
    "blanket",
    "throw",
    # --- plants -----------------------------------------------------------
    #
    # "plant" and "flower pot" were present; a large indoor tree is often
    # neither to the detector.
    "tree",
    "potted plant",
    "indoor tree",
    "planter",
    # --- decor and small objects ------------------------------------------
    #
    # Small but visually distinct things that stand on the furniture this
    # pipeline already finds, and which otherwise become holes in it for the
    # same reason the cushions did.
    "basket",
    "tray",
    "bowl",
    "book",
    "books",
    "candle",
    "sculpture",
    "ornament",
    "bottle",
)

# Which of the above name a mirror. A detection carrying any of these labels
# goes to MIRRORS_ONLY.png and is kept out of ALL_OBJECTS.png.
#
# A mirrored *thing* is a mirror: a mirror door, a mirrored window and a
# mirrored wardrobe all reflect the room and all belong in the mirror layer. A
# plain "door" or "window" does not, and stays with the ordinary objects.
#
# That is why these are matched as whole phrases and never as substrings. A
# substring test would send every door in the building to the mirror layer the
# moment "mirror door" joined this list.
MIRROR_PROMPTS: frozenset[str] = frozenset(
    {
        "mirror",
        "wall mirror",
        "bathroom mirror",
        "full length mirror",
        "mirror partition",
        "decorative mirror",
        "mirror door",
        "mirrored window",
        "mirror panel",
        "mirrored wardrobe",
        "vanity mirror",
        "framed mirror",
        "mirror cabinet",
        "mirrored partition",
        "mirrored wall panel",
        "mirrored door",
    }
)

# Detections that legitimately *are* glazing, and so keep their own glass.
#
# Glazing found inside any other object's mask is the window behind it — the
# view between a plant's leaves, say — and is trimmed out. These labels are
# exempt from that, or a detected window would be erased by the very class that
# describes it. A "mirrored window" is absent on purpose: it is handled as a
# mirror, not as a window.
GLAZING_LABELS: frozenset[str] = frozenset({"window", "windowpane", "glass", "door", "blind"})

# Never objects, whatever a model calls them. These are the surfaces the tile
# renderer owns; lifting one out would delete the very thing being retiled.
STRUCTURAL_LABELS: frozenset[str] = frozenset(
    {
        "floor",
        "wall",
        "ceiling",
        "roof",
        "room surface",
        "tile",
        "grout",
        "shadow",
        "lighting",
        "reflection",
        "texture",
        "sky",
        "ground",
    }
)


@dataclass(frozen=True)
class DetectionConfig:
    """Grounding DINO thresholds and duplicate handling."""

    # A box scoring below this on its best phrase is not a detection.
    #
    # Lower than the 0.35 usually quoted for Grounding DINO, and deliberately.
    # The reference implementation isolates each phrase with a text
    # self-attention mask; the ONNX export has no such input, so all the
    # prompts share one sequence and every score is diluted by how many
    # phrases are being asked about at once. Measured on this export: a clearly
    # visible chair in a real photo scores 0.58, while correct-but-ordinary
    # detections land between 0.17 and 0.27. A 0.35 threshold returns nothing
    # at all. Retune this if the prompt list changes size substantially.
    box_threshold: float = 0.15

    # Floor on a phrase's strongest word piece. Stops a phrase whose words are
    # each weakly present from accumulating its way past `box_threshold`.
    text_threshold: float = 0.10

    # Two boxes overlapping by more than this are candidates for merging.
    # Applied class-agnostically: "sofa" and "couch" on one object must collapse.
    duplicate_iou: float = 0.65

    # Containment is the other half of duplicate handling. "large sofa" can sit
    # wholly inside "sofa" with a modest IoU, so intersection-over-smaller
    # catches what IoU alone misses.
    duplicate_containment: float = 0.80

    # How similar in size two boxes must be before containment is allowed to
    # merge them: the smaller must be at least this fraction of the larger.
    #
    # Containment on its own cannot tell "one phrase fired twice on one object"
    # from "two different objects, one in front of the other". Measured on a
    # bedroom photo, Grounding DINO boxed the bed as "chair" at 135,960 px and
    # an armchair standing in front of it as "chair" at 5,886 px. The armchair
    # scored higher, contained-ness was 1.0, the labels matched — so the bed,
    # the largest object in the room, was merged away and never reached SAM.
    #
    # Size is what separates the two readings. One phrase firing twice on one
    # object gives boxes of comparable extent; a box 23x the area of the one it
    # contains is a different object. Set low enough to keep merging the case
    # this rule exists for ("large sofa" inside "sofa") and high enough that a
    # chair cannot swallow a bed.
    duplicate_area_ratio: float = 0.35

    # How much of an object's mask must be ADE20K "mirror" pixels for it to be
    # a mirror on that evidence alone, whatever Grounding DINO called it.
    #
    # This is what keeps a mirror out of ALL_OBJECTS.png even when the detector
    # names it something else entirely.
    mirror_pixel_share: float = 0.50

    # The same test, relaxed, for a detection Grounding DINO *also* called a
    # mirror. Two independent models agreeing needs less pixel evidence than
    # one model asserting it by itself.
    mirror_corroborated_share: float = 0.20

    # How much real ADE20K `mirror` evidence a mask needs before glazing is
    # allowed to top it up towards `mirror_corroborated_share`.
    #
    # Glazing support exists because a mirrored wall panel is often read as
    # "glass" rather than "mirror" — but a plain window is *entirely* glazing,
    # so letting glazing stand alone makes every window a mirror the moment a
    # mirror phrase lands on it. Measured across four rooms, that is exactly
    # what happened: a kitchen window, a dining window and a set of glass patio
    # doors all reached MIRRORS_ONLY.png on 100%, 55% and 95% "reflective
    # surface" with **zero** mirror pixels between them.
    #
    # The two genuine mirrors in the same set measured 0.998 and 0.249 on the
    # mirror class, and the four false ones measured 0.000. So glazing may
    # supplement mirror evidence but never replace it: a surface that reads as
    # no mirror at all is not a mirror, however reflective the glass is.
    mirror_glazing_needs_mirror_share: float = 0.05

    # A detection lying this far inside the mirror surface that the detector did
    # *not* call a mirror is something seen in the mirror, not something
    # standing in the room, and is dropped rather than written to either layer.
    #
    # Measured: the test room's mirror produced a "chair" detection whose mask
    # was 100% mirror pixels and whose box matched the mirror exactly — a chair
    # reflected in it. Promoting that to the mirror layer duplicates the mirror;
    # keeping it as an object puts a phantom chair over the retiled floor.
    mirror_reflection_share: float = 0.60

    # How many phrases may share one forward pass.
    #
    # This is the single biggest lever on recall, and the reason is the export.
    # The reference implementation isolates each phrase with a text
    # self-attention mask; this ONNX export has no such input, so every prompt
    # shares one sequence and each phrase's score is diluted by how many others
    # are being asked about at once. The longer the prompt list, the more every
    # score collapses — so a *permissive* prompt list silently costs recall,
    # which is the opposite of what it is for.
    #
    # Measured on one bedroom photo, varying only this value:
    #
    #     phrases/pass   raw detections   "window"   "lamp"   "coffee table"
    #     64 (one pass)       27             --        --          --
    #     16                  48           0.605       --        0.164
    #      8                  68           0.654     0.696       0.174
    #      4                  84           0.707     0.683        --
    #
    # A lamp that scores 0.696 on its own is invisible at 0.15 when 64 prompts
    # share the sequence. The bed, the lamps, the artwork and the glazing were
    # all missing from that room for this reason alone.
    #
    # The cost is linear and real: the vision backbone re-runs every pass, at
    # roughly 7 seconds each on CPU regardless of how many phrases the pass
    # carries. Eight is where recall stops improving much and latency is still
    # tolerable; raise it to trade detections for speed.
    max_phrases_per_pass: int = 8

    # Sliced inference (SAHI), for objects too small to survive the resize.
    #
    # Grounding DINO's export takes a fixed 800x800 image, so *every* photo
    # wider than that is downsampled before the model sees it — a 1600px working
    # image loses half its linear detail, and a lamp 30 px across arrives at 15.
    # Running the detector again on overlapping crops puts those objects back at
    # something near their original scale, which is the only way the small and
    # distant ones become visible to it at all.
    #
    # The whole frame is always scanned first; slices are additional evidence,
    # and `filtering.deduplicate` merges what both passes find. Overlap exists
    # so an object straddling a seam is whole in at least one slice.
    #
    # This multiplies detection time by (1 + rows*cols) — measured at 62s -> 273s
    # on a 4000x2668 upload — so the gate matters as much as the mechanism.
    #
    # It is set just under `max_inference_edge`, which is what makes it a test
    # for "this photo was high-resolution": the working copy only reaches this
    # size when the original was at least this big, and that is exactly when the
    # downscale to 800 has thrown detail away. Measured, the same 2x2 slicing on
    # two rooms:
    #
    #     4000x2668 (working 1600, a 2.5x downscale)   7 -> 62 small objects
    #     1280x864  (working 1280, no downscale)      20 -> 78 small objects
    #
    # The second already found its small objects without slicing, and paying 5x
    # the detection time for it is not worth it; the first could not. So photos
    # at or above this size slice, and ordinary ones keep their current speed.
    sahi_min_edge: int = 1500
    sahi_rows: int = 2
    sahi_columns: int = 2
    sahi_overlap: float = 0.2

    # A slice is an arbitrary crop, so a box filling one carries no information
    # about where an object ends — it is the crop's own edge. Such boxes are the
    # characteristic false positive of sliced inference and are dropped.
    sahi_max_box_tile_fraction: float = 0.92

    # Detections are considered largest-score-first and capped here, so a busy
    # photo cannot turn into a hundred SAM calls.
    #
    # Raised from 40 when sliced inference arrived, but not for the reason that
    # sounds obvious. Slicing produces far more raw detections — 73 to 275 on
    # one photo — and the worry was that this cap, which sorts by confidence,
    # would discard the small objects slicing exists to find, since those score
    # lowest. Measured, it does not: those 275 collapse to 43 unique detections
    # in `deduplicate`, because the frame pass and four overlapping slices each
    # see the same objects. The cap was costing 3 detections and one small
    # object, not dozens.
    #
    # 80 is set so the cap stops binding on a sliced photo at all, which is
    # cheap: what reaches SAM is the deduplicated count, and that is bounded by
    # how many objects are really in the room.
    max_detections: int = 80

    # A box covering more of the frame than this is the room, not an object.
    max_box_frame_fraction: float = 0.92

    # A box smaller than this fraction of the frame is noise.
    min_box_frame_fraction: float = 0.0008


@dataclass(frozen=True)
class SegmentationConfig:
    """SAM 2 mask selection and validation (the measurable heuristics)."""

    # Mask logits above this are foreground. SAM 2 emits raw logits, so 0.0 is
    # the natural decision boundary.
    mask_logit_threshold: float = 0.0

    # A mask must keep at least this fraction of its area inside the DINO box
    # that prompted it. Below that, SAM has wandered off the object.
    min_inside_box: float = 0.55

    # A mask filling less than this fraction of its prompting box is a fragment
    # (a cushion instead of the sofa), not the object that was asked for.
    min_mask_to_bbox: float = 0.08

    # A mask covering more of the frame than this is SAM escaping into the room.
    max_frame_fraction: float = 0.85

    # Absolute floor on a mask, in pixels, before anything else is considered.
    min_mask_pixels: int = 160

    # How the three SAM candidates are ranked: agreement with the prompting box
    # against SAM's own predicted IoU.
    box_agreement_weight: float = 0.6

    # A mask whose pixels are this fraction confident wall/floor/ceiling is the
    # surface itself, not an object standing on it.
    max_surface_overlap: float = 0.75

    # How strongly surface overlap counts against a candidate when choosing
    # between SAM's three. At 0.5 a mask that is entirely wall ranks at half
    # what it otherwise would, which is enough to lose to a clean alternative
    # and not enough to matter when every candidate sits against a surface.
    surface_rank_penalty: float = 0.5

    # How large a mask must be, as a fraction of the frame, before high surface
    # overlap is allowed to reject it at all.
    #
    # Overlap alone cannot tell "this mask IS the wall" from "this mask is an
    # object standing against the wall", because the surface model labels both
    # the same way. On a bare kitchen it called the countertop 48% floor / 46%
    # wall and the counter supports 56% wall / 21% floor — the fixtures were
    # indistinguishable from the room by class.
    #
    # Size separates them. The case this veto exists for, a polished floor that
    # Grounding DINO called "table", was 19.6% of the frame at 99.6% surface.
    # The kitchen's counter supports were 0.6-1.5% of the frame at 90-100%
    # surface. A whole wall or floor is large by definition; a fixture in front
    # of one is not, however much surface the model thinks it sees there.
    surface_veto_min_frame: float = 0.05

    # ...but a mask this far into being a surface is one at any size, and the
    # size guard above must not shelter it.
    #
    # The guard exists so a small fixture standing against a wall is not
    # condemned for reading as wall. It was doing more than that: measured
    # across 517 accepted objects, five masks of 90-100% wall/floor/ceiling had
    # been accepted purely because each covered slightly under 5% of its frame
    # — among them a 504,391 px mask that is 100% surface and a 383,739 px one
    # at 98.9%. Those are slabs of wall in ALL_OBJECTS.png, and the clean room
    # had them cut out of it.
    #
    # 0.90 is chosen from the measured gap rather than picked: over those same
    # 517 objects the highest-scoring *real* object is a kitchen counter at
    # 0.743 — dark granite against a dark tiled wall — and the next value above
    # it is 0.903. Nothing legitimate sits between.
    surface_veto_always: float = 0.90

    # The same idea applied to a detection *box*, before SAM has run — and
    # deliberately far stricter, because a box is a far weaker piece of
    # evidence than a mask.
    #
    # A box tightly around a real object standing against a wall is mostly wall
    # even when the object is perfectly detected: the box contains the object
    # plus all the surface visible around it. Measured on a bare kitchen, the
    # box test at 0.75 rejected 18 of 19 detections — the sink, the countertop
    # and the window included — leaving a single 2,382-pixel fragment, because
    # every fixture in that room is flush against tiled walls.
    #
    # So this bar is set where only a box that is *essentially nothing but*
    # surface fails it, which is the case it was introduced for: a polished
    # floor that Grounding DINO called "table", whose box measured 100% floor.
    # Deciding whether a detection is really a surface is left to the mask
    # test above, which is where the evidence actually is.
    max_box_surface_overlap: float = 0.98

    # How far the structural map is eroded before it is used to trim or reject.
    # Only pixels well inside a surface count, so an object standing against a
    # wall keeps its own edge.
    surface_guard_px: int = 3

    # Which surfaces may have pixels trimmed out of an accepted mask.
    #
    # All three, because a cut-out picks up flaps of wall off a sofa arm and
    # strips of ceiling above a wardrobe just as readily as it picks up floor.
    #
    # This was briefly floor-only. Including "wall" used to punch holes through
    # a beige headboard that SegFormer had misread as wall, and restricting the
    # trim was the quickest way to stop that. `trim_confidence_margin` is the
    # better fix — it protects the headboard's confident interior directly — so
    # the wall and ceiling can be trimmed again without that cost.
    trim_surfaces: tuple[str, ...] = ("floor", "wall", "ceiling")

    # How deep into a mask the surface trim may reach, as a fraction of the
    # photo's longest edge, with a floor in pixels so it stays meaningful on a
    # small image.
    #
    # This is what separates contamination from object. A flap of wall hanging
    # off a sofa arm or a strip of floor under a chair is *at the edge* of the
    # mask by construction; a patch of headboard the surface model misread as
    # wall is deep inside it. Restricting the trim to a band around the boundary
    # removes the first without touching the second.
    #
    # SAM's own confidence was measured as an alternative and rejected: it left
    # more contamination than this does on every object tested (chair 35 surface
    # pixels against 0, plant 747 against 0, drawer 638 against 31) because much
    # of the floor a mask bleeds onto is floor SAM is confident about.
    trim_edge_band_fraction: float = 0.0075
    trim_edge_band_min_px: int = 6

    # Trimming that removes more than this much of a mask is disagreement about
    # the object rather than its edge, and is reverted.
    min_surviving_after_trim: float = 0.45

    # A note for anyone who looks at a cut-out and sees the trim's staircase.
    #
    # The trim cuts on SegFormer's map, which is decided far below the photo's
    # resolution, so where the cut runs along a real edge it runs along it
    # roughly — measured at a 4-8 px staircase biting into a plant pot standing
    # against a dark wall. Undoing that needs a rule for telling a strip the
    # trim over-cut from a strip it correctly removed, and three were built and
    # measured on that room:
    #
    #   re-snap from the untrimmed mask with the colour guided filter
    #       fixed the pot, and put 3,512 confident floor pixels back into the
    #       layer, because a flat strip of floor has no gradient for the filter
    #       to stop at.
    #   re-snap from the trimmed edge
    #       safe (+161 surface pixels) and almost useless: 203 pixels returned
    #       and the staircase stayed.
    #   compare the photo's gradient along the cut against the gradient just
    #   outside the strip
    #       unsound. The floor strip under a skirting board scored 4.92 — its
    #       cut is a soft shadow line and the floor outside it is full of grout
    #       joints — while the pot strip that should have returned scored 0.38.
    #       The signal points the wrong way on both real cases.
    #
    # So the staircase stays. Putting floor and wall pixels into an object PNG
    # is the worse error, and none of the three rules avoided it while actually
    # fixing the edge.


@dataclass(frozen=True)
class CleanupConfig:
    """OpenCV post-processing. Cleanup, never shrinking."""

    # Components below this fraction of the mask's own area are debris.
    min_component_fraction: float = 0.02

    # Enclosed gaps up to this fraction of the mask may be filled.
    #
    # Size is the weak half of this rule and was never the half doing the work:
    # what tells a pinhole from a real gap is *what is behind it*, and
    # `fill_small_holes` already refuses any gap that is mostly room surface.
    # The cap used to sit at 0.05, which blocked the case it should have
    # allowed. Measured on a living room, every enclosed gap fell cleanly into
    # one of two kinds:
    #
    #   272 px, 95% floor      a gap between a table's legs and braces
    #    90 px, 87% wall       gaps behind a cabinet
    #   223 px, 88% wall
    #    62 px, 87% wall
    #  1459 px, 0% floor/wall  objects standing on another object
    #
    # The last one is the defect: a vase and books resting on a table are cut
    # out of the table's mask by SAM and are not detected as objects of their
    # own, so the layer ended up with a hole where they stand and the room
    # showed through the furniture. At 5.58% of its mask it was refused by the
    # old cap for being large, when being large is exactly what it is.
    #
    # Everything a gap of this kind contains is a movable object at its own
    # coordinates, so sealing it writes those objects' own pixels into the
    # layer, which is what the layer is for.
    max_hole_fraction: float = 0.35

    # Radius for the closing that seals hairline breaks. Deliberately small:
    # closing only adds pixels, so this cannot eat a chair leg, and a large
    # radius would bridge genuinely separate objects.
    close_radius: int = 2

    # Radius for the opening that removes single-pixel spurs. Smaller than the
    # closing radius, because opening *removes* pixels and the requirement is
    # cleanup, not erosion.
    open_radius: int = 1

    # Contours shorter than this many points are not object outlines.
    min_contour_points: int = 5


@dataclass(frozen=True)
class OverlapConfig:
    """How two object masks that claim the same pixels are settled."""

    # Below this overlap the two masks are simply neighbours and are left alone.
    contest_threshold: float = 0.15

    # Quality difference needed for the better mask to take contested pixels.
    # Within this margin the masks are left as they are rather than risking a
    # bite out of a real boundary.
    decisive_margin: float = 0.08

    # A weaker mask with this much of itself inside the stronger one is not a
    # second object, it is the same object detected twice. Measured on a room
    # with one floor lamp: three separate "lamp" detections each sat 94-99%
    # inside the winning "pendant light" mask, were carved down to 292, 120 and
    # 2 pixels, and those crumbs were then written into ALL_OBJECTS.png as
    # specks beside the lamp. Worse, the mask that lost held the lamp's pole,
    # which the winner did not, so the pole disappeared from the output.
    #
    # Above this share the two masks are unioned instead: one object, whole,
    # with nothing of either version thrown away.
    merge_containment: float = 0.60

    # ...and below that, how much of a mask may be carved away at all. In the
    # same room a plant stood in front of a window: 23% of the plant was
    # contested and taken, which removed a piece of the pot's rim — a visible
    # bite out of a real boundary. Both masks composite into the same layer, so
    # sharing those pixels costs nothing, while cutting them costs the edge.
    max_bite: float = 0.10

    # A non-mirror detection lying this far inside a mirror is the room seen in
    # the mirror, not a second piece of furniture standing where the mirror is.
    reflection_containment: float = 0.70


@dataclass(frozen=True)
class ResidualConfig:
    """
    The sweep for objects Grounding DINO never proposed.

    Every failure that survives into `CLEAN_ROOM.png` has the same shape: the
    object was not in the mask, so `inpaint` never had a hole to fill there. A
    floor lamp is the clearest case — the detector boxes the *shade*, because
    that is what reads as "lamp", and SAM is then held inside that box by
    `min_inside_box`. The arm and the pole are outside it and structurally
    unreachable, so they stay in the emptied room as a dark line on the wall.

    Growing the prompt box and re-running SAM was tried first and does not work:
    the shade's mask fills the shade's box on all four sides, so there is no
    direction to grow, and expanding anyway let SAM escape onto the tiled wall
    (+1217% on that lamp, the added pixels almost entirely wall).

    What does work is already in hand. SegFormer labels all 150 ADE20K classes
    and this pipeline kept five. A pixel the class map calls neither a room
    surface, nor architecture, nor glazing — and which no accepted mask claims —
    is by construction something standing in the room that nothing lifted out.
    On the room above the leftovers were named exactly: 6,526 px "plant" (the
    missing leaves), 2,004 px "lamp" (the arm and pole), 1,442 px "pot", 2,309
    px "armchair" (edges the mask cut short).

    Those regions are then box-prompted through SAM so what joins the layer is a
    real mask rather than a threshold artefact. The encoder has already run for
    this photo, so the whole sweep is decoder calls — measured at 1.8s for 25
    regions, against the 66s the detector costs.
    """

    # Whether the sweep runs at all.
    enabled: bool = True

    # A residual region smaller than this is noise — compression speckle along a
    # real boundary, or a stray pixel of the class map.
    min_region_px: int = 120

    # ...and one larger than this fraction of the frame is not a prop the
    # detector missed, it is the surface model disagreeing with itself about the
    # room. Lifting that into the object layer would cut a hole in the wall.
    max_region_frame_fraction: float = 0.10

    # How much of a SAM mask grown from a residual region may be confident
    # wall/floor/ceiling before it is refused. This is the guard that stops the
    # sweep annexing the surface behind a thin object: the lamp's pole is not
    # wall, the lit patch of tile around it is.
    max_surface_share: float = 0.50

    # How much of the region SAM's mask has to actually cover before it is
    # preferred to the region itself. Below this SAM answered a different
    # question, and the class map's own region is the safer mask.
    min_region_agreement: float = 0.30

    # Closing radius applied to the raw residual before components are found, so
    # a leaf broken into three specks by the class map is one region.
    close_radius: int = 2

    # How many regions may be swept, strongest first. A bound on the decoder
    # cost for a photo whose class map is unusually fragmented.
    max_regions: int = 40


@dataclass(frozen=True)
class ExtractionConfig:
    """The whole pipeline's configuration."""

    prompts: tuple[str, ...] = OBJECT_PROMPTS
    mirror_prompts: frozenset[str] = MIRROR_PROMPTS
    structural_labels: frozenset[str] = STRUCTURAL_LABELS
    glazing_labels: frozenset[str] = GLAZING_LABELS

    detection: DetectionConfig = field(default_factory=DetectionConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    cleanup: CleanupConfig = field(default_factory=CleanupConfig)
    overlap: OverlapConfig = field(default_factory=OverlapConfig)
    residual: ResidualConfig = field(default_factory=ResidualConfig)

    # Longest edge the *models* see. The photo itself is never permanently
    # resized: every mask is mapped back to the original pixel dimensions.
    max_inference_edge: int = 1600

    debug: bool = False

    def is_mirror(self, label: str) -> bool:
        return label.strip().lower() in self.mirror_prompts

    def is_glazing(self, label: str) -> bool:
        """
        Whether a detection is itself a window or glass door.

        Such a detection keeps its own glazing; glazing found inside anything
        else is the window *behind* it, and comes out. Without this a detected
        window would be trimmed away by the very class that describes it.
        """
        return label.strip().lower() in self.glazing_labels

    def is_structural(self, label: str) -> bool:
        return label.strip().lower() in self.structural_labels

    @property
    def text_prompt(self) -> str:
        """
        The prompt list as Grounding DINO wants it: lowercase phrases separated
        by ". ", terminated. The model was trained on this exact shape.
        """
        return ". ".join(phrase.strip().lower() for phrase in self.prompts) + "."


# An override file lets a deployment retune without editing code.
OVERRIDE_PATH = Path(
    os.environ.get(
        "EXTRACTION_CONFIG",
        str(Path(__file__).resolve().parent.parent / "extraction_config.json"),
    )
)

_SECTIONS = {
    "detection": DetectionConfig,
    "segmentation": SegmentationConfig,
    "cleanup": CleanupConfig,
    "overlap": OverlapConfig,
    "residual": ResidualConfig,
}


def load(debug: bool | None = None) -> ExtractionConfig:
    """
    The active configuration: defaults, plus `extraction_config.json` if present.

    Unknown keys are ignored rather than raising — an override file that has
    drifted ahead of the code should not take the server down with it.
    """
    config = ExtractionConfig()

    if OVERRIDE_PATH.exists():
        try:
            raw = json.loads(OVERRIDE_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raw = {}

        if isinstance(raw.get("prompts"), list) and raw["prompts"]:
            config = replace(config, prompts=tuple(str(p) for p in raw["prompts"]))

        if isinstance(raw.get("mirror_prompts"), list):
            config = replace(
                config, mirror_prompts=frozenset(str(p).lower() for p in raw["mirror_prompts"])
            )

        for name, section_type in _SECTIONS.items():
            values = raw.get(name)

            if not isinstance(values, dict):
                continue

            allowed = {f for f in section_type.__dataclass_fields__}

            current = getattr(config, name)

            config = replace(
                config,
                **{name: replace(current, **{k: v for k, v in values.items() if k in allowed})},
            )

        if isinstance(raw.get("max_inference_edge"), int):
            config = replace(config, max_inference_edge=raw["max_inference_edge"])

    if debug is None:
        debug = os.environ.get("EXTRACTION_DEBUG", "").strip().lower() in {"1", "true", "yes"}

    return replace(config, debug=bool(debug))
