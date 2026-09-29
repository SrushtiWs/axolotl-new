
from pathlib import Path
import json

import cv2
import numpy as np

from PIL import (
    Image,
    ImageDraw
)

import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE = Path("/workspace/axolotl")

PROD = (
    BASE
    / "test07"
    / "production_pipeline"
)


MASTER = (
    PROD
    / "stage01_master"
    / "00_master_input.png"
)


STAGE05B = (
    PROD
    / "stage05_clean_room_with_props"
    / "08_neutral_prop_detection_master.png"
)


PROTECTION = (
    BASE
    / "test07"
    / "runs"
    / "room03_bathroom"
    / "stages"
    / "01v4b_tight_behind_glass_protection"
    / "00_tight_behind_glass_protection.png"
)


# Florence result containing 015 towel-bar candidate.
C14A = (
    PROD
    / "stage06_prop_layer"
    / "06c14a_florence_multiobject_relocalization"
    / "00_stage06c14a_results.json"
)


OUT = (
    PROD
    / "stage05_clean_room_with_props"
)

OUTPUT_PATH = (
    OUT
    / "09_final_prop_detection_master.png"
)


RESTORE_MASK_PATH = (
    OUT
    / "09_independent_behind_glass_restore_mask.png"
)


EXCLUDED_MASK_PATH = (
    OUT
    / "09_glass_owned_excluded_mask.png"
)


STATE_PATH = (
    OUT
    / "09_final_prop_detection_master_state.json"
)


# ============================================================
# HELPERS
# ============================================================

def bbox_intersection_fraction(
    component_bbox,
    target_bbox,
):

    cx1, cy1, cx2, cy2 = component_bbox
    tx1, ty1, tx2, ty2 = target_bbox

    ix1 = max(cx1, tx1)
    iy1 = max(cy1, ty1)
    ix2 = min(cx2, tx2)
    iy2 = min(cy2, ty2)

    iw = max(
        0.0,
        ix2 - ix1
    )

    ih = max(
        0.0,
        iy2 - iy1
    )

    intersection = iw * ih

    component_area = max(
        1.0,
        (cx2 - cx1)
        *
        (cy2 - cy1)
    )

    return float(
        intersection
        /
        component_area
    )


def mask_overlap_fraction(
    component_mask,
    bbox,
):

    H, W = component_mask.shape

    x1, y1, x2, y2 = [
        int(round(v))
        for v in bbox
    ]

    x1 = max(
        0,
        min(W - 1, x1)
    )

    y1 = max(
        0,
        min(H - 1, y1)
    )

    x2 = max(
        x1 + 1,
        min(W, x2)
    )

    y2 = max(
        y1 + 1,
        min(H, y2)
    )

    target = np.zeros(
        (H, W),
        dtype=bool
    )

    target[
        y1:y2,
        x1:x2
    ] = True

    area = int(
        component_mask.sum()
    )

    if area == 0:
        return 0.0

    overlap = int(
        (
            component_mask
            &
            target
        ).sum()
    )

    return float(
        overlap
        /
        area
    )


def draw_mask_contours(
    image,
    mask,
    color,
    width=2
):

    result = image.copy()

    draw = ImageDraw.Draw(
        result
    )

    contours, _ = cv2.findContours(
        (
            mask.astype(np.uint8)
            *
            255
        ),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:

        pts = [
            (
                int(p[0][0]),
                int(p[0][1])
            )
            for p in contour
        ]

        if len(pts) >= 2:

            draw.line(
                pts + [pts[0]],
                fill=color,
                width=width
            )

    return result


def show(
    image,
    title,
    figsize=(8, 9)
):

    plt.figure(
        figsize=figsize
    )

    plt.imshow(
        image
    )

    plt.title(
        title
    )

    plt.axis(
        "off"
    )

    plt.show()


# ============================================================
# VALIDATE
# ============================================================

for path in [
    MASTER,
    STAGE05B,
    PROTECTION,
    C14A,
]:

    if not path.exists():

        raise FileNotFoundError(
            path
        )


# ============================================================
# LOAD IMAGES
# ============================================================

master_pil = Image.open(
    MASTER
).convert(
    "RGB"
)

stage05b_pil = Image.open(
    STAGE05B
).convert(
    "RGB"
)


master = np.asarray(
    master_pil
)

stage05b = np.asarray(
    stage05b_pil
)


H, W = master.shape[:2]


if stage05b.shape != master.shape:

    raise RuntimeError(
        "Stage05B size differs from Stage01."
    )


protection = (
    np.asarray(
        Image.open(
            PROTECTION
        ).convert(
            "L"
        )
    )
    >
    0
)


if protection.shape != (
    H,
    W
):

    raise RuntimeError(
        "Protection mask size mismatch."
    )


# ============================================================
# LOAD 015 GLASS-OWNED TOWEL-BAR LOCALIZATION
# ============================================================

c14a = json.loads(
    C14A.read_text(
        encoding="utf-8"
    )
)


row015 = next(
    row
    for row in c14a
    if int(
        row[
            "inventory_id"
        ]
    ) == 15
)


candidate015 = next(
    c
    for c in row015[
        "candidates"
    ]
    if int(
        c[
            "candidate_index"
        ]
    ) == 1
)


glass_hardware_bbox = [
    float(v)
    for v in candidate015[
        "bbox"
    ]
]


print("=" * 110)
print("PRODUCTION STAGE 05C")
print("RESTORE INDEPENDENT BEHIND-GLASS PROPS")
print("=" * 110)

print()
print(
    "PROTECTION PIXELS:",
    int(
        protection.sum()
    )
)

print(
    "015 GLASS-HARDWARE BBOX:",
    [
        round(v, 1)
        for v in glass_hardware_bbox
    ]
)


# ============================================================
# CONNECTED COMPONENT ANALYSIS
# ============================================================

num_labels, labels, stats, _ = (
    cv2.connectedComponentsWithStats(
        protection.astype(
            np.uint8
        ),
        connectivity=8
    )
)


components = []


for label_id in range(
    1,
    num_labels
):

    area = int(
        stats[
            label_id,
            cv2.CC_STAT_AREA
        ]
    )

    if area <= 0:
        continue


    x = int(
        stats[
            label_id,
            cv2.CC_STAT_LEFT
        ]
    )

    y = int(
        stats[
            label_id,
            cv2.CC_STAT_TOP
        ]
    )

    w = int(
        stats[
            label_id,
            cv2.CC_STAT_WIDTH
        ]
    )

    h = int(
        stats[
            label_id,
            cv2.CC_STAT_HEIGHT
        ]
    )


    bbox = [
        x,
        y,
        x + w,
        y + h,
    ]


    component_mask = (
        labels
        ==
        label_id
    )


    overlap_with_glass_hardware = (
        mask_overlap_fraction(
            component_mask,
            glass_hardware_bbox
        )
    )


    bbox_overlap = (
        bbox_intersection_fraction(
            bbox,
            glass_hardware_bbox
        )
    )


    # --------------------------------------------------------
    # Ownership rule
    #
    # If the protected connected component substantially
    # overlaps the independently localized glass-mounted
    # towel bar, Stage04 owns it.
    #
    # Otherwise it is independent behind-glass prop evidence.
    # --------------------------------------------------------

    glass_owned = bool(
        overlap_with_glass_hardware
        >=
        0.25
        or
        bbox_overlap
        >=
        0.25
    )


    components.append({

        "component_id":
            int(label_id),

        "area":
            area,

        "bbox":
            bbox,

        "mask":
            component_mask,

        "overlap_with_015":
            overlap_with_glass_hardware,

        "bbox_overlap_with_015":
            bbox_overlap,

        "ownership":
            (
                "STAGE04_GLASS_SYSTEM"
                if glass_owned
                else
                "STAGE06_INDEPENDENT_PROP"
            ),
    })


# ============================================================
# BUILD RESTORE / EXCLUSION MASKS
# ============================================================

restore_mask = np.zeros(
    (H, W),
    dtype=bool
)

excluded_mask = np.zeros(
    (H, W),
    dtype=bool
)


for item in components:

    if (
        item[
            "ownership"
        ]
        ==
        "STAGE04_GLASS_SYSTEM"
    ):

        excluded_mask |= item[
            "mask"
        ]

    else:

        restore_mask |= item[
            "mask"
        ]


# ============================================================
# RESTORE EXACT STAGE01 RGB
# ============================================================

final = stage05b.copy()


final[
    restore_mask
] = master[
    restore_mask
]


# ============================================================
# SAVE
# ============================================================

Image.fromarray(
    restore_mask.astype(
        np.uint8
    )
    *
    255
).save(
    RESTORE_MASK_PATH
)


Image.fromarray(
    excluded_mask.astype(
        np.uint8
    )
    *
    255
).save(
    EXCLUDED_MASK_PATH
)


Image.fromarray(
    final
).save(
    OUTPUT_PATH
)


# ============================================================
# COMPONENT JSON
# ============================================================

component_json = []


for item in components:

    component_json.append({

        "component_id":
            item[
                "component_id"
            ],

        "area":
            item[
                "area"
            ],

        "bbox":
            item[
                "bbox"
            ],

        "overlap_with_015":
            item[
                "overlap_with_015"
            ],

        "bbox_overlap_with_015":
            item[
                "bbox_overlap_with_015"
            ],

        "ownership":
            item[
                "ownership"
            ],
    })


state = {

    "stage":
        "05C",

    "purpose":
        (
            "final neutral Stage06 detection master with "
            "independent behind-glass props restored"
        ),

    "source_stage05b":
        str(
            STAGE05B
        ),

    "rgb_source":
        str(
            MASTER
        ),

    "protection_source":
        str(
            PROTECTION
        ),

    "glass_hardware_reference":
        {
            "inventory_id":
                15,

            "bbox":
                glass_hardware_bbox,
        },

    "components":
        component_json,

    "restore_pixels":
        int(
            restore_mask.sum()
        ),

    "excluded_glass_owned_pixels":
        int(
            excluded_mask.sum()
        ),

    "output":
        str(
            OUTPUT_PATH
        ),

    "usage_rule":
        {
            "stage06_detection_source":
                str(
                    OUTPUT_PATH
                ),

            "final_rgb_source":
                str(
                    MASTER
                ),
        },

    "status":
        "REQUIRES_VISUAL_AUDIT",
}


STATE_PATH.write_text(
    json.dumps(
        state,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# ============================================================
# PRINT COMPONENT STATE
# ============================================================

print()
print("-" * 110)
print("CONNECTED COMPONENTS")
print("-" * 110)


for item in components:

    print(
        "component #{:02d} | area={} | bbox={} | overlap015={:.4f} | {}".format(

            item[
                "component_id"
            ],

            item[
                "area"
            ],

            item[
                "bbox"
            ],

            item[
                "overlap_with_015"
            ],

            item[
                "ownership"
            ],
        )
    )


print()
print(
    "RESTORED PIXELS:",
    int(
        restore_mask.sum()
    )
)

print(
    "GLASS-OWNED EXCLUDED PIXELS:",
    int(
        excluded_mask.sum()
    )
)


print()
print(
    "FINAL STAGE05 DETECTION MASTER:",
    OUTPUT_PATH
)

print(
    "STATE:",
    STATE_PATH
)


# ============================================================
# INLINE AUDIT PREVIEW 1
# COMPONENT OWNERSHIP
#
# green = restored into Stage05
# red   = Stage04 glass-owned; stays removed
# ============================================================

ownership_preview = master_pil.copy()


ownership_preview = draw_mask_contours(
    ownership_preview,
    restore_mask,
    "lime",
    3
)


ownership_preview = draw_mask_contours(
    ownership_preview,
    excluded_mask,
    "red",
    3
)


show(
    ownership_preview,
    (
        "Stage05C Ownership Audit "
        "GREEN=Restore Independent Props | "
        "RED=Keep With Glass"
    )
)


# ============================================================
# INLINE AUDIT PREVIEW 2
# BEFORE
# ============================================================

show(
    stage05b_pil,
    (
        "BEFORE — Stage05B "
        "(Independent Shower Fixture Missing)"
    )
)


# ============================================================
# INLINE AUDIT PREVIEW 3
# AFTER
# ============================================================

show(
    Image.fromarray(
        final
    ),
    (
        "AFTER — Stage05C Final Prop-Detection Master"
    )
)


# ============================================================
# INLINE AUDIT PREVIEW 4
# RESTORED PIXELS ONLY
# ============================================================

isolated = np.full_like(
    master,
    235
)


isolated[
    restore_mask
] = master[
    restore_mask
]


show(
    Image.fromarray(
        isolated
    ),
    (
        "Stage05C — Restored Independent Behind-Glass Props Only"
    )
)


print()
print(
    "NO AI MODEL WAS RUN."
)

print(
    "NO STAGE06 PROP MASK WAS CREATED."
)
