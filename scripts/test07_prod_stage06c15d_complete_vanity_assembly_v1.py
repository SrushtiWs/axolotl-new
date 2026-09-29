
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

STAGE06 = (
    PROD
    / "stage06_prop_layer"
)

MASTER = (
    PROD
    / "stage01_master"
    / "00_master_input.png"
)


# Main vanity SAM2 stage
VANITY_RESULT = (
    STAGE06
    / "06c15c_complete_vanity_sam2"
    / "00_stage06c15c_result.json"
)


# Verified faucet
FAUCET_RESULT = (
    STAGE06
    / "06c10b_florence_faucet_sam2_duplicate_state"
    / "01_stage06c10b_result.json"
)


# Verified handle
HANDLE_RESULT = (
    STAGE06
    / "06c11c_florence_handle_sam2"
    / "00_stage06c11c_result.json"
)


# Provisional sink evidence
SINK_RESULT = (
    STAGE06
    / "06c7a_sam2_alternative_candidate_evaluation"
    / "00_sam2_candidate_evaluation.json"
)


OUT = (
    STAGE06
    / "06c15d_complete_vanity_assembly"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

OBJECTS = (
    OUT
    / "objects"
)

OBJECTS.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# HELPERS
# ============================================================

def load_json(path):

    if not path.exists():

        raise FileNotFoundError(
            path
        )

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def load_mask(
    path,
    W,
    H
):

    path = Path(
        path
    )

    if not path.exists():

        raise FileNotFoundError(
            path
        )


    mask = np.asarray(
        Image.open(
            path
        ).convert(
            "L"
        )
    )


    if mask.shape != (
        H,
        W
    ):

        raise RuntimeError(
            f"Mask size mismatch: {path} "
            f"{mask.shape} != {(H, W)}"
        )


    return (
        mask
        >
        0
    )


def overlap_fraction(
    child,
    parent
):

    area = int(
        child.sum()
    )

    if area == 0:
        return 0.0


    intersection = int(
        (
            child
            &
            parent
        ).sum()
    )


    return float(
        intersection
        /
        area
    )


def added_pixels(
    child,
    base
):

    return int(
        (
            child
            &
            ~base
        ).sum()
    )


def find_sink_024(
    data
):

    # --------------------------------------------------------
    # Search flexibly because 06C7A is an older diagnostic
    # structure.
    # --------------------------------------------------------

    rows = (
        data
        if isinstance(
            data,
            list
        )
        else data.get(
            "results",
            data.get(
                "objects",
                []
            )
        )
    )


    candidates = []


    for row in rows:

        if not isinstance(
            row,
            dict
        ):
            continue


        try:

            iid = int(
                row.get(
                    "inventory_id",
                    -1
                )
            )

        except Exception:

            continue


        if iid != 24:
            continue


        mask_path = row.get(
            "mask_path"
        )


        if not mask_path:
            continue


        candidates.append(
            row
        )


    if not candidates:

        return None


    # Prefer cluster #1 because it was the visually plausible
    # 024 sink candidate from 06C7A.
    cluster1 = [

        row

        for row in candidates

        if int(
            row.get(
                "cluster_rank",
                -1
            )
        ) == 1
    ]


    if cluster1:

        return cluster1[0]


    # Otherwise highest quality/SAM fallback.
    candidates.sort(

        key=lambda r:
            float(
                r.get(
                    "quality",
                    r.get(
                        "sam2_score",
                        r.get(
                            "sam_score",
                            0.0
                        )
                    )
                )
            ),

        reverse=True
    )


    return candidates[0]


def contours_on_preview(
    preview,
    mask,
    line_color,
    width=2
):

    draw = ImageDraw.Draw(
        preview
    )


    contours, _ = cv2.findContours(

        (
            mask.astype(
                np.uint8
            )
            *
            255
        ),

        cv2.RETR_EXTERNAL,

        cv2.CHAIN_APPROX_SIMPLE
    )


    for contour in contours:

        points = [

            (
                int(
                    p[0][0]
                ),
                int(
                    p[0][1]
                )
            )

            for p in contour
        ]


        if len(points) >= 2:

            draw.line(
                points
                +
                [
                    points[0]
                ],
                fill=line_color,
                width=width
            )


def show(
    path,
    title,
    figsize=(8, 9)
):

    img = Image.open(
        path
    )

    plt.figure(
        figsize=figsize
    )

    plt.imshow(
        img
    )

    plt.title(
        title
    )

    plt.axis(
        "off"
    )

    plt.show()


# ============================================================
# MASTER
# ============================================================

master_pil = Image.open(
    MASTER
).convert(
    "RGB"
)

master = np.asarray(
    master_pil
)

W, H = master_pil.size


# ============================================================
# MAIN VANITY MASK
# ============================================================

vanity_state = load_json(
    VANITY_RESULT
)


candidate_previews = vanity_state[
    "candidate_previews"
]


candidate1 = next(

    row

    for row in candidate_previews

    if int(
        row[
            "rank"
        ]
    ) == 1
)


main_mask = load_mask(
    candidate1[
        "mask_path"
    ],
    W,
    H
)


# ============================================================
# FAUCET
# ============================================================

faucet_state = load_json(
    FAUCET_RESULT
)


faucet_path = (
    faucet_state[
        "primary"
    ][
        "mask_path"
    ]
)


faucet_mask = load_mask(
    faucet_path,
    W,
    H
)


# ============================================================
# HANDLE
# ============================================================

handle_state = load_json(
    HANDLE_RESULT
)


handle_mask = load_mask(
    handle_state[
        "mask_path"
    ],
    W,
    H
)


# ============================================================
# SINK
# ============================================================

sink_mask = None
sink_row = None


if SINK_RESULT.exists():

    sink_data = load_json(
        SINK_RESULT
    )

    sink_row = find_sink_024(
        sink_data
    )


    if (
        sink_row is not None
        and
        sink_row.get(
            "mask_path"
        )
    ):

        sink_mask = load_mask(
            sink_row[
                "mask_path"
            ],
            W,
            H
        )


# ============================================================
# COVERAGE BEFORE ASSEMBLY
# ============================================================

before = {

    "010_faucet_coverage":
        overlap_fraction(
            faucet_mask,
            main_mask
        ),

    "021_handle_coverage":
        overlap_fraction(
            handle_mask,
            main_mask
        ),

    "010_faucet_missing_pixels":
        added_pixels(
            faucet_mask,
            main_mask
        ),

    "021_handle_missing_pixels":
        added_pixels(
            handle_mask,
            main_mask
        ),
}


if sink_mask is not None:

    before[
        "024_sink_coverage"
    ] = overlap_fraction(
        sink_mask,
        main_mask
    )

    before[
        "024_sink_missing_pixels"
    ] = added_pixels(
        sink_mask,
        main_mask
    )


# ============================================================
# BUILD ONE MAIN-PROP MASK
# ============================================================

assembled = (
    main_mask.copy()
)


# ------------------------------------------------------------
# Attached child evidence is merged INTO THE MAIN PROP.
# It is not exported as independent final layers.
# ------------------------------------------------------------

assembled |= faucet_mask
assembled |= handle_mask


if sink_mask is not None:

    assembled |= sink_mask


# ============================================================
# COMPONENT CONTRIBUTIONS
# ============================================================

contributions = {

    "main_06C15C_pixels":
        int(
            main_mask.sum()
        ),

    "010_faucet_added_pixels":
        added_pixels(
            faucet_mask,
            main_mask
        ),

    "021_handle_added_pixels":
        added_pixels(
            handle_mask,
            main_mask
        ),

    "final_assembled_pixels":
        int(
            assembled.sum()
        ),
}


if sink_mask is not None:

    base_before_sink = (
        main_mask
        |
        faucet_mask
        |
        handle_mask
    )

    contributions[
        "024_sink_added_pixels"
    ] = added_pixels(
        sink_mask,
        base_before_sink
    )


# ============================================================
# FINAL COVERAGE
# ============================================================

after = {

    "010_faucet_coverage":
        overlap_fraction(
            faucet_mask,
            assembled
        ),

    "021_handle_coverage":
        overlap_fraction(
            handle_mask,
            assembled
        ),
}


if sink_mask is not None:

    after[
        "024_sink_coverage"
    ] = overlap_fraction(
        sink_mask,
        assembled
    )


# ============================================================
# SAVE ASSEMBLED MASK
# ============================================================

MASK_PATH = (
    OBJECTS
    / "023_complete_vanity_system_mask.png"
)


Image.fromarray(
    assembled.astype(
        np.uint8
    )
    *
    255
).save(
    MASK_PATH
)


# ============================================================
# EXACT RGB RGBA
# ============================================================

rgba = np.zeros(
    (
        H,
        W,
        4
    ),
    dtype=np.uint8
)


rgba[
    :,
    :,
    :3
] = master


rgba[
    :,
    :,
    3
] = (
    assembled.astype(
        np.uint8
    )
    *
    255
)


RGBA_PATH = (
    OBJECTS
    / "023_complete_vanity_system_rgba.png"
)


Image.fromarray(
    rgba,
    mode="RGBA"
).save(
    RGBA_PATH
)


# ============================================================
# PREVIEW 1 — BASE MASK
# ============================================================

base_preview = master_pil.copy()


contours_on_preview(
    base_preview,
    main_mask,
    "lime",
    2
)


BASE_PREVIEW_PATH = (
    OUT
    / "01_before_component_completion.png"
)


base_preview.save(
    BASE_PREVIEW_PATH
)


# ============================================================
# PREVIEW 2 — COMPONENT CONTRIBUTION AUDIT
#
# green  = base main vanity
# red    = faucet child evidence
# blue   = handle child evidence
# yellow = sink evidence
# ============================================================

component_preview = master_pil.copy()


contours_on_preview(
    component_preview,
    main_mask,
    "lime",
    2
)


contours_on_preview(
    component_preview,
    faucet_mask,
    "red",
    3
)


contours_on_preview(
    component_preview,
    handle_mask,
    "blue",
    3
)


if sink_mask is not None:

    contours_on_preview(
        component_preview,
        sink_mask,
        "yellow",
        2
    )


COMPONENT_PREVIEW_PATH = (
    OUT
    / "02_component_evidence_overlay.png"
)


component_preview.save(
    COMPONENT_PREVIEW_PATH
)


# ============================================================
# PREVIEW 3 — FINAL ONE-PROP MASK
# ============================================================

final_preview = master_pil.copy()


contours_on_preview(
    final_preview,
    assembled,
    "lime",
    3
)


FINAL_PREVIEW_PATH = (
    OUT
    / "03_complete_vanity_system_preview.png"
)


final_preview.save(
    FINAL_PREVIEW_PATH
)


# ============================================================
# TRANSPARENCY PREVIEW
# ============================================================

checker = np.full(
    (
        H,
        W,
        3
    ),
    230,
    dtype=np.uint8
)


checker[
    assembled
] = master[
    assembled
]


TRANSPARENCY_PREVIEW_PATH = (
    OUT
    / "04_complete_vanity_transparency_preview.png"
)


Image.fromarray(
    checker
).save(
    TRANSPARENCY_PREVIEW_PATH
)


# ============================================================
# RESULT
# ============================================================

result = {

    "stage":
        "06C15D",

    "inventory_id":
        23,

    "inventory_name":
        "complete vanity system",

    "ownership_rule":
        "MAIN_PROP_OWNS_ATTACHED_COMPONENTS",

    "base_main_mask":
        candidate1[
            "mask_path"
        ],

    "internal_component_evidence": {

        "10_faucet":
            faucet_path,

        "21_handle":
            handle_state[
                "mask_path"
            ],

        "24_sink":
            (
                sink_row.get(
                    "mask_path"
                )
                if sink_row
                else None
            ),
    },

    "coverage_before":
        before,

    "component_contributions":
        contributions,

    "coverage_after":
        after,

    "assembled_mask_path":
        str(
            MASK_PATH
        ),

    "assembled_rgba_path":
        str(
            RGBA_PATH
        ),

    "status":
        "REQUIRES_VISUAL_FINAL_MAIN_PROP_AUDIT",

    "rules": [

        (
            "component masks are internal evidence only"
        ),

        (
            "only the assembled vanity system is eligible "
            "for final prop output"
        ),

        (
            "010,021,024 must not enter final prop union "
            "independently"
        ),

        (
            "exact RGB comes from Stage01 master"
        ),

        (
            "no global prop union performed"
        ),
    ],
}


RESULT_PATH = (
    OUT
    / "00_stage06c15d_result.json"
)


RESULT_PATH.write_text(
    json.dumps(
        result,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# ============================================================
# PRINT
# ============================================================

print("=" * 110)
print("PRODUCTION STAGE 06C15D RESULT")
print("=" * 110)


print()
print(
    "BASE MAIN MASK PIXELS:",
    contributions[
        "main_06C15C_pixels"
    ]
)


print(
    "FAUCET ADDED:",
    contributions[
        "010_faucet_added_pixels"
    ]
)


print(
    "HANDLE ADDED:",
    contributions[
        "021_handle_added_pixels"
    ]
)


print(
    "SINK ADDED:",
    contributions.get(
        "024_sink_added_pixels"
    )
)


print(
    "FINAL VANITY PIXELS:",
    contributions[
        "final_assembled_pixels"
    ]
)


print()
print(
    "COVERAGE BEFORE:"
)

for key, value in before.items():

    print(
        " ",
        key,
        "=",
        (
            round(
                value,
                4
            )
            if isinstance(
                value,
                float
            )
            else value
        )
    )


print()
print(
    "COVERAGE AFTER:"
)

for key, value in after.items():

    print(
        " ",
        key,
        "=",
        round(
            value,
            4
        )
    )


print()
print(
    "RESULT:",
    RESULT_PATH
)

print(
    "FINAL MASK:",
    MASK_PATH
)

print(
    "FINAL RGBA:",
    RGBA_PATH
)

print()
print(
    "NO GLOBAL PROP UNION WAS CREATED."
)


# ============================================================
# INLINE VISUAL VERIFICATION
# ============================================================

print()
print("=" * 110)
print("INLINE VISUAL VERIFICATION")
print("=" * 110)


show(
    BASE_PREVIEW_PATH,
    (
        "06C15D — BEFORE COMPONENT COMPLETION "
        "(BASE VANITY)"
    )
)


show(
    COMPONENT_PREVIEW_PATH,
    (
        "06C15D — COMPONENT EVIDENCE "
        "GREEN=BASE | RED=FAUCET | BLUE=HANDLE | YELLOW=SINK"
    )
)


show(
    FINAL_PREVIEW_PATH,
    (
        "06C15D — COMPLETE VANITY SYSTEM "
        "(ONE FINAL MAIN-PROP CANDIDATE)"
    )
)


show(
    TRANSPARENCY_PREVIEW_PATH,
    (
        "06C15D — VANITY EXTRACTION PREVIEW"
    )
)
