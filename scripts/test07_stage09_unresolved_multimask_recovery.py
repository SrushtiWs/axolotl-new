
from pathlib import Path
import argparse
import gc
import json
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw


# ============================================================
# SAM2 IMPORT
# ============================================================

for source in [
    Path("/workspace/sam2_src"),
    Path("/workspace/axolotl/sam2"),
]:
    if source.exists() and str(source) not in sys.path:
        sys.path.insert(0, str(source))

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# ============================================================
# TEST07 — STAGE09
#
# UNRESOLVED INSTANCE → SAM2 MULTIMASK RECOVERY AUDIT
#
# INPUT:
#   locked resized master
#   Stage08 final trusted foreground
#   Stage06 unresolved candidates
#
# PURPOSE:
#   Try to obtain a better object-local mask for unresolved
#   physical instances.
#
# IMPORTANT:
#   - unresolved instances only
#   - NO merge into Stage08 final mask
#   - evaluate all SAM2 multimasks
#   - keep masks spatially local
#   - preserve inventory_id
#   - Stage08 remains frozen
#   - exact RGB comes from locked master
# ============================================================


SAM2_CONFIG = (
    "configs/sam2.1/"
    "sam2.1_hiera_l.yaml"
)

LOCAL_MARGIN = 0.10

MIN_RECOVERY_PIXELS = 6
MIN_CONTAINMENT = 0.78
MAX_AREA_RATIO = 1.35
MIN_PROPOSAL_SCORE = 0.55


# ============================================================
# HELPERS
# ============================================================

def find_checkpoint(explicit=None):

    candidates = []

    if explicit:
        candidates.append(Path(explicit))

    candidates += [
        Path(
            "/workspace/axolotl/test06a_temp/"
            "models/sam2.1_hiera_large.pt"
        ),
        Path(
            "/workspace/models/"
            "sam2.1_hiera_large.pt"
        ),
        Path(
            "/workspace/sam2_src/checkpoints/"
            "sam2.1_hiera_large.pt"
        ),
    ]

    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        "SAM2 checkpoint not found."
    )


def safe_name(text):

    return "".join(
        c if c.isalnum() else "_"
        for c in str(text).lower()
    )[:50]


def clip_box(box, W, H):

    x1, y1, x2, y2 = [
        float(x)
        for x in box
    ]

    x1 = int(
        max(
            0,
            min(
                W - 1,
                round(x1)
            )
        )
    )

    y1 = int(
        max(
            0,
            min(
                H - 1,
                round(y1)
            )
        )
    )

    x2 = int(
        max(
            x1 + 1,
            min(
                W,
                round(x2)
            )
        )
    )

    y2 = int(
        max(
            y1 + 1,
            min(
                H,
                round(y2)
            )
        )
    )

    return (
        x1,
        y1,
        x2,
        y2
    )


def largest_component(mask):

    mask = mask.astype(bool)

    H, W = mask.shape

    visited = np.zeros_like(
        mask,
        dtype=bool
    )

    best = np.zeros_like(
        mask,
        dtype=bool
    )

    best_area = 0

    ys, xs = np.where(mask)

    for sx, sy in zip(xs, ys):

        if visited[sy, sx]:
            continue

        stack = [
            (
                int(sx),
                int(sy)
            )
        ]

        visited[sy, sx] = True

        coords = []

        while stack:

            x, y = stack.pop()

            coords.append(
                (
                    x,
                    y
                )
            )

            for nx, ny in (
                (x - 1, y),
                (x + 1, y),
                (x, y - 1),
                (x, y + 1)
            ):

                if (
                    0 <= nx < W
                    and
                    0 <= ny < H
                    and
                    mask[ny, nx]
                    and
                    not visited[ny, nx]
                ):

                    visited[ny, nx] = True

                    stack.append(
                        (
                            nx,
                            ny
                        )
                    )

        if len(coords) > best_area:

            best_area = len(coords)

            best[:] = False

            for x, y in coords:
                best[y, x] = True

    return best


def component_containing_point(
    mask,
    px,
    py
):

    mask = mask.astype(bool)

    H, W = mask.shape

    if not (
        0 <= px < W
        and
        0 <= py < H
    ):
        return np.zeros_like(mask)

    if not mask[py, px]:
        return np.zeros_like(mask)

    out = np.zeros_like(
        mask,
        dtype=bool
    )

    stack = [
        (
            px,
            py
        )
    ]

    out[py, px] = True

    while stack:

        x, y = stack.pop()

        for nx, ny in (
            (x - 1, y),
            (x + 1, y),
            (x, y - 1),
            (x, y + 1)
        ):

            if (
                0 <= nx < W
                and
                0 <= ny < H
                and
                mask[ny, nx]
                and
                not out[ny, nx]
            ):

                out[ny, nx] = True

                stack.append(
                    (
                        nx,
                        ny
                    )
                )

    return out


def mask_bbox(mask):

    ys, xs = np.where(mask)

    if len(xs) == 0:
        return None

    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1
    )


def score_mask(
    mask,
    box,
    sam_score,
    base
):

    x1, y1, x2, y2 = box

    box_area = max(
        1,
        (
            x2 - x1
        )
        *
        (
            y2 - y1
        )
    )

    area = int(
        mask.sum()
    )

    if area == 0:

        return (
            -999.0,
            {}
        )

    inside = int(
        mask[
            y1:y2,
            x1:x2
        ].sum()
    )

    outside = (
        area
        -
        inside
    )

    containment = (
        inside
        /
        max(
            1,
            area
        )
    )

    area_ratio = (
        area
        /
        box_area
    )

    missing = (
        mask
        &
        (~base)
    )

    missing_pixels = int(
        missing.sum()
    )

    missing_ratio = (
        missing_pixels
        /
        max(
            1,
            area
        )
    )

    mb = mask_bbox(
        mask
    )

    if mb is None:

        bbox_fill = 1.0

    else:

        mx1, my1, mx2, my2 = mb

        mb_area = max(
            1,
            (
                mx2 - mx1
            )
            *
            (
                my2 - my1
            )
        )

        bbox_fill = (
            area
            /
            mb_area
        )

    giant_penalty = 0.0

    if area_ratio > 2.0:

        giant_penalty += min(
            2.0,
            (
                area_ratio
                -
                2.0
            )
            *
            0.50
        )

    if containment < 0.75:

        giant_penalty += (
            0.75
            -
            containment
        ) * 3.0

    structural_penalty = 0.0

    if (
        area_ratio > 0.65
        and
        bbox_fill > 0.72
    ):

        structural_penalty += 0.65

    score = (

        float(
            sam_score
        )
        *
        1.40

        +

        containment
        *
        1.60

        +

        missing_ratio
        *
        0.35

        -

        abs(
            min(
                area_ratio,
                3.0
            )
            -
            0.55
        )
        *
        0.20

        -

        giant_penalty

        -

        structural_penalty
    )

    metrics = {

        "area":
            area,

        "box_area":
            box_area,

        "area_ratio":
            round(
                float(
                    area_ratio
                ),
                4
            ),

        "inside":
            inside,

        "outside":
            outside,

        "containment":
            round(
                float(
                    containment
                ),
                4
            ),

        "missing_pixels":
            missing_pixels,

        "missing_ratio":
            round(
                float(
                    missing_ratio
                ),
                4
            ),

        "bbox_fill":
            round(
                float(
                    bbox_fill
                ),
                4
            ),

        "sam_score":
            round(
                float(
                    sam_score
                ),
                4
            ),

        "stage09_score":
            round(
                float(
                    score
                ),
                4
            )
    }

    return (
        score,
        metrics
    )


# ============================================================
# MAIN
# ============================================================

def run(
    master_path,
    base_mask_path,
    unresolved_json,
    output_dir,
    checkpoint_path=None
):

    master_path = Path(
        master_path
    )

    base_mask_path = Path(
        base_mask_path
    )

    unresolved_json = Path(
        unresolved_json
    )

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    object_dir = (
        output_dir
        /
        "objects"
    )

    object_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for p in [
        master_path,
        base_mask_path,
        unresolved_json
    ]:

        if not p.exists():
            raise FileNotFoundError(p)

    checkpoint = find_checkpoint(
        checkpoint_path
    )

    # ========================================================
    # LOAD INPUTS
    # ========================================================

    master_pil = (
        Image.open(
            master_path
        )
        .convert(
            "RGB"
        )
    )

    rgb = np.asarray(
        master_pil
    )

    H, W = (
        rgb.shape[:2]
    )

    base = (
        np.asarray(
            Image.open(
                base_mask_path
            )
            .convert(
                "L"
            )
        )
        >
        127
    )

    if base.shape != (
        H,
        W
    ):
        raise RuntimeError(
            "Base mask shape mismatch."
        )

    with open(
        unresolved_json,
        "r",
        encoding="utf-8"
    ) as f:

        unresolved = json.load(
            f
        )

    print(
        "=" * 90
    )

    print(
        "TEST07 STAGE09"
    )

    print(
        "UNRESOLVED INSTANCE MULTIMASK RECOVERY"
    )

    print(
        "=" * 90
    )

    print(
        "INPUT UNRESOLVED CANDIDATES:",
        len(
            unresolved
        )
    )

    print(
        "BASE PIXELS:",
        int(
            base.sum()
        )
    )

    print(
        "MERGE INTO BASE:",
        "NO"
    )

    # ========================================================
    # SAM2
    # ========================================================

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print(
        "Loading SAM2..."
    )

    model = build_sam2(
        SAM2_CONFIG,
        str(
            checkpoint
        ),
        device=device
    )

    predictor = (
        SAM2ImagePredictor(
            model
        )
    )

    predictor.set_image(
        rgb
    )

    print(
        "✅ SAM2 READY"
    )

    # ========================================================
    # RUN PER UNRESOLVED INSTANCE
    # ========================================================

    results = []

    proposal_union = np.zeros(
        (
            H,
            W
        ),
        dtype=bool
    )

    accepted_union = np.zeros(
        (
            H,
            W
        ),
        dtype=bool
    )

    for n, row in enumerate(
        unresolved,
        start=1
    ):

        iid = int(
            row[
                "inventory_id"
            ]
        )

        name = str(
            row[
                "inventory_name"
            ]
        )

        box = clip_box(
            row[
                "bbox"
            ],
            W,
            H
        )

        x1, y1, x2, y2 = box

        print()
        print(
            "-" * 90
        )

        print(
            f"[{n}/{len(unresolved)}]",
            f"ID={iid:02d}",
            name
        )

        print(
            "BOX:",
            box
        )

        input_box = np.asarray(
            [
                x1,
                y1,
                x2,
                y2
            ],
            dtype=np.float32
        )

        with torch.inference_mode():

            masks, scores, logits = (
                predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box,
                    multimask_output=True
                )
            )

        proposals = []

        for k in range(
            len(
                masks
            )
        ):

            m = (
                masks[k]
                .astype(bool)
            )

            # --------------------------------------------
            # LOCAL GATE
            # --------------------------------------------

            bw = (
                x2 - x1
            )

            bh = (
                y2 - y1
            )

            margin_x = max(
                4,
                int(
                    round(
                        bw
                        *
                        LOCAL_MARGIN
                    )
                )
            )

            margin_y = max(
                4,
                int(
                    round(
                        bh
                        *
                        LOCAL_MARGIN
                    )
                )
            )

            gx1 = max(
                0,
                x1 - margin_x
            )

            gy1 = max(
                0,
                y1 - margin_y
            )

            gx2 = min(
                W,
                x2 + margin_x
            )

            gy2 = min(
                H,
                y2 + margin_y
            )

            gate = np.zeros(
                (
                    H,
                    W
                ),
                dtype=bool
            )

            gate[
                gy1:gy2,
                gx1:gx2
            ] = True

            m = (
                m
                &
                gate
            )

            # --------------------------------------------
            # DOMINANT OBJECT COMPONENT
            # --------------------------------------------

            cx = int(
                round(
                    (
                        x1 + x2
                    )
                    /
                    2
                )
            )

            cy = int(
                round(
                    (
                        y1 + y2
                    )
                    /
                    2
                )
            )

            center_component = (
                component_containing_point(
                    m,
                    cx,
                    cy
                )
            )

            if center_component.sum() > 0:

                cleaned = (
                    center_component
                )

            else:

                cleaned = (
                    largest_component(
                        m
                    )
                )

            score, metrics = (
                score_mask(
                    cleaned,
                    box,
                    float(
                        scores[k]
                    ),
                    base
                )
            )

            proposals.append({

                "index":
                    int(k),

                "mask":
                    cleaned,

                "score":
                    float(
                        score
                    ),

                "metrics":
                    metrics
            })

            print(
                f"  MASK {k}:",
                f"SAM={float(scores[k]):.3f}",
                f"SCORE={score:.3f}",
                f"area={metrics.get('area', 0)}",
                f"ratio={metrics.get('area_ratio', 0)}",
                f"contain={metrics.get('containment', 0)}",
                f"missing={metrics.get('missing_pixels', 0)}"
            )

        if not proposals:

            results.append({

                "inventory_id":
                    iid,

                "inventory_name":
                    name,

                "decision":
                    "REJECT",

                "reasons": [
                    "no_sam2_proposals"
                ]
            })

            continue

        proposals.sort(
            key=lambda x:
                x[
                    "score"
                ],
            reverse=True
        )

        best = (
            proposals[
                0
            ]
        )

        best_mask = (
            best[
                "mask"
            ]
        )

        metrics = (
            best[
                "metrics"
            ]
        )

        recovery = (
            best_mask
            &
            (~base)
        )

        proposal_union |= (
            recovery
        )

        recovery_pixels = int(
            recovery.sum()
        )

        area_ratio = float(
            metrics.get(
                "area_ratio",
                999
            )
        )

        containment = float(
            metrics.get(
                "containment",
                0
            )
        )

        bbox_fill = float(
            metrics.get(
                "bbox_fill",
                1
            )
        )

        missing_ratio = float(
            metrics.get(
                "missing_ratio",
                0
            )
        )

        accept = True

        reasons = []

        if (
            recovery_pixels
            <
            MIN_RECOVERY_PIXELS
        ):

            accept = False

            reasons.append(
                "almost_no_new_pixels"
            )

        if (
            containment
            <
            MIN_CONTAINMENT
        ):

            accept = False

            reasons.append(
                "poor_box_containment"
            )

        if (
            area_ratio
            >
            MAX_AREA_RATIO
        ):

            accept = False

            reasons.append(
                "mask_too_large_for_box"
            )

        if (
            area_ratio > 0.75
            and
            bbox_fill > 0.82
            and
            missing_ratio > 0.55
        ):

            accept = False

            reasons.append(
                "dense_structural_region"
            )

        if (
            best[
                "score"
            ]
            <
            MIN_PROPOSAL_SCORE
        ):

            accept = False

            reasons.append(
                "low_stage09_score"
            )


        # ----------------------------------------------------
        # CATASTROPHIC GLOBAL-MASK SAFETY
        #
        # A bad source box can be so large that ordinary
        # containment metrics become meaningless.
        #
        # Reject proposals that recover an implausibly large
        # portion of the full image or behave like a broad
        # structural/background slab.
        # ----------------------------------------------------

        image_recovery_ratio = (
            recovery_pixels
            /
            float(
                W * H
            )
        )


        candidate_mask_area = int(
            best_mask.sum()
        )


        image_mask_ratio = (
            candidate_mask_area
            /
            float(
                W * H
            )
        )


        if (
            image_recovery_ratio
            >
            0.08
        ):

            accept = False

            reasons.append(
                "catastrophic_global_recovery"
            )


        if (
            image_mask_ratio
            >
            0.18
            and
            missing_ratio
            >
            0.45
        ):

            accept = False

            reasons.append(
                "broad_structural_mask"
            )

        # ----------------------------------------------------
        # EXTRA TEST07 SAFETY
        #
        # Stage09 is proposal-only. Even ACCEPT does NOT merge
        # into Stage08 automatically.
        # ----------------------------------------------------

        decision = (
            "PROPOSAL_ACCEPT"
            if accept
            else
            "PROPOSAL_REJECT"
        )

        if accept:

            accepted_union |= (
                recovery
            )

        print(
            "BEST:",
            best[
                "index"
            ],
            "|",
            decision,
            "| recovery:",
            recovery_pixels
        )

        if reasons:

            print(
                "REASONS:",
                ", ".join(
                    reasons
                )
            )

        prefix = (
            f"{iid:02d}_"
            f"{safe_name(name)}"
        )

        Image.fromarray(
            best_mask.astype(
                np.uint8
            )
            *
            255
        ).save(
            object_dir
            /
            f"{prefix}_best_mask.png"
        )

        Image.fromarray(
            recovery.astype(
                np.uint8
            )
            *
            255
        ).save(
            object_dir
            /
            f"{prefix}_recovery.png"
        )

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
        ] = rgb

        rgba[
            :,
            :,
            3
        ] = (
            best_mask.astype(
                np.uint8
            )
            *
            255
        )

        Image.fromarray(
            rgba,
            mode="RGBA"
        ).save(
            object_dir
            /
            f"{prefix}_rgba.png"
        )

        results.append({

            "inventory_id":
                iid,

            "inventory_name":
                name,

            "source_bbox":
                list(
                    box
                ),

            "source_stage05_rescue":
                bool(
                    row.get(
                        "stage05_rescue",
                        False
                    )
                ),

            "best_multimask_index":
                int(
                    best[
                        "index"
                    ]
                ),

            "decision":
                decision,

            "reasons":
                reasons,

            "recovery_pixels":
                recovery_pixels,

            "metrics":
                metrics,

            "all_multimask_metrics": [

                {
                    "index":
                        int(
                            p[
                                "index"
                            ]
                        ),

                    "score":
                        round(
                            float(
                                p[
                                    "score"
                                ]
                            ),
                            4
                        ),

                    "metrics":
                        p[
                            "metrics"
                        ]

                }

                for p in proposals
            ]
        })

    # ========================================================
    # SAVE DIAGNOSTICS ONLY
    # ========================================================

    proposal_path = (
        output_dir
        /
        "00_all_unresolved_recovery_proposals.png"
    )

    accepted_path = (
        output_dir
        /
        "01_accepted_unresolved_recovery_proposals.png"
    )

    Image.fromarray(
        proposal_union.astype(
            np.uint8
        )
        *
        255
    ).save(
        proposal_path
    )

    Image.fromarray(
        accepted_union.astype(
            np.uint8
        )
        *
        255
    ).save(
        accepted_path
    )

    result_path = (
        output_dir
        /
        "02_unresolved_multimask_results.json"
    )

    result_path.write_text(
        json.dumps(
            results,
            indent=2
        )
    )

    accepted_results = [
        r
        for r in results
        if r.get(
            "decision"
        )
        ==
        "PROPOSAL_ACCEPT"
    ]

    rejected_results = [
        r
        for r in results
        if r.get(
            "decision"
        )
        !=
        "PROPOSAL_ACCEPT"
    ]

    report = {

        "experiment":
            "TEST07_STAGE09",

        "method":
            "unresolved-instance SAM2 multimask proposal audit",

        "input_unresolved_candidates":
            len(
                unresolved
            ),

        "proposal_accept_count":
            len(
                accepted_results
            ),

        "proposal_reject_count":
            len(
                rejected_results
            ),

        "proposal_recovery_pixels":
            int(
                proposal_union.sum()
            ),

        "accepted_proposal_pixels":
            int(
                accepted_union.sum()
            ),

        "base_mask_unchanged":
            True,

        "objects":
            results,

        "rules": [
            "only Stage06 unresolved instances are processed",
            "all SAM2 multimasks are evaluated",
            "mask is constrained to local candidate neighborhood",
            "dominant connected component is retained",
            "Stage08 foreground is never modified",
            "proposal acceptance does not mean automatic merge",
            "exact RGB comes only from locked resized master"
        ]
    }

    report_path = (
        output_dir
        /
        "stage09_report.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2
        )
    )

    # ========================================================
    # RESULT
    # ========================================================

    print()
    print(
        "=" * 90
    )

    print(
        "TEST07 STAGE09 RESULT"
    )

    print(
        "=" * 90
    )

    print(
        "UNRESOLVED INPUT:",
        len(
            unresolved
        )
    )

    print(
        "PROPOSAL ACCEPT:",
        len(
            accepted_results
        )
    )

    print(
        "PROPOSAL REJECT:",
        len(
            rejected_results
        )
    )

    print(
        "ACCEPTED PROPOSAL PIXELS:",
        int(
            accepted_union.sum()
        )
    )

    print()
    print(
        "PER INSTANCE:"
    )

    for row in results:

        print(
            f'{int(row["inventory_id"]):02d}.',
            row[
                "inventory_name"
            ],
            "→",
            row[
                "decision"
            ],
            "| recovery:",
            row.get(
                "recovery_pixels",
                0
            ),
            "|",
            ",".join(
                row.get(
                    "reasons",
                    []
                )
            )
        )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "STAGE08 FINAL MASK WAS NOT MODIFIED."
    )

    print(
        "THESE ARE RECOVERY PROPOSALS ONLY."
    )

    del predictor
    del model

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--master",
        required=True
    )

    parser.add_argument(
        "--base-mask",
        required=True
    )

    parser.add_argument(
        "--unresolved-json",
        required=True
    )

    parser.add_argument(
        "--output-dir",
        required=True
    )

    parser.add_argument(
        "--sam2-checkpoint",
        default=None
    )

    args = parser.parse_args()

    run(
        master_path=
            args.master,

        base_mask_path=
            args.base_mask,

        unresolved_json=
            args.unresolved_json,

        output_dir=
            args.output_dir,

        checkpoint_path=
            args.sam2_checkpoint
    )
