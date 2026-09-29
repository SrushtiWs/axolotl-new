
from pathlib import Path
import json
import gc
from collections import Counter

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

MASTER_PATH = (
    PROD
    / "stage05_clean_room_with_props"
    / "12_final_canonical_prop_detection_master.png"
)

D1D5_PATH = (
    STAGE06
    / "06d1d5_sam2_structure_overlap_audit"
    / "00_stage06d1d5_result.json"
)

OUT = (
    STAGE06
    / "06d1e_mask_physical_instance_dedup"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

CLUSTER_DIR = (
    OUT
    / "clusters"
)

CLUSTER_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# DUPLICATE THRESHOLDS
#
# Deliberately strict.
#
# We want SAME PHYSICAL INSTANCE,
# not simply touching / nested objects.
# ============================================================

MIN_DUPLICATE_IOU = 0.72

MIN_BIDIRECTIONAL_CONTAINMENT = 0.82

MIN_AREA_RATIO = 0.55

MAX_AREA_RATIO = 1.82


# ============================================================
# VALIDATE
# ============================================================

for path in [
    MASTER_PATH,
    D1D5_PATH,
]:
    if not path.exists():
        raise FileNotFoundError(path)


master_pil = Image.open(
    MASTER_PATH
).convert("RGB")

master_np = np.asarray(
    master_pil
)

H, W = master_np.shape[:2]


d1d5 = json.loads(
    D1D5_PATH.read_text(
        encoding="utf-8"
    )
)

rows = d1d5[
    "results"
]


# ============================================================
# HELPERS
# ============================================================

def show(
    image_or_path,
    title,
    figsize=(8, 9)
):

    if isinstance(
        image_or_path,
        (str, Path)
    ):
        image = Image.open(image_or_path)
    else:
        image = image_or_path

    plt.figure(
        figsize=figsize
    )

    plt.imshow(image)

    plt.title(title)

    plt.axis("off")

    plt.show()


def load_mask(path):

    return (
        np.asarray(
            Image.open(path).convert("L")
        )
        >
        0
    )


def intersection_area(a, b):

    return int(
        (
            a & b
        ).sum()
    )


def mask_area(mask):

    return int(
        mask.sum()
    )


def iou(a, b):

    inter = intersection_area(
        a,
        b
    )

    union = int(
        (
            a | b
        ).sum()
    )

    if union <= 0:
        return 0.0

    return float(
        inter / union
    )


def directional_containment(
    a,
    b
):
    """
    Fraction of A contained in B.
    """

    area_a = mask_area(a)

    if area_a <= 0:
        return 0.0

    return float(
        intersection_area(a, b)
        /
        area_a
    )


def area_ratio(a, b):

    aa = mask_area(a)
    ab = mask_area(b)

    if aa <= 0 or ab <= 0:
        return 0.0

    return float(
        aa / ab
    )


def duplicate_metrics(a, b):

    pair_iou = iou(
        a,
        b
    )

    a_in_b = directional_containment(
        a,
        b
    )

    b_in_a = directional_containment(
        b,
        a
    )

    ratio = area_ratio(
        a,
        b
    )


    near_same_size = bool(
        MIN_AREA_RATIO
        <=
        ratio
        <=
        MAX_AREA_RATIO
    )


    duplicate = bool(

        pair_iou
        >=
        MIN_DUPLICATE_IOU

        or

        (
            a_in_b
            >=
            MIN_BIDIRECTIONAL_CONTAINMENT

            and

            b_in_a
            >=
            MIN_BIDIRECTIONAL_CONTAINMENT

            and

            near_same_size
        )
    )


    return {

        "duplicate":
            duplicate,

        "iou":
            pair_iou,

        "a_in_b":
            a_in_b,

        "b_in_a":
            b_in_a,

        "area_ratio":
            ratio,
    }


def draw_outline(
    image,
    mask,
    color,
    width=2
):

    result = image.copy()

    contours, _ = cv2.findContours(
        (
            mask.astype(np.uint8)
            *
            255
        ),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    draw = ImageDraw.Draw(
        result
    )


    for contour in contours:

        points = [

            (
                int(p[0][0]),
                int(p[0][1])
            )

            for p in contour
        ]

        if len(points) >= 2:

            draw.line(
                points + [points[0]],
                fill=color,
                width=width
            )


    return result


# ============================================================
# PREPARE VALID-MASK EVIDENCE
# ============================================================

eligible = []

geometry_failed = []


for row in rows:

    mask_path = row.get(
        "mask_path"
    )


    if (
        not mask_path
        or
        not Path(mask_path).exists()
    ):

        failed = dict(row)

        failed[
            "dedup_status"
        ] = "NO_MASK"

        geometry_failed.append(
            failed
        )

        continue


    # --------------------------------------------------------
    # Geometry failures are NOT used as cluster anchors.
    #
    # Preserve them separately so they cannot create bad
    # automatic physical instances.
    # --------------------------------------------------------

    if not row.get(
        "geometry_pass",
        False
    ):

        failed = dict(row)

        failed[
            "dedup_status"
        ] = "GEOMETRY_FAIL_HELD_OUT"

        geometry_failed.append(
            failed
        )

        continue


    mask = load_mask(
        mask_path
    )


    item = dict(row)

    item[
        "_mask"
    ] = mask

    item[
        "_area"
    ] = mask_area(
        mask
    )


    eligible.append(
        item
    )


print("=" * 110)
print("PRODUCTION STAGE 06D1E")
print("MASK-BASED PHYSICAL-INSTANCE DEDUPLICATION")
print("=" * 110)

print()
print(
    "D1D5 INPUT:",
    len(rows)
)

print(
    "ELIGIBLE GEOMETRY-PASS MASKS:",
    len(eligible)
)

print(
    "HELD-OUT GEOMETRY FAILURES:",
    len(geometry_failed)
)


# ============================================================
# PAIRWISE NEAR-IDENTICAL MASK GRAPH
# ============================================================

edges = []


for i in range(
    len(eligible)
):

    for j in range(
        i + 1,
        len(eligible)
    ):

        a = eligible[i]
        b = eligible[j]


        metrics = duplicate_metrics(
            a[
                "_mask"
            ],
            b[
                "_mask"
            ]
        )


        if metrics[
            "duplicate"
        ]:

            edges.append({

                "i":
                    i,

                "j":
                    j,

                "a_id":
                    a[
                        "evidence_id"
                    ],

                "b_id":
                    b[
                        "evidence_id"
                    ],

                **metrics,
            })


print()
print(
    "NEAR-IDENTICAL MASK EDGES:",
    len(edges)
)


# ============================================================
# UNION FIND
# ============================================================

parent = list(
    range(
        len(eligible)
    )
)


def find(x):

    while parent[x] != x:

        parent[x] = parent[
            parent[x]
        ]

        x = parent[x]

    return x


def union(a, b):

    ra = find(a)
    rb = find(b)

    if ra != rb:

        parent[rb] = ra


for edge in edges:

    union(
        edge[
            "i"
        ],
        edge[
            "j"
        ]
    )


# ============================================================
# CREATE CLUSTERS
# ============================================================

cluster_index_map = {}


for idx in range(
    len(eligible)
):

    root = find(idx)

    cluster_index_map.setdefault(
        root,
        []
    ).append(
        idx
    )


raw_clusters = list(
    cluster_index_map.values()
)


# ============================================================
# BUILD CLUSTER RECORDS
# ============================================================

clusters = []


for cluster_number, indexes in enumerate(
    raw_clusters,
    start=1
):

    members = [

        eligible[i]

        for i in indexes
    ]


    # --------------------------------------------------------
    # Canonical mask:
    #
    # pixel majority when >1 member,
    # otherwise member mask.
    #
    # This suppresses occasional fringe differences.
    # --------------------------------------------------------

    stack = np.stack(
        [
            member[
                "_mask"
            ].astype(
                np.uint8
            )

            for member in members
        ],
        axis=0
    )


    threshold = (
        len(members)
        //
        2
        +
        1
    )


    canonical_mask = (
        stack.sum(
            axis=0
        )
        >=
        threshold
    )


    # --------------------------------------------------------
    # For 2-member cluster, majority threshold=2 means
    # intersection. If that becomes implausibly small,
    # use higher-SAM member instead.
    # --------------------------------------------------------

    if (
        len(members) == 2

        and

        canonical_mask.sum()
        <
        0.60
        *
        min(
            members[0][
                "_area"
            ],
            members[1][
                "_area"
            ]
        )
    ):

        best_member = max(
            members,
            key=lambda x:
                (
                    float(
                        x.get(
                            "sam_score",
                            0.0
                        )
                    ),
                    float(
                        x.get(
                            "geometry_quality",
                            0.0
                        )
                    ),
                )
        )

        canonical_mask = (
            best_member[
                "_mask"
            ].copy()
        )


    # --------------------------------------------------------
    # Names are evidence only.
    # Do NOT choose semantic winner automatically.
    # --------------------------------------------------------

    names = [

        str(
            member.get(
                "name",
                ""
            )
        )

        for member in members
    ]


    name_counts = Counter(
        name.lower()
        for name in names
        if name
    )


    unique_names = sorted(
        set(
            names
        )
    )


    semantic_conflict = bool(
        len(
            {
                name.lower()
                for name in names
                if name
            }
        )
        >
        1
    )


    cluster_id = (
        f"I{cluster_number:02d}"
    )


    # --------------------------------------------------------
    # Save mask
    # --------------------------------------------------------

    mask_path = (
        CLUSTER_DIR
        /
        f"{cluster_id}_canonical_mask.png"
    )


    Image.fromarray(
        canonical_mask.astype(
            np.uint8
        )
        *
        255
    ).save(
        mask_path
    )


    # --------------------------------------------------------
    # Preview
    # --------------------------------------------------------

    preview = master_pil.copy()


    preview = draw_outline(
        preview,
        canonical_mask,
        "lime",
        3
    )


    draw = ImageDraw.Draw(
        preview
    )


    ys, xs = np.where(
        canonical_mask
    )


    if len(xs) > 0:

        x1 = int(xs.min())
        y1 = int(ys.min())
        x2 = int(xs.max())
        y2 = int(ys.max())


        draw.rectangle(
            [
                x1,
                y1,
                x2,
                y2
            ],
            outline="red",
            width=2
        )


    preview_path = (
        CLUSTER_DIR
        /
        f"{cluster_id}_preview.png"
    )


    preview.save(
        preview_path
    )


    # --------------------------------------------------------
    # Within-cluster pair metrics
    # --------------------------------------------------------

    pair_metrics = []


    for a_index in range(
        len(members)
    ):

        for b_index in range(
            a_index + 1,
            len(members)
        ):

            metrics = duplicate_metrics(
                members[
                    a_index
                ][
                    "_mask"
                ],
                members[
                    b_index
                ][
                    "_mask"
                ]
            )


            pair_metrics.append({

                "a":
                    members[
                        a_index
                    ][
                        "evidence_id"
                    ],

                "b":
                    members[
                        b_index
                    ][
                        "evidence_id"
                    ],

                **metrics,
            })


    cluster = {

        "instance_id":
            cluster_id,

        "member_count":
            len(
                members
            ),

        "evidence_ids":
            [
                member[
                    "evidence_id"
                ]
                for member
                in members
            ],

        "observed_names":
            names,

        "unique_names":
            unique_names,

        "name_frequency":
            dict(
                name_counts
            ),

        "semantic_conflict":
            semantic_conflict,

        "canonical_mask_path":
            str(
                mask_path
            ),

        "preview_path":
            str(
                preview_path
            ),

        "canonical_area":
            int(
                canonical_mask.sum()
            ),

        "pair_metrics":
            pair_metrics,

        "member_metadata":
            [

                {
                    k:
                        v

                    for k, v
                    in member.items()

                    if not k.startswith(
                        "_"
                    )
                }

                for member
                in members
            ],
    }


    clusters.append(
        cluster
    )


# ============================================================
# CROSS-INSTANCE OVERLAP DIAGNOSTIC
#
# IMPORTANT:
# Not merging here.
#
# This is for the NEXT connected/grouping stage.
# ============================================================

cross_instance_pairs = []


cluster_masks = {

    cluster[
        "instance_id"
    ]:
        load_mask(
            cluster[
                "canonical_mask_path"
            ]
        )

    for cluster
    in clusters
}


for i in range(
    len(clusters)
):

    for j in range(
        i + 1,
        len(clusters)
    ):

        a = clusters[i]
        b = clusters[j]


        ma = cluster_masks[
            a[
                "instance_id"
            ]
        ]

        mb = cluster_masks[
            b[
                "instance_id"
            ]
        ]


        inter = intersection_area(
            ma,
            mb
        )


        if inter <= 0:
            continue


        pair_iou = iou(
            ma,
            mb
        )


        a_in_b = directional_containment(
            ma,
            mb
        )


        b_in_a = directional_containment(
            mb,
            ma
        )


        cross_instance_pairs.append({

            "a_instance":
                a[
                    "instance_id"
                ],

            "b_instance":
                b[
                    "instance_id"
                ],

            "intersection_pixels":
                inter,

            "iou":
                pair_iou,

            "a_in_b":
                a_in_b,

            "b_in_a":
                b_in_a,
        })


# ============================================================
# SAVE
# ============================================================

RESULT = {

    "stage":
        "06D1E",

    "input_stage":
        "06D1D5",

    "duplicate_thresholds": {

        "min_iou":
            MIN_DUPLICATE_IOU,

        "min_bidirectional_containment":
            MIN_BIDIRECTIONAL_CONTAINMENT,

        "min_area_ratio":
            MIN_AREA_RATIO,

        "max_area_ratio":
            MAX_AREA_RATIO,
    },

    "input_evidence_count":
        len(
            rows
        ),

    "eligible_geometry_pass_count":
        len(
            eligible
        ),

    "geometry_failed_held_out":
        [

            {
                k:
                    v

                for k, v
                in row.items()

                if k != "_mask"
            }

            for row
            in geometry_failed
        ],

    "near_identical_edges":
        edges,

    "physical_instance_clusters":
        clusters,

    "cross_instance_overlap_diagnostic":
        cross_instance_pairs,

    "status":
        "REQUIRES_INSTANCE_CLUSTER_AUDIT",

    "rules": [

        "near-identical masks may deduplicate",

        "semantic labels do not drive mask clustering",

        "touching but materially different masks remain separate",

        "geometry-failed evidence cannot create automatic clusters",

        "semantic conflicts remain unresolved",

        "no connected-prop grouping performed",

        "no final mask union created",
    ],
}


RESULT_PATH = (
    OUT
    / "00_stage06d1e_result.json"
)


RESULT_PATH.write_text(
    json.dumps(
        RESULT,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# ============================================================
# PRINT
# ============================================================

print()
print("=" * 110)
print("PRODUCTION STAGE 06D1E RESULT")
print("=" * 110)

print()
print(
    "INPUT EVIDENCE:",
    len(
        rows
    )
)

print(
    "ELIGIBLE MASKS:",
    len(
        eligible
    )
)

print(
    "GEOMETRY FAIL HELD OUT:",
    len(
        geometry_failed
    )
)

print(
    "NEAR-IDENTICAL EDGES:",
    len(
        edges
    )
)

print(
    "PHYSICAL INSTANCE CLUSTERS:",
    len(
        clusters
    )
)


print()
print("=" * 110)
print("INSTANCE CLUSTERS")
print("=" * 110)


for cluster in clusters:

    print()

    print(
        "{} | members={} | conflict={}".format(

            cluster[
                "instance_id"
            ],

            cluster[
                "member_count"
            ],

            cluster[
                "semantic_conflict"
            ],
        )
    )

    print(
        "  EVIDENCE:",
        cluster[
            "evidence_ids"
        ]
    )

    print(
        "  NAMES:",
        cluster[
            "unique_names"
        ]
    )

    print(
        "  AREA:",
        cluster[
            "canonical_area"
        ]
    )


print()
print("=" * 110)
print("GEOMETRY-FAILED EVIDENCE HELD OUT")
print("=" * 110)


for row in geometry_failed:

    print(
        "{} | {} | {}".format(

            row.get(
                "evidence_id"
            ),

            row.get(
                "name"
            ),

            row.get(
                "state"
            ),
        )
    )


print()
print(
    "CROSS-INSTANCE OVERLAP PAIRS:",
    len(
        cross_instance_pairs
    )
)

print(
    "RESULT JSON:",
    RESULT_PATH
)

print()
print(
    "NO CONNECTED-PROP GROUPING WAS PERFORMED."
)

print(
    "NO FINAL PROP MASK UNION WAS CREATED."
)


# ============================================================
# INLINE CLUSTER AUDIT
#
# Print:
# 1. every multi-evidence cluster
# 2. every semantic-conflict cluster
# 3. first few singletons
# ============================================================

print()
print("=" * 110)
print("INLINE 06D1E INSTANCE CLUSTER AUDIT")
print("=" * 110)


review_clusters = []

seen = set()


for cluster in clusters:

    if (
        cluster[
            "member_count"
        ]
        >
        1

        or

        cluster[
            "semantic_conflict"
        ]
    ):

        review_clusters.append(
            cluster
        )

        seen.add(
            cluster[
                "instance_id"
            ]
        )


# Add first 6 singletons as sanity check
for cluster in clusters:

    if (
        cluster[
            "instance_id"
        ]
        in
        seen
    ):
        continue

    review_clusters.append(
        cluster
    )

    if len(
        [
            c
            for c in review_clusters
            if c[
                "member_count"
            ]
            ==
            1
        ]
    ) >= 6:
        break


for cluster in review_clusters:

    show(

        cluster[
            "preview_path"
        ],

        (
            f'{cluster["instance_id"]} | '
            f'members={cluster["member_count"]} | '
            f'names={cluster["unique_names"]}'
        ),
    )


# ============================================================
# PRINT HIGH-OVERLAP DISTINCT CLUSTER PAIRS
#
# Useful for the later touching/connected-prop stage.
# ============================================================

interesting_cross = [

    row

    for row in cross_instance_pairs

    if (
        row[
            "iou"
        ]
        >=
        0.10

        or

        row[
            "a_in_b"
        ]
        >=
        0.50

        or

        row[
            "b_in_a"
        ]
        >=
        0.50
    )
]


interesting_cross.sort(

    key=lambda row:
        max(
            row[
                "iou"
            ],
            row[
                "a_in_b"
            ],
            row[
                "b_in_a"
            ],
        ),

    reverse=True
)


print()
print("=" * 110)
print("HIGH-OVERLAP DISTINCT INSTANCE PAIRS")
print("=" * 110)


for row in interesting_cross:

    print(

        "{} <-> {} | IoU={:.3f} | "
        "AinB={:.3f} | BinA={:.3f}".format(

            row[
                "a_instance"
            ],

            row[
                "b_instance"
            ],

            row[
                "iou"
            ],

            row[
                "a_in_b"
            ],

            row[
                "b_in_a"
            ],
        )
    )
