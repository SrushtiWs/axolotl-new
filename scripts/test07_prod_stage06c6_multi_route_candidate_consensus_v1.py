
from pathlib import Path
import json
import math


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


INVENTORY_JSON = (
    STAGE06
    / "06c2_lost_instance_existence_audit"
    / "00a_physical_inventory_bridge.json"
)


EXISTENCE_JSON = (
    STAGE06
    / "06c2_lost_instance_existence_audit"
    / "00_existence_audit.json"
)


B06_PATH = (
    STAGE06
    / "06b_source_crop_grounding"
    / "05_parent_mapped_dino_candidates_flat.json"
)


C3_PATH = (
    STAGE06
    / "06c3_local_tile_rescue"
    / "00_local_tile_rescue_candidates.json"
)


C5_PATH = (
    STAGE06
    / "06c5_highres_local_pyramid_rescue"
    / "00_highres_pyramid_candidates.json"
)


OUT = (
    STAGE06
    / "06c6_multi_route_candidate_consensus"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIG
# ============================================================

# Cluster compatibility:
#
# IoU >= threshold
# OR
# center distance relative to smaller box diagonal <= threshold

IOU_CLUSTER_THRESHOLD = 0.35

CENTER_DISTANCE_RATIO_THRESHOLD = 0.60


# Strong route diversity bonus.
ROUTE_BONUS = {
    1: 0.00,
    2: 0.18,
    3: 0.32,
}


# Small support-count bonus, capped.
MAX_SUPPORT_BONUS = 0.18


# ============================================================
# HELPERS
# ============================================================

def clamp_box(box, W, H):

    x1, y1, x2, y2 = [
        float(v)
        for v in box
    ]

    x1 = max(
        0.0,
        min(
            float(W),
            x1
        )
    )

    y1 = max(
        0.0,
        min(
            float(H),
            y1
        )
    )

    x2 = max(
        0.0,
        min(
            float(W),
            x2
        )
    )

    y2 = max(
        0.0,
        min(
            float(H),
            y2
        )
    )


    if x2 < x1:
        x1, x2 = x2, x1

    if y2 < y1:
        y1, y2 = y2, y1


    return [
        x1,
        y1,
        x2,
        y2
    ]


def box_area(box):

    x1, y1, x2, y2 = box

    return max(
        0.0,
        x2 - x1
    ) * max(
        0.0,
        y2 - y1
    )


def box_center(box):

    x1, y1, x2, y2 = box

    return (
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0
    )


def box_diag(box):

    x1, y1, x2, y2 = box

    return math.hypot(
        max(
            0.0,
            x2 - x1
        ),
        max(
            0.0,
            y2 - y1
        )
    )


def iou(a, b):

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b


    ix1 = max(
        ax1,
        bx1
    )

    iy1 = max(
        ay1,
        by1
    )

    ix2 = min(
        ax2,
        bx2
    )

    iy2 = min(
        ay2,
        by2
    )


    iw = max(
        0.0,
        ix2 - ix1
    )

    ih = max(
        0.0,
        iy2 - iy1
    )


    inter = (
        iw
        *
        ih
    )


    union = (
        box_area(a)
        +
        box_area(b)
        -
        inter
    )


    if union <= 0:
        return 0.0


    return float(
        inter
        /
        union
    )


def center_distance_ratio(a, b):

    acx, acy = box_center(a)
    bcx, bcy = box_center(b)


    d = math.hypot(
        acx - bcx,
        acy - bcy
    )


    scale = max(
        4.0,
        min(
            box_diag(a),
            box_diag(b)
        )
    )


    return (
        d
        /
        scale
    )


def compatible(a, b):

    ov = iou(
        a["bbox"],
        b["bbox"]
    )


    if ov >= IOU_CLUSTER_THRESHOLD:

        return True


    cdr = center_distance_ratio(
        a["bbox"],
        b["bbox"]
    )


    return (
        cdr
        <=
        CENTER_DISTANCE_RATIO_THRESHOLD
    )


def weighted_median(values, weights):

    pairs = sorted(
        zip(
            values,
            weights
        ),
        key=lambda x:
            x[0]
    )


    total = sum(
        weights
    )


    if total <= 0:

        return float(
            values[
                len(values)
                //
                2
            ]
        )


    running = 0.0

    target = (
        total
        /
        2.0
    )


    for value, weight in pairs:

        running += weight

        if running >= target:

            return float(
                value
            )


    return float(
        pairs[-1][0]
    )


def canonical_box(rows):

    weights = [
        max(
            0.05,
            float(
                r.get(
                    "score",
                    0.0
                )
            )
        )
        for r in rows
    ]


    coords = list(
        zip(
            *[
                r["bbox"]
                for r in rows
            ]
        )
    )


    return [

        weighted_median(
            list(coords[k]),
            weights
        )

        for k in range(4)
    ]


# ============================================================
# VALIDATE
# ============================================================

print("=" * 110)
print("PRODUCTION STAGE 06C6")
print("MULTI-ROUTE SPATIAL CANDIDATE CONSENSUS")
print("=" * 110)


required = [
    MASTER,
    INVENTORY_JSON,
    EXISTENCE_JSON,
    B06_PATH,
    C3_PATH,
    C5_PATH,
]


for path in required:

    print(
        "✅" if path.exists() else "❌",
        path
    )

    if not path.exists():

        raise FileNotFoundError(
            path
        )


# ============================================================
# MASTER SIZE
# ============================================================

from PIL import Image

W, H = Image.open(
    MASTER
).size


print()
print(
    "MASTER:",
    W,
    "x",
    H
)


# ============================================================
# PRESENT IDS
# ============================================================

existence = json.loads(
    EXISTENCE_JSON.read_text(
        encoding="utf-8"
    )
)


present_ids = sorted(
    int(
        row[
            "inventory_id"
        ]
    )
    for row in existence
    if row.get(
        "decision"
    ) == "PRESENT"
)


print(
    "PRESENT IDS:",
    present_ids
)


# ============================================================
# INVENTORY
# ============================================================

inventory = json.loads(
    INVENTORY_JSON.read_text(
        encoding="utf-8"
    )
)["objects"]


inventory_by_id = {

    int(
        row[
            "id"
        ]
    ):
        row

    for row in inventory
}


# ============================================================
# NORMALIZE 06B
# ============================================================

raw_06b = json.loads(
    B06_PATH.read_text(
        encoding="utf-8"
    )
)


norm_06b = []


for row in raw_06b:

    pid = int(
        row[
            "parent_inventory_id"
        ]
    )


    if pid not in present_ids:
        continue


    norm_06b.append({

        "physical_inventory_id":
            pid,

        "inventory_name":
            row.get(
                "parent_inventory_name",
                row.get(
                    "inventory_name",
                    ""
                )
            ),

        "grounding_phrase":
            row.get(
                "grounding_phrase",
                ""
            ),

        "bbox":
            clamp_box(
                row[
                    "bbox"
                ],
                W,
                H
            ),

        "score":
            float(
                row.get(
                    "score",
                    0.0
                )
            ),

        "route":
            "06B_SOURCE_CROP",

        "source_region":
            row.get(
                "source_region"
            ),

        "source_bbox":
            row.get(
                "source_bbox"
            ),

        "observation_id":
            row.get(
                "observation_id"
            ),

        "rank_for_observation":
            row.get(
                "rank_for_observation"
            ),

        "dino_label":
            row.get(
                "dino_label"
            ),

        "original_record":
            row,
    })


# ============================================================
# NORMALIZE 06C3
# ============================================================

raw_c3 = json.loads(
    C3_PATH.read_text(
        encoding="utf-8"
    )
)


norm_c3 = []


for row in raw_c3:

    pid = int(
        row[
            "inventory_id"
        ]
    )


    if pid not in present_ids:
        continue


    norm_c3.append({

        "physical_inventory_id":
            pid,

        "inventory_name":
            row.get(
                "inventory_name",
                ""
            ),

        "grounding_phrase":
            row.get(
                "grounding_phrase",
                ""
            ),

        "bbox":
            clamp_box(
                row[
                    "bbox"
                ],
                W,
                H
            ),

        "score":
            float(
                row.get(
                    "score",
                    0.0
                )
            ),

        "route":
            "06C3_LOCAL_TILE",

        "source_region":
            row.get(
                "source_region"
            ),

        "source_bbox":
            row.get(
                "source_bbox"
            ),

        "dino_label":
            row.get(
                "dino_label"
            ),

        "original_record":
            row,
    })


# ============================================================
# NORMALIZE 06C5
# ============================================================

raw_c5 = json.loads(
    C5_PATH.read_text(
        encoding="utf-8"
    )
)


norm_c5 = []


for row in raw_c5:

    pid = int(
        row[
            "inventory_id"
        ]
    )


    if pid not in present_ids:
        continue


    norm_c5.append({

        "physical_inventory_id":
            pid,

        "inventory_name":
            row.get(
                "inventory_name",
                ""
            ),

        "grounding_phrase":
            row.get(
                "grounding_phrase",
                ""
            ),

        "bbox":
            clamp_box(
                row[
                    "bbox"
                ],
                W,
                H
            ),

        "score":
            float(
                row.get(
                    "score",
                    0.0
                )
            ),

        "route":
            "06C5_HIGHRES",

        "source_region":
            row.get(
                "source_region"
            ),

        "pyramid_view":
            row.get(
                "pyramid_view"
            ),

        "dino_query":
            row.get(
                "dino_query"
            ),

        "dino_label":
            row.get(
                "dino_label"
            ),

        "original_record":
            row,
    })


all_rows = (
    norm_06b
    +
    norm_c3
    +
    norm_c5
)


print()
print(
    "NORMALIZED 06B:",
    len(
        norm_06b
    )
)

print(
    "NORMALIZED 06C3:",
    len(
        norm_c3
    )
)

print(
    "NORMALIZED 06C5:",
    len(
        norm_c5
    )
)

print(
    "TOTAL PRESENT-INSTANCE PROPOSALS:",
    len(
        all_rows
    )
)


# ============================================================
# CLUSTER PER PHYSICAL INSTANCE
#
# Greedy deterministic clustering:
# proposals sorted by detector score descending.
#
# A proposal joins the cluster with maximum compatibility.
# ============================================================

consensus_output = []

canonical_seeds = []


for iid in present_ids:

    obj = inventory_by_id[
        iid
    ]


    rows = [
        r
        for r in all_rows
        if int(
            r[
                "physical_inventory_id"
            ]
        ) == iid
    ]


    rows.sort(
        key=lambda r:
            (
                -float(
                    r[
                        "score"
                    ]
                ),
                r[
                    "route"
                ],
                str(
                    r.get(
                        "source_region"
                    )
                ),
            )
    )


    clusters = []


    for row in rows:

        best_cluster_index = None
        best_similarity = -1.0


        for ci, cluster in enumerate(
            clusters
        ):

            representative = (
                cluster[
                    "representative"
                ]
            )


            ov = iou(
                row[
                    "bbox"
                ],
                representative[
                    "bbox"
                ]
            )


            cdr = center_distance_ratio(
                row[
                    "bbox"
                ],
                representative[
                    "bbox"
                ]
            )


            similarity = max(
                ov,
                max(
                    0.0,
                    1.0 - cdr
                )
            )


            if (
                compatible(
                    row,
                    representative
                )
                and
                similarity
                >
                best_similarity
            ):

                best_cluster_index = ci
                best_similarity = similarity


        if best_cluster_index is None:

            clusters.append({

                "representative":
                    row,

                "members":
                    [
                        row
                    ]
            })

        else:

            clusters[
                best_cluster_index
            ][
                "members"
            ].append(
                row
            )


            # Recompute canonical representative box,
            # but keep highest scoring proposal metadata.
            members = clusters[
                best_cluster_index
            ][
                "members"
            ]


            highest = max(
                members,
                key=lambda r:
                    float(
                        r[
                            "score"
                        ]
                    )
            )


            updated = dict(
                highest
            )


            updated[
                "bbox"
            ] = canonical_box(
                members
            )


            clusters[
                best_cluster_index
            ][
                "representative"
            ] = updated


    # ========================================================
    # SCORE CLUSTERS
    # ========================================================

    cluster_rows = []


    for ci, cluster in enumerate(
        clusters,
        start=1
    ):

        members = cluster[
            "members"
        ]


        routes = sorted(
            {
                r[
                    "route"
                ]
                for r in members
            }
        )


        route_count = len(
            routes
        )


        max_score = max(
            float(
                r[
                    "score"
                ]
            )
            for r in members
        )


        mean_score = sum(
            float(
                r[
                    "score"
                ]
            )
            for r in members
        ) / max(
            1,
            len(
                members
            )
        )


        support_bonus = min(
            MAX_SUPPORT_BONUS,
            max(
                0,
                len(
                    members
                )
                -
                1
            )
            *
            0.025
        )


        route_bonus = ROUTE_BONUS.get(
            route_count,
            ROUTE_BONUS[
                3
            ]
        )


        consensus_score = (
            0.60
            *
            max_score
            +
            0.40
            *
            mean_score
            +
            route_bonus
            +
            support_bonus
        )


        box = canonical_box(
            members
        )


        source_regions = sorted(
            {
                str(
                    r.get(
                        "source_region"
                    )
                )
                for r in members
                if r.get(
                    "source_region"
                )
            }
        )


        c5_views = sorted(
            {
                str(
                    r.get(
                        "pyramid_view"
                    )
                )
                for r in members
                if r.get(
                    "pyramid_view"
                )
            }
        )


        c5_queries = sorted(
            {
                str(
                    r.get(
                        "dino_query"
                    )
                )
                for r in members
                if r.get(
                    "dino_query"
                )
            }
        )


        cluster_rows.append({

            "cluster_id":
                ci,

            "bbox":
                box,

            "proposal_count":
                len(
                    members
                ),

            "routes":
                routes,

            "route_count":
                route_count,

            "source_regions":
                source_regions,

            "highres_views":
                c5_views,

            "highres_queries":
                c5_queries,

            "max_dino_score":
                max_score,

            "mean_dino_score":
                mean_score,

            "route_bonus":
                route_bonus,

            "support_bonus":
                support_bonus,

            "consensus_score":
                consensus_score,

            "members":
                members,
        })


    cluster_rows.sort(
        key=lambda c:
            (
                -int(
                    c[
                        "route_count"
                    ]
                ),
                -float(
                    c[
                        "consensus_score"
                    ]
                ),
                -int(
                    c[
                        "proposal_count"
                    ]
                ),
                -float(
                    c[
                        "max_dino_score"
                    ]
                ),
            )
    )


    best_cluster = (
        cluster_rows[0]
        if cluster_rows
        else None
    )


    instance_record = {

        "inventory_id":
            iid,

        "inventory_name":
            obj.get(
                "name",
                ""
            ),

        "grounding_phrase":
            obj.get(
                "grounding_phrase",
                ""
            ),

        "total_proposals":
            len(
                rows
            ),

        "cluster_count":
            len(
                cluster_rows
            ),

        "clusters":
            cluster_rows,
    }


    consensus_output.append(
        instance_record
    )


    if best_cluster is not None:

        canonical_seeds.append({

            "inventory_id":
                iid,

            "inventory_name":
                obj.get(
                    "name",
                    ""
                ),

            "grounding_phrase":
                obj.get(
                    "grounding_phrase",
                    ""
                ),

            "bbox":
                best_cluster[
                    "bbox"
                ],

            "score":
                best_cluster[
                    "max_dino_score"
                ],

            "consensus_score":
                best_cluster[
                    "consensus_score"
                ],

            "consensus_routes":
                best_cluster[
                    "routes"
                ],

            "consensus_route_count":
                best_cluster[
                    "route_count"
                ],

            "consensus_proposal_count":
                best_cluster[
                    "proposal_count"
                ],

            "consensus_source_regions":
                best_cluster[
                    "source_regions"
                ],

            "consensus_highres_views":
                best_cluster[
                    "highres_views"
                ],

            "consensus_highres_queries":
                best_cluster[
                    "highres_queries"
                ],

            "candidate_origin":
                "06C6_MULTI_ROUTE_CONSENSUS",
        })


# ============================================================
# CROSS-INSTANCE POTENTIAL COLLISIONS
#
# Diagnostic only.
# Do NOT merge here.
# ============================================================

potential_collisions = []


for i in range(
    len(
        canonical_seeds
    )
):

    for j in range(
        i + 1,
        len(
            canonical_seeds
        )
    ):

        a = canonical_seeds[
            i
        ]

        b = canonical_seeds[
            j
        ]


        ov = iou(
            a[
                "bbox"
            ],
            b[
                "bbox"
            ]
        )


        containment_ab = (
            box_area(
                [
                    max(
                        a["bbox"][0],
                        b["bbox"][0]
                    ),
                    max(
                        a["bbox"][1],
                        b["bbox"][1]
                    ),
                    min(
                        a["bbox"][2],
                        b["bbox"][2]
                    ),
                    min(
                        a["bbox"][3],
                        b["bbox"][3]
                    ),
                ]
            )
            /
            max(
                1.0,
                min(
                    box_area(
                        a[
                            "bbox"
                        ]
                    ),
                    box_area(
                        b[
                            "bbox"
                        ]
                    )
                )
            )
        )


        if (
            ov >= 0.55
            or
            containment_ab >= 0.80
        ):

            potential_collisions.append({

                "id_a":
                    a[
                        "inventory_id"
                    ],

                "name_a":
                    a[
                        "inventory_name"
                    ],

                "id_b":
                    b[
                        "inventory_id"
                    ],

                "name_b":
                    b[
                        "inventory_name"
                    ],

                "bbox_iou":
                    ov,

                "smaller_box_containment":
                    containment_ab,

                "status":
                    "POTENTIAL_INSTANCE_COLLISION",
            })


# ============================================================
# SAVE
# ============================================================

ALL_PATH = (
    OUT
    / "00_multi_route_consensus_all.json"
)


SEED_PATH = (
    OUT
    / "01_canonical_sam2_seed_boxes.json"
)


COLLISION_PATH = (
    OUT
    / "02_potential_instance_collisions.json"
)


ALL_PATH.write_text(
    json.dumps(
        consensus_output,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


SEED_PATH.write_text(
    json.dumps(
        canonical_seeds,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


COLLISION_PATH.write_text(
    json.dumps(
        potential_collisions,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# ============================================================
# REPORT
# ============================================================

print()
print("=" * 110)
print("PRODUCTION STAGE 06C6 RESULT")
print("=" * 110)


print(
    "PRESENT INSTANCES:",
    len(
        present_ids
    )
)


print(
    "TOTAL NORMALIZED PROPOSALS:",
    len(
        all_rows
    )
)


print(
    "CANONICAL SAM2 SEEDS:",
    len(
        canonical_seeds
    )
)


print(
    "POTENTIAL CROSS-INSTANCE COLLISIONS:",
    len(
        potential_collisions
    )
)


print()
print(
    "PER PRESENT PHYSICAL INSTANCE:"
)


for instance in consensus_output:

    iid = instance[
        "inventory_id"
    ]

    name = instance[
        "inventory_name"
    ]


    if not instance[
        "clusters"
    ]:

        print(
            f"{iid:03d}. {name} | NO CLUSTER"
        )

        continue


    best = instance[
        "clusters"
    ][0]


    print(
        "{:03d}. {} | proposals={} | clusters={} | best_support={} routes={} | max_dino={:.4f} | consensus={:.4f} | bbox={}".format(

            iid,

            name,

            instance[
                "total_proposals"
            ],

            instance[
                "cluster_count"
            ],

            best[
                "proposal_count"
            ],

            best[
                "routes"
            ],

            best[
                "max_dino_score"
            ],

            best[
                "consensus_score"
            ],

            [
                round(
                    v,
                    1
                )
                for v in best[
                    "bbox"
                ]
            ],
        )
    )


print()
print(
    "POTENTIAL COLLISIONS:"
)


if potential_collisions:

    for row in potential_collisions:

        print(
            "{:03d} {} ↔ {:03d} {} | IoU={:.4f} | containment={:.4f}".format(

                row[
                    "id_a"
                ],

                row[
                    "name_a"
                ],

                row[
                    "id_b"
                ],

                row[
                    "name_b"
                ],

                row[
                    "bbox_iou"
                ],

                row[
                    "smaller_box_containment"
                ],
            )
        )

else:

    print(
        "NONE"
    )


print()
print(
    "CANONICAL SEEDS:",
    SEED_PATH
)

print(
    "FULL CONSENSUS:",
    ALL_PATH
)

print(
    "COLLISION DIAGNOSTIC:",
    COLLISION_PATH
)


print()
print(
    "IMPORTANT:"
)

print(
    "These are canonical LOCALIZATION SEEDS only."
)

print(
    "No Qwen box verification used."
)

print(
    "No SAM2 segmentation has been run."
)
