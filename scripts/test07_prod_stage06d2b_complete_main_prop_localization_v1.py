
from pathlib import Path
import json
import gc
import math

import numpy as np
import torch

from PIL import (
    Image,
    ImageDraw
)

import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE = Path(
    "/workspace/axolotl"
)

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

INVENTORY_PATH = (
    STAGE06
    / "06d2a2_verified_main_prop_state"
    / "00_stage06d2a2_verified_main_prop_state.json"
)

OUT = (
    STAGE06
    / "06d2b_complete_main_prop_localization"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

PREVIEW_DIR = (
    OUT
    / "previews"
)

PREVIEW_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CACHE = (
    "/workspace/data/huggingface-cache"
)

MODEL_ID = (
    "florence-community/Florence-2-large"
)


# ============================================================
# CONFIG
# ============================================================

TOP_K_PER_QUERY = 5

CLUSTER_IOU = 0.35

CLUSTER_CENTER_DISTANCE = 0.55

MAX_CLUSTERS_PER_PROP = 5


# ============================================================
# VALIDATE
# ============================================================

for path in [
    MASTER_PATH,
    INVENTORY_PATH,
]:

    if not path.exists():

        raise FileNotFoundError(
            path
        )


master = Image.open(
    MASTER_PATH
).convert(
    "RGB"
)

W, H = master.size


inventory = json.loads(
    INVENTORY_PATH.read_text(
        encoding="utf-8"
    )
)


main_props = inventory[
    "main_props"
]


print("=" * 110)
print("PRODUCTION STAGE 06D2B")
print("COMPLETE MAIN-PROP LOCALIZATION")
print("=" * 110)

print()
print(
    "FINAL MAIN PROP TARGETS:",
    len(
        main_props
    )
)

for prop in main_props:

    print(
        prop[
            "main_prop_id"
        ],
        "—",
        prop[
            "main_prop_name"
        ]
    )


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

        image = Image.open(
            image_or_path
        )

    else:

        image = image_or_path


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


def clamp_box(
    box
):

    x1, y1, x2, y2 = [
        float(v)
        for v in box
    ]


    x1 = max(
        0.0,
        min(
            float(W - 1),
            x1
        )
    )

    y1 = max(
        0.0,
        min(
            float(H - 1),
            y1
        )
    )

    x2 = max(
        x1 + 1.0,
        min(
            float(W),
            x2
        )
    )

    y2 = max(
        y1 + 1.0,
        min(
            float(H),
            y2
        )
    )


    return [
        x1,
        y1,
        x2,
        y2
    ]


def bbox_area(
    box
):

    return (
        max(
            0.0,
            box[2] - box[0]
        )
        *
        max(
            0.0,
            box[3] - box[1]
        )
    )


def bbox_intersection(
    a,
    b
):

    ix1 = max(
        a[0],
        b[0]
    )

    iy1 = max(
        a[1],
        b[1]
    )

    ix2 = min(
        a[2],
        b[2]
    )

    iy2 = min(
        a[3],
        b[3]
    )


    return (
        max(
            0.0,
            ix2 - ix1
        )
        *
        max(
            0.0,
            iy2 - iy1
        )
    )


def bbox_iou(
    a,
    b
):

    inter = bbox_intersection(
        a,
        b
    )

    union = (
        bbox_area(a)
        +
        bbox_area(b)
        -
        inter
    )

    return float(
        inter
        /
        max(
            1.0,
            union
        )
    )


def center_distance(
    a,
    b
):

    acx = (
        a[0] + a[2]
    ) / 2.0

    acy = (
        a[1] + a[3]
    ) / 2.0


    bcx = (
        b[0] + b[2]
    ) / 2.0

    bcy = (
        b[1] + b[3]
    ) / 2.0


    dist = math.sqrt(
        (
            acx - bcx
        ) ** 2
        +
        (
            acy - bcy
        ) ** 2
    )


    diag_a = math.sqrt(
        (
            a[2] - a[0]
        ) ** 2
        +
        (
            a[3] - a[1]
        ) ** 2
    )


    diag_b = math.sqrt(
        (
            b[2] - b[0]
        ) ** 2
        +
        (
            b[3] - b[1]
        ) ** 2
    )


    reference = max(
        1.0,
        min(
            diag_a,
            diag_b
        )
    )


    return float(
        dist
        /
        reference
    )


def compatible(
    a,
    b
):

    return bool(

        bbox_iou(
            a,
            b
        )
        >=
        CLUSTER_IOU

        or

        center_distance(
            a,
            b
        )
        <=
        CLUSTER_CENTER_DISTANCE
    )


def box_median(
    boxes
):

    array = np.asarray(
        boxes,
        dtype=np.float32
    )

    return [

        float(v)

        for v in np.median(
            array,
            axis=0
        )
    ]


def extract_florence_boxes(
    parsed
):

    result = []


    if not isinstance(
        parsed,
        dict
    ):

        return result


    for _, value in parsed.items():

        if not isinstance(
            value,
            dict
        ):

            continue


        boxes = value.get(
            "bboxes",
            []
        )

        labels = value.get(
            "labels",
            []
        )

        scores = value.get(
            "scores",
            []
        )


        for i, box in enumerate(
            boxes
        ):

            if (
                not isinstance(
                    box,
                    (list, tuple)
                )
                or
                len(box) != 4
            ):

                continue


            result.append({

                "bbox":
                    clamp_box(
                        box
                    ),

                "label":
                    (
                        str(
                            labels[i]
                        )
                        if
                        i < len(
                            labels
                        )
                        else ""
                    ),

                "score":
                    (
                        float(
                            scores[i]
                        )
                        if
                        i < len(
                            scores
                        )
                        else None
                    ),
            })


    return result


# ============================================================
# BUILD WHOLE-PROP QUERY VARIANTS
# ============================================================

def build_queries(
    prop
):

    name = str(
        prop[
            "main_prop_name"
        ]
    ).strip()


    members = [

        str(x).strip()

        for x in prop.get(
            "visible_members",
            []
        )

        if str(x).strip()
    ]


    prop_type = str(
        prop.get(
            "prop_type",
            ""
        )
    )


    member_text = ", ".join(
        members
    )


    queries = []


    # --------------------------------------------------------
    # Query 1 — canonical complete prop name
    # --------------------------------------------------------

    queries.append(
        f"the complete {name}"
    )


    # --------------------------------------------------------
    # Query 2 — explicit complete system + member checklist
    # --------------------------------------------------------

    if members:

        queries.append(
            (
                f"the entire visible {name} including "
                f"{member_text}"
            )
        )


    # --------------------------------------------------------
    # Query 3 differs by connected vs standalone.
    # --------------------------------------------------------

    if prop_type == "CONNECTED_SYSTEM":

        queries.append(
            (
                f"the whole connected physical assembly of "
                f"{member_text}"
            )
        )

    else:

        queries.append(
            f"the complete visible {name} fixture"
        )


    # --------------------------------------------------------
    # Deduplicate text
    # --------------------------------------------------------

    unique = []

    seen = set()


    for query in queries:

        key = query.lower().strip()

        if key in seen:
            continue

        seen.add(
            key
        )

        unique.append(
            query
        )


    return unique


# ============================================================
# LOAD FLORENCE
# ============================================================

from transformers import (
    AutoProcessor,
    AutoModelForMultimodalLM,
)


print()
print(
    "Loading Florence-2 Large..."
)


processor = (
    AutoProcessor
    .from_pretrained(
        MODEL_ID,
        cache_dir=CACHE
    )
)


model = (
    AutoModelForMultimodalLM
    .from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        device_map="auto",
        cache_dir=CACHE
    )
)


model.eval()


print(
    "✅ FLORENCE READY"
)


# ============================================================
# RUN FLORENCE
# ============================================================

def run_florence(
    phrase
):

    task = (
        "<CAPTION_TO_PHRASE_GROUNDING>"
    )


    inputs = processor(
        text=(
            task
            +
            phrase
        ),
        images=master,
        return_tensors="pt"
    )


    inputs = {

        k:
            (
                v.to(
                    model.device
                )
                if hasattr(
                    v,
                    "to"
                )
                else v
            )

        for k, v in inputs.items()
    }


    with torch.inference_mode():

        generated = model.generate(
            **inputs,
            max_new_tokens=256,
            num_beams=3,
            do_sample=False
        )


    text = (
        processor
        .batch_decode(
            generated,
            skip_special_tokens=False
        )[0]
    )


    parsed = (
        processor
        .post_process_generation(
            text,
            task=task,
            image_size=master.size
        )
    )


    return (
        extract_florence_boxes(
            parsed
        )
    )


# ============================================================
# LOCALIZE EACH FINAL MAIN PROP
# ============================================================

results = []


for prop in main_props:

    prop_id = prop[
        "main_prop_id"
    ]

    prop_name = prop[
        "main_prop_name"
    ]

    prop_type = prop[
        "prop_type"
    ]


    print()
    print("=" * 110)

    print(
        f"{prop_id}. {prop_name}"
    )

    print("=" * 110)


    queries = build_queries(
        prop
    )


    proposals = []


    for query_index, query in enumerate(
        queries,
        start=1
    ):

        boxes = run_florence(
            query
        )


        boxes = boxes[
            :TOP_K_PER_QUERY
        ]


        print()
        print(
            f"QUERY {query_index}:",
            query
        )

        print(
            "BOXES:",
            len(
                boxes
            )
        )


        for box_index, row in enumerate(
            boxes,
            start=1
        ):

            proposal = {

                "query_index":
                    query_index,

                "query":
                    query,

                "box_index":
                    box_index,

                "bbox":
                    row[
                        "bbox"
                    ],

                "label":
                    row[
                        "label"
                    ],

                "score":
                    row[
                        "score"
                    ],

                "area":
                    bbox_area(
                        row[
                            "bbox"
                        ]
                    ),
            }


            proposals.append(
                proposal
            )


            print(
                "  #{} bbox={} label={}".format(

                    box_index,

                    [
                        round(v, 1)
                        for v in row[
                            "bbox"
                        ]
                    ],

                    row[
                        "label"
                    ],
                )
            )


    # ========================================================
    # CLUSTER PROPOSALS
    # ========================================================

    clusters = []


    # --------------------------------------------------------
    # Larger boxes first for connected systems.
    #
    # For standalone fixtures area sorting is less important.
    # --------------------------------------------------------

    if prop_type == "CONNECTED_SYSTEM":

        ordered = sorted(
            proposals,
            key=lambda row:
                row[
                    "area"
                ],
            reverse=True
        )

    else:

        ordered = list(
            proposals
        )


    for proposal in ordered:

        assigned = False


        for cluster in clusters:

            if compatible(
                proposal[
                    "bbox"
                ],
                cluster[
                    "bbox"
                ]
            ):

                cluster[
                    "members"
                ].append(
                    proposal
                )


                cluster[
                    "bbox"
                ] = box_median(
                    [
                        member[
                            "bbox"
                        ]
                        for member
                        in cluster[
                            "members"
                        ]
                    ]
                )


                assigned = True
                break


        if not assigned:

            clusters.append({

                "bbox":
                    list(
                        proposal[
                            "bbox"
                        ]
                    ),

                "members":
                    [
                        proposal
                    ],
            })


    # ========================================================
    # CLUSTER METADATA
    # ========================================================

    cluster_rows = []


    for cluster_index, cluster in enumerate(
        clusters,
        start=1
    ):

        query_support = sorted({

            member[
                "query_index"
            ]

            for member
            in cluster[
                "members"
            ]
        })


        max_area = max(

            member[
                "area"
            ]

            for member
            in cluster[
                "members"
            ]
        )


        median_area = float(
            np.median(
                [
                    member[
                        "area"
                    ]

                    for member
                    in cluster[
                        "members"
                    ]
                ]
            )
        )


        # ----------------------------------------------------
        # Rank:
        #
        # 1. number of different whole-prop queries supporting
        # 2. proposal count
        # 3. for connected systems prefer larger complete box
        # ----------------------------------------------------

        support_score = (

            len(
                query_support
            )
            *
            1000

            +

            len(
                cluster[
                    "members"
                ]
            )
            *
            100
        )


        if prop_type == "CONNECTED_SYSTEM":

            support_score += (
                median_area
                /
                max(
                    1.0,
                    W * H
                )
                *
                100
            )


        cluster_rows.append({

            "cluster_index":
                cluster_index,

            "bbox":
                clamp_box(
                    cluster[
                        "bbox"
                    ]
                ),

            "query_support":
                query_support,

            "query_support_count":
                len(
                    query_support
                ),

            "proposal_count":
                len(
                    cluster[
                        "members"
                    ]
                ),

            "median_area":
                median_area,

            "max_area":
                max_area,

            "ranking_score":
                float(
                    support_score
                ),

            "members":
                cluster[
                    "members"
                ],
        })


    cluster_rows.sort(
        key=lambda row:
            row[
                "ranking_score"
            ],
        reverse=True
    )


    cluster_rows = cluster_rows[
        :MAX_CLUSTERS_PER_PROP
    ]


    # Re-rank visible indices 1..N
    for rank, cluster in enumerate(
        cluster_rows,
        start=1
    ):

        cluster[
            "rank"
        ] = rank


    # ========================================================
    # PREVIEW
    # ========================================================

    preview = master.copy()

    draw = ImageDraw.Draw(
        preview
    )


    for cluster in cluster_rows:

        x1, y1, x2, y2 = (
            cluster[
                "bbox"
            ]
        )


        rank = cluster[
            "rank"
        ]


        draw.rectangle(
            [
                x1,
                y1,
                x2,
                y2
            ],
            outline="red",
            width=3
        )


        draw.rectangle(
            [
                x1,
                y1,
                x1 + 42,
                y1 + 25
            ],
            fill="white",
            outline="red"
        )


        draw.text(
            (
                x1 + 6,
                y1 + 4
            ),
            str(
                rank
            ),
            fill="red"
        )


    preview_path = (
        PREVIEW_DIR
        /
        f"{prop_id}_complete_prop_candidates.png"
    )


    preview.save(
        preview_path
    )


    # ========================================================
    # SAVE PROP RESULT
    # ========================================================

    result = {

        "main_prop_id":
            prop_id,

        "main_prop_name":
            prop_name,

        "prop_type":
            prop_type,

        "visible_members":
            prop.get(
                "visible_members",
                []
            ),

        "queries":
            queries,

        "raw_proposal_count":
            len(
                proposals
            ),

        "candidate_clusters":
            cluster_rows,

        "preview_path":
            str(
                preview_path
            ),

        "status":
            (
                "CANDIDATES_READY"
                if cluster_rows
                else
                "NO_LOCALIZATION"
            ),
    }


    results.append(
        result
    )


    print()
    print(
        "RAW PROPOSALS:",
        len(
            proposals
        )
    )

    print(
        "FINAL CANDIDATE CLUSTERS:",
        len(
            cluster_rows
        )
    )


    for cluster in cluster_rows:

        print(
            "  rank={} | queries={} | proposals={} | "
            "bbox={} | median_area={:.1f}".format(

                cluster[
                    "rank"
                ],

                cluster[
                    "query_support"
                ],

                cluster[
                    "proposal_count"
                ],

                [
                    round(v, 1)
                    for v in cluster[
                        "bbox"
                    ]
                ],

                cluster[
                    "median_area"
                ],
            )
        )


# ============================================================
# UNLOAD MODEL
# ============================================================

del model
del processor

gc.collect()

if torch.cuda.is_available():

    torch.cuda.empty_cache()


# ============================================================
# SAVE GLOBAL RESULT
# ============================================================

FINAL_STATE = {

    "stage":
        "06D2B",

    "input_inventory":
        str(
            INVENTORY_PATH
        ),

    "detection_master":
        str(
            MASTER_PATH
        ),

    "model":
        MODEL_ID,

    "main_prop_count":
        len(
            main_props
        ),

    "results":
        results,

    "status":
        "REQUIRES_COMPLETE_MAIN_PROP_LOCALIZATION_AUDIT",

    "rules": [

        "only final main props are localization targets",

        "child components are not independently localized",

        "connected systems are queried as complete assemblies",

        "no SAM2 was run",

        "no final prop mask was created",

        "Stage01 remains final RGB source",
    ],
}


RESULT_PATH = (
    OUT
    / "00_stage06d2b_result.json"
)


RESULT_PATH.write_text(
    json.dumps(
        FINAL_STATE,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print()
print("=" * 110)
print("PRODUCTION STAGE 06D2B RESULT")
print("=" * 110)


for result in results:

    print()
    print(
        "{}. {} | status={} | candidates={}".format(

            result[
                "main_prop_id"
            ],

            result[
                "main_prop_name"
            ],

            result[
                "status"
            ],

            len(
                result[
                    "candidate_clusters"
                ]
            ),
        )
    )


    for cluster in result[
        "candidate_clusters"
    ]:

        print(
            "   #{} bbox={} | query_support={} | proposals={}".format(

                cluster[
                    "rank"
                ],

                [
                    round(v, 1)
                    for v in cluster[
                        "bbox"
                    ]
                ],

                cluster[
                    "query_support_count"
                ],

                cluster[
                    "proposal_count"
                ],
            )
        )


print()
print(
    "RESULT JSON:",
    RESULT_PATH
)

print()
print(
    "NO CHILD-COMPONENT LOCALIZATION WAS PERFORMED."
)

print(
    "NO SAM2 OR FINAL MASK UNION WAS RUN."
)


# ============================================================
# INLINE REVIEW — ALL SIX FINAL PROP TARGETS
# ============================================================

print()
print("=" * 110)
print("INLINE 06D2B COMPLETE MAIN-PROP LOCALIZATION AUDIT")
print("=" * 110)


for result in results:

    show(

        result[
            "preview_path"
        ],

        (
            f'{result["main_prop_id"]} — '
            f'{result["main_prop_name"]}\n'
            f'Red Boxes = Whole Main-Prop Candidate Clusters'
        ),

        figsize=(8, 9)
    )
