
from pathlib import Path
import json
import math

import cv2
import numpy as np
from PIL import Image

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

ROOM_PATH = (
    PROD
    / "stage07_empty_room"
    / "07a_qwen_empty_room"
    / "00_stage07_empty_room_candidate.png"
)

E1_STATE_PATH = (
    PROD
    / "stage08_tile_application"
    / "08e1_floor_line_geometry"
    / "00_stage08e1_result.json"
)

OUT = (
    PROD
    / "stage08_tile_application"
    / "08e2_floor_vanishing_points"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# VALIDATE / LOAD
# ============================================================

for p in [
    ROOM_PATH,
    E1_STATE_PATH,
]:
    if not p.exists():
        raise FileNotFoundError(p)


room = Image.open(
    ROOM_PATH
).convert("RGB")

room_np = np.asarray(room)

H, W = room_np.shape[:2]


state = json.loads(
    E1_STATE_PATH.read_text(
        encoding="utf-8"
    )
)


segments = state.get(
    "segments",
    []
)


if len(segments) < 4:
    raise RuntimeError(
        "Too few 08E1 segments."
    )


print("=" * 110)
print("STAGE 08E2 — FLOOR VANISHING POINTS")
print("=" * 110)

print()
print(
    "SEGMENTS:",
    len(segments)
)


# ============================================================
# LINE REPRESENTATION
#
# image points -> homogeneous line:
# l = p1 × p2
# ============================================================

def line_from_segment(row):

    x1, y1 = row["p1"]
    x2, y2 = row["p2"]

    p1 = np.array(
        [x1, y1, 1.0],
        dtype=np.float64
    )

    p2 = np.array(
        [x2, y2, 1.0],
        dtype=np.float64
    )

    line = np.cross(
        p1,
        p2
    )

    norm = math.hypot(
        line[0],
        line[1]
    )

    if norm < 1e-9:
        return None

    line /= norm

    return line


# ============================================================
# GROUP BY 08E1 CLUSTER
# ============================================================

clusters = {
    0: [],
    1: [],
}


for row in segments:

    cid = int(
        row.get(
            "cluster",
            -1
        )
    )

    if cid not in clusters:
        continue

    line = line_from_segment(
        row
    )

    if line is None:
        continue

    clusters[cid].append(
        {
            "line":
                line,

            "segment":
                row,

            "weight":
                float(
                    row.get(
                        "length",
                        1.0
                    )
                )
        }
    )


for cid in [0, 1]:

    print()
    print(
        "CLUSTER",
        cid,
        "LINES:",
        len(
            clusters[cid]
        )
    )


# ============================================================
# PAIRWISE INTERSECTIONS
# ============================================================

def pair_intersections(rows):

    pts = []

    for i in range(len(rows)):

        for j in range(
            i + 1,
            len(rows)
        ):

            l1 = rows[i]["line"]
            l2 = rows[j]["line"]

            p = np.cross(
                l1,
                l2
            )

            if abs(
                p[2]
            ) < 1e-8:
                continue

            x = p[0] / p[2]
            y = p[1] / p[2]

            if not (
                np.isfinite(x)
                and
                np.isfinite(y)
            ):
                continue

            # Reject absurd numerical intersections.
            if (
                abs(x) > 100000
                or
                abs(y) > 100000
            ):
                continue

            weight = math.sqrt(
                rows[i]["weight"]
                *
                rows[j]["weight"]
            )

            pts.append(
                [
                    float(x),
                    float(y),
                    float(weight)
                ]
            )

    return pts


# ============================================================
# WEIGHTED ROBUST CENTER
# ============================================================

def robust_vp(rows):

    intersections = pair_intersections(
        rows
    )

    # --------------------------------------------------------
    # If enough pair intersections exist:
    # robust median + MAD filtering
    # --------------------------------------------------------

    if len(intersections) >= 3:

        arr = np.asarray(
            intersections,
            dtype=np.float64
        )

        xy = arr[:, :2]
        weights = arr[:, 2]

        center0 = np.median(
            xy,
            axis=0
        )

        dist = np.linalg.norm(
            xy - center0,
            axis=1
        )

        med = np.median(
            dist
        )

        mad = np.median(
            np.abs(
                dist - med
            )
        )

        scale = max(
            5.0,
            1.4826 * mad
        )

        keep = (
            dist
            <=
            med + 2.5 * scale
        )

        if keep.sum() >= 2:

            xy_keep = xy[
                keep
            ]

            w_keep = weights[
                keep
            ]

            vp = np.average(
                xy_keep,
                axis=0,
                weights=w_keep
            )

            return (
                vp,
                intersections,
                keep.tolist(),
                "PAIRWISE_ROBUST"
            )


    # --------------------------------------------------------
    # Fallback:
    # weighted least-squares intersection of all lines.
    #
    # Solve:
    # a*x + b*y + c = 0
    # --------------------------------------------------------

    if len(rows) >= 2:

        A = []
        b = []
        w = []

        for row in rows:

            a, bb, c = row["line"]

            A.append(
                [a, bb]
            )

            b.append(
                -c
            )

            w.append(
                row["weight"]
            )

        A = np.asarray(
            A,
            dtype=np.float64
        )

        b = np.asarray(
            b,
            dtype=np.float64
        )

        w = np.asarray(
            w,
            dtype=np.float64
        )

        sw = np.sqrt(
            w / max(
                w.mean(),
                1e-9
            )
        )

        Aw = (
            A
            *
            sw[:, None]
        )

        bw = (
            b
            *
            sw
        )

        vp, _, _, _ = np.linalg.lstsq(
            Aw,
            bw,
            rcond=None
        )

        return (
            vp,
            intersections,
            [],
            "WEIGHTED_LEAST_SQUARES"
        )


    raise RuntimeError(
        "Cluster has fewer than 2 usable lines."
    )


# ============================================================
# ESTIMATE BOTH VPS
# ============================================================

vp_results = {}


for cid in [0, 1]:

    vp, intersections, kept, method = robust_vp(
        clusters[cid]
    )

    vp_results[
        cid
    ] = {
        "vp":
            [
                float(
                    vp[0]
                ),
                float(
                    vp[1]
                )
            ],

        "method":
            method,

        "pair_intersections":
            intersections,

        "kept_pair_flags":
            kept
    }


    print()
    print(
        "CLUSTER",
        cid,
        "VP:",
        [
            round(
                float(vp[0]),
                2
            ),
            round(
                float(vp[1]),
                2
            )
        ],
        "| METHOD:",
        method
    )


# ============================================================
# HORIZON LINE
#
# Both floor-direction vanishing points must lie on the floor
# horizon / camera horizon.
# ============================================================

vp0 = np.array(
    [
        vp_results[0]["vp"][0],
        vp_results[0]["vp"][1],
        1.0
    ],
    dtype=np.float64
)

vp1 = np.array(
    [
        vp_results[1]["vp"][0],
        vp_results[1]["vp"][1],
        1.0
    ],
    dtype=np.float64
)


horizon = np.cross(
    vp0,
    vp1
)


hnorm = math.hypot(
    horizon[0],
    horizon[1]
)


if hnorm > 1e-9:
    horizon /= hnorm


# ============================================================
# VISUALIZATION
#
# Extend each detected segment toward its cluster VP.
# ============================================================

preview = room_np.copy()


colors = {
    0: (255, 0, 0),
    1: (0, 255, 255),
}


for cid in [0, 1]:

    vp = np.asarray(
        vp_results[cid]["vp"],
        dtype=np.float64
    )


    for row in clusters[cid]:

        seg = row["segment"]

        p1 = np.asarray(
            seg["p1"],
            dtype=np.float64
        )

        p2 = np.asarray(
            seg["p2"],
            dtype=np.float64
        )


        # Use segment midpoint and draw toward VP,
        # clipped by OpenCV to image bounds.
        midpoint = (
            p1 + p2
        ) / 2.0


        direction = (
            midpoint - vp
        )


        norm = np.linalg.norm(
            direction
        )


        if norm < 1e-6:
            continue


        direction /= norm


        # create a very long line through VP and midpoint
        q1 = (
            vp
            -
            direction
            *
            5000.0
        )

        q2 = (
            vp
            +
            direction
            *
            5000.0
        )


        pt1 = (
            int(
                round(
                    q1[0]
                )
            ),
            int(
                round(
                    q1[1]
                )
            )
        )

        pt2 = (
            int(
                round(
                    q2[0]
                )
            ),
            int(
                round(
                    q2[1]
                )
            )
        )


        # clip infinite-ish line to image
        ok, c1, c2 = cv2.clipLine(
            (
                0,
                0,
                W,
                H
            ),
            pt1,
            pt2
        )


        if ok:

            cv2.line(
                preview,
                c1,
                c2,
                colors[cid],
                1,
                cv2.LINE_AA
            )


# ============================================================
# DRAW HORIZON IF IT CROSSES IMAGE
# ============================================================

a, b, c = horizon


if abs(b) > 1e-9:

    y_left = (
        -c
        -
        a * 0
    ) / b

    y_right = (
        -c
        -
        a * (W - 1)
    ) / b


    p1_h = (
        0,
        int(
            round(
                y_left
            )
        )
    )

    p2_h = (
        W - 1,
        int(
            round(
                y_right
            )
        )
    )


    ok, hp1, hp2 = cv2.clipLine(
        (
            0,
            0,
            W,
            H
        ),
        p1_h,
        p2_h
    )


    if ok:

        cv2.line(
            preview,
            hp1,
            hp2,
            (
                255,
                0,
                255
            ),
            2,
            cv2.LINE_AA
        )


# ============================================================
# SECOND VISUAL:
# detected segments only + VP labels in expanded coordinates
# ============================================================

segment_preview = room_np.copy()


for row in segments:

    cid = int(
        row["cluster"]
    )

    cv2.line(
        segment_preview,
        tuple(
            row["p1"]
        ),
        tuple(
            row["p2"]
        ),
        colors[cid],
        2,
        cv2.LINE_AA
    )


# ============================================================
# 4-PANEL AUDIT
# ============================================================

fig, axes = plt.subplots(
    1,
    4,
    figsize=(
        18,
        7
    )
)


axes[0].imshow(
    room
)

axes[0].set_title(
    "1. Stage07A"
)

axes[0].axis(
    "off"
)


axes[1].imshow(
    segment_preview
)

axes[1].set_title(
    "2. 08E1 Line Clusters"
)

axes[1].axis(
    "off"
)


axes[2].imshow(
    preview
)

axes[2].set_title(
    "3. Lines Extended Toward VPs\n"
    "MAGENTA = horizon if visible"
)

axes[2].axis(
    "off"
)


# ------------------------------------------------------------
# VP coordinate plot
# ------------------------------------------------------------

axes[3].imshow(
    room,
    alpha=0.35
)


v0 = vp_results[0]["vp"]
v1 = vp_results[1]["vp"]


axes[3].scatter(
    [
        v0[0],
        v1[0]
    ],
    [
        v0[1],
        v1[1]
    ],
    s=80
)


axes[3].text(
    v0[0],
    v0[1],
    " VP0",
    fontsize=9
)

axes[3].text(
    v1[0],
    v1[1],
    " VP1",
    fontsize=9
)


# Expand plot so off-image VPs can be seen,
# but cap to a reasonable diagnostic range.
all_x = [
    0,
    W,
    v0[0],
    v1[0]
]

all_y = [
    0,
    H,
    v0[1],
    v1[1]
]


xmin = max(
    -3000,
    min(all_x) - 100
)

xmax = min(
    3000,
    max(all_x) + 100
)

ymin = max(
    -3000,
    min(all_y) - 100
)

ymax = min(
    3000,
    max(all_y) + 100
)


axes[3].set_xlim(
    xmin,
    xmax
)

axes[3].set_ylim(
    ymax,
    ymin
)

axes[3].set_title(
    "4. Estimated Vanishing Points"
)


plt.tight_layout()


AUDIT_PATH = (
    OUT
    / "01_stage08e2_vanishing_point_audit.png"
)


plt.savefig(
    AUDIT_PATH,
    dpi=150,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# STATE
# ============================================================

STATE = {

    "stage":
        "08E2",

    "purpose":
        "ESTIMATE_TWO_FLOOR_VANISHING_POINTS",

    "source_state":
        str(
            E1_STATE_PATH
        ),

    "vanishing_points": {

        "cluster_0":
            vp_results[0],

        "cluster_1":
            vp_results[1]
    },

    "horizon_line_abc":
        [
            float(
                horizon[0]
            ),
            float(
                horizon[1]
            ),
            float(
                horizon[2]
            )
        ],

    "status":
        "REQUIRES_VANISHING_POINT_VISUAL_AUDIT",

    "next_if_pass":
        "08E3_METRIC_FLOOR_COORDINATE_CALIBRATION",

    "outputs": {

        "audit":
            str(
                AUDIT_PATH
            )
    }
}


STATE_PATH = (
    OUT
    / "00_stage08e2_result.json"
)


STATE_PATH.write_text(
    json.dumps(
        STATE,
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
print("STAGE 08E2 RESULT")
print("=" * 110)

print()

for cid in [0, 1]:

    row = vp_results[cid]

    print(
        "VP",
        cid,
        "=",
        [
            round(
                row["vp"][0],
                2
            ),
            round(
                row["vp"][1],
                2
            )
        ],
        "|",
        row["method"]
    )


print()
print(
    "HORIZON [a,b,c]:",
    [
        round(
            float(x),
            6
        )
        for x in horizon
    ]
)

print()
print(
    "AUDIT:",
    AUDIT_PATH
)

print(
    "STATE:",
    STATE_PATH
)

print()
print(
    "NO TILE WAS APPLIED."
)
