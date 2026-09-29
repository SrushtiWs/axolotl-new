
from pathlib import Path
import json

import numpy as np
from PIL import Image, ImageDraw

import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector


# ============================================================
# PATHS
# ============================================================

BASE = Path("/workspace/axolotl")

PROD = (
    BASE
    / "test07"
    / "production_pipeline"
)

IMAGE_PATH = (
    PROD
    / "stage01_master"
    / "00_master_input.png"
)

FLOOR_MASK_PATH = (
    PROD
    / "stage08_tile_application"
    / "08b2_floor_plus_screeding_target"
    / "01_main_floor_mask.png"
)

OUT = (
    PROD
    / "stage07_empty_room"
    / "07e0_shadow_ground_truth"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD
# ============================================================

for p in [
    IMAGE_PATH,
    FLOOR_MASK_PATH,
]:
    if not p.exists():
        raise FileNotFoundError(p)


image = Image.open(
    IMAGE_PATH
).convert("RGB")

image_np = np.asarray(image)

W, H = image.size


floor_mask = (
    np.asarray(
        Image.open(
            FLOOR_MASK_PATH
        ).convert("L")
    ) > 127
)


# ============================================================
# STATE
# ============================================================

current_object = "P01"

polygons = {
    "P01": [],
    "P02": [],
}

saved_lines = []


# ============================================================
# FIGURE
# ============================================================

plt.close("all")

fig, ax = plt.subplots(
    figsize=(8, 10)
)

ax.imshow(image_np)

ax.set_xlim(
    0,
    W
)

ax.set_ylim(
    H,
    0
)

ax.axis("off")


def update_title():

    if current_object == "P01":

        obj = "P01 VANITY SHADOW"

    else:

        obj = "P02 TOILET SHADOW"

    ax.set_title(
        f"{obj}\n"
        "Draw + close polygon = AUTO SAVE\n"
        "N=next | U=undo | C=clear | S=save",
        fontsize=11
    )

    fig.canvas.draw_idle()


update_title()


# ============================================================
# REDRAW SAVED POLYGONS
# ============================================================

def redraw():

    global saved_lines

    for line in saved_lines:
        try:
            line.remove()
        except Exception:
            pass

    saved_lines = []


    for obj in [
        "P01",
        "P02",
    ]:

        color = (
            "red"
            if obj == "P01"
            else "cyan"
        )


        for poly in polygons[obj]:

            if len(poly) < 3:
                continue

            pts = np.asarray(
                poly,
                dtype=np.float32
            )

            closed = np.vstack(
                [
                    pts,
                    pts[0]
                ]
            )


            line, = ax.plot(
                closed[:, 0],
                closed[:, 1],
                color=color,
                linewidth=2.2
            )

            saved_lines.append(line)


    fig.canvas.draw_idle()


# ============================================================
# POLYGON CALLBACK
#
# IMPORTANT:
# Completed polygon is saved automatically.
# ============================================================

def on_polygon_complete(verts):

    if len(verts) < 3:
        return


    poly = [
        [
            float(x),
            float(y)
        ]
        for x, y in verts
    ]


    polygons[
        current_object
    ].append(
        poly
    )


    print(
        f"✅ {current_object} polygon "
        f"#{len(polygons[current_object])} saved automatically."
    )


    redraw()


selector = PolygonSelector(
    ax,
    on_polygon_complete,
    useblit=True
)


# ============================================================
# MASK CREATION
# ============================================================

def polygons_to_mask(poly_list):

    mask_img = Image.new(
        "L",
        (
            W,
            H
        ),
        0
    )


    draw = ImageDraw.Draw(
        mask_img
    )


    for poly in poly_list:

        if len(poly) < 3:
            continue


        pts = [
            (
                int(round(x)),
                int(round(y))
            )
            for x, y in poly
        ]


        draw.polygon(
            pts,
            fill=255
        )


    mask = (
        np.asarray(mask_img)
        >
        127
    )


    # Ground truth only exists on the floor.
    mask &= floor_mask


    return mask


# ============================================================
# SAVE
# ============================================================

def save_ground_truth():

    if len(polygons["P01"]) == 0:

        print(
            "❌ P01 has no polygon."
        )

        return


    if len(polygons["P02"]) == 0:

        print(
            "❌ P02 has no polygon."
        )

        return


    p01_mask = polygons_to_mask(
        polygons["P01"]
    )

    p02_mask = polygons_to_mask(
        polygons["P02"]
    )


    combined = (
        p01_mask
        |
        p02_mask
    )


    # --------------------------------------------------------
    # OUTPUT PATHS
    # --------------------------------------------------------

    P01_PATH = (
        OUT
        / "01_p01_vanity_shadow_ground_truth.png"
    )

    P02_PATH = (
        OUT
        / "02_p02_toilet_shadow_ground_truth.png"
    )

    COMBINED_PATH = (
        OUT
        / "03_combined_floor_shadow_ground_truth.png"
    )

    OVERLAY_PATH = (
        OUT
        / "04_ground_truth_overlay.png"
    )


    # --------------------------------------------------------
    # SAVE MASKS
    # --------------------------------------------------------

    Image.fromarray(
        p01_mask.astype(np.uint8) * 255
    ).save(
        P01_PATH
    )


    Image.fromarray(
        p02_mask.astype(np.uint8) * 255
    ).save(
        P02_PATH
    )


    Image.fromarray(
        combined.astype(np.uint8) * 255
    ).save(
        COMBINED_PATH
    )


    # --------------------------------------------------------
    # OVERLAY
    # --------------------------------------------------------

    overlay = image_np.astype(
        np.float32
    ).copy()


    overlay[
        p01_mask
    ] = (

        overlay[
            p01_mask
        ]
        *
        0.42

        +

        np.array(
            [255, 0, 0],
            dtype=np.float32
        )
        *
        0.58
    )


    overlay[
        p02_mask
    ] = (

        overlay[
            p02_mask
        ]
        *
        0.42

        +

        np.array(
            [0, 255, 255],
            dtype=np.float32
        )
        *
        0.58
    )


    overlay = np.clip(
        overlay,
        0,
        255
    ).astype(
        np.uint8
    )


    Image.fromarray(
        overlay
    ).save(
        OVERLAY_PATH
    )


    # --------------------------------------------------------
    # STATE
    # --------------------------------------------------------

    STATE = {

        "stage":
            "07E0",

        "purpose":
            "TEST07_MANUAL_SHADOW_GROUND_TRUTH",

        "production_component":
            False,

        "classes": {

            "P01":
                "vanity floor cast/contact shadow",

            "P02":
                "toilet floor cast/contact shadow"
        },

        "polygon_counts": {

            "P01":
                len(
                    polygons["P01"]
                ),

            "P02":
                len(
                    polygons["P02"]
                )
        },

        "pixel_counts": {

            "P01":
                int(
                    p01_mask.sum()
                ),

            "P02":
                int(
                    p02_mask.sum()
                ),

            "combined":
                int(
                    combined.sum()
                )
        },

        "polygons":
            polygons,

        "outputs": {

            "p01":
                str(P01_PATH),

            "p02":
                str(P02_PATH),

            "combined":
                str(COMBINED_PATH),

            "overlay":
                str(OVERLAY_PATH)
        },

        "status":
            "GROUND_TRUTH_CREATED"
    }


    STATE_PATH = (
        OUT
        / "00_stage07e0_ground_truth.json"
    )


    STATE_PATH.write_text(
        json.dumps(
            STATE,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )


    print()
    print("=" * 100)
    print("✅ STAGE 07E0 GROUND TRUTH SAVED")
    print("=" * 100)

    print()

    print(
        "P01 POLYGONS:",
        len(
            polygons["P01"]
        )
    )

    print(
        "P01 PIXELS:",
        int(
            p01_mask.sum()
        )
    )

    print()

    print(
        "P02 POLYGONS:",
        len(
            polygons["P02"]
        )
    )

    print(
        "P02 PIXELS:",
        int(
            p02_mask.sum()
        )
    )

    print()

    print(
        "COMBINED PIXELS:",
        int(
            combined.sum()
        )
    )

    print()

    print(
        "OVERLAY:",
        OVERLAY_PATH
    )

    print(
        "STATE:",
        STATE_PATH
    )


# ============================================================
# KEYBOARD CONTROLS
# ============================================================

def on_key(event):

    global current_object


    key = (
        event.key.lower()
        if event.key is not None
        else ""
    )


    # --------------------------------------------------------
    # N = NEXT OBJECT
    # --------------------------------------------------------

    if key == "n":

        if current_object == "P01":

            if len(
                polygons["P01"]
            ) == 0:

                print(
                    "❌ Draw P01 first."
                )

                return


            current_object = "P02"

            print()
            print(
                "➡️ NOW ANNOTATING P02 TOILET SHADOW"
            )

            update_title()


        else:

            print(
                "Already annotating P02."
            )


    # --------------------------------------------------------
    # U = UNDO
    # --------------------------------------------------------

    elif key == "u":

        if polygons[
            current_object
        ]:

            polygons[
                current_object
            ].pop()

            print(
                f"↩️ Removed last {current_object} polygon."
            )

            redraw()

        else:

            print(
                f"No {current_object} polygon to undo."
            )


    # --------------------------------------------------------
    # C = CLEAR CURRENT
    # --------------------------------------------------------

    elif key == "c":

        polygons[
            current_object
        ] = []

        print(
            f"🗑️ Cleared all {current_object} polygons."
        )

        redraw()


    # --------------------------------------------------------
    # S = SAVE
    # --------------------------------------------------------

    elif key == "s":

        save_ground_truth()


fig.canvas.mpl_connect(
    "key_press_event",
    on_key
)


# ============================================================
# FINAL INSTRUCTIONS
# ============================================================

print()
print("=" * 100)
print("07E0 KEYBOARD ANNOTATION ACTIVE")
print("=" * 100)

print()

print("STEP 1:")
print("Draw and CLOSE P01 vanity-shadow polygon.")
print("It saves automatically.")

print()
print("STEP 2:")
print("Press N on keyboard.")

print()
print("STEP 3:")
print("Draw and CLOSE P02 toilet-shadow polygon.")
print("It saves automatically.")

print()
print("STEP 4:")
print("Press S to save ground truth.")

print()
print("OTHER KEYS:")
print("U = undo last polygon")
print("C = clear current object's polygons")

print()
print("RED  = P01 vanity")
print("CYAN = P02 toilet")

print()
print("IMPORTANT:")
print("Click once inside the figure before pressing N/U/C/S.")
