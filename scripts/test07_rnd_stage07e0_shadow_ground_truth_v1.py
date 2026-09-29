
from pathlib import Path
import json

import numpy as np
from PIL import Image, ImageDraw

import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector

import ipywidgets as widgets
from IPython.display import display


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

image_np = np.asarray(
    image
)

W, H = image.size


floor_mask = (
    np.asarray(
        Image.open(
            FLOOR_MASK_PATH
        ).convert("L")
    ) > 127
)


# ============================================================
# ANNOTATION STATE
# ============================================================

current_object = "P01"

current_vertices = []

polygons = {
    "P01": [],
    "P02": [],
}


# ============================================================
# FIGURE
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 10)
)

ax.imshow(
    image_np
)

ax.set_xlim(
    0,
    W
)

ax.set_ylim(
    H,
    0
)

ax.set_title(
    "P01 VANITY SHADOW\n"
    "Draw polygon around actual FLOOR shadow only"
)

ax.axis(
    "off"
)


# ============================================================
# DISPLAYED POLYGON PATCHES
# ============================================================

polygon_lines = []


def redraw_saved_polygons():

    global polygon_lines

    for line in polygon_lines:
        try:
            line.remove()
        except Exception:
            pass

    polygon_lines = []


    styles = {
        "P01": ("red", "P01"),
        "P02": ("cyan", "P02"),
    }


    for obj in ["P01", "P02"]:

        color, label = styles[obj]

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

            polygon_lines.append(
                line
            )


    fig.canvas.draw_idle()


# ============================================================
# POLYGON SELECT CALLBACK
# ============================================================

def on_polygon_selected(verts):

    global current_vertices

    current_vertices = [
        [
            float(x),
            float(y)
        ]
        for x, y in verts
    ]

    status.value = (
        f"<b>Current polygon:</b> "
        f"{len(current_vertices)} vertices. "
        f"Click <b>Add Polygon</b> to accept it."
    )


selector = PolygonSelector(
    ax,
    on_polygon_selected,
    useblit=True
)


# ============================================================
# BUTTONS
# ============================================================

add_button = widgets.Button(
    description="Add Polygon",
    button_style="success"
)

undo_button = widgets.Button(
    description="Undo Last",
    button_style="warning"
)

reset_current_button = widgets.Button(
    description="Clear Current Object"
)

next_button = widgets.Button(
    description="Finish P01 → P02",
    button_style="info"
)

save_button = widgets.Button(
    description="Save Ground Truth",
    button_style="success",
    disabled=True
)


status = widgets.HTML(
    value=(
        "<b>Current object: P01 VANITY.</b> "
        "Draw the true vanity shadow on the floor."
    )
)


# ============================================================
# ADD POLYGON
# ============================================================

def add_polygon(_):

    global current_vertices

    if len(current_vertices) < 3:

        status.value = (
            "<b>Polygon not added:</b> "
            "draw at least 3 vertices first."
        )

        return


    polygons[
        current_object
    ].append(
        current_vertices.copy()
    )


    n = len(
        polygons[current_object]
    )


    status.value = (
        f"<b>{current_object}:</b> "
        f"polygon #{n} added."
    )


    current_vertices = []


    try:
        selector.clear()
    except Exception:
        pass


    redraw_saved_polygons()


# ============================================================
# UNDO
# ============================================================

def undo_last(_):

    if len(
        polygons[current_object]
    ) == 0:

        status.value = (
            f"<b>{current_object}:</b> "
            "no saved polygon to undo."
        )

        return


    polygons[
        current_object
    ].pop()


    redraw_saved_polygons()


    status.value = (
        f"<b>{current_object}:</b> "
        "last polygon removed."
    )


# ============================================================
# CLEAR CURRENT OBJECT
# ============================================================

def clear_current(_):

    global current_vertices

    polygons[
        current_object
    ] = []

    current_vertices = []


    try:
        selector.clear()
    except Exception:
        pass


    redraw_saved_polygons()


    status.value = (
        f"<b>{current_object} cleared.</b> "
        "Draw again."
    )


# ============================================================
# SWITCH P01 -> P02
# ============================================================

def finish_p01(_):

    global current_object
    global current_vertices


    if current_object != "P01":
        return


    if len(
        polygons["P01"]
    ) == 0:

        status.value = (
            "<b>P01 has no polygon yet.</b> "
            "Annotate the vanity shadow first."
        )

        return


    current_object = "P02"

    current_vertices = []


    try:
        selector.clear()
    except Exception:
        pass


    ax.set_title(
        "P02 TOILET SHADOW\n"
        "Draw polygon around actual FLOOR shadow only"
    )


    next_button.disabled = True

    save_button.disabled = False


    status.value = (
        "<b>Current object: P02 TOILET.</b> "
        "Draw the true toilet shadow on the floor."
    )


    fig.canvas.draw_idle()


# ============================================================
# POLYGONS -> MASK
# ============================================================

def polygons_to_mask(
    polygon_list
):

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


    for poly in polygon_list:

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
        np.asarray(
            mask_img
        ) > 127
    )


    # Ground truth is restricted to actual floor only.
    mask &= floor_mask


    return mask


# ============================================================
# SAVE
# ============================================================

def save_ground_truth(_):

    global current_vertices


    if current_object != "P02":

        status.value = (
            "<b>Finish P01 first.</b>"
        )

        return


    if len(
        polygons["P02"]
    ) == 0:

        status.value = (
            "<b>P02 has no saved polygon.</b> "
            "Draw the toilet shadow and click Add Polygon."
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
    # SAVE BINARY MASKS
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


    # P01 = red
    overlay[
        p01_mask
    ] = (
        overlay[
            p01_mask
        ] * 0.45
        +
        np.array(
            [255, 0, 0],
            dtype=np.float32
        ) * 0.55
    )


    # P02 = cyan
    overlay[
        p02_mask
    ] = (
        overlay[
            p02_mask
        ] * 0.45
        +
        np.array(
            [0, 255, 255],
            dtype=np.float32
        ) * 0.55
    )


    overlay = np.clip(
        overlay,
        0,
        255
    ).astype(
        np.uint8
    )


    OVERLAY_PATH = (
        OUT
        / "04_ground_truth_overlay.png"
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

        "production_runtime_component":
            False,

        "annotation_source":
            str(
                IMAGE_PATH
            ),

        "floor_mask":
            str(
                FLOOR_MASK_PATH
            ),

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

        "polygons": polygons,

        "outputs": {

            "p01":
                str(
                    P01_PATH
                ),

            "p02":
                str(
                    P02_PATH
                ),

            "combined":
                str(
                    COMBINED_PATH
                ),

            "overlay":
                str(
                    OVERLAY_PATH
                )
        },

        "status":
            "MANUAL_GROUND_TRUTH_CREATED"
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


    status.value = (
        "<b>✅ Ground truth saved successfully.</b><br>"
        f"P01 pixels: {int(p01_mask.sum())}<br>"
        f"P02 pixels: {int(p02_mask.sum())}<br>"
        f"Combined: {int(combined.sum())}<br>"
        f"Folder: {OUT}"
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
# CONNECT BUTTONS
# ============================================================

add_button.on_click(
    add_polygon
)

undo_button.on_click(
    undo_last
)

reset_current_button.on_click(
    clear_current
)

next_button.on_click(
    finish_p01
)

save_button.on_click(
    save_ground_truth
)


# ============================================================
# UI
# ============================================================

controls = widgets.HBox(
    [
        add_button,
        undo_button,
        reset_current_button,
        next_button,
        save_button,
    ]
)


display(
    status
)

display(
    controls
)


print()
print("=" * 100)
print("07E0 ANNOTATION ACTIVE")
print("=" * 100)

print()
print("1. Draw P01 vanity floor-shadow polygon.")
print("2. Click Add Polygon.")
print("3. Add additional P01 polygons if required.")
print("4. Click Finish P01 → P02.")
print("5. Draw P02 toilet shadow.")
print("6. Click Add Polygon.")
print("7. Click Save Ground Truth.")
print()
print("RED outlines = saved P01 polygons")
print("CYAN outlines = saved P02 polygons")
