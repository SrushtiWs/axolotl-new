
from pathlib import Path
import argparse
import json

import torch
from PIL import Image, ImageDraw

from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
)


MODEL_ID = "IDEA-Research/grounding-dino-base"

UPSCALE = 4

BOX_THRESHOLD = 0.15
TEXT_THRESHOLD = 0.12

TOP_K_PER_VIEW = 5

SUB_OVERLAP = 0.20


def run_dino(
    image,
    phrase,
    model,
    processor,
    device,
):

    W, H = image.size

    text = phrase.strip().lower()

    if not text.endswith("."):
        text += "."

    inputs = processor(
        images=image,
        text=text,
        return_tensors="pt"
    ).to(device)

    with torch.inference_mode():

        outputs = model(
            **inputs
        )

    result = (
        processor
        .post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            target_sizes=[
                (H, W)
            ]
        )[0]
    )

    boxes = result.get(
        "boxes",
        []
    )

    scores = result.get(
        "scores",
        []
    )

    labels = result.get(
        "text_labels",
        result.get(
            "labels",
            []
        )
    )

    if torch.is_tensor(boxes):
        boxes = boxes.detach().cpu().numpy()

    if torch.is_tensor(scores):
        scores = scores.detach().cpu().numpy()

    rows = []

    for i in range(
        len(boxes)
    ):

        rows.append({
            "bbox":
                [
                    float(v)
                    for v in boxes[i]
                ],

            "score":
                float(
                    scores[i]
                ),

            "label":
                (
                    str(labels[i])
                    if i < len(labels)
                    else ""
                )
        })

    rows.sort(
        key=lambda r:
            r["score"],
        reverse=True
    )

    return rows[
        :TOP_K_PER_VIEW
    ]


def make_subviews(
    image
):

    W, H = image.size

    half_w = W / 2
    half_h = H / 2

    mx = int(
        half_w
        *
        SUB_OVERLAP
    )

    my = int(
        half_h
        *
        SUB_OVERLAP
    )

    definitions = [
        ("tl", 0, 0),
        ("tr", 1, 0),
        ("bl", 0, 1),
        ("br", 1, 1),
    ]

    views = []

    for name, col, row in definitions:

        x1 = max(
            0,
            int(
                col * half_w
            ) - mx
        )

        y1 = max(
            0,
            int(
                row * half_h
            ) - my
        )

        x2 = min(
            W,
            int(
                (col + 1)
                *
                half_w
            ) + mx
        )

        y2 = min(
            H,
            int(
                (row + 1)
                *
                half_h
            ) + my
        )

        views.append({
            "name":
                name,

            "bbox":
                [
                    x1,
                    y1,
                    x2,
                    y2
                ],

            "image":
                image.crop(
                    (
                        x1,
                        y1,
                        x2,
                        y2
                    )
                )
        })

    return views


def run(
    master_path,
    inventory_json,
    existence_json,
    tiled_candidates_json,
    tile_dir,
    output_dir,
    model_cache,
):

    master_path = Path(
        master_path
    )

    inventory_json = Path(
        inventory_json
    )

    existence_json = Path(
        existence_json
    )

    tiled_candidates_json = Path(
        tiled_candidates_json
    )

    tile_dir = Path(
        tile_dir
    )

    output_dir = Path(
        output_dir
    )

    model_cache = Path(
        model_cache
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    master = (
        Image.open(
            master_path
        )
        .convert("RGB")
    )

    inventory = json.loads(
        inventory_json.read_text()
    )["objects"]

    existence = json.loads(
        existence_json.read_text()
    )

    tiled = json.loads(
        tiled_candidates_json.read_text()
    )

    inventory_by_id = {
        int(row["id"]):
            row
        for row in inventory
    }

    present_ids = {
        int(row["inventory_id"])
        for row in existence
        if row["decision"] == "PRESENT"
    }

    region_bbox = {}

    for row in tiled:

        region = row.get(
            "source_region"
        )

        bbox = row.get(
            "source_bbox"
        )

        if region and bbox:
            region_bbox[
                region
            ] = bbox

    print("=" * 90)
    print("TEST07 STAGE04.8")
    print("HIGH-RES LOCAL PYRAMID DINO RESCUE")
    print("=" * 90)

    print(
        "PRESENT IDS:",
        sorted(
            present_ids
        )
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print(
        "Loading Grounding DINO..."
    )

    processor = (
        AutoProcessor
        .from_pretrained(
            MODEL_ID,
            cache_dir=str(
                model_cache
            )
        )
    )

    model = (
        AutoModelForZeroShotObjectDetection
        .from_pretrained(
            MODEL_ID,
            cache_dir=str(
                model_cache
            )
        )
        .to(device)
    )

    model.eval()

    print(
        "✅ DINO READY"
    )

    all_rows = []

    for iid in sorted(
        present_ids
    ):

        obj = inventory_by_id[
            iid
        ]

        regions = obj.get(
            "source_regions",
            []
        )

        if isinstance(
            regions,
            str
        ):
            regions = [
                regions
            ]

        print()
        print("=" * 80)

        print(
            f'{iid:02d}. '
            f'{obj["name"]} | '
            f'{obj["grounding_phrase"]}'
        )

        for region in regions:

            tile_path = (
                tile_dir
                /
                f"{region}.png"
            )

            if not tile_path.exists():
                continue

            if region not in region_bbox:
                continue

            tile = (
                Image.open(
                    tile_path
                )
                .convert("RGB")
            )

            original_tw, original_th = (
                tile.size
            )

            upscaled = tile.resize(
                (
                    original_tw
                    *
                    UPSCALE,

                    original_th
                    *
                    UPSCALE
                ),
                Image.Resampling.LANCZOS
            )

            views = [
                {
                    "name":
                        "full",

                    "bbox":
                        [
                            0,
                            0,
                            upscaled.width,
                            upscaled.height
                        ],

                    "image":
                        upscaled
                }
            ]

            views.extend(
                make_subviews(
                    upscaled
                )
            )

            master_x1, master_y1, _, _ = (
                region_bbox[
                    region
                ]
            )

            for view in views:

                view_x1, view_y1, _, _ = (
                    view["bbox"]
                )

                # Two textual prompts:
                # specific phrase + generic object name.
                phrases = []

                for phrase in [
                    obj[
                        "grounding_phrase"
                    ],
                    obj[
                        "name"
                    ]
                ]:

                    if (
                        phrase
                        and
                        phrase not in phrases
                    ):
                        phrases.append(
                            phrase
                        )

                for phrase in phrases:

                    rows = run_dino(
                        view[
                            "image"
                        ],
                        phrase,
                        model,
                        processor,
                        device
                    )

                    for row in rows:

                        ux1, uy1, ux2, uy2 = (
                            row[
                                "bbox"
                            ]
                        )

                        # View coords
                        # -> upscaled tile coords.
                        ux1 += view_x1
                        ux2 += view_x1
                        uy1 += view_y1
                        uy2 += view_y1

                        # Upscaled tile coords
                        # -> original tile coords.
                        tx1 = ux1 / UPSCALE
                        ty1 = uy1 / UPSCALE
                        tx2 = ux2 / UPSCALE
                        ty2 = uy2 / UPSCALE

                        # Tile coords -> master.
                        global_box = [
                            tx1 + master_x1,
                            ty1 + master_y1,
                            tx2 + master_x1,
                            ty2 + master_y1,
                        ]

                        all_rows.append({
                            "inventory_id":
                                iid,

                            "inventory_name":
                                obj[
                                    "name"
                                ],

                            "grounding_phrase":
                                obj[
                                    "grounding_phrase"
                                ],

                            "dino_query":
                                phrase,

                            "source_region":
                                region,

                            "pyramid_view":
                                view[
                                    "name"
                                ],

                            "bbox":
                                global_box,

                            "score":
                                row[
                                    "score"
                                ],

                            "dino_label":
                                row[
                                    "label"
                                ],

                            "rescue_source":
                                "stage04_8_highres_local_pyramid",
                        })

        item_rows = [
            row
            for row in all_rows
            if row[
                "inventory_id"
            ] == iid
        ]

        item_rows.sort(
            key=lambda r:
                r[
                    "score"
                ],
            reverse=True
        )

        print(
            "Candidates:",
            len(
                item_rows
            )
        )

        if item_rows:

            print(
                "Best:",
                round(
                    item_rows[0][
                        "score"
                    ],
                    4
                ),
                "|",
                [
                    round(
                        v,
                        1
                    )
                    for v in item_rows[
                        0
                    ][
                        "bbox"
                    ]
                ],
                "| view=",
                item_rows[
                    0
                ][
                    "pyramid_view"
                ],
                "| query=",
                item_rows[
                    0
                ][
                    "dino_query"
                ]
            )

    # ========================================================
    # KEEP TOP UNIQUE BOXES PER INSTANCE
    # ========================================================

    reduced = []

    for iid in sorted(
        present_ids
    ):

        rows = [
            r
            for r in all_rows
            if r[
                "inventory_id"
            ] == iid
        ]

        rows.sort(
            key=lambda r:
                r[
                    "score"
                ],
            reverse=True
        )

        unique = []

        for row in rows:

            box = row[
                "bbox"
            ]

            duplicate = False

            for existing in unique:

                eb = existing[
                    "bbox"
                ]

                delta = sum(
                    abs(
                        box[k]
                        -
                        eb[k]
                    )
                    for k in range(4)
                )

                if delta < 3.0:
                    duplicate = True
                    break

            if duplicate:
                continue

            unique.append(
                row
            )

            if len(
                unique
            ) >= 8:
                break

        reduced.extend(
            unique
        )

    candidate_path = (
        output_dir
        /
        "00_highres_pyramid_candidates.json"
    )

    candidate_path.write_text(
        json.dumps(
            reduced,
            indent=2
        )
    )

    preview = master.copy()

    draw = ImageDraw.Draw(
        preview
    )

    for row in reduced:

        x1, y1, x2, y2 = (
            row[
                "bbox"
            ]
        )

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

        draw.text(
            (
                x1,
                max(
                    0,
                    y1 - 12
                )
            ),
            (
                f'{row["inventory_id"]}:'
                f'{row["inventory_name"]}'
                f' {row["score"]:.2f}'
            ),
            fill="red"
        )

    preview_path = (
        output_dir
        /
        "01_highres_pyramid_preview.png"
    )

    preview.save(
        preview_path
    )

    report = {
        "stage":
            "TEST07_STAGE048",

        "present_ids":
            sorted(
                present_ids
            ),

        "upscale":
            UPSCALE,

        "total_raw_candidates":
            len(
                all_rows
            ),

        "reduced_candidates":
            len(
                reduced
            ),

        "candidate_json":
            str(
                candidate_path
            ),

        "preview":
            str(
                preview_path
            ),
    }

    (
        output_dir
        /
        "stage048_report.json"
    ).write_text(
        json.dumps(
            report,
            indent=2
        )
    )

    print()
    print("=" * 90)
    print("TEST07 STAGE04.8 RESULT")
    print("=" * 90)

    print(
        "RAW CANDIDATES:",
        len(
            all_rows
        )
    )

    print(
        "REDUCED CANDIDATES:",
        len(
            reduced
        )
    )

    print()
    print("PER INSTANCE:")

    for iid in sorted(
        present_ids
    ):

        rows = [
            r
            for r in reduced
            if r[
                "inventory_id"
            ] == iid
        ]

        print(
            f'{iid:02d}. '
            f'{inventory_by_id[iid]["name"]} '
            f'→ {len(rows)} candidate(s)'
            f' | best='
            f'{max([r["score"] for r in rows], default=None)}'
        )

    print()
    print(
        "PREVIEW:"
    )

    print(
        preview_path
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--master",
        required=True
    )

    parser.add_argument(
        "--inventory-json",
        required=True
    )

    parser.add_argument(
        "--existence-json",
        required=True
    )

    parser.add_argument(
        "--tiled-candidates-json",
        required=True
    )

    parser.add_argument(
        "--tile-dir",
        required=True
    )

    parser.add_argument(
        "--output-dir",
        required=True
    )

    parser.add_argument(
        "--model-cache",
        default=
            "/workspace/data/huggingface-cache"
    )

    args = parser.parse_args()

    run(
        master_path=
            args.master,

        inventory_json=
            args.inventory_json,

        existence_json=
            args.existence_json,

        tiled_candidates_json=
            args.tiled_candidates_json,

        tile_dir=
            args.tile_dir,

        output_dir=
            args.output_dir,

        model_cache=
            args.model_cache,
    )
