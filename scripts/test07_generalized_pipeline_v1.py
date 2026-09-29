
from pathlib import Path
import json
import shutil
import argparse


# ============================================================
# TEST07 GENERALIZED PIPELINE V1
#
# PURPOSE:
#   Standardize folder structure and stage handoff for
#   cross-room generalization testing.
#
# IMPORTANT:
#   This runner does NOT yet execute every model stage.
#   First version validates inputs, builds workspace,
#   records the frozen stage order, and prepares dynamic paths.
#
# NO TEST06 FIXED PIXEL COUNTS.
# NO FIXED ROOM TYPE.
# NO FIXED OBJECT LIST.
# ============================================================


STAGES = [
    ("01", "structure_inverse_base"),
    ("02", "dynamic_inventory"),
    ("03", "inventory_grounding"),
    ("04", "candidate_verification"),
    ("05", "generalized_candidate_filter"),
    ("06", "tight_box_verification"),
    ("07", "sam2_safe_recovery"),
    ("08", "aq_seeded_local_recovery"),
    ("09", "sam2_multimask_selection"),
    ("10", "generalized_component_filter"),
    ("11", "fresh_crop_object_segmentation"),
    ("12", "proximity_fragment_discovery"),
    ("13", "strict_fragment_merge"),
    ("14", "small_object_discovery"),
    ("15", "verified_small_object_merge"),
    ("16", "residual_audit"),
]


def prepare_room(
    input_image,
    room_id,
    test07_root
):

    input_image = Path(input_image)
    test07_root = Path(test07_root)

    if not input_image.exists():
        raise FileNotFoundError(
            input_image
        )

    room_root = (
        test07_root
        / "runs"
        / room_id
    )

    input_dir = (
        room_root
        / "input"
    )

    stages_dir = (
        room_root
        / "stages"
    )

    reports_dir = (
        room_root
        / "reports"
    )

    final_dir = (
        room_root
        / "final"
    )


    for folder in [
        room_root,
        input_dir,
        stages_dir,
        reports_dir,
        final_dir,
    ]:

        folder.mkdir(
            parents=True,
            exist_ok=True
        )


    # ========================================================
    # COPY INPUT
    # ========================================================

    suffix = (
        input_image.suffix.lower()
        if input_image.suffix
        else ".png"
    )

    copied_input = (
        input_dir
        / f"original{suffix}"
    )


    if (
        input_image.resolve()
        != copied_input.resolve()
    ):

        shutil.copy2(
            input_image,
            copied_input
        )


    # ========================================================
    # CREATE STAGE DIRECTORIES
    # ========================================================

    stage_paths = {}


    for stage_id, stage_name in STAGES:

        path = (
            stages_dir
            / f"{stage_id}_{stage_name}"
        )

        path.mkdir(
            parents=True,
            exist_ok=True
        )

        stage_paths[
            stage_id
        ] = str(
            path
        )


    # ========================================================
    # RUN MANIFEST
    # ========================================================

    manifest = {
        "pipeline":
            "TEST07_GENERALIZED_PIPELINE_V1",

        "room_id":
            room_id,

        "input_original":
            str(
                input_image
            ),

        "input_copied":
            str(
                copied_input
            ),

        "room_root":
            str(
                room_root
            ),

        "stage_order": [
            {
                "stage_id":
                    stage_id,

                "name":
                    stage_name,

                "output_dir":
                    stage_paths[
                        stage_id
                    ]
            }
            for stage_id, stage_name
            in STAGES
        ],

        "rules": [
            "same V1 stage order for every room",
            "no TEST06 hardcoded pixel counts",
            "no fixed room type",
            "no fixed object inventory",
            "no room-specific coordinates",
            "no per-room threshold changes during benchmark",
            "all integrity checks are dynamic",
            "original RGB must remain unchanged in extracted props",
        ]
    }


    manifest_path = (
        room_root
        / "run_manifest.json"
    )


    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2
        )
    )


    # ========================================================
    # STATUS FILE
    # ========================================================

    status = {
        "room_id":
            room_id,

        "current_stage":
            "NOT_STARTED",

        "completed_stages":
            [],

        "failed_stage":
            None,

        "notes":
            []
    }


    status_path = (
        room_root
        / "status.json"
    )


    status_path.write_text(
        json.dumps(
            status,
            indent=2
        )
    )


    print(
        "=" * 90
    )

    print(
        "TEST07 ROOM WORKSPACE READY"
    )

    print(
        "=" * 90
    )

    print(
        "ROOM ID:",
        room_id
    )

    print(
        "INPUT:",
        copied_input
    )

    print(
        "ROOM ROOT:",
        room_root
    )

    print(
        "STAGES:",
        len(
            STAGES
        )
    )

    print()
    print(
        "MANIFEST:"
    )

    print(
        manifest_path
    )

    print()
    print(
        "STATUS:"
    )

    print(
        status_path
    )

    return manifest


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True
    )

    parser.add_argument(
        "--room-id",
        required=True
    )

    parser.add_argument(
        "--test07-root",
        default=
            "/workspace/axolotl/test07"
    )

    args = parser.parse_args()


    prepare_room(
        input_image=
            args.input,

        room_id=
            args.room_id,

        test07_root=
            args.test07_root
    )
