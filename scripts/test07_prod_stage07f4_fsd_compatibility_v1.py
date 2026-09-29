
from pathlib import Path
import sys
import json
import hashlib
import gc
import traceback

import torch


# ============================================================
# PATHS
# ============================================================

BASE = Path(
    "/workspace/axolotl"
)

FSD_DIR = (
    BASE
    / "test07"
    / "models"
    / "FSD"
)

CKPT_PATH = (
    FSD_DIR
    / "ckpt"
    / "FSD_best.ckpt"
)

OUT = (
    BASE
    / "test07"
    / "production_pipeline"
    / "stage07_empty_room"
    / "07f4_fsd_compatibility"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


EXPECTED_SIZE = (
    209_842_750
)

EXPECTED_SHA256 = (
    "a408c3e9f6cb4c6b304c5cfe742346e97f6434c73e9116b78a362a7180cd58f2"
)


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path):

    h = hashlib.sha256()

    with open(
        path,
        "rb"
    ) as f:

        while True:

            chunk = f.read(
                8 * 1024 * 1024
            )

            if not chunk:
                break

            h.update(
                chunk
            )

    return h.hexdigest()


def count_parameters(model):

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return total, trainable


# ============================================================
# HEADER
# ============================================================

print("=" * 110)
print("07F4 — FSD ARCHITECTURE / CHECKPOINT COMPATIBILITY")
print("=" * 110)

print()

print(
    "Python:",
    sys.version.split()[0]
)

print(
    "PyTorch:",
    torch.__version__
)

print(
    "CUDA available:",
    torch.cuda.is_available()
)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# VERIFY CHECKPOINT AGAIN
# ============================================================

if not CKPT_PATH.exists():

    raise FileNotFoundError(
        CKPT_PATH
    )


size = CKPT_PATH.stat().st_size

print()
print(
    "Checkpoint size:",
    size
)


if size != EXPECTED_SIZE:

    raise RuntimeError(
        "Checkpoint size is incorrect."
    )


sha = sha256_file(
    CKPT_PATH
)


print(
    "Checkpoint SHA256:",
    sha
)


if sha != EXPECTED_SHA256:

    raise RuntimeError(
        "Checkpoint SHA256 verification failed."
    )


print(
    "✅ CHECKPOINT VERIFIED"
)


# ============================================================
# ADD FSD REPOSITORY TO PYTHON IMPORT PATH
# ============================================================

fsd_string = str(
    FSD_DIR
)


if fsd_string not in sys.path:

    sys.path.insert(
        0,
        fsd_string
    )


print()
print(
    "FSD repository added to sys.path:"
)

print(
    FSD_DIR
)


# ============================================================
# IMPORT FDRNET
# ============================================================

IMPORT_OK = False
MODEL_BUILD_OK = False
CHECKPOINT_LOAD_OK = False
STRICT_LOAD_OK = False
CUDA_MOVE_OK = False


import_error = None
build_error = None
checkpoint_error = None
strict_error = None
cuda_error = None


try:

    print()
    print(
        "Importing networks.fdrnet.FDRNet..."
    )

    from networks.fdrnet import FDRNet

    IMPORT_OK = True

    print(
        "✅ FDRNet IMPORT SUCCESS"
    )


except Exception as e:

    import_error = repr(
        e
    )

    print()
    print(
        "❌ FDRNet IMPORT FAILED"
    )

    traceback.print_exc()


# ============================================================
# CONSTRUCT MODEL
#
# Exact demo architecture, except:
#
# use_pretrained=False
#
# We intentionally disable downloading EfficientNet weights.
# The FSD checkpoint should contain learned backbone weights.
# ============================================================

model = None


if IMPORT_OK:

    try:

        print()
        print("=" * 110)
        print("MODEL CONSTRUCTION")
        print("=" * 110)

        model = FDRNet(
            backbone="efficientnet-b3",
            proj_planes=16,
            pred_planes=32,

            # IMPORTANT:
            # prevent external backbone download
            use_pretrained=False,

            fix_backbone=False,
            has_se=False,
            dropout_2d=0,
            normalize=True,
            mu_init=0.5,
            reweight_mode="manual",
        )

        MODEL_BUILD_OK = True

        total_params, trainable_params = (
            count_parameters(
                model
            )
        )

        print()
        print(
            "✅ MODEL CONSTRUCTION SUCCESS"
        )

        print(
            "Total parameters:",
            f"{total_params:,}"
        )

        print(
            "Trainable parameters:",
            f"{trainable_params:,}"
        )


    except Exception as e:

        build_error = repr(
            e
        )

        print()
        print(
            "❌ MODEL CONSTRUCTION FAILED"
        )

        traceback.print_exc()


# ============================================================
# LOAD CHECKPOINT
#
# First try PyTorch 2.6 safe weights-only mode.
# If this checkpoint format requires normal pickle loading,
# fall back explicitly because this is the official checkpoint
# whose SHA256 we verified above.
# ============================================================

ckpt = None
checkpoint_load_method = None


if MODEL_BUILD_OK:

    print()
    print("=" * 110)
    print("CHECKPOINT DESERIALIZATION")
    print("=" * 110)

    try:

        ckpt = torch.load(
            CKPT_PATH,
            map_location="cpu",
            weights_only=True,
        )

        checkpoint_load_method = (
            "weights_only=True"
        )

        CHECKPOINT_LOAD_OK = True

        print()
        print(
            "✅ CHECKPOINT DESERIALIZED"
        )

        print(
            "Method:",
            checkpoint_load_method
        )


    except Exception as safe_error:

        print()
        print(
            "weights_only=True could not load checkpoint."
        )

        print(
            "Trying verified official checkpoint with "
            "weights_only=False..."
        )

        try:

            ckpt = torch.load(
                CKPT_PATH,
                map_location="cpu",
                weights_only=False,
            )

            checkpoint_load_method = (
                "weights_only=False"
            )

            CHECKPOINT_LOAD_OK = True

            print()
            print(
                "✅ CHECKPOINT DESERIALIZED"
            )

            print(
                "Method:",
                checkpoint_load_method
            )


        except Exception as e:

            checkpoint_error = repr(
                e
            )

            print()
            print(
                "❌ CHECKPOINT DESERIALIZATION FAILED"
            )

            traceback.print_exc()


# ============================================================
# INSPECT CHECKPOINT STRUCTURE
# ============================================================

state_dict = None


if CHECKPOINT_LOAD_OK:

    print()
    print("=" * 110)
    print("CHECKPOINT STRUCTURE")
    print("=" * 110)

    print()
    print(
        "Checkpoint Python type:",
        type(
            ckpt
        )
    )


    if isinstance(
        ckpt,
        dict
    ):

        print(
            "Top-level keys:"
        )

        for key in ckpt.keys():

            print(
                " -",
                key
            )


        if "model" in ckpt:

            state_dict = ckpt[
                "model"
            ]

            print()
            print(
                "✅ Found ckpt['model']"
            )


        elif "state_dict" in ckpt:

            state_dict = ckpt[
                "state_dict"
            ]

            print()
            print(
                "Using ckpt['state_dict']"
            )


        else:

            tensor_like = all(
                torch.is_tensor(v)
                for v in ckpt.values()
            )

            if tensor_like:

                state_dict = ckpt

                print()
                print(
                    "Checkpoint itself appears to be state_dict."
                )


    if state_dict is None:

        raise RuntimeError(
            "Could not locate model state_dict "
            "inside checkpoint."
        )


    print(
        "State-dict entries:",
        len(
            state_dict
        )
    )


    print()
    print(
        "First 12 state-dict keys:"
    )


    for key in list(
        state_dict.keys()
    )[:12]:

        value = state_dict[
            key
        ]

        shape = (
            tuple(
                value.shape
            )
            if torch.is_tensor(
                value
            )
            else type(
                value
            )
        )

        print(
            " ",
            key,
            shape
        )


# ============================================================
# STRICT MODEL LOAD
# ============================================================

missing_keys = []
unexpected_keys = []


if (
    CHECKPOINT_LOAD_OK
    and
    state_dict is not None
):

    print()
    print("=" * 110)
    print("STRICT STATE-DICT LOAD")
    print("=" * 110)


    try:

        result = model.load_state_dict(
            state_dict,
            strict=True
        )

        STRICT_LOAD_OK = True

        print()
        print(
            "✅ STRICT STATE-DICT LOAD SUCCESS"
        )


        missing_keys = list(
            getattr(
                result,
                "missing_keys",
                []
            )
        )

        unexpected_keys = list(
            getattr(
                result,
                "unexpected_keys",
                []
            )
        )


    except Exception as e:

        strict_error = repr(
            e
        )

        print()
        print(
            "❌ STRICT STATE-DICT LOAD FAILED"
        )

        print()
        print(
            "Running non-strict diagnostic only..."
        )


        try:

            result = model.load_state_dict(
                state_dict,
                strict=False
            )

            missing_keys = list(
                result.missing_keys
            )

            unexpected_keys = list(
                result.unexpected_keys
            )


            print()
            print(
                "MISSING KEYS:",
                len(
                    missing_keys
                )
            )


            for key in missing_keys[:30]:

                print(
                    "  MISSING:",
                    key
                )


            print()
            print(
                "UNEXPECTED KEYS:",
                len(
                    unexpected_keys
                )
            )


            for key in unexpected_keys[:30]:

                print(
                    "  UNEXPECTED:",
                    key
                )


        except Exception:

            traceback.print_exc()


# ============================================================
# CUDA MOVE TEST
#
# Still NO inference.
# ============================================================

if STRICT_LOAD_OK:

    print()
    print("=" * 110)
    print("CUDA MODEL MOVE TEST")
    print("=" * 110)


    if torch.cuda.is_available():

        try:

            model = model.cuda()

            model.eval()

            CUDA_MOVE_OK = True


            allocated = (
                torch.cuda.memory_allocated()
                /
                1024**3
            )


            reserved = (
                torch.cuda.memory_reserved()
                /
                1024**3
            )


            print()
            print(
                "✅ MODEL MOVED TO CUDA"
            )

            print(
                "Allocated VRAM GB:",
                round(
                    allocated,
                    3
                )
            )

            print(
                "Reserved VRAM GB:",
                round(
                    reserved,
                    3
                )
            )


        except Exception as e:

            cuda_error = repr(
                e
            )

            print()
            print(
                "❌ CUDA MOVE FAILED"
            )

            traceback.print_exc()


    else:

        print(
            "CUDA unavailable; skipped."
        )


# ============================================================
# RESULT
# ============================================================

overall_compatible = bool(
    IMPORT_OK
    and
    MODEL_BUILD_OK
    and
    CHECKPOINT_LOAD_OK
    and
    STRICT_LOAD_OK
)


print()
print("=" * 110)
print("07F4 RESULT")
print("=" * 110)

print()

print(
    "FDRNet import:",
    "✅ PASS"
    if IMPORT_OK
    else "❌ FAIL"
)

print(
    "Model construction:",
    "✅ PASS"
    if MODEL_BUILD_OK
    else "❌ FAIL"
)

print(
    "Checkpoint deserialize:",
    "✅ PASS"
    if CHECKPOINT_LOAD_OK
    else "❌ FAIL"
)

print(
    "Strict state-dict:",
    "✅ PASS"
    if STRICT_LOAD_OK
    else "❌ FAIL"
)

print(
    "CUDA move:",
    (
        "✅ PASS"
        if CUDA_MOVE_OK
        else "❌/SKIPPED"
    )
)


print()

if overall_compatible:

    print(
        "✅ FSD IS ARCHITECTURALLY COMPATIBLE "
        "WITH CURRENT ENVIRONMENT"
    )

    print()
    print(
        "NO NEW PYTHON ENVIRONMENT REQUIRED SO FAR."
    )

else:

    print(
        "⚠️ FSD CURRENT-ENVIRONMENT COMPATIBILITY "
        "NOT YET CONFIRMED."
    )


print()
print(
    "NO IMAGE INFERENCE WAS PERFORMED."
)


# ============================================================
# SAVE RESULT
# ============================================================

STATE = {

    "stage":
        "07F4",

    "purpose":
        "FSD_CURRENT_ENVIRONMENT_COMPATIBILITY",

    "python":
        sys.version.split()[0],

    "torch":
        torch.__version__,

    "checkpoint": {

        "path":
            str(
                CKPT_PATH
            ),

        "size":
            size,

        "sha256":
            sha,

        "load_method":
            checkpoint_load_method
    },

    "results": {

        "import_ok":
            IMPORT_OK,

        "model_build_ok":
            MODEL_BUILD_OK,

        "checkpoint_load_ok":
            CHECKPOINT_LOAD_OK,

        "strict_load_ok":
            STRICT_LOAD_OK,

        "cuda_move_ok":
            CUDA_MOVE_OK,

        "overall_compatible":
            overall_compatible,

        "missing_keys":
            missing_keys,

        "unexpected_keys":
            unexpected_keys,
    },

    "errors": {

        "import":
            import_error,

        "build":
            build_error,

        "checkpoint":
            checkpoint_error,

        "strict":
            strict_error,

        "cuda":
            cuda_error,
    },

    "next_if_pass":
        "07F5_SINGLE_STAGE01_FSD_INFERENCE",

    "status":
        (
            "PASS"
            if overall_compatible
            else "REQUIRES_COMPATIBILITY_FIX"
        )
}


STATE_PATH = (
    OUT
    / "00_stage07f4_result.json"
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
print(
    "STATE:",
    STATE_PATH
)


# ============================================================
# CLEANUP
# ============================================================

try:

    del state_dict

except Exception:

    pass


try:

    del ckpt

except Exception:

    pass


try:

    del model

except Exception:

    pass


gc.collect()


if torch.cuda.is_available():

    torch.cuda.empty_cache()

