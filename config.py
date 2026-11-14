from pathlib import Path

DATA_ROOT = Path(
    "/kaggle/input/datasets/haiderrasoolqadri/"
    "dfdc-10k-preprocessed-face-images"
)

PROJECT_DIR = Path("/kaggle/working/dfdc_reproduction")
MANIFEST_PATH = PROJECT_DIR / "dfdc_split_manifest.csv"
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
RESULTS_DIR = PROJECT_DIR / "results"

SEED = 42
IMAGE_SIZE = 299
CLASS_TO_IDX = {"REAL": 0, "FAKE": 1}
NUM_WORKERS = 2

MODEL_CONFIGS = {
    "xception": {
        "batch_size": 32,
        "learning_rate": 1e-4,
        "weight_decay": 0.0,
        "full_epochs": 30,
        "patience": 5,
    },
    "vgg16": {
        "batch_size": 8,
        "learning_rate": 1e-4,
        "weight_decay": 0.0,
        "full_epochs": 20,
        "patience": 5,
    },
    "resnet50": {
        "batch_size": 16,
        "learning_rate": 5e-5,
        "weight_decay": 1e-5,
        "full_epochs": 25,
        "patience": 5,
    },
    "resnet101": {
        "batch_size": 16,
        "learning_rate": 5e-5,
        "weight_decay": 1e-5,
        "full_epochs": 25,
        "patience": 5,
    },
}
