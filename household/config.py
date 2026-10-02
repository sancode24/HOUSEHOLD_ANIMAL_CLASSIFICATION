"""
config.py
Central configuration. Every path and hyperparameter lives here so that all
scripts (prepare_data, train, evaluate, visualise) stay consistent.
"""

from pathlib import Path  # Path gives OS-independent file paths (works on Windows/Linux/Mac)

# ----------------------------------------------------------------------------- Paths
ROOT = Path(__file__).resolve().parent        # folder that contains this file = project root
RAW_DIR = ROOT / "data" / "raw" / "train"     # Kaggle train.zip extracted here (cat.0.jpg, dog.0.jpg, ...)
DATA_DIR = ROOT / "data" / "split"            # prepare_data.py writes train/ val/ test/ sub-folders here
OUT_DIR = ROOT / "outputs"                    # everything the scripts generate goes under outputs/
WEIGHTS_DIR = OUT_DIR / "weights"             # trained model weights (*.weights.h5)
HISTORY_DIR = OUT_DIR / "history"             # per-epoch accuracy/loss logs (JSON + CSV) used for plots
FIG_DIR = OUT_DIR / "figures"                 # all saved plots, activation grids and heatmaps

# ----------------------------------------------------------------------------- Data
IMG_SIZE = (150, 150)                         # every image is resized to 150x150 (originals avg ~350x500)
BATCH_SIZE = 32                               # number of images per gradient-descent step
SEED = 42                                     # fixed seed -> identical split & shuffling on every run
SPLIT = {"train": 0.80, "val": 0.16, "test": 0.04}  # paper: 16 % validation, 4 % test, rest training
CLASS_NAMES = ["cat", "dog"]                  # label 0 = cat, label 1 = dog
CLASS_DIRS = ["cats", "dogs"]                 # folder names inside train/ val/ test/ (same order as above)

# ----------------------------------------------------------------------------- Training
EPOCHS = 30                                   # default epochs; the paper used 100 (pass --epochs 100)
LR = {                                        # learning rate for each model variant
    "vgg6_noreg": 1e-4,                       # VGG-6, no dropout / no augmentation -> demonstrates overfitting
    "vgg6": 1e-4,                             # VGG-6 + dropout + data augmentation (paper's "VGG06_base")
    "vgg16": 2e-5,                            # frozen VGG-16 conv base + new classifier ("VGG16_base")
    "vgg16_ft": 1e-5,                         # fine-tune block5 ("VGG16_fine"): tiny LR so pretrained
}                                             #   features are nudged, not destroyed


def ensure_dirs():
    """Create every output folder if it does not exist yet."""
    for d in (WEIGHTS_DIR, HISTORY_DIR, FIG_DIR):  # loop over the three output folders
        d.mkdir(parents=True, exist_ok=True)       # parents=True makes intermediate dirs; exist_ok avoids errors
