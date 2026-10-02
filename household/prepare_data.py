"""
prepare_data.py  (STEP 1)
Validates the Kaggle Dogs-vs-Cats images (drops corrupted files) and splits them
80 / 16 / 4 into train / val / test folders laid out as:

    data/split/train/cats/*.jpg   data/split/train/dogs/*.jpg
    data/split/val/cats/*.jpg     data/split/val/dogs/*.jpg
    data/split/test/cats/*.jpg    data/split/test/dogs/*.jpg

Usage:
    python prepare_data.py                     # full 25,000 images
    python prepare_data.py --max-per-class 2000  # small subset for quick CPU runs
"""

import argparse                # parses command-line flags like --max-per-class
import random                  # seeded shuffling so the split is reproducible
import shutil                  # copying files / deleting old split folders

from PIL import Image          # Pillow: used only to check that each image actually decodes

import config                  # our central paths & hyperparameters


def is_valid(path):
    """Return True if the file opens AND fully decodes as an image, else False."""
    try:                                   # any decoding error is caught below
        with Image.open(path) as im:       # opens the file header (cheap)
            im.load()                      # forces a full pixel decode; truncated/corrupt files raise here
        return True                        # decoded fine -> keep it
    except Exception:                      # PIL raises various error types for bad files
        return False                       # corrupted -> drop it


def main():
    parser = argparse.ArgumentParser(description="Validate and split Dogs-vs-Cats images.")   # CLI parser
    parser.add_argument("--max-per-class", type=int, default=None,                         # optional cap
                        help="use only N images per class (for quick experiments)")
    args = parser.parse_args()                                                             # read the flags

    if not config.RAW_DIR.exists():                                    # raw images missing -> stop early
        raise SystemExit(f"Raw images not found at {config.RAW_DIR}. See README (Dataset section).")

    if config.DATA_DIR.exists():                                       # an older split exists
        shutil.rmtree(config.DATA_DIR)                                 # delete it so splits never mix

    rng = random.Random(config.SEED)                                   # private RNG seeded for reproducibility

    for cls, cls_dir in zip(config.CLASS_NAMES, config.CLASS_DIRS):    # ("cat","cats"), ("dog","dogs")
        files = sorted(config.RAW_DIR.glob(f"{cls}.*.jpg"))            # Kaggle names files cat.123.jpg / dog.45.jpg
        good = [f for f in files if is_valid(f)]                       # keep only images that decode
        print(f"{cls}: {len(files)} files, {len(files) - len(good)} corrupted removed")

        rng.shuffle(good)                                              # random order before splitting
        if args.max_per_class:                                         # optional subset for fast runs
            good = good[: args.max_per_class]                          # take the first N after shuffling

        n = len(good)                                                  # images available for this class
        n_train = int(n * config.SPLIT["train"])                       # 80 % -> training
        n_val = int(n * config.SPLIT["val"])                           # 16 % -> validation
        parts = {                                                      # slice the shuffled list into 3 parts
            "train": good[:n_train],                                   # first 80 %
            "val": good[n_train:n_train + n_val],                      # next 16 %
            "test": good[n_train + n_val:],                            # remaining ~4 %
        }

        for split, split_files in parts.items():                       # write each part to disk
            dst = config.DATA_DIR / split / cls_dir                    # e.g. data/split/train/cats
            dst.mkdir(parents=True, exist_ok=True)                     # create the folder
            for f in split_files:                                      # copy every image
                shutil.copy2(f, dst / f.name)                          # copy2 also preserves file metadata
            print(f"  {split:<5} -> {len(split_files):>5} images")      # report counts


if __name__ == "__main__":     # run main() only when executed directly, not when imported
    main()
