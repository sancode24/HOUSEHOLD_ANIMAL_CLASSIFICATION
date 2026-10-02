"""
data.py
tf.data input pipeline: loads images from data/split/<split>/<class>/,
applies data augmentation to the training set, and provides helpers to load
single images for prediction / visualisation.
"""

import tensorflow as tf            # tf.data (fast input pipelines) and tensor ops
from tensorflow import keras       # high-level API: image loading + augmentation layers

import config                      # paths, image size, batch size, seed


def get_augmenter():
    """Random transforms applied to TRAINING images only (fights overfitting)."""
    return keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal"),   # mirror left<->right (a flipped cat is still a cat)
            keras.layers.RandomRotation(0.1),        # rotate by up to +/-10 % of a full turn (~36 deg)
            keras.layers.RandomZoom(0.2),            # zoom in/out by up to 20 %
        ],
        name="augmentation",                         # name shows up in summaries/debugging
    )


def load_split(split, shuffle):
    """Return a batched tf.data.Dataset of (images, labels) for 'train', 'val' or 'test'."""
    return keras.utils.image_dataset_from_directory(
        str(config.DATA_DIR / split),     # e.g. data/split/train (str() for older TF versions)
        labels="inferred",                # label = sub-folder name
        label_mode="binary",              # labels are floats 0.0 / 1.0 -> matches a sigmoid output
        class_names=config.CLASS_DIRS,    # fixes order: cats -> 0, dogs -> 1
        color_mode="rgb",                 # grayscale images are converted to 3 channels
        image_size=config.IMG_SIZE,       # resize every image to 150x150 (bilinear)
        batch_size=config.BATCH_SIZE,     # 32 images per batch
        shuffle=shuffle,                  # shuffle train; keep val/test order fixed
        seed=config.SEED,                 # reproducible shuffling
    )                                     # pixel values stay float32 in [0, 255]; models rescale internally


def get_datasets(augment):
    """Return (train_ds, val_ds). If augment=True, training images are randomly transformed each epoch."""
    train_ds = load_split("train", shuffle=True)          # training set, shuffled
    val_ds = load_split("val", shuffle=False)             # validation set, fixed order
    if augment:                                           # only for regularised models
        aug = get_augmenter()                             # build augmentation pipeline once
        train_ds = train_ds.map(                          # apply it to every training batch
            lambda x, y: (aug(x, training=True), y),      # training=True activates the random ops
            num_parallel_calls=tf.data.AUTOTUNE,          # augment batches in parallel on CPU
        )
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)        # prepare next batch while GPU trains on current
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)            # same for validation
    return train_ds, val_ds


def load_image(path):
    """Load one image file as a (1, 150, 150, 3) float32 batch in [0, 255] for model input."""
    img = keras.utils.load_img(path, target_size=config.IMG_SIZE,   # read + resize with Pillow
                               interpolation="bilinear")            # same resize method as training
    arr = keras.utils.img_to_array(img)                             # PIL image -> float32 array (H, W, 3)
    return arr[None]                                                # add batch dimension -> (1, H, W, 3)


def default_test_images(n_per_class=1):
    """Pick the first n images of each class from the test split (used when no --image is given)."""
    paths = []                                                      # collected file paths
    for cls_dir in config.CLASS_DIRS:                               # cats, then dogs
        files = sorted((config.DATA_DIR / "test" / cls_dir).glob("*.jpg"))  # deterministic order
        paths += files[:n_per_class]                                # take the first n
    return paths
