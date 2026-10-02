"""
models.py
Model definitions:
  vgg6_noreg : VGG-6 from scratch, no dropout/augmentation (shows overfitting)     -> paper Sec 4.1
  vgg6       : VGG-6 + dropout + augmentation               (paper "VGG06_base")  -> paper Sec 4.3
  vgg16      : VGG-16 ImageNet conv base FROZEN + new head  (paper "VGG16_base")  -> paper Sec 4.2
  vgg16_ft   : vgg16 with block5 unfrozen, fine-tuned       (paper "VGG16_fine")  -> paper Sec 4.3

Every model takes raw images in [0, 255] and does its own preprocessing
internally, so training, evaluation and the live demo all feed identical input.
The final Dense layer is named "dense_2" and outputs a LOGIT; a separate sigmoid
layer turns it into a probability. Grad-CAM uses the logit (see gradcam.py).
"""

import tensorflow as tf                    # tensor ops for the custom preprocessing layer
from tensorflow import keras               # Model / Input / applications
from tensorflow.keras import layers        # Conv2D, Dense, Dropout, ...

import config                              # image size, weights folder

LAST_CONV = {                              # last conv layer of each model (target layer for Grad-CAM)
    "vgg6_noreg": "conv2d_4",
    "vgg6": "conv2d_4",
    "vgg16": "block5_conv3",
    "vgg16_ft": "block5_conv3",
}
AUGMENT = {                                # whether each model trains on augmented images
    "vgg6_noreg": False,                   # deliberately unregularised
    "vgg6": True,
    "vgg16": True,
    "vgg16_ft": True,
}
LOGIT_LAYER = "dense_2"                    # name of the final (pre-sigmoid) Dense layer in all models


class VGGPreprocess(layers.Layer):
    """ImageNet 'caffe' preprocessing that VGG-16's pretrained weights expect:
    RGB -> BGR channel order, then subtract the ImageNet per-channel mean. No trainable weights."""

    MEAN_BGR = [103.939, 116.779, 123.68]                        # ImageNet mean pixel (B, G, R)

    def call(self, x):                                           # x: (batch, H, W, 3) RGB in [0, 255]
        x = tf.reverse(x, axis=[-1])                             # flip last axis: RGB -> BGR
        return x - tf.constant(self.MEAN_BGR, dtype=x.dtype)     # zero-centre each channel


def build_vgg6(dropout=True):
    """VGG-6 from scratch (paper Fig. 3): 4 x [Conv3x3 + MaxPool] blocks, then Flatten -> Dense -> Dense.
    '6' = 4 conv layers + 2 dense layers that carry weights."""
    inputs = keras.Input(shape=config.IMG_SIZE + (3,), name="image")             # (150, 150, 3) input
    x = layers.Rescaling(1.0 / 255, name="rescale")(inputs)                       # [0,255] -> [0,1]
    for i, filters in enumerate([32, 64, 128, 128], start=1):                     # filters double with depth
        x = layers.Conv2D(filters, 3, padding="same", activation="relu",          # 3x3 conv, 'same' keeps HxW
                          name=f"conv2d_{i}")(x)                                  # names match the paper
        x = layers.MaxPooling2D(2, name=f"max_pooling2d_{i}")(x)                  # halve H and W
    x = layers.Flatten(name="flatten_1")(x)                                       # 9x9x128 -> 10368 vector
    if dropout:                                                                   # regularised variant only
        x = layers.Dropout(0.5, name="dropout")(x)                                # randomly zero 50 % of units
    x = layers.Dense(512, activation="relu", name="dense_1")(x)                   # fully connected hidden layer
    x = layers.Dense(1, name=LOGIT_LAYER)(x)                                      # 1 logit: >0 means "dog"
    outputs = layers.Activation("sigmoid", name="prob")(x)                        # P(dog) in [0, 1]
    return keras.Model(inputs, outputs, name="vgg6")                              # wire input -> output


def build_vgg16(pretrained=True):
    """Transfer learning (paper Sec 4.2): VGG-16 conv base (ImageNet weights, frozen) + new binary head."""
    inputs = keras.Input(shape=config.IMG_SIZE + (3,), name="image")             # (150, 150, 3) input
    x = VGGPreprocess(name="vgg_preprocess")(inputs)                              # BGR + mean subtraction
    base = keras.applications.VGG16(                                              # the 13 conv layers of VGG-16
        include_top=False,                                                        # drop ImageNet's 1000-class head
        weights="imagenet" if pretrained else None,                               # None when we load our own weights
        input_tensor=x,                                                           # build layers directly on our graph
        input_shape=config.IMG_SIZE + (3,),                                       # 150x150x3 -> output 4x4x512
    )
    for layer in base.layers:                                                     # every layer of the conv base
        layer.trainable = False                                                   # freeze: weights not updated
    x = layers.Flatten(name="flatten_1")(base.output)                             # 4x4x512 -> 8192 vector
    x = layers.Dense(256, activation="relu", name="dense_1")(x)                   # new hidden layer (trainable)
    x = layers.Dropout(0.5, name="dropout")(x)                                    # regularise the new head
    x = layers.Dense(1, name=LOGIT_LAYER)(x)                                      # 1 logit
    outputs = layers.Activation("sigmoid", name="prob")(x)                        # P(dog)
    return keras.Model(inputs, outputs, name="vgg16_transfer")


def unfreeze_block5(model):
    """Fine-tuning (paper Sec 4.3): unfreeze from 'block5_conv1' onward; blocks 1-4 stay frozen
    because early layers learn generic edges/textures that transfer as-is."""
    for layer in model.layers:                                   # walk every layer in the model
        if layer.name.startswith("block"):                       # only VGG-16 conv-base layers
            layer.trainable = layer.name.startswith("block5")    # True for block5_*, False for block1-4
    return model


def build_model(name, pretrained=True):
    """Build an (untrained or ImageNet-initialised) model by its variant name."""
    if name == "vgg6_noreg":
        return build_vgg6(dropout=False)
    if name == "vgg6":
        return build_vgg6(dropout=True)
    if name == "vgg16":
        return build_vgg16(pretrained)
    if name == "vgg16_ft":
        return unfreeze_block5(build_vgg16(pretrained))         # same architecture, block5 trainable
    raise ValueError(f"Unknown model '{name}'. Choose from {list(LAST_CONV)}")


def weights_path(name):
    """Where the trained weights of a model variant are stored."""
    return config.WEIGHTS_DIR / f"{name}.weights.h5"            # Keras requires the .weights.h5 suffix


def load_trained(name):
    """Rebuild the architecture and load trained weights (no ImageNet download needed)."""
    path = weights_path(name)                                    # expected weights file
    if not path.exists():                                        # model not trained yet
        raise SystemExit(f"{path} not found. Run: python train.py --model {name}")
    model = build_model(name, pretrained=False)                  # same layers, random init
    model.load_weights(str(path))                                # overwrite with trained weights
    return model
