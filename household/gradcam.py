"""
gradcam.py  (STEP 6 + LIVE DEMO)
Reproduces paper Fig. 6 / Sec 5.2: class-activation heatmaps showing WHICH
image regions drove the cat/dog decision.

We use Grad-CAM (Selvaraju et al., 2017), the generalisation of CAM: original
CAM needs a Global-Average-Pooling layer right before the classifier, which our
Flatten-based heads don't have. Grad-CAM gets the per-filter importance weights
from gradients instead, so it works on any CNN without changing the architecture.

Usage:
    python gradcam.py                                   # 2 cats + 2 dogs from test set
    python gradcam.py --images pet1.jpg pet2.jpg --show # live demo on any photos
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

import config
import models
from data import default_test_images, load_image


def gradcam(model, img, conv_name):
    """Return (heatmap in [0,1] at conv resolution, P(dog)) for one image batch of size 1."""
    grad_model = keras.Model(                                   # model with two outputs:
        model.inputs,
        [model.get_layer(conv_name).output,                     # 1) last conv feature maps A^k
         model.get_layer(models.LOGIT_LAYER).output],           # 2) the pre-sigmoid logit
    )
    with tf.GradientTape() as tape:                             # record ops to compute gradients
        conv_out, logit = grad_model(img, training=False)       # forward pass (dropout off)
        logit = logit[:, 0]                                     # shape (1,)
        score = tf.where(logit >= 0, logit, -logit)             # score of PREDICTED class:
                                                                #   +logit for dog, -logit for cat
    grads = tape.gradient(score, conv_out)                      # d(score)/d(A^k): (1, h, w, C)
    weights = tf.reduce_mean(grads, axis=(1, 2))                # alpha_k = spatial mean of gradients: (1, C)
    cam = tf.einsum("bhwc,bc->bhw", conv_out, weights)          # weighted sum of feature maps: (1, h, w)
    cam = tf.nn.relu(cam)[0].numpy()                            # keep only positive evidence
    cam = cam / (cam.max() + 1e-8)                              # normalise to [0, 1]
    prob_dog = float(tf.sigmoid(logit)[0])                      # logit -> probability
    return cam, prob_dog
    # Using the logit (not the sigmoid output) matters: at p~1.0 the sigmoid saturates and its
    # gradient underflows to ~0, which would give an empty heatmap for confident predictions.


def overlay(image_path, cam):
    """Upsample the heatmap to the ORIGINAL photo size and return (photo, heatmap) arrays."""
    photo = keras.utils.img_to_array(keras.utils.load_img(image_path)).astype("uint8")  # full-res RGB
    h, w = photo.shape[:2]                                                               # original size
    heat = tf.image.resize(cam[..., None], (h, w)).numpy()[..., 0]                       # bilinear upsample
    return photo, heat


def main():
    parser = argparse.ArgumentParser(description="Grad-CAM heatmaps / live demo.")
    parser.add_argument("--model", default="vgg16_ft", choices=list(models.LAST_CONV))
    parser.add_argument("--images", nargs="+", default=None, help="image paths (default: test samples)")
    parser.add_argument("--show", action="store_true", help="open a window (for the live demo)")
    args = parser.parse_args()

    config.ensure_dirs()
    paths = args.images or default_test_images(2)                     # default: 2 cats + 2 dogs
    model = models.load_trained(args.model)
    conv_name = models.LAST_CONV[args.model]                          # last conv layer = best spatial semantics

    n = len(paths)
    cols = min(n, 2)                                                  # 2 columns like the paper's figure
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)

    for ax, path in zip(axes.ravel(), paths):                         # one panel per image
        cam, p_dog = gradcam(model, load_image(path), conv_name)      # heatmap + probability
        label = "dog" if p_dog >= 0.5 else "cat"                      # predicted class
        conf = p_dog if label == "dog" else 1 - p_dog                 # confidence of that class
        photo, heat = overlay(path, cam)
        ax.imshow(photo)                                              # original image
        ax.imshow(heat, cmap="jet", alpha=0.4)                        # semi-transparent heatmap on top
        ax.set_title(f"{label} {conf * 100:.2f}%", fontsize=14)       # e.g. "cat 99.87%"
        ax.axis("off")
        print(f"{path}: {label} ({conf * 100:.2f}%)")

    for ax in axes.ravel()[n:]:                                       # hide unused panels
        ax.axis("off")

    out = config.FIG_DIR / f"gradcam_{args.model}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    if args.show:                                                     # interactive window for the demo
        plt.show()


if __name__ == "__main__":
    main()
