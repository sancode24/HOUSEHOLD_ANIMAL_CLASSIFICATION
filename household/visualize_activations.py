"""
visualize_activations.py  (STEP 5)
Reproduces paper Fig. 5 / Sec 5.1: runs one image through the trained model and
shows the feature map of every filter in the first few conv layers as a grid.
Early layers ~ edge detectors; deeper layers ~ abstract, sparse (more blank filters).

Usage:
    python visualize_activations.py                           # first dog in test set, VGG-6
    python visualize_activations.py --image my_dog.jpg --model vgg16_ft --layers 4
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

import config
import models
from data import default_test_images, load_image


def make_grid(act, per_row):
    """Tile all channels of one feature map (H, W, C) into a single 2-D image."""
    size, n_ch = act.shape[0], act.shape[-1]          # spatial size and number of filters
    n_rows = n_ch // per_row                          # e.g. 128 filters / 16 per row = 8 rows
    grid = np.zeros((size * n_rows, size * per_row))  # empty canvas
    for r in range(n_rows):                           # each row of tiles
        for c in range(per_row):                      # each column of tiles
            ch = act[:, :, r * per_row + c].copy()    # one filter's activation map
            if ch.std() > 0:                          # dead filters (all zeros) stay blank
                ch = (ch - ch.mean()) / ch.std()      # standardise for visible contrast
            ch = np.clip(ch * 64 + 128, 0, 255)       # map to displayable 0-255 range
            grid[r * size:(r + 1) * size, c * size:(c + 1) * size] = ch   # paste tile
    return grid


def main():
    parser = argparse.ArgumentParser(description="Visualise intermediate conv-layer activations.")
    parser.add_argument("--model", default="vgg6", choices=list(models.LAST_CONV))
    parser.add_argument("--image", default=None, help="image path (default: first test dog)")
    parser.add_argument("--layers", type=int, default=4, help="how many conv layers to show")
    parser.add_argument("--per-row", type=int, default=16, help="filters per row in the grid")
    args = parser.parse_args()

    config.ensure_dirs()
    image_path = args.image or default_test_images(1)[-1]              # [-1] = the dog
    model = models.load_trained(args.model)                            # trained network

    conv_names = [l.name for l in model.layers                         # names of all conv layers...
                  if isinstance(l, keras.layers.Conv2D)][: args.layers]  # ...first N of them
    act_model = keras.Model(model.inputs,                              # new model: same input,
                            [model.get_layer(n).output for n in conv_names])  # outputs = conv activations

    img = load_image(image_path)                                       # (1, 150, 150, 3)
    acts = act_model.predict(img, verbose=0)                           # list: one array per layer
    if not isinstance(acts, list):                                     # single layer -> wrap in list
        acts = [acts]

    for name, act in zip(conv_names, acts):                            # one figure per layer
        grid = make_grid(act[0], args.per_row)                         # act[0] drops batch dimension
        size = act.shape[1]                                            # tile size in pixels
        plt.figure(figsize=(grid.shape[1] / size, grid.shape[0] / size))  # 1 inch per tile
        plt.imshow(grid, aspect="auto", cmap="viridis")                # same colormap as the paper
        plt.title(f"{name}  ({act.shape[-1]} filters, {size}x{size})")
        plt.axis("off")
        out = config.FIG_DIR / f"activations_{args.model}_{name}.png"
        plt.savefig(out, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
