"""
plot_history.py  (STEP 4)
Reproduces paper Fig. 4: training (dots) vs validation (dashed) accuracy and
loss for every model that has been trained.

Usage:
    python plot_history.py
"""

import json                              # read history files written by train.py

import matplotlib.pyplot as plt          # plotting

import config

STYLE = {                                # model -> (colour, legend label), colours match the paper
    "vgg6_noreg": ("tab:orange", "VGG06_noreg"),
    "vgg6": ("red", "VGG06_base"),
    "vgg16": ("black", "VGG16_base"),
    "vgg16_ft": ("blue", "VGG16_fine"),
}


def main():
    config.ensure_dirs()                                             # outputs/figures must exist
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))                  # left: accuracy, right: loss
    plotted = 0                                                      # count models actually found

    for name, (color, label) in STYLE.items():                       # each model variant
        path = config.HISTORY_DIR / f"{name}.json"                   # its history file
        if not path.exists():                                        # not trained -> skip
            continue
        hist = json.loads(path.read_text())                          # dict of metric -> list per epoch
        epochs = range(1, len(hist["accuracy"]) + 1)                 # x-axis: 1..N
        for ax, metric in zip(axes, ["accuracy", "loss"]):           # same drawing for both panels
            ax.plot(epochs, hist[metric], ".", color=color,          # training curve as dots
                    label=f"{label}_tra")
            ax.plot(epochs, hist[f"val_{metric}"], "--", color=color,  # validation curve dashed
                    linewidth=0.9, label=f"{label}_val")
        plotted += 1

    if plotted == 0:
        raise SystemExit("No history files found. Run train.py first.")

    for ax, metric in zip(axes, ["Accuracy", "Loss"]):               # label both panels
        ax.set_xlabel("Epochs")
        ax.set_ylabel(metric)
        ax.set_ylim(0, 1.0)                                          # same 0-1 scale as the paper
        ax.grid(alpha=0.3)                                           # light grid for readability
        ax.legend(fontsize=8)
    fig.suptitle("Training (tra) vs Validation (val)")               # paper's figure title
    out = config.FIG_DIR / "accuracy_loss.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")                   # save PNG
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
