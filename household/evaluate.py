"""
evaluate.py  (STEP 3)
Evaluates every trained model on the held-out TEST split (never seen during
training or model selection): accuracy, precision/recall/F1, confusion matrix.

Usage:
    python evaluate.py                       # all models that have weights
    python evaluate.py --models vgg6 vgg16_ft
"""

import argparse                                        # command-line flags

import matplotlib.pyplot as plt                        # plotting confusion matrices
import numpy as np                                     # array maths
from sklearn.metrics import (ConfusionMatrixDisplay,   # draws a confusion matrix
                             classification_report,    # precision / recall / F1 table
                             confusion_matrix)         # computes the 2x2 matrix

import config
import models
from data import load_split


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models on the test split.")
    parser.add_argument("--models", nargs="+", default=None)        # optional explicit list
    args = parser.parse_args()

    config.ensure_dirs()                                            # make sure outputs/figures exists
    names = args.models or [n for n in models.LAST_CONV              # default: every variant...
                            if models.weights_path(n).exists()]     # ...that has been trained
    if not names:
        raise SystemExit("No trained weights found. Run train.py first.")

    test_ds = load_split("test", shuffle=False)                     # fixed order so labels align with preds
    y_true = np.concatenate([y.numpy() for _, y in test_ds]).ravel().astype(int)  # ground-truth 0/1 labels

    summary = []                                                    # rows for the final table
    for name in names:                                              # evaluate each model
        model = models.load_trained(name)                           # architecture + trained weights
        probs = model.predict(test_ds, verbose=0).ravel()           # P(dog) for every test image
        y_pred = (probs >= 0.5).astype(int)                         # threshold at 0.5 -> class 0/1
        acc = float((y_pred == y_true).mean())                      # fraction correct

        print(f"\n===== {name} =====")
        print(classification_report(y_true, y_pred,                 # per-class precision / recall / F1
                                    target_names=config.CLASS_NAMES, digits=4))

        cm = confusion_matrix(y_true, y_pred)                       # rows = true class, cols = predicted
        ConfusionMatrixDisplay(cm, display_labels=config.CLASS_NAMES).plot(cmap="Blues")  # draw it
        plt.title(f"{name} - test accuracy {acc:.4f}")              # title with accuracy
        plt.savefig(config.FIG_DIR / f"confusion_{name}.png", dpi=150, bbox_inches="tight")  # save PNG
        plt.close()                                                 # free memory before next model

        summary.append((name, acc))                                 # remember for summary table

    print("\nModel          Test accuracy")                          # compact comparison table
    for name, acc in summary:
        print(f"{name:<14} {acc:.4f}")


if __name__ == "__main__":
    main()
