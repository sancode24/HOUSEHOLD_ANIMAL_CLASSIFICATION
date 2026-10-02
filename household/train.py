"""
train.py  (STEP 2)
Trains one model variant and saves its best weights + per-epoch history.

Usage (run in this order; vgg16_ft starts from the trained vgg16 weights):
    python train.py --model vgg6_noreg   # optional: shows overfitting
    python train.py --model vgg6
    python train.py --model vgg16
    python train.py --model vgg16_ft
Add --epochs 100 to match the paper's Fig. 4.
"""

import argparse                    # command-line flags
import json                        # save training history for plot_history.py

import tensorflow as tf            # seeding
from tensorflow import keras       # optimizers, callbacks

import config                      # hyperparameters & paths
import models                      # model builders
from data import get_datasets      # tf.data pipelines


def main():
    parser = argparse.ArgumentParser(description="Train a model variant.")
    parser.add_argument("--model", required=True, choices=list(models.LAST_CONV))  # which variant
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)                # number of passes over data
    parser.add_argument("--patience", type=int, default=0,                          # 0 = no early stopping
                        help="stop if val_loss doesn't improve for N epochs (0 disables)")
    args = parser.parse_args()

    config.ensure_dirs()                                   # create outputs/ sub-folders
    tf.keras.utils.set_random_seed(config.SEED)            # seeds Python, NumPy and TF in one call

    train_ds, val_ds = get_datasets(augment=models.AUGMENT[args.model])  # augmentation on/off per variant

    if args.model == "vgg16_ft":                                         # fine-tuning continues from vgg16
        src = models.weights_path("vgg16")                               # trained frozen-base weights
        if not src.exists():                                             # must train vgg16 first
            raise SystemExit("Train the frozen model first: python train.py --model vgg16")
        model = models.build_model("vgg16", pretrained=False)            # same architecture, all conv frozen
        model.load_weights(str(src))                                     # ImageNet conv + our trained head
        models.unfreeze_block5(model)                                    # now make block5 trainable
    else:
        model = models.build_model(args.model)                           # fresh model (VGG-16 gets ImageNet)

    model.compile(                                                       # compile AFTER (un)freezing
        optimizer=keras.optimizers.RMSprop(learning_rate=config.LR[args.model]),  # paper uses RMSprop
        loss="binary_crossentropy",                                      # standard loss for 2-class sigmoid
        metrics=["accuracy"],                                            # report accuracy every epoch
    )
    model.summary()                                                      # prints layers + trainable params

    ckpt_path = models.weights_path(args.model)                          # outputs/weights/<model>.weights.h5
    callbacks = [
        keras.callbacks.ModelCheckpoint(                                 # save weights whenever val improves
            str(ckpt_path),
            monitor="val_accuracy",                                      # metric that decides "best"
            save_best_only=True,                                         # keep only the best epoch
            save_weights_only=True,                                      # weights only -> robust across TF versions
            verbose=1,                                                   # print when it saves
        ),
        keras.callbacks.CSVLogger(str(config.HISTORY_DIR / f"{args.model}.csv")),  # per-epoch log (spreadsheet)
    ]
    if args.patience > 0:                                                # optional early stopping
        callbacks.append(keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=args.patience, restore_best_weights=True))

    history = model.fit(                                                 # the actual training loop
        train_ds,                                                        # training batches
        validation_data=val_ds,                                          # evaluated after every epoch
        epochs=args.epochs,
        callbacks=callbacks,
    )

    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}  # numpy floats -> JSON floats
    with open(config.HISTORY_DIR / f"{args.model}.json", "w") as f:             # outputs/history/<model>.json
        json.dump(hist, f, indent=2)                                            # used by plot_history.py

    best = max(hist["val_accuracy"])                                     # best validation accuracy reached
    print(f"\nDone. Best val_accuracy = {best:.4f}. Weights -> {ckpt_path}")


if __name__ == "__main__":
    main()
