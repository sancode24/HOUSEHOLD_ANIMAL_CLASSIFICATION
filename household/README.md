# Household Animals Classification Using Deep Learning

UE24CS352A Machine Learning – Mini-Project. Cats vs Dogs classification, based on Lin (CS229, Stanford 2020). It covers four pieces of work:

1. **VGG-6 from scratch.** The baseline is first trained without regularisation, which shows the overfitting. It is then retrained with dropout and data augmentation.
2. **Transfer learning.** The VGG-16 ImageNet conv base is kept frozen, and only a new classifier head is trained.
3. **Fine-tuning.** The model is unfrozen from `block5_conv1` onward and trained further.
4. **Visualisation.** This includes intermediate-layer activations and Grad-CAM class-activation heatmaps.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
You need Python 3.9–3.12. A GPU is strongly recommended for the full dataset; Google Colab works fine.

## Dataset
This project uses the Kaggle [Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/data) dataset: 25,000 labelled JPEGs, 12,500 per class.

1. Accept the competition rules on Kaggle.
2. Put your API token at `~/.kaggle/kaggle.json`.
3. Download and extract:
```bash
kaggle competitions download -c dogs-vs-cats -p data/raw
cd data/raw && unzip -q dogs-vs-cats.zip && unzip -q train.zip && cd ../..
# expected: data/raw/train/cat.0.jpg ... dog.12499.jpg
```

## Run (in order)
```bash
python prepare_data.py                    # drop corrupt files, split 80/16/4 into data/split/
python train.py --model vgg6_noreg        # optional: overfitting baseline
python train.py --model vgg6              # VGG-6 + dropout + augmentation
python train.py --model vgg16             # frozen VGG-16 + new head
python train.py --model vgg16_ft          # fine-tune block5 (needs vgg16 weights)
python evaluate.py                        # test-set accuracy, F1, confusion matrices
python plot_history.py                    # Fig. 4: accuracy/loss curves
python visualize_activations.py           # Fig. 5: conv-layer feature maps
python gradcam.py                         # Fig. 6: Grad-CAM heatmaps
```
A few options:
- `--epochs 100` reproduces the paper's training length. The default is 30.
- `--patience 10` enables early stopping.
- `prepare_data.py --max-per-class 2000` gives a quick subset for CPU runs.

**Live demo:** `python gradcam.py --images path/to/pet.jpg --show`

## Project structure
| File | Purpose |
|---|---|
| `config.py` | Paths and hyperparameters |
| `prepare_data.py` | Image validation and train/val/test split |
| `data.py` | tf.data pipelines, augmentation, and image loading |
| `models.py` | VGG-6, VGG-16 transfer, and fine-tune definitions |
| `train.py` | Training: best weights go to `outputs/weights/`, history to `outputs/history/` |
| `evaluate.py` | Test-set metrics and confusion matrices |
| `plot_history.py` | Learning curves |
| `visualize_activations.py` | Intermediate activations |
| `gradcam.py` | Grad-CAM heatmaps and the live demo |

All figures are saved to `outputs/figures/`.

## Results
| Model | Test accuracy |
|---|---|
| VGG-6 (no reg.) | _fill in_ |
| VGG-6 (dropout + aug.) | _fill in_ |
| VGG-16 frozen | _fill in_ |
| VGG-16 fine-tuned | _fill in_ |

## References
- L. Lin, *Household Animals Classification Using Deep Learning*, CS229 Stanford, 2020.
- K. Simonyan & A. Zisserman, *Very Deep Convolutional Networks for Large-Scale Image Recognition*, 2014.
- R. Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*, ICCV 2017.
