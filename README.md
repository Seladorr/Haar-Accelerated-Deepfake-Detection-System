# Haar-Accelerated Deepfake Detection System

Video deepfake detection using a **ResNeXt-50 + LSTM** classifier with **Squeeze-and-Excitation** and **Spatial Attention**, where the usual deep-CNN face-detection step (`face_recognition`) is replaced by OpenCV's lightweight **Haar Cascade** detector to make preprocessing much faster and dependency-light.

Full write-up: [`dlcv_final.pdf`](dlcv_final.pdf) — *Deepfake Detection with Lightweight Attention-Based Architecture*.

## How it works

```
input video ──► extract ≤150 frames ──► Haar Cascade face detect (grayscale)
            ──► crop first face, resize 112×112 ──► face-only .mp4
            ──► ResNeXt-50 (ImageNet) feature maps ──► SE block ──► Spatial Attention ──► avg-pool (2048-d / frame)
            ──► LSTM (1 layer, 2048 hidden) ──► LayerNorm ──► mean over time
            ──► residual head (FC ─ GELU ─ Dropout 0.4 ─ +skip ─ FC) ──► REAL / FAKE
```

At prediction time the class-activation map of the last frame is overlaid on the face and saved as a heatmap (`enhanced_heatmap.png`).

## Repository layout

| Path | What it is |
|---|---|
| `New_preprocessing (1).ipynb` | **Step 1** – Haar Cascade face cropping: `real_yt/` → `real_yt_face_videos/`, `fake_yt/` → `fake_yt_face_videos/` |
| `New_Model_and_train.ipynb` | **Step 2** – dataset loading, 80/20 split, `EnhancedModel` definition and training |
| `New_Predict.ipynb` | **Step 3** – load a checkpoint, classify videos, generate heatmaps |
| `Codefiles/` | Baseline pipeline (`face_recognition` preprocessing + plain ResNeXt-LSTM `Model`). Contains the `train_epoch` / `test` / `AverageMeter` / `calculate_accuracy` helpers |
| `Old_preprocessing.ipynb`, `Predict.ipynb`, `Model_and_train_csv.ipynb` | Earlier baseline versions |
| `New_Model_and_train copy.ipynb` | Experimental EfficientNet-B3 + channel-attention variant |
| `final_model/checkpoint{1-4}.pt` | Weights for the **enhanced** model (`EnhancedModel`) — `checkpoint3.pt` is the one used in `New_Predict.ipynb` |
| `model/checkpoint*.pt` | Weights for the **baseline** ResNeXt-LSTM (`Model`) |
| `DeepfakeDetector/`, `Codefiles/checkpoint.pt`, `PREV_VID/checkpoint.pt`, `trial2*.pt` | Intermediate / experimental checkpoints |
| `*.png` | Training curves, overfitting plots and prediction heatmaps |

## Setup

### 1. Clone (with the model weights)

The `.pt` checkpoints (~3 GB) are stored with **Git LFS**. Install it first, otherwise you'll only get small pointer files:

```bash
git lfs install
git clone https://github.com/Seladorr/Haar-Accelerated-Deepfake-Detection-System.git
cd Haar-Accelerated-Deepfake-Detection-System
```

### 2. Python environment

Python 3.9+ is recommended.

```bash
python -m venv venv
# Windows: venv\Scripts\activate   |   Linux/macOS: source venv/bin/activate
pip install torch torchvision opencv-python numpy matplotlib scikit-learn seaborn pandas tqdm jupyter
```

A CUDA GPU (or Apple Silicon `mps`) is strongly recommended for training. `face_recognition` (which needs `dlib`) is only required for the baseline notebooks in `Codefiles/` — the Haar pipeline doesn't use it, so you can delete that import from the notebooks if you don't want to install it.

### 3. Dataset

Videos are **not** included in this repo (too large). The project used a balanced subset of **FaceForensics++**: 1,000 real and 1,000 deepfake videos. Place them like this in the project root:

```
real_yt/   *.mp4   (real videos)
fake_yt/   *.mp4   (deepfake videos)
```

## Running

Launch Jupyter from the project root (all notebook paths are relative to it):

```bash
jupyter notebook
```

### Step 1 – Preprocess (Haar Cascade face cropping)
Run all cells of **`New_preprocessing (1).ipynb`**. For every video it reads up to 150 frames, detects the face with `haarcascade_frontalface_default.xml`, crops & resizes it to 112×112 and writes it to `real_yt_face_videos/` / `fake_yt_face_videos/`. Already-processed videos are skipped, so it is safe to re-run.

### Step 2 – Train
Open **`New_Model_and_train.ipynb`** and run the cells in order.

- Labels: `1 = REAL`, `0 = FAKE`; stratified 80/20 split (`random_state=42`) → 1,600 train / 400 val.
- Settings: sequence length 10 frames, batch size 4, Adam (`lr=1e-5`, `weight_decay=1e-5`), cross-entropy, 20 epochs.
- The training-loop helpers (`train_epoch`, `test`, `AverageMeter`, `calculate_accuracy`) live in `Codefiles/New_Model_and_train.ipynb` — copy that cell in and run it before the training cell. `train_epoch` saves weights to `./checkpoint.pt` after every epoch.
- The device line is set to `mps`/`cpu`; on an NVIDIA GPU change it to `torch.device("cuda" if torch.cuda.is_available() else "cpu")`.

### Step 3 – Predict
Open **`New_Predict.ipynb`**:

1. Run the model-definition and helper cells.
2. Point `path_to_model` at a checkpoint, e.g. `./final_model/checkpoint3.pt` (the notebook currently has an absolute macOS path — change it). If you're on a CPU-only machine use `torch.load(path, map_location="cpu")`.
3. Set `path_to_videos` to the videos you want to test and run the prediction cell. It prints `REAL` / `FAKE` with a confidence score and saves the attention heatmap to `enhanced_heatmap.png`.

Example output from the notebook:

```
./real_yt/130.mp4          confidence 98.47  → REAL
fake_yt/042_084.mp4        confidence 97.66  → FAKE
```

## Results

Enhanced model on the 400-video validation split (label order: Fake, Real):

|              | Pred. Fake | Pred. Real |
|---|---|---|
| **Actual Fake** | 129 | 71 |
| **Actual Real** | 35 | 165 |

Accuracy **73.5%** from this confusion matrix (the report quotes 76.5%; the best per-epoch validation accuracy logged during training was 76.0%). The model catches real videos well but produces a fair number of fake→real misses; training curves (`New-overfit-1.png`, `New-Overfit2.png`) show overfitting after ~3 epochs, so more data / stronger augmentation is the main avenue for improvement.

Haar Cascade preprocessing ran up to ~5× faster than the `face_recognition` CNN detector in our tests, with no extra deep-learning dependency.

## Limitations

- Haar Cascades only handle reasonably frontal, well-lit faces; profile or occluded faces are skipped.
- Only the first detected face per frame is used.
- Trained on a small FaceForensics++ subset, so generalisation to other generators is untested.

## Authors

Harsh Oza, Satish Panda, Shreyash Kumar, Neerav Sethi — DLCV course project, IIT Kanpur.

## References

- Hu et al., *Squeeze-and-Excitation Networks*, CVPR 2018
- Woo et al., *CBAM: Convolutional Block Attention Module*, ECCV 2018
- Viola & Jones, *Rapid Object Detection using a Boosted Cascade of Simple Features*, CVPR 2001
- Rössler et al., *FaceForensics++*, ICCV 2019
