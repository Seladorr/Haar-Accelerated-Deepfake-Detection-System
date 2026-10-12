# Haar-Accelerated Deepfake Detection

This was our course project for DLCV (Deep Learning for Computer Vision) at IIT Kanpur. We're three undergrads who wanted to see if we could take a well-known deepfake detector and make it faster and lighter without wrecking the accuracy.

The detector we started from is the usual **ResNeXt + LSTM** setup: a CNN looks at each frame, an LSTM looks at how frames change over time, and deepfakes tend to slip up on both. The annoying part was the very first step — finding the face in every frame using the `face_recognition` library. It's slow, it drags in `dlib`, and it was a pain to install on our laptops.

So we did two things:

1. **Swapped the face detector for OpenCV's Haar Cascade.** Yes, the old-school Viola-Jones one from 2001. It's not fancy, but deepfake datasets are mostly people looking straight at the camera, and for that it works fine. In our tests it was up to ~5× faster.
2. **Added a bit of attention to the model.** We put a Squeeze-and-Excitation block and a spatial attention block on top of the ResNeXt features, plus a small residual head at the end.

Our full report is in [`dlcv_final.pdf`](dlcv_final.pdf) if you want the details.

## The pipeline

```
video ─► grab up to 150 frames ─► Haar Cascade finds the face ─► crop + resize to 112×112
      ─► ResNeXt-50 features ─► SE block ─► spatial attention ─► one 2048-d vector per frame
      ─► LSTM over the frames ─► average ─► small residual classifier ─► REAL or FAKE
```

When predicting, we also draw a heatmap over the face showing where the model was "looking" (`enhanced_heatmap.png`).

## What's in here

| File / folder | What it is |
|---|---|
| `New_preprocessing.ipynb` | Step 1: crops faces out of the raw videos with Haar Cascade |
| `New_Model_and_train.ipynb` | Step 2: builds the model and trains it |
| `New_Predict.ipynb` | Step 3: loads a trained model and tells you if a video is real or fake |
| `final_model/` | Trained weights for our final model (`checkpoint3.pt` is the one we use for predictions) |
| `model/` | Weights for the original ResNeXt + LSTM, without our changes, for comparison |
| `Codefiles/`, `Old_preprocessing.ipynb`, `Predict.ipynb`, `Model_and_train_csv.ipynb` | Our earlier attempts using `face_recognition`. Kept around for reference |
| `Experimental_EfficientNet_train.ipynb` | We also tried EfficientNet-B3 as the backbone. Didn't end up using it |
| `DeepfakeDetector/`, `trial2*.pt`, `PREV_VID/`, `Codefiles/checkpoint.pt` | Checkpoints from various experiments along the way |
| `*.png` | Training curves, overfitting plots, heatmaps |

## Getting it running

**1. Clone it.** The model weights use Git LFS, so install that first or you'll just get tiny placeholder files instead of the actual `.pt` files.

```bash
git lfs install
git clone https://github.com/Seladorr/Haar-Accelerated-Deepfake-Detection-System.git
cd Haar-Accelerated-Deepfake-Detection-System
```

**2. Install the stuff.** We used Python 3.9.

```bash
python -m venv venv
venv\Scripts\activate          # on Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
```

Heads up: OpenCV 5 moved the Haar Cascade stuff out of the main package, so `requirements.txt` pins `opencv-python<5`. Don't upgrade it or the face detection breaks.

The notebooks will use your GPU if you have one (CUDA or Apple Silicon), otherwise CPU. Training on CPU will take forever, but predicting is fine.

**3. Get the videos.** We used 1,000 real and 1,000 fake videos from **FaceForensics++**. Put them in the project folder like this:

```
real_yt/   ← real videos (.mp4)
fake_yt/   ← fake videos (.mp4)
```

**4. Open Jupyter** from the project folder (all the paths in the notebooks are relative):

```bash
jupyter notebook
```

### Step 1: crop the faces

Run everything in `New_preprocessing.ipynb`. It goes through each video, finds the face in each frame, crops it to 112×112 and saves a new face-only video into `real_yt_face_videos/` and `fake_yt_face_videos/`. It skips videos it's already done, so you can stop and restart it.

### Step 2: train

Run `New_Model_and_train.ipynb` top to bottom. What we used:

- 80/20 split → 1,600 videos to train on, 400 to test on
- 10 frames per video, batch size 4
- Adam, learning rate `1e-5`, weight decay `1e-5`, 20 epochs
- Labels: `1` = real, `0` = fake

It saves `checkpoint.pt` after every epoch. If you just want to use our weights, skip the training cell and run the "Load a saved checkpoint" one instead.

### Step 3: predict

Run `New_Predict.ipynb`. It loads `final_model/checkpoint3.pt`. To try your own videos, change `path_to_videos`:

- normal videos → `crop_faces=True` (it'll find the faces for you)
- videos you already ran through step 1 → `crop_faces=False`

You'll get REAL/FAKE, a confidence score, and the heatmap. At the end there's a section that runs the original model on the same videos so you can compare.

What it looks like:

```
./real_yt/130.mp4       confidence 98.96  → REAL   ✓
fake_yt/042_084.mp4     confidence 99.03  → FAKE   ✓
fake_yt/295_099.mp4     confidence 57.52  → REAL   ✗  (this one fools it)
```

## How well does it work?

Honestly, okay but not amazing. On the 400 test videos:

|                 | said Fake | said Real |
|---|---|---|
| **actually Fake** | 129 | 71 |
| **actually Real** | 35 | 165 |

That's **73.5%** accuracy (best single epoch during training hit 76%). It's pretty good at recognising real videos, but lets a fair number of fakes through. The training curves (`New-overfit-1.png`, `New-Overfit2.png`) show it starts overfitting after about 3 epochs. With more data or better augmentation it would probably do better; we just ran out of time (and GPU).

The main win was speed: the Haar Cascade preprocessing was way faster than `face_recognition`, and we didn't need to install any extra deep learning libraries just to find faces.

## Things it's bad at

- Faces that aren't facing the camera. Haar Cascade just doesn't find them.
- Videos with more than one person. We only take the first face it finds.
- Other kinds of deepfakes. We only trained on a small slice of FaceForensics++, so who knows how it does on anything else.

## Stuff we built on

- Hu et al., *Squeeze-and-Excitation Networks*, CVPR 2018
- Woo et al., *CBAM: Convolutional Block Attention Module*, ECCV 2018
- Viola & Jones, *Rapid Object Detection using a Boosted Cascade of Simple Features*, CVPR 2001
- Rössler et al., *FaceForensics++*, ICCV 2019
