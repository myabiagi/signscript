# SignScript

SignScript is a computer vision project for recognizing American Sign Language (ASL) fingerspelling using a webcam, MediaPipe hand landmarks, and machine learning.

The project targets the 26-letter ASL fingerspelling alphabet:

- **24 static letters:** A–Z, excluding J and Z.
- **2 dynamic letters:** J and Z, which require motion recognition.

**Current implementation:** Static letters and motion-based **J/Z** are supported in a browser demo. MediaPipe runs in the browser; only landmarks are sent to a local FastAPI server (no video upload).

## Features

- Browser live demo (`python dev.py start`) with webcam + landmark overlay
- MediaPipe detection of 21 hand landmarks
- Left- and right-hand normalization
- 108-dimensional static hand features
- Random Forest classification of 24 static letters
- Sequence Random Forest for J/Z strokes
- Shared motion thresholds between Python and the browser (`/api/config`)
- Pretrained classifiers included; no retraining required to run the demo

## Technologies

- Python
- FastAPI / Uvicorn
- OpenCV (dataset prepare / collect)
- MediaPipe Hand Landmarker
- NumPy
- scikit-learn
- joblib

## Getting Started

### 1. Extract the project zip

Unzip the submitted project archive and open a terminal in the extracted `signscript` folder.

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Confirm the model files

After extracting, `models/` should contain:

```text
models/
├── hand_landmarker.task           # MediaPipe (prepare / train)
├── static_classifier.joblib       # 24 static letters
└── dynamic_rf_jz_sequence.joblib  # J/Z
```

If `hand_landmarker.task` is missing:

**Windows PowerShell:**

```powershell
New-Item -ItemType Directory -Force models

Invoke-WebRequest `
  -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" `
  -OutFile "models/hand_landmarker.task"
```

**macOS:**

```bash
mkdir -p models
curl -L \
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" \
  -o models/hand_landmarker.task
```

The browser demo loads MediaPipe from a CDN; the `.task` file is only needed for dataset prepare / train.

## Run SignScript

```bash
python dev.py start
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000), click **Enable camera**, allow webcam access, and sign.

- Hold a still handshape for static letters.
- Draw a clear **J** hook or **Z** zigzag for motion letters.
- Keep the hand close, centered, and well lit.

**The training datasets are not required to run the supplied classifiers.**

## Reproducing Training

Download (in this order):

1. [ASL Dynamic Gestures — Mendeley](https://data.mendeley.com/datasets/8fmvr9m98w/3) → `ASL_dynamic.zip` (J/Z motion clips)
2. [ASL-HG — Mendeley](https://data.mendeley.com/datasets/j4y5w2c8w9/1) → `ASL_Raw_Images.zip` (static letters)
3. [Pugeault & Bowden — ASL Finger Spelling](https://empslocal.ex.ac.uk/people/staff/np331/index.php?section=FingerSpellingDataset) → **Dataset A: 5 users (easy)**

### Static datasets

Extract `ASL_Raw_Images.zip` and Dataset A into:

```text
data/
└── static/
    ├── asl_hg/
    │   ├── A/
    │   ├── B/
    │   └── ...
    └── pugeault_bowden/
        ├── A/
        │   ├── a/
        │   ├── b/
        │   └── ...
        ├── B/
        ├── C/
        ├── D/
        └── E/
```

ASL-HG uses images like `P1_A_1.jpg` by letter. Pugeault Dataset A uses `color_*.png` by participant and letter.

```bash
python dev.py prepare-static asl-hg
python dev.py prepare-static pugeault
python dev.py train-static final --install
python dev.py eval-static
```

`train-static final --install` writes `models/static_classifier.joblib`. Without `--install`, it still saves `models/static_classifier_retrained.joblib`.

Held-out ASL-HG evaluation of the shipped static model: **95.21% accuracy** and **0.9501 macro-F1** on 4,800 images.

### Dynamic datasets (J / Z)

Extract `ASL_dynamic.zip` into:

```text
data/
└── ASL_dynamic/
    ├── J/.../frame_*.jpg
    ├── Z/...
    ├── I/...
    └── D/...
```

J and Z are motion positives; I and D are used as negatives during training.

```bash
python dev.py prepare-dynamic
python dev.py train-dynamic
python dev.py eval-dynamic
```

This writes `models/dynamic_rf_jz_sequence.joblib` (metrics under `results/dynamic/`).

## Dev commands

```text
python dev.py start
python dev.py prepare-static | train-static | eval-static
python dev.py prepare-dynamic | train-dynamic | eval-dynamic
```
