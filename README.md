# SignScript

SignScript is a computer vision project for recognizing American Sign Language (ASL) fingerspelling using a webcam, MediaPipe hand landmarks, and machine learning.

The project targets the 26-letter ASL fingerspelling alphabet:

- **24 static letters:** A–Z, excluding J and Z.
- **2 dynamic letters:** J and Z, which require motion recognition.

**Current implementation:** The 24 static letters are supported. Dynamic J/Z recognition is under development.

## Features

- Live webcam input through OpenCV
- MediaPipe detection and visualization of 21 hand landmarks
- Left- and right-hand normalization
- 108-dimensional static hand features
- Random Forest classification of 24 static letters
- Prediction smoothing using six recent predictions and four matching votes
- Stable letter display during uncertain transitions
- Pretrained classifier included; no retraining required to run the application

## Technologies

- Python
- OpenCV
- MediaPipe Hand Landmarker
- NumPy
- scikit-learn
- joblib

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/myabiagi/signscript.git
cd signscript
```

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Download the MediaPipe model

SignScript uses MediaPipe's pretrained Hand Landmarker for detecting hands.

The required model must be saved as `models/hand_landmarker.task`.

**Windows PowerShell:**

```powershell
New-Item -ItemType Directory -Force models

Invoke-WebRequest `
  -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" `
  -OutFile "models/hand_landmarker.task"
```

**macOS/Linux:**

```bash
mkdir -p models

curl -L \
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" \
  -o models/hand_landmarker.task
```

After setup, the models directory should contain:

```text
models/
├── hand_landmarker.task
└── static_classifier.joblib
```

`static_classifier.joblib` is the trained SignScript classifier included in the repository.

## Run SignScript

```bash
python dev.py start
```

Show one static ASL handshape to the webcam.

The application displays the predicted letter, detected handedness, and hand landmarks.

To reduce flickering, a letter must receive at least four matching predictions within a six-frame window. Until then, the last stable prediction remains displayed.

Press **Q** or close the webcam window to exit.

**The training datasets are not required to run the supplied classifier.**

## Reproducing Training (Optional)

The repository also includes preparation, training, and evaluation scripts.

Download the two static-letter datasets:

- [ASL-HG — Mendeley Data](https://data.mendeley.com/datasets/j4y5w2c8w9/1)
- [Pugeault & Bowden — ASL Finger Spelling Dataset A](https://empslocal.ex.ac.uk/people/staff/np331/index.php?section=FingerSpellingDataset)

Extract them into:

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

ASL-HG contains images named like `P1_A_1.jpg`, organized by letter. Pugeault Dataset A contains RGB images named `color_*.png`, organized by participant and letter.

Prepare the datasets:

```bash
python dev.py prepare-static asl-hg
python dev.py prepare-static pugeault
```

Train the static models:

```bash
python dev.py train-static knn
python dev.py train-static rf
python dev.py train-static geometric-knn
python dev.py train-static geometric-rf
python dev.py train-static final
```

`train-static final` saves the reproduced model as `models/static_classifier_retrained.joblib`, without overwriting the supplied classifier.

Evaluate the supplied final model on the held-out ASL-HG test participants:

```bash
python dev.py eval-static
```

The original evaluation produced **95.21% accuracy and 0.9501 macro-F1** on 4,800 held-out static-letter images. Retraining in a different software environment may produce different results.

The model evaluation uses individual images; it does not measure continuous real-time fingerspelling accuracy.

### Improving live webcam accuracy (domain mix)

Offline ASL-HG accuracy does not fully transfer to a laptop webcam. To close that gap:

1. Download [ASL Alphabet Hand Landmarks](https://www.kaggle.com/datasets/borisgraudt/asl-alphabet-hand-landmarks) (MediaPipe `.npy` from webcam capture).
2. Extract so letter folders sit here (A/, B/, … — J/Z are ignored for static):

```text
data/static/graudt_landmarks/
├── A/
├── B/
└── ...
```

3. Prepare + optionally capture your own hard letters (A/S, M/N, …):

```bash
python dev.py prepare-static graudt
python dev.py collect-static          # SPACE saves current letter
python dev.py prepare-static webcam
python dev.py train-static webcam-mix --install
python dev.py start
```

`webcam-mix` keeps ASL-HG/Pugeault when those caches exist, adds Graudt + your webcam landmarks (up-weighted), and `--install` replaces `models/static_classifier.joblib` for live use.

## Project Report

The project report documents the computer vision methodology, datasets, experiments, evaluation results, failure analysis, and limitations.

## References

- [ASL-HG Dataset](https://data.mendeley.com/datasets/j4y5w2c8w9/1)
- [Pugeault & Bowden ASL Finger Spelling Dataset](https://empslocal.ex.ac.uk/people/staff/np331/index.php?section=FingerSpellingDataset)
- [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
- [scikit-learn](https://scikit-learn.org/)
