# SignScript

Real-time American Sign Language (ASL) fingerspelling and isolated sign recognition using computer vision and machine learning.

## Overview

SignScript is a Python-based computer vision application designed to recognize ASL fingerspelling and common isolated signs, converting recognized signs into English text.

The project uses Google's MediaPipe Hand Landmarker for hand tracking and OpenCV for real-time video processing. Machine learning classifiers will be trained to recognize individual letters and selected ASL signs.

## Features

### Implemented
- Real-time webcam capture using OpenCV
- Hand detection for up to two hands
- Visualization of 21 landmarks per detected hand
- Customizable hand landmark colours
- Close the application using the window's X button
- Hand landmark detection for images and video
- Normalization of hand landmarks for position, size, and handedness
- Preparation of labelled ASL alphabet features for model training

### Planned
- Recognition of 24 static ASL alphabet letters
- Recognition of 2 dynamic ASL alphabet letters (J and Z)
- Recognition of selected common ASL signs
- Automatic letter and word recognition without manual mode selection
- English text output
- Machine learning model comparisons and evaluation

## Technologies

- Python 3.11
- OpenCV
- Google MediaPipe
- NumPy
- scikit-learn
- Matplotlib
- joblib

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/myabiagi/signscript.git
cd signscript
```

### 2. Create a virtual environment

Windows (PowerShell):

```powershell
py -3.11 -m venv .venv
```

macOS:

```bash
python3.11 -m venv .venv
```

### 3. Activate the virtual environment

Windows (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS:

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Download the MediaPipe hand landmark model

The application uses Google's pretrained MediaPipe Hand Landmarker model.

The model is not included in the repository and must be downloaded separately.

Windows (PowerShell):

```powershell
New-Item -ItemType Directory -Force models

Invoke-WebRequest `
    -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" `
    -OutFile "models/hand_landmarker.task"
```

macOS:

```bash
mkdir -p models

curl -L \
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" \
    -o models/hand_landmarker.task
```

The model should be saved to:

    models/hand_landmarker.task

## Usage

Run all commands from the repository root with the virtual environment activated.

### Start SignScript

```bash
python dev.py start
```

This launches the webcam application, detects hands, and displays their landmarks.

Close the webcam window using the X button.

## Reproducing the Experiments

To reproduce Experiment 1 (KNN vs. Random Forest):

1. Download SignAlphaSet:
   https://data.mendeley.com/datasets/8fmvr9m98w/3

2. Extract the images into:
   `data/alphabet/signalphaset/`

3. Prepare features:

   ```bash
   python dev.py prepare-alphabet
   ```

4. Train and evaluate:

   ```bash
   python dev.py train-alphabet
   ```

Experiment outputs are saved in `results/alphabet/`.
