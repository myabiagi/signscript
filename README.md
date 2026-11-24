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
- Hand landmark detection for both images and video
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
- scikit-learn (planned for classification)
- Matplotlib (planned for evaluation)

## Installation

### 1. Clone the repository

```powershell
git clone https://github.com/myabiagi/signscript.git
cd signscript
```

### 2. Create a virtual environment

```powershell
py -3.11 -m venv .venv
```

### 3. Activate the virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 5. Download the MediaPipe hand landmark model

The application uses Google's pretrained MediaPipe Hand Landmarker model.

Create the `models` directory:

```powershell
New-Item -ItemType Directory -Force models
```

Download the model:

```powershell
Invoke-WebRequest `
    -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" `
    -OutFile "models/hand_landmarker.task"
```

## Alphabet Dataset

SignScript uses the SignAlphaSet dataset for static ASL alphabet recognition.

Dataset source:
https://data.mendeley.com/datasets/8fmvr9m98w/3

Download `SignAlphaSet.zip` and extract the images into:

  data/alphabet/signalphaset/

The extracted directory should contain 26 folders, labelled A through Z, with 1,000 JPG images per letter.

The static alphabet classifier uses 24 letters, excluding J and Z, which require motion recognition.

The dataset is not included in this repository and must be downloaded separately.

## Alphabet Feature Preparation

After downloading and extracting SignAlphaSet, run:

```powershell
python src/prepare_alphabet.py
```

This script:
- Randomly samples 200 images per static letter using seed 42
- Extracts 21 hand landmarks using MediaPipe
- Normalizes the landmarks into 63 numerical features
- Records hand detection statistics
- Saves the processed features to `data/processed/alphabet_features.npz`

The generated features will be used to train and evaluate KNN and Random Forest classifiers.

## Usage

With your virtual environment activated, run the application from the repository root:

```powershell
python src/main.py
```

The application opens a webcam window and displays detected hand landmarks.
