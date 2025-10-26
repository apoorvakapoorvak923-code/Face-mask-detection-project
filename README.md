# Face-mask-detection-project
Overview

This project implements a real-time face mask detection system using Convolutional Neural Networks (CNN) and OpenCV. It detects whether a person is wearing a mask or not and provides an alert sound if a mask is not detected.

This system can be useful in public areas to encourage mask-wearing during pandemics or for monitoring safety compliance.

Features

Detects faces in real-time using a webcam.

Predicts mask-wearing status (Mask or No Mask) using a trained CNN model.

Plays an alert sound when a face without a mask is detected.

Lightweight and easy to run.

Folder Structure
FACE-MASK-DETECTION/
│
├── data/                   # Contains training/testing images (with_mask/without_mask)
├── venv/                   # Python virtual environment
│   ├── Include/
│   ├── Lib/
│   ├── Scripts/
│   └── share/
├── pyvenv.cfg              # Virtual environment config
├── alert.mp3               # Sound alert when mask is not detected
├── detect_mask_live.py     # Real-time mask detection script
├── mask_detector.h5        # Trained CNN model
└── train_mask_detector.py  # Script to train your mask detection model

Requirements

Python 3.10+

TensorFlow

OpenCV

NumPy

Playsound

Install dependencies using:

pip install -r requirements.txt


requirements.txt:

tensorflow
opencv-python
numpy
playsound

Usage
1. Train the Model
If you want to retrain the model:
python train_mask_detector.py
Make sure the data/ folder contains subfolders:
with_mask/ – images of people wearing masks
without_mask/ – images of people without masks
The trained model will be saved as mask_detector.h5.

2. Run Real-Time Detection
python detect_mask_live.py
A webcam window will open.
Faces will be detected and labeled as Mask or No Mask.
Plays alert.mp3 if a face without a mask is detected.
Press q to quit.
Notes
Ensure mask_detector.h5 and alert.mp3 are in the same folder as the detection script.
The model input size is 150x150 pixels, so resizing is handled in the script.
The project uses Haar Cascade Classifier for face detection and CNN for mask classification.
Author
Apoorva K
License
This project is open-source. Feel free to use and modify it for educational purposes
