# Speech Emotion Recognition (SER) using CNN

**Author:** Akash Maderna
**Student ID:** 2025A8PS1195P
**Date:** Feb 2025

## Project Overview
This project builds a 2D Convolutional Neural Network (CNN) to classify speech emotions from the RAVDESS dataset. The model takes raw audio waveforms, converts them into Log-Mel Spectrograms, and predicts one of 8 emotions: Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, and Surprised.

## Model Performance
* **Accuracy:** ~84%
* **Macro F1-Score:** ~0.83
* **Gender Bias Gap:** ~6% (Male vs Female accuracy difference)

## Training Details
To achieve high accuracy and mitigate overfitting/bias, the following strategies were used:

### 1. Data Augmentation (The "3x" Strategy)
Every audio file was processed into 3 versions to triple the dataset size:
* **Original:** Clean log-mel spectrogram.
* **Noise Injection:** Added white noise to improve robustness against real-world static.
* **Pitch Shifting (Bias Fix):** * Male voices were shifted **UP** (+2.5 steps).
    * Female voices were shifted **DOWN** (-2.5 steps).
    * *Impact:* This successfully reduced the model's tendency to confuse low-pitched male voices with "Sad" or "Calm."

### 2. Model Architecture
* **Input:** 128x130x1 Log-Mel Spectrograms.
* **Backbone:** Custom 4-block 2D CNN.
* **Regularization:** BatchNormalization after every convolution and Dropout (0.2 - 0.5) to prevent overfitting.
* **Optimization:** Adam Optimizer (LR=0.001) with `ReduceLROnPlateau`.
* **Class Balancing:** Applied `class_weights` during training to force the model to focus on under-represented classes like "Sad."

## Files in Repository
* `Notebook.ipynb`: Complete source code including EDA, Preprocessing, Training, and Evaluation.
* `predict.py`: Standalone inference script for testing new audio files.
* `final_ser_model_weighted.keras`: Saved model weights.
* `requirements.txt`: List of dependencies.

## How to Run Inference
To predict the emotion of an audio file, run the following command in your terminal:

```bash
python predict.py --file "path/to/your/audio_file.wav"
