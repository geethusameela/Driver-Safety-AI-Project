# AI-Based Real-Time Driver Drowsiness and Distraction Detection System

## Project Overview

This project presents an AI-based real-time driver safety monitoring system designed to detect driver drowsiness and distraction using computer vision and machine learning.

The system uses a webcam to monitor the driver's face and extracts facial features such as Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), and head movement. These features are then analysed using rule-based methods and a Random Forest machine learning model to classify the driver's current state.

The system provides real-time safety information including driver status, risk score, machine learning prediction, confidence, eye status, mouth status, head direction, blink count, and yawn count.

## Main Objectives

- Detect driver drowsiness in real time.
- Detect driver distraction through head direction.
- Extract facial features using MediaPipe Face Mesh.
- Calculate Eye Aspect Ratio (EAR) for eye closure analysis.
- Calculate Mouth Aspect Ratio (MAR) for yawning detection.
- Develop a machine learning classification model.
- Compare different machine learning algorithms.
- Integrate the trained model into a real-time monitoring system.
- Provide a visual driver safety monitoring interface.

## Technologies Used

- Python
- OpenCV
- MediaPipe
- NumPy
- Pandas
- Scikit-learn
- Joblib
- Streamlit
- Matplotlib

## System Architecture

```text
Webcam
   ↓
Face Detection
   ↓
MediaPipe Face Landmarks
   ↓
Feature Extraction
   ├── EAR
   ├── MAR
   ├── Head Horizontal
   └── Head Vertical
   ↓
Rule-Based Analysis + Machine Learning
   ↓
Driver State Classification
   ├── ALERT
   ├── DROWSY
   └── DISTRACTED
   ↓
Risk Assessment
   ↓
Real-Time Safety Dashboard
