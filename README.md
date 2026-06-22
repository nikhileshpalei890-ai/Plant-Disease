---
title: Plant Disease Detector
emoji: 🌿
colorFrom: green
colorTo: green
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
---

# 🌿 Plant Disease Detector

A deep learning web app that classifies plant leaf diseases from images.

## Model
- **Architecture:** MobileNetV2 (Transfer Learning)
- **Training:** PlantVillage Dataset → PlantDoc Domain Adaptation
- **Classes:** 38 plant-disease combinations
- **Accuracy:** ~95% on PlantVillage test set

## How to Use
1. Upload a clear photo of a plant leaf
2. Click **Detect Disease**
3. Get the disease name and confidence score

## Tech Stack
- TensorFlow / Keras
- Streamlit
- Hugging Face Spaces

## Project
Field Work Project — 4th Semester  
B.Sc. Computer Science  
Utkal University
