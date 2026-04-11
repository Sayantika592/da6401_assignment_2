# DA6401 Assignment 2 – Multi-Task Perception Pipeline

## Links
- GitHub Repository: ADD_YOUR_GITHUB_LINK_HERE  
- W&B Report: https://wandb.ai/me22b190-indian-institute-of-technology-madras/oxford_pet_multitask/reports/DA6401-Assignment-2--VmlldzoxNjQ4OTQ5Ng?accessToken=rpmc2ebz0b82skj1hlvzdpwngi82pk9i2x62wq0u9jo7sw65cn791q9h85x70ir1

---

## Overview
This project implements a unified multi-task learning pipeline for pet images using a VGG11-based architecture. The model performs:

1. Image Classification – Predicts one of 37 pet breeds  
2. Object Localization – Predicts bounding box coordinates  
3. Semantic Segmentation – Generates pixel-wise trimap masks  

All tasks share a common VGG11 encoder backbone, with separate task-specific heads.

---

## Model Architecture

### Shared Backbone
- Custom implementation of VGG11  
- Includes Batch Normalization and Custom Dropout  

### Task Heads
- Classification Head: Fully connected layers  
- Localization Head: Regression (x_center, y_center, width, height)  
- Segmentation Head: U-Net style decoder with skip connections  

---

## Key Features

- VGG11 implemented from scratch  
- Custom Dropout layer (no built-in dropout used)  
- Custom IoU Loss for localization  
- U-Net style segmentation with transposed convolutions  
- Unified multi-task forward pass  
- Experiment tracking using Weights & Biases  

---

## Training and Evaluation

### Classification
- Metrics: Accuracy, Loss  

### Localization
- Metric: IoU Loss  

### Segmentation
- Metrics:
  - Pixel Accuracy  
  - Dice Score  

---

## Observations

- Batch Normalization improved training stability and convergence speed  
- Dropout reduced overfitting in fully connected layers  
- Fine-tuning the encoder improved multi-task performance  
- Pixel Accuracy remained high due to class imbalance, while Dice Score better reflected segmentation quality  

---

## Challenges

- Task interference due to shared backbone  
- Localization instability in complex backgrounds  
- Segmentation degradation in low-contrast or cluttered scenes  

---

## How to Run

### Install dependencies
```bash
pip install -r requirements.txt
python train.py
python inference.py

'''
.
├── checkpoints/
├── data/
│   └── pets_dataset.py
├── losses/
│   └── iou_loss.py
├── models/
│   ├── classification.py
│   ├── localization.py
│   ├── segmentation.py
│   ├── multitask.py
│   ├── vgg11.py
│   └── layers.py
├── inference.py
├── train.py
├── requirements.txt
└── README.md
'''
Name: Sayantika Chakraborty, Roll No.: ME22B190
