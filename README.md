# 🧘 Yoga Pose Classifier

A deep learning web application that classifies yoga poses from images using a MobileNetV2 transfer learning model trained with TensorFlow.

## 📋 Project Overview

This project was built as part of a Deep Learning exam at ESPRIT. It classifies yoga poses into two categories:
- **Downward Dog** (Adho Mukha Svanasana)
- **Goddess Pose** (Utkata Konasana)

## 🏆 Model Performance

| Model | Validation Accuracy |
|---|---|
| Basic CNN (from scratch) | 58.2% |
| Deep CNN (from scratch) | 55.7% |
| **MobileNetV2 (Transfer Learning)** | **98.7%** ✅ |

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- Streamlit
- MobileNetV2 (pretrained on ImageNet)

## 🚀 Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Project Structure

```
yoga_app/
├── app.py               ← Streamlit web application
├── yoga_model.keras     ← Trained MobileNetV2 model
├── requirements.txt     ← Python dependencies
└── README.md            ← This file
```

## 📓 Notebook

The full Jupyter notebook with all training steps, EDA, and model evaluation is included: `yoga_classification.ipynb`

## 👨‍🎓 Author

Deep Learning Project — ESPRIT School of Engineering
