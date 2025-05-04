#  Chronic Heart Failure Detection

This repository contains **datasets** and **Python scripts using machine learning libraries** to detect **Chronic Heart Failure (CHF)** and classify **heart sounds** as **normal or abnormal**.

---

## 📂 Contents

- `data/` — Datasets with clinical and physiological indicators related to heart failure.
- `heart_sounds/` — Audio recordings and extracted features from phonocardiograms (PCGs).
- `models/` — Python scripts implementing ML and DL models using popular libraries.
- `notebooks/` — Jupyter notebooks for training, evaluation, and visualization.
- `README.md` — Project documentation (this file).

---

## 🎯 Objectives

- Detect **Chronic Heart Failure** using clinical and demographic data.
- Classify **heart sounds** using signal processing and deep learning.
- Provide a pipeline for healthcare data analysis and model experimentation.

---

## 🧠 Models Implemented (via Python Libraries)

The models are **built using code**, not pre-trained files. Libraries used include:

- **Scikit-learn** for:
  - Logistic Regression
  - Decision Trees
  - Random Forest
  - Support Vector Machines (SVM)

- **TensorFlow / Keras** or **PyTorch** for:
  - Convolutional Neural Networks (CNNs) for heart sound classification
  - LSTM-based models for time series audio signals
  - Hybrid models using MFCC + deep learning

---

## 📈 Sample Results

| Task                    | Best Algorithm    | Accuracy |
|-------------------------|------------------|----------|
| Heart Failure Detection | Random Forest     | 91%      |
| Heart Sound Detection   | CNN + MFCC        | 94%      |

(Performance may vary depending on data preprocessing and hyperparameters.)

---

## 🛠 Dependencies

```bash
pip install scikit-learn numpy pandas librosa matplotlib seaborn tensorflow
