# Butterfly Image Classification 🦋

## 🚀 Overview

This repository demonstrates an end-to-end machine learning pipeline for classifying butterfly species from images. Starting from baseline classical models (Logistic Regression, SVM) and culminating in a Convolutional Neural Network (CNN), it showcases data acquisition, preprocessing, exploratory analysis, model training, evaluation, and visualization.

**Key Highlights:**
- **Dataset:** [Kaggle Butterfly Image Classification](https://www.kaggle.com/datasets/phucthaiv02/butterfly-image-classification)
- **Models:** Logistic Regression, Support Vector Machine (SVM), and CNN (TensorFlow/Keras)
- **Best Performance:** CNN yielded **XX%** test accuracy (see “Results” below)
- **Languages & Tools:** Python, Jupyter Notebook, scikit-learn, TensorFlow, matplotlib, seaborn

---

## 📂 Repository Structure

```text
├── assets/                  # Optional: images for README (e.g. banner.png)
├── data/                    # Downloaded Kaggle dataset here
│   ├── train/               # Training images (subfolders per species)
│   └── test/                # Testing images (subfolders per species)
├── notebooks/
│   ├── 1_logistic_regression.ipynb   # Baseline logistic regression
│   ├── 2_svm_classification.ipynb    # Baseline SVM
│   └── 3_cnn_model.ipynb             # Final CNN pipeline
├── requirements.txt         # Python dependencies
└── README.md                # You are here!
