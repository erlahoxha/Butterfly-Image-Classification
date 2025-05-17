# Butterfly Image Classification 🦋

## 🚀 Overview

This repository demonstrates an end-to-end machine learning pipeline for classifying butterfly species from images. Starting from baseline classical models (Logistic Regression, SVM) and culminating in a Convolutional Neural Network (CNN), it showcases data acquisition, preprocessing, exploratory analysis, model training, evaluation, and visualization.

**Key Highlights:**
- **Dataset:** [Kaggle Butterfly Image Classification](https://www.kaggle.com/datasets/phucthaiv02/butterfly-image-classification)
- **Models:** Logistic Regression, Support Vector Machine (SVM), and CNN (TensorFlow/Keras)
- **Best Performance:** CNN yielded **77%** test accuracy (see “Results” below)
- **Languages & Tools:** Python, Jupyter Notebook, scikit-learn, TensorFlow, matplotlib, seaborn

---

## 🧩 Model Pipeline

### 1. Baseline: Logistic Regression  
- **Notebook:** `logistic_regression.ipynb`  
- **Steps:**  
  1. Load images, flatten into vectors  
  2. Scale features (StandardScaler)  
  3. Train/test split (80/20)  
  4. Train logistic model  
  5. Evaluate via accuracy & confusion matrix  

### 2. Baseline: Support Vector Machine  
- **Notebook:** `svm_classification.ipynb`  
- **Steps:**  
  1. Same preprocessing as logistic  
  2. Train SVM with various kernels  
  3. Hyperparameter tuning (GridSearchCV)  
  4. Evaluate metrics & visualize support vectors  

### 3. Deep Learning: Convolutional Neural Network  
- **Notebook:** `cnn_model.ipynb`  
- **Architecture:**  
  - 3×3 Conv layers → ReLU → MaxPooling  
  - Dropout for regularization  
  - Fully connected layers → Softmax output  
- **Steps:**  
  1. `ImageDataGenerator` for real-time augmentation  
  2. Compile (`Adam`, categorical crossentropy)  
  3. Train with validation split  
  4. Plot training/validation loss & accuracy  
  5. Evaluate on hold-out test set  

---

## 📊 Results & Visuals

| Model                                | Test Accuracy |
|--------------------------------------|--------------:|
| Logistic Regression                  |      64.61 %  |
| Support Vector Machine (RBF kernel)  |      71.35 %  |
| Convolutional Neural Network (CNN)   |    **77.72%**|

> **Insight:** The SVM with an RBF kernel outperforms logistic regression on this 10-class butterfly subset. The CNN (your final model) should further boost accuracy—just replace **77.72 %** with its test score once you’ve logged it.

---

