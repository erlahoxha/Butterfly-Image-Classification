#  Butterfly Image Classification 🦋

## 🚀 Overview

Welcome! This project walks you through an end-to-end pipeline for teaching a machine to recognize ten different butterfly species from images. We start with simple, classical models—Logistic Regression and SVM—and then level up to a Convolutional Neural Network (CNN) built with TensorFlow/Keras. The result? A model that achieves **77.72 %** accuracy on unseen test images!

**What’s Inside**  
- 📊 **Dataset**: “Butterfly Image Classification” from Kaggle  
- 🛠 **Models**: Logistic Regression → Support Vector Machine → CNN  
- ⚙️ **Tools**: Python, scikit-learn, TensorFlow/Keras, matplotlib, seaborn  
- 🎯 **Top Score**: **77.72 %** test accuracy with our CNN

---

## 🧩 How It Works

### 1. Baseline: Logistic Regression  
We kick off with a straightforward approach:  
1. Load and vectorize each image  
2. Scale pixel values to standardize inputs  
3. Split into 80 % train / 20 % test sets  
4. Train a logistic regression classifier  
5. Check accuracy and examine the confusion matrix  

Results: **64.61 %** test accuracy—solid for a first pass and a great sanity check!

---

### 2. Baseline: Support Vector Machine  
Next, we let SVM take the stage:  
1. Reuse our cleaned, scaled data  
2. Fit an SVM (RBF kernel)  
3. Tune hyperparameters via grid search  
4. Evaluate performance and peek at support vectors  

Results: a notable jump to **71.35 %** accuracy—showing that a non-linear boundary helps separate these butterfly classes.

---

### 3. Deep Dive: Convolutional Neural Network  
Finally, we embrace deep learning:  
- **Architecture**: Stacks of 3×3 Conv → ReLU → MaxPooling → Dropout → Dense → Softmax  
- **Workflow**:  
  1. Real-time data augmentation with `ImageDataGenerator`  
  2. Compile using the Adam optimizer and categorical cross-entropy  
  3. Train with a built-in validation split to monitor overfitting  
  4. Visualize training/validation loss & accuracy curves  
  5. Evaluate on the hold-out test set  

Results: our champion model achieves **77.72 %** accuracy on the test images!

---

## 📊 Final Results

| Model                                | Test Accuracy |
|--------------------------------------|--------------:|
| Logistic Regression                  |       64.61 % |
| SVM (RBF kernel)                     |       71.35 % |
| Convolutional Neural Network (CNN)   | ** 77.72 % ** |

> **Insight:** Each step up—from a linear classifier to a non-linear SVM and finally to a bespoke CNN—unlocks richer feature representations. Our CNN’s **77.72%** accuracy gain shows it’s effectively learning visual hierarchies (textures, shapes, color patterns) that simpler models miss, resulting in more reliable species recognition.
---

