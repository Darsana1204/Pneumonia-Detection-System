# Pneumonia Detection from Chest X-Ray Images using Image Processing and Deep Learning

This project focuses on automating pneumonia detection from chest X-ray images using deep learning and image processing techniques.  
The aim is to support early diagnosis by improving detection accuracy and interpretability through AI-based solutions.



# Project Overview
This study implements and compares multiple deep learning and machine learning models — **CNN**, **SVM**, and **XGBoost** — for pneumonia classification.  
To enhance model performance, **Contrast Limited Adaptive Histogram Equalization (CLAHE)** was applied to improve feature visibility in chest X-ray images.



# 🧠 Model Pipeline

1. **Preprocessing and Image Enhancement**
   - Grayscale conversion and resizing  
   - CLAHE for local contrast enhancement  
   - Noise reduction and normalization  

2. **Data Augmentation**
   - Random rotation, zoom, and flipping using **Albumentations**  
   - Balanced dataset for improved generalization  

3. **Model Development**
   - Custom **CNN** designed for binary classification  
   - Classical models: **SVM** and **XGBoost** for feature-based learning  
   - Comparative analysis of all models based on accuracy, precision, recall, F1-score, and AUC  

---

## 🧩 Technologies Used
**Languages:** Python  
**Libraries:** TensorFlow, Keras, OpenCV, scikit-learn, Albumentations, Pillow, NumPy, Pandas, Matplotlib, Seaborn, XGBoost  
**Environment:** VS Code
**Version Control:** Git, GitHub  

---

## 📊 Results

## 🔹 Model Comparison
| Model       | Accuracy (%)  | Precision (%) | Recall (%) | F1-Score (%) | AUC  |
|-------------|---------------|---------------|------------|--------------|------|
| **SVM**     | 80.93         | 77.05         | 98.97      | 86.64        | 0.95 |
| **CNN**     | 80.77         | 76.89         | 98.97      | 86.54        | 0.94 |
| **XGBoost** | 79.65         | 76.04         | 98.46      | 85.81        | 0.93 |

> 🏆 **Best Performing Model:** SVM — achieved highest accuracy (80.93%) and balanced precision-recall performance (F1: 86.64%).

---

### 🔹 Image Enhancement Metrics (CLAHE)
| Metric        | Before Enhancement | After Enhancement | 
|---------------|--------------------|-------------------|
| **MSE**       | 5.5745             | 18.6693           |
| **PSNR (dB)** | 41.4117            | 35.9772           |
| **SSIM**      | 0.9452             | 0.8834            | 

> CLAHE significantly improved local contrast and feature detail visibility, enhancing pneumonia region clarity in X-rays.

---

## 💡 Key Highlights
- Applied **prompt engineering** to optimize model generation and fine-tuning using AI tools such as **Copilot** and **ChatGPT**.  
- Compared classical ML and deep learning models to identify optimal architecture for X-ray classification.  
- Enhanced interpretability using **SHAP** and visualization techniques.  
- Balanced dataset and preprocessing pipeline improved model robustness and generalization.  

---

## 📂 Dataset
- Source: [Kaggle - Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)  
- Classes: **Normal**, **Pneumonia** 
- Total Images: ~6,000 (Train + Test + validation)

---

## 🔒 Repository Privacy
This repository is kept **private** while the project is under IEEE conference submission review.  
Access can be granted upon request for academic or evaluation purposes.

---

## ✨ Author
**Darsana C K**  
B.Tech in Electronics and Communication Engineering  
[LinkedIn](https://www.linkedin.com/in/darsana-c-k-279094244/) • [Email](mailto:darsanakrishnakumar1204@gmail.com)

---

> “AI-assisted diagnosis holds the potential to empower healthcare professionals and make early detection more accessible.”
