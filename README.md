<div align="center">

# 🩸 AI-Based Anemia Detection System 🩸

*A Machine Learning–powered web application for early anemia risk detection*

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Flask](https://img.shields.io/badge/Flask-Web%20Framework-black)
![ML](https://img.shields.io/badge/Machine%20Learning-Random%20Forest-green)
![Status](https://img.shields.io/badge/Status-Working-success)

</div>

---

## 📌 Introduction

Anemia is a common medical condition caused by a deficiency of red blood cells or hemoglobin, leading to reduced oxygen delivery to body tissues. Traditional diagnosis requires laboratory blood tests, which may not always be easily accessible.

This project presents an **AI-based Anemia Detection System** that uses **Machine Learning techniques** to predict whether a patient is **Anemic or Non-Anemic** based on key clinical parameters. The system is deployed as a **Flask web application**, allowing users to interact with the model through a simple interface.

---

## 🎯 Project Objectives

- To develop a machine learning model for anemia prediction  
- To analyze the importance of clinical features affecting anemia  
- To handle class imbalance using data resampling techniques  
- To deploy the trained model using a Flask web application  
- To provide a user-friendly interface for prediction  

---

## 🧠 Methodology

1. **Data Collection & Preprocessing**
   - Dataset includes blood-related parameters such as hemoglobin, MCV, MCH, and gender
   - Missing values handled and features standardized

2. **Class Imbalance Handling**
   - SMOTE (Synthetic Minority Oversampling Technique) applied to balance classes

3. **Model Training**
   - Random Forest Classifier selected due to high accuracy and robustness
   - Model evaluated using accuracy and ROC-AUC metrics

4. **Model Deployment**
   - Trained model saved and integrated into a Flask application
   - Predictions generated through a web interface

---

## ⚙️ Technologies Used

- **Programming Language:** Python  
- **Web Framework:** Flask  
- **Machine Learning:** Scikit-learn  
- **Data Processing:** Pandas, NumPy  
- **Visualization:** Matplotlib  
- **Model:** Random Forest Classifier  

---

## 📊 Dataset Features

- Hemoglobin (Hb)
- Mean Corpuscular Volume (MCV)
- Mean Corpuscular Hemoglobin (MCH)
- Mean Corpuscular Hemoglobin Concentration (MCHC)
- Gender

---

## 📈 Model Performance

| Algorithm            | Accuracy |
|---------------------|----------|
| Random Forest       | ~99%     |
| Logistic Regression | ~98%     |
| Support Vector मशीन | ~90%     |
| K-Nearest Neighbors | ~87%     |

**Random Forest** achieved the best performance and was selected for deployment.

---

## 🧪 Feature Importance (Random Forest)

- Hemoglobin: Highest impact on prediction  
- Gender: Moderate contribution  
- MCH, MCV, MCHC: Supporting features  

---📊 Performance Evaluation
Model Comparison Analysis

Algorithm	Accuracy	AUC
Random Forest	99%	99%
Logistic Regression	98%	98%
SVM	90%	90%
KNN	87%	87%
Random Forest Classifier demonstrates superior performance across both metrics.

## 🖥️ System Architecture
User Input
↓
Flask Web Interface
↓
Pre-processing Module
↓
Trained ML Model
↓
Prediction Result

---

## 📂 Project Structure

```bash
Anemia_detection/
│
├── app.py
├── requirements.txt
├── utils.py
├── model/
│   └── random_forest_classifier.pkl
├── static/
│   └── style.css
├── templates/
│   └── index.html
└── README.md
