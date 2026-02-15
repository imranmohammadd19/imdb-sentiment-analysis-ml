# 🎬 IMDb Sentiment Analysis (Machine Learning Project)

## 📌 Project Overview

This project builds an end-to-end Machine Learning pipeline to classify movie reviews as **Positive** or **Negative** using the IMDb 50K Movie Reviews dataset.

The system uses:
- SQLite (for database storage)
- TF-IDF (for text feature extraction)
- Logistic Regression & Naive Bayes (for classification)
- Scikit-learn (for ML pipeline)
- Joblib (for model persistence)

---

## 🧠 Problem Statement

Given a movie review (text), predict whether the sentiment is:

- ✅ Positive
- ❌ Negative

## 📊 Dataset

- **IMDb 50K Movie Reviews**
- 50,000 labeled reviews
- Balanced dataset:
  - 25,000 Positive
  - 25,000 Negative

The dataset is stored in a SQLite database for structured querying.

---

## 🏗️ Project Architecture
IMDb CSV
↓
Load into SQLite database
↓
Load from SQL into Pandas
↓
Train/Test Split
↓
TF-IDF Vectorization (5000 features)
↓
Logistic Regression Training
↓
Model Evaluation
↓
Save Model (.pkl files)
↓
Prediction via CLI


---

## 🔬 Machine Learning Approach

### 1️⃣ Data Preprocessing
- Convert sentiment labels to numeric (Positive = 1, Negative = 0)
- Split dataset using stratified train-test split (80/20)

### 2️⃣ Feature Engineering
- TF-IDF Vectorization
- Maximum 5000 most important words
- Converts text → numerical feature vectors

### 3️⃣ Model Training
Two models were tested:
- Logistic Regression
- Multinomial Naive Bayes

### 4️⃣ Evaluation
- Accuracy Score
- Confusion Matrix
- 5-Fold Cross Validation

---

## 📈 Results

- Logistic Regression Accuracy: ~0.88–0.90
- Strong baseline performance for classical ML
- Balanced performance across both classes



