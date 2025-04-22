# 🎬 TMDB Movie Success Prediction App

[![View App](https://img.shields.io/badge/Streamlit-Live_App-green?logo=streamlit)](https://movie-success-predictor-gvawzufjqfp7ugepcgajot.streamlit.app/)

This is an end-to-end machine learning project that predicts whether a movie will be **successful** based on its metadata — such as budget, genres, cast, crew, and more — using data from **The Movie Database (TMDB)**.

The final product is a **Streamlit web app** where users can input movie features and get a prediction in real-time.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [Machine Learning](#machine-learning)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Demo](#demo)
- [Screenshots](#screenshots)
- [Author](#author)

---

## 🔍 Overview

This project follows the complete machine learning workflow:

- 📂 Data cleaning and feature engineering
- 🔍 Exploratory data analysis
- 🤖 Model building and evaluation
- 🖥️ Deployment as an interactive web app

---

## ✨ Features

- Predict whether a movie is likely to be **Successful** or **Not Successful**
- User-friendly interface via **Streamlit**
- Covers the full ML lifecycle: data cleaning → modeling → deployment
- Based on real-world dataset with budget, revenue, genres, cast, etc.

---

## 💻 Tech Stack

- **Python**
- **Pandas, NumPy**
- **Matplotlib, Seaborn**
- **Scikit-learn**
- **Streamlit**
- **Jupyter Notebook**

---

## 📁 Dataset

- **Source**: [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
- Includes metadata of over 5000 movies including:
  - Title, budget, revenue
  - Genres, cast, crew
  - Popularity, release date, runtime

---

## 🤖 Machine Learning

The model used is **Logistic Regression**, a supervised learning algorithm ideal for binary classification (Success vs Not Success).

### 🧠 Key Concepts Used:
- **Feature Engineering**: Created `profit`, `ROI`, and a `success_class` label.
- **Model**: `LogisticRegression()` from `sklearn.linear_model`
- **Train-Test Split**: 80/20
- **Confusion Matrix**: For visualizing TP, FP, FN, TN
- **Classification Report**: Precision, Recall, F1-score
- **ROC Curve & AUC**: For model performance evaluation

### ✅ Evaluation Metrics:
- **Accuracy**
- **Precision & Recall**
- **F1 Score**
- **AUC (Area Under the Curve)**

---


