# 🎬 TMDB Movie Success Prediction App

[![View App](https://img.shields.io/badge/Streamlit-Live_App-green?logo=streamlit)](https://movie-success-predictor-gvawzufjqfp7ugepcgajot.streamlit.app/)

### 🚀 Live App  
[Click here to try the Movie Success Predictor](https://movie-success-predictor-gvawzufjqfp7ugepcgajot.streamlit.app/)

This is an end-to-end machine learning project that predicts whether a movie will be **successful** based on its metadata, budget, revenue, genres, and more — using data from **The Movie Database (TMDB)**.

The final output is a **Streamlit web app** that allows users to input movie attributes and get a success prediction in real time.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Demo](#demo)
- [Screenshots](#screenshots)
- [Author](#author)

---

## 🔍 Overview

This project is a complete machine learning pipeline:

- 📂 **Data Cleaning**: Merging movie metadata and credits, handling nested JSON columns, removing nulls, and preprocessing.
- 🧪 **Feature Engineering**: Creating profit, ROI, success classification, extracting top genres, cast, crew, etc.
- 📊 **EDA & Visualization**: Exploring patterns in successful movies vs. unsuccessful ones.
- 🤖 **Modeling**: Logistic Regression for binary classification.
- 🚀 **Deployment**: Interactive Streamlit app for real-time prediction.

---

## ✨ Features

- Predict if a movie is likely to be **Successful** or **Not Successful** based on key features.
- Clean, interactive **Streamlit UI**.
- Handles real-world messy data (nested JSON, missing values).
- Clearly structured machine learning pipeline.
- Feature importance and intuitive visualizations.

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
- Contains metadata about 5000+ movies: budget, revenue, cast, crew, genres, etc.

---

## 📂 Project Structure

r
