
# Employee Performance Predictor using Data Analytics & Machine Learning

## Project Overview

This project is an AI-powered HR analytics system that predicts employee performance based on key workplace attributes such as experience, salary, training hours, attendance, and productivity metrics.

It helps HR teams make data-driven decisions for promotions, training, and performance improvement strategies.

---

## Problem Statement

Manual employee evaluation is:
- Time-consuming
- Biased
- Inconsistent

This project automates performance prediction using Machine Learning.

---

## Solution

We built a machine learning system that:
- Generates synthetic HR data
- Analyzes employee behavior patterns
- Trains a Random Forest model
- Predicts employee performance in real-time
- Provides HR recommendations

---

## Tech Stack

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Streamlit (UI)
- Joblib (model saving)

---

## Project Workflow

Data Generation → EDA → Feature Engineering → Model Training → Evaluation → Prediction System → Streamlit Dashboard

---

## Features

- Synthetic HR dataset generation
- Exploratory Data Analysis (EDA)
- Machine Learning classification model
- Real-time performance prediction
- Interactive Streamlit dashboard
- HR decision recommendations

---

## Model Performance

- Algorithm: Random Forest Classifier
- Evaluation: Accuracy + Classification Report
- Improved using stratified sampling

---

## How to Run the Project

### 1. Install dependencies
```bash
pip install -r requirements.txt
```
### 2. Generate dataset
```bash
python src/data_generator.py
```
### 3. rain model
```bash
python src/model.py
```
### 4. Run Streamlit app
```bash
streamlit run src/app.py
```

---

## Business Impact

- Helps HR teams identify high performers
- Supports promotion decisions
- Identifies training needs
- Reduces manual evaluation bias

---

## Future Improvements

- Deploy on cloud (Streamlit Cloud / Render)
- Add real HR dataset
- Add employee attrition prediction
- Integrate dashboard analytics
- Add role-based access system

---

## Author

Bindu P
