# Decision Tree Classifier - Customer Purchase Prediction App

A Machine Learning web application built using **Streamlit** and **Decision Tree Classifier** to predict whether a customer is likely to purchase a product based on customer behavior and financial details.

## Live Demo

https://decision-tree-classifier1.streamlit.app/

---

# Project Overview

This project demonstrates a complete **Machine Learning Classification workflow** including:

- Data Collection
- Data Cleaning
- Outlier Detection and Treatment
- Feature Encoding
- Model Training
- Model Evaluation
- Model Deployment using Streamlit

The application predicts whether a customer is:

- Likely to Purchase
- Not Likely to Purchase

based on customer information.

Decision Trees are highly interpretable machine learning algorithms widely used for classification and decision-making systems. :contentReference[oaicite:0]{index=0}

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Seaborn
- Pickle

---

# Machine Learning Algorithm

## Decision Tree Classifier

Decision Tree is a supervised machine learning algorithm used for classification and regression tasks.

The model works by:
- Splitting data into branches
- Creating decision rules
- Predicting outcomes based on conditions

In this project:
- `1` → Customer Purchases Product
- `0` → Customer Does Not Purchase Product

Decision Trees are popular because they are easy to understand and visualize. :contentReference[oaicite:1]{index=1}

---

# Dataset Information

The dataset contains:

- 5000 rows
- 8 columns
- Balanced classes

## Features

| Feature | Description |
|---|---|
| Age | Customer age |
| Salary | Annual salary |
| CreditScore | Customer credit score |
| Gender | Male/Female |
| SpendingScore | Spending behavior score |
| MembershipYears | Years of membership |
| PreviousPurchases | Previous purchases count |
| Purchased | Target variable |

---

# Project Workflow

## 1. Data Preprocessing

- Removed duplicate rows
- Checked missing values
- Statistical analysis using `describe()`

---

## 2. Outlier Detection using IQR

Outliers were detected using the IQR (Interquartile Range) method.

Formula:

:contentReference[oaicite:2]{index=2}

Lower Bound:

:contentReference[oaicite:3]{index=3}

Upper Bound:

:contentReference[oaicite:4]{index=4}

---

## 3. Outlier Treatment

Detected outliers were treated by replacing extreme values with lower and upper bounds using NumPy operations.

---

## 4. Feature Encoding

Categorical data (`Gender`) was converted into numerical form using:

```python
from sklearn.preprocessing import LabelEncoder
