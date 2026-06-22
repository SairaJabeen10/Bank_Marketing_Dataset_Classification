# Multi-Model Ensemble for Noisy Data Classification

## Overview

This project implements a complete Machine Learning pipeline for classifying customer responses using the **Bank Marketing Dataset**. The dataset contains missing values, noisy entries, and categorical features, making it a suitable real-world classification problem.

The project focuses on data preprocessing, model training, ensemble learning, and performance evaluation to improve classification accuracy and robustness against noisy data.

---

## Objectives

* Handle missing and noisy data effectively.
* Preprocess categorical and numerical features.
* Train and compare multiple supervised learning algorithms.
* Apply ensemble learning techniques to improve performance.
* Analyze classification results using evaluation metrics and confusion matrices.
* Reduce overfitting and improve model generalization.

---

## Dataset

**Dataset:** Bank Marketing Dataset

The dataset contains customer information and marketing campaign details used to predict whether a customer will subscribe to a term deposit.

### Features

* Demographic information
* Job type
* Marital status
* Education
* Contact information
* Previous campaign outcomes
* Other banking-related attributes

### Target Variable

* `y` → Customer subscription decision (Yes/No)

---

## Data Preprocessing

### 1. Data Loading

The dataset is loaded using Pandas with a semicolon (`;`) separator.

```python
import pandas as pd

df = pd.read_csv("bank.csv", sep=";")
```

### 2. Handling Missing and Noisy Data

* Replaced `"unknown"` values with `NaN`
* Removed incomplete records using `dropna()`

### 3. Feature Encoding

Categorical features were converted into numerical values using **Label Encoding**.

### 4. Feature Scaling

All features were standardized using **StandardScaler** to ensure consistent feature ranges.

### 5. Train-Test Split

The dataset was divided into:

* Training Set: 80%
* Testing Set: 20%

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

---

## Machine Learning Models

### 1. Decision Tree Classifier

* Controlled depth (`max_depth=5`)
* Captures non-linear relationships
* Prone to overfitting without regularization

### 2. Gaussian Naive Bayes

* Probabilistic classifier
* Robust against noisy data
* Assumes feature independence

### 3. Support Vector Machine (SVM)

* Uses RBF Kernel
* Effective in high-dimensional spaces
* Requires feature scaling

---

## Ensemble Learning

### Bagging Classifier

A Bagging ensemble was implemented using **50 Decision Trees**.

Benefits:

* Reduces variance
* Improves robustness
* Minimizes overfitting
* Produces more stable predictions

```python
from sklearn.ensemble import BaggingClassifier
```

---

## Evaluation Metrics

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

```python
from sklearn.metrics import classification_report
```

---

## Results Summary

| Model            | Performance                                     |
| ---------------- | ----------------------------------------------- |
| Decision Tree    | Good performance but susceptible to overfitting |
| Naive Bayes      | Consistent and noise-resistant                  |
| SVM              | High classification accuracy                    |
| Bagging Ensemble | Best overall performance                        |

### Key Findings

* Feature scaling significantly improved SVM performance.
* Naive Bayes remained stable even with noisy data.
* Decision Trees required depth control to avoid overfitting.
* Bagging Ensemble achieved the most reliable results and reduced misclassifications.

---

## Overfitting Prevention

Techniques used:

* Limiting Decision Tree depth
* SVM regularization
* Feature scaling
* Data cleaning
* Ensemble learning (Bagging)

These methods improved model generalization on unseen data.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn

---

## Project Structure

```text
├── dataset/
│   └── bank.csv
├── notebook/
│   └── ML_Assignment.ipynb
├── report/
│   └── ML_Assignment_Report.pdf
├── README.md
└── requirements.txt
```

---

## Conclusion

This project demonstrates a complete Machine Learning workflow for solving a noisy real-world classification problem. Through effective preprocessing, multiple classification models, and Bagging ensemble learning, the system achieved improved accuracy, robustness, and generalization performance. The results highlight the importance of data cleaning, feature engineering, and ensemble methods in building reliable machine learning solutions.

---

## Author

**Saira Jabeen**
BS Artificial Intelligence
