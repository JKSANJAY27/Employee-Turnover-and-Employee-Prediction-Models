# 🧠 Employee Turnover and Attrition Prediction

This project presents an end-to-end data analysis and machine learning pipeline to **predict employee turnover** (whether an employee will leave the company) and **attrition** (an employee’s voluntary/involuntary exit). Two datasets are used to train predictive models, analyze employee behavior, and identify key factors affecting attrition.

---

## 📁 Project Structure

├── HR.csv # Dataset 1 - Turnover Data
├── Employee-Attrition.csv # Dataset 2 - IBM HR Analytics Data
├── notebook.ipynb # Jupyter Notebook with all code and visualizations
└── README.md # This file

---

## 📊 Datasets

1. **HR.csv**  
   - Contains employee satisfaction, evaluation, work hours, accidents, promotions, departments, and salary.  
   - Target variable: `left` (1 = employee left, 0 = stayed)

2. **Employee-Attrition.csv**  
   - IBM’s HR Analytics Employee Attrition data.  
   - Richer dataset with 35 features.  
   - Target variable: `Attrition` (Yes/No)

---

## 🧰 Tools and Libraries

- Python 3.x
- `pandas`, `numpy`
- `matplotlib`, `seaborn`, `plotly`
- `sklearn`, `imblearn`
- `RandomForestClassifier`, `LogisticRegression`
- `RFE`, `SMOTE`

---

## 🔍 Key Steps

### 1. **Turnover Prediction (HR.csv)**

- **Preprocessing**
  - Renamed columns, merged similar departments
  - One-hot encoding for categorical variables

- **Feature Selection**
  - Used Recursive Feature Elimination (RFE) with Logistic Regression
  - Selected top 10 features

- **Modeling**
  - Compared Logistic Regression and Random Forest models
  - Evaluated using accuracy, confusion matrix, ROC-AUC

- **Results**
  - Logistic Regression Accuracy: **77%**
  - Random Forest Accuracy: **97.9%**
  - Key Feature Importance (Random Forest):
    - `satisfaction_level`: **50.39%**
    - `time_spend_company`: **26.51%**

---

### 2. **Attrition Prediction (Employee-Attrition.csv)**

- **Exploratory Data Analysis**
  - KDE plots to analyze key relationships (e.g., Age vs Total Working Years)
  - Pearson correlation heatmap for numerical features

- **Preprocessing**
  - One-hot encoding of categorical variables
  - Feature scaling and label transformation

- **Handling Imbalance**
  - Applied **SMOTE** to balance the dataset

- **Modeling**
  - Trained a **Random Forest Classifier**
  - Tuned with custom hyperparameters

- **Results**
  - Accuracy: **85.3%**
  - Precision/Recall for class `1` (Attrition): **0.57 / 0.49**
  - Feature importance visualized using interactive Plotly chart

---

## 📈 Visualizations

- Heatmaps (confusion matrix, feature correlation)
- KDE plots for feature relationships
- ROC curves for both models
- Plotly scatter for feature importance

---

## ✅ Key Insights

- **Satisfaction level**, **evaluation score**, and **time at company** are critical indicators for turnover.
- For attrition, **work-life balance**, **job involvement**, and **distance from home** have moderate predictive power.
- **Random Forest models** consistently outperform Logistic Regression, especially in non-linear and imbalanced data scenarios.

---
