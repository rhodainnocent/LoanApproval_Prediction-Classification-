# Loan Approval Prediction System

## Overview

### Problem Definition
Manual loan underwriting is a critical and time-consuming process for financial institutions. This project addresses the challenge by building a machine learning classification model that predicts whether a loan application will be approved (`1` = Loan Approved, `0` = Loan Not Approved) based on applicant attributes such as income, education, credit history, employment status, and loan amount. The objective is to provide lenders with a reliable tool to accelerate decision-making, improve accuracy, and minimize credit default risk.

### Machine Learning Workflow
To build and evaluate an optimal predictive system, this project follows a standard, end-to-end machine learning lifecycle:
1. **Exploratory Data Analysis (EDA):** Investigating dataset distributions, identifying data types, checking for duplicates, and spotting missing values.
2. **Data Preprocessing & Cleaning:** Imputing missing values using statistically sound methods and encoding categorical features into numerical formats.
3. **Data Splitting & Scaling:** Partitioning the dataset using stratified sampling to preserve class distributions, followed by feature scaling via `StandardScaler`.
4. **Model Training & Baseline Evaluation:** Training multiple baseline classification algorithms (Logistic Regression, Decision Tree, Random Forest, SVM) and evaluating initial performance.
5. **Hyperparameter Tuning & Optimization:** Utilizing `GridSearchCV` with 5-fold cross-validation (`cv=5`) to fine-tune model parameters (such as `n_estimators` and `max_depth`) to prevent overfitting and boost generalization.
6. **Final Evaluation & Model Selection:** Comparing final model metrics, analyzing confusion matrices, and selecting the top-performing models for production readiness.

---

## Technical Stack
* **Language:** Python
* **Data Processing & Analysis:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn (Logistic Regression, Decision Tree, Random Forest, Support Vector Classifier, StandardScaler, GridSearchCV)

---

## Data Pipeline and Processing

### Data Cleaning and Missing Value Imputation Details
During exploratory data analysis, several features were identified with missing records. These were handled using targeted imputation strategies:

| Feature | Missing Count | Imputation Method | Justification |
| :--- | :--- | :--- | :--- |
| **Gender** | 13 | Mode (Most frequent value) | Categorical variable representing binary demographics; mode replacement preserves the natural category distribution without introducing artificial categories. |
| **Married** | 3 | Mode (Most frequent value) | Binary marital status category; mode replacement is standard for low-frequency categorical gaps to maintain class proportions. |
| **Dependents** | 15 | Mode (Most frequent value) | Categorical/ordinal count of dependents; mode imputation prevents fractional values and preserves the discrete nature of household sizes. |
| **Self_Employed** | 32 | Mode (Most frequent value) | Binary employment status category; mode replacement avoids biasing the binary classification split. |
| **LoanAmount** | 22 | Median | Continuous numerical financial data; the median is robust against extreme income and loan size outliers present in the dataset. |
| **Credit_History** | 50 | Median | Binary indicator stored as float; median imputation maintains the 0/1 binary boundary while reflecting the central tendency. |
| **Loan_Amount_Term** | 14 | Median | Discrete numerical duration data; median filling avoids distortion from uncommon loan term lengths. |

### Feature Encoding and Removal
* Non-informative identifiers (`Loan_ID`) were dropped prior to training.
* Categorical columns were mapped to numerical representations (`Gender`, `Married`, `Education`, `Self_Employed`, `Loan_Status`).
* Ordinal variables (`Dependents`, `Property_Area`) were converted directly into structured integer values (including replacing `'3+'` string categories with numeric representations).

### Splitting and Scaling
* The dataset was partitioned into training (80%) and testing (20%) sets using stratified sampling (`stratify=y`) to maintain the exact ratio of approved vs. rejected loans across subsets.
* Features were standardized using `StandardScaler` to ensure uniform scaling for linear models and support vector machines.

---

## Model Evaluation, Tuning, and Results

Five machine learning models were trained and evaluated on the test set. Hyperparameter tuning via `GridSearchCV` was applied to optimize ensemble models and enhance performance.

| Model | Description | Accuracy |
| :--- | :--- | :--- |
| **Logistic Regression** | Linear classifier trained on scaled features | **86.18%** |
| **Random Forest (Tuned)** | Ensemble model optimized using GridSearchCV (n=200, max_depth=8) | **86.18%** |
| **Random Forest (Baseline)** | Baseline ensemble model with 100 trees and max depth of 3 | 85.37% |
| **Support Vector Machine (SVM)** | Non-linear classifier with Radial Basis Function (RBF) kernel | 85.37% |
| **Decision Tree** | Single decision tree classifier restricted to max depth of 3 | 84.55% |

### Key Observations
* **Logistic Regression** and the **Tuned Random Forest** achieved the highest test accuracy at **86.18%**, demonstrating that both linear and tree-based ensemble approaches handle the feature space effectively after proper tuning.
* Feature analysis and decision tree plotting revealed that **Credit History** is the single most dominant factor influencing loan approval predictions.
* Confusion matrix evaluation confirmed high precision and recall stability, validating the model's reliability for deployment.
