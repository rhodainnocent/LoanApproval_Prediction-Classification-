# Loan Approval Prediction System

## Project Overview
This project builds a machine learning classification model to predict whether a loan application will be approved (`1` = Approved, `0` = Not Approved). The goal is to help financial institutions automate loan decisions, speed up processing, and mitigate credit default risks.

## Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn

## Workflow & Methodology
1. **Data Cleaning & Imputation:** Handled missing values by filling categorical features (`Gender`, `Married`, `Dependents`, `Self_Employed`) with the mode, and numerical/binary features (`LoanAmount`, `Credit_History`, `Loan_Amount_Term`) with the median. Dropped non-informative identifiers (`Loan_ID`).
2. **Preprocessing & Encoding:** Mapped categorical variables into numerical format and partitioned the data into an 80/20 train-test split using stratified sampling to preserve class proportions. Features were standardized using `StandardScaler`.
3. **Model Training & Tuning:** Trained multiple classifiers including Logistic Regression, Decision Tree, Random Forest, and SVM. Performed hyperparameter tuning using `GridSearchCV` with 5-fold cross-validation to optimize model parameters.

## Results & Performance

| Model | Description | Accuracy |
| :--- | :--- | :--- |
| **Logistic Regression** | Linear classifier on scaled features | **86.18%** |
| **Random Forest (Tuned)** | Optimized ensemble model via GridSearchCV | **86.18%** |
| **Support Vector Machine** | Non-linear classifier with RBF kernel | 85.37% |
| **Random Forest (Baseline)** | Standard ensemble model | 85.37% |
| **Decision Tree** | Single tree-based classifier | 84.55% |

### Key Takeaways
* **Logistic Regression** and the **Tuned Random Forest** tied for the highest test accuracy at **86.18%**.
* **Credit History** emerged as the single most critical factor influencing loan approval outcomes.
