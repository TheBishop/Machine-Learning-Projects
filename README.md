# Credit Risk Prediction Using Ensemble Learning

## Project Overview

This project builds and evaluates multiple ensemble machine learning models to predict loan repayment outcomes. The objective is to identify borrowers who are likely to **not fully repay a loan** (`not.fully.paid = 1`).

Credit risk modeling is a core function in financial institutions. Accurately detecting potential defaulters reduces financial loss and improves portfolio risk management.

---

## Business Objective

In credit risk applications, failing to detect a defaulter (False Negative) is more costly than incorrectly flagging a reliable borrower (False Positive).

Therefore, this project prioritizes:

* **Recall for the default class (minority class)**
* Balanced performance across classes
* Robust model comparison using cross-validation

---

## Dataset Description

The dataset contains 9,578 loan records with 14 features describing borrower credit profile and loan characteristics.

### Key Features

* `credit.policy` — Whether the customer meets credit underwriting criteria
* `purpose` — Loan purpose (encoded categorical variable)
* `int.rate` — Interest rate of the loan
* `installment` — Monthly installment amount
* `log.annual.inc` — Log of annual income
* `dti` — Debt-to-income ratio
* `fico` — FICO credit score
* `days.with.cr.line` — Length of credit history
* `revol.bal` — Revolving balance
* `revol.util` — Revolving credit utilization rate
* `inq.last.6mths` — Number of credit inquiries in the last 6 months
* `delinq.2yrs` — Delinquencies in the past 2 years
* `pub.rec` — Number of derogatory public records

### Target Variable

* `not.fully.paid`

  * 0 = Fully repaid
  * 1 = Not fully repaid (default)

The dataset is imbalanced (~16% defaulters), making recall optimization critical.

---

## Methodology

### 1. Exploratory Data Analysis (EDA)

* Distribution analysis of FICO scores
* Correlation heatmap
* Loan purpose distribution
* Class imbalance assessment

### 2. Data Preparation

* Label encoding of categorical variable (`purpose`)
* Stratified train-test split to preserve class distribution
* Cross-validation using StratifiedKFold

### 3. Models Implemented

* Decision Tree (Baseline)
* Bagging with Decision Trees
* AdaBoost
* Random Forest
* Gradient Boosting

Class imbalance was addressed using:

* `class_weight='balanced'`
* Recall-focused evaluation metrics

---

## Model Evaluation

Evaluation metrics used:

* Confusion Matrix
* Classification Report
* Weighted Recall
* Recall for defaulters (primary metric)

### Key Findings

* The baseline Decision Tree failed to detect defaulters.
* Ensemble methods significantly improved minority class detection.
* **AdaBoost achieved the highest recall for defaulters (~33%)**, making it the strongest candidate for risk-sensitive applications.
* Random Forest provided strong overall stability but lower minority recall.
* Gradient Boosting favored overall accuracy over minority detection.

---

## Conclusion

Ensemble learning methods substantially outperform a single decision tree in credit risk prediction tasks.

For financial institutions prioritizing default detection, AdaBoost with class balancing offers the most practical trade-off between recall and generalization.

---

## Future Improvements

* Implement Gradient Boosting frameworks (XGBoost / LightGBM)
* Perform threshold optimization based on business cost
* Apply SHAP for model explainability
* Build a reusable training pipeline
* Deploy as a REST API for real-world integration

---

## Tech Stack

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

---

## Author

Richmond Dzahene
Machine Learning & AI Development

---

*This project demonstrates practical application of ensemble learning techniques to real-world financial risk modeling problems.*
