# Credit-Risk-Probability-Model-for-Alternative-Data-wek_4

## Overview
This project implements an end-to-end **Credit Risk Model** for Bati Bank's Buy-Now-Pay-Later (BNPL) service. Using transactional and behavioral data from an eCommerce platform, we estimate the probability of customer default and generate credit scores to inform loan approvals, credit limits, and repayment terms.

---

## Business Need
Bati Bank partners with an eCommerce platform to offer BNPL services. The goal is to:

- Identify high-risk vs. low-risk customers
- Automate loan approval decisions
- Recommend optimal loan amounts and durations
- Transform behavioral data into actionable risk signals

---

## Dataset Fields
Key fields include:

- **TransactionId, BatchId, AccountId, SubscriptionId, CustomerId** – Unique identifiers  
- **CurrencyCode, CountryCode, ProviderId, ProductId, ProductCategory** – Transaction and product details  
- **ChannelId** – Web, Android, iOS, Pay Later, Checkout  
- **Amount / Value** – Transaction value (positive = debit, negative = credit)  
- **TransactionStartTime** – Timestamp  
- **PricingStrategy** – Merchant pricing category  
- **FraudResult** – 1 = fraud, 0 = no fraud  

---

## Week 4 Tasks


1. ## Exploratory Data Analysis (EDA)

   - The EDA phase focuses on understanding customer transaction behavior and preparing the data for proxy-based credit risk modeling. We examine the structure and quality of the dataset, explore numerical and categorical feature patterns, analyze transaction distributions, check for missing values and outliers, and study correlations to identify meaningful predictors.

2. ## Exploratory Data Analysis (EDA)
 **Proxy Variable Creation**  
   - Define a proxy variable to label customers as **high risk** (bad) or **low risk** (good) based on RFM behavior patterns.

1. **Feature Engineering**  
   - Select observable features highly correlated with default behavior.
   - Transform raw data into predictive inputs for modeling.

2. **Model Development**  
   - Train **classification models** (e.g., Logistic Regression, XGBoost) to predict credit risk probability.
   - Train **regression/severity models** to estimate potential loss amounts.

3. **Credit Scoring**  
   - Convert risk probabilities into **credit scores** for loan eligibility decisions.

4. **Loan Recommendation**  
   - Predict **optimal loan amount and duration** based on customer risk and historical patterns.

5. **Model Evaluation & Interpretability**  
   - Evaluate using Accuracy, Precision, Recall, F1-score, ROC-AUC, RMSE, MAE, R².
   - Use **SHAP / LIME** for feature importance and explainable predictions.

---


## Project Structure

```bash
credit-risk-model/
├── .github/workflows/ci.yml     # CI/CD
├── data/                        # Add this folder to .gitignore
│   ├── raw/                     # Raw data
│   └── processed/               # Processed/cleaned data
├── notebooks/
│   └── eda.ipynb                # Exploratory Data Analysis
├── src/
│   ├── __init__.py
│   ├── data_processing.py       # Feature engineering
│   ├── train.py                 # Model training
│   ├── predict.py               # Inference & scoring
│   └── api/
│       ├── main.py              # FastAPI app
│       └── pydantic_models.py   # Request/response schemas
├── tests/
│   └── test_data_processing.py  # Unit tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md



Credit Scoring Business Understanding.
---
### 1. How does the Basel II Accord’s emphasis on risk measurement influence our need for an interpretable and well-documented model?
**Answer:**  
Basel II emphasizes quantifying credit risk to ensure banks hold sufficient capital against potential losses. Models must be transparent, interpretable, and well-documented so regulators can understand predictions, data inputs, and assumptions. Poorly documented or opaque models can lead to compliance issues, misestimation of capital requirements, and reputational risks.

---

### 2. Since we lack a direct "default" label, why is creating a proxy variable necessary, and what are the potential business risks of making predictions based on this proxy?
**Answer:**  
When historical default data is unavailable, proxies (e.g., delinquency flags, late payments, credit utilization) are used to approximate default risk. Using proxies carries risks: misclassification of borrowers, inaccurate risk estimation, and potentially flawed business decisions, which could result in financial losses or poor risk management practices.

---

### 3. What are the key trade-offs between using a simple, interpretable model (like Logistic Regression with WoE) versus a complex, high-performance model (like Gradient Boosting) in a regulated financial context?
**Answer:**  
- **Simple models (Logistic Regression + WoE):** Highly interpretable, easy to document, and regulator-friendly. Limitations include lower predictive performance on complex data.  
- **Complex models (Gradient Boosting, XGBoost):** Capture nonlinear relationships and improve accuracy, but are harder to interpret, explain, and justify for compliance purposes.  
Organizations must balance predictive power with regulatory transparency and interpretability.

---
