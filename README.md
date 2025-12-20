# Credit-Risk-Probability-Model-for-Alternative-Data-wek_4

## Overview
This project implements an end-to-end **Credit Risk Model** for Bati Bank's Buy-Now-Pay-Later (BNPL) service. Using transactional and behavioral data from an eCommerce platform, we estimate the probability of customer default and generate credit scores to inform loan approvals, credit limits, and repayment terms.

---

## Business Need
Bati Bank partners with an eCommerce platform to offer BNPL services. The goals are:

- Identify high-risk vs. low-risk customers
- Automate loan approval decisions
- Recommend optimal loan amounts and durations
- Transform behavioral data into actionable risk signals

**Proxy Variable Justification:**  
Since historical default data is unavailable, a **proxy variable** based on RFM (Recency, Frequency, Monetary) clustering is used to label high-risk customers. This allows for risk-based decision-making despite lacking explicit default labels.

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

## Final Report Tasks

1. **Exploratory Data Analysis (EDA)**  
   - Understand customer transaction behavior and data quality  
   - Identify outliers, missing values, correlations, and feature distributions  
   - Visualizations: histograms, boxplots, and heatmaps

2. **Proxy Target Variable Engineering**  
   - Use **RFM analysis** to segment customers  
   - Apply **K-Means clustering** to label high-risk vs. low-risk customers

3. **Feature Engineering**  
   - Aggregate per-customer metrics (total/avg transaction amount, counts, etc.)  
   - Create date/time features from transaction timestamps  
   - Encode categorical variables (Label Encoding / WoE)  
   - Scale numerical features for modeling

4. **Model Development**  
   - Train **classification models** (Logistic Regression, XGBoost, Random Forest)  
   - Hyperparameter tuning via **GridSearchCV**  
   - Evaluate models using **Accuracy, Precision, Recall, F1-score, ROC-AUC**

5. **Credit Scoring**  
   - Convert predicted probabilities into credit scores  
   - Define risk thresholds for BNPL loan approvals

6. **Loan Recommendation**  
   - Suggest **optimal loan amounts and repayment durations**  
   - Based on predicted risk and historical repayment patterns

7. **API & Deployment**  
   - **FastAPI** endpoints for inference and scoring  
   - Sample request/response:
     ```json
     POST /predict
     {
       "CustomerId": 12345,
       "TransactionHistory": [...]
     }
     
     Response:
     {
       "credit_score": 780,
       "risk_label": "Low Risk",
       "recommended_loan": 1500
     }
     ```
   - Dockerized for production deployment  
   - CI/CD pipelines track unit tests and automated builds

8. **Model Logging & Interpretability**  
   - Track experiments using **MLflow**  
   - Use **SHAP / LIME** for feature importance  
   - Placeholder screenshots:
     - MLflow dashboard
     - CI/CD status
     - Docker container running

9. **Limitations of Proxy-Based Approach**  
   - Proxy may misclassify customers due to imperfect risk signals  
   - Does not replace true default data; predictions may be biased  
   - Requires regular validation and monitoring in production

---

## Project Structure

```bash
credit-risk-model/
├── .github/workflows/ci.yml       # CI/CD pipeline
├── data/                          # Add to .gitignore
│   ├── raw/                       # Raw dataset
│   └── processed/                 # Cleaned & engineered data
├── notebooks/
│   └── eda.ipynb                  # Exploratory Data Analysis
├── src/
│   ├── __init__.py
│   ├── data_processing.py         # Feature engineering
│   ├── train.py                   # Model training
│   ├── predict.py                 # Inference / API
│   └── api/
│       ├── main.py                # FastAPI application
│       └── pydantic_models.py     # Request/response schemas
├── tests/
│   └── test_data_processing.py    # Unit tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md
