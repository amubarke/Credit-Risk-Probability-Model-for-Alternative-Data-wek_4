# Task 4 – Proxy Target Variable Engineering
# Objective: Create a proxy credit risk target using RFM analysis and K-Means clustering

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# --------------------------------------------------
# 1. RFM FEATURE CALCULATION
# --------------------------------------------------
class RFMCalculator(BaseEstimator, TransformerMixin):
    def __init__(self, customer_col="CustomerId",
                 date_col="TransactionStartTime",
                 amount_col="Amount",
                 snapshot_date=None):
        self.customer_col = customer_col
        self.date_col = date_col
        self.amount_col = amount_col
        self.snapshot_date = snapshot_date

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df[self.date_col] = pd.to_datetime(df[self.date_col])

        snapshot = self.snapshot_date or df[self.date_col].max() + pd.Timedelta(days=1)

        rfm = (
            df.groupby(self.customer_col)
              .agg(
                  recency=(self.date_col, lambda x: (snapshot - x.max()).days),
                  frequency=(self.date_col, "count"),
                  monetary=(self.amount_col, "sum")
              )
              .reset_index()
        )
        return rfm



# --------------------------------------------------
# 2. CUSTOMER CLUSTERING (K-MEANS)
# --------------------------------------------------
class RFMClustering(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=3, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)

    def fit(self, X, y=None):
        features = X[["recency", "frequency", "monetary"]]
        scaled = self.scaler.fit_transform(features)
        self.kmeans.fit(scaled)
        return self

    def transform(self, X):
        df = X.copy()
        scaled = self.scaler.transform(df[["recency", "frequency", "monetary"]])
        df["cluster"] = self.kmeans.predict(scaled)
        return df



# --------------------------------------------------
# 3. HIGH-RISK LABEL ASSIGNMENT
# --------------------------------------------------
class HighRiskLabeler(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        cluster_summary = (
            X.groupby("cluster")[["frequency", "monetary"]]
              .mean()
        )

        self.high_risk_cluster_ = cluster_summary.sum(axis=1).idxmin()
        return self

    def transform(self, X):
        df = X.copy()
        df["is_high_risk"] = (df["cluster"] == self.high_risk_cluster_).astype(int)
        return df[[ "CustomerId", "is_high_risk" ]]


# --------------------------------------------------
# 4. FULL RFM PIPELINE
# --------------------------------------------------
def build_rfm_pipeline():
    return Pipeline(steps=[
        ("rfm", RFMCalculator()),
        ("cluster", RFMClustering()),
        ("label", HighRiskLabeler())
    ])
    return pipeline


