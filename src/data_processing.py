# End-to-End Data Processing Pipeline (Steps 1–6)
# Author: Machine Learning Engineer
# Purpose: Transform raw transaction data into model-ready format using sklearn.pipeline.Pipeline

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer

# WoE / IV
from xverse.transformer import WOE


# --------------------------------------------------
# 1. AGGREGATE FEATURES PER CUSTOMER
# --------------------------------------------------
class AggregateFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, customer_col="CustomerId", amount_col="Amount"):
        self.customer_col = customer_col
        self.amount_col = amount_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        agg_df = df.groupby(self.customer_col)[self.amount_col].agg([
            ("total_transaction_amount", "sum"),
            ("avg_transaction_amount", "mean"),
            ("transaction_count", "count"),
            ("std_transaction_amount", "std")
        ]).reset_index()

        df = df.merge(agg_df, on=self.customer_col, how="left")
        return df


# --------------------------------------------------
# 2. DATE / TIME FEATURE EXTRACTION
# --------------------------------------------------
class DateTimeFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, datetime_col="TransactionStartTime"):
        self.datetime_col = datetime_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df[self.datetime_col] = pd.to_datetime(df[self.datetime_col])

        df["transaction_hour"] = df[self.datetime_col].dt.hour
        df["transaction_day"] = df[self.datetime_col].dt.day
        df["transaction_month"] = df[self.datetime_col].dt.month
        df["transaction_year"] = df[self.datetime_col].dt.year

        return df


# --------------------------------------------------
# 3. LABEL ENCODING FOR CATEGORICAL FEATURES
# --------------------------------------------------
class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_cols):
        self.categorical_cols = categorical_cols
        self.encoders = {}

    def fit(self, X, y=None):
        for col in self.categorical_cols:
            le = LabelEncoder()
            le.fit(X[col].astype(str))
            self.encoders[col] = le
        return self

    def transform(self, X):
        df = X.copy()
        for col, le in self.encoders.items():
            df[col] = le.transform(df[col].astype(str))
        return df


# --------------------------------------------------
# 4. MISSING VALUE HANDLING
# --------------------------------------------------
class MissingValueHandler(BaseEstimator, TransformerMixin):
    def __init__(self, numeric_cols, categorical_cols):
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.num_imputer = SimpleImputer(strategy="mean")
        self.cat_imputer = SimpleImputer(strategy="most_frequent")

    def fit(self, X, y=None):
        self.num_imputer.fit(X[self.numeric_cols])
        self.cat_imputer.fit(X[self.categorical_cols])
        return self

    def transform(self, X):
        df = X.copy()
        df[self.numeric_cols] = self.num_imputer.transform(df[self.numeric_cols])
        df[self.categorical_cols] = self.cat_imputer.transform(df[self.categorical_cols])
        return df


# --------------------------------------------------
# 5. FEATURE SCALING
# --------------------------------------------------
class FeatureScaler(BaseEstimator, TransformerMixin):
    def __init__(self, numeric_cols, method="standard"):
        self.numeric_cols = numeric_cols
        self.method = method
        self.scaler = None

    def fit(self, X, y=None):
        self.scaler = MinMaxScaler() if self.method == "normalize" else StandardScaler()
        self.scaler.fit(X[self.numeric_cols])
        return self

    def transform(self, X):
        df = X.copy()
        df[self.numeric_cols] = self.scaler.transform(df[self.numeric_cols])
        return df


# --------------------------------------------------
# 6. WEIGHT OF EVIDENCE (WoE) TRANSFORMATION
# --------------------------------------------------
class WoETransformer(BaseEstimator, TransformerMixin):
    def __init__(self, target_col="FraudResult"):
        self.target_col = target_col
        self.woe = WOE()

    def fit(self, X, y=None):
        self.woe.fit(X, y)
        return self

    def transform(self, X):
        return self.woe.transform(X)


# --------------------------------------------------
# FULL PIPELINE
# --------------------------------------------------
def build_pipeline(categorical_cols, numeric_cols, scaling_method="standard"):
    pipeline = Pipeline(steps=[
        ("aggregate_features", AggregateFeatures()),
        ("datetime_features", DateTimeFeatures()),
        ("missing_values", MissingValueHandler(numeric_cols, categorical_cols)),
        ("categorical_encoding", CategoricalEncoder(categorical_cols)),
        ("scaling", FeatureScaler(numeric_cols, method=scaling_method)),
        ("woe", WoETransformer())
    ])
    return pipeline



