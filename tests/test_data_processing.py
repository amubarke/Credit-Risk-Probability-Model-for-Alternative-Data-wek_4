import pandas as pd
import numpy as np

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from src.data_processing import (
    AggregateFeatures,
    DateTimeFeatures,
    build_pipeline
)
def test_aggregate_features_output_columns():
    df = pd.DataFrame({
        "CustomerId": ["C1", "C1", "C2"],
        "TransactionId": ["T1", "T2", "T3"],
        "Amount": [100, 200, 50]
    })

    transformer = AggregateFeatures()
    result = transformer.fit_transform(df)

    expected_columns = {
        "CustomerId",
        "total_transaction_amount",
        "avg_transaction_amount",
        "transaction_count",
        "std_transaction_amount"
    }

    assert expected_columns.issubset(result.columns)



def test_datetime_features_created():
    df = pd.DataFrame({
        "TransactionStartTime": [
            "2018-11-27 20:29:09",
            "2018-12-01 10:15:00"
        ]
    })

    transformer = DateTimeFeatures()
    result = transformer.fit_transform(df)

    expected_columns = {
        "transaction_hour",
        "transaction_day",
        "transaction_month",
        "transaction_year"
    }

    assert expected_columns.issubset(result.columns)


