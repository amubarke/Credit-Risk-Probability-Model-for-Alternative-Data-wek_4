# src/api/pydantic_models.py
from pydantic import BaseModel

from pydantic import BaseModel
from typing import Optional

class CustomerData(BaseModel):
    AccountId: int
    SubscriptionId: int
    CustomerId: int
    CurrencyCode: int
    CountryCode: int
    ProviderId: int
    ProductId: int
    ProductCategory: Optional[str] = "A"         # default value
    ChannelId: int
    Amount: float
    Value: float
    PricingStrategy: Optional[str] = "Standard" # default value
    total_transaction_amount: float
    avg_transaction_amount: float
    transaction_count: int
    std_transaction_amount: float
    transaction_hour: int
    transaction_day: int
    transaction_month: int
    transaction_year: int
    TransactionHour: int
    TransactionDay: int
    TransactionMonth: int
    TransactionYear: int
    TransactionWeekday: int


class PredictionResponse(BaseModel):
    prediction_class: int
    risk_probability: float
    model_source: str
