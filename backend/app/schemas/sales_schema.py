from datetime import date, datetime
from typing import Optional
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    predict_year: int = Field(..., ge=2000, examples=[2026])
    predict_month: int = Field(..., ge=1, le=12, examples=[3])


class SalesOut(BaseModel):
    id: int
    product_id: int
    date: date
    sales: float
    profit: float
    marketing_spend: float
    region: str

    class Config:
        from_attributes = True


class PredictionOut(BaseModel):
    id: int
    product_id: int
    month: str = Field(..., examples=["2026-01"])
    predicted_sales: float
    predicted_profit: float
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class PredictionRunResponse(BaseModel):
    message: str
    count: int
    month: str

    class Config:
        from_attributes = True
