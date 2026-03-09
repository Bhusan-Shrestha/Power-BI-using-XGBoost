from dataclasses import dataclass


@dataclass
class SalesFeatures:
    month: int
    marketing_spend: float
    region: str
    previous_sales: float
