from sqlalchemy import Column, Integer, Float, String, DateTime, Boolean, Text
from sqlalchemy.sql import func
from core.database import Base


class Customer(Base):
    __tablename__ = "customers"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False)
    email = Column(String(200), unique=True)
    phone = Column(String(50))
    country = Column(String(100))
    region = Column(String(100))
    segment = Column(String(50))  # enterprise, smb, individual
    lifetime_value = Column(Float, default=0.0)
    churn_risk_score = Column(Float, default=0.0)
    acquisition_channel = Column(String(100))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
