from sqlalchemy import Date, DateTime, Float, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .db import Base


class Product(Base):
    __tablename__ = "products"

    product_id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    product_name: Mapped[str] = mapped_column(String(120), nullable=False)
    category: Mapped[str] = mapped_column(String(80), nullable=False)

    sales = relationship("MonthlySales", back_populates="product")
    predictions = relationship("Prediction", back_populates="product")


class MonthlySales(Base):
    __tablename__ = "monthly_sales"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.product_id"), nullable=False)
    date: Mapped[Date] = mapped_column(Date, nullable=False)
    sales: Mapped[float] = mapped_column(Float, nullable=False)
    profit: Mapped[float] = mapped_column(Float, nullable=False)
    marketing_spend: Mapped[float] = mapped_column(Float, nullable=False)
    region: Mapped[str] = mapped_column(String(50), nullable=False)

    product = relationship("Product", back_populates="sales")


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    product_id: Mapped[int] = mapped_column(ForeignKey("products.product_id"), nullable=False)
    month: Mapped[str] = mapped_column(String(7), nullable=False)
    predicted_sales: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_profit: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[DateTime] = mapped_column(DateTime, server_default=func.now())

    product = relationship("Product", back_populates="predictions")
