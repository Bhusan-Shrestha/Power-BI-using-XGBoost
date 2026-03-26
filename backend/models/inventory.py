from sqlalchemy import Column, Integer, Float, String, DateTime, ForeignKey, Boolean
from sqlalchemy.sql import func
from core.database import Base


class Inventory(Base):
    __tablename__ = "inventory"

    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    warehouse = Column(String(100), nullable=False)
    quantity_on_hand = Column(Integer, default=0)
    reorder_point = Column(Integer, default=50)
    reorder_quantity = Column(Integer, default=200)
    last_restocked = Column(DateTime(timezone=True))
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class InventoryTransaction(Base):
    __tablename__ = "inventory_transactions"

    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    transaction_type = Column(String(20))  # in, out, adjustment
    quantity = Column(Integer, nullable=False)
    reference = Column(String(100))
    transaction_date = Column(DateTime(timezone=True), server_default=func.now())
