import asyncio
import random
from datetime import datetime, timedelta
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from core.config import settings
from core.security import hash_password
from models.user import User, UserRole
from models.sales import Product, Sale
from models.inventory import Inventory
from models.customer import Customer
from core.database import Base

CATEGORIES = ["Electronics", "Clothing", "Food & Beverage", "Home & Garden", "Sports", "Books"]
REGIONS = ["North", "South", "East", "West", "Central"]
CHANNELS = ["online", "retail", "wholesale"]
SEGMENTS = ["enterprise", "smb", "individual"]

async def seed():
    engine = create_async_engine(settings.DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    Session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as db:
        print("Seeding users...")
        admin = User(email="admin@bi.com", full_name="Admin User",
                     hashed_password=hash_password("admin123"), role=UserRole.admin)
        standard_user = User(email="user@bi.com", full_name="Standard User",
                     hashed_password=hash_password("user123"), role=UserRole.user)
        db.add_all([admin, standard_user])
        await db.flush()

        print("Seeding products...")
        products = []
        for i in range(50):
            p = Product(
                name=f"Product {i+1:03d}",
                category=random.choice(CATEGORIES),
                sku=f"SKU-{i+1:05d}",
                price=round(random.uniform(10, 500), 2),
                cost=round(random.uniform(5, 200), 2),
            )
            db.add(p)
            products.append(p)
        await db.flush()

        print("Seeding customers...")
        customers = []
        for i in range(200):
            c = Customer(
                name=f"Customer {i+1}",
                email=f"customer{i+1}@example.com",
                country="USA",
                region=random.choice(REGIONS),
                segment=random.choice(SEGMENTS),
                lifetime_value=round(random.uniform(100, 50000), 2),
                churn_risk_score=round(random.uniform(0, 1), 3),
                acquisition_channel=random.choice(["organic", "paid", "referral", "direct"]),
            )
            db.add(c)
            customers.append(c)
        await db.flush()

        print("Seeding sales (2 years of data)...")
        start_date = datetime.now() - timedelta(days=730)
        for i in range(5000):
            sale_date = start_date + timedelta(days=random.randint(0, 730))
            product = random.choice(products)
            qty = random.randint(1, 20)
            sale = Sale(
                product_id=product.id,
                customer_id=random.choice(customers).id if random.random() > 0.3 else None,
                quantity=qty,
                unit_price=product.price,
                total_amount=round(product.price * qty * random.uniform(0.85, 1.0), 2),
                discount=round(random.uniform(0, 0.15), 3),
                region=random.choice(REGIONS),
                channel=random.choice(CHANNELS),
                sale_date=sale_date,
            )
            db.add(sale)

        print("Seeding inventory...")
        for product in products:
            inv = Inventory(
                product_id=product.id,
                warehouse=f"WH-{random.choice(['A', 'B', 'C'])}",
                quantity_on_hand=random.randint(0, 1000),
                reorder_point=random.randint(20, 100),
                reorder_quantity=random.randint(100, 500),
            )
            db.add(inv)

        await db.commit()
        print("✅ Database seeded successfully!")
        print("   Login: admin@bi.com / admin123")

if __name__ == "__main__":
    asyncio.run(seed())
