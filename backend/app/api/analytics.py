from datetime import date

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func
from sqlalchemy.orm import Session
from ..database.db import get_db
from ..database.tables import MonthlySales, Product

router = APIRouter(prefix="/analytics", tags=["analytics"])


def _filtered_rows(
    db: Session,
    start_date: date | None = None,
    end_date: date | None = None,
    region: str | None = None,
    product_id: int | None = None,
    category: str | None = None,
):
    query = db.query(MonthlySales, Product).join(Product, Product.product_id == MonthlySales.product_id)

    if start_date:
        query = query.filter(MonthlySales.date >= start_date)
    if end_date:
        query = query.filter(MonthlySales.date <= end_date)
    if region:
        query = query.filter(func.lower(MonthlySales.region) == region.strip().lower())
    if product_id:
        query = query.filter(MonthlySales.product_id == product_id)
    if category:
        query = query.filter(func.lower(Product.category) == category.strip().lower())

    return query.order_by(MonthlySales.date.asc()).all()


@router.get("/filters")
def get_filter_options(db: Session = Depends(get_db)):
    min_date, max_date = db.query(
        func.min(MonthlySales.date),
        func.max(MonthlySales.date),
    ).one()

    regions = [
        row[0]
        for row in db.query(MonthlySales.region)
        .distinct()
        .order_by(MonthlySales.region.asc())
        .all()
    ]
    categories = [
        row[0]
        for row in db.query(Product.category).distinct().order_by(Product.category.asc()).all()
    ]
    products = [
        {"product_id": product_id, "product_name": product_name}
        for product_id, product_name in db.query(Product.product_id, Product.product_name)
        .order_by(Product.product_name.asc())
        .all()
    ]

    return {
        "min_date": min_date.isoformat() if min_date else None,
        "max_date": max_date.isoformat() if max_date else None,
        "regions": regions,
        "categories": categories,
        "products": products,
    }


@router.get("/summary")
def get_summary(
    start_date: date | None = None,
    end_date: date | None = None,
    region: str | None = None,
    product_id: int | None = None,
    category: str | None = None,
    db: Session = Depends(get_db),
):
    rows = _filtered_rows(db, start_date, end_date, region, product_id, category)
    total_sales = sum(float(sale.sales) for sale, _ in rows)
    total_profit = sum(float(sale.profit) for sale, _ in rows)
    record_count = len(rows)

    avg_margin = (total_profit / total_sales * 100.0) if total_sales else 0.0
    return {
        "total_sales": float(total_sales),
        "total_profit": float(total_profit),
        "avg_margin": float(avg_margin),
        "record_count": record_count,
    }


@router.get("/regional")
def get_regional_performance(
    start_date: date | None = None,
    end_date: date | None = None,
    product_id: int | None = None,
    category: str | None = None,
    db: Session = Depends(get_db),
):
    rows = _filtered_rows(db, start_date, end_date, None, product_id, category)
    grouped = {}
    for sale, _ in rows:
        grouped.setdefault(sale.region, {"sales": 0.0, "profit": 0.0})
        grouped[sale.region]["sales"] += float(sale.sales)
        grouped[sale.region]["profit"] += float(sale.profit)

    return [
        {
            "region": region,
            "sales": float(values["sales"]),
            "profit": float(values["profit"]),
        }
        for region, values in sorted(grouped.items(), key=lambda item: item[0])
    ]


@router.get("/trend")
def get_monthly_trend(
    start_date: date | None = None,
    end_date: date | None = None,
    region: str | None = None,
    product_id: int | None = None,
    category: str | None = None,
    db: Session = Depends(get_db),
):
    rows = _filtered_rows(db, start_date, end_date, region, product_id, category)
    grouped = {}
    for sale, _ in rows:
        month = sale.date.strftime("%Y-%m")
        grouped.setdefault(month, {"sales": 0.0, "profit": 0.0, "marketing_spend": 0.0})
        grouped[month]["sales"] += float(sale.sales)
        grouped[month]["profit"] += float(sale.profit)
        grouped[month]["marketing_spend"] += float(sale.marketing_spend)

    return [
        {
            "month": month,
            "sales": float(values["sales"]),
            "profit": float(values["profit"]),
            "marketing_spend": float(values["marketing_spend"]),
        }
        for month, values in sorted(grouped.items(), key=lambda item: item[0])
    ]


@router.get("/products")
def get_product_performance(
    start_date: date | None = None,
    end_date: date | None = None,
    region: str | None = None,
    category: str | None = None,
    limit: int = Query(default=10, ge=1, le=100),
    db: Session = Depends(get_db),
):
    rows = _filtered_rows(db, start_date, end_date, region, None, category)
    grouped = {}
    for sale, product in rows:
        key = product.product_id
        grouped.setdefault(
            key,
            {
                "product_id": product.product_id,
                "product_name": product.product_name,
                "category": product.category,
                "sales": 0.0,
                "profit": 0.0,
            },
        )
        grouped[key]["sales"] += float(sale.sales)
        grouped[key]["profit"] += float(sale.profit)

    result = sorted(grouped.values(), key=lambda item: item["sales"], reverse=True)[:limit]
    return result


@router.get("/category")
def get_category_performance(
    start_date: date | None = None,
    end_date: date | None = None,
    region: str | None = None,
    product_id: int | None = None,
    db: Session = Depends(get_db),
):
    rows = _filtered_rows(db, start_date, end_date, region, product_id, None)
    grouped = {}
    for sale, product in rows:
        grouped.setdefault(product.category, {"sales": 0.0, "profit": 0.0})
        grouped[product.category]["sales"] += float(sale.sales)
        grouped[product.category]["profit"] += float(sale.profit)

    return [
        {
            "category": category,
            "sales": float(values["sales"]),
            "profit": float(values["profit"]),
        }
        for category, values in sorted(grouped.items(), key=lambda item: item[0])
    ]


@router.get("/records")
def get_sales_records(
    start_date: date | None = None,
    end_date: date | None = None,
    region: str | None = None,
    product_id: int | None = None,
    category: str | None = None,
    limit: int = Query(default=200, ge=1, le=5000),
    db: Session = Depends(get_db),
):
    rows = _filtered_rows(db, start_date, end_date, region, product_id, category)
    rows = rows[:limit]

    return [
        {
            "id": sale.id,
            "product_id": sale.product_id,
            "product_name": product.product_name,
            "category": product.category,
            "date": sale.date.isoformat(),
            "region": sale.region,
            "marketing_spend": float(sale.marketing_spend),
            "sales": float(sale.sales),
            "profit": float(sale.profit),
            "margin_pct": float((sale.profit / sale.sales * 100.0) if sale.sales else 0.0),
        }
        for sale, product in rows
    ]
