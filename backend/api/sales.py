from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from core.database import get_db

router = APIRouter()

PERIOD_MAP = {
    "7d":  "7 days",
    "30d": "30 days",
    "90d": "90 days",
    "1y":  "365 days",
}

VALID_CHANNELS = {"online", "retail", "wholesale"}


class SaleUpsertRequest(BaseModel):
    product_id: int
    customer_id: int | None = None
    quantity: int = Field(..., gt=0)
    unit_price: float | None = Field(default=None, gt=0)
    discount: float = Field(default=0.0, ge=0.0, le=1.0)
    region: str | None = None
    channel: str = "retail"
    sale_date: datetime | None = None


def _is_before_today(value: datetime) -> bool:
    return value.date() < datetime.now(timezone.utc).date()


async def _fetch_product(db: AsyncSession, product_id: int):
    result = await db.execute(text("""
        SELECT id, name, sku, price, cost
        FROM products
        WHERE id = :product_id
    """), {"product_id": product_id})
    return result.fetchone()


async def _fetch_customer(db: AsyncSession, customer_id: int):
    result = await db.execute(text("""
        SELECT id, name
        FROM customers
        WHERE id = :customer_id
    """), {"customer_id": customer_id})
    return result.fetchone()


async def _fetch_sale(db: AsyncSession, sale_id: int):
    result = await db.execute(text("""
        SELECT id, product_id, quantity
        FROM sales
        WHERE id = :sale_id
    """), {"sale_id": sale_id})
    return result.fetchone()


async def _adjust_inventory_for_sale(db: AsyncSession, product_id: int, delta_quantity: int):
    if delta_quantity == 0:
        return

    if delta_quantity > 0:
        inv_rows = await db.execute(text("""
            SELECT id, quantity_on_hand
            FROM inventory
            WHERE product_id = :product_id
            ORDER BY quantity_on_hand DESC, id ASC
            FOR UPDATE
        """), {"product_id": product_id})
        rows = inv_rows.fetchall()

        remaining = delta_quantity
        for row in rows:
            if remaining <= 0:
                break
            available = row.quantity_on_hand or 0
            if available <= 0:
                continue
            consume = min(available, remaining)
            await db.execute(text("""
                UPDATE inventory
                SET quantity_on_hand = quantity_on_hand - :consume,
                    updated_at = NOW()
                WHERE id = :inventory_id
            """), {"consume": consume, "inventory_id": row.id})
            remaining -= consume

        if remaining > 0:
            raise HTTPException(
                status_code=400,
                detail="Insufficient inventory for this product to complete the sale.",
            )
        return

    restock_qty = abs(delta_quantity)
    target = await db.execute(text("""
        SELECT id
        FROM inventory
        WHERE product_id = :product_id
        ORDER BY quantity_on_hand DESC, id ASC
        LIMIT 1
        FOR UPDATE
    """), {"product_id": product_id})
    target_row = target.fetchone()
    if not target_row:
        await db.execute(text("""
            INSERT INTO inventory (
                product_id, warehouse, quantity_on_hand,
                reorder_point, reorder_quantity, last_restocked, updated_at
            ) VALUES (
                :product_id, 'WH-A', :quantity_on_hand,
                50, 200, NOW(), NOW()
            )
        """), {"product_id": product_id, "quantity_on_hand": restock_qty})
        return

    await db.execute(text("""
        UPDATE inventory
        SET quantity_on_hand = quantity_on_hand + :restock_qty,
            updated_at = NOW()
        WHERE id = :inventory_id
    """), {"restock_qty": restock_qty, "inventory_id": target_row.id})


def _normalize_channel(channel: str) -> str:
    normalized = (channel or "retail").strip().lower()
    if normalized not in VALID_CHANNELS:
        raise HTTPException(
            status_code=400,
            detail="Channel must be one of: online, retail, wholesale.",
        )
    return normalized


async def _sale_detail(db: AsyncSession, sale_id: int):
    result = await db.execute(text("""
        SELECT
            s.id,
            s.product_id,
            p.name AS product_name,
            p.sku,
            p.category,
            s.customer_id,
            s.quantity,
            s.unit_price,
            s.discount,
            s.total_amount,
            ROUND((s.quantity * s.unit_price)::numeric, 2) AS gross_sales,
            ROUND((s.total_amount - (COALESCE(p.cost, 0) * s.quantity))::numeric, 2) AS gross_profit,
            CASE
                WHEN s.total_amount > 0 THEN ROUND((((s.total_amount - (COALESCE(p.cost, 0) * s.quantity)) / s.total_amount) * 100)::numeric, 2)
                ELSE 0
            END AS margin_pct,
            s.region,
            s.channel,
            s.sale_date
        FROM sales s
        JOIN products p ON p.id = s.product_id
        WHERE s.id = :sale_id
    """), {"sale_id": sale_id})
    row = result.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Sale not found.")
    return dict(row._mapping)


@router.get("/products")
async def list_products(db: AsyncSession = Depends(get_db)):
    result = await db.execute(text("""
        SELECT id, name, sku, category, price, cost
        FROM products
        ORDER BY name ASC
    """))
    return [dict(r._mapping) for r in result.fetchall()]


@router.get("/customers")
async def list_customers(db: AsyncSession = Depends(get_db)):
    result = await db.execute(text("""
        SELECT id, name
        FROM customers
        ORDER BY name ASC, id ASC
    """))
    return [dict(r._mapping) for r in result.fetchall()]


@router.get("/records")
async def list_sales_records(
    limit: int = Query(50, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(text("""
        SELECT
            s.id,
            s.product_id,
            p.name AS product_name,
            p.sku,
            p.category,
            s.customer_id,
            s.quantity,
            s.unit_price,
            s.discount,
            s.total_amount,
            ROUND((s.quantity * s.unit_price)::numeric, 2) AS gross_sales,
            ROUND((s.total_amount - (COALESCE(p.cost, 0) * s.quantity))::numeric, 2) AS gross_profit,
            CASE
                WHEN s.total_amount > 0 THEN ROUND((((s.total_amount - (COALESCE(p.cost, 0) * s.quantity)) / s.total_amount) * 100)::numeric, 2)
                ELSE 0
            END AS margin_pct,
            s.region,
            s.channel,
            s.sale_date
        FROM sales s
        JOIN products p ON p.id = s.product_id
        ORDER BY s.sale_date DESC, s.id DESC
        LIMIT :limit
    """), {"limit": limit})
    return [dict(r._mapping) for r in result.fetchall()]


@router.put("/records/{sale_id}")
async def update_sale(
    sale_id: int,
    payload: SaleUpsertRequest,
    db: AsyncSession = Depends(get_db),
):
    existing = await _fetch_sale(db, sale_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Sale not found.")

    product = await _fetch_product(db, payload.product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found.")

    if payload.customer_id is not None:
        customer = await _fetch_customer(db, payload.customer_id)
        if not customer:
            raise HTTPException(status_code=404, detail="Customer not found.")

    channel = _normalize_channel(payload.channel)
    unit_price = float(payload.unit_price if payload.unit_price is not None else product.price)
    if unit_price <= 0:
        raise HTTPException(status_code=400, detail="Unit price must be greater than 0.")

    sale_date = payload.sale_date or datetime.utcnow()
    if _is_before_today(sale_date):
        raise HTTPException(status_code=400, detail="Sale date cannot be before today.")

    quantity = int(payload.quantity)
    discount = float(payload.discount or 0.0)
    total_amount = round((unit_price * quantity) * (1 - discount), 2)

    if existing.product_id != payload.product_id:
        await _adjust_inventory_for_sale(db, existing.product_id, -int(existing.quantity))
        await _adjust_inventory_for_sale(db, payload.product_id, quantity)
    else:
        delta = quantity - int(existing.quantity)
        await _adjust_inventory_for_sale(db, payload.product_id, delta)

    await db.execute(text("""
        UPDATE sales
        SET
            product_id = :product_id,
            customer_id = :customer_id,
            quantity = :quantity,
            unit_price = :unit_price,
            total_amount = :total_amount,
            discount = :discount,
            region = :region,
            channel = :channel,
            sale_date = :sale_date
        WHERE id = :sale_id
    """), {
        "sale_id": sale_id,
        "product_id": payload.product_id,
        "customer_id": payload.customer_id,
        "quantity": quantity,
        "unit_price": unit_price,
        "total_amount": total_amount,
        "discount": discount,
        "region": payload.region,
        "channel": channel,
        "sale_date": sale_date,
    })

    return await _sale_detail(db, sale_id)


@router.get("/summary")
async def get_sales_summary(
    period: str = Query("30d"),
    db: AsyncSession = Depends(get_db),
):
    interval = PERIOD_MAP.get(period, "30 days")
    try:
        result = await db.execute(text(f"""
            SELECT
                ROUND(COALESCE(SUM(total_amount), 0)::numeric, 2) AS revenue,
                COUNT(*) AS orders,
                COALESCE(SUM(quantity), 0) AS units_sold,
                ROUND(COALESCE(AVG(total_amount), 0)::numeric, 2) AS avg_order_value,
                ROUND(COALESCE(SUM(discount * total_amount), 0)::numeric, 2) AS total_discounts
            FROM sales
            WHERE sale_date >= NOW() - INTERVAL '{interval}'
        """))
        row = result.fetchone()
        return {
            "revenue": float(row.revenue),
            "orders": row.orders,
            "units_sold": row.units_sold,
            "avg_order_value": float(row.avg_order_value),
            "total_discounts": float(row.total_discounts),
            "period": period,
        }
    except Exception as e:
        return {"error": str(e)}


@router.get("/by-category")
async def sales_by_category(
    period: str = Query("30d"),
    db: AsyncSession = Depends(get_db),
):
    interval = PERIOD_MAP.get(period, "30 days")
    try:
        result = await db.execute(text(f"""
            SELECT
                p.category,
                ROUND(SUM(s.total_amount)::numeric, 2) AS revenue,
                COUNT(s.id) AS orders,
                SUM(s.quantity) AS units
            FROM sales s
            JOIN products p ON s.product_id = p.id
            WHERE s.sale_date >= NOW() - INTERVAL '{interval}'
            GROUP BY p.category
            ORDER BY revenue DESC
        """))
        return [
            {
                "category": r.category,
                "revenue": float(r.revenue),
                "orders": r.orders,
                "units": r.units,
            }
            for r in result.fetchall()
        ]
    except Exception as e:
        return {"error": str(e)}


@router.get("/by-region")
async def sales_by_region(
    period: str = Query("30d"),
    db: AsyncSession = Depends(get_db),
):
    interval = PERIOD_MAP.get(period, "30 days")
    try:
        result = await db.execute(text(f"""
            SELECT
                region,
                ROUND(SUM(total_amount)::numeric, 2) AS revenue,
                COUNT(*) AS orders
            FROM sales
            WHERE sale_date >= NOW() - INTERVAL '{interval}'
            GROUP BY region
            ORDER BY revenue DESC
        """))
        return [
            {
                "region": r.region,
                "revenue": float(r.revenue),
                "orders": r.orders,
            }
            for r in result.fetchall()
        ]
    except Exception as e:
        return {"error": str(e)}


@router.get("/monthly-trend")
async def monthly_trend(db: AsyncSession = Depends(get_db)):
    try:
        result = await db.execute(text("""
            SELECT
                TO_CHAR(sale_date, 'YYYY-MM') AS month,
                ROUND(SUM(total_amount)::numeric, 2) AS revenue,
                COUNT(*) AS orders
            FROM sales
            WHERE sale_date >= NOW() - INTERVAL '365 days'
            GROUP BY TO_CHAR(sale_date, 'YYYY-MM')
            ORDER BY month ASC
        """))
        return [
            {
                "month": r.month,
                "revenue": float(r.revenue),
                "orders": r.orders,
            }
            for r in result.fetchall()
        ]
    except Exception as e:
        return {"error": str(e)}


@router.get("/top-products")
async def top_products(
    limit: int = Query(10),
    period: str = Query("30d"),
    db: AsyncSession = Depends(get_db),
):
    interval = PERIOD_MAP.get(period, "30 days")
    try:
        result = await db.execute(text(f"""
            SELECT
                p.name,
                p.category,
                p.sku,
                ROUND(SUM(s.total_amount)::numeric, 2) AS revenue,
                SUM(s.quantity) AS units
            FROM sales s
            JOIN products p ON s.product_id = p.id
            WHERE s.sale_date >= NOW() - INTERVAL '{interval}'
            GROUP BY p.id, p.name, p.category, p.sku
            ORDER BY revenue DESC
            LIMIT {limit}
        """))
        return [
            {
                "name": r.name,
                "category": r.category,
                "sku": r.sku,
                "revenue": float(r.revenue),
                "units": r.units,
            }
            for r in result.fetchall()
        ]
    except Exception as e:
        return {"error": str(e)}