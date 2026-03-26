from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from core.database import get_db

router = APIRouter()


@router.get("/kpis")
async def get_kpis(db: AsyncSession = Depends(get_db)):
    try:
        r1 = await db.execute(text("""
            SELECT
                COALESCE(SUM(total_amount), 0) as revenue,
                COUNT(*) as orders,
                COALESCE(AVG(total_amount), 0) as avg_order
            FROM sales
            WHERE sale_date >= NOW() - INTERVAL '30 days'
        """))
        row = r1.fetchone()

        r2 = await db.execute(text("""
            SELECT COALESCE(SUM(total_amount), 0) as prev_revenue
            FROM sales
            WHERE sale_date >= NOW() - INTERVAL '60 days'
              AND sale_date < NOW() - INTERVAL '30 days'
        """))
        prev = r2.fetchone()

        r3 = await db.execute(text(
            "SELECT COUNT(*) as total FROM customers"
        ))
        cust = r3.fetchone()

        r4 = await db.execute(text("""
            SELECT COUNT(*) as alerts FROM inventory
            WHERE quantity_on_hand <= reorder_point
        """))
        alerts = r4.fetchone()

        revenue = float(row.revenue)
        prev_revenue = float(prev.prev_revenue)
        growth = round(
            ((revenue - prev_revenue) / prev_revenue * 100)
            if prev_revenue > 0 else 0, 1
        )

        return {
            "total_revenue": round(revenue, 2),
            "total_orders": row.orders,
            "avg_order_value": round(float(row.avg_order), 2),
            "total_customers": cust.total,
            "low_stock_alerts": alerts.alerts,
            "revenue_growth": growth,
        }
    except Exception as e:
        return {"error": str(e)}


@router.get("/revenue-trend")
async def get_revenue_trend(db: AsyncSession = Depends(get_db)):
    result = await db.execute(text("""
        SELECT
            DATE(sale_date) as date,
            ROUND(SUM(total_amount)::numeric, 2) as revenue,
            COUNT(*) as orders
        FROM sales
        WHERE sale_date >= NOW() - INTERVAL '90 days'
        GROUP BY DATE(sale_date)
        ORDER BY date ASC
    """))
    rows = result.fetchall()
    return [
        {
            "date": str(r.date),
            "revenue": float(r.revenue),
            "orders": r.orders
        }
        for r in rows
    ]


@router.get("/top-products")
async def get_top_products(db: AsyncSession = Depends(get_db)):
    result = await db.execute(text("""
        SELECT
            p.name,
            p.category,
            ROUND(SUM(s.total_amount)::numeric, 2) as revenue,
            SUM(s.quantity) as units
        FROM sales s
        JOIN products p ON s.product_id = p.id
        WHERE s.sale_date >= NOW() - INTERVAL '30 days'
        GROUP BY p.id, p.name, p.category
        ORDER BY revenue DESC
        LIMIT 5
    """))
    rows = result.fetchall()
    return [
        {
            "name": r.name,
            "category": r.category,
            "revenue": float(r.revenue),
            "units": r.units,
        }
        for r in rows
    ]


@router.get("/sales-by-channel")
async def sales_by_channel(db: AsyncSession = Depends(get_db)):
    result = await db.execute(text("""
        SELECT
            channel,
            ROUND(SUM(total_amount)::numeric, 2) as revenue,
            COUNT(*) as orders
        FROM sales
        WHERE sale_date >= NOW() - INTERVAL '30 days'
        GROUP BY channel
        ORDER BY revenue DESC
    """))
    return [
        {
            "channel": r.channel,
            "revenue": float(r.revenue),
            "orders": r.orders,
        }
        for r in result.fetchall()
    ]
