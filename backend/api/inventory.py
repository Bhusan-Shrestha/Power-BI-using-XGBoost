from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from core.database import get_db

router = APIRouter()
DEFAULT_WAREHOUSE = "Kathmandu"


class InventoryUpsertRequest(BaseModel):
    product_id: int
    warehouse: str = Field(..., min_length=2, max_length=100)
    quantity_on_hand: int = Field(default=0, ge=0)
    reorder_point: int = Field(default=50, ge=0)
    reorder_quantity: int = Field(default=200, gt=0)
    last_restocked: datetime | None = None


def _is_before_today(value: datetime) -> bool:
    return value.date() < datetime.now(timezone.utc).date()


@router.get("/products")
async def list_products(db: AsyncSession = Depends(get_db)):
    result = await db.execute(text("""
        SELECT id, name, sku, category, price, cost
        FROM products
        ORDER BY name ASC
    """))
    return [dict(r._mapping) for r in result.fetchall()]


@router.get("/product-defaults/{product_id}")
async def product_inventory_defaults(
    product_id: int,
    db: AsyncSession = Depends(get_db),
):
    product = await db.execute(text("""
        SELECT id FROM products WHERE id = :product_id
    """), {"product_id": product_id})
    if not product.fetchone():
        raise HTTPException(status_code=404, detail="Product not found.")

    existing = await db.execute(text("""
        SELECT quantity_on_hand, reorder_point, reorder_quantity, warehouse
        FROM inventory
        WHERE product_id = :product_id
          AND LOWER(warehouse) = LOWER(:warehouse)
        ORDER BY id DESC
        LIMIT 1
    """), {
        "product_id": product_id,
        "warehouse": DEFAULT_WAREHOUSE,
    })
    row = existing.fetchone()
    source_warehouse = DEFAULT_WAREHOUSE

    if not row:
        fallback = await db.execute(text("""
            SELECT quantity_on_hand, reorder_point, reorder_quantity, warehouse
            FROM inventory
            WHERE product_id = :product_id
            ORDER BY id DESC
            LIMIT 1
        """), {"product_id": product_id})
        row = fallback.fetchone()
        if row:
            source_warehouse = row.warehouse or DEFAULT_WAREHOUSE

    if not row:
        return {
            "warehouse": DEFAULT_WAREHOUSE,
            "quantity_on_hand": 0,
            "reorder_point": 50,
            "reorder_quantity": 200,
            "source": "default",
        }

    return {
        "warehouse": source_warehouse,
        "quantity_on_hand": int(row.quantity_on_hand or 0),
        "reorder_point": int(row.reorder_point or 50),
        "reorder_quantity": int(row.reorder_quantity or 200),
        "source": "database",
    }


@router.get("/records")
async def inventory_records(
    limit: int = Query(100, ge=1, le=1000),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(text("""
        SELECT
            i.id,
            i.product_id,
            p.name,
            p.sku,
            p.category,
            i.warehouse,
            i.quantity_on_hand,
            i.reorder_point,
            i.reorder_quantity,
            i.last_restocked,
            CASE
                WHEN i.quantity_on_hand <= 0 THEN 'out_of_stock'
                WHEN i.quantity_on_hand <= i.reorder_point THEN 'low_stock'
                ELSE 'in_stock'
            END AS status
        FROM inventory i
        JOIN products p ON p.id = i.product_id
        ORDER BY i.quantity_on_hand ASC, i.id DESC
        LIMIT :limit
    """), {"limit": limit})
    return [dict(r._mapping) for r in result.fetchall()]


@router.put("/records/{inventory_id}")
async def update_inventory_record(
    inventory_id: int,
    payload: InventoryUpsertRequest,
    db: AsyncSession = Depends(get_db),
):
    current = await db.execute(text("""
        SELECT id
        FROM inventory
        WHERE id = :inventory_id
    """), {"inventory_id": inventory_id})
    if not current.fetchone():
        raise HTTPException(status_code=404, detail="Inventory record not found.")

    product = await db.execute(text("""
        SELECT id FROM products WHERE id = :product_id
    """), {"product_id": payload.product_id})
    if not product.fetchone():
        raise HTTPException(status_code=404, detail="Product not found.")

    selected_warehouse = DEFAULT_WAREHOUSE

    restocked_at = payload.last_restocked or datetime.utcnow()
    if _is_before_today(restocked_at):
        raise HTTPException(status_code=400, detail="Restock date cannot be before today.")

    duplicate = await db.execute(text("""
        SELECT id
        FROM inventory
        WHERE product_id = :product_id
          AND LOWER(warehouse) = LOWER(:warehouse)
          AND id <> :inventory_id
        LIMIT 1
    """), {
        "product_id": payload.product_id,
        "warehouse": selected_warehouse,
        "inventory_id": inventory_id,
    })
    if duplicate.fetchone():
        raise HTTPException(
            status_code=400,
            detail="Another inventory record for this product and warehouse already exists.",
        )

    await db.execute(text("""
        UPDATE inventory
        SET
            product_id = :product_id,
            warehouse = :warehouse,
            quantity_on_hand = :quantity_on_hand,
            reorder_point = :reorder_point,
            reorder_quantity = :reorder_quantity,
            last_restocked = :last_restocked,
            updated_at = NOW()
        WHERE id = :inventory_id
    """), {
        "inventory_id": inventory_id,
        "product_id": payload.product_id,
        "warehouse": selected_warehouse,
        "quantity_on_hand": payload.quantity_on_hand,
        "reorder_point": payload.reorder_point,
        "reorder_quantity": payload.reorder_quantity,
        "last_restocked": restocked_at,
    })

    row = await db.execute(text("""
        SELECT
            i.id,
            i.product_id,
            p.name,
            p.sku,
            p.category,
            i.warehouse,
            i.quantity_on_hand,
            i.reorder_point,
            i.reorder_quantity,
            i.last_restocked,
            CASE
                WHEN i.quantity_on_hand <= 0 THEN 'out_of_stock'
                WHEN i.quantity_on_hand <= i.reorder_point THEN 'low_stock'
                ELSE 'in_stock'
            END AS status
        FROM inventory i
        JOIN products p ON p.id = i.product_id
        WHERE i.id = :inventory_id
    """), {"inventory_id": inventory_id})
    updated = row.fetchone()
    if not updated:
        raise HTTPException(status_code=404, detail="Inventory record not found after update.")
    return dict(updated._mapping)


@router.get("/status")
async def inventory_status(db: AsyncSession = Depends(get_db)):
    try:
        result = await db.execute(text("""
            SELECT
                p.name, p.sku, p.category,
                i.quantity_on_hand,
                i.reorder_point,
                i.reorder_quantity,
                i.warehouse,
                CASE
                    WHEN i.quantity_on_hand <= 0 THEN 'out_of_stock'
                    WHEN i.quantity_on_hand <= i.reorder_point THEN 'low_stock'
                    ELSE 'in_stock'
                END AS status
            FROM inventory i
            JOIN products p ON i.product_id = p.id
            ORDER BY i.quantity_on_hand ASC
        """))
        return [dict(r._mapping) for r in result.fetchall()]
    except Exception as e:
        return {"error": str(e)}


@router.get("/alerts")
async def inventory_alerts(db: AsyncSession = Depends(get_db)):
    try:
        result = await db.execute(text("""
            SELECT
                p.name, p.sku,
                i.quantity_on_hand,
                i.reorder_point,
                i.warehouse
            FROM inventory i
            JOIN products p ON i.product_id = p.id
            WHERE i.quantity_on_hand <= i.reorder_point
            ORDER BY i.quantity_on_hand ASC
        """))
        rows = result.fetchall()
        return {
            "count": len(rows),
            "alerts": [dict(r._mapping) for r in rows],
        }
    except Exception as e:
        return {"error": str(e)}


@router.get("/summary")
async def inventory_summary(db: AsyncSession = Depends(get_db)):
    try:
        result = await db.execute(text("""
            SELECT
                COUNT(*) AS total_products,
                SUM(CASE WHEN quantity_on_hand <= 0 THEN 1 ELSE 0 END) AS out_of_stock,
                SUM(CASE WHEN quantity_on_hand > 0
                    AND quantity_on_hand <= reorder_point THEN 1 ELSE 0 END) AS low_stock,
                SUM(CASE WHEN quantity_on_hand > reorder_point THEN 1 ELSE 0 END) AS in_stock,
                SUM(quantity_on_hand) AS total_units
            FROM inventory
        """))
        row = result.fetchone()
        if not row:
            return {
                "total_products": 0,
                "out_of_stock": 0,
                "low_stock": 0,
                "in_stock": 0,
                "total_units": 0,
            }
        return dict(row._mapping)
    except Exception as e:
        return {"error": str(e)}