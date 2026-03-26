from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, EmailStr, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from core.database import get_db

router = APIRouter()


class CustomerUpsertRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=200)
    email: EmailStr
    phone: str = Field(..., min_length=10, max_length=10)
    country: str | None = Field(default=None, max_length=100)
    region: str | None = Field(default=None, max_length=100)
    segment: str = Field(default="individual", max_length=50)
    lifetime_value: float = Field(default=0.0, ge=0)
    churn_risk_score: float = Field(default=0.0, ge=0, le=1)
    acquisition_channel: str | None = Field(default=None, max_length=100)
    is_active: bool = True

    @field_validator("phone")
    @classmethod
    def validate_phone(cls, value: str) -> str:
        phone = value.strip()
        if not phone.isdigit() or len(phone) != 10:
            raise ValueError("Phone number must be exactly 10 digits.")
        return phone


@router.get("/records")
async def customer_records(
    limit: int = Query(200, ge=1, le=1000),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(text("""
        SELECT
            id,
            name,
            email,
            phone,
            country,
            region,
            segment,
            ROUND(COALESCE(lifetime_value, 0)::numeric, 2) AS lifetime_value,
            ROUND(COALESCE(churn_risk_score, 0)::numeric, 3) AS churn_risk_score,
            acquisition_channel,
            is_active,
            created_at
        FROM customers
        ORDER BY created_at DESC, id DESC
        LIMIT :limit
    """), {"limit": limit})
    return [dict(r._mapping) for r in result.fetchall()]


@router.put("/records/{customer_id}")
async def update_customer(
    customer_id: int,
    payload: CustomerUpsertRequest,
    db: AsyncSession = Depends(get_db),
):
    current = await db.execute(text("""
        SELECT id
        FROM customers
        WHERE id = :customer_id
    """), {"customer_id": customer_id})
    if not current.fetchone():
        raise HTTPException(status_code=404, detail="Customer not found.")

    duplicate = await db.execute(text("""
        SELECT id
        FROM customers
        WHERE LOWER(email) = LOWER(:email)
          AND id <> :customer_id
        LIMIT 1
    """), {
        "email": payload.email.strip(),
        "customer_id": customer_id,
    })
    if duplicate.fetchone():
        raise HTTPException(status_code=400, detail="Another customer already uses this email.")

    await db.execute(text("""
        UPDATE customers
        SET
            name = :name,
            email = :email,
            phone = :phone,
            country = :country,
            region = :region,
            segment = :segment,
            lifetime_value = :lifetime_value,
            churn_risk_score = :churn_risk_score,
            acquisition_channel = :acquisition_channel,
            is_active = :is_active
        WHERE id = :customer_id
    """), {
        "customer_id": customer_id,
        "name": payload.name.strip(),
        "email": payload.email.strip(),
        "phone": payload.phone,
        "country": payload.country,
        "region": payload.region,
        "segment": payload.segment,
        "lifetime_value": payload.lifetime_value,
        "churn_risk_score": payload.churn_risk_score,
        "acquisition_channel": payload.acquisition_channel,
        "is_active": payload.is_active,
    })

    row = await db.execute(text("""
        SELECT
            id,
            name,
            email,
            phone,
            country,
            region,
            segment,
            ROUND(COALESCE(lifetime_value, 0)::numeric, 2) AS lifetime_value,
            ROUND(COALESCE(churn_risk_score, 0)::numeric, 3) AS churn_risk_score,
            acquisition_channel,
            is_active,
            created_at
        FROM customers
        WHERE id = :customer_id
    """), {"customer_id": customer_id})
    return dict(row.fetchone()._mapping)


@router.get("/summary")
async def customer_summary(db: AsyncSession = Depends(get_db)):
    try:
        result = await db.execute(text("""
            SELECT
                COUNT(*) AS total,
                COUNT(CASE WHEN is_active = true THEN 1 END) AS active,
                ROUND(AVG(lifetime_value)::numeric, 2) AS avg_ltv,
                ROUND(SUM(lifetime_value)::numeric, 2) AS total_ltv
            FROM customers
        """))
        row = result.fetchone()
        return dict(row._mapping)
    except Exception as e:
        return {"error": str(e)}


@router.get("/segments")
async def customer_segments(db: AsyncSession = Depends(get_db)):
    try:
        result = await db.execute(text("""
            SELECT
                segment,
                COUNT(*) AS count,
                ROUND(AVG(lifetime_value)::numeric, 2) AS avg_ltv,
                ROUND(SUM(lifetime_value)::numeric, 2) AS total_ltv
            FROM customers
            GROUP BY segment
            ORDER BY total_ltv DESC
        """))
        return [dict(r._mapping) for r in result.fetchall()]
    except Exception as e:
        return {"error": str(e)}


@router.get("/churn-risk")
async def churn_risk(
    threshold: float = Query(0.6),
    db: AsyncSession = Depends(get_db),
):
    try:
        result = await db.execute(text(f"""
            SELECT
                name,
                email,
                phone,
                segment,
                ROUND(churn_risk_score::numeric, 3) AS churn_risk_score,
                ROUND(lifetime_value::numeric, 2) AS lifetime_value,
                region
            FROM customers
            WHERE churn_risk_score >= {threshold}
            ORDER BY churn_risk_score DESC
            LIMIT 50
        """))
        return [dict(r._mapping) for r in result.fetchall()]
    except Exception as e:
        return {"error": str(e)}


@router.get("/acquisition")
async def acquisition_channels(db: AsyncSession = Depends(get_db)):
    try:
        result = await db.execute(text("""
            SELECT
                acquisition_channel,
                COUNT(*) AS customers,
                ROUND(AVG(lifetime_value)::numeric, 2) AS avg_ltv
            FROM customers
            GROUP BY acquisition_channel
            ORDER BY customers DESC
        """))
        return [dict(r._mapping) for r in result.fetchall()]
    except Exception as e:
        return {"error": str(e)}