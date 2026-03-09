import os
from pathlib import Path
import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session
from ..database.db import get_db
from ..database.tables import MonthlySales, Product
from ..services.data_cleaning import clean_sales_dataframe

router = APIRouter()


@router.post("/upload")
async def upload_sales_file(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename.lower().endswith((".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="Only Excel files are supported")

    upload_dir = Path(os.getenv("UPLOAD_DIR", "backend/uploads"))
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / file.filename

    with open(file_path, "wb") as f:
        f.write(await file.read())

    try:
        raw_df = pd.read_excel(file_path)
        cleaned = clean_sales_dataframe(raw_df)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid file format: {exc}") from exc

    upserted_products = 0
    inserted_sales = 0

    for _, row in cleaned.iterrows():
        product = db.query(Product).filter(Product.product_id == int(row["product_id"])).first()
        if product is None:
            product = Product(
                product_id=int(row["product_id"]),
                product_name=row["product_name"],
                category=row["category"],
            )
            db.add(product)
            upserted_products += 1
        else:
            product.product_name = row["product_name"]
            product.category = row["category"]

        db.add(
            MonthlySales(
                product_id=int(row["product_id"]),
                date=row["date"],
                sales=float(row["sales"]),
                profit=float(row["profit"]),
                marketing_spend=float(row["marketing_spend"]),
                region=row["region"],
            )
        )
        inserted_sales += 1

    db.commit()
    return {
        "message": "File uploaded and processed successfully",
        "products_upserted": upserted_products,
        "sales_rows_inserted": inserted_sales,
    }


@router.get("/sales")
def get_sales(db: Session = Depends(get_db)):
    rows = (
        db.query(MonthlySales, Product)
        .join(Product, Product.product_id == MonthlySales.product_id)
        .order_by(MonthlySales.date.asc())
        .all()
    )
    return [
        {
            "id": sale.id,
            "product_id": sale.product_id,
            "product_name": product.product_name,
            "category": product.category,
            "date": sale.date.isoformat(),
            "sales": sale.sales,
            "profit": sale.profit,
            "marketing_spend": sale.marketing_spend,
            "region": sale.region,
        }
        for sale, product in rows
    ]
