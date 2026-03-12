from fastapi import APIRouter, File, HTTPException, UploadFile

from ..services.prediction_service import PredictionService

router = APIRouter()
service = PredictionService()


@router.post("/upload")
async def upload_sales_file(file: UploadFile = File(...)):
    try:
        result = await service.save_upload(file)
        overview = service.get_dashboard_overview()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Upload failed: {exc}") from exc

    return {
        "message": (
            f"Data uploaded for {result['latest_month']}. Inserted: {result['rows']}, Updated: {result.get('updated', 0)}."
            if result["merged"]
            else f"Data uploaded. {result['latest_month']} already exists, so no new rows were added."
        ),
        "input_file": result["path"].name,
        "input_path": str(result["path"]),
        "products_upserted": result["products"],
        "sales_rows_inserted": result["rows"],
        "sales_rows_updated": result.get("updated", 0),
        "merged": result["merged"],
        "latest_month": result["latest_month"],
        "overview": overview,
    }
