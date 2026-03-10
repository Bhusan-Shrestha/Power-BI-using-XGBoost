from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from ..services.prediction_service import PredictionService

router = APIRouter()
service = PredictionService()


@router.post("/predict")
async def generate_predictions(request: Request):
    content_type = request.headers.get("content-type", "").lower()

    # Support multipart predict requests where the file is sent directly to /predict.
    if "multipart/form-data" in content_type:
        form = await request.form()
        file_obj: Any = form.get("file")
        if file_obj is not None and hasattr(file_obj, "filename"):
            await service.save_upload(file_obj)

    # Keep compatibility with JSON payloads used by older clients.
    elif "application/json" in content_type:
        await request.body()

    try:
        result = service.run_prediction()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    return {
        "message": "Prediction generated successfully",
        "count": len(result.get("predictions", [])),
        "month": result.get("predicting_month"),
        "input_file": result.get("input_file"),
        "output_file": result.get("download_file"),
        "download_url": result.get("download_url"),
        "predictions": result.get("predictions", []),
    }


@router.get("/predictions")
def get_predictions():
    return service.read_latest_predictions()


@router.get("/download/{filename}")
def download_prediction_file(filename: str):
    path = Path(service.output_dir) / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")

    return FileResponse(
        path=str(path),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=filename,
    )
