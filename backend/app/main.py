from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.prediction import router as prediction_router
from .api.upload import router as upload_router

app = FastAPI(title="Sales AI Platform API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload_router, tags=["upload"])
app.include_router(prediction_router, tags=["prediction"])


@app.get("/")
def health_check():
    return {"status": "ok", "service": "sales-ai-platform"}
