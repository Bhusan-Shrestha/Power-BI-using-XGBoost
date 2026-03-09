from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api.upload import router as upload_router
from .api.prediction import router as prediction_router
from .api.analytics import router as analytics_router
from .database.db import Base, engine

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Sales AI Platform API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload_router, tags=["upload"])
app.include_router(prediction_router, tags=["prediction"])
app.include_router(analytics_router)


@app.get("/")
def health_check():
    return {"status": "ok", "service": "sales-ai-platform"}
