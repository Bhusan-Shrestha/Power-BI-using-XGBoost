from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from contextlib import asynccontextmanager
import asyncio
import logging
from core.database import engine, Base
from core.db_bootstrap import seed_default_users, normalize_product_categories
from api import forecast


logger = logging.getLogger(__name__)


async def _forecast_refresh_daemon() -> None:
    while True:
        try:
            await forecast.ensure_monthly_forecast_snapshot()
        except Exception as exc:
            logger.warning("Monthly forecast daemon run failed: %s", exc)
        await asyncio.sleep(24 * 60 * 60)


@asynccontextmanager
async def lifespan(app: FastAPI):
    forecast_task = None
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    # Ensure first-run environments always have baseline login accounts.
    await seed_default_users()
    await normalize_product_categories()
    try:
        await forecast.ensure_monthly_forecast_snapshot()
    except Exception as exc:
        # Startup should continue even if forecast generation temporarily fails.
        logger.warning("Monthly forecast bootstrap failed: %s", exc)
    forecast_task = asyncio.create_task(_forecast_refresh_daemon())
    yield
    if forecast_task:
        forecast_task.cancel()
        try:
            await forecast_task
        except asyncio.CancelledError:
            pass
    await engine.dispose()


app = FastAPI(
    title="AI-Powered BI Platform",
    description="Business Intelligence with ML forecasting and AI chatbot",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from api import auth, sales, inventory, customers, chatbot, dashboard, system

app.include_router(auth.router,      prefix="/api/auth",      tags=["Auth"])
app.include_router(dashboard.router, prefix="/api/dashboard", tags=["Dashboard"])
app.include_router(sales.router,     prefix="/api/sales",     tags=["Sales"])
app.include_router(inventory.router, prefix="/api/inventory", tags=["Inventory"])
app.include_router(customers.router, prefix="/api/customers", tags=["Customers"])
app.include_router(forecast.router,  prefix="/api/forecast",  tags=["Forecast"])
app.include_router(chatbot.router,   prefix="/api/chatbot",   tags=["Chatbot"])
app.include_router(system.router,    prefix="/api/system",    tags=["System"])


@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}


@app.get("/")
async def root():
    return {
        "message": "BI Platform API is running!",
        "docs": "http://localhost:8000/docs"
    }