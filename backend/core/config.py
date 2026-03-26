from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(
            str(Path(__file__).resolve().parent.parent / ".env"),
            str(Path(__file__).resolve().parent.parent.parent / ".env"),
        ),
        extra="ignore",
    )

    APP_NAME: str = "BI Platform"
    APP_ENV: str = "development"
    SECRET_KEY: str = "bi-platform-secret-key-2024"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440
    ALGORITHM: str = "HS256"
    AUTO_CREATE_TABLES: bool = True
    CORS_ORIGINS: str = "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173"

    # PostgreSQL connection
    DATABASE_URL: str = "postgresql+asyncpg://biuser:bipassword@localhost:5432/biplatform"

    OPENAI_API_KEY: str = ""


settings = Settings()