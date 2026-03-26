import asyncio
import os
import sys

from sqlalchemy import text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.database import engine


async def ensure_schema() -> None:
    async with engine.begin() as conn:
        await conn.execute(text("CREATE SCHEMA IF NOT EXISTS biuser AUTHORIZATION biuser"))


if __name__ == "__main__":
    asyncio.run(ensure_schema())
    print("Schema check complete")
