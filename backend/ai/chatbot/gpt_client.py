from openai import AsyncOpenAI
from core.config import settings
from typing import AsyncGenerator

client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

SYSTEM_PROMPT = """
You are an expert Business Intelligence user assistant.
You have access to data about Sales, Inventory, and Customers.
Answer questions clearly and concisely. When presenting numbers,
format them with proper units (e.g. $12,450 or 24.5%).
If asked to chart something, describe the data you would use.
"""


class GPTClient:
    async def ask(self, message: str, context: str = "") -> str:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": message},
            ],
            max_tokens=500,
            temperature=0.3,
        )
        return response.choices[0].message.content

    async def stream(self, message: str) -> AsyncGenerator[str, None]:
        stream = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": message},
            ],
            stream=True,
            max_tokens=500,
        )
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
