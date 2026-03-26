from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from core.config import settings
import re
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db

router = APIRouter()

SYSTEM_PROMPT = """You are an expert Business Intelligence user assistant.
You help analyze sales, inventory, and customer data.
Keep answers short, clear and actionable."""

NLQ_PATTERNS = {
    r"top.*product|best.*product|highest.*sale": "top_products",
    r"low.*stock|reorder|out of stock": "low_stock",
    r"churn|at.?risk|losing customer": "churn_risk",
    r"revenue|income|earning|sales total": "revenue",
    r"region|area|location|province": "region",
    r"customer|client|buyer": "customers",
    r"inventory|stock|warehouse": "inventory",
    r"forecast|predict|future|next month": "forecast",
    r"anomal|unusual|spike|drop": "anomalies",
    r"growth|increase|trend": "growth",
    r"channel|online|retail|wholesale": "channel",
    r"hello|hi|hey|help": "greeting",
}


def _format_npr(value: float | int | None) -> str:
    num = float(value or 0)
    return f"NPR {num:,.2f}"


def _match_intent(query: str) -> str | None:
    q = query.lower()
    for pattern, intent in NLQ_PATTERNS.items():
        if re.search(pattern, q):
            return intent
    return None


async def _data_backed_answer(intent: str, db: AsyncSession) -> str | None:
    if intent == "greeting":
        return "Hello! Ask one of the predefined BI questions and I will answer from your current data."

    if intent == "top_products":
        rows = (await db.execute(text("""
            SELECT p.name, SUM(s.total_amount) AS revenue
            FROM sales s
            JOIN products p ON p.id = s.product_id
            WHERE s.sale_date >= NOW() - INTERVAL '30 days'
            GROUP BY p.name
            ORDER BY revenue DESC
            LIMIT 5
        """))).fetchall()
        if not rows:
            return "No sales data found for the last 30 days."
        lines = [f"{i + 1}. {r.name}: {_format_npr(r.revenue)}" for i, r in enumerate(rows)]
        return "Top products by revenue (last 30 days):\n" + "\n".join(lines)

    if intent == "low_stock":
        rows = (await db.execute(text("""
            SELECT p.name, i.quantity_on_hand, i.reorder_point
            FROM inventory i
            JOIN products p ON p.id = i.product_id
            WHERE i.quantity_on_hand <= i.reorder_point
            ORDER BY i.quantity_on_hand ASC, p.name ASC
            LIMIT 5
        """))).fetchall()
        if not rows:
            return "Great news: there are no low stock alerts right now."
        details = ", ".join([f"{r.name} ({r.quantity_on_hand}/{r.reorder_point})" for r in rows])
        return f"Low stock alerts detected for {len(rows)} products. Critical items: {details}."

    if intent == "churn_risk":
        row = (await db.execute(text("""
            SELECT COUNT(*) AS total,
                   COUNT(CASE WHEN churn_risk_score >= 0.8 THEN 1 END) AS high
            FROM customers
        """))).fetchone()
        top = (await db.execute(text("""
            SELECT name, churn_risk_score
            FROM customers
            WHERE churn_risk_score >= 0.8
            ORDER BY churn_risk_score DESC
            LIMIT 3
        """))).fetchall()
        if not row or int(row.total or 0) == 0:
            return "No customer records are available yet."
        if int(row.high or 0) == 0:
            return f"No customers are currently in high churn risk. Total customers: {int(row.total)}."
        top_txt = ", ".join([f"{t.name} ({float(t.churn_risk_score):.2f})" for t in top])
        return f"High churn risk customers: {int(row.high)} out of {int(row.total)}. Top risks: {top_txt}."

    if intent == "revenue":
        row = (await db.execute(text("""
            SELECT
                COALESCE(SUM(CASE WHEN sale_date >= NOW() - INTERVAL '30 days' THEN total_amount END), 0) AS curr,
                COALESCE(SUM(CASE WHEN sale_date < NOW() - INTERVAL '30 days'
                                   AND sale_date >= NOW() - INTERVAL '60 days' THEN total_amount END), 0) AS prev
            FROM sales
        """))).fetchone()
        region = (await db.execute(text("""
            SELECT COALESCE(region, 'Unknown') AS region, SUM(total_amount) AS revenue
            FROM sales
            WHERE sale_date >= NOW() - INTERVAL '30 days'
            GROUP BY COALESCE(region, 'Unknown')
            ORDER BY revenue DESC
            LIMIT 1
        """))).fetchone()
        curr = float(row.curr or 0)
        prev = float(row.prev or 0)
        growth = ((curr - prev) / prev * 100) if prev > 0 else 0
        region_txt = region.region if region else "N/A"
        return f"Revenue (last 30 days): {_format_npr(curr)}. Change vs previous 30 days: {growth:.2f}%. Best region: {region_txt}."

    if intent == "region":
        rows = (await db.execute(text("""
            SELECT COALESCE(region, 'Unknown') AS region, SUM(total_amount) AS revenue
            FROM sales
            WHERE sale_date >= NOW() - INTERVAL '30 days'
            GROUP BY COALESCE(region, 'Unknown')
            ORDER BY revenue DESC
        """))).fetchall()
        if not rows:
            return "No regional sales data available for the last 30 days."
        total = sum(float(r.revenue or 0) for r in rows)
        parts = [f"{r.region} {((float(r.revenue or 0) / total) * 100 if total > 0 else 0):.1f}%" for r in rows[:5]]
        return "Sales by region (last 30 days): " + ", ".join(parts) + "."

    if intent == "customers":
        summary = (await db.execute(text("""
            SELECT COUNT(*) AS total,
                   COUNT(CASE WHEN is_active THEN 1 END) AS active,
                   COALESCE(AVG(lifetime_value), 0) AS avg_ltv
            FROM customers
        """))).fetchone()
        segments = (await db.execute(text("""
            SELECT segment, COALESCE(AVG(lifetime_value), 0) AS avg_ltv
            FROM customers
            GROUP BY segment
            ORDER BY avg_ltv DESC
            LIMIT 2
        """))).fetchall()
        seg_txt = ", ".join([f"{s.segment}: {_format_npr(s.avg_ltv)} avg LTV" for s in segments]) if segments else "No segment data"
        return f"Customers: {int(summary.total or 0)} total, {int(summary.active or 0)} active, overall avg LTV {_format_npr(summary.avg_ltv)}. {seg_txt}."

    if intent == "inventory":
        row = (await db.execute(text("""
            SELECT COUNT(*) AS total,
                   SUM(CASE WHEN quantity_on_hand > reorder_point THEN 1 ELSE 0 END) AS in_stock,
                   SUM(CASE WHEN quantity_on_hand > 0 AND quantity_on_hand <= reorder_point THEN 1 ELSE 0 END) AS low_stock,
                   SUM(CASE WHEN quantity_on_hand <= 0 THEN 1 ELSE 0 END) AS out_of_stock
            FROM inventory
        """))).fetchone()
        return (
            f"Inventory summary: {int(row.total or 0)} products, "
            f"{int(row.in_stock or 0)} in stock, {int(row.low_stock or 0)} low stock, "
            f"{int(row.out_of_stock or 0)} out of stock."
        )

    if intent == "forecast":
        row = (await db.execute(text("""
            SELECT forecast_month, summary, generated_at
            FROM monthly_sales_forecasts
            ORDER BY forecast_month DESC
            LIMIT 1
        """))).fetchone()
        if not row:
            return "Monthly model forecast is not available yet. Open Forecast page once to generate it."

        summary = row.summary or {}
        month = str(row.forecast_month)
        total_units = float(summary.get("total_predicted_units", 0) or 0)
        top_product = str(summary.get("top_product", "-"))
        top_units = float(summary.get("top_predicted_units", 0) or 0)
        return (
            f"Model forecast for next 30 days ({month}): total {total_units:,.0f} units. "
            f"Top product: {top_product} ({top_units:,.0f} units)."
        )

    if intent == "anomalies":
        rows = (await db.execute(text("""
            WITH daily AS (
                SELECT DATE(sale_date) AS day, SUM(total_amount) AS amount
                FROM sales
                WHERE sale_date >= NOW() - INTERVAL '30 days'
                GROUP BY DATE(sale_date)
            ), stats AS (
                SELECT AVG(amount) AS avg_amount FROM daily
            )
            SELECT d.day, d.amount
            FROM daily d, stats s
            WHERE d.amount >= s.avg_amount * 1.8 OR d.amount <= s.avg_amount * 0.5
            ORDER BY d.day DESC
            LIMIT 5
        """))).fetchall()
        if not rows:
            return "No major sales anomalies detected in the last 30 days."
        items = ", ".join([f"{r.day}: {_format_npr(r.amount)}" for r in rows])
        return f"Potential anomalies in last 30 days: {items}."

    if intent == "growth":
        row = (await db.execute(text("""
            SELECT
                COALESCE(SUM(CASE WHEN sale_date >= NOW() - INTERVAL '30 days' THEN total_amount END), 0) AS rev_curr,
                COALESCE(SUM(CASE WHEN sale_date < NOW() - INTERVAL '30 days'
                                   AND sale_date >= NOW() - INTERVAL '60 days' THEN total_amount END), 0) AS rev_prev,
                COALESCE(COUNT(CASE WHEN sale_date >= NOW() - INTERVAL '30 days' THEN 1 END), 0) AS ord_curr,
                COALESCE(COUNT(CASE WHEN sale_date < NOW() - INTERVAL '30 days'
                                     AND sale_date >= NOW() - INTERVAL '60 days' THEN 1 END), 0) AS ord_prev
            FROM sales
        """))).fetchone()
        rev_curr = float(row.rev_curr or 0)
        rev_prev = float(row.rev_prev or 0)
        ord_curr = int(row.ord_curr or 0)
        ord_prev = int(row.ord_prev or 0)
        rev_growth = ((rev_curr - rev_prev) / rev_prev * 100) if rev_prev > 0 else 0
        ord_growth = ((ord_curr - ord_prev) / ord_prev * 100) if ord_prev > 0 else 0
        return f"Growth trend (30d vs previous 30d): Revenue {rev_growth:.2f}%, Orders {ord_growth:.2f}%."

    if intent == "channel":
        rows = (await db.execute(text("""
            SELECT channel, SUM(total_amount) AS revenue
            FROM sales
            WHERE sale_date >= NOW() - INTERVAL '30 days'
            GROUP BY channel
            ORDER BY revenue DESC
        """))).fetchall()
        if not rows:
            return "No channel data available for the last 30 days."
        best = rows[0]
        breakdown = ", ".join([f"{r.channel}: {_format_npr(r.revenue)}" for r in rows])
        return f"Best performing sales channel: {best.channel}. Breakdown: {breakdown}."

    return None


def nlq_match(query: str):
    return _match_intent(query)


async def ask_gpt(message: str):
    if not settings.OPENAI_API_KEY:
        return None
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": message},
            ],
            max_tokens=400,
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        return None


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


@router.post("/chat")
async def chat(req: ChatRequest, db: AsyncSession = Depends(get_db)):
    # 1. Match predefined intent and answer from current database data
    intent = nlq_match(req.message)
    if intent:
        result = await _data_backed_answer(intent, db)
        if result:
            return {
                "type": "insight",
                "intent": intent,
                "query": req.message,
                "response": result,
                "source": "local",
            }

    # 2. Try OpenAI if key is set
    if settings.OPENAI_API_KEY:
        gpt_result = await ask_gpt(req.message)
        if gpt_result:
            return {"type": "ai", "query": req.message,
                    "response": gpt_result, "source": "openai"}

    # 3. Fallback
    return {
        "type": "fallback",
        "query": req.message,
        "response": "Try asking about: sales, inventory, customers, forecasts, or anomalies.",
        "source": "fallback",
    }


@router.get("/suggestions")
async def get_suggestions():
    return {
        "suggestions": [
            "What are my top products by revenue?",
            "Which customers are at churn risk?",
            "Show me low stock alerts",
            "What is the revenue forecast for next month?",
            "Which region has the highest sales?",
            "Are there any anomalies in revenue?",
            "What is our growth trend?",
            "Which sales channel performs best?",
        ]
    }


@router.websocket("/ws/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            intent = nlq_match(data)
            if intent:
                await websocket.send_text(
                    "Predefined question recognized. Use /chat endpoint for data-backed answer."
                )
                continue
            await websocket.send_text(
                "Ask me about sales, inventory, customers or forecasts."
            )
    except WebSocketDisconnect:
        pass