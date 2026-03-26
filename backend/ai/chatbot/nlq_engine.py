from typing import Optional, Dict, Any
import re


QUERY_PATTERNS = {
    r"(top|best)\s*(\d+)?\s*product": "top_products",
    r"revenue\s*(today|this week|this month)": "revenue_period",
    r"low\s*stock|reorder": "low_stock",
    r"churn|at risk customer": "churn_risk",
    r"(sales|revenue) by (region|country)": "sales_by_region",
    r"anomal|unusual|spike|drop": "anomalies",
}

CANNED_QUERIES = {
    "top_products": "SELECT p.name, SUM(s.total_amount) AS revenue FROM sales s JOIN products p ON s.product_id = p.id GROUP BY p.name ORDER BY revenue DESC LIMIT 10",
    "low_stock": "SELECT p.name, i.quantity_on_hand, i.reorder_point FROM inventory i JOIN products p ON i.product_id = p.id WHERE i.quantity_on_hand <= i.reorder_point",
    "churn_risk": "SELECT name, churn_risk_score, lifetime_value FROM customers WHERE churn_risk_score >= 0.6 ORDER BY churn_risk_score DESC LIMIT 20",
    "sales_by_region": "SELECT region, SUM(total_amount) AS revenue FROM sales WHERE sale_date >= NOW() - INTERVAL '30 days' GROUP BY region ORDER BY revenue DESC",
}


class NLQEngine:
    async def process(self, query: str) -> Optional[Dict[str, Any]]:
        query_lower = query.lower()
        for pattern, query_type in QUERY_PATTERNS.items():
            if re.search(pattern, query_lower):
                sql = CANNED_QUERIES.get(query_type)
                if sql:
                    return {"type": query_type, "sql": sql, "description": f"Query: {query_type.replace('_', ' ').title()}"}
        return None
