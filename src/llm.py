import json
import os

from cerebras.cloud.sdk import Cerebras

from config import Config
from db import connect_readonly, describe_schema
from dotenv import load_dotenv


load_dotenv()
client = Cerebras(api_key=os.environ["CEREBRAS_API_KEY"])

NOTES = """
- Data covers Sept 2016 to Oct 2018 (order_purchase_timestamp). If the user
  says "last month", "this year", "recently" etc., there is no data for the
  real current date: say so in "message" and use the most recent period in
  the data instead.
- Timestamps are TEXT ('YYYY-MM-DD HH:MM:SS'). Use SQLite date functions:
  date(), strftime('%Y-%m', col), julianday(a) - julianday(b) for days between.
- CUSTOMERS: customer_id is a NEW id for EVERY order, so grouping by it always
  gives 1 order per customer. For anything per customer (top customers, orders
  per customer, repeat buyers, number of customers) JOIN customers and use
  customers.customer_unique_id. Example:
    SELECT c.customer_unique_id, COUNT(*) AS n_orders
    FROM orders o JOIN customers c ON c.customer_id = o.customer_id
    GROUP BY c.customer_unique_id ORDER BY n_orders DESC LIMIT 10;
- ORDER STATUS: 'delivered' = completed (~97% of orders). 'shipped' = still in
  transit (rare). Others: canceled, unavailable, invoiced, processing,
  created, approved. For "sold"/"completed" orders use 'delivered' and say so.
- REVIEWS: an order can have more than one review. Aggregate per order first
  (e.g. AVG(review_score) GROUP BY order_id) before joining to orders, or
  orders get double-counted. review_score is 1-5.
- DELIVERY: delivery dates are NULL for orders that never arrived; filter
  them out for delivery-time questions. Late = order_delivered_customer_date >
  order_estimated_delivery_date. Delivery time in days =
  julianday(order_delivered_customer_date) - julianday(order_purchase_timestamp).
- NOT IN THIS DATABASE: prices, revenue, payments, products, categories,
  sellers, customer age/gender. Questions needing these -> "cannot_answer".
"""

SYSTEM = """You answer questions about an e-commerce SQLite database by writing SQL.

Schema (with sample rows):
{schema}

Domain notes (follow these strictly, they override your assumptions):
{notes}

Reply with ONLY a JSON object, no other text:
{{"action": "sql" | "clarify" | "cannot_answer", "sql": "...", "message": "..."}}

- action="sql": the question is answerable. Write ONE read-only SQLite query
  (SELECT or WITH only). Use only tables/columns that exist above. Add a LIMIT
  unless the result is an aggregate. Put every assumption in "message"
  (e.g. "Interpreted 'best customers' as most orders").
- For any per-customer question, GROUP BY customers.customer_unique_id,
  NEVER by customer_id.
- action="clarify": too ambiguous to guess. Ask one short question in "message".
- action="cannot_answer": the data can't answer it (missing columns, off-topic,
  a request to change data, gibberish). Explain briefly in "message".
"""


def build_system_prompt(schema: str) -> str:
    return SYSTEM.format(schema=schema, notes=NOTES)


def parse_json(text: str) -> dict:
    """Models sometimes wrap JSON in ```json fences or add text; dig it out."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


def ask_for_sql(system: str, messages: list):
    """Returns (raw_text, decision_dict). raw_text is needed for the retry loop."""
    resp = client.chat.completions.create(
        model=Config.MODEL,
        messages=[{"role": "system", "content": system}] + messages,
        response_format={"type": "json_object"},
        temperature=0,
    )
    raw = resp.choices[0].message.content
    try:
        decision = parse_json(raw)
    except json.JSONDecodeError:
        decision = {
            "action": "cannot_answer",
            "message": "Model returned invalid JSON.",
        }
    return raw, decision


def explain(
    question: str, sql: str, assumption: str, cols, rows, truncated: bool
) -> str:
    preview = [dict(zip(cols, r)) for r in rows[:50]]
    prompt = (
        f"Question: {question}\n\nSQL used:\n{sql}\n\n"
        f"Assumptions: {assumption or 'none'}\n\n"
        f"Result ({len(rows)}{'+' if truncated else ''} rows, first 50 shown):\n"
        f"{json.dumps(preview, default=str, ensure_ascii=False)}\n\n"
        "Answer the question in 2-4 plain sentences using these results. "
        "Mention any assumption. If the result is empty, say so and suggest why "
        "(e.g. date outside the data range). Do not invent numbers."
    )
    resp = client.chat.completions.create(
        model=Config.MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return resp.choices[0].message.content.strip()
