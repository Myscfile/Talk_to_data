"""LLM layer: builds the prompt, asks the model for SQL, and explains results.

Contains no database code; app.py connects this to db.py.
"""

import hashlib
import json
import os

from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv

from config import Config

load_dotenv(Config.BASE_DIR / ".env")

if "CEREBRAS_API_KEY" not in os.environ:
    raise RuntimeError(
        "CEREBRAS_API_KEY is not set. Add it to a .env file in the project folder."
    )

client = Cerebras(api_key=os.environ["CEREBRAS_API_KEY"])

VALID_ACTIONS = {"sql", "clarify", "cannot_answer"}
PREVIEW_ROWS = 50  # rows sent to the explainer; more costs tokens without adding much

# Domain knowledge the model can't infer from the schema alone.
# Every rule here exists because the model got it wrong without it.
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
  orders get double-counted. review_score is 1-5. About 1% of orders have no
  review at all: use LEFT JOIN when the question is about all orders, and a
  plain JOIN only when it is about reviewed orders.
- DELIVERY: delivery dates are NULL for orders that never arrived; filter
  them out for delivery-time questions with
  order_delivered_customer_date IS NOT NULL. Do not use
  order_status = 'delivered' as a stand-in: a few delivered orders have no
  delivery date and a few non-delivered ones have one. Late = order_delivered_customer_date >
  order_estimated_delivery_date. Delivery time in days =
  julianday(order_delivered_customer_date) - julianday(order_purchase_timestamp).
- NOT IN THIS DATABASE: prices, revenue, payments, products, categories,
  sellers, customer age/gender. Questions needing these -> "cannot_answer".
"""

# Double braces {{ }} are literal braces; single braces are filled by .format().
SYSTEM = """You answer questions about an e-commerce SQLite database by writing SQL.

Schema (with sample rows):
{schema}

Domain notes (follow these strictly, they override your assumptions):
{notes}

Reply with ONLY a JSON object, no other text:
{{"action": "sql" | "clarify" | "cannot_answer", "sql": "...", "message": "...", "suggestions": []}}

- action="sql": the question is answerable. Write ONE read-only SQLite query
  (SELECT or WITH only). Use only tables/columns that exist above. Add a LIMIT
  unless the result is an aggregate. Put every assumption in "message"
  (e.g. "Interpreted 'best customers' as most orders").
- For any per-customer question, GROUP BY customers.customer_unique_id,
  NEVER by customer_id.
- Add context columns AFTER the figure that was asked for (it must be the first
  column): when counting a subset (late, canceled, repeat customers), also
  return the size of the whole group it comes from; when averaging, also
  return COUNT(*) of the rows averaged.
- action="clarify": too ambiguous to guess. Ask one short question in "message".
  In "suggestions" give 2-3 complete questions that settle the ambiguity
  (for "best customers": one by number of orders, one by review score).
- action="cannot_answer": the data can't answer it (missing columns, off-topic,
  a request to change data, gibberish). In "message" explain why, saying which
  data is missing. In "suggestions" give 2-3 complete questions that this data
  CAN answer and that are close to what the user wanted.
- Every suggestion must be answerable with the tables above and must make sense
  on its own (never "and for 2018?"). For action="sql" leave "suggestions" empty.
"""


def build_system_prompt(schema: str) -> str:
    """Combine the schema description and domain notes into the system prompt."""
    return SYSTEM.format(schema=schema, notes=NOTES)


def prompt_hash() -> str:
    """Short fingerprint of the prompt text (template + notes, not the schema).

    Stored with a validated model so we can tell when the prompt has been
    edited since the model was last evaluated.
    """
    return hashlib.sha256((SYSTEM + NOTES).encode("utf-8")).hexdigest()[:12]


def parse_json(text: str) -> dict:
    """Parse the model's reply as JSON, tolerating extra text around it.

    Models sometimes wrap the JSON in ```json fences or add a sentence before
    it, so if a direct parse fails we take everything from the first '{' to
    the last '}'.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


def ask_for_sql(system: str, messages: list[dict]) -> tuple[str, dict]:
    """Ask the model to decide how to handle the question.

    Returns (raw_text, decision). decision always has a valid "action";
    malformed replies are turned into "cannot_answer" so the caller never has
    to handle broken output. raw_text is kept so the retry loop in app.py can
    show the model its own previous answer.
    """
    resp = client.chat.completions.create(
        model=Config.MODEL,
        messages=[{"role": "system", "content": system}] + messages,
        response_format={"type": "json_object"},
        temperature=0,  # same question -> same SQL, which makes testing reproducible
    )
    raw = resp.choices[0].message.content or ""

    try:
        decision = parse_json(raw)
    except json.JSONDecodeError:
        return raw, {
            "action": "cannot_answer",
            "message": "Model returned invalid JSON.",
        }

    if decision.get("action") not in VALID_ACTIONS:
        return raw, {
            "action": "cannot_answer",
            "message": "Model returned an unknown action.",
        }
    return raw, decision


def explain(
    question: str,
    sql: str,
    assumption: str,
    cols: list[str],
    rows: list[tuple],
    truncated: bool,
) -> str:
    """Turn the query result into a short plain-language answer.

    The model only sees the actual result rows, and is told not to invent
    numbers, so the answer stays grounded in what the database returned.
    """
    preview = [dict(zip(cols, r)) for r in rows[:PREVIEW_ROWS]]
    prompt = (
        f"Question: {question}\n\nSQL used:\n{sql}\n\n"
        f"Assumptions: {assumption or 'none'}\n\n"
        f"Result ({len(rows)}{'+' if truncated else ''} rows, "
        f"first {PREVIEW_ROWS} shown):\n"
        f"{json.dumps(preview, default=str, ensure_ascii=False)}\n\n"
        "Answer the question in 2-4 plain sentences using these results. "
        "Do not invent numbers. The assumptions are shown to the reader "
        "separately, so do not repeat them.\n\n"
        "Then add a short 'Context' list of at most 3 bullets, only where it "
        "applies:\n"
        "- Base: if the result holds a subset and the whole it comes from, give "
        "the share, e.g. '7,827 of 96,476 delivered orders (8.1%)'.\n"
        "- Thin data: if a figure rests on few rows (under about 100), or on the "
        "first or last months of the data (it runs Sept 2016 to Oct 2018 and is "
        "sparse at both ends), say it may not be representative.\n"
        "- Reference: if the result offers a comparison (overall vs month, "
        "average vs median), put the figure next to it.\n"
        "- Other reading: if the question could reasonably mean something else "
        "and give a different answer, name that reading briefly, without a "
        "number you were not shown.\n\n"
        "If the result is empty, say so and suggest why (e.g. date outside the "
        "data range). If the result looks suspicious (e.g. all values tied, all "
        "counts 1, or suspiciously few rows), say so plainly instead of "
        "presenting it as a finding.\n\n"
        "Finish with a line 'FOLLOW-UPS:' and then up to 3 lines starting with "
        "'- ', each a complete question that makes sense on its own, goes a "
        "step deeper than this answer, and can be answered from the orders, "
        "customers and order_reviews tables (never prices, revenue, products or "
        "sellers). Write nothing after them."
    )
    resp = client.chat.completions.create(
        model=Config.MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return (resp.choices[0].message.content or "").strip()


if __name__ == "__main__":
    from db import connect_readonly, describe_schema

    conn = connect_readonly(Config.DB_PATH)
    system = build_system_prompt(describe_schema(conn))
    _, decision = ask_for_sql(
        system, [{"role": "user", "content": "How many orders are there?"}]
    )
    print(decision)
