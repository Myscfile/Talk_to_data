import json
import os

from cerebras.cloud.sdk import Cerebras

from config import Config
from db import connect_readonly, describe_schema
from dotenv import load_dotenv


load_dotenv()
client = Cerebras(api_key=os.environ["CEREBRAS_API_KEY"])

# Hand-written knowledge about the data. This is the biggest quality lever:
# make sure table/column names match YOUR database.
NOTES = """
- Data covers roughly Sept 2016 to Oct 2018. If the user says "last month",
  "this year" etc., say there is no data for the current date and offer the
  most recent period in the data instead.
- Timestamps are TEXT ('YYYY-MM-DD HH:MM:SS'). Use SQLite date functions:
  date(), strftime('%Y-%m', col), julianday(a) - julianday(b).
- customer_id is unique PER ORDER. To count real people / repeat buyers use
  customer_unique_id from the customers table.
- An order can have more than one review: aggregate reviews per order
  (e.g. AVG(review_score)) before joining, to avoid double-counting.
- order_status values: delivered, shipped, canceled, unavailable, invoiced,
  processing, created, approved.
- Late delivery = order_delivered_customer_date > order_estimated_delivery_date.
"""

SYSTEM = """You answer questions about an e-commerce SQLite database by writing SQL.

Schema (with sample rows):
{schema}

Domain notes:
{notes}

Reply with ONLY a JSON object, no other text:
{{"action": "sql" | "clarify" | "cannot_answer", "sql": "...", "message": "..."}}

- action="sql": the question is answerable. Write ONE read-only SQLite query
  (SELECT or WITH only). Use only tables/columns that exist above. Add a LIMIT
  unless the result is an aggregate. Put any assumption in "message"
  (e.g. "Interpreted 'best customers' as highest total spend").
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


DB_PATH = Config.DB_PATH
conn = connect_readonly(DB_PATH)
schema = describe_schema(conn)
system = build_system_prompt(schema)
_, decision = ask_for_sql(
    system, [{"role": "user", "content": "How many orders are there?"}]
)
print(decision)
