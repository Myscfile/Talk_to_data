"""Command-line app: ask a question, get SQL, results and a plain-language answer.

Flow per question:
  1. LLM decides: write SQL, ask to clarify, or say it can't answer.
  2. The SQL is checked (read-only, known pitfalls) before it runs.
  3. If a check fails or the query errors, the reason goes back to the LLM
     so it can fix its own query (up to MAX_FIX_ATTEMPTS times).
  4. The result is explained in plain language by a second LLM call.
"""

import logging
import sqlite3
from dataclasses import dataclass, field

from config import Config
from db import (
    check_pitfalls,
    connect_readonly,
    describe_schema,
    is_read_only,
    run_query,
)
from llm import ask_for_sql, build_system_prompt, explain

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


@dataclass
class Result:
    """Everything the user sees for one question."""

    status: str  # "ok", "clarify", "cannot_answer" or "error"
    message: str
    sql: str = ""
    columns: list[str] = field(default_factory=list)
    rows: list[tuple] = field(default_factory=list)


def _send_back(messages: list[dict], raw: str, feedback: str) -> None:
    """Add the model's previous reply plus our feedback, so it can retry."""
    messages.append({"role": "assistant", "content": raw})
    messages.append(
        {"role": "user", "content": f"{feedback} Reply with the corrected JSON."}
    )


def answer(conn: sqlite3.Connection, system: str, question: str) -> Result:
    """Answer one question, letting the model fix its own SQL if needed."""
    messages = [{"role": "user", "content": question}]
    sql = ""

    for attempt in range(Config.MAX_FIX_ATTEMPTS + 1):
        raw, decision = ask_for_sql(system, messages)
        action = decision["action"]
        msg = decision.get("message", "")
        sql = decision.get("sql", "")

        if action != "sql":
            return Result(status=action, message=msg)

        # Hard refusal: never retry a write attempt, just stop.
        if not is_read_only(sql):
            return Result(
                "cannot_answer", "Refused: only read-only queries are allowed.", sql
            )

        problem = check_pitfalls(sql)
        if problem:
            log.info("Guard rejected query (attempt %d): %s", attempt + 1, problem)
            _send_back(messages, raw, problem)
            continue

        try:
            cols, rows, truncated = run_query(conn, sql)
        except sqlite3.Error as e:
            log.info("Query failed (attempt %d): %s", attempt + 1, e)
            _send_back(messages, raw, f"That query failed: {e}.")
            continue

        explanation = explain(question, sql, msg, cols, rows, truncated)
        return Result("ok", explanation, sql, cols, rows)

    return Result("error", "Couldn't produce a working query. Try rephrasing.", sql)


LABELS = {
    "ok": "Answer",
    "clarify": "Need clarification",
    "cannot_answer": "Can't answer",
    "error": "Error",
}


def print_result(result: Result, max_rows: int = 10) -> None:
    """Print the SQL, a preview of the rows and the answer."""
    if result.sql:
        print(f"\nSQL:\n{result.sql}")
    if result.rows:
        print("\n" + " | ".join(result.columns))
        for row in result.rows[:max_rows]:
            print(" | ".join(str(v) for v in row))
        if len(result.rows) > max_rows:
            print(f"... ({len(result.rows)} rows)")
    print(f"\n{LABELS.get(result.status, 'Result')}: {result.message}\n")


def main() -> None:
    try:
        conn = connect_readonly(Config.DB_PATH)
    except FileNotFoundError as e:
        print(e)
        return

    # Built once at startup: restart the app after editing the notes in llm.py.
    system = build_system_prompt(describe_schema(conn))
    print("Ask a question about the data (Ctrl+C to quit).")

    while True:
        try:
            question = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not question:
            continue
        try:
            print_result(answer(conn, system, question))
        except Exception:
            # Top-level safety net (network down, API error, ...): keep the app
            # running and show the error instead of crashing the whole session.
            log.exception("Something went wrong answering that question")

    conn.close()


if __name__ == "__main__":
    main()
