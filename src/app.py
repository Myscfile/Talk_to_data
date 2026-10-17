"""Command-line app: ask a question, get SQL, results and a plain-language answer.

Flow per question:
  1. LLM decides: write SQL, ask to clarify, or say it can't answer.
  2. The SQL is checked (read-only, known pitfalls) before it runs.
  3. If a check fails or the query errors, the reason goes back to the LLM
     so it can fix its own query (up to MAX_FIX_ATTEMPTS times).
  4. The result is explained in plain language by a second LLM call.
"""

import json
import logging
import sqlite3
import time
import uuid
from dataclasses import dataclass, field

from config import Config
from db import (
    check_pitfalls,
    connect_readonly,
    describe_schema,
    is_read_only,
    run_query,
)
from llm import ask_for_sql, build_system_prompt, explain, prompt_hash
from logger import log_interaction, setup_logging

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


def answer(
    conn: sqlite3.Connection, system: str, question: str, question_id: str = ""
) -> Result:
    """Answer one question, letting the model fix its own SQL if needed.

    question_id only tags the diagnostic log lines for this question.
    """
    messages = [{"role": "user", "content": question}]
    sql = ""

    for attempt in range(Config.MAX_FIX_ATTEMPTS + 1):
        raw, decision = ask_for_sql(system, messages)
        action = decision["action"]
        msg = decision.get("message", "")
        sql = decision.get("sql", "")
        log.info(
            "[%s] attempt %d: action=%s sql=%s",
            question_id, attempt + 1, action, sql,
        )

        if action != "sql":
            return Result(status=action, message=msg)

        # Hard refusal: never retry a write attempt, just stop.
        if not is_read_only(sql):
            log.warning("[%s] refused non-read-only query: %s", question_id, sql)
            return Result(
                "cannot_answer", "Refused: only read-only queries are allowed.", sql
            )

        problem = check_pitfalls(sql)
        if problem:
            log.info(
                "[%s] guard rejected (attempt %d): %s",
                question_id, attempt + 1, problem,
            )
            _send_back(messages, raw, problem)
            continue

        try:
            cols, rows, truncated = run_query(conn, sql)
        except sqlite3.Error as e:
            log.info(
                "[%s] query failed (attempt %d): %s", question_id, attempt + 1, e
            )
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


def warn_if_not_validated() -> None:
    """Warn at startup if the model/prompt in use hasn't passed the evaluation.

    Only a warning: the app still runs, but the answers are unverified.
    """
    try:
        validated = json.loads(Config.VALIDATED_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        log.warning("No validated model on record. Run eval/run_eval.py.")
        return
    if validated.get("model") != Config.MODEL:
        log.warning(
            "Model %s is not the validated one (%s). Run eval/run_eval.py.",
            Config.MODEL, validated.get("model"),
        )
    elif validated.get("prompt_hash") != prompt_hash():
        log.warning("The prompt changed since validation. Run eval/run_eval.py.")


def main() -> None:
    setup_logging()
    log.info("start: db=%s model=%s", Config.DB_PATH, Config.MODEL)
    warn_if_not_validated()
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
        question_id = uuid.uuid4().hex[:8]
        log.info("[%s] question: %s", question_id, question)
        started = time.perf_counter()
        try:
            result = answer(conn, system, question, question_id)
            print_result(result)
        except Exception:
            # Top-level safety net (network down, API error, ...): keep the app
            # running and show the error instead of crashing the whole session.
            log.exception("[%s] Something went wrong answering that question", question_id)
            result = Result("error", "")
        log_interaction(
            question_id=question_id,
            question=question,
            status=result.status,
            answer=result.message,
            sql=result.sql,
            row_count=len(result.rows),
            seconds=round(time.perf_counter() - started, 2),
        )

    conn.close()


if __name__ == "__main__":
    main()
