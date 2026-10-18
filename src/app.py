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
from llm import (
    ask_for_sql,
    build_system_prompt,
    explain,
    prompt_hash,
    reset_usage,
    usage,
)
from logger import TOKEN_FIELDS, log_interaction, setup_logging
from suggestions import clean_questions, split_followups

log = logging.getLogger(__name__)


@dataclass
class Result:
    """Everything the user sees for one question."""

    status: str  # "ok", "clarify", "cannot_answer" or "error"
    message: str
    sql: str = ""
    columns: list[str] = field(default_factory=list)
    rows: list[tuple] = field(default_factory=list)
    assumption: str = ""  # how the model read the question, shown separately
    suggestions: list[str] = field(default_factory=list)  # questions to try next
    query_seconds: float = 0.0  # time spent running SQL, over all attempts
    llm_seconds: float = 0.0  # time spent waiting for the model, over all calls
    llm_calls: int = 0  # number of model calls (SQL attempts plus the explanation)
    # Tokens over all model calls; reasoning is part of completion, cached part of prompt.
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    total_tokens: int = 0


def _examples() -> list[str]:
    """Questions known to work, offered when the model has nothing better."""
    return list(Config.EXAMPLE_QUESTIONS[:3])


_years: set[int] = set()  # the years the data covers, read from the database once


def _data_years(conn: sqlite3.Connection) -> set[int] | None:
    """Years that have orders, so a suggested follow ups can't point outside the data."""
    if not _years:
        first, last = conn.execute(
            "SELECT MIN(order_purchase_timestamp), MAX(order_purchase_timestamp) FROM orders"
        ).fetchone()
        if first and last:
            _years.update(range(int(first[:4]), int(last[:4]) + 1))
    return _years or None


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

    This is the whole pipeline for one question. It returns a Result, and every
    exit path fills in the same timing and token counts.

    The loop runs at most MAX_FIX_ATTEMPTS + 1 times. Each pass:

    1. Asks the model for a decision (ask_for_sql). The model answers with one of
       three actions:
       - "clarify" or "cannot_answer": return at once, with the model's message
         and suggested questions (cannot_answer falls back to the example questions
         if the model gave none).
       - "sql": go on to the checks below.
    2. Refuses any query that is not a single read-only SELECT/WITH (is_read_only).
       A refusal is final: the query is never retried.
    3. Checks the query for known data pitfalls (check_pitfalls). If it fails,
       the reason is sent back to the model and the loop continues.
    4. Runs the query (run_query) with a time limit and a row cap. If SQLite
       reports an error or the time limit is hit, the error is sent back to the
       model and the loop continues.
    5. If the query ran, asks the model to explain the rows (explain). The
       explanation and the follow-up questions are split apart (split_followups),
       and the function returns status "ok".

    Feedback to the model (_send_back) adds the model's own previous reply and the
    reason it failed, so the next attempt can correct it. If every attempt fails,
    the function returns status "error" with the last SQL and the example questions.

    question_id only tags the diagnostic log lines for this question. The
    token counters are reset here so each question's tokens are counted alone.
    """
    messages = [{"role": "user", "content": question}]
    sql = ""
    query_seconds = 0.0  # summed over attempts, so a slow failed query still counts
    llm_seconds, llm_calls = 0.0, 0  # the same for time spent waiting on the model

    reset_usage()  # count this question's tokens from zero

    def timings() -> dict:
        """Time and token counters, for whichever Result this question ends with."""
        return {
            **usage(),
            "query_seconds": round(query_seconds, 3),
            "llm_seconds": round(llm_seconds, 3),
            "llm_calls": llm_calls,
        }

    for attempt in range(Config.MAX_FIX_ATTEMPTS + 1):
        llm_started = time.perf_counter()
        tokens_before = usage()["total_tokens"]
        raw, decision = ask_for_sql(system, messages)
        took = time.perf_counter() - llm_started
        tokens = usage()["total_tokens"] - tokens_before
        llm_seconds, llm_calls = llm_seconds + took, llm_calls + 1
        action = decision["action"]
        msg = decision.get("message", "")
        sql = decision.get("sql", "")
        log.info(
            "[%s] attempt %d: action=%s (model %.2f s, %d tokens) sql=%s",
            question_id,
            attempt + 1,
            action,
            took,
            tokens,
            sql,
        )

        if action != "sql":
            suggestions = clean_questions(
                decision.get("suggestions"), _data_years(conn)
            )
            if not suggestions and action == "cannot_answer":
                suggestions = _examples()
            return Result(
                status=action,
                message=msg,
                suggestions=suggestions,
                **timings(),
            )

        # Hard refusal: never retry a write attempt, just stop.
        if not is_read_only(sql):
            log.warning("[%s] refused non-read-only query: %s", question_id, sql)
            return Result(
                "cannot_answer",
                "Refused: only read-only queries are allowed.",
                sql,
                suggestions=_examples(),
                **timings(),
            )

        problem = check_pitfalls(sql)
        if problem:
            log.info(
                "[%s] guard rejected (attempt %d): %s",
                question_id,
                attempt + 1,
                problem,
            )
            _send_back(messages, raw, problem)
            continue

        query_started = time.perf_counter()
        try:
            cols, rows, truncated = run_query(conn, sql)
        except sqlite3.Error as e:
            took = time.perf_counter() - query_started
            query_seconds += took
            log.info(
                "[%s] query failed after %.2f s (attempt %d): %s",
                question_id,
                took,
                attempt + 1,
                e,
            )
            _send_back(messages, raw, f"That query failed: {e}.")
            continue
        took = time.perf_counter() - query_started
        query_seconds += took
        log.info("[%s] query ran in %.3f s, %d rows", question_id, took, len(rows))

        llm_started = time.perf_counter()
        tokens_before = usage()["total_tokens"]
        explained = explain(question, sql, msg, cols, rows, truncated)
        took = time.perf_counter() - llm_started
        tokens = usage()["total_tokens"] - tokens_before
        llm_seconds, llm_calls = llm_seconds + took, llm_calls + 1
        log.info(
            "[%s] explanation written (model %.2f s, %d tokens)",
            question_id,
            took,
            tokens,
        )
        explanation, follow_ups = split_followups(explained, _data_years(conn))
        return Result(
            "ok",
            explanation,
            sql,
            cols,
            rows,
            assumption=msg,
            suggestions=follow_ups,
            **timings(),
        )

    return Result(
        "error",
        "Couldn't produce a working query. Try rephrasing, or pick one of these.",
        sql,
        suggestions=_examples(),
        **timings(),
    )


def ask(
    conn: sqlite3.Connection,
    system: str,
    question: str,
    *,
    source: str = "cli",
    follow_up_of: str = "",
    session_id: str = "",
) -> tuple[Result, str]:
    """Answer one question and log it: the part both front ends share.

    Wraps answer() with a question_id, timing, a safety net for unexpected errors
    (network down, API error, ...) and the interaction record in json_log/. Never
    raises. Returns the Result and the question_id, so the dashboard can link a
    follow-up question to the answer it came from.

    source, follow_up_of and session_id are only stored in the log record; see
    log_interaction() for what they mean.
    """
    question_id = uuid.uuid4().hex[:8]
    log.info("[%s] question (%s): %s", question_id, source, question)
    started = time.perf_counter()
    try:
        result = answer(conn, system, question, question_id)
    except Exception:
        log.exception("[%s] Something went wrong answering that question", question_id)
        result = Result("error", "Something went wrong answering that. Please try again.")

    try:
        log_interaction(
            question_id=question_id,
            question=question,
            status=result.status,
            answer=result.message,
            sql=result.sql,
            row_count=len(result.rows),
            seconds=round(time.perf_counter() - started, 2),
            assumption=result.assumption,
            suggestions=result.suggestions,
            source=source,
            follow_up_of=follow_up_of,
            session_id=session_id,
            query_seconds=result.query_seconds,
            llm_seconds=result.llm_seconds,
            llm_calls=result.llm_calls,
            tokens={field: getattr(result, field) for field in TOKEN_FIELDS},
        )
    except Exception:  # a logging problem must never break the answer
        log.exception("[%s] Could not write the interaction record", question_id)
    return result, question_id


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
    if result.assumption:
        print(f"\nHow I read your question: {result.assumption}")
    print(f"\n{LABELS.get(result.status, 'Result')}: {result.message}\n")
    if result.suggestions:
        print("You could ask:")
        for question in result.suggestions:
            print(f"  - {question}")
        print()


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
            Config.MODEL,
            validated.get("model"),
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
        result, _ = ask(conn, system, question)
        print_result(result)

    conn.close()


if __name__ == "__main__":
    main()
