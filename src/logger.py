"""Logging in two parts, each with its own file per program run.

- Interaction records (json_log/*.jsonl): one JSON object per question, with
  the same fields every time. Meant to be loaded for analysis, a DB insert or
  an API.
- Diagnostic log (logs/*.log): plain text with retries, rejected queries and
  errors. Meant for debugging a session. Lines carry the question_id so they
  can be matched to a record.

File names are <date>_<time>_<run_id>, so each run gets its own pair.
"""

import json
import logging
import uuid
from datetime import datetime

from config import Config

RUN_ID = uuid.uuid4().hex[:8]

# Token counts stored with every question (llm.py counts them per model call).
TOKEN_FIELDS = (
    "prompt_tokens", "completion_tokens", "reasoning_tokens", "cached_tokens", "total_tokens"
)

# Set by setup_logging(); the file itself is created on the first record.
_json_file = None


def setup_logging() -> None:
    """Send diagnostics to a text file (and warnings to the terminal)."""
    global _json_file
    Config.LOG_DIR.mkdir(exist_ok=True)
    Config.JSON_LOG_DIR.mkdir(exist_ok=True)
    stamp = f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{RUN_ID}"
    _json_file = Config.JSON_LOG_DIR / f"{stamp}.jsonl"

    file_handler = logging.FileHandler(Config.LOG_DIR / f"{stamp}.log", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )

    # The terminal already shows answers via print(); only surface problems here.
    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    console.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    logging.basicConfig(level=logging.INFO, handlers=[console, file_handler])
    # The API client logs every HTTP request at INFO; that is noise in our log.
    logging.getLogger("httpx").setLevel(logging.WARNING)


def log_interaction(
    question_id: str,
    question: str,
    status: str,
    answer: str,
    sql: str,
    row_count: int,
    seconds: float,
    assumption: str = "",
    suggestions: list[str] | None = None,
    source: str = "cli",
    follow_up_of: str = "",
    session_id: str = "",
    query_seconds: float = 0.0,
    llm_seconds: float = 0.0,
    llm_calls: int = 0,
    tokens: dict | None = None,
) -> None:
    """Append one question/answer record (one JSON line) to this run's file.

    source is where the question came from: "cli" (typed in the terminal),
    "typed" (typed in the dashboard), "example" (a welcome-message button) or
    "follow_up" (a suggestion under an answer; follow_up_of is that answer's
    question_id). session_id tells dashboard visitors apart within one run.
    seconds is the whole answer. Of that, query_seconds is the time spent running
    SQL and llm_seconds the time spent waiting for the model, both summed over
    attempts; llm_calls counts the model calls. tokens holds the TOKEN_FIELDS
    counts over all model calls (reasoning is part of completion, cached part
    of prompt).
    """
    record = {
        "question_id": question_id,
        "run_id": RUN_ID,
        "session_id": session_id,
        "ts": datetime.now().isoformat(timespec="seconds"),
        "source": source,
        "follow_up_of": follow_up_of,
        "question": question,
        "status": status,
        "answer": answer,
        "assumption": assumption,
        "sql": sql,
        "row_count": row_count,
        "suggestions": suggestions or [],
        "seconds": seconds,
        "query_seconds": query_seconds,
        "llm_seconds": llm_seconds,
        "llm_calls": llm_calls,
        **{field: (tokens or {}).get(field, 0) for field in TOKEN_FIELDS},
    }
    with open(_json_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
