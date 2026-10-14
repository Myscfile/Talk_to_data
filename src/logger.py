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
) -> None:
    """Append one question/answer record (one JSON line) to this run's file."""
    record = {
        "question_id": question_id,
        "run_id": RUN_ID,
        "ts": datetime.now().isoformat(timespec="seconds"),
        "question": question,
        "status": status,
        "answer": answer,
        "sql": sql,
        "row_count": row_count,
        "seconds": seconds,
    }
    with open(_json_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
