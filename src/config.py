"""Central settings for the app.

All tunable values live here so they can be changed in one place.
Secrets (API keys) are NOT stored here; they come from the .env file.
"""

from pathlib import Path


class Config:
    """App-wide settings, used as Config.NAME (no instance needed)."""

    # Paths are built from this file's location, so they work no matter
    # which folder the app is started from. parent.parent = project root.
    BASE_DIR = Path(__file__).resolve().parent.parent
    DB_PATH = str(BASE_DIR / "data" / "sqlite" / "database.db")
    LOG_DIR = BASE_DIR / "logs"  # plain-text diagnostics, for debugging
    JSON_LOG_DIR = BASE_DIR / "json_log"  # one JSON record per question

    # Query safety limits
    QUERY_TIMEOUT_S = 5  # abort queries running longer than this (seconds)
    MAX_ROWS = 200  # max rows returned to the user and the explainer

    # LLM settings
    MODEL = "gpt-oss-120b"  # Cerebras model name
    MAX_FIX_ATTEMPTS = 2  # times the model may retry after a failed/rejected query

    # Evaluation (eval/run_eval.py)
    EVAL_DIR = BASE_DIR / "eval"
    VALIDATED_PATH = EVAL_DIR / "validated_model.json"  # written when a run passes
    EVAL_MAX_QUESTIONS = 10  # keeps a run quick and cheap
    EVAL_PASS_RATE = 0.9  # share of questions that must pass to validate a model
    EVAL_TOLERANCE = 0.005  # decimals may differ by 0.5%; whole numbers must match

    # Words that make a query non-read-only. A first check only: the
    # read-only database connection is the real protection.
    BANNED_KEYWORDS = (
        "insert ",
        "update ",
        "delete ",
        "drop ",
        "alter ",
        "attach ",
        "pragma ",
        "create ",
        "replace ",
        "vacuum",
    )
