from pathlib import Path


class Config:
    BASE_DIR = Path(__file__).resolve().parent.parent
    DB_PATH = str(BASE_DIR / "data" / "sqlite" / "database.db")
    QUERY_TIMEOUT_S = 5
    MAX_ROWS = 200
    MODEL = "gpt-oss-120b"
    MAX_FIX_ATTEMPTS = 2
