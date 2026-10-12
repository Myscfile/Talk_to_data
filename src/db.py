import sqlite3
import os
import sys
import time
from pathlib import Path
from config import Config


# config variables
DB_PATH = Config.DB_PATH
QUERY_TIMEOUT_S = Config.QUERY_TIMEOUT_S
MAX_ROWS = Config.MAX_ROWS


# Connects to database
def connect_readonly(path: str) -> sqlite3.Connection:
    """Read-only at the SQLite level: writes fail even if the LLM tries one."""
    if not os.path.exists(path):
        sys.exit(f"Database not found: {path} (check DB_PATH in config.py)")
    uri = Path(path).resolve().as_uri() + "?mode=ro"
    return sqlite3.connect(uri, uri=True)


# Describes the data
def describe_schema(conn: sqlite3.Connection) -> str:
    out = []
    tables = conn.execute(
        "SELECT name, sql FROM sqlite_master "
        "WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    for name, ddl in tables:
        cur = conn.execute(f'SELECT * FROM "{name}" LIMIT 1')
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
        count = conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
        out.append(
            f"{ddl};\n-- {count:,} rows. Sample:\n-- {cols}\n"
            + "\n".join(f"-- {r}" for r in rows)
        )
    return "\n\n".join(out)


# Safegaurd to make sure the query is valid
def is_read_only(sql: str) -> bool:
    s = sql.strip().lstrip("(").lower()
    if not (s.startswith("select") or s.startswith("with")):
        return False
    banned = (
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
    return not any(b in s for b in banned) and ";" not in s.rstrip(";")


# Runs the query in the database
def run_query(conn: sqlite3.Connection, sql: str):
    deadline = time.time() + QUERY_TIMEOUT_S
    # Abort long-running queries (returns non-zero -> interrupt).
    conn.set_progress_handler(lambda: int(time.time() > deadline), 10_000)
    try:
        cur = conn.execute(sql)
        cols = [d[0] for d in cur.description] if cur.description else []
        rows = cur.fetchmany(MAX_ROWS + 1)
        truncated = len(rows) > MAX_ROWS
        return cols, rows[:MAX_ROWS], truncated
    finally:
        conn.set_progress_handler(None, 0)
