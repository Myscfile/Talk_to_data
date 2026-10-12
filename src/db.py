"""Database access: read-only connection, schema description, and safe query execution."""

import sqlite3
import time
from pathlib import Path
from config import Config


def connect_readonly(path: str) -> sqlite3.Connection:
    """Open the database in read-only mode.

    Read-only is enforced by SQLite itself, so even if the LLM produces a
    write query that slips past is_read_only(), the write will fail.
    """
    db_file = Path(path)
    if not db_file.exists():
        raise FileNotFoundError(
            f"Database not found: {path} (check DB_PATH in config.py)"
        )
    uri = db_file.resolve().as_uri() + "?mode=ro"
    return sqlite3.connect(uri, uri=True)


def describe_schema(conn: sqlite3.Connection) -> str:
    """Build a text description of every table for the LLM prompt.

    Includes the CREATE statement, row count and one sample row per table.
    The sample row shows the model real value formats (e.g. how dates look),
    which the column types alone don't reveal.
    """
    parts = []
    tables = conn.execute(
        "SELECT name, sql FROM sqlite_master "
        "WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    for name, ddl in tables:
        cur = conn.execute(f'SELECT * FROM "{name}" LIMIT 1')
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
        count = conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
        parts.append(
            f"{ddl};\n-- {count:,} rows. Sample:\n-- {cols}\n"
            + "\n".join(f"-- {r}" for r in rows)
        )
    return "\n\n".join(parts)


def is_read_only(sql: str) -> bool:
    """Return True if the query looks like a single read-only SELECT/WITH.

    This is a first line of defence; the read-only connection is the real
    guarantee. Rejects multiple statements (';' in the middle) to block
    tricks like 'SELECT 1; DROP TABLE orders'.
    """
    s = sql.strip().lstrip("(").lower()
    if not s.startswith(("select", "with")):
        return False
    if any(keyword in s for keyword in Config.BANNED_KEYWORDS):
        return False
    return ";" not in s.rstrip(";")  # a trailing ';' is fine, one in the middle is not


def run_query(
    conn: sqlite3.Connection, sql: str
) -> tuple[list[str], list[tuple], bool]:
    """Run a query with a time limit and a row limit.

    Returns (column_names, rows, truncated). truncated is True when the
    query returned more than MAX_ROWS rows and the rest were cut off.
    """
    deadline = time.time() + Config.QUERY_TIMEOUT_S
    # SQLite calls this every 10,000 steps; returning 1 aborts the query.
    # Protects against runaway queries (e.g. an accidental cross join).
    conn.set_progress_handler(lambda: int(time.time() > deadline), 10_000)
    try:
        cur = conn.execute(sql)
        cols = [d[0] for d in cur.description] if cur.description else []
        # Fetch one extra row so we can tell whether results were truncated.
        rows = cur.fetchmany(Config.MAX_ROWS + 1)
        return cols, rows[: Config.MAX_ROWS], len(rows) > Config.MAX_ROWS
    finally:
        conn.set_progress_handler(None, 0)


if __name__ == "__main__":
    print(describe_schema(connect_readonly(Config.DB_PATH)))
