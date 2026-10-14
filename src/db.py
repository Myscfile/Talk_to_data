"""Database access: read-only connection, schema description, and safe query execution."""

import re
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


def check_pitfalls(sql: str) -> str:
    """Return a fix-it message if the query hits a known data pitfall, else "".

    Unlike is_read_only(), this is not about safety: the query is valid but
    would give a misleading answer. The message goes back to the LLM so it can
    rewrite the query. Regex-based, so it is a heuristic and can miss cases.
    """
    # Lower-case, collapse whitespace and drop string literals so that text
    # inside quotes can't trigger a match.
    s = re.sub(r"'[^']*'", "''", sql.lower())
    s = re.sub(r"\s+", " ", s)

    # 1. customer_id is a new id for every order, so grouping or counting
    #    distinct values of it gives one order per "customer".
    group_clauses = re.findall(
        r"group by (.*?)(?= having | order by | limit |\)|$)", s
    )
    if any(re.search(r"\bcustomer_id\b", clause) for clause in group_clauses) or (
        re.search(r"count\s*\(\s*distinct\s+(?:\w+\.)?customer_id\s*\)", s)
    ):
        return (
            "customer_id is a new id for every order, so grouping or counting "
            "by it treats each order as a different customer. JOIN customers "
            "and use customers.customer_unique_id instead."
        )

    # 2. An order can have several reviews, so joining order_reviews straight
    #    onto orders duplicates orders. A GROUP BY on order_id anywhere in the
    #    query is taken as "reviews were aggregated per order first".
    joins_reviews = re.search(r"\bjoin order_reviews\b", s) or (
        re.search(r"\bfrom order_reviews\b", s) and re.search(r"\bjoin orders\b", s)
    )
    if joins_reviews and not any(
        re.search(r"\border_id\b", clause) for clause in group_clauses
    ):
        return (
            "An order can have more than one review, so joining order_reviews "
            "directly duplicates orders. Aggregate reviews per order first "
            "(e.g. a subquery with GROUP BY order_id) and join that."
        )

    return ""


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
