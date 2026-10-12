import sqlite3
from config import Config
from db import connect_readonly, describe_schema, is_read_only, run_query
from llm import ask_for_sql, build_system_prompt, explain


MAX_FIX_ATTEMPTS = Config.MAX_FIX_ATTEMPTS


def answer(conn, system: str, question: str) -> dict:
    messages = [{"role": "user", "content": question}]
    sql = ""

    for _ in range(MAX_FIX_ATTEMPTS + 1):
        raw, decision = ask_for_sql(system, messages)
        action = decision.get("action")
        msg = decision.get("message", "")
        sql = decision.get("sql", "")

        if action != "sql":
            return {"status": action, "message": msg}

        if not is_read_only(sql):
            return {
                "status": "cannot_answer",
                "message": "Refused: only read-only queries are allowed.",
                "sql": sql,
            }

        try:
            cols, rows, truncated = run_query(conn, sql)
        except sqlite3.Error as e:
            # Give the model its own answer + the error, let it fix the query.
            messages += [
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": f"That query failed: {e}. Fix it and reply with the JSON again.",
                },
            ]
            continue

        return {
            "status": "ok",
            "message": explain(question, sql, msg, cols, rows, truncated),
            "sql": sql,
            "columns": cols,
            "rows": rows,
        }

    return {
        "status": "error",
        "message": "Couldn't produce a working query. Try rephrasing.",
        "sql": sql,
    }


def print_result(r: dict):
    if r.get("sql"):
        print(f"\nSQL:\n{r['sql']}")
    if r.get("rows") is not None:
        print("\n" + " | ".join(r["columns"]))
        for row in r["rows"][:10]:
            print(" | ".join(str(v) for v in row))
        if len(r["rows"]) > 10:
            print(f"... ({len(r['rows'])} rows)")
    label = {
        "ok": "Answer",
        "clarify": "Need clarification",
        "cannot_answer": "Can't answer",
        "error": "Error",
    }.get(r["status"], "Result")
    print(f"\n{label}: {r['message']}\n")


def main():
    conn = connect_readonly(Config.DB_PATH)
    system = build_system_prompt(describe_schema(conn))
    print("Ask a question about the data (Ctrl+C to quit).")
    while True:
        try:
            q = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if q:
            print_result(answer(conn, system, q))


if __name__ == "__main__":
    main()
