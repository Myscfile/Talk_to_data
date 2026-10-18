"""Evaluate the model against the known answers in eval/questions.json.

For each question the real pipeline (app.answer) runs, and the numbers its SQL
returns are compared with the known answer. Behaviour questions (refusals)
only check the status. If enough questions pass, the model is recorded in
eval/validated_model.json.

Each question is also timed (the total, the time spent waiting for the model and
the time spent running SQL) and its model tokens are counted. A summary of those
goes into the results file and validated_model.json.

Usage (from the project folder):
    python eval/run_eval.py                     full run; may update validated_model.json
    python eval/run_eval.py --only q01,q05      run some questions; never validates
    python eval/run_eval.py --with-explanation  also time the explanation step, so the
                                                timings match what a user waits for

This calls the LLM API, so it needs the API key and costs a few requests.
"""

import argparse
import hashlib
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import app  # noqa: E402
from compare import results_match  # noqa: E402
from config import Config  # noqa: E402
from db import connect_readonly, describe_schema, run_query  # noqa: E402
from llm import build_system_prompt, prompt_hash  # noqa: E402
from logger import TOKEN_FIELDS  # noqa: E402

QUESTIONS_PATH = Config.EVAL_DIR / "questions.json"
RESULTS_DIR = Config.EVAL_DIR / "results"


def find_stale_answers(questions: list[dict], conn) -> list[str]:
    """Return ids of questions whose stored answer no longer matches the database."""
    stale = []
    for q in questions:
        for answer in q.get("answers", []):
            _, rows, _ = run_query(conn, answer["sql"])
            if not results_match(answer["rows"], rows, True, 1e-9):
                stale.append(q["id"])
                break
    return stale


def check_question(
    q: dict, conn, system: str, with_explanation: bool
) -> tuple[bool, str, str, dict]:
    """Run one question. Returns (passed, reason, sql the model used, timing)."""
    started = time.perf_counter()
    try:
        result = app.answer(conn, system, q["question"])
    except Exception as e:  # network/API problems count as a failure, not a crash
        timing = {"seconds": round(time.perf_counter() - started, 2),
                  "llm_seconds": None, "llm_calls": None, "query_seconds": None,
                  **dict.fromkeys(TOKEN_FIELDS)}
        return False, f"exception: {e}", "", timing
    calls = result.llm_calls
    if not with_explanation and result.status == "ok":
        calls -= 1  # the explanation runs once per answered question; here it was skipped
    timing = {
        "seconds": round(time.perf_counter() - started, 2),
        "llm_seconds": result.llm_seconds,
        "llm_calls": calls,
        "query_seconds": result.query_seconds,
        **{field: getattr(result, field) for field in TOKEN_FIELDS},
    }

    if result.status not in q["expect"]:
        return False, f"status {result.status!r}, expected {q['expect']}", result.sql, timing

    answers = q.get("answers")
    if not answers:  # behaviour question: the status was all we checked
        return True, f"status {result.status!r}", result.sql, timing

    for answer in answers:
        if results_match(answer["rows"], result.rows, q["ordered"], Config.EVAL_TOLERANCE):
            return True, f"matches '{answer['label']}'", result.sql, timing
    got = [list(r) for r in result.rows[:3]]
    want = [a["rows"][:3] for a in answers]
    return False, f"got {got}, expected one of {want}", result.sql, timing


def token_summary(outcomes: list[dict]) -> dict:
    """Token totals over the run, plus the median per question that finished."""
    done = [o["timing"] for o in outcomes if o["timing"]["total_tokens"] is not None]
    if not done:
        return {"questions_counted": 0}
    return {
        "questions_counted": len(done),
        **{f"sum_{field}": sum(t[field] for t in done) for field in TOKEN_FIELDS},
        "median_total_tokens": round(statistics.median(t["total_tokens"] for t in done)),
    }


def timing_summary(outcomes: list[dict], with_explanation: bool) -> dict:
    """Median and slowest times over the questions that finished, plus model totals."""
    done = [o["timing"] for o in outcomes if o["timing"]["llm_seconds"] is not None]
    if not done:
        return {"explanation_included": with_explanation, "questions_timed": 0}
    seconds = [t["seconds"] for t in done]
    llm = [t["llm_seconds"] for t in done]
    query = [t["query_seconds"] for t in done]
    slowest = max(outcomes, key=lambda o: o["timing"]["seconds"])
    return {
        "explanation_included": with_explanation,
        "questions_timed": len(done),
        "median_seconds": round(statistics.median(seconds), 2),
        "slowest_seconds": round(max(seconds), 2),
        "slowest_question": slowest["id"],
        "median_llm_seconds": round(statistics.median(llm), 2),
        "median_query_seconds": round(statistics.median(query), 3),
        "llm_share": round(sum(llm) / sum(seconds), 2) if sum(seconds) else None,
        "total_llm_calls": sum(t["llm_calls"] for t in done),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--only", help="comma-separated question ids to run")
    parser.add_argument(
        "--with-explanation",
        action="store_true",
        help="also run the explanation step, so timings match what a user waits for "
        "(about twice the API calls)",
    )
    args = parser.parse_args()

    raw = QUESTIONS_PATH.read_bytes()
    questions = json.loads(raw)["questions"]
    if len(questions) > Config.EVAL_MAX_QUESTIONS:
        print(
            f"questions.json has {len(questions)} questions; "
            f"the limit is {Config.EVAL_MAX_QUESTIONS} (EVAL_MAX_QUESTIONS in config.py)."
        )
        return 2
    full_run = args.only is None
    if args.only:
        wanted = set(args.only.split(","))
        questions = [q for q in questions if q["id"] in wanted]

    conn = connect_readonly(Config.DB_PATH)
    stale = find_stale_answers(questions, conn)
    if stale:
        print(f"The database no longer matches the stored answers for: {', '.join(stale)}")
        print("Update eval/questions.json before evaluating.")
        return 2

    # The explanation is not what is evaluated, so by default its API call is
    # skipped to save requests; then the timings cover writing the SQL only.
    if not args.with_explanation:
        app.explain = lambda *args, **kwargs: ""
    system = build_system_prompt(describe_schema(conn))

    scope = "full answers" if args.with_explanation else "SQL step only, explanation skipped"
    print(f"Evaluating {Config.MODEL} on {len(questions)} questions (timing: {scope})\n")
    outcomes = []
    for q in questions:
        passed, reason, sql, timing = check_question(q, conn, system, args.with_explanation)
        if timing["llm_seconds"] is None:
            took = f"{timing['seconds']:.2f} s"
        else:
            took = (f"{timing['seconds']:.2f} s: model {timing['llm_seconds']:.2f} s in "
                    f"{timing['llm_calls']} call(s), query {timing['query_seconds']:.3f} s")
        print(f"{'PASS' if passed else 'FAIL'}  {q['id']}  {q['question']}  ({took})")
        if not passed:
            print(f"      {reason}\n      SQL: {sql}")
        outcomes.append(
            {"id": q["id"], "question": q["question"], "passed": passed,
             "reason": reason, "sql": sql, "timing": timing}
        )

    passed_count = sum(o["passed"] for o in outcomes)
    total = len(outcomes)
    pass_rate = round(passed_count / total, 3) if total else 0.0
    record = {
        "model": Config.MODEL,
        "validated_at": datetime.now().isoformat(timespec="seconds"),
        "passed": passed_count,
        "total": total,
        "pass_rate": pass_rate,
        "failed_ids": [o["id"] for o in outcomes if not o["passed"]],
        "prompt_hash": prompt_hash(),
        "eval_set_hash": hashlib.sha256(raw).hexdigest()[:12],
        "timing": timing_summary(outcomes, args.with_explanation),
        "tokens": token_summary(outcomes),
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    (RESULTS_DIR / f"{stamp}.json").write_text(
        json.dumps({**record, "results": outcomes}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\n{passed_count}/{total} passed ({pass_rate:.0%})")
    t = record["timing"]
    if t["questions_timed"]:
        print(
            f"Time per question ({scope}): median {t['median_seconds']} s, slowest "
            f"{t['slowest_seconds']} s ({t['slowest_question']}). Model: median "
            f"{t['median_llm_seconds']} s, {t['llm_share']:.0%} of the total time, "
            f"{t['total_llm_calls']} calls. Query: median {t['median_query_seconds']} s."
        )
    k = record["tokens"]
    if k["questions_counted"]:
        print(
            f"Tokens ({scope}): {k['sum_total_tokens']:,} in total, median "
            f"{k['median_total_tokens']:,} per question. Prompt {k['sum_prompt_tokens']:,} "
            f"(cached {k['sum_cached_tokens']:,}), completion {k['sum_completion_tokens']:,} "
            f"(reasoning {k['sum_reasoning_tokens']:,})."
        )
    meets_bar = pass_rate >= Config.EVAL_PASS_RATE
    if full_run and meets_bar:
        Config.VALIDATED_PATH.write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"Model validated -> {Config.VALIDATED_PATH.name}")
    elif not full_run:
        print("Partial run: validated_model.json not updated.")
    else:
        print(f"Below the {Config.EVAL_PASS_RATE:.0%} bar: validated_model.json not updated.")
    return 0 if meets_bar else 1


if __name__ == "__main__":
    sys.exit(main())
