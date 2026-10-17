"""Evaluate the model against the known answers in eval/questions.json.

For each question the real pipeline (app.answer) runs, and the numbers its SQL
returns are compared with the known answer. Behaviour questions (refusals)
only check the status. If enough questions pass, the model is recorded in
eval/validated_model.json.

Usage (from the project folder):
    python eval/run_eval.py                  full run; may update validated_model.json
    python eval/run_eval.py --only q01,q05   run some questions; never validates

This calls the LLM API, so it needs the API key and costs a few requests.
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import app  # noqa: E402
from compare import results_match  # noqa: E402
from config import Config  # noqa: E402
from db import connect_readonly, describe_schema, run_query  # noqa: E402
from llm import build_system_prompt, prompt_hash  # noqa: E402

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


def check_question(q: dict, conn, system: str) -> tuple[bool, str, str]:
    """Run one question. Returns (passed, reason, sql the model used)."""
    try:
        result = app.answer(conn, system, q["question"])
    except Exception as e:  # network/API problems count as a failure, not a crash
        return False, f"exception: {e}", ""

    if result.status not in q["expect"]:
        return False, f"status {result.status!r}, expected {q['expect']}", result.sql

    answers = q.get("answers")
    if not answers:  # behaviour question: the status was all we checked
        return True, f"status {result.status!r}", result.sql

    for answer in answers:
        if results_match(answer["rows"], result.rows, q["ordered"], Config.EVAL_TOLERANCE):
            return True, f"matches '{answer['label']}'", result.sql
    got = [list(r) for r in result.rows[:3]]
    want = [a["rows"][:3] for a in answers]
    return False, f"got {got}, expected one of {want}", result.sql


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--only", help="comma-separated question ids to run")
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

    # The explanation step is not evaluated; skip its API call to save requests.
    app.explain = lambda *args, **kwargs: ""
    system = build_system_prompt(describe_schema(conn))

    print(f"Evaluating {Config.MODEL} on {len(questions)} questions\n")
    outcomes = []
    for q in questions:
        passed, reason, sql = check_question(q, conn, system)
        print(f"{'PASS' if passed else 'FAIL'}  {q['id']}  {q['question']}")
        if not passed:
            print(f"      {reason}\n      SQL: {sql}")
        outcomes.append(
            {"id": q["id"], "question": q["question"], "passed": passed,
             "reason": reason, "sql": sql}
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
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    (RESULTS_DIR / f"{stamp}.json").write_text(
        json.dumps({**record, "results": outcomes}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\n{passed_count}/{total} passed ({pass_rate:.0%})")
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
