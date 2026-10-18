# Talk with your data: meet Capi

Ask questions in plain English about the Olist Brazilian e-commerce data. Capi, a
capybara, writes the SQL, checks it, runs it on a read-only database and explains
the answer. It also tells you how it read your question and suggests what to ask
next.

The goal is that someone who does not know the database can still see and
understand what is going on.

## Quick start

Requires Python 3.13 (other recent 3.x versions will likely work) and an API key
for [Cerebras](https://cloud.cerebras.ai), which hosts the model (`gpt-oss-120b`).

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; on macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Create a file named `.env` in the project folder:

```
CEREBRAS_API_KEY=your_key_here
```

Then, from the project folder:

| What | Command |
|---|---|
| Dashboard (chat in the browser) | `streamlit run src/dashboard.py` |
| Command-line version | `python src/app.py` |
| Unit tests (no API key needed) | `python -m unittest discover tests` |
| Evaluation (uses the API) | `python eval/run_eval.py` |

The dashboard opens at `http://127.0.0.1:8501`. It only listens on your own
machine (`.streamlit/config.toml`).

## The data

The SQLite database `data/sqlite/database.db` is included, so there is no
loading step. It holds three tables from the Olist CSVs in `data/csv`: `orders`,
`order_reviews` and `customers` (the assignment's `order_customers`). The data
runs from September 2016 to October 2018 and is thin at both ends.

Things the app knows about the data, because the model got them wrong without
them (see `NOTES` in `src/llm.py`):

- `customer_id` is new for every order. People are counted with
  `customer_unique_id` (99,441 ids, 96,096 people).
- An order can have more than one review, so reviews are aggregated per order
  before joining.
- There are no prices, revenue, products or sellers in these tables. Such
  questions are answered with "can't answer" and suggestions that do work.

## How a question is answered

```
question
  -> model call 1: write SQL, ask to clarify, or say it can't be answered
  -> safety check: only a single read-only SELECT/WITH        (refuse, no retry)
  -> pitfall check: known traps such as grouping by customer_id (send back, retry)
  -> run on a read-only connection, with a time limit and a row cap
     (on an error or timeout: send back, retry)
  -> model call 2: explain the rows, add context and follow-up questions
```

The model may retry a few times (`MAX_FIX_ATTEMPTS` in `src/config.py`). The real
protection against writes is the read-only database connection; the text check is
a first filter.

Each answer shows:

- the answer itself, with **context** (the base a number comes from, thin data,
  other ways to read the question),
- **how the question was read** (the model's assumption),
- the **SQL used** (collapsed),
- **follow-up questions** as buttons, filtered so they stay inside the data.

## Project layout

```
src/
  app.py          the pipeline (answer) and the command-line version
  dashboard.py    the Streamlit chat and data overview
  llm.py          prompts, model calls, token counting
  db.py           read-only connection, safety and pitfall checks, running queries
  suggestions.py  reading and filtering follow-up questions
  logger.py       logging
  config.py       all settings in one place
eval/
  questions.json      10 questions with known answers
  run_eval.py         runs them and compares the numbers
  validated_model.json  written when a model passes
tests/              unit tests for the checks, comparison and suggestions
assets/             avatars and the walking capybara
```

## Evaluation

`eval/questions.json` holds 10 questions: 7 with known numeric answers and 3 that
should be refused or clarified. The runner sends each through the real pipeline
and compares the **numbers** the SQL returns with the known answer (whole numbers
exactly, decimals within 0.5%), not the wording of the explanation.

```bash
python eval/run_eval.py                     # SQL step only, cheaper
python eval/run_eval.py --with-explanation  # full answers, realistic timings
python eval/run_eval.py --only q01,q05      # a few questions
```

It also reports time per question (split into model and query time) and token
usage. A full run that passes 90% is recorded in `eval/validated_model.json`
together with a hash of the prompt; the command-line version warns at start-up
when the model or prompt has changed since then.

## Logs

Each run writes two files, both linked by a `question_id`:

- `json_log/*.jsonl`: one record per question, for analysis. It includes the
  question, answer, SQL, where the question came from (typed, example button or
  follow-up), and the time and tokens used.
- `logs/*.log`: plain text, for debugging: each attempt, rejected queries and
  errors.

## Known limitations

- Each question is answered on its own; there is no memory of the conversation.
- The checks for pitfalls and for follow-up questions are rule-based, so they
  can miss cases.
- The context bullets and suggestions are written by the model and are not
  verified.
- The evaluation set is small (10 questions).
- Some dashboard behaviour (keeping the page at the top on load, the pinned
  overview) depends on Streamlit internals; `requirements.txt` pins the version.
