"""Helpers for the suggested questions: parse them from model output and clean them.

Kept free of LLM and database code so it can be tested without an API key.
The model does not always follow the requested format exactly (bold headings,
no colon, special hyphens, other bullet characters), so parsing is forgiving.
"""

import re

MAX_SUGGESTIONS = 3
MAX_LENGTH = 160  # a suggestion is a short question, not a paragraph

# Hyphen-like characters: '-' plus the Unicode hyphens and dashes (U+2010 to U+2015),
# because the model likes to write "FOLLOW<non-breaking hyphen>UPS".
_DASHES = "\\-\u2010-\u2015"
# A heading line such as FOLLOW-UPS:, **Follow-ups**, ### Follow-up questions
_HEADING = re.compile(
    rf"\n[#*_\s]*follow[\s{_DASHES}]*ups?(?:\s+questions?)?[#*_\s]*:?[#*_\s]*\n",
    re.IGNORECASE,
)
# A list marker at the start of a line: '-', '*', a bullet dot, or '1.' / '1)'
_BULLET = re.compile(rf"^\s*(?:[*•{_DASHES}]|\d+[.)])\s+")


# Things this dataset cannot answer. The prompts say so too, but a model can ignore
# a prompt, so suggestions are checked here before they become buttons.
_NO_DATA_TOPICS = re.compile(
    r"\b(prices?|pricing|revenues?|payments?|paid|products?|sellers?|freight|"
    r"profits?|costs?|categor(?:y|ies)|gender|age)\b",
    re.IGNORECASE,
)
_FUTURE = re.compile(r"\b(next\s+(?:month|quarter|year)|forecast|predict\w*|future)\b", re.IGNORECASE)
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")


def in_scope(question: str, years: set[int] | None = None) -> bool:
    """False if the question names a year outside the data or a topic it lacks.

    years is the set of years the data covers; without it, years are not checked.
    """
    if years is not None and any(int(y) not in years for y in _YEAR.findall(question)):
        return False
    return not (_NO_DATA_TOPICS.search(question) or _FUTURE.search(question))


def clean_questions(value, years: set[int] | None = None) -> list[str]:
    """Keep up to three usable, in-scope questions from whatever the model sent back.

    Anything that is not a list of short strings is dropped, so a badly formed
    reply simply means no suggestions instead of an error. Questions about years
    outside the data, or about things the data does not have, are dropped too.
    """
    if not isinstance(value, list):
        return []
    questions = []
    for item in value:
        if not isinstance(item, str):
            continue
        question = _BULLET.sub("", item.strip()).strip()
        if (
            8 <= len(question) <= MAX_LENGTH
            and question not in questions
            and in_scope(question, years)
        ):
            questions.append(question)
    return questions[:MAX_SUGGESTIONS]


def _trailing_questions(text: str, years: set[int] | None) -> tuple[str, list[str]]:
    """Fallback when there is no heading: bullet lines that are questions, at the end."""
    lines = text.rstrip().splitlines()
    found = []
    while lines and (
        not lines[-1].strip()
        or (_BULLET.match(lines[-1]) and lines[-1].rstrip().endswith("?"))
    ):
        line = lines.pop()
        if line.strip():
            found.append(line.strip())
    if not found:
        return text.strip(), []
    return "\n".join(lines).strip(), clean_questions(found[::-1], years)


def split_followups(text: str, years: set[int] | None = None) -> tuple[str, list[str]]:
    """Separate the explanation from the follow-up questions at the end of it.

    Returns (explanation, questions). Looks for a 'FOLLOW-UPS' heading first,
    then for trailing bullet lines that are questions. If neither is there, the
    whole text is the explanation and there are no questions.
    """
    full = "\n" + text.strip() + "\n"
    match = _HEADING.search(full)
    if not match:
        return _trailing_questions(text, years)
    tail = full[match.end() :]
    lines = [ln.strip() for ln in tail.splitlines() if _BULLET.match(ln)]
    return full[: match.start()].strip(), clean_questions(lines, years)
