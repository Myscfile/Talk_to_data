"""Compare the numbers a query returned with the known answer.

Only numeric columns are compared; text columns (ids, names) and column names
are ignored. Whole numbers must match exactly, decimals within a tolerance.
"""


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def numeric_columns(rows: list) -> list[list]:
    """Return the columns whose values are all numbers, one list per column."""
    if not rows:
        return []
    columns = list(zip(*rows))
    return [list(col) for col in columns if all(_is_number(v) for v in col)]


def _close(expected, actual, tolerance: float) -> bool:
    if isinstance(expected, int):  # counts and other whole numbers: exact
        return actual == expected
    return abs(actual - expected) <= tolerance * max(abs(expected), abs(actual))


def _column_matches(expected: list, actual: list, ordered: bool, tolerance: float) -> bool:
    if len(expected) != len(actual):
        return False
    if not ordered:
        expected, actual = sorted(expected), sorted(actual)
    return all(_close(e, a, tolerance) for e, a in zip(expected, actual))


def results_match(
    expected_rows: list, actual_rows: list, ordered: bool, tolerance: float
) -> bool:
    """True if every numeric column of the known answer appears in the result.

    The result may have extra columns and a different column order. It must
    have the same number of rows. Row order only matters when ordered=True.
    """
    if len(expected_rows) != len(actual_rows):
        return False
    expected_cols = numeric_columns(expected_rows)
    actual_cols = numeric_columns(actual_rows)
    return bool(expected_cols) and all(
        any(_column_matches(e, a, ordered, tolerance) for a in actual_cols)
        for e in expected_cols
    )
