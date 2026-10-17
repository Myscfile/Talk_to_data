"""Fast tests for the code that needs no LLM: SQL checks and answer comparison.

Run from the project folder:  python -m unittest discover tests
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from compare import results_match  # noqa: E402
from db import check_pitfalls, is_read_only  # noqa: E402


class IsReadOnly(unittest.TestCase):
    def test_allows_select_and_with(self):
        self.assertTrue(is_read_only("SELECT COUNT(*) FROM orders"))
        self.assertTrue(is_read_only("WITH t AS (SELECT 1) SELECT * FROM t;"))

    def test_rejects_writes(self):
        for sql in ("DROP TABLE orders", "DELETE FROM orders", "UPDATE orders SET x=1"):
            self.assertFalse(is_read_only(sql), sql)

    def test_rejects_multiple_statements(self):
        self.assertFalse(is_read_only("SELECT 1; DROP TABLE orders"))


class CheckPitfalls(unittest.TestCase):
    def test_flags_customer_id_grouping(self):
        for sql in (
            "SELECT customer_id, COUNT(*) FROM orders GROUP BY customer_id",
            "select o.customer_id, count(*) from orders o group by o.customer_id",
            "SELECT COUNT(DISTINCT customer_id) FROM orders",
        ):
            self.assertNotEqual(check_pitfalls(sql), "", sql)

    def test_flags_unaggregated_review_join(self):
        sql = (
            "SELECT AVG(r.review_score) FROM orders o "
            "JOIN order_reviews r ON r.order_id = o.order_id"
        )
        self.assertNotEqual(check_pitfalls(sql), "")

    def test_allows_correct_queries(self):
        for sql in (
            "SELECT c.customer_unique_id, COUNT(*) FROM orders o JOIN customers c "
            "ON c.customer_id = o.customer_id GROUP BY c.customer_unique_id",
            "SELECT COUNT(DISTINCT customer_unique_id) FROM customers",
            "SELECT AVG(s) FROM orders o JOIN (SELECT order_id, AVG(review_score) s "
            "FROM order_reviews GROUP BY order_id) r ON r.order_id = o.order_id",
            "SELECT review_score, COUNT(*) FROM order_reviews GROUP BY review_score",
            "SELECT COUNT(*) FROM customers WHERE customer_city = 'group by customer_id'",
        ):
            self.assertEqual(check_pitfalls(sql), "", sql)


class ResultsMatch(unittest.TestCase):
    TOL = 0.005

    def test_whole_numbers_must_match_exactly(self):
        self.assertTrue(results_match([[96096]], [[96096]], False, self.TOL))
        self.assertTrue(results_match([[96096]], [[96096.0]], False, self.TOL))
        self.assertFalse(results_match([[96096]], [[96097]], False, self.TOL))

    def test_decimals_use_tolerance(self):
        self.assertTrue(results_match([[12.5587]], [[12.56]], False, self.TOL))
        self.assertFalse(results_match([[12.5587]], [[13.0]], False, self.TOL))

    def test_text_and_column_order_are_ignored(self):
        self.assertTrue(results_match([["SP", 40302]], [[40302, "sao paulo"]], False, self.TOL))

    def test_extra_numeric_columns_are_allowed(self):
        self.assertTrue(results_match([[40302]], [["SP", 40302, 0.42]], False, self.TOL))

    def test_row_order_matters_only_when_ordered(self):
        expected, actual = [[17], [9], [7]], [[7], [17], [9]]
        self.assertTrue(results_match(expected, actual, False, self.TOL))
        self.assertFalse(results_match(expected, actual, True, self.TOL))

    def test_row_count_must_match(self):
        self.assertFalse(results_match([[1], [2]], [[1]], False, self.TOL))
        self.assertFalse(results_match([[1]], [], False, self.TOL))


if __name__ == "__main__":
    unittest.main()
