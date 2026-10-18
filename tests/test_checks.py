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
from suggestions import clean_questions, split_followups  # noqa: E402


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


class Suggestions(unittest.TestCase):
    ANSWER = "There are 99,441 orders.\n\n**Context**\n- Base: all orders."

    def test_splits_a_plain_followups_list(self):
        text = self.ANSWER + "\n\nFOLLOW-UPS:\n- How many orders were canceled?\n- Which state has the most orders?"
        explanation, questions = split_followups(text)
        self.assertEqual(explanation, self.ANSWER)
        self.assertEqual(questions, ["How many orders were canceled?", "Which state has the most orders?"])

    def test_tolerates_bold_headings_other_bullets_and_case(self):
        text = self.ANSWER + "\n\n**Follow-ups:**\n* How many orders were canceled?\n* What is the average review score?"
        explanation, questions = split_followups(text)
        self.assertEqual(explanation, self.ANSWER)
        self.assertEqual(len(questions), 2)

    def test_no_list_means_no_questions(self):
        self.assertEqual(split_followups(self.ANSWER), (self.ANSWER, []))
        self.assertEqual(split_followups(""), ("", []))

    def test_context_bullets_are_not_taken_as_questions(self):
        # Bullets before the FOLLOW-UPS line belong to the explanation.
        explanation, questions = split_followups(self.ANSWER + "\n\nFOLLOW-UPS:\n- How many orders were canceled?")
        self.assertIn("Base: all orders.", explanation)
        self.assertEqual(questions, ["How many orders were canceled?"])

    def test_at_most_three_and_no_duplicates(self):
        raw = ["How many orders were canceled?"] * 2 + [f"Question number {i} about orders?" for i in range(5)]
        cleaned = clean_questions(raw)
        self.assertEqual(len(cleaned), 3)
        self.assertEqual(len(set(cleaned)), 3)

    QUESTIONS = "- How many customers placed orders in all three years?\n- Which two-year combinations have the most customers?"
    BODY = "Six hundred ninety customers ordered in exactly two years.\n\n**Context**\n- 690 of 96,096 customers (0.7 %)."

    def test_heading_variants_the_model_really_uses(self):
        headings = [
            "**FOLLOW-UPS**",  # bold, no colon (this one broke the dashboard)
            "FOLLOW‑UPS:",  # non-breaking hyphen
            "FOLLOW‑UPS",
            "### Follow-ups",
            "Follow-up questions:",
            "**Follow-ups:**",
            "Follow ups:",
        ]
        for heading in headings:
            explanation, questions = split_followups(f"{self.BODY}\n\n{heading}\n{self.QUESTIONS}")
            self.assertEqual(len(questions), 2, ascii(heading))
            self.assertEqual(explanation, self.BODY, ascii(heading))  # heading not left in the text

    def test_other_bullet_styles(self):
        for marker in ("•", "*", "1.", "1)", "–"):
            text = f"{self.BODY}\n\n**FOLLOW-UPS**\n" + self.QUESTIONS.replace("- ", marker + " ")
            self.assertEqual(len(split_followups(text)[1]), 2, marker)

    def test_questions_without_any_heading_are_found_at_the_end(self):
        explanation, questions = split_followups(f"{self.BODY}\n\n{self.QUESTIONS}")
        self.assertEqual(len(questions), 2)
        self.assertEqual(explanation, self.BODY)

    def test_trailing_context_bullets_are_not_mistaken_for_questions(self):
        text = f"{self.BODY}\n- Thin data: few rows.\n- Reference: overall is 4.09."
        self.assertEqual(split_followups(text), (text, []))

    def test_heading_without_bullets_gives_no_questions(self):
        explanation, questions = split_followups(f"{self.BODY}\n\n**FOLLOW-UPS**\n")
        self.assertEqual(questions, [])
        self.assertEqual(explanation, self.BODY)

    YEARS = {2016, 2017, 2018}

    def test_years_outside_the_data_are_dropped(self):
        raw = [
            "How many orders were placed in 2019?",
            "Did late deliveries rise in 2020 compared with 2018?",
            "How many orders were placed in 2017?",
            "How many orders were placed in 2016, 2017 and 2018 together?",
        ]
        self.assertEqual(
            clean_questions(raw, self.YEARS),
            ["How many orders were placed in 2017?", "How many orders were placed in 2016, 2017 and 2018 together?"],
        )

    def test_topics_and_future_the_data_lacks_are_dropped(self):
        for question in (
            "What is the revenue per order?",
            "Which product categories sell best?",
            "What is the average price of delivered orders?",
            "Which sellers have the slowest delivery?",
            "How many orders will there be next year?",
            "Can you forecast orders for the coming months?",
        ):
            self.assertEqual(clean_questions([question], self.YEARS), [], question)

    def test_normal_questions_are_kept(self):
        questions = [
            "What is the average delivery time in days?",
            "Who are the top 10 customers by number of orders?",
            "Which state has the most customers?",  # contains no banned word
            "What is the average review score by month?",  # 'average' is not 'age'
        ]
        self.assertEqual(len(clean_questions(questions[:3], self.YEARS)), 3)
        self.assertEqual(clean_questions(questions[3:], self.YEARS), questions[3:])

    def test_without_years_only_topics_are_checked(self):
        self.assertEqual(len(clean_questions(["How many orders were placed in 2019?"])), 1)
        self.assertEqual(clean_questions(["What is the revenue per order?"]), [])

    def test_split_followups_filters_too(self):
        text = (
            "There are 99,441 orders.\n\nFOLLOW-UPS:\n"
            "- What happened to orders in 2019?\n"
            "- How many orders were canceled?\n"
            "- What is the total revenue?"
        )
        explanation, questions = split_followups(text, self.YEARS)
        self.assertEqual(questions, ["How many orders were canceled?"])
        self.assertNotIn("FOLLOW", explanation)

    def test_bad_input_gives_no_suggestions(self):
        for value in (None, "a string", 5, {"a": 1}, [None, 3, "", "hi"], ["x" * 500]):
            self.assertEqual(clean_questions(value), [], value)

    def test_numbering_and_bullets_are_stripped(self):
        self.assertEqual(
            clean_questions(["1. How many orders were canceled?", "- Which state has the most orders?"]),
            ["How many orders were canceled?", "Which state has the most orders?"],
        )


if __name__ == "__main__":
    unittest.main()
