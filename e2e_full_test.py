import unittest
import json
import os
import sys

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath('.'))

from objective import ObjectiveTest
from subjective import SubjectiveTest
from mcq import MCQTest
from BERT_translate_custom import translate_questions, translate_text
from validation import (
    evaluate_mcq_response,
    evaluate_objective_response,
    evaluate_subjective_response,
    evaluate_response
)
from app import app, CURRENT_QUIZ

class TestQGenAtlasE2E(unittest.TestCase):

    def setUp(self):
        self.sample_text = (
            "India achieved Independence Day on 15th August 1947 after long freedom struggle against British rule. "
            "It was the work of an enormous number of ordinary people spread across the whole country. "
            "Bhagat Singh was only twenty-three years old when he was hanged for his sacrifice in the freedom movement. "
            "Mahatma Gandhi led the non-violent freedom movement and non-cooperation across the country. "
            "Natural Language Processing (NLP) is a subfield of artificial intelligence that focuses on human language."
        )
        self.app = app.test_client()
        self.app.testing = True

    def test_mcq_binary_scoring(self):
        print("\n--- Testing MCQ Binary Scoring Engine (100% or 0% Only) ---")
        correct_eval = evaluate_mcq_response("23 years", "23 years", user_option_id="opt_correct", correct_option_id="opt_correct")
        self.assertEqual(correct_eval["similarity_percentage"], 100.0)
        self.assertEqual(correct_eval["feedback"], "Correct")

        wrong_eval = evaluate_mcq_response("reflection", "non-cooperation", user_option_id="opt_d_1", correct_option_id="opt_correct")
        self.assertEqual(wrong_eval["similarity_percentage"], 0.0)
        self.assertEqual(wrong_eval["feedback"], "Incorrect")

        # Verify 7/10 overall test score calculation equals 70.0%
        scores = [100.0]*7 + [0.0]*3
        avg_score = round(sum(scores) / len(scores), 2)
        self.assertEqual(avg_score, 70.0)
        print(f"MCQ Correct Match: {correct_eval['similarity_percentage']}% ({correct_eval['feedback']})")
        print(f"MCQ Wrong Option: {wrong_eval['similarity_percentage']}% ({wrong_eval['feedback']})")
        print(f"7/10 Test Average Score: {avg_score}%")

    def test_subjective_contradiction_detection(self):
        print("\n--- Testing Subjective Contradiction Penalty (Violence vs Non-Violence) ---")
        expected = "Mahatma Gandhi led the non-violent freedom movement and non-cooperation across the country."
        student_contradict = "Mahatma Gandhi led the movement through violence and military resistance against the British."

        eval_res = evaluate_subjective_response(student_contradict, expected)
        self.assertEqual(eval_res["similarity_percentage"], 0.0)
        self.assertIn("Incorrect", eval_res["feedback"])
        self.assertIn("Contradiction", eval_res["feedback"])
        print(f"Contradiction Result: {eval_res['similarity_percentage']}% ({eval_res['feedback']})")

    def test_subjective_numeric_fact_mismatch(self):
        print("\n--- Testing Subjective Numeric Mismatch Penalty (25 vs 23) ---")
        expected = "Bhagat Singh was only twenty-three years old when he was hanged for his sacrifice."
        student_wrong_age = "Bhagat Singh was twenty-five years old when he was hanged."

        eval_res = evaluate_subjective_response(student_wrong_age, expected)
        self.assertEqual(eval_res["similarity_percentage"], 0.0)
        self.assertIn("Incorrect", eval_res["feedback"])
        print(f"Numeric Mismatch Result: {eval_res['similarity_percentage']}% ({eval_res['feedback']})")

    def test_subjective_paraphrase_success(self):
        print("\n--- Testing Subjective Paraphrase Success ---")
        expected = "It was the work of an enormous number of ordinary people spread across the whole country."
        student_paraphrase = "It means that a very large number of ordinary people from across India took part in the freedom struggle."

        eval_res = evaluate_subjective_response(student_paraphrase, expected)
        self.assertGreaterEqual(eval_res["similarity_percentage"], 60.0)
        self.assertIn(eval_res["feedback"], ["Excellent", "Good", "Partial Match"])
        print(f"Paraphrase Result: {eval_res['similarity_percentage']}% ({eval_res['feedback']})")

    def test_qa_unit_generation_alignment(self):
        print("\n--- Testing QA Unit Alignment & Quality Filtering ---")
        sub_gen = SubjectiveTest(self.sample_text, num_questions=5)
        questions = sub_gen.generate_questions()

        for q in questions:
            self.assertNotIn("enormous number", q["question"].lower())
            self.assertNotIn("what is year", q["question"].lower())
            print(f"Generated Q: {q['question']}")
            print(f"Reference Answer: {q['answer']}\n")

if __name__ == '__main__':
    unittest.main()
