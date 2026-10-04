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
from validation import compute_raw_levenshtein, evaluate_objective_response, evaluate_subjective_response, evaluate_response
from app import app, CURRENT_QUIZ

class TestQGenAtlasE2E(unittest.TestCase):

    def setUp(self):
        self.sample_text = (
            "India achieved Independence Day on 15th August 1947 after long freedom struggle against British rule. "
            "Bhagat Singh was only twenty-three years old when he was hanged for his sacrifice in the freedom movement. "
            "Mahatma Gandhi led the non-violent freedom movement across the country. "
            "Natural Language Processing (NLP) is a subfield of artificial intelligence that focuses on human language. "
            "Response validation evaluates user answers using Levenshtein distance to calculate match percentage accurately."
        )
        self.app = app.test_client()
        self.app.testing = True

    def test_priority1_scoring_bug_fix(self):
        print("\n--- Testing Priority 1: Mathematical Levenshtein vs Final Score Separation ---")
        student = "a"
        expected = "It is celebrated every year on the 15th of August, and in 2026 it falls on a Saturday."

        sim_pct, dist = compute_raw_levenshtein(student, expected)
        self.assertEqual(dist, 83)
        self.assertLess(sim_pct, 5.0)  # ~2.41%
        print(f"Raw Levenshtein: Distance = {dist}, Character Similarity = {sim_pct}%")

        # Objective Phrase Match Test
        target_phrase = "own laws"
        student_phrase = "India did not make its own laws"
        eval_phrase = evaluate_objective_response(student_phrase, target_phrase)
        self.assertEqual(eval_phrase["similarity_percentage"], 100.0)
        self.assertIn("Phrase Match", eval_phrase["feedback"])
        print(f"Phrase Match Result: Levenshtein Similarity = {eval_phrase['levenshtein_similarity']}%, Final Evaluated Score = {eval_phrase['similarity_percentage']}% ({eval_phrase['feedback']})")

    def test_priority2_factual_error_detection(self):
        print("\n--- Testing Priority 2 & 6: Factual Mismatch Penalty ---")
        expected = "India achieved independence from British rule in 1947."
        student_wrong_year = "India achieved independence from British rule in 1950."
        student_correct = "India became free from British rule in 1947."

        eval_wrong = evaluate_subjective_response(student_wrong_year, expected)
        eval_correct = evaluate_subjective_response(student_correct, expected)

        self.assertTrue(eval_wrong["factual_mismatch"])
        self.assertLess(eval_wrong["similarity_percentage"], 40.0)
        self.assertIn("Factual error", eval_wrong["feedback"])

        self.assertFalse(eval_correct["factual_mismatch"])
        self.assertGreaterEqual(eval_correct["similarity_percentage"], 80.0)

        print(f"Correct Year (1947): {eval_correct['similarity_percentage']}% ({eval_correct['feedback']})")
        print(f"Wrong Year (1950): {eval_wrong['similarity_percentage']}% ({eval_wrong['feedback']})")

    def test_priority4_concept_entity_filtering(self):
        print("\n--- Testing Priority 4: Concept Filtering (Reject 'year', 'time', 'work') ---")
        sub_gen = SubjectiveTest(self.sample_text, num_questions=5)
        questions = sub_gen.generate_questions()

        for q in questions:
            concept = q["target_concept"].lower()
            self.assertNotIn(concept, ['year', 'years', 'time', 'work', 'day', 'date', 'the text'])
            print(f"Extracted Concept: '{q['target_concept']}' -> Q: {q['question']}")

    def test_priority5_qa_unit_alignment(self):
        print("\n--- Testing Priority 5: Question + Reference Answer Alignment ---")
        sub_gen = SubjectiveTest(self.sample_text, num_questions=5)
        questions = sub_gen.generate_questions()

        for q in questions:
            print(f"Q: {q['question']}")
            print(f"Ref Answer: {q['answer']}\n")
            if "old" in q["question"].lower():
                self.assertIn("twenty-three", q["answer"].lower())

    def test_priority6_mcq_plausible_distractors(self):
        print("\n--- Testing Priority 6: MCQ Plausible Distractor Generation ---")
        mcq_gen = MCQTest(self.sample_text, num_questions=3)
        questions = mcq_gen.generate_questions()

        for q in questions:
            self.assertEqual(len(q["options"]), 4)
            self.assertIn(q["answer"], q["options"])
            print(f"MCQ Q: {q['question']}")
            print(f"Options: {q['options']}")
            print(f"Target: {q['answer']}\n")

if __name__ == '__main__':
    unittest.main()
