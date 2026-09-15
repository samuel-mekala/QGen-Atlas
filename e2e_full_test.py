import unittest
import json
import os
import sys

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath('.'))

from objective import ObjectiveTest
from subjective import SubjectiveTest
from BERT_translate_custom import translate_questions, translate_text
from validation import compute_levenshtein_similarity, evaluate_response
from app import app, CURRENT_QUIZ

class TestQGenAtlasE2E(unittest.TestCase):

    def setUp(self):
        self.sample_text = (
            "Natural Language Processing (NLP) is a subfield of artificial intelligence that focuses "
            "on the interaction between computers and human language. Query generation involves "
            "automatically extracting key information and creating relevant questions based on source text. "
            "Multilingual NLP frameworks enhance accessibility by providing translation across multiple target languages. "
            "Response validation evaluates user answers against expected reference answers using algorithms like "
            "Levenshtein distance to calculate match percentage accurately."
        )
        self.app = app.test_client()
        self.app.testing = True

    def test_objective_generation(self):
        print("\n--- Testing Objective Questions Generation ---")
        generator = ObjectiveTest(self.sample_text, num_questions=3)
        questions = generator.generate_questions()
        self.assertGreaterEqual(len(questions), 1)
        for q in questions:
            self.assertIn("________", q["question"])
            self.assertIsNotNone(q["answer"])
            self.assertTrue(len(q["answer"]) > 0)
            print(f"Objective Q{q['id']}: {q['question']}")
            print(f"Target Answer: {q['answer']}\n")

    def test_subjective_generation(self):
        print("\n--- Testing Subjective Questions Generation ---")
        generator = SubjectiveTest(self.sample_text, num_questions=3)
        questions = generator.generate_questions()
        self.assertGreaterEqual(len(questions), 1)
        for q in questions:
            self.assertTrue(any(pattern in q["question"] for pattern in ["What is", "Explain", "Describe", "Discuss", "role"]))
            self.assertIsNotNone(q["answer"])
            print(f"Subjective Q{q['id']}: {q['question']}")
            print(f"Target Context Answer: {q['answer']}\n")

    def test_levenshtein_validation_precision(self):
        print("\n--- Testing Levenshtein Validation Scoring ---")
        target = "Natural Language Processing"
        
        # Exact match -> 100%
        res_exact = evaluate_response("Natural Language Processing", target)
        self.assertEqual(res_exact["similarity_percentage"], 100.0)
        self.assertEqual(res_exact["feedback"], "Excellent")
        
        # Minor typo -> High match
        res_typo = evaluate_response("Natural Language Processin", target)
        self.assertGreaterEqual(res_typo["similarity_percentage"], 90.0)
        
        # Partial response
        res_partial = evaluate_response("Natural Language", target)
        self.assertTrue(50.0 <= res_partial["similarity_percentage"] <= 85.0)
        
        # Completely wrong
        res_wrong = evaluate_response("Quantum Computing Physics", target)
        self.assertLess(res_wrong["similarity_percentage"], 40.0)
        self.assertEqual(res_wrong["feedback"], "Incorrect")

        print(f"Exact Match: {res_exact['similarity_percentage']}% ({res_exact['feedback']})")
        print(f"Minor Typo Match: {res_typo['similarity_percentage']}% ({res_typo['feedback']})")
        print(f"Partial Match: {res_partial['similarity_percentage']}% ({res_partial['feedback']})")
        print(f"Wrong Answer: {res_wrong['similarity_percentage']}% ({res_wrong['feedback']})")

    def test_flask_end_to_end_flow(self):
        print("\n--- Testing Full Flask Web Application Flow ---")
        # 1. GET Homepage
        res_home = self.app.get('/')
        self.assertEqual(res_home.status_code, 200)
        self.assertIn(b"Multilingual Query Generation & Validation", res_home.data)

        # 2. POST /generate (Generate Objective Test in Spanish)
        post_data = {
            "input_text": self.sample_text,
            "num_questions": "3",
            "test_type": "objective",
            "target_lang": "es"
        }
        res_gen = self.app.post('/generate', data=post_data, follow_redirects=True)
        self.assertEqual(res_gen.status_code, 200)
        self.assertIn(b"Generated Assessment Session", res_gen.data)
        
        # Verify active quiz session in server state
        self.assertIn("questions", CURRENT_QUIZ)
        questions = CURRENT_QUIZ["questions"]
        self.assertEqual(len(questions), 3)

        # 3. POST /validate (Submit User Answers)
        val_data = {}
        for q in questions:
            val_data[f"user_answer_{q['id']}"] = q["answer"]  # Submit correct target answers
            
        res_val = self.app.post('/validate', data=val_data, follow_redirects=True)
        self.assertEqual(res_val.status_code, 200)
        self.assertIn(b"Response Validation Scorecard", res_val.data)
        self.assertIn(b"100", res_val.data)  # 100% score for exact match
        print("End-to-end web flow completed cleanly with 100% score validation!")

if __name__ == '__main__':
    unittest.main()
