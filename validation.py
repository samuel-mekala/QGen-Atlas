import Levenshtein
import re
from rapidfuzz import fuzz

STOPWORDS = {
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'in', 'on', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
    'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to',
    'from', 'up', 'down', 'out', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
    'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
    'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
    'than', 'too', 'very', 'can', 'will', 'just', 'should', 'now', 'of', 'and'
}

CONTRACTIONS = {
    "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "can't": "cannot", "won't": "will not", "isn't": "is not",
    "aren't": "are not", "wasn't": "was not", "weren't": "were not",
    "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
    "it's": "it is", "they're": "they are", "we're": "we are", "you're": "you are"
}

NUMBER_WORDS = {
    "twenty-three": "23", "twenty three": "23",
    "twenty-five": "25", "twenty five": "25",
    "twenty-one": "21", "twenty one": "21",
    "twenty-seven": "27", "twenty seven": "27",
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14", "fifteen": "15",
    "sixteen": "16", "seventeen": "17", "eighteen": "18", "nineteen": "19", "twenty": "20"
}

ANTONYM_PAIRS = [
    ("violence", "nonviolence"),
    ("violence", "nonviolent"),
    ("violent", "nonviolent"),
    ("military", "nonviolent"),
    ("peaceful", "violent"),
    ("war", "peace"),
    ("failed", "succeeded"),
    ("false", "true"),
    ("incorrect", "correct")
]

def expand_contractions(text):
    if not text:
        return ""
    words = text.split()
    expanded = [CONTRACTIONS.get(w.lower(), w) for w in words]
    return " ".join(expanded)

def normalize_text(text):
    if not text:
        return ""
    text = expand_contractions(text)
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)  # replace punctuation with space
    text = re.sub(r'\s+', ' ', text).strip() # normalize whitespace
    return text

def normalize_numeric_text(text):
    if not text:
        return ""
    text_lower = expand_contractions(text).lower().strip()
    for word, num in NUMBER_WORDS.items():
        text_lower = re.sub(r'\b' + re.escape(word) + r'\b', num, text_lower)
    text_lower = re.sub(r'[^\w\s]', ' ', text_lower)
    text_lower = re.sub(r'\s+', ' ', text_lower).strip()
    return text_lower

def extract_content_tokens(text):
    norm = normalize_text(text)
    tokens = norm.split()
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]

def extract_numbers_and_facts(text):
    if not text:
        return set()
    norm = normalize_numeric_text(text)
    return set(re.findall(r'\b\d+\b', norm))

def check_factual_contradiction(user_text, expected_text):
    u_tokens = set(normalize_text(user_text).split())
    e_tokens = set(normalize_text(expected_text).split())

    # 1. Antonym / Contradiction Check
    if 'violence' in u_tokens and ('nonviolent' in e_tokens or 'nonviolence' in e_tokens or ('non' in e_tokens and 'violent' in e_tokens)):
        return True, "Contradiction detected: 'violence' vs 'non-violence'"

    if ('nonviolent' in u_tokens or 'nonviolence' in u_tokens or ('non' in u_tokens and 'violent' in u_tokens)) and 'violence' in e_tokens:
        if 'nonviolent' not in e_tokens and 'nonviolence' not in e_tokens:
            return True, "Contradiction detected: 'non-violence' vs 'violence'"

    for word1, word2 in ANTONYM_PAIRS:
        if word1 in u_tokens and word2 in e_tokens:
            return True, f"Contradiction detected: '{word1}' vs '{word2}'"
        if word2 in u_tokens and word1 in e_tokens:
            return True, f"Contradiction detected: '{word2}' vs '{word1}'"

    # 2. Numeric & Date Conflicts
    u_facts = extract_numbers_and_facts(user_text)
    e_facts = extract_numbers_and_facts(expected_text)

    if e_facts:
        conflicting = [f for f in u_facts if f not in e_facts]
        if conflicting and not any(f in u_facts for f in e_facts):
            return True, f"Factual mismatch: {', '.join(conflicting)} vs {', '.join(e_facts)}"

    return False, None

def compute_raw_levenshtein(user_response, expected_answer):
    norm_user = normalize_text(user_response)
    norm_expected = normalize_text(expected_answer)

    if not norm_user and not norm_expected:
        return 100.0, 0
    if not norm_user or not norm_expected:
        return 0.0, max(len(norm_user), len(norm_expected))

    distance = Levenshtein.distance(norm_user, norm_expected)
    max_len = max(len(norm_user), len(norm_expected))
    sim_pct = round((1 - (distance / max_len)) * 100, 2)
    sim_pct = max(0.0, min(100.0, sim_pct))

    return sim_pct, distance

def evaluate_mcq_response(user_response, expected_answer, user_option_id=None, correct_option_id=None):
    """
    MCQ Independent Engine: Strictly Binary (100% Correct or 0% Incorrect).
    """
    if user_option_id is not None and correct_option_id is not None:
        is_correct = (str(user_option_id).strip() == str(correct_option_id).strip())
    else:
        norm_user = normalize_numeric_text(user_response)
        norm_expected = normalize_numeric_text(expected_answer)
        is_correct = (norm_user == norm_expected)

    if is_correct:
        return {
            "user_response": user_response,
            "expected_answer": expected_answer,
            "similarity_percentage": 100.0,
            "levenshtein_similarity": 100.0,
            "levenshtein_distance": 0,
            "feedback": "Correct",
            "status_color": "success"
        }
    else:
        raw_lev_pct, lev_dist = compute_raw_levenshtein(user_response, expected_answer)
        return {
            "user_response": user_response,
            "expected_answer": expected_answer,
            "similarity_percentage": 0.0,
            "levenshtein_similarity": raw_lev_pct,
            "levenshtein_distance": lev_dist,
            "feedback": "Incorrect",
            "status_color": "danger"
        }

def evaluate_objective_response(user_response, expected_answer):
    """
    Objective Fill-in-the-Blank Engine:
    Exact Match -> Numeric Equivalence -> Substring/Phrase Match -> Levenshtein Near Match -> Incorrect.
    """
    norm_user = normalize_numeric_text(user_response)
    norm_expected = normalize_numeric_text(expected_answer)
    raw_lev_pct, lev_dist = compute_raw_levenshtein(user_response, expected_answer)

    if not norm_user:
        return {
            "user_response": user_response,
            "expected_answer": expected_answer,
            "similarity_percentage": 0.0,
            "levenshtein_similarity": 0.0,
            "levenshtein_distance": lev_dist,
            "feedback": "Incorrect",
            "status_color": "danger"
        }

    # 1. Exact Match
    if norm_user == norm_expected:
        return {
            "user_response": user_response,
            "expected_answer": expected_answer,
            "similarity_percentage": 100.0,
            "levenshtein_similarity": 100.0,
            "levenshtein_distance": lev_dist,
            "feedback": "Exact Match",
            "status_color": "success"
        }

    # 2. Phrase Substring Inclusion
    if norm_expected and norm_expected in norm_user:
        return {
            "user_response": user_response,
            "expected_answer": expected_answer,
            "similarity_percentage": 100.0,
            "levenshtein_similarity": raw_lev_pct,
            "levenshtein_distance": lev_dist,
            "feedback": "Equivalent Answer (Phrase Match)",
            "status_color": "success"
        }

    # 3. Typo / Near Match
    if raw_lev_pct >= 85.0:
        return {
            "user_response": user_response,
            "expected_answer": expected_answer,
            "similarity_percentage": raw_lev_pct,
            "levenshtein_similarity": raw_lev_pct,
            "levenshtein_distance": lev_dist,
            "feedback": "Near Match",
            "status_color": "info"
        }

    return {
        "user_response": user_response,
        "expected_answer": expected_answer,
        "similarity_percentage": 0.0,
        "levenshtein_similarity": raw_lev_pct,
        "levenshtein_distance": lev_dist,
        "feedback": "Incorrect",
        "status_color": "danger"
    }

def evaluate_subjective_response(user_response, expected_answer):
    """
    Subjective Engine:
    Semantic Token Fuzzy Ratios + Concept Coverage + Factual Consistency Check
    Antonym / Contradiction penalty -> 0% Incorrect.
    """
    norm_user = normalize_text(user_response)
    norm_expected = normalize_text(expected_answer)
    raw_lev_pct, lev_dist = compute_raw_levenshtein(user_response, expected_answer)

    if not norm_user:
        return {
            "user_response": user_response,
            "expected_answer": expected_answer,
            "similarity_percentage": 0.0,
            "levenshtein_similarity": 0.0,
            "levenshtein_distance": lev_dist,
            "feedback": "Incorrect",
            "status_color": "danger"
        }

    # 1. Contradiction & Fact Conflict Check
    is_contradiction, error_reason = check_factual_contradiction(user_response, expected_answer)
    if is_contradiction:
        return {
            "user_response": user_response,
            "expected_answer": expected_answer,
            "similarity_percentage": 0.0,
            "levenshtein_similarity": raw_lev_pct,
            "levenshtein_distance": lev_dist,
            "factual_mismatch": True,
            "feedback": f"Incorrect ({error_reason})",
            "status_color": "danger"
        }

    # 2. Semantic Token Fuzzy Ratios
    semantic_score = fuzz.token_set_ratio(norm_user, norm_expected)
    token_sort_score = fuzz.token_sort_ratio(norm_user, norm_expected)
    max_semantic = max(semantic_score, token_sort_score)

    # 3. Concept Keyword Coverage
    user_content = extract_content_tokens(user_response)
    expected_content = extract_content_tokens(expected_answer)
    if expected_content:
        matches = sum(1 for tok in expected_content if any(tok in u_tok or u_tok in tok for u_tok in user_content))
        concept_coverage_score = (matches / len(expected_content)) * 100.0
    else:
        concept_coverage_score = max_semantic

    final_score = max(max_semantic, concept_coverage_score)
    final_score = round(max(0.0, min(100.0, float(final_score))), 2)

    if final_score >= 90.0:
        feedback = "Excellent"
        status_color = "success"
    elif final_score >= 70.0:
        feedback = "Good"
        status_color = "info"
    elif final_score >= 40.0:
        feedback = "Partial Match"
        status_color = "warning"
    else:
        feedback = "Incorrect"
        status_color = "danger"

    return {
        "user_response": user_response,
        "expected_answer": expected_answer,
        "similarity_percentage": final_score,
        "levenshtein_similarity": raw_lev_pct,
        "levenshtein_distance": lev_dist,
        "feedback": feedback,
        "status_color": status_color
    }

def evaluate_response(user_response, expected_answer, question_type="objective", user_option_id=None, correct_option_id=None):
    if question_type == "mcq":
        return evaluate_mcq_response(user_response, expected_answer, user_option_id, correct_option_id)
    elif question_type == "objective":
        return evaluate_objective_response(user_response, expected_answer)
    else:
        return evaluate_subjective_response(user_response, expected_answer)

compute_semantic_similarity = compute_raw_levenshtein
compute_levenshtein_similarity = compute_raw_levenshtein
