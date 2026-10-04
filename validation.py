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
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    text = re.sub(r'\s+', ' ', text)     # normalize whitespace
    return text

def extract_content_tokens(text):
    norm = normalize_text(text)
    tokens = norm.split()
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]

def extract_facts(text):
    """Extract numbers, dates, years, and digit sequences."""
    if not text:
        return set()
    numbers = set(re.findall(r'\b\d+\b', text))
    return numbers

def check_factual_consistency(user_text, expected_text):
    """
    Checks if student answer contains conflicting factual numbers/dates
    compared to the reference answer.
    """
    u_facts = extract_facts(user_text)
    e_facts = extract_facts(expected_text)

    if not e_facts:
        return False, None  # No numbers in reference answer to conflict with

    # If reference answer has facts (e.g. 1947), but student provided different numbers (e.g. 1950)
    conflicting_facts = [f for f in u_facts if f not in e_facts]
    if conflicting_facts and not any(f in u_facts for f in e_facts):
        return True, conflicting_facts

    return False, None

def compute_raw_levenshtein(user_response, expected_answer):
    """
    Calculates pure mathematical Levenshtein distance & similarity percentage.
    Formula: (1 - Distance / max(len(A), len(B))) * 100
    """
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

def is_acronym_match(user_text, expected_text):
    u_clean = normalize_text(user_text).upper()
    e_clean = normalize_text(expected_text).upper()

    e_words = extract_content_tokens(expected_text)
    if len(e_words) > 1:
        e_acronym = "".join([w[0].upper() for w in e_words])
        if u_clean == e_acronym:
            return True

    u_words = extract_content_tokens(user_text)
    if len(u_words) > 1:
        u_acronym = "".join([w[0].upper() for w in u_words])
        if e_clean == u_acronym:
            return True

    return False

def evaluate_objective_response(user_response, expected_answer):
    """
    Objective Evaluation Engine (Exact Match -> Phrase Inclusion -> Levenshtein Fallback).
    Maintains true character Levenshtein similarity while scoring phrase matches correctly.
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

    # 1. Exact Normalized Match
    if norm_user == norm_expected:
        return {
            "user_response": user_response,
            "expected_answer": expected_answer,
            "similarity_percentage": 100.0,
            "levenshtein_similarity": 100.0,
            "levenshtein_distance": lev_dist,
            "feedback": "Excellent (Exact Match)",
            "status_color": "success"
        }

    # 2. Acronym Match
    if is_acronym_match(user_response, expected_answer):
        return {
            "user_response": user_response,
            "expected_answer": expected_answer,
            "similarity_percentage": 100.0,
            "levenshtein_similarity": raw_lev_pct,
            "levenshtein_distance": lev_dist,
            "feedback": "Excellent (Acronym Match)",
            "status_color": "success"
        }

    # 3. Substring / Phrase Inclusion Match
    # E.g. Expected: "own laws", User: "India did not make its own laws"
    if norm_expected and norm_expected in norm_user:
        return {
            "user_response": user_response,
            "expected_answer": expected_answer,
            "similarity_percentage": 100.0,
            "levenshtein_similarity": raw_lev_pct,
            "levenshtein_distance": lev_dist,
            "feedback": "Excellent (Phrase Match)",
            "status_color": "success"
        }

    # 4. High-Fuzzy Match or Levenshtein score
    token_set = fuzz.token_set_ratio(norm_user, norm_expected)
    final_score = max(token_set, raw_lev_pct)

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
        "similarity_percentage": round(final_score, 2),
        "levenshtein_similarity": raw_lev_pct,
        "levenshtein_distance": lev_dist,
        "feedback": feedback,
        "status_color": status_color
    }

def evaluate_subjective_response(user_response, expected_answer):
    """
    Subjective Evaluation Engine combining:
    - 20% Lexical Character Similarity
    - 80% Semantic & Fuzzy Token Coverage
    - Factual Error Detection Penalty (e.g. 1950 vs 1947)
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

    # 1. Factual Error Check
    has_factual_error, bad_facts = check_factual_consistency(user_response, expected_answer)

    # 2. Semantic Token Coverage
    token_set_pct = fuzz.token_set_ratio(norm_user, norm_expected)
    token_sort_pct = fuzz.token_sort_ratio(norm_user, norm_expected)

    user_content = extract_content_tokens(user_response)
    expected_content = extract_content_tokens(expected_answer)

    if expected_content:
        matches = sum(1 for tok in expected_content if any(tok in u_tok or u_tok in tok for u_tok in user_content))
        keyword_coverage_pct = (matches / len(expected_content)) * 100.0
    else:
        keyword_coverage_pct = raw_lev_pct

    semantic_score = max(token_set_pct, token_sort_pct, keyword_coverage_pct)

    # 3. Weighted Final Score: 20% Lexical + 80% Semantic
    combined_score = (0.20 * raw_lev_pct) + (0.80 * semantic_score)

    # 4. Apply Factual Mismatch Penalty
    if has_factual_error:
        combined_score *= 0.30  # Heavy penalty for factual mismatch (e.g. 1950 vs 1947)

    final_score = round(max(0.0, min(100.0, combined_score)), 2)

    # Grading Threshold Brackets
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
        if has_factual_error:
            feedback = f"Incorrect (Factual error: {', '.join(bad_facts)})"
        else:
            feedback = "Incorrect"
        status_color = "danger"

    return {
        "user_response": user_response,
        "expected_answer": expected_answer,
        "similarity_percentage": final_score,
        "levenshtein_similarity": raw_lev_pct,
        "levenshtein_distance": lev_dist,
        "factual_mismatch": has_factual_error,
        "feedback": feedback,
        "status_color": status_color
    }

def evaluate_response(user_response, expected_answer, question_type="objective"):
    if question_type in ["objective", "mcq"]:
        return evaluate_objective_response(user_response, expected_answer)
    else:
        return evaluate_subjective_response(user_response, expected_answer)

# Alias for backward compatibility
compute_semantic_similarity = compute_raw_levenshtein
compute_levenshtein_similarity = compute_raw_levenshtein
