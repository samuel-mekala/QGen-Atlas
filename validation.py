import Levenshtein
import re
from rapidfuzz import fuzz

STOPWORDS = {
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'in', 'on', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
    'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to',
    'from', 'up', 'down', 'in', 'out', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
    'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
    'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
    'than', 'too', 'very', 'can', 'will', 'just', 'should', 'now', 'of', 'and'
}

def normalize_text(text):
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    text = re.sub(r'\s+', ' ', text)     # normalize spaces
    return text

def extract_content_tokens(text):
    norm = normalize_text(text)
    tokens = norm.split()
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]

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

def compute_semantic_similarity(user_response, expected_answer):
    norm_user = normalize_text(user_response)
    norm_expected = normalize_text(expected_answer)
    
    if not norm_user and not norm_expected:
        return 100.0, 0
    if not norm_user or not norm_expected:
        return 0.0, max(len(norm_user), len(norm_expected))

    if is_acronym_match(user_response, expected_answer):
        return 100.0, 0

    lev_dist = Levenshtein.distance(norm_user, norm_expected)
    max_len = max(len(norm_user), len(norm_expected))
    lev_pct = (1 - (lev_dist / max_len)) * 100 if max_len > 0 else 100.0

    token_set_pct = fuzz.token_set_ratio(norm_user, norm_expected)
    token_sort_pct = fuzz.token_sort_ratio(norm_user, norm_expected)

    user_content = extract_content_tokens(user_response)
    expected_content = extract_content_tokens(expected_answer)
    
    if expected_content:
        matches = sum(1 for tok in expected_content if any(tok in u_tok or u_tok in tok for u_tok in user_content))
        keyword_coverage_pct = (matches / len(expected_content)) * 100.0
    else:
        keyword_coverage_pct = lev_pct

    hybrid_score = max(token_set_pct, token_sort_pct, keyword_coverage_pct, lev_pct)
    hybrid_score = round(max(0.0, min(100.0, float(hybrid_score))), 2)

    return hybrid_score, lev_dist

# Alias for backward compatibility
compute_levenshtein_similarity = compute_semantic_similarity

def evaluate_response(user_response, expected_answer, question_type="objective"):
    similarity_pct, distance = compute_semantic_similarity(user_response, expected_answer)
    
    if similarity_pct >= 80.0:
        feedback = "Excellent"
        status_color = "success"
    elif similarity_pct >= 60.0:
        feedback = "Good"
        status_color = "info"
    elif similarity_pct >= 35.0:
        feedback = "Partial Match"
        status_color = "warning"
    else:
        feedback = "Incorrect"
        status_color = "danger"
        
    return {
        "user_response": user_response,
        "expected_answer": expected_answer,
        "similarity_percentage": similarity_pct,
        "levenshtein_distance": distance,
        "feedback": feedback,
        "status_color": status_color
    }
