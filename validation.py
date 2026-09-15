import Levenshtein
import re

def normalize_text(text):
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    text = re.sub(r'\s+', ' ', text)     # normalize spaces
    return text

def compute_levenshtein_similarity(user_response, expected_answer):
    """
    Computes similarity percentage between user response and expected answer
    using the Levenshtein distance algorithm.
    """
    norm_user = normalize_text(user_response)
    norm_expected = normalize_text(expected_answer)
    
    if not norm_user and not norm_expected:
        return 100.0, 0
    if not norm_user or not norm_expected:
        return 0.0, max(len(norm_user), len(norm_expected))
    
    distance = Levenshtein.distance(norm_user, norm_expected)
    max_len = max(len(norm_user), len(norm_expected))
    
    similarity_pct = round((1 - (distance / max_len)) * 100, 2)
    similarity_pct = max(0.0, min(100.0, similarity_pct))
    
    return similarity_pct, distance

def evaluate_response(user_response, expected_answer, question_type="objective"):
    """
    Evaluates a user response and returns detailed feedback metrics.
    """
    similarity_pct, distance = compute_levenshtein_similarity(user_response, expected_answer)
    
    # Feedback classification based on similarity threshold
    if similarity_pct >= 85.0:
        feedback = "Excellent"
        status_color = "success"
    elif similarity_pct >= 65.0:
        feedback = "Good"
        status_color = "info"
    elif similarity_pct >= 40.0:
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
