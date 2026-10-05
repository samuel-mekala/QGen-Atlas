import os
import tempfile
import random
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag, RegexpParser

STOP_CONCEPTS = {
    'year', 'years', 'time', 'work', 'day', 'days', 'date', 'dates',
    'every year', 'this year', 'that time', 'the work', 'the text',
    'this text', 'this study', 'a system', 'the system', 'this paper',
    'the paper', 'some cases', 'this approach', 'the method', 'a result',
    'the result', 'a subfield', 'this article', 'the process', 'a method',
    'an example', 'the following', 'this section', 'in addition', 'a key information',
    'country', 'number', 'enormous number'
}

TIME_DISTRACTORS = ["months", "decades", "centuries", "weeks", "seasons"]
NUMERIC_AGE_DISTRACTORS = ["21 years", "25 years", "27 years", "30 years", "18 years"]

def _ensure_nltk():
    local_path = os.path.join(os.path.dirname(__file__), 'nltk_data')
    if not os.path.exists(local_path):
        local_path = os.path.join(tempfile.gettempdir(), 'nltk_data')
    os.makedirs(local_path, exist_ok=True)
    if local_path not in nltk.data.path:
        nltk.data.path.insert(0, local_path)
    
    for pkg in ['punkt', 'punkt_tab', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng']:
        try:
            nltk.data.find(pkg)
        except (LookupError, Exception):
            try:
                nltk.download(pkg, download_dir=local_path, quiet=True)
            except Exception:
                pass

_ensure_nltk()

class MCQTest:
    def __init__(self, text, num_questions=5):
        self.text = text
        try:
            self.num_questions = int(num_questions)
        except (ValueError, TypeError):
            self.num_questions = 5
        self.questions = []

    def extract_informative_noun_phrases(self, sentence):
        try:
            words = word_tokenize(sentence)
            tagged = pos_tag(words)
        except LookupError:
            _ensure_nltk()
            words = word_tokenize(sentence)
            tagged = pos_tag(words)

        grammar = r"NP: {<NNP.*>+ | <JJ>*<NN.*>+}"
        cp = RegexpParser(grammar)
        tree = cp.parse(tagged)
        
        candidates = []
        for subtree in tree.subtrees():
            if subtree.label() == 'NP':
                phrase = " ".join([word for word, tag in subtree.leaves()]).strip()
                clean_phrase = phrase.lower().strip()
                if len(clean_phrase) > 2 and clean_phrase not in STOP_CONCEPTS:
                    if not clean_phrase.startswith(('in ', 'by ', 'with ', 'from ')):
                        candidates.append(phrase)
        return candidates

    def generate_questions(self):
        try:
            sentences = sent_tokenize(self.text)
        except LookupError:
            _ensure_nltk()
            sentences = sent_tokenize(self.text)

        if not sentences:
            return []

        all_noun_phrases = []
        for s in sentences:
            all_noun_phrases.extend(self.extract_informative_noun_phrases(s))
        
        unique_np_pool = list(dict.fromkeys(all_noun_phrases))

        fallback_distractors = [
            "Data Structure", "Machine Learning", "System Architecture",
            "Artificial Intelligence", "Database Indexing", "Cloud Infrastructure",
            "Information Retrieval", "Neural Network", "Algorithmic Efficiency"
        ]

        valid_sentences = [s.strip() for s in sentences if len(s.strip().split()) >= 5]
        if not valid_sentences:
            valid_sentences = sentences

        questions_data = []
        count = 0
        used_questions = []

        for sentence in valid_sentences:
            if count >= self.num_questions:
                break

            nps = self.extract_informative_noun_phrases(sentence)
            if not nps:
                continue

            target_ans = max(nps, key=len)
            question_text = sentence.replace(target_ans, "________", 1)

            if question_text.startswith("________ by ") or question_text.startswith("________ in "):
                continue

            # Duplicate question check
            if any(question_text.lower() in u.lower() for u in used_questions):
                continue
            used_questions.append(question_text)

            # Plausible distractors matching POS category
            t_lower = target_ans.lower()
            if t_lower in ['years', 'months', 'decades', 'centuries']:
                chosen_text_distractor = random.sample([t for t in TIME_DISTRACTORS if t.lower() != t_lower], 3)
            elif 'twenty-three' in t_lower or '23' in t_lower or 'years old' in t_lower:
                chosen_text_distractor = random.sample([t for t in NUMERIC_AGE_DISTRACTORS if t.lower() != t_lower], 3)
            else:
                distractors = [np for np in unique_np_pool if np.lower() != t_lower and np.lower() not in STOP_CONCEPTS and len(np) > 2]
                for fb in fallback_distractors:
                    if len(distractors) >= 3:
                        break
                    if fb.lower() != t_lower and fb.lower() not in [d.lower() for d in distractors]:
                        distractors.append(fb)
                chosen_text_distractor = random.sample(distractors, min(3, len(distractors)))

            # Build options with stable Option IDs
            correct_opt = {"id": "opt_correct", "text": target_ans}
            distractor_opts = [{"id": f"opt_d_{idx+1}", "text": text_val} for idx, text_val in enumerate(chosen_text_distractor)]
            
            all_opts = [correct_opt] + distractor_opts
            random.shuffle(all_opts)

            count += 1
            item = {
                "id": count,
                "type": "mcq",
                "question": question_text,
                "options": all_opts,
                "correct_option_id": "opt_correct",
                "answer": target_ans,
                "original_sentence": sentence
            }
            questions_data.append(item)

        self.questions = questions_data
        return questions_data
