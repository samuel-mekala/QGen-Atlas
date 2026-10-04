import os
import tempfile
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
    'an example', 'the following', 'this section', 'in addition', 'a key information'
}

STOP_WORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
    'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'it', 'its', 'they',
    'them', 'their', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was',
    'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
    'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as',
    'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between'
}

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

class ObjectiveTest:
    def __init__(self, text, num_questions=5):
        self.text = text
        try:
            self.num_questions = int(num_questions)
        except (ValueError, TypeError):
            self.num_questions = 5
        self.questions = []
        self.answers = []

    def extract_informative_noun_phrases(self, sentence):
        try:
            words = word_tokenize(sentence)
            tagged = pos_tag(words)
        except LookupError:
            _ensure_nltk()
            words = word_tokenize(sentence)
            tagged = pos_tag(words)

        # Prioritize Proper Nouns (NNP) first, then multi-word Noun Phrases
        grammar = r"NP: {<NNP.*>+ | <JJ>*<NN.*>+}"
        cp = RegexpParser(grammar)
        tree = cp.parse(tagged)
        
        candidates = []
        for subtree in tree.subtrees():
            if subtree.label() == 'NP':
                phrase = " ".join([word for word, tag in subtree.leaves()]).strip()
                clean_phrase = phrase.lower().strip()

                if len(clean_phrase) <= 2 or clean_phrase in STOP_CONCEPTS:
                    continue
                if clean_phrase in STOP_WORDS or clean_phrase.startswith(('in ', 'by ', 'with ', 'from ')):
                    continue
                if not any(c.isalpha() for c in clean_phrase):
                    continue

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

        valid_sentences = [s.strip() for s in sentences if len(s.strip().split()) >= 6]
        if not valid_sentences:
            valid_sentences = sentences

        questions_data = []
        count = 0
        used_answers = set()

        for sentence in valid_sentences:
            if count >= self.num_questions:
                break

            nps = self.extract_informative_noun_phrases(sentence)
            if not nps:
                continue

            target_key = None
            for np in sorted(nps, key=len, reverse=True):
                if np.lower() not in used_answers and np.lower() not in STOP_CONCEPTS:
                    target_key = np
                    used_answers.add(np.lower())
                    break
            
            if not target_key:
                target_key = nps[0]

            question_text = sentence.replace(target_key, "________", 1)

            if question_text.startswith("________ by ") or question_text.startswith("________ in "):
                continue

            count += 1
            item = {
                "id": count,
                "type": "objective",
                "question": question_text,
                "answer": target_key,
                "original_sentence": sentence
            }
            questions_data.append(item)

        self.questions = questions_data
        return questions_data
