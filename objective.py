import os
import tempfile
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag, RegexpParser

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

    def extract_noun_phrases(self, sentence):
        """Extract noun phrases using chunking regex grammar."""
        try:
            words = word_tokenize(sentence)
            tagged = pos_tag(words)
        except LookupError:
            _ensure_nltk()
            words = word_tokenize(sentence)
            tagged = pos_tag(words)

        grammar = r"NP: {<DT>?<JJ>*<NN.*>+}"
        cp = RegexpParser(grammar)
        tree = cp.parse(tagged)
        
        noun_phrases = []
        for subtree in tree.subtrees():
            if subtree.label() == 'NP':
                phrase = " ".join([word for word, tag in subtree.leaves()])
                if len(phrase.strip()) > 2:
                    noun_phrases.append(phrase)
        return noun_phrases

    def generate_questions(self):
        try:
            sentences = sent_tokenize(self.text)
        except LookupError:
            _ensure_nltk()
            sentences = sent_tokenize(self.text)

        if not sentences:
            return []

        valid_sentences = [s.strip() for s in sentences if len(s.strip().split()) >= 5]
        if not valid_sentences:
            valid_sentences = sentences

        questions_data = []
        count = 0

        for sentence in valid_sentences:
            if count >= self.num_questions:
                break
            
            noun_phrases = self.extract_noun_phrases(sentence)
            if not noun_phrases:
                try:
                    words = word_tokenize(sentence)
                    tagged = pos_tag(words)
                except LookupError:
                    _ensure_nltk()
                    words = word_tokenize(sentence)
                    tagged = pos_tag(words)
                nouns = [w for w, t in tagged if t.startswith('NN') and len(w) > 2]
                if nouns:
                    target_key = nouns[0]
                else:
                    continue
            else:
                target_key = max(noun_phrases, key=len)

            question_text = sentence.replace(target_key, "________", 1)
            
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
