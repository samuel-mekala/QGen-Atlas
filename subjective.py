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

class SubjectiveTest:
    def __init__(self, text, num_questions=5):
        self.text = text
        try:
            self.num_questions = int(num_questions)
        except (ValueError, TypeError):
            self.num_questions = 5
        self.questions = []
        self.templates = [
            "What is {term}?",
            "Explain the concept of {term}.",
            "Describe the significance of {term}.",
            "What role does {term} play according to the text?",
            "Discuss {term} in detail."
        ]

    def extract_key_concepts(self, sentence):
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
        
        phrases = []
        for subtree in tree.subtrees():
            if subtree.label() == 'NP':
                phrase = " ".join([word for word, tag in subtree.leaves()])
                if len(phrase.strip().split()) >= 1 and len(phrase) > 3:
                    phrases.append(phrase)
        return phrases

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
        used_concepts = set()

        for idx, sentence in enumerate(valid_sentences):
            if count >= self.num_questions:
                break
            
            concepts = self.extract_key_concepts(sentence)
            if not concepts:
                continue

            target_concept = None
            for c in concepts:
                c_clean = c.lower().strip()
                if c_clean not in used_concepts and c_clean not in ['the text', 'this study', 'a system']:
                    target_concept = c
                    used_concepts.add(c_clean)
                    break
            
            if not target_concept:
                target_concept = concepts[0]

            template = self.templates[count % len(self.templates)]
            question_text = template.format(term=target_concept)
            
            count += 1
            item = {
                "id": count,
                "type": "subjective",
                "question": question_text,
                "target_concept": target_concept,
                "answer": sentence,
                "original_sentence": sentence
            }
            questions_data.append(item)

        self.questions = questions_data
        return questions_data
