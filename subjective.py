import os
import tempfile
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag, RegexpParser

GENERIC_BLACKLIST = {
    'this text', 'the text', 'this study', 'a system', 'the system',
    'this paper', 'the paper', 'some cases', 'this approach', 'the method',
    'a result', 'the result', 'a subfield', 'this article', 'the process',
    'a method', 'an example', 'the following', 'this section', 'in addition'
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

class SubjectiveTest:
    def __init__(self, text, num_questions=5):
        self.text = text
        try:
            self.num_questions = int(num_questions)
        except (ValueError, TypeError):
            self.num_questions = 5
        self.questions = []

    def extract_core_subject(self, sentence):
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
                clean_phrase = phrase.lower()
                if len(clean_phrase) > 2 and clean_phrase not in GENERIC_BLACKLIST:
                    candidates.append(phrase)
                    
        return candidates

    def select_semantically_aligned_question(self, concept, sentence):
        """
        Selects a question template that semantically matches the verb action
        in the target reference sentence.
        """
        s_lower = sentence.lower()

        # 1. Definition / Identity sentences (is, are, refers to, defines)
        if any(v in s_lower for v in [' is ', ' are ', ' refers to ', ' defines ', ' means ']):
            return f"What is {concept}?"

        # 2. Functional / Utility sentences (enables, provides, allows, facilitates)
        elif any(v in s_lower for v in ['enables', 'provides', 'allows', 'helps', 'facilitates', 'enhances']):
            return f"What is the role and purpose of {concept}?"

        # 3. Measurement / Calculation sentences (calculates, measures, computes, evaluates)
        elif any(v in s_lower for v in ['calculates', 'measures', 'computes', 'evaluates', 'scores']):
            return f"How does {concept} evaluate or measure data?"

        # 4. Process / Method sentences (involves, consists of, uses, applies)
        elif any(v in s_lower for v in ['involves', 'consists', 'uses', 'applies', 'utilizes']):
            return f"Explain how {concept} works according to the text."

        # Default fallback
        return f"Discuss {concept} in detail based on the text."

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

        for sentence in valid_sentences:
            if count >= self.num_questions:
                break
            
            concepts = self.extract_core_subject(sentence)
            if not concepts:
                continue

            target_concept = None
            for c in concepts:
                c_clean = c.lower().strip()
                if c_clean not in used_concepts and c_clean not in GENERIC_BLACKLIST:
                    target_concept = c
                    used_concepts.add(c_clean)
                    break
            
            if not target_concept:
                target_concept = concepts[0]

            # Generate semantically aligned question stem
            question_text = self.select_semantically_aligned_question(target_concept, sentence)
            
            count += 1
            item = {
                "id": count,
                "type": "subjective",
                "question": question_text,
                "target_concept": target_concept,
                "answer": sentence,  # Reference answer sentence matches question intent
                "original_sentence": sentence
            }
            questions_data.append(item)

        self.questions = questions_data
        return questions_data
