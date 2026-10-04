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

    def extract_core_entities(self, sentence):
        """
        Extracts key entities, proper nouns (NNP), and informative concepts,
        filtering out generic terms like 'year', 'time', 'work'.
        """
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

                if len(clean_phrase) <= 2 or clean_phrase in STOP_CONCEPTS:
                    continue
                if clean_phrase.startswith(('in ', 'by ', 'with ', 'from ', 'the ', 'this ')):
                    clean_phrase_core = re.sub(r'^(the|this|a|an|in|by|with|from)\s+', '', clean_phrase)
                    if clean_phrase_core in STOP_CONCEPTS or len(clean_phrase_core) <= 2:
                        continue
                candidates.append(phrase)
                    
        return candidates

    def generate_aligned_qa_unit(self, concept, sentence):
        """
        Generates a question and reference answer as a SINGLE unified unit,
        ensuring question intent matches reference answer facts.
        """
        s_lower = sentence.lower()

        # 1. Age / Numeric Duration (e.g. "twenty-three years old", "hanged at age 23")
        if any(w in s_lower for w in ['years old', 'age of', 'hanged', 'born in', 'died in']):
            return f"How old was {concept} when this event occurred according to the text?", sentence

        # 2. Date / Historical Event Timing (e.g. "15th August 1947", "celebrated on")
        elif any(w in s_lower for w in ['celebrated on', 'achieved on', '1947', '1950', 'august', 'date']):
            return f"When is {concept} celebrated or observed according to the passage?", sentence

        # 3. Role / Contribution (e.g. "led", "contributed", "role", "fought", "sacrificed")
        elif any(w in s_lower for w in ['led', 'contributed', 'played', 'fought', 'sacrificed', 'role', 'leader']):
            return f"What contribution or role does {concept} have in the text?", sentence

        # 4. Definition / Identity (is, are, refers to, defines)
        elif any(w in s_lower for w in [' is ', ' are ', ' refers to ', ' defines ', ' means ']):
            return f"What is {concept}?", sentence

        # 5. Process / Function (enables, provides, allows, facilitates, calculates, measures)
        elif any(w in s_lower for w in ['enables', 'provides', 'allows', 'calculates', 'measures', 'computes']):
            return f"What function or purpose does {concept} serve?", sentence

        # Default fallback
        return f"Explain the significance of {concept} as described in the passage.", sentence

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
            
            concepts = self.extract_core_entities(sentence)
            if not concepts:
                continue

            target_concept = None
            for c in concepts:
                c_clean = c.lower().strip()
                if c_clean not in used_concepts and c_clean not in STOP_CONCEPTS:
                    target_concept = c
                    used_concepts.add(c_clean)
                    break
            
            if not target_concept:
                target_concept = concepts[0]

            # Generate unified Q&A unit
            question_text, reference_answer = self.generate_aligned_qa_unit(target_concept, sentence)
            
            count += 1
            item = {
                "id": count,
                "type": "subjective",
                "question": question_text,
                "target_concept": target_concept,
                "answer": reference_answer,
                "original_sentence": sentence
            }
            questions_data.append(item)

        self.questions = questions_data
        return questions_data
