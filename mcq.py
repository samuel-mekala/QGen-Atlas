import os
import tempfile
import random
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

class MCQTest:
    def __init__(self, text, num_questions=5):
        self.text = text
        try:
            self.num_questions = int(num_questions)
        except (ValueError, TypeError):
            self.num_questions = 5
        self.questions = []

    def extract_noun_phrases(self, sentence):
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
                    noun_phrases.append(phrase.strip())
        return noun_phrases

    def generate_questions(self):
        try:
            sentences = sent_tokenize(self.text)
        except LookupError:
            _ensure_nltk()
            sentences = sent_tokenize(self.text)

        if not sentences:
            return []

        # Collect pool of all noun phrases from full text for distractor pool
        all_noun_phrases = []
        for s in sentences:
            all_noun_phrases.extend(self.extract_noun_phrases(s))
        
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

        for sentence in valid_sentences:
            if count >= self.num_questions:
                break

            nps = self.extract_noun_phrases(sentence)
            if not nps:
                continue

            target_ans = max(nps, key=len)
            question_text = sentence.replace(target_ans, "________", 1)

            # Build distractors distinct from target_ans
            distractors = [np for np in unique_np_pool if np.lower() != target_ans.lower() and len(np) > 2]
            
            for fb in fallback_distractors:
                if len(distractors) >= 3:
                    break
                if fb.lower() != target_ans.lower() and fb.lower() not in [d.lower() for d in distractors]:
                    distractors.append(fb)

            chosen_distractors = random.sample(distractors, min(3, len(distractors)))
            
            options_pool = [target_ans] + chosen_distractors
            random.shuffle(options_pool)

            count += 1
            item = {
                "id": count,
                "type": "mcq",
                "question": question_text,
                "options": options_pool,
                "answer": target_ans,
                "original_sentence": sentence
            }
            questions_data.append(item)

        self.questions = questions_data
        return questions_data
