from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import nltk

NLTK_LOCAL_PATH = os.path.join(os.path.dirname(__file__), 'nltk_data')
if os.path.exists(NLTK_LOCAL_PATH) and NLTK_LOCAL_PATH not in nltk.data.path:
    nltk.data.path.insert(0, NLTK_LOCAL_PATH)

from objective import ObjectiveTest
from subjective import SubjectiveTest
from BERT_translate_custom import translate_questions, translate_text, SUPPORTED_LANGUAGES
from validation import evaluate_response

app = Flask(__name__)
app.secret_key = "qgen_atlas_secret_key_super_secure"

# In-memory session store for active quiz items
CURRENT_QUIZ = {}

@app.route('/')
def index():
    default_text = (
        "Natural Language Processing (NLP) is a subfield of artificial intelligence that focuses "
        "on the interaction between computers and human language. Query generation involves "
        "automatically extracting key information and creating relevant questions based on source text. "
        "Multilingual NLP frameworks enhance accessibility by providing translation across multiple target languages. "
        "Response validation evaluates user answers against expected reference answers using algorithms like "
        "Levenshtein distance to calculate match percentage accurately."
    )
    return render_template('index.html', languages=SUPPORTED_LANGUAGES, default_text=default_text)

@app.route('/generate', methods=['POST'])
def generate():
    text = request.form.get('input_text', '').strip()
    num_questions = int(request.form.get('num_questions', 5))
    test_type = request.form.get('test_type', 'objective')
    target_lang = request.form.get('target_lang', 'en')

    if not text:
        return redirect(url_for('index'))

    if test_type == 'objective':
        generator = ObjectiveTest(text, num_questions=num_questions)
    else:
        generator = SubjectiveTest(text, num_questions=num_questions)

    raw_questions = generator.generate_questions()

    if target_lang != 'en':
        translated_questions = translate_questions(raw_questions, target_lang=target_lang)
    else:
        translated_questions = raw_questions

    CURRENT_QUIZ['questions'] = translated_questions
    CURRENT_QUIZ['test_type'] = test_type
    CURRENT_QUIZ['target_lang'] = target_lang
    CURRENT_QUIZ['source_text'] = text

    return render_template('quiz.html', 
                           questions=translated_questions, 
                           test_type=test_type, 
                           target_lang=target_lang,
                           lang_name=SUPPORTED_LANGUAGES.get(target_lang, target_lang))

@app.route('/validate', methods=['POST'])
def validate():
    questions = CURRENT_QUIZ.get('questions', [])
    if not questions:
        return redirect(url_for('index'))

    results = []
    total_score = 0.0

    for item in questions:
        q_id = str(item['id'])
        user_ans = request.form.get(f'user_answer_{q_id}', '').strip()
        expected_ans = item['answer']
        
        eval_res = evaluate_response(user_ans, expected_ans, question_type=CURRENT_QUIZ.get('test_type', 'objective'))
        eval_res['question'] = item['question']
        eval_res['question_id'] = item['id']
        
        results.append(eval_res)
        total_score += eval_res['similarity_percentage']

    avg_score = round(total_score / len(results), 2) if results else 0.0

    return render_template('results.html', 
                           results=results, 
                           avg_score=avg_score, 
                           test_type=CURRENT_QUIZ.get('test_type', 'objective'),
                           target_lang=CURRENT_QUIZ.get('target_lang', 'en'),
                           lang_name=SUPPORTED_LANGUAGES.get(CURRENT_QUIZ.get('target_lang', 'en'), 'English'))

@app.route('/api/generate', methods=['POST'])
def api_generate():
    data = request.get_json() or {}
    text = data.get('text', '')
    num_q = data.get('num_questions', 5)
    test_type = data.get('test_type', 'objective')
    target_lang = data.get('target_lang', 'en')

    if test_type == 'objective':
        gen = ObjectiveTest(text, num_questions=num_q)
    else:
        gen = SubjectiveTest(text, num_questions=num_q)

    questions = gen.generate_questions()
    if target_lang != 'en':
        questions = translate_questions(questions, target_lang=target_lang)

    return jsonify({"status": "success", "questions": questions})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)
