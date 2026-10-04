from deep_translator import GoogleTranslator
import logging

logging.basicConfig(level=logging.INFO)

SUPPORTED_LANGUAGES = {
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'hi': 'Hindi',
    'ta': 'Tamil',
    'te': 'Telugu',
    'zh-CN': 'Chinese (Simplified)',
    'ja': 'Japanese',
    'ar': 'Arabic',
    'ru': 'Russian',
    'pt': 'Portuguese',
    'it': 'Italian'
}

def translate_text(text, target_lang='en'):
    if not text or target_lang == 'en':
        return text
    
    try:
        translator = GoogleTranslator(source='auto', target=target_lang)
        return translator.translate(text) or text
    except Exception as e:
        logging.error(f"Translation error: {e}")
        return text

def translate_questions(questions_list, target_lang='en'):
    if target_lang == 'en' or not questions_list:
        return questions_list
    
    try:
        translator = GoogleTranslator(source='auto', target=target_lang)
        
        translated_list = []
        for q in questions_list:
            q_copy = dict(q)
            try:
                q_copy['question'] = translator.translate(q['question']) or q['question']
                q_copy['answer'] = translator.translate(q['answer']) or q['answer']
                if 'options' in q and isinstance(q['options'], list):
                    q_copy['options'] = [translator.translate(opt) or opt for opt in q['options']]
                q_copy['translated_lang'] = target_lang
            except Exception as item_err:
                logging.error(f"Translation item error: {item_err}")
            translated_list.append(q_copy)
            
        return translated_list
    except Exception as e:
        logging.error(f"Batch translation error: {e}")
        return questions_list
