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
        
        texts_to_translate = []
        for q in questions_list:
            texts_to_translate.append(q['question'])
            texts_to_translate.append(q['answer'])
            
        translated_texts = translator.translate_batch(texts_to_translate)
        
        translated_list = []
        for i, q in enumerate(questions_list):
            q_copy = dict(q)
            q_copy['question'] = translated_texts[i * 2] if i * 2 < len(translated_texts) else q['question']
            q_copy['answer'] = translated_texts[i * 2 + 1] if i * 2 + 1 < len(translated_texts) else q['answer']
            q_copy['translated_lang'] = target_lang
            translated_list.append(q_copy)
            
        return translated_list
    except Exception as e:
        logging.error(f"Batch translation error: {e}")
        return questions_list
