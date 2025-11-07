from transformers import pipeline

def translate_text(text, target_lang="ta"):
    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-ta")
    result = translator(text)
    return result[0]['translation_text']