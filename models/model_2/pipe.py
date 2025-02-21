import re
import nltk
import string
import pymorphy2
from nltk.corpus import stopwords

nltk.download('stopwords')

# Инициализируем лемматизатор и список стоп-слов
morph = pymorphy2.MorphAnalyzer()
stop_words = set(stopwords.words('russian'))

def clean_text_for_person(text):
    if not isinstance(text, str):  # Проверяем, что текст - строка
        return [""]

    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))  # Удаляем пунктуацию
    tokens = text.split()  # Разбиваем на слова
    tokens = [morph.parse(word)[0].normal_form for word in tokens if word not in stop_words]  # Лемматизация
    tokens = [word for word in tokens if len(word) > 1]  # Убираем короткие слова
    cleaned_text = " ".join(tokens)

    return [cleaned_text]

def clean_text(text):
    clean_text = clean_text_for_person(text)
    new_text = re.sub(r'[^а-яё\s]', '', clean_text)
    return new_text
