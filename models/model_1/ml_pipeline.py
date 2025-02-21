import re
import nltk
import string
import pymorphy2
from nltk.corpus import stopwords

import pandas as pd
import numpy as numpy
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

nltk.download('stopwords')

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

# vectorizer = joblib.load('models/model_1/vectorizer.joblib')
# encoder = joblib.load('models/model_1/LabelEncoder.joblib')

def pipeline(text):
    clean_text = clean_text_for_person(text)
    vectorizer = joblib.load('models/model_1/vectorizer.joblib')
    vectorize = vectorizer.transform(clean_text)
    return(vectorize)

def decode(preds):
    encoder = joblib.load('models/model_1/LabelEncoder.joblib')
    decoded_message = encoder.inverse_transform(preds)
    return decoded_message