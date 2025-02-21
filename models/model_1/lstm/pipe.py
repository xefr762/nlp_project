import re
import nltk
import string
import pymorphy2
from nltk.corpus import stopwords
import numpy as np

import torch
from torch import nn

from collections import Counter

nltk.download('stopwords')

import gensim
from gensim.models import Word2Vec

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

model = Word2Vec.load("w2v.model")

def get_words_by_freq(sorted_words: list[tuple[str, int]], n: int = 10) -> list:
    return list(filter(lambda x: x[1] > n, sorted_words))

def preprocessor(message):
    
    corpus = [word for text in message for word in text.split()]
    sorted_words = Counter(corpus).most_common()
    porog = 5

    reviews_int = []
    for text in message:
        r = [vocab_to_int[word] for word in text.split() if vocab_to_int.get(word)]
        reviews_int.append(r)

    sorted_words = get_words_by_freq(sorted_words, porog)
    vocab_to_int = {w:i+1 for i, (w,c) in enumerate(sorted_words)}

    VOCAB_SIZE = len(vocab_to_int) + 1
    EMBEDDING_DIM = 128

    embedding_matrix = np.zeros((VOCAB_SIZE, EMBEDDING_DIM))

    # Бежим по всем словам словаря: если слово есть в word2vec, 
    # достаем его вектор; если слова нет, то распечатываем его и пропускаем
    for word, i in vocab_to_int.items():
            embedding_vector = model.wv[word]
            embedding_matrix[i] = embedding_vector
    embedding_layer = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix))

    return embedding_layer, reviews_int

HIDDEN_SIZE = 32
SEQ_LEN = 80
BATCH_SIZE = 256
EMBEDDING_DIM = 128

def padding(review_int):
    features = np.zeros((len(review_int), 80), dtype=int)

    for i, review in enumerate(review_int):
        if not isinstance(review, list):
            raise ValueError(f"Ошибка: ожидался список, но получено {type(review)}")
        if len(review) == 0:
            continue  
        features[i, -min(len(review), 80):] = np.array(review[:80])
    
    return features