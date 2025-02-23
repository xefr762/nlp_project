import re
import nltk
import string
import pymorphy2
import numpy as np
import pandas as pd
import torch
from collections import Counter
from nltk.corpus import stopwords
from gensim.models import Word2Vec

nltk.download('stopwords')
stop_words = set(stopwords.words('russian'))

morph = pymorphy2.MorphAnalyzer()

# Функция очистки и лемматизации текста
def clean_text(text):    
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))                          # Удаляем пунктуацию
    tokens = text.split()                                                                     # Разбиваем на слова
    tokens = [morph.parse(word)[0].normal_form for word in tokens if word not in stop_words]  # Лемматизация
    tokens = [word for word in tokens if len(word) > 1]                                       # Убираем короткие слова
    return tokens                                                                             # Возвращаем список токенов

# Загружаем word2vec
w2v_model = Word2Vec.load("w2v.model")

def cc_cleaner(text):
    clean_text = text
    new_text = re.sub(r'[^а-яё\s]', '', clean_text)
    return new_text

# Построение словаря
def build_vocab(corpus, min_freq=5):
    all_words = [word for text in corpus for word in text]
    word_counts = Counter(all_words)
    sorted_words = [word for word, count in word_counts.items() if count >= min_freq]
    vocab_to_int = {word: idx + 1 for idx, word in enumerate(sorted_words)}
    return vocab_to_int

# загружаем тренировочный датасет и строим словарь.
df = pd.read_csv('save_true.csv')
corpus = df['Content'].astype('str').apply(cc_cleaner).to_list()
vocab_to_int = build_vocab([text.split() for text in corpus], min_freq=5)

# Текст в числа
def text_to_sequence(tokens, vocab_to_int):
    return [vocab_to_int[word] for word in tokens if word in vocab_to_int]

# Функция для паддинга
def padding(sequence, max_length=80):
    padded = np.zeros(max_length, dtype=int)
    sequence = sequence[:max_length]
    padded[-len(sequence):] = sequence
    return padded

# Создаём embedding_matrix из Word2Vec
def create_embedding_matrix(vocab_to_int, w2v_model, embedding_dim=128):
    vocab_size = len(vocab_to_int) + 1
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    for word, idx in vocab_to_int.items():
        if word in w2v_model.wv:
            embedding_matrix[idx] = w2v_model.wv[word]
    
    return torch.FloatTensor(embedding_matrix)

# Функция паредобработки текста от пользователя
def preprocess_user_input(text, vocab_to_int, max_length=80):
    tokens = clean_text(text)
    sequence = text_to_sequence(tokens, vocab_to_int)                    # Преобразование в индексы
    padded_sequence = padding(sequence, max_length)                      # Паддинг

    return torch.tensor(padded_sequence, dtype=torch.long).unsqueeze(0)

# Генерируем embedding_matrix
embedding_matrix = create_embedding_matrix(vocab_to_int, w2v_model)
