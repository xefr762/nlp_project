import streamlit as st
import torch
import time
from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def run():
    st.title("🎬 Классификация отзывов на фильмы")
    st.divider()
    st.write('На данной странице пользователь может ввести текст отзыва на фильм и получить пресказание классификации по 3 моделям:')
    st.write('1. Классический ML-алгоритм LogReg (на основе TF-IDF)')
    st.write('2. LSTM с attention-механизмом')
    st.write('3. BERT-based модель (rubert-tiny-sentiment-balanced)')
    st.write('Каждое предсказание сопровождается временем его вычисления. Также представлена сравнительная таблица метрик f1-macro для всех моделей')
    st.divider()
    model_choose = st.radio('Выберете модель', options = ['ML', 'LSTM', 'Bert'])
    st.write("Введите текст отзыва, и модель предскажет его тональность.")

    user_input = st.text_area("Введите текст отзыва:")
    
    if st.button("Анализировать"):
        if model_choose == 'ML':
            pass #TODO
        elif model_choose == 'LSTM':
            pass #TODO
        else:
            @st.cache_resource
            def load_model():
                model = ReviewBert()
                model.load_state_dict(torch.load("models/model_1/bert/clf_bert_v1.pth", map_location="cpu"))
                model.eval()
                return model
            @st.cache_resource
            def load_tokenizer():
                return AutoTokenizer.from_pretrained("cointegrated/rubert-tiny-sentiment-balanced")
            
            class ReviewBert(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.bert = AutoModelForSequenceClassification.from_pretrained('cointegrated/rubert-tiny-sentiment-balanced')
                    for param in self.bert.parameters():
                        param.requires_grad = False
                    self.bert.classifier = torch.nn.Sequential(
                        torch.nn.Linear(312, 256),
                        torch.nn.Tanh(),
                        torch.nn.Dropout(),
                        torch.nn.Linear(256, 128),
                        torch.nn.Tanh(),
                        torch.nn.Dropout(),
                        torch.nn.Linear(128,3)
                    )
                    for param in self.bert.classifier.parameters():
                        param.requires_grad = True
                def forward(self, input_ids, attention_mask):
                    outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                    return outputs.logits

            model = load_model()
            tokenizer = load_tokenizer()
            def tokenize_texts(texts):
                return tokenizer(texts, padding="max_length", max_length=50, truncation=True, return_tensors="pt")

            def eval_clf(text):
                start_time = time.time()
                encoded = tokenize_texts(text)
                logits = model(encoded['input_ids'], encoded['attention_mask'])
                prob = torch.softmax(logits, dim=1)
                cls = prob.argmax().item()
                end_time = time.time()
                elapsed_time = end_time - start_time
                return cls, elapsed_time

            if user_input.strip():
                res, time_s = eval_clf(user_input)
                sentiment_dict = {"Положительный": 0, "Отрицательный": 1, "Нейтральный": 2}
                reversed_dict = {v: k for k, v in sentiment_dict.items()}
                st.write(f"Отзыв скорее всего : **{reversed_dict[res]}**. Выполнено за : {time_s:.4f}")
            else:
                st.warning("Введите текст перед оценкой.")

    if st.button('Показать метрики модели'):
        st.image('images/loss_ROCAUC_cls.png', caption='Loss ROC-AUC', use_container_width=True)

