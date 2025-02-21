import streamlit as st
import torch
import time
from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Загрузка токенизатора и модели с кешированием
@st.cache_resource
def load_model():
    model = ToxicBert()
    model.load_state_dict(torch.load("models/model_2/toxic_bert_v1.pth", map_location="cpu"))
    model.eval()
    return model

@st.cache_resource
def load_tokenizer():
    return AutoTokenizer.from_pretrained("cointegrated/rubert-tiny-toxicity")

# Определение модели
class ToxicBert(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained("cointegrated/rubert-tiny-toxicity")
        for param in self.bert.parameters():
            param.requires_grad = False
        
        hidden_size = self.bert.config.hidden_size  # Получаем размер скрытого слоя
        self.bert.classifier = nn.Linear(hidden_size, 1)  # Исправленный слой
        
        for param in self.bert.classifier.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

# Глобальные переменные с кешем
model = load_model()
tokenizer = load_tokenizer()

def tokenize_texts(texts):
    return tokenizer(texts, padding="max_length", max_length=50, truncation=True, return_tensors="pt")

def eval_toxicity(text):
    start_time = time.time()
    encoded = tokenize_texts(text)
    logits = model(encoded["input_ids"], encoded["attention_mask"]).squeeze().item()
    prob = torch.sigmoid(torch.tensor(logits)).item()
    end_time = time.time()
    elapsed_time = end_time - start_time
    return prob, elapsed_time

# Streamlit UI
def run():
    st.title("⚠️ Оценка токсичности текста")
    st.divider()
    st.write("На этой странице пользователь может ввести текст сообщения, а кастомная дообученная модель на основе rubert-tiny-toxicity оценит степень его токсичности.")
    st.divider()
    
    user_input = st.text_area("Введите текст:")

    if st.button("Проверить токсичность"):
        if user_input.strip():
            res, time = eval_toxicity(user_input)
            st.write(f"Оценка токсичности: **{res:.4f}**. Выполнено за : {time:.4f}")
        else:
            st.warning("Введите текст перед оценкой.")

    if st.button('Показать метрики модели'):
        st.image('images/loss_ROCAUC.png', caption='Loss ROC-AUC', use_container_width=True)
        st.image('images/pr_curve.png', caption='Preccision / Recall')