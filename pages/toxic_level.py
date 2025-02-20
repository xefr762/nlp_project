import streamlit as st

def run():
    st.title("⚠️ Оценка токсичности текста")
    st.divider()
    st.write('На этой странице пользователь может ввести текст сообщения, а модель rubert-tiny-toxicity оценит степень его токсичности.')
    st.divider()
    st.write("Введите текст, чтобы модель оценила его токсичность.")
    
    user_input = st.text_area("Введите текст:")
    
    if st.button("Проверить токсичность"):
        st.write("💡 Оценка токсичности: ... (здесь будет вывод модели)")

if __name__ == "__main__":
    run()
