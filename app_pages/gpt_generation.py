import streamlit as st

def run():
    st.title("🤖 Генерация текста GPT")
    st.divider()
    st.write('Здесь пользователь может вводить промпт, а GPT-модель будет генерировать текст.')
    st.write('Настройки генерации:')
    st.write('Длина выходного текста')
    st.write('Количество генераций')
    st.write('Температура / top-k / top-p')
    st.divider()
    
    prompt = st.text_area("Введите промпт:")
    
    if st.button("Сгенерировать"):
        st.write("✍️ Генерируем текст... (здесь будет логика модели)")

if __name__ == "__main__":
    run()
