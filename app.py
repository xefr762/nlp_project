import streamlit as st 

#Основная страница  ----------------------------
# боковая панель
st.sidebar.title('Меню')
st.sidebar.write('Выберите страницу:')
# вставить 3 страницы
page_main = st.sidebar.button('Главная', use_container_width=True)
page01 = st.sidebar.button('1. Классификация отзывов на фильмы')
page02 = st.sidebar.button('2. Оценка степени токсичности')
page03 = st.sidebar.button('3. Генерация текста GPT-моделью')

if page_main:
    st.title('Обработка естесственного языка • Natural Language Processing')

    st.divider()

    st.subheader('Основные задачи проекта:')
    st.write('1. Классификация отзывов на фильмы')
    st.write('2. Оценка степени токсичности')  
    st.write('3. Генерация текста GPT-моделью')

    st.divider()

    st.subheader('Участники проекта:')
    st.write('Илья Крючков')
    st.write('Илья Тыщенко')
    st.write('Владислав Мороз')

elif page01:
    import pages.movie_reviews
    pages.movie_reviews.run()
elif page02:
    import pages.toxic_level
    pages.toxic_level.run()
elif page03:
    import pages.gpt_generation
    pages.gpt_generation.run()