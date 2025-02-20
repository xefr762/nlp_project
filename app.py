import streamlit as st 

#Основная страница  ----------------------------
# боковая панель
st.sidebar.title('Меню')
st.sidebar.write('Выберите страницу:')
# вставить 3 страницы
page01 = st.sidebar.button('1. Классификация отзывов на фильмы')
page02 = st.sidebar.button('2. Оценка степени токсичности')
page03 = st.sidebar.button('3. Генерация текста GPT-моделью')

# основная панель
st.title('Обработка естесственного языка • Natural Language Processing')
st.divider()
st.subheader('Основные задачи проекта:')
st.write('1. Классификация отзывов на фильмы')
st.write('2. Оценка степени токсичности')  
st.write('3. Генерация текста GPT-моделью')

st.divider()

st.subheader('Классификация отзывов на фильмы')
st.write('На данной странице пользователь может ввести текст отзыва на фильм и получить пресказание классификации по 3 моделям:')
st.write('1. Классический ML-алгоритм (на основе BagOfWords/TF-IDF)')
st.write('2. RNN/LSTM (предпочтительно с attention-механизмом)')
st.write('3. BERT-based модель')
st.write('Каждое предсказание сопровождается временем его вычисления. Также представлена сравнительная таблица метрик f1-macro для всех моделей')

st.divider()

st.subheader('Оценка степени токсичности')
st.write('На этой странице пользователь может ввести текст сообщения, а модель rubert-tiny-toxicity оценит степень его токсичности.')

st.divider()

st.subheader('Генерация текста GPT-моделью')
st.write('Здесь пользователь может вводить промпт, а GPT-модель будет генерировать текст.')
st.write('Настройки генерации:')
st.write('Длина выходного текста')
st.write('Количество генераций')
st.write('Температура / top-k / top-p')


# Приложение создано в рамках учебного проекта и демонстрирует различные подходы к обработке текста с помощью современных NLP-моделей.
# page03 = st.Page("pages/page_03.py", title = '1. Детекция лиц (YOLO)')
# page031 = st.Page("pages/page_031.py", title = '* Описание модели')

# page04 = st.Page("pages/page_04.py", title = '2a. Сегментация аэрокосм. снимков (UNET)')
# page041 = st.Page("pages/page_041.py", title = '* Описание модели')
# page042 = st.Page("pages/page_042.py", title = '2b. Сегментация аэрокосм. снимков (YOLO)')
# page043 = st.Page("pages/page_043.py", title = '* Описание модели')

# page05 = st.Page("pages/page_05.py", title = '3a. Детекция ветрогенераторов (YOLO)')
# page051 = st.Page("pages/page_051.py", title = '* Описание модели')
# page052 = st.Page("pages/page_052.py", title = '3b. Детекция ветрогенераторов (...)')
# page053 = st.Page("pages/page_053.py", title = '* Описание модели')

# page06 = st.Page("pages/page_06.py", title = '4. Семантическая детекция ветрогенераторов (YOLO)')
# page061 = st.Page("pages/page_061.py", title = '* Описание модели')

# pg = st.navigation([page01,  page02, 
#                     page03,  page031, 
#                     page04,  page041, 
#                     page042, page043, 
#                     page05,  page051,
#                     page052, page053,
#                     page06,  page061
#                     ], expanded=True)
# pg.run()


st.sidebar.title('Команда проекта: ')
st.sidebar.write('Илья Крючков')
st.sidebar.write('Илья Тыщенко')
st.sidebar.write('Владислав Морозов')
