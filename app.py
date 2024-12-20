import streamlit as st
import nltk
import numpy as np
import pymorphy2
import requests
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from natasha import MorphVocab, Doc, Segmenter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from googlesearch import search
from bs4 import BeautifulSoup
from PIL import Image
import io

morph = pymorphy2.MorphAnalyzer()
morph_vocab = MorphVocab()
segmenter = Segmenter()

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

stop_words = set(stopwords.words('russian'))
not_about_tourism = ['личный', 'кабинет', 'тур', 'согласие', 'вход', 'офис', 'агентство', 'компания', 'ул', 'поиск',
                     'набор', 'реестр', 'вариант', 'зима', 'день', 'январь', 'декабрь', 'дата', 'год', 'средний',
                     'зимний', 'выше', 'высокий', 'область', 'край', 'регион', 'проживание', 'новый', 'каникулы', 'вид',
                     'сложность', 'лето', 'весна', 'осень', 'человек', 'весь', 'отдых', 'нужно', 'россия', 'среднее',
                     'тот', 'который', 'рубль', 'руб', 'ввод', 'неверный', 'нагрузка', 'февраль', 'март', 'ввод',
                     'серия', 'дед', 'ноябрь', 'апрель', 'этот', 'данный', 'статья', 'страница', 'другой', 'свой',
                     'русский', 'дек', 'апр', 'км', 'ной', 'наш']

def get_search_results(query, num_results=5):
    try:
        results = search(query, num_results=num_results, lang='ru')
        return list(results)
    except Exception as e:
        st.write(f"Ошибка при поиске: {e}")
        return []

def extract_text_from_url(url):
    try:
        response = requests.get(url)
        response.encoding = response.apparent_encoding
        soup = BeautifulSoup(response.text, 'html.parser')
        body = soup.find('body')
        if body:
            for script in body(["script", "style"]):
                script.decompose()
            text = body.get_text(separator=' ')
            return text
        else:
            return ""
    except Exception as e:
        st.write(f"Ошибка при извлечении текста из {url}: {e}")
        return ""

def get_pos(word):
    parsed = morph.parse(word)
    if parsed:
        return parsed[0].tag.POS
    return None

def is_noun_or_adjective(word):
    pos = get_pos(word)
    return pos in ('NOUN', 'ADJF', 'ADJS')

def process_query(query, num_results=5):
    urls = get_search_results(query, num_results=num_results)
    all_text = []
    for url in urls:
        text = extract_text_from_url(url).strip()
        all_text.append(text)

    full_text = ' '.join(all_text)
    tokens = word_tokenize(full_text)

    filtered_tokens = []
    normal_forms_cache = {}

    for word in tokens:
        lower_word = word.lower()
        if word.isalpha() and lower_word not in stop_words:
            if is_noun_or_adjective(lower_word):
                if lower_word not in normal_forms_cache:
                    parsed_word = morph.parse(lower_word)[0]
                    normal_forms_cache[lower_word] = parsed_word.normal_form
                infinitive = normal_forms_cache[lower_word]
                if infinitive not in not_about_tourism:
                    filtered_tokens.append(infinitive.capitalize())

    word_freq = Counter(filtered_tokens)
    return word_freq.most_common(50)

st.title("Визуализация облака слов для запросов")

query = st.text_input("Введите ваш запрос для парсинга", value="поездки на новогодних каникулах по России в 2025")
num_results = st.number_input("Количество результатов для парсинга (больше число - точнее подобраны слова, меньше число - быстрее генерация", min_value=1, max_value=20, value=5)

st.write("Настройки облака слов:")

bg_color = st.color_picker("Выберите цвет фона облака", "#FFFFFF")

custom_colors = ['#A54040', '#B96E6E', '#CD9C9C', '#98C665', '#B0D28A', '#C7DDAD', '#A57865', '#BA988A']
palettes = {
    "Пастельная": ['#F1B2B2', '#F1D7B2', '#F1F1B2', '#B2F1B2', '#B2F1D7', '#E0B0FF'],
    "Тёмная": ['#8B0000', '#006400', '#B22222', '#483D8B', '#D2691E'],
    "Яркая": ['#FF0000', '#DAA520', '#00FF00', '#0000FF', '#9400D3', '#00FFFF']
}

use_single_color = st.checkbox("Использовать один цвет для всех слов")

if use_single_color:
    word_color = st.color_picker("Выберите цвет слов", "#000000")

    def color_func_single(*args, **kwargs):
        return word_color

    color_func = color_func_single
else:
    palette_choice = st.selectbox("Выберите палитру цветов", ["Пастельная", "Тёмная", "Яркая", "Made by shpingalety"])
    if palette_choice == "Made by shpingalety":
        def color_func_random(*args, **kwargs):
            return np.random.choice(custom_colors)
        color_func = color_func_random
    else:
        selected_palette = palettes[palette_choice]
        def color_func_palette(*args, **kwargs):
            return np.random.choice(selected_palette)
        color_func = color_func_palette

font_choice = st.selectbox("Выберите шрифт", ["Roboto", "Ubuntu", "Montserrat"])
font_paths = {
    "Roboto": "fonts/Roboto-Regular.ttf",
    "Ubuntu": "fonts/Ubuntu-Regular.ttf",
    "Montserrat": "fonts/Montserrat-Medium.ttf"
}
font_path = font_paths[font_choice]



mask_option = st.selectbox("Выберите форму облака", ("Прямоугольник", "Звезда", "Птица счастья"))
mask = None
if mask_option != "Прямоугольник":
    mask_file = {"Звезда": "masks/star.png",
                 "Птица счастья": 'masks/bird.png'}[mask_option]
    try:
        mask_image = Image.open(mask_file).convert("RGB")
        mask = np.array(mask_image)
    except:
        st.write("Не удалось загрузить маску. Проверьте путь и наличие файла.")
        mask = None

if st.button("Сгенерировать облако слов"):
    most_common_words = process_query(query, num_results=num_results)
    word_freq_dict = dict(most_common_words)

    wc = WordCloud(width=800, height=400, background_color=bg_color,
                   color_func=color_func,
                   font_path=font_path if font_path else None,
                   mask=mask)
    wc.generate_from_frequencies(word_freq_dict)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

    buf = io.BytesIO()
    wc.to_image().save(buf, format='PNG')
    buf.seek(0)

    st.download_button("Скачать облако слов", data=buf, file_name="wordcloud.png", mime="image/png")
