# Oblako — сервис по созданию облака слов
![Проект закрыт](https://img.shields.io/badge/%D0%9F%D1%80%D0%BE%D0%B5%D0%BA%D1%82%20%D0%B7%D0%B0%D0%BA%D1%80%D1%8B%D1%82-red)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![BeautifulSoup4](https://img.shields.io/badge/BeautifulSoup4-4C4C4C)
![matplotlib](https://img.shields.io/badge/matplotlib-11557C?logo=matplotlib&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)

***Проект для регионального этапа хакатона МПИТ. (1 место)***

## Основная информация
Веб-приложение реализовано на **Streamlit**, на вход принимается запрос, по нему приложение парсит тексты из интернета, фильтрует и выделяет ключевые слова (оставляем существительные и прилагательные, ибо они несут основную смысловую нагрузку), дальше создается облако слов.

Пользователь может вручную выбрать одну из предложеннзы палитр, фон, шрифт и форму облака слов, так же есть возможность скачать итоговое изображение.

***Стек проекта: Python, Streamlit, pymorphy2, natasha, BeautifulSoup4, matplotlib, numpy.***


## Быстрый старт (Python 3.12)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

*P.S. если сервис внезапно упал, тыкните меня, восстановлю обратно,
а еще вы все так же можете глянуть демо [(тык!)](https://disk.360.yandex.ru/i/ivcFtdSBPqqbKQ).*

***made by shpingalety, 2024***
