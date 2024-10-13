from pathlib import Path

import streamlit as st
from ultralytics import YOLO
import yaml
from src.service.upload import upload_serial_photos
from src.service.relabel import relabel_serial_photos
import pandas as pd
import shutil

# Settings
st.set_page_config(layout="wide")


def get_report():
    dataframes = st.session_state["dataframes"]

    result = pd.concat([x for x in dataframes.values()], ignore_index=True)
    st.dataframe(result)


def reboot():
    st.session_state = {}


# Создаем сессию для сохранения данных между изменениями этапов
if "language" not in st.session_state:
    st.session_state["language"] = None
if "serial_photo_pairs" not in st.session_state:
    st.session_state["serial_photo_pairs"] = []
if "is_uploaded" not in st.session_state:
    st.session_state["is_uploaded"] = False
if "step" not in st.session_state:
    st.session_state["step"] = "upload"  # Начинаем с первого контейнера

# Выбор языка
st.sidebar.title("Выберите язык / Choose a language / 选择语言")
language_options = {"Русский": "rus", "English": "eng", "中文": "zh"}
language_selection = st.sidebar.selectbox(
    "",
    options=list(language_options.values()),  # Выбор на основе кодов языка
    format_func=lambda code: list(language_options.keys())[
        list(language_options.values()).index(code)
    ],  # Отображаем полные названия
    index=0,
)
st.session_state["language"] = language_selection

# Получаекм инференсер
inferencer = YOLO("weights/best.pt")
# Загрузка текстов из YAML в Python словарь
with open("src/service/texts.yaml", "r", encoding="utf-8") as file:
    texts = yaml.safe_load(file)

# Создаем временную директорию
upload_path = Path("upload_data")
upload_path.mkdir(exist_ok=True)

# Переключаем язык на тот который в боковом меню
if st.session_state["language"] is not None:
    language = st.session_state["language"]

    # Используем контейнеры для создания независимых областей для добавления серийных номеров и фотографий
    # Этап 1: Загрузка файлов и ввод серийных номеров. На этом этапе фотографии сохраняются в папку.
    # после чего шаг переключается на 2

    if st.session_state["step"] == "upload":
        with st.container():
            # Заголовок сервиса
            st.title(texts["welcome"][language])
            upload_serial_photos(texts, language, upload_path, inferencer)

    # Этап 3: Делаем проверку фотографий и ручную переразметку
    if st.session_state["step"] == "relabel":
        # Переключаем контейнер чтобы исчезла форма загрузки файлов
        relabel_dir = Path("relabel_data")

        if relabel_dir.is_dir():
            shutil.rmtree(relabel_dir)
        relabel_dir.mkdir(exist_ok=True)

        images_for_relabel = list(st.session_state["image_for_label"].values())

        if len(images_for_relabel[0]) > 0:
            for x in images_for_relabel:
                for k in x:
                    image_path = upload_path / k
                    xml_path = image_path.parent / (image_path.stem + ".xml")
                    shutil.copy(image_path, relabel_dir)
                    shutil.copy(xml_path, relabel_dir)

            with st.container():
                st.subheader(texts["instruction_container_2"][language])
                # задаем имена классов на выбранном языке
                label_keys = [
                    "dead_pixel",
                    "scratch",
                    "missing_screw",
                    "keyboard_damage",
                    "broken_lock",
                ]
                relabel_serial_photos(relabel_dir, label_keys)
        else:
            st.session_state["step"] = "final"

    if st.session_state["step"] == "final":
        with st.container():
            st.title(texts["final_report"][language])
            get_report()
            st.button("Вернуться к загрузке", on_click=reboot)
