import os
from typing import List, Tuple

import pandas as pd
import streamlit as st
from numpy.typing import NDArray
from ultralytics import YOLO

from pathlib import Path

import xml.etree.ElementTree as ET
from PIL import Image


# Функция для создания XML файла с аннотациями
def save_boxes_to_xml(image_path, boxes):
    """
    Создает XML файл с таким же именем, как изображение, и записывает в него координаты боксов.

    :param image_path: Путь к изображению.
    :param boxes: Список боксов в формате [(x_min, y_min, x_max, y_max), ...].
    :param output_dir: Папка, в которую сохраняются XML файлы.
    """

    image_path = Path(image_path)
    # Открываем изображение для получения его размеров
    with Image.open(image_path) as img:
        width, height = img.size
        depth = len(
            img.getbands(),
        )  # Количество каналов (например, RGB = 3, Grayscale = 1)

    # Извлекаем имя файла без расширения
    base_filename = str(image_path.stem)
    output_dir = image_path.parent
    # Создаем путь для xml файла
    xml_filename = str(image_path.parent / f"{base_filename}.xml")

    image_path = str(image_path)
    # Создаем корневой элемент <annotation>
    annotation = ET.Element("annotation")

    # Добавляем элемент <folder>
    folder = ET.SubElement(annotation, "folder")
    folder.text = str(output_dir.name)

    # Добавляем элемент <filename>
    filename = ET.SubElement(annotation, "filename")
    filename.text = os.path.basename(image_path)

    # Добавляем элемент <path>
    path = ET.SubElement(annotation, "path")
    path.text = image_path

    # Добавляем элемент <source>
    source = ET.SubElement(annotation, "source")
    database = ET.SubElement(source, "database")
    database.text = "Unknown"

    # Добавляем элемент <size>
    size = ET.SubElement(annotation, "size")
    width_elem = ET.SubElement(size, "width")
    width_elem.text = str(width)
    height_elem = ET.SubElement(size, "height")
    height_elem.text = str(height)
    depth_elem = ET.SubElement(size, "depth")
    depth_elem.text = str(depth)

    # Добавляем элемент <segmented>
    segmented = ET.SubElement(annotation, "segmented")
    segmented.text = "0"

    # Для каждого бокса создаем объект <object>
    for box in boxes:
        obj = ET.SubElement(annotation, "object")

        # Имя объекта
        name = ET.SubElement(obj, "name")
        name.text = "dead_pixel"  # Задайте необходимое имя объекта

        pose = ET.SubElement(obj, "pose")
        pose.text = "Unspecified"

        truncated = ET.SubElement(obj, "truncated")
        truncated.text = "0"

        difficult = ET.SubElement(obj, "difficult")
        difficult.text = "0"

        # Добавляем координаты <bndbox>
        bndbox = ET.SubElement(obj, "bndbox")
        xmin = ET.SubElement(bndbox, "xmin")
        xmin.text = str(int(box[0]))
        ymin = ET.SubElement(bndbox, "ymin")
        ymin.text = str(int(box[1]))
        xmax = ET.SubElement(bndbox, "xmax")
        xmax.text = str(int(box[2]))
        ymax = ET.SubElement(bndbox, "ymax")
        ymax.text = str(int(box[3]))

    # Преобразуем структуру в XML и сохраняем в файл
    tree = ET.ElementTree(annotation)
    tree.write(xml_filename, encoding="utf-8", xml_declaration=True)


def file_selector(folder_path="./data"):
    filenames = os.listdir(folder_path)

    selected_filename = st.selectbox("Select a file", filenames)
    return os.path.join(folder_path, selected_filename)

def detect_anomalies(
    image_files: list,
    inferencer: YOLO,
) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Perform anomaly detection on the input tensor image using the provided inferencer.

    Args:
        tensor_image (torch.Tensor): Input tensor image.
        inferencer (TorchInferencer): Inferencer object for anomaly detection.
        classifier: Inferencer object for classification anomalies

    Returns:
        Tuple[NDArray, NDArray, NDArray]: Result image, anomaly map, and heatmap.
    """

    predictions = inferencer(image_files)
    # original_images: NDArray = predictions.results
    boxes: list[int] = [x.boxes.xyxy.cpu().numpy() for x in predictions]

    # Сохранение боксов для каждого изображения
    for i, image_path in enumerate(image_files):
        save_boxes_to_xml(image_path, boxes[i])

    scores = [x.boxes.conf.cpu().numpy() for x in predictions]
    result = pd.DataFrame()
    for filename, scores in zip(image_files, scores):
        count = 0
        for s in scores:
            if s > 0.5:
                count += 1
        df = pd.DataFrame(
            {"Имя файла": Path(filename).name, "Количество дефектов": [count]},
        )
        result = pd.concat([result, df], ignore_index=True)

    return boxes, result
