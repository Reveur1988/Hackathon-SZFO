import os
from typing import List, Tuple

import pandas as pd
import streamlit as st
from numpy.typing import NDArray
from ultralytics import YOLO

from anomalib.deploy import TorchInferencer

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
        xmin.text = str(int(box[2]))
        ymin = ET.SubElement(bndbox, "ymin")
        ymin.text = str(int(box[3]))
        xmax = ET.SubElement(bndbox, "xmax")
        xmax.text = str(int(box[0]))
        ymax = ET.SubElement(bndbox, "ymax")
        ymax.text = str(int(box[1]))

    # Преобразуем структуру в XML и сохраняем в файл
    tree = ET.ElementTree(annotation)
    tree.write(xml_filename, encoding="utf-8", xml_declaration=True)


def file_selector(folder_path="./data"):
    filenames = os.listdir(folder_path)

    selected_filename = st.selectbox("Select a file", filenames)
    return os.path.join(folder_path, selected_filename)


@st.cache_resource
def get_inferencer(path: str = "weights/torch/model.pt", device: str = "gpu"):
    if os.path.exists(path):
        # Если файл модели существует, используем реальный TorchInferencer
        inferencer = TorchInferencer(
            path=path,
            device=device,
        )
        return inferencer
    else:
        import numpy as np

        # Если файла модели нет, выводим предупреждение и используем заглушку
        st.warning(f"Модель не найдена по пути: {path}. Используется заглушка.")

        # Заглушка для инференсера
        class MockInferencer:
            def predict(self, image):
                # Имитируем результат предсказания
                class MockPredictions:
                    def __init__(self):
                        self.image = (np.random.rand(224, 224, 3) * 255).astype(
                            np.uint8,
                        )  # Фиктивное изображение
                        self.pred_boxes = [
                            [30, 40, 150, 200],  # Имитируем координаты коробок
                            [80, 100, 180, 220],
                        ]
                        self.anomaly_map = np.random.rand(
                            224,
                            224,
                        )  # Имитируем карту аномалий
                        self.heat_map = np.random.rand(
                            224,
                            224,
                        )  # Имитируем тепловую карту

                # Возвращаем фальшивые предсказания
                return MockPredictions()

        return MockInferencer()


def get_stats(
    boxes: List[List[int]],
    scores: List[float],
    classes: List[str],
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Returns the count of each anomaly class detected based on a score threshold.

    :param boxes: A list of bounding boxes, each as [x_min, y_min, x_max, y_max].
    :param scores: A list of confidence scores for each bounding box.
    :param classes: A list of class names corresponding to each detected anomaly.
    :param threshold: The threshold above which a detection is considered valid.
    :return: A DataFrame with the count of each class.
    """
    # Проверка на соответствие длины списков
    if len(boxes) != len(scores) or len(scores) != len(classes):
        raise ValueError("The lengths of boxes, scores, and classes must be the same.")

    # Фильтруем аномалии по заданному порогу
    filtered_data = [
        (cls, score) for cls, score in zip(classes, scores) if score >= threshold
    ]

    # Извлекаем классы, прошедшие порог
    filtered_classes = [cls for cls, _ in filtered_data]

    # Подсчитываем количество каждого класса
    class_counts = pd.Series(filtered_classes).value_counts().reset_index()
    class_counts.columns = ["class_name", "count"]

    return class_counts


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
    boxes: list[int] = [x.boxes.xywh.cpu().numpy() for x in predictions]

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
