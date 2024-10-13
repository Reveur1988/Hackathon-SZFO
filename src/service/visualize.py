import cv2
from numpy.typing import NDArray


def resize_image(
    image: NDArray,
    max_width: int = 800,
    max_height: int = 800,
) -> NDArray:
    """
    Resizes an image represented as an NDArray while maintaining aspect ratio
    if it exceeds the max width or height.

    :param image: The input image as a NumPy array (BGR format).
    :param max_width: The maximum width allowed for the image.
    :param max_height: The maximum height allowed for the image.
    :return: The resized image as a NumPy array.
    """
    # Получаем текущие размеры изображения
    original_height, original_width = image.shape[:2]

    # Проверка, нужно ли изменение размера
    if original_width > max_width or original_height > max_height:
        # Вычисляем соотношение сторон
        aspect_ratio = original_width / original_height

        # Определяем новые размеры с сохранением пропорций
        if aspect_ratio > 1:  # Ширина больше высоты
            new_width = min(original_width, max_width)
            new_height = int(new_width / aspect_ratio)
        else:  # Высота больше ширины или квадратное изображение
            new_height = min(original_height, max_height)
            new_width = int(new_height * aspect_ratio)

        # Изменяем размер изображения с помощью OpenCV
        resized_image = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA,
        )
    else:
        # Если размеры меньше максимальных, возвращаем оригинальное изображение
        resized_image = image

    return resized_image


def draw_rounded_boxes(
    image,
    boxes,
    pred_score,
    threshold=0.5,
    color=(0, 255, 0),
    thickness=2,
    radius=50,
) -> NDArray:
    """
    Draws rounded rectangles for each bounding box on the image.

    :param image: The input image as a NumPy array (H, W, C).
    :param boxes: A list of bounding boxes, each as [x_min, y_min, x_max, y_max].
    :param color: The color of the rectangle in BGR format (default is green).
    :param thickness: Thickness of the rectangle lines.
    :param radius: Radius for the rounded corners.
    :return: Image with drawn rounded rectangles.
    """

    # Iterate through all bounding boxes
    for box, score in zip(boxes, pred_score):
        if score > threshold:
            x_min, y_min, x_max, y_max = map(int, box)

            # Draw the arcs for rounded corners
            cv2.ellipse(
                image,
                (x_min + radius, y_min + radius),
                (radius, radius),
                180,
                0,
                90,
                color,
                thickness,
            )
            cv2.ellipse(
                image,
                (x_max - radius, y_min + radius),
                (radius, radius),
                270,
                0,
                90,
                color,
                thickness,
            )
            cv2.ellipse(
                image,
                (x_max - radius, y_max - radius),
                (radius, radius),
                0,
                0,
                90,
                color,
                thickness,
            )
            cv2.ellipse(
                image,
                (x_min + radius, y_max - radius),
                (radius, radius),
                90,
                0,
                90,
                color,
                thickness,
            )

    return image
