import streamlit as st
from src.service import detect_anomalies
import cv2
from pathlib import Path


def upload_serial_photos(texts, language, upload_path, inferencer):
    st.subheader(texts["instruction_container_1"][language])

    dataframes = {}
    images_for_label = {}
    with st.form(
        key=f"form_{len(st.session_state['serial_photo_pairs'])}",
        clear_on_submit=True,
    ):
        serial_number = st.text_input(
            label=texts["serial_number_label"][language],
            value="",
            key=f"serial_input_{len(st.session_state['serial_photo_pairs'])}",
        )

        uploaded_photos = st.file_uploader(
            label=texts["upload_label"][language],
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key=f"photos_uploader_{len(st.session_state['serial_photo_pairs'])}",
        )

        add_button = st.form_submit_button(label=texts["add_button"][language])

        if add_button:
            if serial_number and uploaded_photos:
                uploaded_photos_list = list(uploaded_photos)
                st.session_state["serial_photo_pairs"].append(
                    {
                        "serial_number": serial_number,
                        "photos": uploaded_photos_list,
                    },
                )
                st.success(texts["data_added"][language])
            else:
                st.warning(texts["please_add_serial_and_photos"][language])

    if st.session_state["serial_photo_pairs"]:
        st.write(texts["instruction_container_1"][language])
        for idx, pair in enumerate(st.session_state["serial_photo_pairs"]):
            with st.expander(
                f"{texts['serial_number'][language]}: {pair['serial_number']}",
            ):
                image_files = []
                serial_number = pair["serial_number"]
                for photo in pair["photos"]:
                    # st.image(photo, caption=photo.name, width=150)

                    new_filename = f"{serial_number}_{photo.name}"
                    photo_path = upload_path / new_filename
                    image_files.append(photo_path)
                    # Сохраняем файл
                    with open(photo_path, "wb") as f:
                        f.write(photo.getbuffer())

                if image_files:
                    boxes, results = detect_anomalies(
                        image_files,
                        inferencer,
                    )
                    dataframes[serial_number] = results
                    select_image_data = st.dataframe(
                        dataframes[serial_number],
                        hide_index=True,
                        on_select="rerun",
                        selection_mode="multi-row",
                    )

                    selected_rows = select_image_data.selection.rows
                    selected_data = results.iloc[selected_rows]
                    images_for_label[serial_number] = list(
                        selected_data["Имя файла"].values,
                    )

                # Количество изображений в строке
                cols_per_row = 5

                # Проходим по всем изображениям
                for i in range(0, len(image_files), cols_per_row):
                    # Создаем столбцы для текущей строки
                    cols = st.columns(cols_per_row)

                    for col, image_path, bounding_boxes in zip(
                        cols,
                        image_files[i : i + cols_per_row],
                        boxes,
                    ):
                        # Открываем изображение
                        image = cv2.imread(image_path)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                        # Нарисуйте каждую bounding box
                        for box in bounding_boxes:
                            x_min, y_min, x_max, y_max = map(int, box)
                            cv2.rectangle(
                                image,
                                (x_min, y_min),
                                (x_max, y_max),
                                (255, 0, 0),
                                3,
                            )  # Красная рамка, толщина 3 пикселя

                        # Выводим изображение в соответствующем столбце
                        col.image(
                            image,
                            use_column_width=True,
                            caption=Path(image_path).name,
                        )
    v = list(images_for_label.values())
    if v and len(v[0]) > 0:
        text_button = texts["final_relabel_button"][language]
    else:
        text_button = texts["final_report_button"][language]

    if st.button(text_button, key="final_submit"):
        if st.session_state["serial_photo_pairs"]:
            st.success(texts["data_successfully_submitted"][language])
            st.session_state["image_for_label"] = images_for_label
            st.session_state["dataframes"] = dataframes
            st.session_state["step"] = "relabel"

        else:
            st.warning(texts["please_add_serial_and_photos_warning"][language])
