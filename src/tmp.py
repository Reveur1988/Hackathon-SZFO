#     if uploaded_file:
#         # В правой колонке - отображение изображения и результатов
#         image = Image.open(uploaded_file)
#         # tensor_image = read_image(uploaded_file, as_tensor=True)
#         st.divider()
#         # Создаем временную директорию
#         temp_dir = tempfile.mkdtemp()

#         # Извлекаем имя файла изображения
#         image_name = uploaded_file.name

#         # Путь для сохранения изображения в временной директории
#         temp_image_path = Path(temp_dir) / image_name
#         print(temp_image_path)
#         # Копируем изображение во временную директорию
#         cv2.imwrite(temp_image_path, np.array(image))

#         # Анализ изображения
#         results = detect_anomalies(
#             [temp_image_path],
#             inferencer,
#         )

#         # Создание колонок для отображения результатов
#         # col1, _, col2, _, col3, _ = st.columns([3, 1, 3, 1, 3, 1])

#         # # Отображение оригинального изображения, карты аномалий и тепловой карты в колонках
#         # with col1:
#         #     st.subheader("Оригинал")
#         #     result_resized_image = resize_image(
#         #         result_image,
#         #         max_width=MAX_WIDTH,
#         #         max_height=MAX_HEIGHT,
#         #     )
#         #     st.image(result_resized_image, use_column_width=True)

#         # with col2:
#         #     st.subheader("Карта аномалий")
#         #     anomaly_map_resized = resize_image(
#         #         anomaly_map,
#         #         max_width=MAX_WIDTH,
#         #         max_height=MAX_HEIGHT,
#         #     )
#         #     st.image(anomaly_map_resized, use_column_width=True)

#         # with col3:
#         #     st.subheader("Тепловая карта")
#         #     heatmap_resized = resize_image(
#         #         heatmap,
#         #         max_width=MAX_WIDTH,
#         #         max_height=MAX_HEIGHT,
#         #     )
#         #     st.image(heatmap_resized, use_column_width=True)

#     with col_right:
#         if uploaded_file:
#             st.subheader("Статистика")
#             st.dataframe(results, hide_index=True)

# # Страница 2: Анализ папки изображений
# elif page == "Анализ папки с изображениями":
#     col_left, _, col_right = st.columns([3, 1, 8])
#     with col_left:
#         st.subheader("Анализ папки с изображениями")

#         # По умолчанию смотрит директори в ./data
#         folder_path = file_selector()

#     if folder_path and st.button("Запустить анализ"):
#         folder = Path(folder_path)

#         if folder.is_dir():
#             # Поиск всех изображений с расширениями .jpg, .jpeg, .png в папке
#             image_files = (
#                 list(folder.rglob("*.jpg"))
#                 + list(folder.rglob("*.jpeg"))
#                 + list(folder.rglob("*.png"))
#             )

#             if image_files:
#                 results = detect_anomalies(
#                     image_files,
#                     inferencer,
#                 )
#                 # for image_file in image_files:
#                 #     image = Image.open(image_file)
#                 #     tensor_image = read_image(str(image_file), as_tensor=True)

#                 #     # TODO batch prediction
#                 #     # Анализ изображения
#                 #     stats, result_image, anomaly_map, heatmap = detect_anomalies(
#                 #         tensor_image,
#                 #         inferencer,
#                 #     )

#                 #     # Преобразуем stats в длинный формат с помощью stack() и добавляем имя файла
#                 #     stacked_stats = stats.stack().reset_index()

#                 #     # Переименование колонок
#                 #     stacked_stats.columns = [
#                 #         "Индекс",  # Первый уровень индекса, если требуется
#                 #         "Вид аномалии",  # Переименование для типа аномалии
#                 #         "Количество",  # Количество аномалий
#                 #     ]
#                 #     stacked_stats["Имя"] = (
#                 #         image_file.name
#                 #     )  # Добавляем колонку с именем файла

#                 #     # Пивотируем таблицу, чтобы метрики стали отдельными колонками
#                 #     pivoted_stats = stacked_stats.pivot_table(
#                 #         index=["Имя"],
#                 #         columns="Вид аномалии",
#                 #         values="Количество",
#                 #     ).reset_index()

#                 #     # Добавляем результаты анализа в список
#                 #     results.append(pivoted_stats)

#                 # Объединяем все DataFrame в один
#                 # final_results_df = pd.concat(results, ignore_index=True)

#                 # Выводим результаты в таблице
#                 st.subheader("Результаты анализа")
#                 st.dataframe(results, hide_index=True)
#             else:
#                 st.warning("В папке нет изображений для анализа.")
#         else:
#             st.warning("Указанная папка не существует.")
