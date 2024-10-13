import os
import subprocess

# Указываем путь к директории с модулем src, чтобы он был доступен для импорта
os.environ["PYTHONPATH"] = os.path.abspath("./")

# Запуск streamlit с указанием app.py
subprocess.run(["streamlit", "run", "src/app.py"])
