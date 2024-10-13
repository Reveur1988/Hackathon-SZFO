.PHONY: *

PYTHON_EXEC := python3.10

CLEARML_PROJECT_NAME := "SZFO"
CLEARML_DATASET_NAME := original


setup_ws:
	poetry env use $(PYTHON_EXEC)
	poetry install --with notebooks
	poetry run pre-commit install
	@echo
	@echo "Virtual environment has been created."
	@echo "Path to Python executable:"
	@echo `poetry env info -p`/bin/python

run_training:
	poetry run $(PYTHON_EXEC) -m src.train

run_service:
	poetry run $(PYTHON_EXEC) -m src.run_streamlit
