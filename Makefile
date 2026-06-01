.PHONY: run ui install help

help:
	@echo "Usage:"
	@echo "  make run     - Run batch pipeline (main.py)"
	@echo "  make ui      - Launch Streamlit interface"
	@echo "  make install - Install dependencies with Poetry"

run:
	poetry run python src/weld_pipeline/main.py

ui:
	poetry run streamlit run streamlit_app.py

install:
	poetry install
