.PHONY: run ui install help sweep plot-sweep

help:
	@echo "Usage:"
	@echo "  make run              - Run batch pipeline (main.py)"
	@echo "  make ui               - Launch Streamlit interface"
	@echo "  make test-porosity    - Interactive porosity test (optional: IMG=path/to/img.jpg)"
	@echo "  make sweep            - Run discontinuity parameter sweep → sweep_results.json"
	@echo "  make plot-sweep       - Plot sweep results → accuracy_surface.png"
	@echo "  make install          - Install dependencies with Poetry"

run:
	poetry run python src/weld_pipeline/main.py

ui:
	poetry run streamlit run streamlit_app.py

test-porosity:
	poetry run python tests/test_porosity.py $(IMG)

sweep:
	poetry run python sweep_test.py

plot-sweep:
	poetry run python plot_3d.py

install:
	poetry install
