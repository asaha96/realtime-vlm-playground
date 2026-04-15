.PHONY: help setup install run dry-run evaluate test lint fmt clean

help:
	@echo "VLM Orchestrator Evaluation"
	@echo "==========================="
	@echo ""
	@echo "  make setup     - Install dependencies"
	@echo "  make dry-run   - Validate inputs without VLM calls"
	@echo "  make run       - Run the pipeline (requires API key + video)"
	@echo "  make evaluate  - Evaluate output against ground truth"
	@echo "  make test      - Run test suite"
	@echo "  make lint      - Check code style"
	@echo "  make clean     - Remove generated files"
	@echo ""
	@echo "Set OPENROUTER_API_KEY before running 'make run'"
	@echo ""

setup: install
	@echo ""
	@echo "Setup complete. Next steps:"
	@echo "  1. export OPENROUTER_API_KEY=your_key"
	@echo "  2. make dry-run"
	@echo "  3. Implement src/run.py"
	@echo "  4. make run"

install:
	pip install --upgrade pip setuptools wheel
	pip install -r requirements.txt

# Run pipeline on a specific video + procedure
# Override VIDEO and PROCEDURE via env or args:
#   make run VIDEO=path/to/video.mp4 PROCEDURE=data/procedures/change_circuit_breaker.json
# Override these when running:
#   make run VIDEO=data/videos_full/CLIP/Export_py/Video_pitchshift.mp4 PROCEDURE=data/clip_procedures/CLIP.json
PROCEDURE ?= data/clip_procedures/R066-15July-Circuit-Breaker-part2.json
VIDEO ?= data_videos/videos_full/R066-15July-Circuit-Breaker-part2/Export_py/Video_pitchshift.mp4
OUTPUT ?= output/events.json
SPEED ?= 1.0

run:
	python src/run.py \
		--procedure $(PROCEDURE) \
		--video $(VIDEO) \
		--output $(OUTPUT) \
		--speed $(SPEED)

dry-run:
	python src/run.py \
		--procedure $(PROCEDURE) \
		--video $(VIDEO) \
		--dry-run

GROUND_TRUTH ?= data/ground_truth_sample/R066-15July-Circuit-Breaker-part2.json

evaluate:
	python -m src.evaluator --predicted $(OUTPUT) --ground-truth $(GROUND_TRUTH) --tolerance 5.0

test:
	pytest tests/ -v --cov=src 2>/dev/null || echo "No tests found yet. Add tests in tests/"

lint:
	pylint src/ || true

fmt:
	black src/ --line-length 100 || echo "black not installed (optional)"

clean:
	rm -rf output/*.json __pycache__ src/__pycache__ .pytest_cache .coverage

.DEFAULT_GOAL := help
