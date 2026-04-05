.PHONY: install install-dev download prepare data train eval baselines profile-vram publish publish-model publish-dataset test lint format clean

install:
	uv sync

install-dev:
	uv sync --extra dev
	uv run pre-commit install

# Data pipeline
download:
	uv run python scripts/download_data.py

prepare:
	uv run python scripts/prepare_dataset.py

data: download prepare

# Training
train:
	uv run axolotl train configs/training/axolotl_v3_2gpu.yaml

# Evaluation
eval:
	uv run python scripts/run_eval.py

baselines:
	uv run python scripts/run_baselines.py

# Profiling
profile-vram:
	uv run python scripts/profile_vram.py

# Publishing
publish-model:
	uv run python scripts/publish_model.py model

publish-dataset:
	uv run python scripts/publish_model.py dataset

publish: publish-model publish-dataset

# Development
test:
	uv run pytest tests/ -v

lint:
	uv run ruff check src/ tests/ scripts/

format:
	uv run ruff format src/ tests/ scripts/
	uv run ruff check --fix src/ tests/ scripts/

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
