.PHONY: install install-dev install-axolotl download prepare data train train-1gpu eval baselines profile-vram publish test lint format clean

install:
	uv sync

install-dev:
	uv sync --extra dev
	uv run pre-commit install

install-axolotl:
	uv sync

# Data pipeline
download:
	uv run python scripts/download_data.py

prepare:
	uv run python scripts/prepare_dataset.py

data: download prepare

# Training with Axolotl
train:
	uv run axolotl train configs/training/axolotl_2gpu.yaml

train-1gpu:
	uv run axolotl train configs/training/axolotl_1gpu.yaml

# Evaluation
eval:
	uv run python scripts/run_eval.py

baselines:
	uv run python scripts/run_baselines.py

# Profiling
profile-vram:
	uv run python scripts/profile_vram.py

# Publishing
publish:
	uv run python scripts/publish_model.py

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
