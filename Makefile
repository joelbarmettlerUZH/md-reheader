.PHONY: install install-dev data train train-debug eval profile-vram test lint format clean

install:
	uv sync

install-dev:
	uv sync --extra dev
	uv run pre-commit install

data:
	uv run python scripts/download_data.py
	uv run python scripts/prepare_dataset.py

train:
	uv run accelerate launch --num_processes 2 scripts/run_training.py

train-debug:
	uv run python scripts/run_training.py \
		training.epochs=1 \
		model.max_seq_len=4096 \
		data.train_path=./data/processed/train_small.jsonl

eval:
	uv run python scripts/run_eval.py

baselines:
	uv run python scripts/run_baselines.py

profile-vram:
	uv run python scripts/profile_vram.py

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
