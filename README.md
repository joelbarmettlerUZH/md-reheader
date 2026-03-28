# md-reheader

A fine-tuned Qwen3-0.6B model that restores correct heading hierarchy in markdown documents up to 32k tokens. Designed to fix flat heading structures produced by PDF-to-markdown parsers.

## Problem

PDF-to-markdown tools (Marker, Nougat, etc.) often lose heading hierarchy, producing documents where all headings are the same level. md-reheader restores the correct heading structure by predicting the appropriate level (1-6) for each heading.

## Approach

- **Model:** Qwen3-0.6B, full fine-tune (not LoRA)
- **Context:** Full 32k token window with sample-packed training
- **Data:** codeparrot/github-code + euirim/goodwiki, corrupted to simulate flat headings
- **Training:** Axolotl with sample packing on 2x RTX 4090 with FSDP

## Quick Start

```bash
# Install dependencies
make install-dev

# Run tests
make test

# Profile VRAM (requires GPU)
make profile-vram

# Download data
make data

# Train
make train

# Evaluate
make eval
```

## Project Structure

```
├── configs/          # Hydra configs for data, eval; Axolotl configs for training
├── src/md_reheader/
│   ├── data/         # Extraction, filtering, corruption, formatting, batching
│   ├── training/     # Training utilities
│   ├── eval/         # Metrics, baselines, error analysis
│   └── inference/    # Prediction pipeline
├── scripts/          # CLI entrypoints
├── tests/            # Unit tests
├── notebooks/        # Exploration and analysis
└── docs/             # Model card, dataset card, VRAM profile
```

## License

Apache 2.0
