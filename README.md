# md-reheader

Restore heading hierarchy in markdown documents using a fine-tuned Qwen3-0.6B model.

PDF-to-markdown tools like Marker, Docling, and MinerU often flatten heading structure — everything becomes `#` or `##`. md-reheader predicts the correct heading levels (H1–H6) from document context and semantics, restoring the original hierarchy.

## Results

Evaluated on 7,321 test documents from GitHub markdown files and Wikipedia articles:

| Metric | Naive Flat | Heuristic | **md-reheader** |
|--------|-----------|-----------|-----------------|
| Exact match | 0.0% | 14.5% | **56.1%** |
| Per-heading accuracy | 13.1% | 49.1% | **80.6%** |
| Hierarchy preservation | 61.3% | 68.6% | **91.0%** |
| Mean absolute error | 1.38 | 0.62 | **0.22** |

Per-level accuracy:

| Level | Accuracy |
|-------|----------|
| H1 | 77% |
| H2 | 85% |
| H3 | 78% |
| H4 | 68% |
| H5 | 45% |
| H6 | 50% |

## Installation

```bash
pip install md-reheader
```

## Usage

```python
from md_reheader.inference.predict import load_model, reheader_document

model, tokenizer = load_model("joelbarmettlerUZH/md-reheader")

flat_markdown = """
# Introduction
# Background
# Methods
# Data Collection
# Results
"""

fixed = reheader_document(flat_markdown, model, tokenizer)
print(fixed)
# # Introduction
# ## Background
# ## Methods
# ### Data Collection
# ## Results
```

The `reheader_document` function handles the full pipeline: extract headings, strip and flatten the document, run inference, and apply predicted levels back to the original text.

## How It Works

### Input preprocessing

The document is preprocessed before inference:

1. All headings are flattened to `# ` (level 1) — the model should not trust input heading levels
2. Body text between headings is truncated to first 128 + last 128 tokens — enough for structural cues without bloating context
3. The total document is truncated to fit within 8k tokens

### Model

The model is a fine-tuned [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) (text-only, 0.6B parameters). It receives the preprocessed document and outputs each heading with the correct `#` prefix:

```
Input:  # Introduction\n...body...\n# Background\n...body...
Output: # Introduction\n## Background
```

The heading text in the output leverages pretraining knowledge — the model knows "Introduction" is typically H1/H2 while "2.1.3 Implementation Details" implies deep nesting.

### Training data

- **codeparrot/github-code**: ~105k markdown files from GitHub repositories
- **euirim/goodwiki**: ~45k Wikipedia articles in markdown format
- Deep documents (heading depth 4+) are oversampled 2–8x to address class imbalance
- Total: ~197k training examples after oversampling
- Split by repository/article title to prevent data leakage

### Training

- Framework: [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) (YAML-driven, no custom training code)
- Full fine-tuning, BF16, DDP across 2x RTX 4090
- 8k sequence length, sample packing
- Learning rate 5e-5, cosine schedule, 2 epochs (epoch 1 checkpoint is optimal)
- Tracked in [Weights & Biases](https://wandb.ai/university-of-zurich/md-reheader)

## Limitations

- **Deep nesting (H5/H6)**: Accuracy drops to 45–50%. The model preserves relative structure but tends to compress deep hierarchies.
- **Irregular document starts**: Documents beginning with metadata or non-standard headings can cause cascading errors.
- **Ambiguous structure**: Heading levels are inherently subjective — "Background" could be H1 or H2 depending on context. The model learns common conventions but cannot resolve genuine ambiguity.
- **Long documents**: Documents exceeding 8k tokens (after stripping) are truncated, so headings beyond the cutoff use fallback levels.

## Development

```bash
git clone https://github.com/joelbarmettlerUZH/md-reheader.git
cd md-reheader

# Install with dev dependencies
uv sync --extra dev

# Run tests
make test

# Run linter
make lint
```

### Reproducing training

```bash
# Install training dependencies
uv sync --extra train

# Download raw data (~150k documents)
make download

# Prepare training data (strip, flatten, oversample)
make prepare

# Train (requires 2x GPU)
make train
```

### Evaluation

```bash
# Run model evaluation on test set
make eval

# Run baseline comparisons
make baselines
```

## Project Structure

```
md-reheader/
├── src/md_reheader/
│   ├── models.py          # Pydantic data models
│   ├── data/
│   │   ├── extract.py     # Heading extraction (markdown-it-py)
│   │   ├── strip.py       # Body stripping + heading flattening
│   │   ├── format.py      # ChatML formatting
│   │   ├── apply.py       # Apply predicted levels to markdown
│   │   ├── filter.py      # Quality filters
│   │   └── corrupt.py     # Corruption strategies (historical)
│   ├── eval/
│   │   ├── metrics.py     # Evaluation metrics
│   │   ├── evaluate.py    # Evaluation pipeline with slicing
│   │   ├── baselines.py   # Heuristic baselines
│   │   └── analysis.py    # Error categorization
│   └── inference/
│       └── predict.py     # Inference pipeline
├── scripts/               # Data download, preparation, evaluation
├── configs/training/      # Axolotl YAML configs
└── tests/                 # 91 tests
```

## License

Code and model weights: [Apache 2.0](LICENSE)

Training data includes content from Wikipedia ([CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)) and GitHub repositories (various open-source licenses).

## Citation

```bibtex
@software{barmettler2026mdreheader,
  author = {Barmettler, Joel},
  title = {md-reheader: Restoring Heading Hierarchy in Markdown Documents},
  year = {2026},
  url = {https://github.com/joelbarmettlerUZH/md-reheader}
}
```
