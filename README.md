# md-reheader

**Restore heading hierarchy in markdown documents** using a fine-tuned 0.6B parameter LLM.

PDF-to-markdown tools like [MinerU](https://github.com/opendatalab/MinerU), [Docling](https://github.com/DS4SD/docling), and [Marker](https://github.com/VikParuchuri/marker) often flatten heading structure. You get this:

```markdown
# API Reference
# Authentication
# Endpoints
# Users
# List Users
# Get User by ID
# Projects
# List Projects
# Error Handling
```

md-reheader restores the correct hierarchy:

```markdown
# API Reference
## Authentication
## Endpoints
### Users
#### List Users
#### Get User by ID
### Projects
#### List Projects
## Error Handling
```

## Installation

```bash
pip install md-reheader
```

Requires Python 3.12+ and PyTorch. Works on both CPU and GPU.

## Quick Start

### Command line

```bash
# Auto-detects GPU if available, otherwise CPU
rehead --input flat.md --output fixed.md

# Force a device
rehead -i flat.md -o fixed.md --gpu
rehead -i flat.md -o fixed.md --cpu

# No --output: prints to stdout, so you can pipe
rehead -i flat.md | tee fixed.md
rehead -i flat.md > fixed.md

# Overwrite an existing output
rehead -i flat.md -o fixed.md --force

# Point at a local checkpoint or a different HF repo
rehead -i flat.md -o fixed.md --model ./checkpoints/epoch-1
```

Nested output directories are created automatically. Run `rehead --help` for the full flag list.

### From a file

```python
from pathlib import Path
from md_reheader.inference.predict import load_model, reheader_document

model, tokenizer = load_model("joelbarmettler/md-reheader")

markdown = Path("document.md").read_text()
fixed = reheader_document(markdown, model, tokenizer)
Path("document_fixed.md").write_text(fixed)
```

### From a string

```python
from md_reheader.inference.predict import load_model, reheader_document

model, tokenizer = load_model("joelbarmettler/md-reheader")

flat_markdown = """\
# Introduction
# Background
# Related Work
# Methods
# Data Collection
# Preprocessing
# Model Architecture
# Results
# Discussion
# Conclusion
"""

fixed = reheader_document(flat_markdown, model, tokenizer)
print(fixed)
# # Introduction
# ## Background
# ### Related Work
# ## Methods
# ### Data Collection
# ### Preprocessing
# ### Model Architecture
# ## Results
# ## Discussion
# ## Conclusion
```

### Post-processing MinerU / Docling output

```python
# After running MinerU or Docling on a PDF:
from md_reheader.inference.predict import load_model, reheader_document

model, tokenizer = load_model("joelbarmettler/md-reheader")

# MinerU outputs markdown with flat headings
mineru_output = open("output/paper.md").read()

# Fix the heading hierarchy
fixed = reheader_document(mineru_output, model, tokenizer)

with open("output/paper_fixed.md", "w") as f:
    f.write(fixed)
```

### GPU vs CPU

```python
# GPU (recommended for batch processing)
model, tokenizer = load_model("joelbarmettler/md-reheader", device="cuda")

# CPU (no GPU required, slower)
model, tokenizer = load_model("joelbarmettler/md-reheader", device="cpu")
```

## Speed

Benchmarked on a single NVIDIA RTX 4090 (BF16) and CPU (float32):

| Document size | GPU (RTX 4090) | CPU |
|--------------|----------------|-----|
| < 1k tokens | 0.4s | 5s |
| 1k-2k tokens | 0.8s | 10s |
| 2k-4k tokens | 1.4s | ~20s |
| 4k-8k tokens | 3.4s | ~60s |

The model processes documents up to 8k tokens (after preprocessing). Longer documents are automatically truncated.

## Evaluation

Evaluated on 7,321 test documents from GitHub markdown files and Wikipedia articles:

| Metric | All-H1 baseline | Heuristic | **md-reheader** |
|--------|-----------|-----------|-----------------|
| Exact match | 0.0% | 14.5% | **56.1%** |
| Per-heading accuracy | 13.1% | 49.1% | **80.6%** |
| Hierarchy preservation | 61.3% | 68.6% | **91.0%** |
| Mean absolute error | 1.38 | 0.62 | **0.22** |

### Per-level accuracy

| | H1 | H2 | H3 | H4 | H5 | H6 |
|---|---|---|---|---|---|---|
| **Accuracy** | 77% | 85% | 78% | 68% | 45% | 50% |

The model is strongest on H1-H3 headings (77-85% accuracy) and still significantly outperforms baselines on deeper levels. Most errors on H4-H6 are off-by-one — the relative structure is preserved even when the absolute level is shifted.

### By document depth

| Max heading depth | Exact match | Per-heading accuracy | Hierarchy |
|---|---|---|---|
| Depth 2 (flat) | 83% | 91% | 95% |
| Depth 3 | 54% | 82% | 90% |
| Depth 4 | 32% | 70% | 88% |
| Depth 5-6 | 33% | 65% | 89% |

### By source

| Source | Exact match | Per-heading accuracy |
|---|---|---|
| GitHub markdown | 49.5% | 74.0% |
| Wikipedia | 71.3% | 95.5% |

## How It Works

1. **Extract** headings from the document using [markdown-it-py](https://github.com/executablebooks/markdown-it-py) (CommonMark parser, correctly skips code blocks)
2. **Flatten** all headings to `# ` (level 1) — the model should not trust input heading levels
3. **Strip** body text to first 128 + last 128 tokens per section — preserves structural cues without bloating context
4. **Predict** heading levels using the fine-tuned Qwen3-0.6B model
5. **Apply** predicted levels back to the original document

The model outputs headings with correct `#` prefixes (e.g., `## Methods`, `### Data Collection`), leveraging pretraining knowledge about heading semantics.

## Limitations

- **Deep nesting (H5/H6)**: Accuracy drops to 45-50%. The model preserves relative structure but tends to compress deep hierarchies by 1-2 levels.
- **Ambiguous structure**: Heading levels are inherently subjective. The model learns common conventions but cannot resolve genuine ambiguity.
- **Long documents**: Documents exceeding ~8k tokens after stripping are truncated from the end. Headings beyond the cutoff retain their original levels.

## Training

The model is a fine-tuned [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) trained on ~197k markdown documents:
- **codeparrot/github-code**: ~105k markdown files from GitHub repositories
- **euirim/goodwiki**: ~45k Wikipedia articles
- Deep documents (depth 4+) oversampled 2-8x to address class imbalance

Trained with [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) on 2x RTX 4090 using DDP, BF16, 8k sequence length with sample packing.

### Reproducing

```bash
git clone https://github.com/joelbarmettlerUZH/md-reheader.git
cd md-reheader

uv sync --extra train    # install training dependencies
make download             # download raw data (~150k documents)
make prepare              # strip, flatten, oversample, format
make train                # train on 2x GPU
make eval                 # evaluate on test set
```

## License

Code and model weights: [Apache 2.0](LICENSE)

Training data includes Wikipedia content ([CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)) and GitHub repositories (various open-source licenses).

## Citation

```bibtex
@software{barmettler2026mdreheader,
  author = {Barmettler, Joel},
  title = {md-reheader: Restoring Heading Hierarchy in Markdown Documents},
  year = {2026},
  url = {https://github.com/joelbarmettlerUZH/md-reheader}
}
```
