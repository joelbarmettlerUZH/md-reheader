---
license: apache-2.0
language:
  - en
tags:
  - markdown
  - document-structure
  - heading-prediction
  - pdf-to-markdown
  - fine-tuned
base_model: Qwen/Qwen3-0.6B
datasets:
  - joelbarmettlerUZH/md-reheader-dataset
metrics:
  - accuracy
  - exact_match
pipeline_tag: text-generation
library_name: transformers
---

# md-reheader

**Restore heading hierarchy in markdown documents** using a fine-tuned Qwen3-0.6B model (0.6B parameters).

PDF-to-markdown tools like [MinerU](https://github.com/opendatalab/MinerU), [Docling](https://github.com/DS4SD/docling), and [Marker](https://github.com/VikParuchuri/marker) often flatten heading structure — everything becomes `#` or `##`. This model predicts the correct heading levels (H1–H6) from document context and semantics.

## Usage

### With the md-reheader package (recommended)

```bash
pip install md-reheader
```

```python
from md_reheader.inference.predict import load_model, reheader_document

model, tokenizer = load_model("joelbarmettlerUZH/md-reheader")

flat_markdown = open("document.md").read()
fixed = reheader_document(flat_markdown, model, tokenizer)
```

The package handles preprocessing (heading flattening, body stripping) and postprocessing (applying predicted levels back to the original document) automatically.

### Direct usage with transformers

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("joelbarmettlerUZH/md-reheader")
model = AutoModelForCausalLM.from_pretrained(
    "joelbarmettlerUZH/md-reheader",
    dtype=torch.bfloat16,
    device_map="auto",
)

messages = [
    {"role": "system", "content": "You are a markdown document structure expert. Given a markdown document with incorrect or flattened heading levels, output each heading with its correct markdown prefix (# for level 1, ## for level 2, etc.), one per line."},
    {"role": "user", "content": "# Introduction\n\nSome text...\n\n# Background\n\nMore text...\n\n# Methods"},
]

input_text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
)
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=4096, do_sample=False)

generated = outputs[0][inputs["input_ids"].shape[1]:]
print(tokenizer.decode(generated, skip_special_tokens=True))
# # Introduction
# ## Background
# ## Methods
```

**Important:** You must pass `enable_thinking=False` to `apply_chat_template` to match the training format. Without this, the model will not generate correctly.

## Evaluation Results

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

### By document depth

| Max depth | Exact match | Accuracy | Hierarchy |
|---|---|---|---|
| Depth 2 | 83% | 91% | 95% |
| Depth 3 | 54% | 82% | 90% |
| Depth 4 | 32% | 70% | 88% |
| Depth 5-6 | 33% | 65% | 89% |

## Training

- **Base model:** [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) (text-only, 0.6B parameters)
- **Framework:** [Axolotl](https://github.com/axolotl-ai-cloud/axolotl)
- **Data:** ~197k markdown documents from GitHub and Wikipedia (with deep-doc oversampling)
- **Hardware:** 2x NVIDIA RTX 4090, DDP, BF16
- **Sequence length:** 8,192 tokens with sample packing
- **Learning rate:** 5e-5 with cosine schedule
- **Epochs:** 2 (epoch 1 checkpoint used — epoch 2 overfits)
- **Effective batch size:** 24

### Input format

During training, documents are preprocessed:
1. All headings flattened to `# ` (level 1)
2. Body text between headings truncated to first 128 + last 128 tokens
3. Total document truncated to fit within 8k tokens

The model then outputs each heading with the correct `#` prefix and heading text.

## Speed

| Document size | GPU (RTX 4090) | CPU |
|---|---|---|
| < 1k tokens | 0.4s | 5s |
| 1k-2k tokens | 0.8s | 10s |
| 2k-4k tokens | 1.4s | ~20s |
| 4k-8k tokens | 3.4s | ~60s |

## Limitations

- **Deep nesting (H5/H6):** Accuracy drops to 45-50%. The model preserves relative structure but compresses deep hierarchies.
- **Ambiguous structure:** Heading levels are inherently subjective. The model learns conventions but cannot resolve genuine ambiguity.
- **Long documents:** Documents exceeding ~8k tokens after preprocessing are truncated.
- **English-centric:** Trained primarily on English documents.

## Citation

```bibtex
@software{barmettler2026mdreheader,
  author = {Barmettler, Joel},
  title = {md-reheader: Restoring Heading Hierarchy in Markdown Documents},
  year = {2026},
  url = {https://github.com/joelbarmettlerUZH/md-reheader}
}
```
