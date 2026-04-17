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
  - joelbarmettler/md-reheader-dataset
metrics:
  - accuracy
  - exact_match
pipeline_tag: text-generation
library_name: transformers
---

<div align="center">

<h1 align="center" style="font-size: 32px">md-reheader</h1>

<p align="center"><strong>Restore heading hierarchy in markdown documents with a fine-tuned 0.6B LLM.</strong></p>

<p align="center">
  <a href="https://pypi.org/project/md-reheader/"><img src="https://img.shields.io/pypi/v/md-reheader?color=blue&label=PyPI" alt="PyPI"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.12%2B-blue" alt="Python 3.12+"></a>
  <a href="https://www.apache.org/licenses/LICENSE-2.0"><img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="Apache 2.0"></a>
  <a href="https://huggingface.co/joelbarmettler/md-reheader"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-yellow" alt="HuggingFace Model"></a>
  <a href="https://huggingface.co/datasets/joelbarmettler/md-reheader-dataset"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-Explore-yellow" alt="HuggingFace Dataset"></a>
  <a href="https://github.com/joelbarmettlerUZH/md-reheader"><img src="https://img.shields.io/github/stars/joelbarmettlerUZH/md-reheader?style=social" alt="GitHub stars"></a>
</p>

</div>

---

## The problem

PDF-to-markdown tools like [MinerU](https://github.com/opendatalab/MinerU), [Docling](https://github.com/DS4SD/docling), and [Marker](https://github.com/VikParuchuri/marker) do great text extraction — then collapse your document structure. Every heading becomes `#` or `##`. TOCs break. RAG chunking breaks. Navigation breaks.

**md-reheader** fixes it. A 0.6B-parameter Qwen3 fine-tune reads the document and predicts the correct H1–H6 level for every heading in a single forward pass.

<p align="center">
  <img src="https://raw.githubusercontent.com/joelbarmettlerUZH/md-reheader/main/docs/hero.png" alt="Source PDF → md-reheader → hierarchy restored vs. flat PDF-parser output" width="800">
</p>

---

## Quick start

### CLI

```bash
pip install md-reheader
rehead --input flat.md --output fixed.md
```

Auto-detects CUDA. Use `--cpu` or `--gpu` to override. Omit `--output` to stream to stdout (pipe-friendly).

```bash
rehead -i flat.md | tee fixed.md      # pipe
rehead -i flat.md --gpu -o out/fixed.md  # creates nested dirs
rehead --help                          # all flags
```

### Python API

```python
from md_reheader.inference.predict import load_model, reheader_document

model, tokenizer = load_model("joelbarmettler/md-reheader")

flat = open("document.md").read()
fixed = reheader_document(flat, model, tokenizer)
```

The package handles preprocessing (flattening + body stripping) and postprocessing (applying predicted levels back to the original document) automatically.

### Direct `transformers` usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("joelbarmettler/md-reheader")
model = AutoModelForCausalLM.from_pretrained(
    "joelbarmettler/md-reheader",
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

> **Important:** pass `enable_thinking=False` to `apply_chat_template`. Without it, the model enters a repetition loop because training used the non-thinking chat format.

---

### Self-host with vLLM

md-reheader exposes the standard OpenAI-compatible chat endpoint when served with [vLLM](https://github.com/vllm-project/vllm) — higher throughput than raw `transformers`, and drop-in client compatibility.

```bash
pip install vllm
vllm serve joelbarmettler/md-reheader --dtype bfloat16 --max-model-len 8192
```

On <10 GB cards (e.g. RTX 2000/3060), add `--enforce-eager --gpu-memory-utilization 0.70` to skip CUDA-graph allocations that otherwise OOM.

### Remote inference (vLLM or any OpenAI-compatible endpoint)

Once a server is running, use md-reheader as a thin client — no local weights needed.

**CLI:**

```bash
rehead -i flat.md -o fixed.md --endpoint http://localhost:8000/v1
# With auth:
rehead -i flat.md -o fixed.md --endpoint https://api.example.com/v1 --api-key sk-xxx
# or set MD_REHEADER_API_KEY in the environment
```

**Python:**

```python
from md_reheader.inference.remote import reheader_document_remote

fixed = reheader_document_remote(
    open("flat.md").read(),
    endpoint="http://localhost:8000/v1",
    model="joelbarmettler/md-reheader",
    api_key=None,  # or a bearer token
)
```

The remote client preprocesses locally (flatten + strip), sends a chat completion to the server with `chat_template_kwargs={"enable_thinking": false}` to match training, and applies predicted levels back to the original document. Identical output to local inference.

## How it works

```
flat markdown  ──►  flatten headings to #  ──►  strip body to 128+128 tokens
                                                         │
                                                         ▼
        restored markdown  ◄──  apply predicted levels  ◄── Qwen3-0.6B (fine-tuned)
```

1. Extract headings with [markdown-it-py](https://github.com/executablebooks/markdown-it-py) (correctly skips code blocks).
2. Flatten every heading to `# ` — the model ignores input levels.
3. Strip each section's body to its first 128 + last 128 tokens — preserves structural cues, kills the context bloat.
4. Qwen3-0.6B predicts the correct `#` prefix per heading.
5. Levels get mapped back to the original document.

---

## Evaluation

Benchmarked on 7,321 held-out documents from GitHub markdown and Wikipedia.

| Metric                  | All-H1 baseline | Heuristic | **md-reheader** |
|-------------------------|:---------------:|:---------:|:---------------:|
| Exact match             | 0.0%            | 14.5%     | **56.1%**       |
| Per-heading accuracy    | 13.1%           | 49.1%     | **80.6%**       |
| Hierarchy preservation  | 61.3%           | 68.6%     | **91.0%**       |
| Mean absolute error     | 1.38            | 0.62      | **0.22**        |

### Per-level accuracy

|          | H1  | H2  | H3  | H4  | H5  | H6  |
|----------|:---:|:---:|:---:|:---:|:---:|:---:|
| Accuracy | 77% | 85% | 78% | 68% | 45% | 50% |

H1–H3 land in the 77–85% band; H5/H6 drop but still beat baselines. Most deep-level errors are off-by-one — the relative structure survives.

### By document depth

| Max depth | Exact match | Per-heading accuracy | Hierarchy |
|-----------|:-----------:|:--------------------:|:---------:|
| Depth 2   | 83%         | 91%                  | 95%       |
| Depth 3   | 54%         | 82%                  | 90%       |
| Depth 4   | 32%         | 70%                  | 88%       |
| Depth 5-6 | 33%         | 65%                  | 89%       |

### By source

| Source            | Exact match | Per-heading accuracy |
|-------------------|:-----------:|:--------------------:|
| GitHub markdown   | 49.5%       | 74.0%                |
| Wikipedia         | 71.3%       | 95.5%                |

---

## Speed

| Document size | RTX 4090 (BF16) | CPU (fp32) |
|---------------|:---------------:|:----------:|
| < 1k tokens   | 0.4s            | 5s         |
| 1k–2k tokens  | 0.8s            | 10s        |
| 2k–4k tokens  | 1.4s            | ~20s       |
| 4k–8k tokens  | 3.4s            | ~60s       |

Documents longer than ~8k tokens (after stripping) are truncated from the tail.

---

## Training

| Item              | Value                                                      |
|-------------------|------------------------------------------------------------|
| Base model        | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) (text-only) |
| Framework         | [Axolotl](https://github.com/axolotl-ai-cloud/axolotl)     |
| Training data     | ~197k markdown docs (GitHub + Wikipedia, depth 4+ oversampled 2–8×) |
| Hardware          | 2× RTX 4090, DDP, BF16                                     |
| Sequence length   | 8,192 tokens with sample packing                           |
| Learning rate     | 5e-5, cosine schedule                                      |
| Epochs            | 2 (epoch-1 checkpoint — epoch 2 overfits)                  |
| Effective batch   | 24                                                         |

### Input format during training

1. All headings flattened to `# `.
2. Body text per section truncated to first 128 + last 128 tokens.
3. Document truncated to 8k tokens.
4. Assistant output: one heading per line with its correct `#` prefix.

---

## Limitations

- **Deep nesting (H5/H6):** accuracy drops to 45–50%. Relative structure is preserved but absolute depth gets compressed by 1–2 levels.
- **Ambiguous structure:** heading levels are subjective. The model learns common conventions; it can't resolve genuine ambiguity.
- **Long documents:** >8k tokens (after stripping) get truncated. Headings past the cutoff retain their input levels.
- **English-centric:** trained primarily on English content.

---

## Author

Built by [Joel Barmettler](https://joelbarmettler.xyz/).

## Citation

```bibtex
@software{barmettler2026mdreheader,
  author = {Barmettler, Joel},
  title  = {md-reheader: Restoring Heading Hierarchy in Markdown Documents},
  year   = {2026},
  url    = {https://github.com/joelbarmettlerUZH/md-reheader}
}
```
