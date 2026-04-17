# md-reheader — Implementation Plan (v5)

## Project Priorities

This is a **learning and showcase project**. The priorities, in order:

1. **Reproducible, well-structured training pipeline** — clean code, config-driven, version-controlled
2. **Rigorous evaluation** — proper metrics, error analysis, comparison baselines
3. **Polished publication** — HuggingFace model card, dataset card, weights, PyPI package
4. **Data collection** — important but not the showcase; keep it simple and sufficient

---

## Code Style

- **Pydantic models** with `Annotated[type, Field()]` for all structured data
- **No file-level docstrings** — the module name and structure should be self-explanatory
- **Comments explain WHY, not WHAT** — no `# extract headings from document` style comments
- **Strict typing** — all function signatures fully typed with `Annotated` for parameters
- **Fail fast** — raise errors early, don't silently return defaults
- **No unnecessary abstractions** — three similar lines > premature helper function
- **ruff** for linting and formatting (line-length 99, py312, select E/F/I/N/W/UP)
- **uv** for dependency management, `pyproject.toml` as single source of truth

---

## Task & Model Design

### The Problem

PDF-to-markdown tools (Marker, Nougat, Docling, MinerU) flatten heading hierarchy — typically
everything becomes H1 or H2. This destroys document structure. A small fine-tuned model can
restore correct heading levels by understanding document semantics and structure.

### Model Choice: Qwen3-0.6B (text-only)

**Important:** Qwen3.5-0.8B is a vision-language model (VLM) — do NOT use it.
We use **Qwen/Qwen3-0.6B** — a pure text-only causal LM, 40k native context, 0.6B parameters.

### Input/Output Format (V3 — Current)

**Input:** Document with all headings flattened to `# ` (level 1), body text stripped
to first 128 + last 128 tokens per section, truncated to fit 8k tokens.

**Output:** Headings with correct `#` prefixes and text, one per line.

```
Input (user message):
# Introduction
First 128 tokens of body... [...] ...last 128 tokens.
# Background
Body text...
# Methods
Body text...

Output (assistant message):
# Introduction
## Background
### Methods
```

The system prompt is: "You are a markdown document structure expert. Given a markdown
document with incorrect or flattened heading levels, output each heading with its correct
markdown prefix (# for level 1, ## for level 2, etc.), one per line."

### Why This Format (Lessons from V1 and V2)

**V1** used full documents (no stripping) with random corruption. Loss was dominated by
heading text reproduction (~98% of output tokens), masking poor level prediction. Trained
at 32k seq_len with FSDP — slow (days) and the model learned to copy text but defaulted
to H2 for levels.

**V2** used marker tokens (`<|object_ref_start|>`) and numeric-only output (`1\n2\n3\n`).
Removing heading text from the output removed too much signal — the model couldn't leverage
pretraining knowledge about heading semantics ("Introduction" = likely H1). Performance
dropped across the board.

**V3** (current) takes the best of both: stripped body + flattened `#` headings from V2's
efficiency gains, but V1's heading-text-in-output to leverage pretraining semantics. The
`#` heading prefix in the input carries meaning the model already understands from
pretraining, unlike arbitrary marker tokens.

### Inference: enable_thinking=False

The Qwen3 chat template has an `enable_thinking` flag. During training, assistant messages
get wrapped in empty `<think>\n\n</think>\n\n` tags. At inference, `enable_thinking=False`
must be passed to `apply_chat_template` to match the training format. Without this, the
model enters a repetition loop.

---

## Data

### Sources

**Primary: `codeparrot/github-code`** (Markdown subset)
- 8.5M markdown files, loaded via raw parquet files
- Filtered by `.md`/`.markdown` file extension on the `path` column
- Bucket-targeted download: fills per-length-bucket targets (< 4k, 4k-8k, 8k-16k, 16k-32k)
- Split by `repo_name` to prevent data leakage

**Supplement: `euirim/goodwiki`** (curated Wikipedia)
- 44.8k high-quality Wikipedia articles in GitHub-flavored Markdown
- Article title prepended as `# {title}`

### Pipeline

1. **Download** (`scripts/download_data.py`): streams from HuggingFace, applies cheap
   filters, saves raw JSONL. Headings extracted via `markdown-it-py`.
2. **Prepare** (`scripts/prepare_dataset.py --version 3`): re-extracts headings, applies
   token-count filter, splits by repo/title, oversamples deep docs (depth 4: 2x, 5: 4x,
   6: 8x), strips body text, flattens headings, truncates to 7500 tokens, formats as
   ChatML, saves JSONL.

### Dataset Stats

| Split | Examples | Notes |
|-------|---------|-------|
| train | ~197k | After oversampling (131k original) |
| val | ~7k | |
| test | ~7k | |

Sources: ~105k github-code + ~45k goodwiki. Oversampling increases depth 4+ from 30% to 53%.

---

## Training with Axolotl

### Why Axolotl

No custom training code. Sample packing, gradient accumulation, DDP, gradient checkpointing,
BF16, W&B — all via YAML config. YAML-driven reproducibility.

### Current Config (V3)

- `sequence_len: 8192` — stripped docs fit within this
- `micro_batch_size: 12`, `gradient_accumulation_steps: 1` (effective batch 24)
- `learning_rate: 5e-5` (higher than default — small output fraction needs stronger signal)
- `num_epochs: 2` (but epoch 1 checkpoint is optimal — epoch 2 overfits)
- DDP across 2x RTX 4090 (no FSDP — 0.6B model fits on single GPU)
- `chat_template: qwen3`, `roles_to_train: assistant`
- `cut_cross_entropy` plugin for memory efficiency

### Training Lessons

- **FSDP is overkill for 0.6B** — DDP is faster, less communication overhead
- **The model overfits in epoch 2** — use the epoch-1 checkpoint (best eval loss)
- **Assistant tokens are ~2% of total** — the loss signal is sparse, so higher LR helps
- **Eval loss ≠ task performance** — V1 had lower eval loss (0.035) than V3 (0.028) but
  V3 has better heading-level accuracy because V1's loss was dominated by text reproduction

---

## Evaluation Results (V3, Best Checkpoint)

### Overall

| Metric | Naive Flat | Heuristic | **V3 Model** |
|--------|-----------|-----------|-------------|
| Exact match | 0.000 | 0.145 | **0.561** |
| Per-heading accuracy | 0.131 | 0.491 | **0.806** |
| Hierarchy preservation | 0.613 | 0.686 | **0.910** |
| MAE | 1.382 | 0.624 | **0.220** |
| Level count match | 1.000 | 1.000 | **0.987** |

### Per-Level Accuracy

| Level | V1 | V3 |
|-------|-----|-----|
| H1 | 0.72 | **0.77** |
| H2 | 0.83 | **0.85** |
| H3 | 0.73 | **0.78** |
| H4 | 0.56 | **0.68** |
| H5 | 0.42 | **0.45** |
| H6 | 0.34 | **0.50** |

### Error Analysis

- **56.1% exact match** (every heading correct)
- **1.3% count mismatch** (vs 8.9% in V1)
- **1.1% inverted hierarchy**
- Most errors are off-by-one at deep levels — reasonable disagreements

### Known Failure Modes

1. **Consistent off-by-one at depth** — model gets relative structure right (86% hierarchy)
   but absolute level shifted by 1. Common with ambiguous nesting.
2. **Cascade from bad start** — if the document starts with irregular structure (e.g.
   Jekyll frontmatter), the model gets confused and everything shifts.
3. **Compression of deep subtrees** — H5/H6 get compressed to H3/H4. The model preserves
   relative structure but underestimates absolute depth.

---

## Repository Structure

```
md-reheader/
├── pyproject.toml               # uv, Python 3.12, PyPI metadata
├── Makefile
├── CLAUDE.md
│
├── configs/
│   └── training/
│       ├── axolotl_v3_2gpu.yaml # Current best config
│       ├── axolotl_2gpu.yaml    # V1 config (historical)
│       ├── axolotl_1gpu.yaml    # V1 single-GPU
│       ├── axolotl_v2_2gpu.yaml # V2 config (historical)
│       └── axolotl_v2_1gpu.yaml # V2 single-GPU
│
├── src/
│   └── md_reheader/
│       ├── __init__.py
│       ├── models.py            # Pydantic models
│       ├── data/
│       │   ├── __init__.py
│       │   ├── extract.py       # Heading extraction via markdown-it-py
│       │   ├── filter.py        # Quality filters
│       │   ├── format.py        # ChatML formatting
│       │   ├── strip.py         # Body stripping + heading flattening (v3)
│       │   └── apply.py         # Apply predicted levels to markdown
│       ├── eval/
│       │   ├── __init__.py
│       │   ├── metrics.py       # Metric functions
│       │   ├── evaluate.py      # Evaluation with slicing
│       │   ├── baselines.py     # Heuristic baselines
│       │   └── analysis.py      # Error categorization
│       └── inference/
│           ├── __init__.py
│           └── predict.py       # Inference pipeline
│
├── scripts/
│   ├── download_data.py
│   ├── prepare_dataset.py
│   ├── run_eval.py
│   ├── run_baselines.py
│   ├── profile_vram.py
│   └── publish_model.py
│
└── tests/
    ├── test_extract.py
    ├── test_metrics.py
    ├── test_format.py
    ├── test_filter.py
    ├── test_prepare.py
    ├── test_strip.py
    ├── test_apply.py
    └── test_batching.py
```

---

## Phases

### Phase 1: Project Bootstrap (DONE)
Python 3.12, uv, Axolotl + flash-attn + cut-cross-entropy as managed deps.

### Phase 2: Data Collection (DONE)
Two-step pipeline: download raw → prepare processed. Bucket-targeted download for
length diversity. 150k raw docs, 197k after oversampling.

### Phase 3: Training (DONE)
Three iterations (V1→V2→V3). V3 is the best: flattened input, stripped body, heading
text in output, oversampled deep docs, higher LR. 2 epochs, epoch-1 checkpoint optimal.

### Phase 4: Evaluation (DONE)
Metrics, baselines, per-level accuracy, error analysis. V3 achieves 56% exact match,
81% per-heading accuracy, 91% hierarchy preservation.

### Phase 5: Publishing (IN PROGRESS)
- [ ] Clean up code for publication
- [ ] PyPI package with inference pipeline (`reheader_document`)
- [ ] HuggingFace model card + model upload
- [ ] HuggingFace dataset card + dataset upload

### Phase 6: Stretch Goals
- HuggingFace Space with Gradio demo
- GGUF export for local inference
- Benchmarking as post-processor for PDF-to-markdown tools

---

## Key Principles

- **Every experiment is tracked in W&B.**
- **Reproducibility is sacred.** Pin dependencies, log seeds, commit configs.
- **Tests protect the pipeline.** 107 tests covering extraction, stripping, formatting, metrics.
- **The write-up tells a story.** "PDF parsers lose heading structure. I fixed it with a tiny
  model that runs anywhere."
