# Heading Level Prediction LLM — Implementation Plan (v4)

## Project Priorities

This is a **learning and showcase project**. The priorities, in order:

1. **Reproducible, well-structured training pipeline** — clean code, config-driven, version-controlled
2. **Rigorous evaluation** — proper metrics, ablations, error analysis, comparison baselines
3. **Polished publication** — HuggingFace model card, dataset card, weights, blog post
4. **Data collection** — important but not the showcase; keep it simple and sufficient

---

## Context Window Strategy

Heading hierarchy is a global document property — a heading's correct level depends on
its relationship to every other heading in the document. Chunking destroys this signal.
This project uses the **full context window** of Qwen3-0.6B (40k native, training up to 32k)
with a mixed-length training strategy that handles documents at their natural lengths.

### Why Long Context Works for This Task

The model reads a long input but produces a relatively short output (the headings with
correct `#` prefixes). It doesn't need to generate coherent long-form text — it needs
to attend over headings and infer structural patterns. The attention patterns are
regular (heading-to-heading relationships, section length ratios, nesting cues), which
is easier than arbitrary long-range dependencies that typically degrade small models
at long context.

### Output Format: Markdown Headings (not integers)

The model outputs newline-separated markdown headings with correct `#` prefixes:

```
# Introduction
## Background
## Methods
### Data Collection
## Results
```

This aligns with the model's pretraining distribution — it has seen millions of
correctly-structured markdown documents, so predicting `## Methods` after
`# Introduction` leverages existing knowledge. A comma-separated integer format
(e.g., `1, 2, 2, 3, 2`) would require learning an arbitrary mapping from scratch.
The heading text also carries signal — "Introduction" is typically H1/H2, while
"2.1.3 Implementation Details" implies deep nesting.

### Model Choice: Qwen3-0.6B (text-only)

**Important:** Qwen3.5-0.8B is a vision-language model (VLM) with a vision encoder that
adds significant memory overhead. We use **Qwen/Qwen3-0.6B** instead — a pure text-only
causal LM with 40k native context window, 28 layers, 0.6B parameters.

### VRAM Profile (Empirically Measured, 2x RTX 4090, 24 GB each)

Single-GPU measurements with BF16 + gradient checkpointing:

```
seq_len    bs    peak VRAM    fits 24GB?
────────────────────────────────────────
   512     16    18,276 MB    Y
  1024      8    18,276 MB    Y
  2048      1     5,433 MB    Y
  2048      2     9,713 MB    Y
  2048      4    18,277 MB    Y
  4096      1     9,714 MB    Y
  4096      2    18,278 MB    Y
  8192      1    18,280 MB    Y
 16384      1    OOM          N
 32768      1    OOM          N
```

Model weights alone: ~1,137 MB in BF16. Activations scale ~2.1 GB per 1k tokens at bs=1.
**Single GPU max: seq_len=8192, bs=1.** For 16k-32k, FSDP across 2 GPUs is required.

---

## Repository Structure

```
md-reheader/
├── README.md
├── LICENSE
├── pyproject.toml               # uv, Python 3.13
├── Makefile
│
├── configs/
│   ├── data/
│   │   ├── github_code.yaml     # codeparrot/github-code config
│   │   └── goodwiki.yaml        # euirim/goodwiki config
│   ├── training/
│   │   ├── full_ft_2gpu.yaml
│   │   └── full_ft_1gpu.yaml
│   └── eval/
│       └── default.yaml
│
├── src/
│   └── md_reheader/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── extract.py       # Heading extraction from markdown
│       │   ├── filter.py        # Quality filters
│       │   ├── corrupt.py       # Corruption strategies
│       │   ├── format.py        # ChatML formatting
│       │   └── batching.py      # Length-bucketed sampler & collation
│       ├── training/
│       │   ├── __init__.py
│       │   └── train.py         # Training entrypoint
│       ├── eval/
│       │   ├── __init__.py
│       │   ├── metrics.py       # All metric functions
│       │   ├── evaluate.py      # Evaluation loop
│       │   ├── baselines.py     # Heuristic & zero-shot baselines
│       │   └── analysis.py      # Error analysis & visualization
│       └── inference/
│           ├── __init__.py
│           └── predict.py       # Inference pipeline
│
├── scripts/
│   ├── download_data.py         # Fetch from HuggingFace datasets
│   ├── prepare_dataset.py       # Corrupt, format, split, save processed JSONL
│   ├── profile_vram.py          # Measure actual VRAM at each sequence length
│   ├── run_training.py          # Launch training with config
│   ├── run_eval.py              # Run evaluation suite
│   ├── publish_model.py         # Push to HuggingFace Hub
│   └── run_baselines.py         # Run all baseline comparisons
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_training_analysis.ipynb
│   └── 03_error_analysis.ipynb
│
├── tests/
│   ├── test_extract.py
│   ├── test_corrupt.py
│   ├── test_metrics.py
│   ├── test_format.py
│   └── test_batching.py
│
└── docs/
    ├── model_card.md
    ├── dataset_card.md
    └── vram_profile.md          # Empirical VRAM measurements
```

---

## Phase 1: Project Bootstrap (DONE)

Completed. Python 3.13, uv, all dependencies at latest versions (torch 2.11,
transformers 5.4, etc.). Hydra configs, pre-commit, ruff, 51 tests passing,
VRAM profiled, git initialized.

---

## Phase 2: Data Collection (DONE)

Two-step pipeline: `download_data.py` saves clean originals, `prepare_dataset.py`
corrupts + formats them. This lets you change corruption strategy without re-downloading.

### Data Sources

**Primary: `codeparrot/github-code`** (Markdown subset)
- 8.5M markdown files, 23 GB, no authentication required
- Loaded via raw parquet files (`datasets>=4.0` dropped loading script support)
- Filtered by `.md`/`.markdown` file extension on the `path` column
- Has `repo_name` for splitting by repo (prevents data leakage)
- Stratified sampling: 70% priority paths (`docs/`, `wiki/`, `guide/`), 30% READMEs
- Second pass for long docs (25k+ chars) from remaining shards to boost 8k-32k bucket

**Supplement: `euirim/goodwiki`** (curated Wikipedia)
- 44.8k high-quality Wikipedia articles in GitHub-flavored Markdown
- Article title prepended as `# {title}` — Wikipedia's title IS the H1
- Different domain from GitHub code docs — improves generalization

**Dropped alternatives:**
- `bigcode/the-stack-v2-dedup` — content not in parquet rows, requires S3 download + auth
- `bigcode/starcoderdata` — gated, no advantage over github-code
- `marcodsn/arxiv-markdown` — generated by docling which flattens headings, no ground truth

### Pipeline

1. **Download** (`scripts/download_data.py`): streams from HuggingFace, applies cheap
   char-length + heading count filters, saves raw JSONL with metadata
2. **Prepare** (`scripts/prepare_dataset.py`): token-count filter, split by repo/title
   (prevents leakage), corrupt headings (mixed strategy), format as ChatML, save JSONL

### Dataset Stats (Actual)

| Split | Examples | Tokens (content) |
|-------|---------|-------------------|
| train | ~130k   | ~322M             |
| val   | ~7k     | ~18M              |
| test  | ~7k     | ~18M              |

Sources: ~100k github-code (70k priority + 30k READMEs) + ~20k long github-code
(25k+ chars) + ~45k goodwiki. Token filter drops <1%.

### Dataset Versioning

Push processed dataset to HuggingFace Hub for reproducibility:

```python
ds = DatasetDict({"train": train, "validation": val, "test": test})
ds.push_to_hub("your-username/markdown-heading-levels", private=False)
```

---

## Phase 3: Training with Axolotl

### Why Axolotl

Evaluated Axolotl, TRL SFTTrainer, torchtune, LLaMA-Factory, Unsloth, Nanotron, and
TorchTitan. Decision factors:

- **No custom training code.** Dynamic padding, sample packing, gradient accumulation,
  FSDP, gradient checkpointing, BF16, and W&B are all commodity features. We should not
  implement any of them. Axolotl provides all of these via YAML config.
- **Sample packing replaces length-bucketed batching.** Axolotl's multipack bin-packing
  concatenates short sequences into full-length training sequences, achieving the same
  efficiency as our custom `LengthBucketSampler` without any custom code.
- **YAML-driven reproducibility.** One config file fully defines a training run. Easy to
  version-control, diff, and reproduce. Ablations are trivial CLI overrides.
- **FSDP + sequence parallelism.** For 16k-32k sequences that OOM on a single GPU,
  Axolotl supports FSDP1/FSDP2 across 2 GPUs and Ring FlashAttention for sequence
  parallelism.
- **Qwen3 support.** Explicit support since Oct 2025, documented in Qwen's own training
  guide.

Dropped alternatives:
- TRL SFTTrainer — strong, but Python-configured (not YAML-first), no length bucketing
- torchtune — open Qwen3 EOS masking issue, smaller community
- Unsloth — multi-GPU is a known weakness, dealbreaker for 16k-32k sequences
- LLaMA-Factory — slowing development, documentation issues
- Nanotron, TorchTitan — pretraining only, no SFT support

### 3.1 — Data Format for Axolotl

Axolotl expects conversations in `sharegpt` format. Our processed JSONL already has
the right structure (`messages` with `role`/`content`). We configure Axolotl to read
this directly:

```yaml
datasets:
  - path: ./data/processed/train.jsonl
    type: sharegpt
    conversation: chatml
```

### 3.2 — Base Training Config

```yaml
# configs/training/axolotl_base.yaml
base_model: Qwen/Qwen3-0.6B
model_type: AutoModelForCausalLM

load_in_8bit: false
load_in_4bit: false
bf16: auto

datasets:
  - path: ./data/processed/train.jsonl
    type: sharegpt
    conversation: chatml

val_set_size: 0  # we have a separate val set
dataset_prepared_path: ./data/axolotl_prepared

sequence_len: 32768
sample_packing: true
pad_to_sequence_len: true

gradient_accumulation_steps: 16
micro_batch_size: 1
num_epochs: 3
learning_rate: 2e-5
lr_scheduler: cosine
warmup_steps: 200
weight_decay: 0.01
optimizer: adamw_torch

gradient_checkpointing: true

logging_steps: 10
eval_steps: 500
save_steps: 1000
save_total_limit: 3
output_dir: ./checkpoints

wandb_project: md-reheader
wandb_run_name: null

seed: 42
```

### 3.3 — Multi-GPU Config (FSDP)

For 2x RTX 4090 with FSDP to handle 16k-32k sequences:

```yaml
# configs/training/axolotl_2gpu.yaml (extends base)
fsdp:
  - full_shard
  - auto_wrap
fsdp_config:
  fsdp_limit_all_gathers: true
  fsdp_sync_module_states: true
  fsdp_offload_params: false
  fsdp_state_dict_type: FULL_STATE_DICT
  fsdp_transformer_layer_cls_to_wrap: Qwen3DecoderLayer
```

Launch: `accelerate launch --num_processes 2 -m axolotl.cli.train configs/training/axolotl_2gpu.yaml`

### 3.4 — Experiment Tracking

Every run tracked in W&B. Axolotl logs loss, learning rate, grad norm, throughput,
and GPU memory automatically. Tag runs for easy filtering:

```yaml
wandb_project: md-reheader
wandb_run_name: full-ft-32k-lr2e5-ep3
```

### 3.5 — Ablation Study Plan

Ablations are CLI overrides on the base config — no code changes needed:

| Experiment ID   | Variable            | Values to Test                | CLI Override Example                     |
|-----------------|---------------------|-------------------------------|------------------------------------------|
| `abl-lr`        | Learning rate       | 1e-5, 2e-5, 5e-5             | `--learning_rate 5e-5`                   |
| `abl-epoch`     | Epochs              | 1, 3, 5                      | `--num_epochs 5`                         |
| `abl-seqlen`    | Max sequence length | 4096, 8192, 16384, 32768     | `--sequence_len 4096`                    |
| `abl-corrupt`   | Corruption mix      | flat-only, mixed, random-only | Re-run prepare_dataset.py, point to data |
| `abl-data`      | Dataset size        | 20k, 50k, 80k, 130k          | Subsample train.jsonl                    |
| `abl-lora`      | LoRA vs full FT     | LoRA r=32, full FT            | Add `adapter: lora` + `lora_r: 32`      |

The `abl-seqlen` ablation validates the full-context architecture decision.
The `abl-data` ablation determines if we need to scale beyond 80k examples.

Run each with 3 seeds (`--seed 42/123/456`) for variance estimates.

---

## Phase 4: Evaluation (The Most Important Phase)

### 4.1 — Metric Suite

Already implemented in `src/md_reheader/eval/metrics.py`:
- Exact match
- Per-heading accuracy
- Hierarchy preservation (pairwise directional accuracy)
- Mean absolute error
- Level count match

### 4.2 — Baselines

- **Naive flat:** All level 1. What the corrupted input already has.
- **Heuristic rules:** First heading = L1, numbered headings infer depth from dots,
  common top-level patterns = L1. Already implemented.
- **Zero-shot Qwen3-0.6B:** Same prompt, no fine-tuning.
- **Zero-shot larger model (Qwen3-1.7B or API model):** Sets the ceiling.

### 4.3 — Evaluation Slicing

Document length is the **primary evaluation axis**. Also slice by source (github_code
vs goodwiki), heading count, max depth, and corruption type.

The **accuracy vs document length chart** comparing fine-tuned model, 4k-training
ablation, zero-shot larger model, and heuristic baseline is the centerpiece.

### 4.4 — Error Analysis

Already implemented in `src/md_reheader/eval/analysis.py`:
- Count mismatch, off-by-one (constant offset), flat prediction
- Inverted hierarchy, "lost in middle" detection
- Confusion matrix, calibration plot, positional accuracy plot

---

## Phase 5: Publishing

### 5.1 — HuggingFace Model Card

Include: model description, training details, evaluation tables (overall + by length +
by source), baseline comparison, ablation results, usage code, limitations.

### 5.2 — Blog Post Structure

1. The problem — visual before/after of flat vs hierarchical headings
2. Why a small fine-tuned model — cost, latency, beating larger models on narrow task
3. Why full context matters — ablation chart
4. Data strategy — two sources, corruption approach, length distribution
5. Training details — W&B dashboard, loss curve, VRAM profile
6. Results — comparison table, accuracy-vs-length chart
7. Error analysis — confusion matrix, "lost in middle", example failures
8. Try it yourself — HuggingFace model, Gradio demo, code snippet

---

## Phase 6: Stretch Goals

- HuggingFace Space with Gradio demo
- GGUF export for local inference
- Benchmarking as post-processor on Marker, Nougat, and other PDF-to-MD tools

---

## Key Principles Throughout

- **Every experiment is tracked in W&B.** No "quick test" runs that go unlogged.
- **Every decision is documented.** Why Qwen3-0.6B not Qwen3.5? Why github-code not Stack v2? Put these in the README or a decisions log.
- **Reproducibility is sacred.** Pin all dependencies, log seeds, commit configs.
- **Tests protect your pipeline.** Corruption logic will change multiple times. Tests catch regressions.
- **The write-up tells a story.** "PDF parsers lose heading structure. I fixed it with a tiny model that runs anywhere — and proved that full-document context is essential."
