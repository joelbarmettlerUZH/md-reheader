# VRAM Profile — Qwen3-0.6B

GPU: NVIDIA GeForce RTX 4090, 24 GB VRAM
Model: Qwen/Qwen3-0.6B (text-only, 0.6B params)
Settings: BF16, gradient checkpointing enabled, single GPU

## Results

| seq_len | bs | peak VRAM (MB) | fits 24GB? |
|---------|----|----------------|------------|
| 512     | 16 | 18,276         | Y          |
| 512     | 32 | OOM            | N          |
| 1024    | 8  | 18,276         | Y          |
| 1024    | 16 | OOM            | N          |
| 2048    | 1  | 5,433          | Y          |
| 2048    | 2  | 9,713          | Y          |
| 2048    | 4  | 18,277         | Y          |
| 2048    | 8  | OOM            | N          |
| 4096    | 1  | 9,714          | Y          |
| 4096    | 2  | 18,278         | Y          |
| 4096    | 4  | OOM            | N          |
| 8192    | 1  | 18,280         | Y          |
| 8192    | 2  | OOM            | N          |
| 16384   | 1  | OOM            | N          |
| 32768   | 1  | OOM            | N          |

## Key Findings

- Model weights alone: ~1,137 MB in BF16
- Activation memory scales roughly linearly: ~2.1 GB/1k tokens at bs=1
- **Single GPU max: seq_len=8192, bs=1** (18.3 GB, 76% utilization)
- For 16k-32k sequences, **FSDP across 2 GPUs is required**
- FSDP shards optimizer states + gradients, freeing ~3-4 GB per GPU for activations

## Recommended Bucket Config

### Single GPU (debug/fast iteration)

| bucket     | max bs | VRAM   |
|------------|--------|--------|
| < 2048     | 4      | ~18 GB |
| 2048-4096  | 2      | ~18 GB |
| 4096-8192  | 1      | ~18 GB |

### 2-GPU FSDP (full training)

| bucket       | max bs/GPU | estimated VRAM/GPU |
|--------------|------------|-------------------|
| < 2048       | 4          | ~10 GB            |
| 2048-4096    | 2          | ~10 GB            |
| 4096-8192    | 1          | ~10 GB            |
| 8192-16384   | 1          | ~18 GB (est.)     |
| 16384-32768  | 1          | ~22 GB (est.)     |

Note: FSDP estimates assume ~50% reduction in fixed memory (weights + optimizer + gradients)
which frees headroom for activations. Needs empirical validation with actual FSDP run.

## Note on Model Choice

The CLAUDE.md references Qwen3.5-0.8B, but this is a VLM (vision-language model) with
significantly higher memory usage. Qwen3-0.6B is the text-only equivalent and is what
we use for training.
