import json
import subprocess
import sys

CONFIGS = [
    (512, 16),
    (512, 32),
    (1024, 16),
    (1024, 8),
    (2048, 1),
    (2048, 2),
    (2048, 4),
    (2048, 8),
    (4096, 1),
    (4096, 2),
    (4096, 4),
    (8192, 1),
    (8192, 2),
    (16384, 1),
    (32768, 1),
]

WORKER_SCRIPT = """
import sys, json
import torch
from transformers import AutoModelForCausalLM

seq_len = int(sys.argv[1])
batch_size = int(sys.argv[2])
model_name = sys.argv[3]

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

model = AutoModelForCausalLM.from_pretrained(
    model_name, dtype=torch.bfloat16, device_map="cuda:0"
)
model.gradient_checkpointing_enable()
model.train()

dummy_input = torch.randint(0, 1000, (batch_size, seq_len), device="cuda:0")
dummy_labels = dummy_input.clone()

output = model(dummy_input, labels=dummy_labels)
output.loss.backward()

peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
print(json.dumps({"peak_mb": peak_mb}))
"""

MODEL = "Qwen/Qwen3-0.6B"


def run_single_config(seq_len: int, batch_size: int) -> float | str:
    result = subprocess.run(
        [sys.executable, "-c", WORKER_SCRIPT, str(seq_len), str(batch_size), MODEL],
        capture_output=True,
        text=True,
        timeout=300,
    )

    if result.returncode != 0:
        stderr = result.stderr.strip()
        if "OutOfMemoryError" in stderr or "CUDA out of memory" in stderr:
            return "OOM"
        return f"ERROR: {stderr[-200:]}"

    for line in result.stdout.strip().splitlines():
        line_data = json.loads(line)
        return line_data["peak_mb"]

    return f"ERROR: no output, stderr={result.stderr[-200:]}"


def main() -> None:
    import torch

    if not torch.cuda.is_available():
        print("CUDA not available. This script requires a GPU.")
        return

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem: float = torch.cuda.get_device_properties(0).total_memory / 1024**2
    print(f"GPU: {gpu_name}, Total VRAM: {gpu_mem:,.0f} MB\n")

    results: dict[tuple[int, int], float | str] = {}
    for seq_len, bs in CONFIGS:
        print(f"Profiling seq_len={seq_len:>6}, bs={bs:>2} ... ", end="", flush=True)
        peak = run_single_config(seq_len, bs)
        results[(seq_len, bs)] = peak
        if isinstance(peak, float):
            fits = "Y" if peak < gpu_mem * 0.95 else "N"
            print(f"{peak:,.0f} MB [{fits}]")
        else:
            print(peak)

    print(f"\n{'=' * 60}")
    print(f"  VRAM Profile Summary — {gpu_name}")
    print(f"{'=' * 60}")
    print(f"  {'seq_len':>8}  {'bs':>4}  {'peak VRAM':>12}  {'fits?':>5}")
    print(f"  {'-' * 8}  {'-' * 4}  {'-' * 12}  {'-' * 5}")
    for (sl, bs), peak in sorted(results.items()):
        if isinstance(peak, float):
            fits = "Y" if peak < gpu_mem * 0.95 else "N"
            print(f"  {sl:>8}  {bs:>4}  {peak:>10,.0f} MB  {'  ' + fits}")
        else:
            print(f"  {sl:>8}  {bs:>4}  {'':>10}{peak:>5}")


if __name__ == "__main__":
    main()
