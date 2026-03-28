from pathlib import Path

from md_reheader.eval.metrics import compute_all_metrics
from md_reheader.models import EvalResult


def run_full_evaluation(
    predictions: list[list[int]],
    ground_truths: list[list[int]],
    metadata_list: list[dict],
    output_dir: Path | None = None,
) -> dict[str, list[EvalResult]]:
    results: list[EvalResult] = []
    for pred, truth, meta in zip(predictions, ground_truths, metadata_list):
        result = compute_all_metrics(pred, truth)
        result.metadata = meta
        results.append(result)

    slices = {"overall": results}

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    return slices
