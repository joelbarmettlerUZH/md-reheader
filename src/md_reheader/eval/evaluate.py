import json
from pathlib import Path

import numpy as np

from md_reheader.eval.metrics import compute_all_metrics
from md_reheader.models import EvalResult

LENGTH_SLICES = [
    (0, 2048, "< 2k"),
    (2048, 4096, "2k-4k"),
    (4096, 8192, "4k-8k"),
    (8192, 16384, "8k-16k"),
    (16384, 32768, "16k-32k"),
]

HEADING_COUNT_SLICES = [
    (3, 5, "3-5 headings"),
    (6, 15, "6-15 headings"),
    (16, 40, "16-40 headings"),
    (41, 999, "40+ headings"),
]


def aggregate_metrics(results: list[EvalResult]) -> dict[str, float]:
    if not results:
        return {}
    return {
        "n": len(results),
        "exact_match": float(np.mean([r.exact_match for r in results])),
        "per_heading_accuracy": float(np.mean([r.per_heading_accuracy for r in results])),
        "hierarchy_preservation": float(np.mean([r.hierarchy_preservation for r in results])),
        "mean_absolute_error": float(
            np.mean([r.mean_absolute_error for r in results if r.level_count_match])
        )
        if any(r.level_count_match for r in results)
        else float("inf"),
        "level_count_match": float(np.mean([r.level_count_match for r in results])),
    }


def slice_results(results: list[EvalResult]) -> dict[str, list[EvalResult]]:
    slices: dict[str, list[EvalResult]] = {"overall": results}

    for lo, hi, label in LENGTH_SLICES:
        slices[f"len:{label}"] = [
            r for r in results if lo <= r.metadata.get("token_count", 0) < hi
        ]

    for lo, hi, label in HEADING_COUNT_SLICES:
        slices[f"headings:{label}"] = [
            r for r in results if lo <= r.metadata.get("heading_count", 0) <= hi
        ]

    for source in ("github_code", "goodwiki"):
        slices[f"source:{source}"] = [
            r for r in results if r.metadata.get("source") == source
        ]

    return slices


def run_full_evaluation(
    predictions: list[list[int]],
    ground_truths: list[list[int]],
    metadata_list: list[dict],
    output_dir: Path | None = None,
) -> dict[str, dict[str, float]]:
    results: list[EvalResult] = []
    for pred, truth, meta in zip(predictions, ground_truths, metadata_list):
        result = compute_all_metrics(pred, truth)
        result.metadata = meta
        results.append(result)

    slices = slice_results(results)
    aggregated = {name: aggregate_metrics(rs) for name, rs in slices.items()}

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(aggregated, f, indent=2)
        with open(output_dir / "per_example.jsonl", "w") as f:
            for r in results:
                f.write(r.model_dump_json() + "\n")

    return aggregated
