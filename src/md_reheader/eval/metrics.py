import numpy as np

from md_reheader.models import EvalResult


def exact_match(pred: list[int], truth: list[int]) -> float:
    return 1.0 if pred == truth else 0.0


def per_heading_accuracy(pred: list[int], truth: list[int]) -> float:
    if len(pred) != len(truth):
        return 0.0
    correct = sum(1 for p, g in zip(pred, truth) if p == g)
    return correct / len(truth)


def hierarchy_preservation(pred: list[int], truth: list[int]) -> float:
    if len(pred) != len(truth) or len(pred) < 2:
        return 0.0
    correct = 0
    for i in range(len(pred) - 1):
        pred_dir = _sign(pred[i + 1] - pred[i])
        truth_dir = _sign(truth[i + 1] - truth[i])
        if pred_dir == truth_dir:
            correct += 1
    return correct / (len(pred) - 1)


def mean_absolute_error(pred: list[int], truth: list[int]) -> float:
    if len(pred) != len(truth):
        return float("inf")
    return float(np.mean([abs(p - g) for p, g in zip(pred, truth)]))


def level_count_match(pred: list[int], truth: list[int]) -> bool:
    return len(pred) == len(truth)


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def compute_all_metrics(pred: list[int], truth: list[int]) -> EvalResult:
    return EvalResult(
        exact_match=exact_match(pred, truth),
        per_heading_accuracy=per_heading_accuracy(pred, truth),
        hierarchy_preservation=hierarchy_preservation(pred, truth),
        mean_absolute_error=mean_absolute_error(pred, truth),
        level_count_match=level_count_match(pred, truth),
        predicted_levels=pred,
        ground_truth_levels=truth,
    )
