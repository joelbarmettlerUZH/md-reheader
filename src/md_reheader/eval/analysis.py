from md_reheader.models import EvalResult


def categorize_errors(results: list[EvalResult]) -> dict[str, list[EvalResult]]:
    categories: dict[str, list[EvalResult]] = {
        "count_mismatch": [],
        "off_by_one": [],
        "flat_prediction": [],
        "inverted_hierarchy": [],
        "lost_in_middle": [],
        "correct": [],
        "other": [],
    }

    for r in results:
        if not r.level_count_match:
            categories["count_mismatch"].append(r)
        elif r.exact_match == 1.0:
            categories["correct"].append(r)
        elif _is_constant_offset(r.predicted_levels, r.ground_truth_levels):
            categories["off_by_one"].append(r)
        elif len(set(r.predicted_levels)) == 1:
            categories["flat_prediction"].append(r)
        elif r.hierarchy_preservation < 0.5:
            categories["inverted_hierarchy"].append(r)
        elif _is_lost_in_middle(r.predicted_levels, r.ground_truth_levels):
            categories["lost_in_middle"].append(r)
        else:
            categories["other"].append(r)

    return categories


def _is_constant_offset(pred: list[int], truth: list[int]) -> bool:
    if len(pred) != len(truth) or not pred:
        return False
    offset = pred[0] - truth[0]
    return all(p - t == offset for p, t in zip(pred, truth))


def _is_lost_in_middle(pred: list[int], truth: list[int]) -> bool:
    """Known failure mode: small models attend well to start/end but degrade in the middle."""
    if len(pred) != len(truth) or len(pred) < 8:
        return False
    quarter = len(pred) // 4

    start_acc = sum(p == t for p, t in zip(pred[:quarter], truth[:quarter])) / quarter
    end_acc = sum(p == t for p, t in zip(pred[-quarter:], truth[-quarter:])) / quarter
    mid_acc = sum(
        p == t for p, t in zip(pred[quarter:-quarter], truth[quarter:-quarter])
    ) / (len(pred) - 2 * quarter)

    return start_acc > 0.7 and end_acc > 0.7 and mid_acc < 0.4
