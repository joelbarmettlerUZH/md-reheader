import json
from pathlib import Path

from md_reheader.data.format import parse_headings_output, parse_levels_from_output
from md_reheader.eval.baselines import heuristic_baseline, naive_flat_baseline
from md_reheader.eval.evaluate import run_full_evaluation


def load_test_data(path: Path) -> list[dict]:
    examples = []
    with open(path) as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def extract_ground_truth(example: dict) -> list[int]:
    assistant_msg = example["messages"][2]["content"]
    return parse_levels_from_output(assistant_msg)


def extract_heading_texts(example: dict) -> list[str]:
    assistant_msg = example["messages"][2]["content"]
    headings = parse_headings_output(assistant_msg)
    return [h.text for h in headings]


def run_baseline(name: str, baseline_fn, examples: list[dict], output_dir: Path) -> None:
    predictions: list[list[int]] = []
    ground_truths: list[list[int]] = []
    metadata_list: list[dict] = []

    for ex in examples:
        texts = extract_heading_texts(ex)
        truth = extract_ground_truth(ex)
        pred = baseline_fn(texts)

        predictions.append(pred)
        ground_truths.append(truth)
        metadata_list.append(ex.get("metadata", {}))

    result_dir = output_dir / name
    aggregated = run_full_evaluation(predictions, ground_truths, metadata_list, result_dir)

    print(f"\n=== {name} ===")
    for slice_name, metrics in sorted(aggregated.items()):
        if not metrics:
            continue
        n = int(metrics["n"])
        print(f"\n  {slice_name} (n={n}):")
        print(f"    Exact match:             {metrics['exact_match']:.4f}")
        print(f"    Per-heading accuracy:    {metrics['per_heading_accuracy']:.4f}")
        print(f"    Hierarchy preservation:  {metrics['hierarchy_preservation']:.4f}")
        print(f"    Mean absolute error:     {metrics['mean_absolute_error']:.4f}")
        print(f"    Level count match:       {metrics['level_count_match']:.4f}")


def main() -> None:
    test_path = Path("./data/processed/test.jsonl")
    output_dir = Path("./results/baselines")

    print(f"Loading test data from {test_path}")
    examples = load_test_data(test_path)
    print(f"  {len(examples)} examples")

    run_baseline("naive_flat", naive_flat_baseline, examples, output_dir)
    run_baseline("heuristic", heuristic_baseline, examples, output_dir)

    print(f"\nBaseline results saved to {output_dir}")


if __name__ == "__main__":
    main()
