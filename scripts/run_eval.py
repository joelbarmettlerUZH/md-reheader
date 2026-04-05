import argparse
import json
import re
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from md_reheader.data.format import parse_levels_from_output, parse_levels_from_output_v2
from md_reheader.data.strip import MARKER_START
from md_reheader.eval.analysis import categorize_errors
from md_reheader.eval.evaluate import run_full_evaluation
from md_reheader.eval.metrics import compute_all_metrics


def load_test_data(path: Path) -> list[dict]:
    examples = []
    with open(path) as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def detect_format(examples: list[dict]) -> str:
    """Detect v1 vs v2 based on assistant output format."""
    sample = examples[0]["messages"][2]["content"]
    if re.match(r"^#{1,6}\s", sample):
        return "v1"
    return "v2"


def extract_ground_truth(example: dict, fmt: str) -> list[int]:
    content = example["messages"][2]["content"]
    if fmt == "v1":
        return parse_levels_from_output(content)
    return parse_levels_from_output_v2(content)


def extract_system_prompt(example: dict) -> str:
    return example["messages"][0]["content"]


def extract_user_content(example: dict) -> str:
    return example["messages"][1]["content"]


def estimate_max_tokens(user_content: str, fmt: str) -> int:
    if fmt == "v2":
        n_headings = user_content.count(MARKER_START)
        # ~3 tokens per heading (digit + newline), with buffer
        return max(n_headings * 5, 64)
    # v1: heading text + # prefix
    n_headings = len(re.findall(r"^#{1,6}\s", user_content, re.MULTILINE))
    return max(n_headings * 40, 256)


def parse_output(response: str, fmt: str) -> list[int]:
    if fmt == "v1":
        return parse_levels_from_output(response)
    return parse_levels_from_output_v2(response)


def predict_batch(
    batch_examples: list[dict],
    model,
    tokenizer,
    max_new_tokens: int,
    system_prompt: str,
) -> list[list[int]]:
    fmt = detect_format(batch_examples)

    prompts = []
    for ex in batch_examples:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": extract_user_content(ex)},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        prompts.append(prompt)

    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    predictions = []
    for j in range(len(batch_examples)):
        prompt_len = inputs["input_ids"][j].ne(tokenizer.pad_token_id).sum().item()
        generated_ids = outputs[j][prompt_len:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        predictions.append(parse_output(response, fmt))

    return predictions


def run_inference(
    examples: list[dict],
    model,
    tokenizer,
    batch_size: int = 8,
) -> list[list[int]]:
    fmt = detect_format(examples)
    system_prompt = extract_system_prompt(examples[0])

    indexed = sorted(
        enumerate(examples),
        key=lambda x: x[1].get("metadata", {}).get("token_count", 0),
    )

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    long_threshold = 8192
    all_predictions: list[tuple[int, list[int]]] = []

    for batch_start in tqdm(range(0, len(indexed), batch_size), desc="Batches"):
        batch = indexed[batch_start : batch_start + batch_size]
        max_tok_count = max(ex.get("metadata", {}).get("token_count", 0) for _, ex in batch)

        if max_tok_count >= long_threshold:
            for orig_idx, ex in batch:
                max_tokens = estimate_max_tokens(extract_user_content(ex), fmt)
                preds = predict_batch([ex], model, tokenizer, max_tokens, system_prompt)
                all_predictions.append((orig_idx, preds[0]))
        else:
            orig_indices = [idx for idx, _ in batch]
            batch_examples = [ex for _, ex in batch]
            max_tokens = max(
                estimate_max_tokens(extract_user_content(ex), fmt) for ex in batch_examples
            )
            preds = predict_batch(batch_examples, model, tokenizer, max_tokens, system_prompt)
            for orig_idx, pred in zip(orig_indices, preds):
                all_predictions.append((orig_idx, pred))

    all_predictions.sort(key=lambda x: x[0])
    return [pred for _, pred in all_predictions]


def print_results(aggregated: dict[str, dict[str, float]]) -> None:
    for slice_name, metrics in sorted(aggregated.items()):
        if not metrics:
            continue
        n = int(metrics["n"])
        print(f"\n{slice_name} (n={n}):")
        print(f"  Exact match:             {metrics['exact_match']:.4f}")
        print(f"  Per-heading accuracy:    {metrics['per_heading_accuracy']:.4f}")
        print(f"  Hierarchy preservation:  {metrics['hierarchy_preservation']:.4f}")
        print(f"  Mean absolute error:     {metrics['mean_absolute_error']:.4f}")
        print(f"  Level count match:       {metrics['level_count_match']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./checkpoints", help="Model path or HF model ID")
    parser.add_argument("--test-data", default="./data/processed/test.jsonl")
    parser.add_argument("--output-dir", default="./results/model")
    parser.add_argument("--device", default="auto", help="Device: auto, cuda:0, cuda:1")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    device_map = {"": args.device} if args.device != "auto" else "auto"

    print(f"Loading model from {args.model} on {args.device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=device_map,
    )
    model.eval()

    print(f"Loading test data from {args.test_data}")
    examples = load_test_data(Path(args.test_data))
    fmt = detect_format(examples)
    print(f"  {len(examples)} examples, format={fmt}, batch_size={args.batch_size}")

    ground_truths = [extract_ground_truth(ex, fmt) for ex in examples]
    metadata_list = [ex.get("metadata", {}) for ex in examples]

    print("Running batched inference...")
    predictions = run_inference(examples, model, tokenizer, batch_size=args.batch_size)

    print("Computing metrics...")
    aggregated = run_full_evaluation(predictions, ground_truths, metadata_list, output_dir)

    print("\n=== Results ===")
    print_results(aggregated)

    all_results = []
    for pred, truth, meta in zip(predictions, ground_truths, metadata_list):
        r = compute_all_metrics(pred, truth)
        r.metadata = meta
        all_results.append(r)

    categories = categorize_errors(all_results)
    print("\n=== Error Analysis ===")
    for cat, items in categories.items():
        print(f"  {cat}: {len(items)} ({100 * len(items) / len(all_results):.1f}%)")

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
