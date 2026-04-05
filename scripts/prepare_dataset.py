import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path

from datasets import Dataset, DatasetDict

from md_reheader.data.extract import extract_headings
from md_reheader.data.filter import compute_token_count, passes_token_filter
from md_reheader.data.format import format_training_example
from md_reheader.data.strip import count_headings, strip_document, truncate_stripped

logger = logging.getLogger(__name__)

DEPTH_MULTIPLIERS = {4: 2, 5: 4, 6: 8}


def load_raw_jsonl(raw_dir: Path) -> list[dict]:
    docs: list[dict] = []
    for jsonl_path in sorted(raw_dir.glob("*_raw.jsonl")):
        logger.info(f"Loading {jsonl_path}")
        with open(jsonl_path) as f:
            for line in f:
                if line.strip():
                    docs.append(json.loads(line))
    logger.info(f"Loaded {len(docs)} raw documents total")
    return docs


def _split_key(doc: dict) -> str:
    meta = doc.get("meta", {})
    return meta.get("repo") or meta.get("title") or ""


def split_by_source_key(
    docs: list[dict],
    train_ratio: float = 0.90,
    val_ratio: float = 0.05,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    by_key: dict[str, list[dict]] = defaultdict(list)
    for doc in docs:
        by_key[_split_key(doc)].append(doc)

    keys = sorted(by_key.keys())
    rng = random.Random(seed)
    rng.shuffle(keys)

    total = len(docs)
    train_target = int(total * train_ratio)
    val_target = int(total * val_ratio)

    train: list[dict] = []
    val: list[dict] = []
    test: list[dict] = []
    train_count = 0
    val_count = 0

    for key in keys:
        group = by_key[key]
        if train_count < train_target:
            train.extend(group)
            train_count += len(group)
        elif val_count < val_target:
            val.extend(group)
            val_count += len(group)
        else:
            test.extend(group)

    logger.info(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test


def oversample_deep_docs(
    docs: list[dict],
    multipliers: dict[int, int] | None = None,
) -> list[dict]:
    multipliers = multipliers or DEPTH_MULTIPLIERS
    result: list[dict] = []
    for doc in docs:
        headings = extract_headings(doc["content"])
        if not headings:
            result.append(doc)
            continue
        max_depth = max(h.level for h in headings)
        mult = 1
        for depth, m in sorted(multipliers.items()):
            if max_depth >= depth:
                mult = m
        result.extend([doc] * mult)
    return result


def process_split(
    docs: list[dict],
    split_name: str,
    seed: int = 42,
) -> list[dict]:
    processed: list[dict] = []
    for _i, doc in enumerate(docs):
        content = doc["content"]
        headings = extract_headings(content)
        if not headings:
            continue
        true_levels = [h.level for h in headings]

        stripped = strip_document(content, headings)
        stripped = truncate_stripped(stripped, max_tokens=7500)

        visible_count = count_headings(stripped)
        true_levels = true_levels[:visible_count]
        headings = headings[:visible_count]
        if not true_levels:
            continue

        token_count = compute_token_count(stripped)

        example = format_training_example(
            stripped_md=stripped,
            headings=headings,
            true_levels=true_levels,
        )

        example.metadata = {
            "source": doc["source"],
            "token_count": token_count,
            "heading_count": len(headings),
            "max_depth": max(true_levels),
            "min_depth": min(true_levels),
            "max_level_gap": doc.get("meta", {}).get("max_level_gap", 0),
            **{k: v for k, v in doc.get("meta", {}).items() if k != "max_level_gap"},
        }
        processed.append(example.model_dump())

    return processed


def save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(records)} examples to {path}")


def prepare_dataset(
    raw_dir: Path = Path("data/raw"),
    output_dir: Path = Path("data/processed"),
    seed: int = 42,
    push_to_hub: bool = False,
    hub_repo: str = "",
) -> None:
    raw_docs = load_raw_jsonl(raw_dir)

    logger.info("Applying token-count filter...")
    before = len(raw_docs)
    docs = [d for d in raw_docs if passes_token_filter(d["content"])]
    logger.info(f"Token filter: {before} -> {len(docs)} ({before - len(docs)} dropped)")

    source_counts: dict[str, int] = defaultdict(int)
    for d in docs:
        source_counts[d["source"]] += 1
    for source, count in sorted(source_counts.items()):
        logger.info(f"  {source}: {count}")

    train, val, test = split_by_source_key(docs, seed=seed)

    logger.info(f"Oversampling deep docs in train split ({len(train)} before)...")
    train = oversample_deep_docs(train)
    logger.info(f"Train after oversampling: {len(train)}")

    for split_name, split_docs in [("train", train), ("val", val), ("test", test)]:
        logger.info(f"Processing {split_name} split ({len(split_docs)} docs)...")
        processed = process_split(split_docs, split_name, seed=seed)
        save_jsonl(processed, output_dir / f"{split_name}.jsonl")

    if push_to_hub and hub_repo:
        logger.info(f"Pushing to HuggingFace Hub: {hub_repo}")
        train_ds = Dataset.from_json(str(output_dir / "train.jsonl"))
        val_ds = Dataset.from_json(str(output_dir / "val.jsonl"))
        test_ds = Dataset.from_json(str(output_dir / "test.jsonl"))
        ds = DatasetDict({"train": train_ds, "validation": val_ds, "test": test_ds})
        ds.push_to_hub(hub_repo, private=False)
        logger.info(f"Pushed to {hub_repo}")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="data/processed")
    args = parser.parse_args()

    prepare_dataset(output_dir=Path(args.output_dir))


if __name__ == "__main__":
    main()
