import hashlib
import json
import logging
import random
from collections import defaultdict
from pathlib import Path

from datasets import Dataset, DatasetDict

from md_reheader.data.corrupt import corrupt_document
from md_reheader.data.filter import compute_token_count, passes_token_filter
from md_reheader.data.format import format_training_example
from md_reheader.models import Heading

logger = logging.getLogger(__name__)


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


def process_split(
    docs: list[dict],
    split_name: str,
    seed: int = 42,
) -> list[dict]:
    processed: list[dict] = []
    for i, doc in enumerate(docs):
        content = doc["content"]
        raw_headings = doc["headings"]
        headings = [Heading(text=h["text"], level=h["level"]) for h in raw_headings]
        true_levels = [h.level for h in headings]

        doc_seed_str = f"{seed}-{split_name}-{i}-{_split_key(doc)}"
        doc_seed = int(hashlib.md5(doc_seed_str.encode()).hexdigest()[:8], 16)

        corrupted = corrupt_document(content, strategy="mixed", seed=doc_seed)

        example = format_training_example(
            corrupted_md=corrupted,
            headings=headings,
            true_levels=true_levels,
        )

        token_count = compute_token_count(content)
        example.metadata = {
            "source": doc["source"],
            "token_count": token_count,
            "heading_count": len(headings),
            "max_depth": max(h.level for h in headings),
            "min_depth": min(h.level for h in headings),
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
    prepare_dataset()


if __name__ == "__main__":
    main()
