import json
import logging
from pathlib import Path
from typing import IO

from datasets import load_dataset

from md_reheader.data.extract import extract_headings
from md_reheader.data.filter import compute_heading_level_gap, passes_cheap_filters
from md_reheader.models import Heading

logger = logging.getLogger(__name__)

PRIORITY_PATTERNS = ["docs/", "doc/", "wiki/", "guide/", "manual/", "tutorial/"]

DEFAULT_BUCKET_TARGETS: dict[str, int] = {
    "< 4k": 50_000,
    "4k-8k": 30_000,
    "8k-16k": 15_000,
    "16k-32k": 10_000,
}

BUCKET_CHAR_RANGES: dict[str, tuple[int, int]] = {
    "< 4k": (0, 16_000),
    "4k-8k": (16_000, 32_000),
    "8k-16k": (32_000, 64_000),
    "16k-32k": (64_000, 150_000),
}


def _is_priority_path(path: str) -> bool:
    path_lower = path.lower()
    return any(p in path_lower for p in PRIORITY_PATTERNS)


def _char_bucket(char_len: int) -> str | None:
    for bucket, (lo, hi) in BUCKET_CHAR_RANGES.items():
        if lo <= char_len < hi:
            return bucket
    return None


def _serialize_headings(headings: list[Heading]) -> list[dict]:
    return [h.model_dump() for h in headings]


def _write_record(f: IO, record: dict) -> None:
    f.write(json.dumps(record, ensure_ascii=False) + "\n")


def download_github_code(
    output_path: Path,
    bucket_targets: dict[str, int] | None = None,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    targets = bucket_targets or DEFAULT_BUCKET_TARGETS
    counts: dict[str, int] = {b: 0 for b in targets}

    ds = load_dataset(
        "parquet",
        data_files="hf://datasets/codeparrot/github-code/data/train-*.parquet",
        streaming=True,
        split="train",
    )

    scanned = 0
    with open(output_path, "w") as f:
        for sample in ds:
            path: str = sample["path"]
            if not path.lower().endswith((".md", ".markdown")):
                continue

            content: str = sample["content"]
            scanned += 1

            bucket = _char_bucket(len(content))
            if bucket is None or counts[bucket] >= targets[bucket]:
                if all(counts[b] >= targets[b] for b in targets):
                    break
                continue

            headings = extract_headings(content)
            if not passes_cheap_filters(content, headings):
                continue

            record = {
                "content": content,
                "headings": _serialize_headings(headings),
                "source": "github_code",
                "meta": {
                    "repo": sample["repo_name"],
                    "path": path,
                    "license": sample["license"],
                    "is_priority_path": _is_priority_path(path),
                    "max_level_gap": compute_heading_level_gap(headings),
                },
            }
            _write_record(f, record)
            counts[bucket] += 1

            total = sum(counts.values())
            if total % 5_000 == 0:
                bucket_str = ", ".join(f"{b}={counts[b]}/{targets[b]}" for b in targets)
                logger.info(f"github-code: {total} total ({bucket_str}, scanned={scanned})")

    total = sum(counts.values())
    bucket_str = ", ".join(f"{b}={counts[b]}/{targets[b]}" for b in targets)
    logger.info(f"github-code done: {total} docs ({bucket_str}, scanned={scanned})")
    return total


def download_goodwiki(output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("euirim/goodwiki", split="train")

    count = 0
    with open(output_path, "w") as f:
        for sample in ds:
            title: str = sample["title"]
            # Wikipedia's title IS the H1; the dataset just doesn't include it in the markdown
            content = f"# {title}\n\n{sample['markdown']}"
            headings = extract_headings(content)

            if not passes_cheap_filters(content, headings):
                continue

            record = {
                "content": content,
                "headings": _serialize_headings(headings),
                "source": "goodwiki",
                "meta": {
                    "title": title,
                    "pageid": sample["pageid"],
                    "categories": sample["categories"],
                    "max_level_gap": compute_heading_level_gap(headings),
                },
            }
            _write_record(f, record)
            count += 1

            if count % 5_000 == 0:
                logger.info(f"goodwiki: {count} docs collected")

    logger.info(f"goodwiki done: {count} docs saved to {output_path}")
    return count


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    raw_dir = Path("data/raw")

    logger.info("=== Downloading github-code (Markdown) ===")
    gc_count = download_github_code(raw_dir / "github_code_raw.jsonl")

    logger.info("=== Downloading goodwiki ===")
    gw_count = download_goodwiki(raw_dir / "goodwiki_raw.jsonl")

    logger.info(f"=== Total raw docs: {gc_count + gw_count} ===")


if __name__ == "__main__":
    main()
