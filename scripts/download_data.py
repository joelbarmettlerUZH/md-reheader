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


def _is_priority_path(path: str) -> bool:
    path_lower = path.lower()
    return any(p in path_lower for p in PRIORITY_PATTERNS)


def _serialize_headings(headings: list[Heading]) -> list[dict]:
    return [h.model_dump() for h in headings]


def _write_record(f: IO, record: dict) -> None:
    f.write(json.dumps(record, ensure_ascii=False) + "\n")


def download_github_code(
    output_path: Path,
    target_count: int = 100_000,
    priority_ratio: float = 0.7,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    max_readmes = int(target_count * (1 - priority_ratio))
    max_priority = target_count - max_readmes

    # Legacy loading script no longer supported in datasets>=4.0; load parquets directly
    ds = load_dataset(
        "parquet",
        data_files="hf://datasets/codeparrot/github-code/data/train-*.parquet",
        streaming=True,
        split="train",
    )

    priority_count = 0
    readme_count = 0
    scanned = 0

    with open(output_path, "w") as f:
        for sample in ds:
            path: str = sample["path"]
            if not path.lower().endswith((".md", ".markdown")):
                continue

            scanned += 1
            content: str = sample["content"]
            is_priority = _is_priority_path(path)

            if is_priority and priority_count >= max_priority:
                continue
            if not is_priority and readme_count >= max_readmes:
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
                    "is_priority_path": is_priority,
                    "max_level_gap": compute_heading_level_gap(headings),
                },
            }
            _write_record(f, record)

            if is_priority:
                priority_count += 1
            else:
                readme_count += 1

            total = priority_count + readme_count
            if total % 5_000 == 0:
                logger.info(
                    f"github-code: {total}/{target_count} "
                    f"(priority={priority_count}, readme={readme_count}, scanned={scanned})"
                )

            if total >= target_count:
                break

    total = priority_count + readme_count
    logger.info(
        f"github-code done: {total} docs saved to {output_path} "
        f"(priority={priority_count}, readme={readme_count}, scanned={scanned})"
    )
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


def download_github_code_long(
    output_path: Path,
    target_count: int = 20_000,
    min_char_length: int = 25_000,
    start_shard: int = 185,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_shards = 1126
    shard_files = [
        f"hf://datasets/codeparrot/github-code/data/train-{i:05d}-of-{total_shards:05d}.parquet"
        for i in range(start_shard, total_shards)
    ]

    ds = load_dataset(
        "parquet",
        data_files=shard_files,
        streaming=True,
        split="train",
    )

    count = 0
    scanned = 0

    with open(output_path, "w") as f:
        for sample in ds:
            path: str = sample["path"]
            if not path.lower().endswith((".md", ".markdown")):
                continue

            content: str = sample["content"]
            scanned += 1

            if len(content) < min_char_length:
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
            count += 1

            if count % 2_000 == 0:
                logger.info(
                    f"github-code-long: {count}/{target_count} (scanned={scanned})"
                )

            if count >= target_count:
                break

    logger.info(
        f"github-code-long done: {count} docs saved to {output_path} (scanned={scanned})"
    )
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

    logger.info("=== Downloading github-code long docs ===")
    gc_long_count = download_github_code_long(raw_dir / "github_code_long_raw.jsonl")

    logger.info(f"=== Total raw docs: {gc_count + gw_count + gc_long_count} ===")


if __name__ == "__main__":
    main()
