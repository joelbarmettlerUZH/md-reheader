import tempfile
from pathlib import Path

from scripts.prepare_dataset import (
    load_raw_jsonl,
    process_split,
    save_jsonl,
    split_by_source_key,
)


def _make_doc(source="github_code", repo="owner/repo1", title=None, n_headings=4):
    headings = [
        {"text": f"Heading {i}", "level": 1 + (i % 3)}
        for i in range(n_headings)
    ]
    content_lines: list[str] = []
    for h in headings:
        content_lines.append(f"{'#' * h['level']} {h['text']}")
        content_lines.append("Some body text here.\n")

    meta: dict = {"max_level_gap": 1}
    if source == "github_code":
        meta["repo"] = repo
        meta["path"] = "docs/guide.md"
        meta["license"] = "mit"
        meta["is_priority_path"] = True
    else:
        meta["title"] = title or "Test Article"
        meta["pageid"] = 12345
        meta["categories"] = ["Test"]

    return {
        "content": "\n".join(content_lines),
        "headings": headings,
        "source": source,
        "meta": meta,
    }


class TestSplitBySourceKey:
    def test_basic_split(self):
        docs = [
            _make_doc(repo="owner/repo1"),
            _make_doc(repo="owner/repo1"),
            _make_doc(repo="owner/repo2"),
            _make_doc(repo="owner/repo3"),
            _make_doc(repo="owner/repo4"),
            _make_doc(repo="owner/repo5"),
            _make_doc(repo="owner/repo6"),
            _make_doc(repo="owner/repo7"),
            _make_doc(repo="owner/repo8"),
            _make_doc(repo="owner/repo9"),
        ]
        train, val, test = split_by_source_key(docs, seed=42)
        assert len(train) + len(val) + len(test) == len(docs)
        assert len(train) >= len(val)
        assert len(train) >= len(test)

    def test_same_repo_same_split(self):
        docs = [
            _make_doc(repo="owner/repo1"),
            _make_doc(repo="owner/repo1"),
            _make_doc(repo="owner/repo1"),
            _make_doc(repo="owner/repo2"),
            _make_doc(repo="owner/repo2"),
        ]
        train, val, test = split_by_source_key(docs, seed=42)

        repo_splits: dict[str, str] = {}
        for split_name, split_docs in [("train", train), ("val", val), ("test", test)]:
            for doc in split_docs:
                repo = doc["meta"]["repo"]
                if repo in repo_splits:
                    assert repo_splits[repo] == split_name, f"Repo {repo} in multiple splits"
                repo_splits[repo] = split_name

    def test_deterministic(self):
        docs = [_make_doc(repo=f"owner/repo{i}") for i in range(20)]
        t1, v1, s1 = split_by_source_key(docs, seed=42)
        t2, v2, s2 = split_by_source_key(docs, seed=42)
        assert len(t1) == len(t2)
        assert len(v1) == len(v2)
        assert len(s1) == len(s2)

    def test_goodwiki_splits_by_title(self):
        docs = [
            _make_doc(source="goodwiki", title="Article A"),
            _make_doc(source="goodwiki", title="Article A"),
            _make_doc(source="goodwiki", title="Article B"),
        ]
        train, val, test = split_by_source_key(docs, seed=42)
        assert len(train) + len(val) + len(test) == 3


class TestProcessSplit:
    def test_produces_chatml_format(self):
        docs = [_make_doc()]
        processed = process_split(docs, "train", seed=42)
        assert len(processed) == 1

        example = processed[0]
        assert "messages" in example
        assert len(example["messages"]) == 3
        assert example["messages"][0]["role"] == "system"
        assert example["messages"][1]["role"] == "user"
        assert example["messages"][2]["role"] == "assistant"

    def test_metadata_preserved(self):
        docs = [_make_doc(repo="owner/myrepo")]
        processed = process_split(docs, "train", seed=42)
        meta = processed[0]["metadata"]
        assert meta["source"] == "github_code"
        assert meta["heading_count"] == 4
        assert "token_count" in meta
        assert "max_depth" in meta
        assert meta["repo"] == "owner/myrepo"

    def test_deterministic_corruption(self):
        docs = [_make_doc()]
        p1 = process_split(docs, "train", seed=42)
        p2 = process_split(docs, "train", seed=42)
        assert p1[0]["messages"][1]["content"] == p2[0]["messages"][1]["content"]


class TestSaveAndLoadJsonl:
    def test_roundtrip(self):
        records = [{"a": 1, "b": "hello"}, {"a": 2, "b": "world"}]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jsonl"
            save_jsonl(records, path)

            raw_path = Path(tmpdir) / "test_raw.jsonl"
            path.rename(raw_path)
            loaded = load_raw_jsonl(Path(tmpdir))
            assert len(loaded) == 2
            assert loaded[0]["a"] == 1
            assert loaded[1]["b"] == "world"
