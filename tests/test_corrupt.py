from md_reheader.data.corrupt import (
    corrupt_document,
    corrupt_flatten_all,
    corrupt_random_levels,
    corrupt_shift_levels,
)


class TestCorruptFlattenAll:
    def test_flatten_preserves_content(self):
        md = "## Hello\nSome text\n### World\nMore text"
        flat = corrupt_flatten_all(md)
        assert "# Hello" in flat
        assert "# World" in flat
        assert "Some text" in flat

    def test_flatten_removes_deeper_levels(self):
        md = "### Deep heading"
        flat = corrupt_flatten_all(md)
        assert flat.startswith("# ")
        assert "###" not in flat

    def test_flatten_already_flat(self):
        md = "# Already flat\n# Also flat"
        flat = corrupt_flatten_all(md)
        assert flat == md


class TestCorruptRandomLevels:
    def test_deterministic_with_seed(self):
        md = "## Hello\n### World\n#### Deep"
        result1 = corrupt_random_levels(md, seed=42)
        result2 = corrupt_random_levels(md, seed=42)
        assert result1 == result2

    def test_preserves_non_heading_lines(self):
        md = "## Heading\nSome text here\nMore text"
        result = corrupt_random_levels(md, seed=1)
        assert "Some text here" in result
        assert "More text" in result


class TestCorruptShiftLevels:
    def test_shift_up(self):
        md = "# Title\n## Section"
        result = corrupt_shift_levels(md, shift=1)
        assert result.startswith("## ")

    def test_clamp_at_six(self):
        md = "###### Deep"
        result = corrupt_shift_levels(md, shift=1)
        assert result.startswith("###### ")


class TestCorruptDocument:
    def test_flatten_strategy(self):
        md = "## Hello\n### World"
        result = corrupt_document(md, strategy="flatten")
        assert result == corrupt_flatten_all(md)

    def test_mixed_strategy_deterministic(self):
        md = "## Hello\n### World"
        r1 = corrupt_document(md, strategy="mixed", seed=42)
        r2 = corrupt_document(md, strategy="mixed", seed=42)
        assert r1 == r2

    def test_unknown_strategy_raises(self):
        import pytest

        with pytest.raises(ValueError, match="Unknown corruption"):
            corrupt_document("## Test", strategy="nonexistent")
