from md_reheader.data.filter import (
    compute_heading_level_gap,
    passes_cheap_filters,
)
from md_reheader.models import Heading

H = Heading


class TestPassesCheapFilters:
    def test_valid_document(self):
        md = "x" * 500
        headings = [H(text="A", level=1), H(text="B", level=2), H(text="C", level=2)]
        assert passes_cheap_filters(md, headings) is True

    def test_too_short(self):
        md = "x" * 100
        headings = [H(text="A", level=1), H(text="B", level=2), H(text="C", level=2)]
        assert passes_cheap_filters(md, headings) is False

    def test_too_long(self):
        md = "x" * 200_000
        headings = [H(text="A", level=1), H(text="B", level=2), H(text="C", level=2)]
        assert passes_cheap_filters(md, headings) is False

    def test_too_few_headings(self):
        md = "x" * 500
        headings = [H(text="A", level=1), H(text="B", level=2)]
        assert passes_cheap_filters(md, headings) is False

    def test_single_level(self):
        md = "x" * 500
        headings = [H(text="A", level=1), H(text="B", level=1), H(text="C", level=1)]
        assert passes_cheap_filters(md, headings) is False

    def test_too_many_headings(self):
        md = "x" * 500
        headings = [H(text=f"H{i}", level=1 + i % 3) for i in range(130)]
        assert passes_cheap_filters(md, headings) is False

    def test_custom_thresholds(self):
        md = "x" * 100
        headings = [H(text="A", level=1), H(text="B", level=2), H(text="C", level=3)]
        assert passes_cheap_filters(md, headings, min_char_length=50) is True
        assert passes_cheap_filters(md, headings, min_char_length=200) is False

    def test_no_level_gap_filter(self):
        md = "x" * 500
        headings = [H(text="A", level=1), H(text="B", level=4), H(text="C", level=6)]
        assert passes_cheap_filters(md, headings) is True


class TestComputeHeadingLevelGap:
    def test_no_gap(self):
        headings = [H(text="A", level=1), H(text="B", level=2), H(text="C", level=3)]
        assert compute_heading_level_gap(headings) == 1

    def test_large_gap(self):
        headings = [H(text="A", level=1), H(text="B", level=4), H(text="C", level=5)]
        assert compute_heading_level_gap(headings) == 3

    def test_single_level(self):
        headings = [H(text="A", level=2), H(text="B", level=2)]
        assert compute_heading_level_gap(headings) == 0

    def test_non_contiguous(self):
        headings = [H(text="A", level=1), H(text="B", level=3), H(text="C", level=6)]
        assert compute_heading_level_gap(headings) == 3
