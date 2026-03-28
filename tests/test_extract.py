from md_reheader.data.extract import extract_headings
from md_reheader.models import Heading

H = Heading


class TestExtractHeadings:
    def test_basic_headings(self):
        md = "# Title\nSome text\n## Section\nMore text\n### Subsection"
        result = extract_headings(md)
        assert result == [
            H(text="Title", level=1), H(text="Section", level=2), H(text="Subsection", level=3),
        ]

    def test_no_headings(self):
        md = "Just some plain text\nwith multiple lines"
        assert extract_headings(md) == []

    def test_all_levels(self):
        md = "# H1\n## H2\n### H3\n#### H4\n##### H5\n###### H6"
        result = extract_headings(md)
        assert len(result) == 6
        assert [h.level for h in result] == [1, 2, 3, 4, 5, 6]

    def test_heading_with_trailing_hashes(self):
        md = "## Section ##"
        result = extract_headings(md)
        assert result == [H(text="Section", level=2)]

    def test_no_space_after_hash_not_heading(self):
        md = "#NotAHeading"
        assert extract_headings(md) == []

    def test_empty_string(self):
        assert extract_headings("") == []

    def test_mixed_content(self):
        md = "Some intro text\n\n# Title\n\nParagraph here\n\n## Sub\n\nMore text"
        result = extract_headings(md)
        assert result == [H(text="Title", level=1), H(text="Sub", level=2)]

    def test_skips_code_blocks(self):
        md = "# Real heading\n\n```bash\n# this is a comment\npip install foo\n```\n\n## Also real"
        result = extract_headings(md)
        assert result == [H(text="Real heading", level=1), H(text="Also real", level=2)]

    def test_skips_nested_code_blocks(self):
        md = "# Before\n\n```python\n# comment 1\n# comment 2\n```\n\n# After"
        result = extract_headings(md)
        assert result == [H(text="Before", level=1), H(text="After", level=1)]

    def test_code_block_with_language_tag(self):
        md = "# Title\n\n```javascript\n// code\n# not a heading\n```\n\n## Section"
        result = extract_headings(md)
        assert result == [H(text="Title", level=1), H(text="Section", level=2)]
