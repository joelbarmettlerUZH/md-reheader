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
        md = "Some intro text\n# Title\nParagraph here\n## Sub\nMore text\nNot a heading"
        result = extract_headings(md)
        assert result == [H(text="Title", level=1), H(text="Sub", level=2)]
