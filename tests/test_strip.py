import re

from md_reheader.data.extract import extract_headings
from md_reheader.data.strip import count_headings, strip_document

SIMPLE_DOC = """\
# Title

Some intro text here.

## Section One

Body of section one with some content.

### Subsection

Subsection content here.

## Section Two

More content here.
"""

DEEP_DOC = """\
# Project

Overview text.

## Module A

Module description.

### Class Foo

Class documentation.

#### Method bar

Method details.

##### Parameter baz

Parameter info.

## Module B

Another module.
"""


class TestStripDocument:
    def test_heading_count_matches(self):
        headings = extract_headings(SIMPLE_DOC)
        stripped = strip_document(SIMPLE_DOC, headings)
        assert count_headings(stripped) == len(headings)

    def test_all_headings_flattened_to_h1(self):
        headings = extract_headings(SIMPLE_DOC)
        stripped = strip_document(SIMPLE_DOC, headings)
        for line in stripped.split("\n"):
            if re.match(r"^#{1,6}\s", line):
                assert line.startswith("# "), f"Expected H1, got: {line}"

    def test_heading_text_preserved(self):
        headings = extract_headings(SIMPLE_DOC)
        stripped = strip_document(SIMPLE_DOC, headings)
        for heading in headings:
            assert f"# {heading.text}" in stripped

    def test_body_text_preserved_short_sections(self):
        headings = extract_headings(SIMPLE_DOC)
        stripped = strip_document(SIMPLE_DOC, headings)
        assert "Some intro text here." in stripped
        assert "Body of section one" in stripped

    def test_deep_doc_all_headings_present(self):
        headings = extract_headings(DEEP_DOC)
        stripped = strip_document(DEEP_DOC, headings)
        assert count_headings(stripped) == 6
        assert "# Parameter baz" in stripped

    def test_no_headings_returns_truncated_body(self):
        text = "Just some plain text without any headings."
        stripped = strip_document(text)
        assert "Just some plain text" in stripped

    def test_code_block_content_preserved(self):
        doc = """\
# Real Heading

```python
# This is a comment, not a heading
def foo():
    pass
```

## Another Heading
"""
        headings = extract_headings(doc)
        stripped = strip_document(doc, headings)
        # markdown-it-py correctly identifies only 2 real headings
        assert len(headings) == 2
        assert "This is a comment" in stripped

    def test_empty_body_between_headings(self):
        doc = """\
# Title
## Section One
## Section Two
"""
        headings = extract_headings(doc)
        stripped = strip_document(doc, headings)
        assert count_headings(stripped) == 3


class TestCountHeadings:
    def test_counts(self):
        text = "# A\n\nsome text\n\n# B\n\nmore text"
        assert count_headings(text) == 2

    def test_zero(self):
        assert count_headings("no headings here") == 0
