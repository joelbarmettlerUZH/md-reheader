import pytest

from md_reheader.data.apply import apply_levels
from md_reheader.data.extract import extract_headings

SIMPLE_DOC = """\
# Title

Some intro text.

## Section One

Body content.

### Subsection

More content.
"""


class TestApplyLevels:
    def test_identity(self):
        """Applying original levels should not change the document."""
        headings = extract_headings(SIMPLE_DOC)
        levels = [h.level for h in headings]
        result = apply_levels(SIMPLE_DOC, levels)
        assert result == SIMPLE_DOC

    def test_round_trip(self):
        """Extract levels, change them, apply, re-extract — should match."""
        new_levels = [1, 3, 5]
        result = apply_levels(SIMPLE_DOC, new_levels)
        headings = extract_headings(result)
        assert [h.level for h in headings] == new_levels

    def test_flatten_all(self):
        new_levels = [2, 2, 2]
        result = apply_levels(SIMPLE_DOC, new_levels)
        headings = extract_headings(result)
        assert all(h.level == 2 for h in headings)

    def test_preserves_heading_text(self):
        new_levels = [1, 3, 5]
        result = apply_levels(SIMPLE_DOC, new_levels)
        headings = extract_headings(result)
        assert headings[0].text == "Title"
        assert headings[1].text == "Section One"
        assert headings[2].text == "Subsection"

    def test_preserves_body_text(self):
        new_levels = [1, 2, 3]
        result = apply_levels(SIMPLE_DOC, new_levels)
        assert "Some intro text." in result
        assert "Body content." in result

    def test_wrong_level_count_raises(self):
        with pytest.raises(ValueError, match="does not match"):
            apply_levels(SIMPLE_DOC, [1, 2, 3, 4])

    def test_code_block_headings_untouched(self):
        doc = """\
# Title

```
# comment in code
```

## Section
"""
        result = apply_levels(doc, [3, 4])
        assert "# comment in code" in result
        headings = extract_headings(result)
        assert headings[0].level == 3
        assert headings[1].level == 4
