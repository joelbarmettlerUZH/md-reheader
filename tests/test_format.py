from md_reheader.data.format import (
    format_headings_output,
    format_training_example,
    parse_headings_output,
    parse_levels_from_output,
)
from md_reheader.models import Heading

H = Heading


class TestFormatHeadingsOutput:
    def test_basic(self):
        headings = [H(text="Intro", level=1), H(text="Methods", level=2), H(text="Data", level=3)]
        result = format_headings_output(headings, [1, 2, 3])
        assert result == "# Intro\n## Methods\n### Data"

    def test_single(self):
        headings = [H(text="Title", level=1)]
        result = format_headings_output(headings, [1])
        assert result == "# Title"

    def test_ignores_original_levels(self):
        headings = [H(text="A", level=1), H(text="B", level=1)]
        result = format_headings_output(headings, [1, 3])
        assert result == "# A\n### B"

    def test_all_six_levels(self):
        headings = [H(text=f"H{i}", level=i) for i in range(1, 7)]
        result = format_headings_output(headings, list(range(1, 7)))
        lines = result.splitlines()
        assert lines[0] == "# H1"
        assert lines[5] == "###### H6"


class TestParseHeadingsOutput:
    def test_basic(self):
        text = "# Intro\n## Methods\n### Data"
        result = parse_headings_output(text)
        assert result == [
            H(text="Intro", level=1), H(text="Methods", level=2), H(text="Data", level=3),
        ]

    def test_with_trailing_whitespace(self):
        text = "  # Title  \n  ## Section  "
        result = parse_headings_output(text)
        assert result == [H(text="Title", level=1), H(text="Section", level=2)]

    def test_skips_blank_lines(self):
        text = "# A\n\n## B\n\n### C"
        result = parse_headings_output(text)
        assert len(result) == 3

    def test_skips_non_heading_lines(self):
        text = "# Title\nSome random text\n## Section"
        result = parse_headings_output(text)
        assert result == [H(text="Title", level=1), H(text="Section", level=2)]

    def test_empty_string(self):
        assert parse_headings_output("") == []

    def test_all_levels(self):
        text = "# L1\n## L2\n### L3\n#### L4\n##### L5\n###### L6"
        result = parse_headings_output(text)
        assert [h.level for h in result] == [1, 2, 3, 4, 5, 6]

    def test_no_space_after_hash_ignored(self):
        text = "#NoSpace\n## Valid"
        result = parse_headings_output(text)
        assert result == [H(text="Valid", level=2)]


class TestParseLevelsFromOutput:
    def test_basic(self):
        text = "# Intro\n## Methods\n## Results"
        assert parse_levels_from_output(text) == [1, 2, 2]

    def test_empty(self):
        assert parse_levels_from_output("") == []


class TestFormatTrainingExample:
    def test_structure(self):
        result = format_training_example(
            stripped_md="# Flat\n# Also flat",
            headings=[H(text="Flat", level=1), H(text="Also flat", level=1)],
            true_levels=[1, 2],
        )
        assert len(result.messages) == 3
        assert result.messages[0].role == "system"
        assert result.messages[1].role == "user"
        assert result.messages[2].role == "assistant"
        assert result.messages[2].content == "# Flat\n## Also flat"

    def test_user_message_is_the_stripped_document(self):
        doc = "# Flat heading\nSome content\n# Another flat"
        result = format_training_example(
            stripped_md=doc,
            headings=[H(text="Flat heading", level=1), H(text="Another flat", level=1)],
            true_levels=[1, 2],
        )
        assert result.messages[1].content == doc

    def test_roundtrip(self):
        headings = [
            H(text="Intro", level=1),
            H(text="Methods", level=2),
            H(text="Data", level=3),
            H(text="Results", level=2),
        ]
        true_levels = [1, 2, 3, 2]
        result = format_training_example(
            stripped_md="# x\n# x\n# x\n# x",
            headings=headings,
            true_levels=true_levels,
        )
        parsed = parse_levels_from_output(result.messages[2].content)
        assert parsed == true_levels
