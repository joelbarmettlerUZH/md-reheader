from markdown_it import MarkdownIt

_md_parser = MarkdownIt("commonmark")


def apply_levels(md_text: str, levels: list[int]) -> str:
    """Replace heading prefixes in a markdown document with the given levels."""
    tokens = _md_parser.parse(md_text)
    lines = md_text.split("\n")

    heading_positions: list[int] = []
    for token in tokens:
        if token.type == "heading_open" and token.map is not None:
            heading_positions.append(token.map[0])

    if len(levels) != len(heading_positions):
        raise ValueError(
            f"Number of levels ({len(levels)}) does not match "
            f"number of headings ({len(heading_positions)})"
        )

    for line_idx, level in zip(reversed(heading_positions), reversed(levels)):
        line = lines[line_idx]
        stripped = line.lstrip("#")
        if stripped and stripped[0] == " ":
            stripped = stripped[1:]
        lines[line_idx] = "#" * level + " " + stripped

    return "\n".join(lines)
