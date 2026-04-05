import re

from md_reheader.models import ChatMessage, Heading, TrainingExample

SYSTEM_PROMPT = (
    "You are a markdown document structure expert. Given a markdown document with "
    "incorrect or flattened heading levels, output each heading with its correct "
    "markdown prefix (# for level 1, ## for level 2, etc.), one per line."
)


def format_headings_output(headings: list[Heading], levels: list[int]) -> str:
    lines: list[str] = []
    for heading, level in zip(headings, levels):
        lines.append(f"{'#' * level} {heading.text}")
    return "\n".join(lines)


def format_training_example(
    stripped_md: str,
    headings: list[Heading],
    true_levels: list[int],
) -> TrainingExample:
    assistant_content = format_headings_output(headings, true_levels)
    return TrainingExample(
        messages=[
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            ChatMessage(role="user", content=stripped_md),
            ChatMessage(role="assistant", content=assistant_content),
        ]
    )


def parse_headings_output(text: str) -> list[Heading]:
    results: list[Heading] = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.match(r"^(#{1,6})\s+(.+)$", line)
        if match:
            level = len(match.group(1))
            heading_text = match.group(2).strip()
            results.append(Heading(text=heading_text, level=level))
    return results


def parse_levels_from_output(text: str) -> list[int]:
    return [h.level for h in parse_headings_output(text)]
