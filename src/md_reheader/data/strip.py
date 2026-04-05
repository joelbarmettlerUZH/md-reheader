import re

from markdown_it import MarkdownIt

from md_reheader.data.filter import _get_tokenizer, compute_token_count
from md_reheader.models import Heading

_md_parser = MarkdownIt("commonmark")


def _find_heading_spans(md_text: str) -> list[tuple[int, int, int, str]]:
    tokens = _md_parser.parse(md_text)
    spans: list[tuple[int, int, int, str]] = []
    for i, token in enumerate(tokens):
        if token.type == "heading_open":
            level = int(token.tag[1])
            inline_token = tokens[i + 1]
            if token.map is not None:
                line_start, line_end = token.map
                spans.append((line_start, line_end, level, inline_token.content))
    return spans


def _truncate_body(text: str, head_tokens: int, tail_tokens: int) -> str:
    max_tokens = head_tokens + tail_tokens
    token_count = compute_token_count(text)
    if token_count <= max_tokens:
        return text

    tok = _get_tokenizer()
    encoded = tok.encode(text)
    head_text = tok.decode(encoded[:head_tokens])
    tail_text = tok.decode(encoded[-tail_tokens:])
    return head_text + "\n[...]\n" + tail_text


def strip_document(
    md_text: str,
    headings: list[Heading] | None = None,
    head_tokens: int = 128,
    tail_tokens: int = 128,
) -> str:
    """Flatten all headings to `# ` and truncate body text between headings."""
    lines = md_text.split("\n")
    spans = _find_heading_spans(md_text)

    if not spans:
        return _truncate_body(md_text, head_tokens, tail_tokens)

    result_parts: list[str] = []

    for idx, (line_start, line_end, _level, heading_text) in enumerate(spans):
        if idx == 0:
            body_lines = lines[:line_start]
        else:
            prev_end = spans[idx - 1][1]
            body_lines = lines[prev_end:line_start]

        body_text = "\n".join(body_lines).strip()
        if body_text:
            truncated = _truncate_body(body_text, head_tokens, tail_tokens)
            result_parts.append(truncated)

        result_parts.append(f"# {heading_text}")

    last_end = spans[-1][1]
    trailing_text = "\n".join(lines[last_end:]).strip()
    if trailing_text:
        result_parts.append(_truncate_body(trailing_text, head_tokens, tail_tokens))

    return "\n\n".join(result_parts)


def count_headings(text: str) -> int:
    return len(re.findall(r"^# ", text, re.MULTILINE))


def truncate_stripped(text: str, max_tokens: int = 7500) -> str:
    """Truncate to fit within max_tokens, cutting at the last complete heading."""
    token_count = compute_token_count(text)
    if token_count <= max_tokens:
        return text

    tok = _get_tokenizer()
    encoded = tok.encode(text)
    truncated_text = tok.decode(encoded[:max_tokens])

    last_heading = truncated_text.rfind("\n# ")
    if last_heading == -1:
        return truncated_text

    next_newline = truncated_text.find("\n", last_heading + 1)
    if next_newline == -1:
        return truncated_text
    return truncated_text[:next_newline]
