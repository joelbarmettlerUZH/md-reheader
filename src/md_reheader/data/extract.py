from markdown_it import MarkdownIt

from md_reheader.models import Heading

_md_parser = MarkdownIt("commonmark")


def extract_headings(md_text: str) -> list[Heading]:
    tokens = _md_parser.parse(md_text)
    headings: list[Heading] = []
    for i, token in enumerate(tokens):
        if token.type == "heading_open":
            level = int(token.tag[1])
            inline_token = tokens[i + 1]
            headings.append(Heading(text=inline_token.content, level=level))
    return headings
