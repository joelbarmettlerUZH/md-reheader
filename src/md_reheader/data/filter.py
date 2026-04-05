from transformers import AutoTokenizer, PreTrainedTokenizerBase

from md_reheader.models import Heading

_tokenizer: PreTrainedTokenizerBase | None = None


def _get_tokenizer() -> PreTrainedTokenizerBase:
    global _tokenizer  # noqa: PLW0603
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    return _tokenizer


def passes_cheap_filters(
    md_text: str,
    headings: list[Heading],
    min_headings: int = 3,
    min_distinct_levels: int = 2,
    min_char_length: int = 300,
    max_char_length: int = 150_000,
    max_headings: int = 120,
) -> bool:
    if len(md_text) < min_char_length or len(md_text) > max_char_length:
        return False
    if len(headings) < min_headings:
        return False
    if len({h.level for h in headings}) < min_distinct_levels:
        return False
    if len(headings) > max_headings:
        return False
    return True


def passes_token_filter(
    md_text: str,
    min_tokens: int = 100,
    max_tokens: int = 31_000,
) -> bool:
    token_count = compute_token_count(md_text)
    return min_tokens <= token_count <= max_tokens


def compute_heading_level_gap(headings: list[Heading]) -> int:
    levels = sorted({h.level for h in headings})
    if len(levels) < 2:
        return 0
    return max(levels[i] - levels[i - 1] for i in range(1, len(levels)))


def compute_token_count(md_text: str) -> int:
    return len(_get_tokenizer().encode(md_text))
