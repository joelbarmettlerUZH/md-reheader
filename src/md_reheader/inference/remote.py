import httpx

from md_reheader.data.apply import apply_levels
from md_reheader.data.extract import extract_headings
from md_reheader.data.format import SYSTEM_PROMPT, parse_levels_from_output
from md_reheader.data.strip import strip_document

DEFAULT_MODEL = "joelbarmettler/md-reheader"


def predict_heading_levels_remote(
    md_text: str,
    endpoint: str,
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    max_new_tokens: int = 1024,
    timeout: float = 300.0,
) -> list[int]:
    stripped = strip_document(md_text)

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": stripped},
        ],
        "max_tokens": max_new_tokens,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    url = endpoint.rstrip("/") + "/chat/completions"
    response = httpx.post(url, headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()

    data = response.json()
    content = data["choices"][0]["message"]["content"]
    return parse_levels_from_output(content)


def reheader_document_remote(
    md_text: str,
    endpoint: str,
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    max_new_tokens: int = 1024,
    timeout: float = 300.0,
) -> str:
    """Reheader a document via an OpenAI-compatible endpoint (e.g. vLLM)."""
    headings = extract_headings(md_text)
    if not headings:
        return md_text

    levels = predict_heading_levels_remote(
        md_text,
        endpoint=endpoint,
        api_key=api_key,
        model=model,
        max_new_tokens=max_new_tokens,
        timeout=timeout,
    )

    if len(levels) < len(headings):
        levels.extend([headings[i].level for i in range(len(levels), len(headings))])
    elif len(levels) > len(headings):
        levels = levels[: len(headings)]

    return apply_levels(md_text, levels)
