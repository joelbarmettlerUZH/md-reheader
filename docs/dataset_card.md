---
license: cc-by-sa-4.0
language:
  - en
tags:
  - markdown
  - document-structure
  - heading-prediction
  - pdf-to-markdown
size_categories:
  - 100K<n<1M
task_categories:
  - text-generation
---

# md-reheader Dataset

Training data for [md-reheader](https://huggingface.co/joelbarmettler/md-reheader), a model that restores heading hierarchy in markdown documents.

## Dataset Description

Each example is a ChatML conversation with three messages:

1. **System:** Instructions for heading level prediction
2. **User:** A markdown document with all headings flattened to `# ` (level 1) and body text stripped to first 128 + last 128 tokens per section
3. **Assistant:** The correct headings with proper `#` prefixes (e.g., `## Methods`, `### Data`)

### Example

```json
{
  "messages": [
    {"role": "system", "content": "You are a markdown document structure expert..."},
    {"role": "user", "content": "# Introduction\n\nFirst 128 tokens...\n\n# Background\n\n..."},
    {"role": "assistant", "content": "# Introduction\n## Background"}
  ],
  "metadata": {
    "source": "goodwiki",
    "token_count": 1632,
    "heading_count": 9,
    "max_depth": 3
  }
}
```

## Dataset Structure

| Split | Examples | Notes |
|-------|---------|-------|
| train | 196,949 | Includes 2-8x oversampling of deep-structure documents |
| validation | 7,321 | |
| test | 7,321 | |

### Metadata fields

- `source`: `github_code` or `goodwiki`
- `token_count`: Token count of the stripped document (Qwen3 tokenizer)
- `heading_count`: Number of headings in the document
- `max_depth`: Maximum heading depth (1-6)
- `min_depth`: Minimum heading depth
- `max_level_gap`: Largest gap between adjacent heading levels

## Sources

### codeparrot/github-code (~105k documents)

Markdown files from GitHub repositories, loaded from the [codeparrot/github-code](https://huggingface.co/datasets/codeparrot/github-code) dataset. Filtered by `.md`/`.markdown` file extension. Bucket-targeted download ensures representation across document lengths (< 4k, 4k-8k, 8k-16k, 16k-32k characters).

Split by repository name to prevent data leakage.

### euirim/goodwiki (~45k documents)

High-quality Wikipedia articles from [euirim/goodwiki](https://huggingface.co/datasets/euirim/goodwiki). Article titles prepended as `# {title}` since Wikipedia's title serves as the H1 heading.

Split by article title to prevent data leakage.

## Processing Pipeline

1. **Download** raw markdown documents from HuggingFace
2. **Filter** by character length (300-150k), heading count (3-120), distinct levels (≥2), token count (100-31k)
3. **Split** by source key (repo name / article title) into 90% train / 5% val / 5% test
4. **Oversample** deep-structure documents in the training split (depth 4: 2x, depth 5: 4x, depth 6: 8x)
5. **Strip** body text to first 128 + last 128 tokens per section
6. **Flatten** all headings to `# ` (level 1)
7. **Truncate** documents exceeding 7,500 tokens
8. **Format** as ChatML conversations

## Heading Level Distribution (training set)

| Level | Count | Percentage |
|-------|-------|-----------|
| H1 | 211,673 | 8.5% |
| H2 | 911,437 | 36.8% |
| H3 | 922,695 | 37.3% |
| H4 | 335,052 | 13.5% |
| H5 | 78,761 | 3.2% |
| H6 | 16,345 | 0.7% |

## License

This dataset is released under **CC-BY-SA 4.0** because it contains derivative content from Wikipedia.

- Wikipedia content: [CC-BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) by Wikimedia contributors
- GitHub code files: various permissive licenses (MIT, Apache 2.0, BSD, etc.) via [codeparrot/github-code](https://huggingface.co/datasets/codeparrot/github-code)
- GoodWiki dataset: [CC-BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) by Euirim Choi

## Author

Built by [Joel Barmettler](https://joelbarmettler.xyz/).

## Citation

```bibtex
@software{barmettler2026mdreheader,
  author = {Barmettler, Joel},
  title = {md-reheader: Restoring Heading Hierarchy in Markdown Documents},
  year = {2026},
  url = {https://github.com/joelbarmettlerUZH/md-reheader}
}
```
