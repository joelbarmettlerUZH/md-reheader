import re

TOP_LEVEL_PATTERNS = [
    "introduction",
    "conclusion",
    "abstract",
    "references",
    "appendix",
    "acknowledgments",
    "background",
    "methods",
    "results",
    "discussion",
    "overview",
    "summary",
    "table of contents",
    "getting started",
    "installation",
    "usage",
    "license",
    "contributing",
]


def naive_flat_baseline(headings: list[str]) -> list[int]:
    return [1] * len(headings)


def heuristic_baseline(headings: list[str]) -> list[int]:
    levels: list[int] = []
    for i, heading in enumerate(headings):
        heading_lower = heading.lower().strip()

        if i == 0:
            levels.append(1)
            continue

        if heading_lower in TOP_LEVEL_PATTERNS:
            levels.append(1)
            continue

        numbering_match = re.match(r"^(\d+\.)+\d*\s", heading)
        if numbering_match:
            depth = numbering_match.group(0).count(".")
            levels.append(min(depth + 1, 6))
            continue

        levels.append(2)

    return levels
