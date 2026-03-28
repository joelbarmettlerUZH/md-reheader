from typing import Annotated

from pydantic import BaseModel, Field


class Heading(BaseModel):
    text: Annotated[str, Field(description="The heading text without # prefix")]
    level: Annotated[int, Field(ge=1, le=6, description="Heading depth, 1 = top-level")]


class ChatMessage(BaseModel):
    role: Annotated[str, Field(description="One of: system, user, assistant")]
    content: Annotated[str, Field(description="Message content")]


class TrainingExample(BaseModel):
    messages: Annotated[list[ChatMessage], Field(description="ChatML conversation")]
    metadata: Annotated[dict, Field(default_factory=dict, description="Source and doc metadata")]


class GitHubCodeMeta(BaseModel):
    repo: Annotated[str, Field(description="GitHub repo name, e.g. owner/repo")]
    path: Annotated[str, Field(description="File path within the repo")]
    license: Annotated[str, Field(description="SPDX license identifier")]
    is_priority_path: Annotated[bool, Field(description="From docs/wiki/guide path")]
    max_level_gap: Annotated[int, Field(ge=0, description="Max gap between adjacent levels")]


class GoodwikiMeta(BaseModel):
    title: Annotated[str, Field(description="Wikipedia article title")]
    pageid: Annotated[int, Field(description="Wikipedia page ID")]
    categories: Annotated[list[str], Field(description="Wikipedia categories")]
    max_level_gap: Annotated[int, Field(ge=0, description="Max gap between adjacent levels")]


class RawDocument(BaseModel):
    content: Annotated[str, Field(description="Full markdown document text")]
    headings: Annotated[list[Heading], Field(description="Extracted headings with levels")]
    source: Annotated[str, Field(description="One of: github_code, goodwiki")]
    meta: Annotated[dict, Field(description="Source-specific metadata")]


class EvalResult(BaseModel):
    exact_match: Annotated[float, Field(ge=0, le=1)]
    per_heading_accuracy: Annotated[float, Field(ge=0, le=1)]
    hierarchy_preservation: Annotated[float, Field(ge=0, le=1)]
    mean_absolute_error: Annotated[float, Field(ge=0, description="Average absolute level error")]
    level_count_match: Annotated[bool, Field(description="Whether predicted count matches truth")]
    predicted_levels: Annotated[list[int], Field(description="Model-predicted heading levels")]
    ground_truth_levels: Annotated[list[int], Field(description="True heading levels")]
    metadata: Annotated[dict, Field(default_factory=dict, description="Source and doc metadata")]
