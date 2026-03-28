from typing import Annotated

from pydantic import BaseModel, Field


class Heading(BaseModel):
    text: Annotated[str, Field(description="The heading text without # prefix")]
    level: Annotated[int, Field(ge=1, le=6, description="Heading depth, 1 = top-level")]


class ChatMessage(BaseModel):
    role: Annotated[str, Field(description="One of: system, user, assistant")]
    content: str


class TrainingExample(BaseModel):
    messages: list[ChatMessage]
    metadata: Annotated[dict, Field(default_factory=dict)]


class GitHubCodeMeta(BaseModel):
    repo: str
    path: str
    license: str
    is_priority_path: bool
    max_level_gap: Annotated[int, Field(ge=0)]


class GoodwikiMeta(BaseModel):
    title: str
    pageid: int
    categories: list[str]
    max_level_gap: Annotated[int, Field(ge=0)]


class RawDocument(BaseModel):
    content: str
    headings: list[Heading]
    source: Annotated[str, Field(description="One of: github_code, goodwiki")]
    meta: Annotated[dict, Field(description="Source-specific metadata")]


class EvalResult(BaseModel):
    exact_match: Annotated[float, Field(ge=0, le=1)]
    per_heading_accuracy: Annotated[float, Field(ge=0, le=1)]
    hierarchy_preservation: Annotated[float, Field(ge=0, le=1)]
    mean_absolute_error: float
    level_count_match: bool
    predicted_levels: list[int]
    ground_truth_levels: list[int]
    metadata: Annotated[dict, Field(default_factory=dict)]
