"""Pydantic v2 request/response models (validation is the first guardrail)."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .settings import get_settings

SLUG = r"^[a-z0-9][a-z0-9_-]{1,62}$"


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    conversation_id: str | None = None
    collection: str = Field(default="cloud", pattern=SLUG)
    model: str | None = Field(
        default=None, max_length=200, description="model id from GET /models; default model if empty"
    )
    document_ids: list[str] | None = Field(
        default=None, max_length=500, description="search only these documents of the notebook (ticked sources)"
    )

    @field_validator("message")
    @classmethod
    def _length(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("empty message")
        if len(v) > get_settings().max_message_chars:
            raise ValueError(f"message longer than {get_settings().max_message_chars} characters")
        return v


class Citation(BaseModel):
    index: int
    document_id: str
    title: str = ""
    snippet: str
    score: float | None = None
    source_document: str | None = None
    page: int | None = None
    section: str | None = None


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    role: str
    content: str
    turn: int
    decision: str | None = None
    p_unanswerable: float | None = None
    latency_ms: int | None = None
    meta: dict = {}
    created_at: datetime


class ConversationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    title: str
    collection_id: str
    created_at: datetime
    updated_at: datetime


class ConversationDetail(ConversationOut):
    messages: list[MessageOut]
    agent_state: dict | None = None


class FeedbackIn(BaseModel):
    message_id: str
    rating: Literal[1, -1]
    reason: Literal["wrong_refusal", "wrong_answer", "bad_citation", "unclear", "other"] | None = None
    comment: str | None = Field(default=None, max_length=2000)


class FeedbackOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    message_id: str
    rating: int
    reason: str | None
    created_at: datetime


class CollectionIn(BaseModel):
    id: str = Field(pattern=SLUG, description="slug, e.g. 'hr-policies'")
    name: str = Field(min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=2000)
    is_public: bool = False


class CollectionUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=2000)


class CollectionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    name: str
    description: str | None
    is_public: bool
    created_at: datetime
    link_access: str = "none"
    role: str | None = None  # the caller's role: owner | editor | viewer | public
    shared: bool = False  # shared with at least one person or by link


class MemberIn(BaseModel):
    principal: str = Field(min_length=3, max_length=320, description="e-mail address or user id")
    role: Literal["viewer", "editor"] = "viewer"


class SharingOut(BaseModel):
    link_access: Literal["none", "viewer"]
    owner: str | None
    members: list[dict]


class SharingUpdate(BaseModel):
    link_access: Literal["none", "viewer"]


class DiscoverIn(BaseModel):
    query: str = Field(min_length=3, max_length=300)
    max_results: int = Field(default=8, ge=1, le=15)


class DiscoverHit(BaseModel):
    title: str
    url: str
    snippet: str = ""
    kind: Literal["url", "youtube"] = "url"


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    collection_id: str
    filename: str
    size_bytes: int
    status: str
    n_passages: int
    n_flagged: int
    error: str | None
    source_type: str = "file"
    source_url: str | None = None
    guide: dict | None = None
    guide_status: str = "none"
    created_at: datetime


class SourceIn(BaseModel):
    """A source that is not a file upload."""

    type: Literal["url", "youtube", "text"]
    url: str | None = Field(default=None, max_length=2000)
    title: str | None = Field(default=None, max_length=200)
    text: str | None = Field(default=None, max_length=2_000_000)
    model: str | None = Field(default=None, max_length=200, description="model for the source guide")

    @model_validator(mode="after")
    def _check(self):
        if self.type in ("url", "youtube") and not (self.url or "").strip():
            raise ValueError("url is required")
        if self.type == "text" and len((self.text or "").strip()) < 20:
            raise ValueError("paste at least a few sentences")
        return self


class PassageOut(BaseModel):
    passage_id: str
    text: str
    page: int | None = None
    section: str | None = None


class NoteIn(BaseModel):
    title: str = Field(default="Untitled note", min_length=1, max_length=300)
    content: str = Field(default="", max_length=50_000)
    kind: Literal["note", "answer"] = "note"
    source_message_id: str | None = None
    citations: list[dict] = Field(default_factory=list, max_length=50)
    pinned: bool = False


class NoteUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=300)
    content: str | None = Field(default=None, max_length=50_000)
    pinned: bool | None = None


class NoteOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    collection_id: str
    title: str
    content: str
    kind: str
    source_message_id: str | None
    citations: list[dict]
    pinned: bool
    created_at: datetime
    updated_at: datetime


class ModelOut(BaseModel):
    id: str
    provider: str
    provider_label: str
    model: str
    name: str
    context_length: int | None = None
    reasoning: bool = False
    featured: bool = False
    calibrated: bool = False
    description: str = ""
    calibration: dict | None = None


class ModelsOut(BaseModel):
    default: str
    models: list[ModelOut]
    providers: list[dict]


class ArtifactIn(BaseModel):
    kind: Literal["briefing", "study_guide", "faq", "timeline", "mindmap", "flashcards", "quiz", "audio"]
    document_ids: list[str] | None = Field(default=None, max_length=500, description="ticked sources (default: all)")
    model: str | None = Field(default=None, max_length=200)
    instructions: str | None = Field(default=None, max_length=500, description="e.g. 'focus on the methods'")


class ArtifactOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    collection_id: str
    kind: str
    title: str
    status: str
    content: str
    data: dict
    document_ids: list
    model: str | None
    instructions: str | None
    error: str | None
    created_at: datetime
    updated_at: datetime
