"""Pydantic v2 request/response models (validation is the first guardrail)."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

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
    created_at: datetime


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
