"""Pydantic v2 schemas for DoubtHub request validation and API response serialization."""

import re
from typing import Any, List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator

_COURSE_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
_EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class DoubtCreate(BaseModel):
    """Validated payload for creating and clustering a new student doubt."""

    model_config = ConfigDict(str_strip_whitespace=True)

    course_id: str = Field(..., min_length=2, max_length=20, description="Course code, e.g. UMA023")
    sheet_no: int = Field(..., ge=1, le=100, description="Tutorial sheet number (1-100)")
    question_no: str = Field(..., min_length=1, max_length=30, description="Question identifier, e.g. 4b")
    doubt_text: str = Field(..., min_length=3, max_length=2000, description="Detailed description of the roadblock")
    ai_failure_reason: str = Field(..., min_length=3, max_length=1000, description="Where generic AI explanations failed")

    @field_validator("course_id")
    @classmethod
    def validate_course_id(cls, value: str) -> str:
        """Normalize course code to uppercase and enforce alphanumeric/hyphen/underscore format."""
        normalized = value.strip().upper()
        if not _COURSE_PATTERN.match(normalized):
            raise ValueError("course_id may only contain letters, numbers, hyphens, and underscores")
        return normalized

    @field_validator("question_no")
    @classmethod
    def normalize_question_no(cls, value: str) -> str:
        """Normalize question identifier to trimmed lowercase."""
        normalized = value.strip().lower()
        if not normalized:
            raise ValueError("question_no cannot be blank")
        return normalized

    @field_validator("doubt_text", "ai_failure_reason")
    @classmethod
    def ensure_non_blank(cls, value: str) -> str:
        """Reject whitespace-only text inputs."""
        if not value.strip():
            raise ValueError("Field cannot be empty or whitespace-only")
        return value.strip()


class DoubtOut(BaseModel):
    """Serialized representation of a stored doubt row."""

    id: int
    course_id: str
    sheet_no: int
    question_no: str
    doubt_text: str
    ai_failure_reason: str
    cluster_id: Optional[int] = None
    created_at: str


class DoubtBriefOut(BaseModel):
    """Concise doubt summary embedded inside TA/student cluster cards."""

    doubt_text: str
    ai_failure_reason: str


class DoubtCreateResponse(BaseModel):
    """Response returned after logging and clustering a doubt."""

    status: str = "success"
    cluster_id: int = Field(..., ge=1)


class ClusterOut(BaseModel):
    """Serialized representation of a roadblock cluster with nested student breakdowns."""

    id: int
    course_id: str
    sheet_no: int
    question_no: str
    title: str
    ai_summary: Optional[str] = None
    upvotes: int = Field(default=1, ge=0)
    is_resolved: int = Field(default=0, ge=0, le=1)
    ta_response: Optional[str] = None
    doubts: List[DoubtBriefOut] = Field(default_factory=list)


class TALoginRequest(BaseModel):
    """Credentials payload for TA deadlock portal authentication."""

    model_config = ConfigDict(str_strip_whitespace=True)

    email: str = Field(..., min_length=5, max_length=120)
    password: str = Field(..., min_length=1, max_length=128)

    @field_validator("email")
    @classmethod
    def validate_email_format(cls, value: str) -> str:
        """Ensure provided university email follows basic email syntax."""
        normalized = value.strip().lower()
        if not _EMAIL_PATTERN.match(normalized):
            raise ValueError("Invalid email address format")
        return normalized


class TALoginResponse(BaseModel):
    """Response returned on successful TA login."""

    status: str = "success"
    token: str


class TAResponseInput(BaseModel):
    """Payload for publishing or updating an official TA clarification."""

    model_config = ConfigDict(str_strip_whitespace=True)

    response_text: str = Field(..., min_length=1, max_length=4000)
    mark_resolved: bool = False

    @field_validator("response_text")
    @classmethod
    def ensure_non_empty_response(cls, value: str) -> str:
        """Reject empty TA responses."""
        if not value.strip():
            raise ValueError("response_text cannot be empty")
        return value.strip()


class CommunitySolutionOut(BaseModel):
    """Public peer solution payload (never includes ownership manage_token)."""

    id: int
    cluster_id: int
    author_name: str = Field(default="Anonymous Hosteller", min_length=1, max_length=80)
    note_text: Optional[str] = Field(default=None, max_length=2000)
    image_filename: Optional[str] = None
    upvotes: int = Field(default=0, ge=0)
    created_at: str


class CommunitySolutionCreateResponse(CommunitySolutionOut):
    """Creation response including the creator's device-scoped manage_token."""

    manage_token: str


class SolutionUpdateInput(BaseModel):
    """Payload for editing an owned peer solution's note text."""

    model_config = ConfigDict(str_strip_whitespace=True)

    note_text: str = Field(..., min_length=1, max_length=2000)


class StatusResponse(BaseModel):
    """Standard success status envelope."""

    status: str = "success"


class UpvoteSolutionResponse(BaseModel):
    """Response returned when upvoting a community solution."""

    status: str = "success"
    upvotes: int = Field(..., ge=0)


class SynthesisResponse(BaseModel):
    """AI-generated TA blackboard guide response."""

    summary: str


class ErrorEnvelope(BaseModel):
    """Standardized JSON error response format."""

    status: str = "error"
    error: str
    detail: Any = None

