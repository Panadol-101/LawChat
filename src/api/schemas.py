from __future__ import annotations

from datetime import date, datetime
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field

from database import LegalStatus


class SupportingQuoteResponse(BaseModel):
    evidence_id: str
    quote: str


class ClaimResponse(BaseModel):
    text: str
    evidence_ids: list[str]
    supporting_quotes: list[SupportingQuoteResponse] = Field(default_factory=list)


class CitationResponse(BaseModel):
    evidence_id: str
    document_id: str
    chunk_id: str
    title: str
    document_number: str | None
    article: str | None
    clause: str | None
    point: str | None
    source_url: str
    as_of: date
    status: str
    status_scope: str
    version_id: str | None = None
    version_source_url: str | None = None
    version_source_revision: str | None = None
    content_valid_from: date | None = None
    content_valid_to: date | None = None


class TimingResponse(BaseModel):
    queue_wait: float = 0.0
    issue_decomposition: float = 0.0
    retrieval: float
    context: float
    generation_and_verification: float
    processing: float = 0.0
    total: float
    llm_ttft: float | None = None
    semantic_attempts: list["GenerationAttemptTelemetryResponse"] = Field(default_factory=list)
    issue_decomposition_attempt: "GenerationAttemptTelemetryResponse | None" = None
    generation_attempts: list["GenerationAttemptTelemetryResponse"] = Field(
        default_factory=list
    )
    stages: list["PipelineStageTimingResponse"] = Field(default_factory=list)


class PipelineStageTimingResponse(BaseModel):
    stage: str
    attempt: int
    seconds: float


class GenerationAttemptTelemetryResponse(BaseModel):
    total_seconds: float
    ttft_seconds: float | None
    prompt_tokens: int | None
    completion_tokens: int | None
    error_code: str | None = None


class VerificationIssueResponse(BaseModel):
    code: str
    message: str
    claim_index: int | None


class VerificationResponse(BaseModel):
    status: str
    issues: list[VerificationIssueResponse]


class RetrievalDiagnosticsResponse(BaseModel):
    sources: list[str]
    searched_candidates: int
    rejected_candidates: int
    rejection_reasons: dict[str, int] = Field(default_factory=dict)
    warnings: list[str]
    timings: dict[str, float]


class ContextDiagnosticsResponse(BaseModel):
    evidence_count: int
    used_tokens: int
    token_budget: int
    dropped_chunk_ids: list[str]
    issue_evidence_counts: dict[str, int] = Field(default_factory=dict)


class EvidenceResponse(CitationResponse):
    text: str
    role: str


class _AnswerResponseBase(BaseModel):
    request_id: str
    query: str
    # The LLM-rewritten standalone question, when the rewrite step ran.
    rewritten_query: str | None = None
    as_of: date
    status: str
    semantic_mode: Literal["off", "shadow", "enforce"] = "off"
    semantic_status: Literal["DISABLED", "NOT_CHECKED", "PASSED", "FLAGGED", "ERROR"] = "DISABLED"
    coverage_status: Literal["NOT_CHECKED", "COMPLETE", "INCOMPLETE", "ERROR"] = "NOT_CHECKED"
    attempts: int
    answer: str
    claims: list[ClaimResponse]
    limitations: list[str]
    confidence: str
    legal_issues: list["LegalIssueResponse"] = Field(default_factory=list)
    issue_resolutions: list[dict] = Field(default_factory=list)
    citations: list[CitationResponse]
    timing: TimingResponse


class LegalIssueResponse(BaseModel):
    issue_id: str
    question: str
    search_query: str


class CompactAnswerResponse(_AnswerResponseBase):
    mode: Literal["compact"] = "compact"


class VerboseAnswerResponse(_AnswerResponseBase):
    mode: Literal["verbose"] = "verbose"
    semantic_observations: list[dict] = Field(default_factory=list)
    verification: VerificationResponse
    retrieval: RetrievalDiagnosticsResponse
    context: ContextDiagnosticsResponse
    evidence: list[EvidenceResponse]


AnswerResponse = Annotated[
    CompactAnswerResponse | VerboseAnswerResponse,
    Field(discriminator="mode"),
]


class ProjectCreateBody(BaseModel):
    name: str = Field(min_length=1, max_length=255)


class ProjectUpdateBody(BaseModel):
    name: str = Field(min_length=1, max_length=255)


class ProjectResponse(BaseModel):
    id: UUID
    name: str
    is_shared: bool = False
    created_at: datetime
    updated_at: datetime


class ConversationCreateBody(BaseModel):
    title: str = Field(default="Cuộc trò chuyện mới", min_length=1, max_length=255)
    project_id: UUID | None = None


class ConversationUpdateBody(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=255)
    pinned: bool | None = None
    project_id: UUID | None = None
    update_project: bool = False


class ConversationResponse(BaseModel):
    id: UUID
    title: str
    project_id: UUID | None
    pinned: bool
    is_shared: bool = False
    created_at: datetime
    updated_at: datetime


class MessageResponse(BaseModel):
    id: UUID
    conversation_id: UUID
    client_message_id: str | None
    role: Literal["user", "assistant"]
    status: Literal["PENDING", "PROCESSING", "COMPLETED", "REFUSED", "FAILED"]
    content: str
    citations: list[dict[str, Any]] = Field(default_factory=list)
    claims: list[dict[str, Any]] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    as_of: date | None
    request_id: str | None
    semantic_status: str | None
    coverage_status: str | None
    confidence: str | None
    error_code: str | None
    created_at: datetime
    completed_at: datetime | None


class ChatBody(BaseModel):
    conversation_id: UUID
    client_message_id: str = Field(min_length=1, max_length=80)
    message: str = Field(min_length=1, max_length=20_000)
    as_of: date | None = None
    limit: int = Field(default=5, ge=1, le=20)
    candidate_limit: int = Field(default=50, ge=1, le=500)
    document_numbers: list[str] = Field(default_factory=list)
    document_types: list[str] = Field(default_factory=list)
    authorities: list[str] = Field(default_factory=list)
    legal_fields: list[str] = Field(default_factory=list)
    statuses: list[LegalStatus] = Field(default_factory=list)
    response_mode: Literal["compact"] = "compact"


class ChatResponse(BaseModel):
    conversation_id: UUID
    user_message_id: UUID
    assistant_message_id: UUID
    result: dict[str, Any]


class ShareResponse(BaseModel):
    token: str


class SharedConversationResponse(BaseModel):
    conversation: ConversationResponse
    messages: list[MessageResponse]
class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=100)
    password: str = Field(min_length=1, max_length=256)


class LoginResponse(BaseModel):
    requires_totp: bool
    message: str


class TOTPRequest(BaseModel):
    username: str = Field(min_length=1, max_length=100)
    password: str = Field(min_length=1, max_length=256)
    code: str = Field(min_length=6, max_length=6)


class UserResponse(BaseModel):
    id: UUID
    username: str
    role: str
    totp_enabled: bool

class UserUsageResponse(BaseModel):
    token_limit: int
    tokens_used: int
    remaining_tokens: int
    period_start: datetime
    period_end: datetime

class AdminUserUsageResponse(BaseModel):
    id: UUID
    username: str
    role: str
    is_active: bool
    token_limit: int | None = None
    tokens_used: int | None = None
    remaining_tokens: int | None = None
    period_start: datetime | None = None
    period_end: datetime | None = None

class AdminUsageUpdateRequest(BaseModel):
    token_limit: int = Field(
        ge=0,
        le=10_000_000_000,
    )

class RegisterRequest(BaseModel):
    username: str = Field(
        min_length=3,
        max_length=50,
        pattern=r"^[a-zA-Z0-9_]+$",
    )
    password: str = Field(min_length=8, max_length=128)
    confirm_password: str = Field(min_length=8, max_length=128)
