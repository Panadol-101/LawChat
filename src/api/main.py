from __future__ import annotations

import asyncio
import json
import os
import re
from contextlib import asynccontextmanager
from contextlib import suppress
from dataclasses import asdict, replace
from datetime import date
from time import perf_counter
from typing import Annotated, Any, Literal
from uuid import UUID, uuid4
COOKIE_SECURE = os.getenv("LAWCHAT_COOKIE_SECURE", "true").lower() == "true"
from fastapi import (
    Cookie,
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Query,
    Request,
    Response,
)
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import text
from starlette.concurrency import run_in_threadpool
from starlette.responses import RedirectResponse, StreamingResponse

from chat import ChatNotFoundError, ChatRepository
from database import (
    DatabaseSettings,
    LegalStatus,
    create_db_engine,
    create_session_factory,
)
from auth.service import (
    authenticate_user,
    create_session,
    get_user_by_session,
    register_user,
    revoke_session,
    verify_user_totp,
    write_audit_log,
)
from indexing import QdrantSettings, SentenceTransformerEmbedder
from rag import (
    GenerationRequest,
    IssueDecomposer,
    IssueDecompositionError,
    IssueDecompositionRequest,
    IssuePlan,
    QueryRewrite,
    QueryRewriteRequest,
    RAGQueueFullError,
    RAGRuntime,
    create_rag_runtime,
    history_for_rewrite,
    requires_case_specific_prediction_refusal,
    retrieve_issue_plan,
    retrieve_issue_plan_async,
)
from retrieval import (
    LegalQueryParser,
    RetrievalRequest,
    RetrievalResponse,
    UnsupportedAsOfDate,
)
from retrieval.query_parser import ensure_supported_as_of
from retrieval.runtime import create_hybrid_retrieval_service
from sources import (
    DocumentNotFoundError,
    SourceResolutionService,
    create_source_resolution_service,
)

from .schemas import (
    AnswerResponse,
    ChatBody,
    ChatResponse,
    CompactAnswerResponse,
    ConversationCreateBody,
    ConversationResponse,
    ConversationUpdateBody,
    MessageResponse,
    ProjectCreateBody,
    ProjectResponse,
    ProjectUpdateBody,
    ShareResponse,
    SharedConversationResponse,
    VerboseAnswerResponse,
    LoginRequest,
    LoginResponse,
    RegisterRequest,
    TOTPRequest,
    UserResponse,
    UserUsageResponse,
    AdminUserUsageResponse,
    AdminUsageUpdateRequest,
)
from .logging import configure_logging, log_event


class SearchBody(BaseModel):
    query: str = Field(min_length=1)
    as_of: date | None = None
    limit: int = Field(default=10, ge=1, le=50)
    candidate_limit: int = Field(default=50, ge=1, le=500)
    document_numbers: list[str] = Field(default_factory=list)
    document_types: list[str] = Field(default_factory=list)
    authorities: list[str] = Field(default_factory=list)
    legal_fields: list[str] = Field(default_factory=list)
    statuses: list[LegalStatus] = Field(default_factory=list)


class AnswerBody(SearchBody):
    limit: int = Field(default=5, ge=1, le=20)
    response_mode: Literal["compact", "verbose"] = "compact"


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    db_engine = create_db_engine(DatabaseSettings.from_env())
    qdrant_settings = QdrantSettings.from_env()
    qdrant_client = qdrant_settings.create_client()
    app.state.db_engine = db_engine
    app.state.qdrant_client = qdrant_client
    app.state.qdrant_settings = qdrant_settings
    app.state.retrieval_service = create_hybrid_retrieval_service(
        db_engine,
        qdrant_client,
        qdrant_settings,
        SentenceTransformerEmbedder.from_env(),
    )
    app.state.rag_runtime = create_rag_runtime()
    session_factory = create_session_factory(db_engine)
    app.state.session_factory = session_factory
    app.state.chat_repository = ChatRepository(session_factory)
    app.state.source_resolution_service = create_source_resolution_service(
        session_factory
    )
    if os.getenv("LAWCHAT_LLM_WARMUP_ENABLED", "false").casefold() in {
        "1", "true", "yes", "on"
    }:
        await asyncio.to_thread(app.state.rag_runtime.llm_health.warmup)
    try:
        yield
    finally:
        qdrant_client.close()
        db_engine.dispose()


app = FastAPI(title="LawChat API", version="0.4.0", lifespan=lifespan)

_cors_origins = [
    item.strip()
    for item in os.getenv(
        "LAWCHAT_CORS_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501"
    ).split(",")
    if item.strip()
]
if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "X-Workspace-ID"],
    )


def get_retrieval_service() -> Any:
    return app.state.retrieval_service


def get_rag_runtime() -> RAGRuntime:
    return app.state.rag_runtime


def get_chat_repository() -> ChatRepository:
    return app.state.chat_repository


def get_source_resolution_service() -> SourceResolutionService:
    return app.state.source_resolution_service

def get_current_user(
    request: Request,
) -> Any:
    session_token = request.cookies.get("lawchat_session")

    if not session_token:
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
        )

    session_factory = app.state.session_factory

    with session_factory() as db:
        user = get_user_by_session(db, session_token)

    if user is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired session",
        )

    return user


CurrentUserDependency = Annotated[Any, Depends(get_current_user)]


def require_admin(
    user: CurrentUserDependency,
) -> Any:
    if user.role != "ADMIN":
        raise HTTPException(
            status_code=403,
            detail="Administrator privileges required",
        )

    return user


AdminUserDependency = Annotated[Any, Depends(require_admin)]

@app.get("/api/v1/admin/ping")
def admin_ping(
    user: AdminUserDependency,
):
    return {
        "ok": True,
        "message": "Admin access granted",
        "username": user.username,
        "role": user.role,
    }

@app.get("/api/v1/admin/hydration-stats")
def admin_hydration_stats(
    user: AdminUserDependency,
):
    """Aggregate hydration rejection counts grouped by reason."""
    session_factory = app.state.session_factory
    with session_factory() as db:
        rows = db.execute(
            text(
                """
                SELECT es.status, count(*) AS cnt
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                JOIN effective_status es ON es.document_id = d.id
                WHERE c.is_indexable
                  AND es.valid_period @> CURRENT_DATE
                GROUP BY es.status
                """
            )
        ).mappings().all()
    return {
        "by_status": {row["status"]: int(row["cnt"]) for row in rows},
        "label": "active valid-period effective_status counts",
    }


@app.get(
    "/api/v1/admin/users",
    response_model=list[AdminUserUsageResponse],
)
def admin_list_users(
    user: AdminUserDependency,
):
    session_factory = app.state.session_factory

    with session_factory() as db:
        result = db.execute(
            text(
                """
                SELECT
                    u.id,
                    u.username,
                    u.role,
                    u.is_active,
                    uu.token_limit,
                    uu.tokens_used,
                    uu.period_start,
                    uu.period_end
                FROM users u
                LEFT JOIN user_usage uu
                    ON uu.user_id = u.id
                ORDER BY u.username
                """
            )
        ).mappings().all()

        users = []

        for row in result:
            token_limit = (
                int(row["token_limit"])
                if row["token_limit"] is not None
                else None
            )

            tokens_used = (
                int(row["tokens_used"])
                if row["tokens_used"] is not None
                else None
            )

            remaining_tokens = (
                max(token_limit - tokens_used, 0)
                if token_limit is not None and tokens_used is not None
                else None
            )

            users.append(
                AdminUserUsageResponse(
                    id=row["id"],
                    username=row["username"],
                    role=row["role"],
                    is_active=row["is_active"],
                    token_limit=token_limit,
                    tokens_used=tokens_used,
                    remaining_tokens=remaining_tokens,
                    period_start=row["period_start"],
                    period_end=row["period_end"],
                )
            )

        return users

@app.patch(
    "/api/v1/admin/users/{user_id}/usage",
    response_model=AdminUserUsageResponse,
)
def admin_update_user_usage(
    user_id: UUID,
    body: AdminUsageUpdateRequest,
    user: AdminUserDependency,
):
    session_factory = app.state.session_factory

    with session_factory() as db:
        result = db.execute(
            text(
                """
                UPDATE user_usage
                SET
                    token_limit = :token_limit,
                    updated_at = CURRENT_TIMESTAMP
                WHERE user_id = :user_id
                RETURNING
                    token_limit,
                    tokens_used,
                    period_start,
                    period_end
                """
            ),
            {
                "user_id": user_id,
                "token_limit": body.token_limit,
            },
        ).mappings().first()

        if result is None:
            raise HTTPException(
                status_code=404,
                detail="User usage quota not found",
            )

        user_result = db.execute(
            text(
                """
                SELECT
                    id,
                    username,
                    role,
                    is_active
                FROM users
                WHERE id = :user_id
                """
            ),
            {
                "user_id": user_id,
            },
        ).mappings().first()

        if user_result is None:
            raise HTTPException(
                status_code=404,
                detail="User not found",
            )

        db.commit()

        token_limit = int(result["token_limit"])
        tokens_used = int(result["tokens_used"])

        return AdminUserUsageResponse(
            id=user_result["id"],
            username=user_result["username"],
            role=user_result["role"],
            is_active=user_result["is_active"],
            token_limit=token_limit,
            tokens_used=tokens_used,
            remaining_tokens=max(
                token_limit - tokens_used,
                0,
            ),
            period_start=result["period_start"],
            period_end=result["period_end"],
        )

def get_workspace_id(
    x_workspace_id: Annotated[str | None, Header()] = None,
) -> str:
    workspace_id = (x_workspace_id or "local").strip()
    if not workspace_id or len(workspace_id) > 100 or not all(
        char.isalnum() or char in "-_." for char in workspace_id
    ):
        raise HTTPException(status_code=400, detail="Invalid X-Workspace-ID")
    return workspace_id


RetrievalServiceDependency = Annotated[Any, Depends(get_retrieval_service)]
RAGRuntimeDependency = Annotated[RAGRuntime, Depends(get_rag_runtime)]
ChatRepositoryDependency = Annotated[ChatRepository, Depends(get_chat_repository)]
WorkspaceDependency = Annotated[str, Depends(get_workspace_id)]
SourceResolutionDependency = Annotated[
    SourceResolutionService, Depends(get_source_resolution_service)
]


@app.post(
    "/api/v1/auth/login",
    response_model=LoginResponse,
)
def auth_login(
    body: LoginRequest,
    request: Request,
    response: Response,
):
    session_factory = app.state.session_factory

    with session_factory() as db:
        user = authenticate_user(
            db,
            body.username,
            body.password,
        )

        if user is None:
            write_audit_log(
                db,
                event="login_failed",
                ip_address=request.client.host if request.client else None,
                user_agent=request.headers.get("user-agent"),
            )

            raise HTTPException(
                status_code=401,
                detail="Invalid username or password",
            )

        write_audit_log(
            db,
            event="login_password_verified",
            user_id=user.id,
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
        )

        # ADMIN: password verified, but TOTP is still required.
        if user.totp_enabled:
            return LoginResponse(
                requires_totp=True,
                message="Password verified. TOTP required.",
            )

        # USER: password is sufficient, create session immediately.
        session_token = create_session(db, user)

        response.set_cookie(
            key="lawchat_session",
            value=session_token,
            httponly=True,
            secure=COOKIE_SECURE,
            samesite="lax",
            max_age=12 * 60 * 60,
            path="/",
        )

        write_audit_log(
            db,
            event="login_success",
            user_id=user.id,
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
        )

        return LoginResponse(
            requires_totp=False,
            message="Login successful.",
        )

@app.post(
    "/api/v1/auth/totp",
    response_model=UserResponse,
)
def auth_totp(
    body: TOTPRequest,
    request: Request,
    response: Response,
):
    session_factory = app.state.session_factory

    with session_factory() as db:
        # Step 1: verify username + password again
        user = authenticate_user(
            db,
            body.username,
            body.password,
        )

        if user is None:
            write_audit_log(
                db,
                event="login_failed",
                ip_address=request.client.host if request.client else None,
                user_agent=request.headers.get("user-agent"),
            )

            raise HTTPException(
                status_code=401,
                detail="Invalid username or password",
            )

        # Step 2: verify TOTP
        if not verify_user_totp(user, body.code):
            write_audit_log(
                db,
                event="totp_failed",
                user_id=user.id,
                ip_address=request.client.host if request.client else None,
                user_agent=request.headers.get("user-agent"),
            )

            raise HTTPException(
                status_code=401,
                detail="Invalid authentication code",
            )

        # Step 3: create authenticated session
        session_token = create_session(db, user)

        response.set_cookie(
            key="lawchat_session",
            value=session_token,
            httponly=True,
            secure=True,
            samesite="lax",
            max_age=12 * 60 * 60,
            path="/",
        )

        write_audit_log(
            db,
            event="login_success",
            user_id=user.id,
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
        )

        return UserResponse(
            id=user.id,
            username=user.username,
            role=user.role,
            totp_enabled=user.totp_enabled,
        )

@app.post(
    "/api/v1/auth/register",
    response_model=UserResponse,
)
def auth_register(
    body: RegisterRequest,
    request: Request,
):
    if body.password != body.confirm_password:
        raise HTTPException(
            status_code=400,
            detail="Passwords do not match",
        )

    session_factory = app.state.session_factory

    with session_factory() as db:
        try:
            user = register_user(
                db,
                body.username,
                body.password,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

        write_audit_log(
            db,
            event="register_success",
            user_id=user.id,
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
        )

        return UserResponse(
            id=user.id,
            username=user.username,
            role=user.role,
            totp_enabled=user.totp_enabled,
        )


@app.get(
    "/api/v1/auth/me",
    response_model=UserResponse,
)
def auth_me(
    user: CurrentUserDependency,
):
    return UserResponse(
        id=user.id,
        username=user.username,
        role=user.role,
        totp_enabled=user.totp_enabled,
    )

@app.get(
    "/api/v1/auth/usage",
    response_model=UserUsageResponse,
)
def auth_usage(
    user: CurrentUserDependency,
):
    session_factory = app.state.session_factory

    with session_factory() as db:
        result = db.execute(
            text(
                """
                SELECT
                    token_limit,
                    tokens_used,
                    period_start,
                    period_end
                FROM user_usage
                WHERE user_id = :user_id
                """
            ),
            {
                "user_id": user.id,
            },
        ).mappings().first()

        if result is None:
            raise HTTPException(
                status_code=404,
                detail="Usage quota not found",
            )

        token_limit = int(result["token_limit"])
        tokens_used = int(result["tokens_used"])

        return UserUsageResponse(
            token_limit=token_limit,
            tokens_used=tokens_used,
            remaining_tokens=max(
                token_limit - tokens_used,
                0,
            ),
            period_start=result["period_start"],
            period_end=result["period_end"],
        )

@app.post("/api/v1/auth/logout")
def auth_logout(
    request: Request,
    response: Response,
):
    session_token = request.cookies.get("lawchat_session")

    if session_token:
        session_factory = app.state.session_factory

        with session_factory() as db:
            revoke_session(db, session_token)

    response.delete_cookie(
        key="lawchat_session",
        path="/",
    )

    return {"message": "Logged out"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, str]:
    try:
        with app.state.db_engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        settings = app.state.qdrant_settings
        alias_target = next(
            (
                item.collection_name
                for item in app.state.qdrant_client.get_aliases().aliases
                if item.alias_name == settings.alias
            ),
            None,
        )
        if alias_target != settings.collection:
            raise RuntimeError(
                f"alias {settings.alias!r} targets {alias_target!r}, "
                f"expected {settings.collection!r}"
            )
        if app.state.rag_runtime.llm_health is not None:
            app.state.rag_runtime.llm_health.check()
    except Exception as exc:
        log_event(
            "readiness_failed",
            dependency="postgres_qdrant_or_llm",
            error_type=type(exc).__name__,
        )
        raise HTTPException(
            status_code=503,
            detail="One or more runtime dependencies are unavailable",
        ) from exc
    return {"status": "ready"}


@app.post("/api/v1/search")
def search(
    body: SearchBody,
    service: RetrievalServiceDependency,
    _admin: AdminUserDependency,
):
    _validate_body(body)
    try:
        response = service.retrieve(_to_retrieval_request(body))
    except UnsupportedAsOfDate as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return jsonable_encoder(asdict(response))


@app.get("/api/v1/documents/{document_id}/source")
def document_source(
    document_id: str,
    service: SourceResolutionDependency,
):
    try:
        return jsonable_encoder(service.describe(document_id))
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/v1/documents/{document_id}/open-source")
async def open_document_source(
    document_id: str,
    service: SourceResolutionDependency,
):
    try:
        resolution = await service.resolve(document_id)
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    log_event(
        "document_source_resolved",
        document_id=str(document_id),
        status=resolution.status,
        cached=resolution.cached,
    )
    return RedirectResponse(
        resolution.url,
        status_code=302,
        headers={"X-LawChat-Source-Resolution": resolution.status},
    )


@app.post("/api/v1/projects", response_model=ProjectResponse, status_code=201)
def create_project(
    current_user: CurrentUserDependency,
    body: ProjectCreateBody,
    repository: ChatRepositoryDependency,
    workspace_id: WorkspaceDependency,
):
    return _project_payload(repository.create_project(current_user.id, workspace_id, body.name))


@app.get("/api/v1/projects", response_model=list[ProjectResponse])
def list_projects(
    current_user: CurrentUserDependency,
    repository: ChatRepositoryDependency,
    workspace_id: WorkspaceDependency,
):
    return [_project_payload(item) for item in repository.list_projects(current_user.id, workspace_id)]


@app.patch("/api/v1/projects/{project_id}", response_model=ProjectResponse)
def update_project(
    project_id: UUID,
    current_user: CurrentUserDependency,
    body: ProjectUpdateBody,
    repository: ChatRepositoryDependency,
    workspace_id: WorkspaceDependency,
):
    try:
        return _project_payload(repository.update_project( current_user.id, workspace_id, project_id, name=body.name,))
    except ChatNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.delete("/api/v1/projects/{project_id}", status_code=204)
def delete_project(
    project_id: UUID,
    current_user: CurrentUserDependency,
    repository: ChatRepositoryDependency,
    workspace_id: WorkspaceDependency,
):
    try:
        repository.delete_project(current_user.id, workspace_id, project_id,)
    except ChatNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return Response(status_code=204)


@app.post(
    "/api/v1/conversations", response_model=ConversationResponse, status_code=201
)
def create_conversation(
    body: ConversationCreateBody,
    current_user: CurrentUserDependency,
    repository: ChatRepositoryDependency,
    workspace_id: WorkspaceDependency,
):
    try:
        item = repository.create_conversation(
                   current_user.id,    
                   workspace_id,
                   body.title,
                   project_id=body.project_id,
        )
    except ChatNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _conversation_payload(item)


@app.get("/api/v1/conversations", response_model=list[ConversationResponse])
def list_conversations(
    repository: ChatRepositoryDependency,
    current_user: CurrentUserDependency,
    workspace_id: WorkspaceDependency,
    project_id: UUID | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
):
    return [
        _conversation_payload(item)
        for item in repository.list_conversations(
            current_user.id, workspace_id, project_id=project_id, limit=limit
        )
    ]


@app.get("/api/v1/conversations/{conversation_id}", response_model=ConversationResponse)
def get_conversation(
    conversation_id: UUID,
    current_user: CurrentUserDependency,
    repository: ChatRepositoryDependency,
    workspace_id: WorkspaceDependency,
):
    try:
        return _conversation_payload(
            repository.get_conversation(current_user.id, workspace_id, conversation_id)
        )
    except ChatNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.patch("/api/v1/conversations/{conversation_id}", response_model=ConversationResponse)
def update_conversation(
    conversation_id: UUID,
    current_user: CurrentUserDependency,
    body: ConversationUpdateBody,
    repository: ChatRepositoryDependency,
    workspace_id: WorkspaceDependency,
):
    updates: dict[str, Any] = {"title": body.title, "pinned": body.pinned}
    if body.update_project:
        updates["project_id"] = body.project_id
    try:
        return _conversation_payload(repository.update_conversation(
           current_user.id, workspace_id, conversation_id, **updates
        ))
    except ChatNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.delete("/api/v1/conversations/{conversation_id}", status_code=204)
def delete_conversation(
    conversation_id: UUID,
    current_user: CurrentUserDependency,
    repository: ChatRepositoryDependency,
    workspace_id: WorkspaceDependency,
):
    try:
        repository.delete_conversation(current_user.id, workspace_id, conversation_id)
    except ChatNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return Response(status_code=204)


@app.get(
    "/api/v1/conversations/{conversation_id}/messages",
    response_model=list[MessageResponse],
)
def list_messages(
    conversation_id: UUID,
    current_user: CurrentUserDependency,
    repository: ChatRepositoryDependency,
    workspace_id: WorkspaceDependency,
):
    try:
        return [
            _message_payload(item)
            for item in repository.list_messages(current_user.id, workspace_id, conversation_id)
        ]
    except ChatNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post(
    "/api/v1/conversations/{conversation_id}/share",
    response_model=ShareResponse,
)
def enable_conversation_share(
    conversation_id: UUID,
    current_user: CurrentUserDependency,
    repository: ChatRepositoryDependency,
    workspace_id: WorkspaceDependency,
):
    try:
        conversation = repository.enable_conversation_share(
            current_user.id, workspace_id, conversation_id
        )
    except ChatNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"token": conversation.share_token}


@app.delete("/api/v1/conversations/{conversation_id}/share", status_code=204)
def disable_conversation_share(
    conversation_id: UUID,
    current_user: CurrentUserDependency,
    repository: ChatRepositoryDependency,
    workspace_id: WorkspaceDependency,
):
    try:
        repository.disable_conversation_share(current_user.id, workspace_id, conversation_id)
    except ChatNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return Response(status_code=204)


@app.get(
    "/api/v1/shared/conversations/{token}",
    response_model=SharedConversationResponse,
)
def get_shared_conversation(
    token: str,
    repository: ChatRepositoryDependency,
):
    try:
        conversation, messages = repository.get_shared_conversation(token)
    except ChatNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "conversation": _conversation_payload(conversation),
        "messages": [_message_payload(item) for item in messages],
    }


@app.post("/api/v1/answer", response_model=AnswerResponse)
async def answer(
    body: AnswerBody,
    http_request: Request,
    retrieval_service: RetrievalServiceDependency,
    rag_runtime: RAGRuntimeDependency,
    _admin: AdminUserDependency,
):
    _validate_body(body)
    request_id = uuid4().hex
    started = perf_counter()
    pipeline_task = asyncio.create_task(
        _execute_answer_pipeline(body, retrieval_service, rag_runtime)
    )
    disconnect_task = asyncio.create_task(_wait_for_disconnect(http_request))
    try:
        done, _pending = await asyncio.wait(
            {pipeline_task, disconnect_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if disconnect_task in done and disconnect_task.result():
            pipeline_task.cancel()
            with suppress(asyncio.CancelledError):
                await pipeline_task
            log_event("rag_request_cancelled", request_id=request_id)
            raise HTTPException(status_code=499, detail="Client disconnected")
        disconnect_task.cancel()
        with suppress(asyncio.CancelledError):
            await disconnect_task
        (
            issue_plan,
            retrieval,
            context,
            result,
            retrieval_seconds,
            context_seconds,
            generation_seconds,
            decomposition_seconds,
            queue_wait_seconds,
            processing_seconds,
            rewrite,
        ) = await pipeline_task
    except UnsupportedAsOfDate as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except IssueDecompositionError as exc:
        log_event("rag_issue_decomposition_failed", request_id=request_id)
        raise HTTPException(
            status_code=503,
            detail="Không thể phân tích đầy đủ các vấn đề trong câu hỏi.",
        ) from exc
    except RAGQueueFullError as exc:
        log_event("rag_queue_rejected", request_id=request_id)
        raise HTTPException(status_code=429, detail="RAG request queue is full") from exc
    except TimeoutError as exc:
        log_event("rag_request_timeout", request_id=request_id)
        raise HTTPException(status_code=504, detail="RAG request deadline exceeded") from exc
    except asyncio.CancelledError:
        pipeline_task.cancel()
        disconnect_task.cancel()
        log_event("rag_request_cancelled", request_id=request_id)
        raise

    evidence_by_id = {item.evidence_id: item for item in context.evidence}
    cited_ids = tuple(
        dict.fromkeys(
            evidence_id
            for claim in result.answer.claims
            for evidence_id in claim.evidence_ids
        )
    )
    citations = [
        _citation_payload(evidence_by_id[evidence_id])
        for evidence_id in cited_ids
        if evidence_id in evidence_by_id
    ]
    timing = {
        "queue_wait": round(queue_wait_seconds, 3),
        "issue_decomposition": round(decomposition_seconds, 3),
        "retrieval": round(retrieval_seconds, 3),
        "context": round(context_seconds, 3),
        "generation_and_verification": round(generation_seconds, 3),
        "processing": round(processing_seconds, 3),
        "total": round(perf_counter() - started, 3),
        "llm_ttft": (
            round(result.telemetry[0].ttft_seconds, 3)
            if result.telemetry and result.telemetry[0].ttft_seconds is not None
            else None
        ),
        "generation_attempts": [asdict(item) for item in result.telemetry],
        "stages": [asdict(item) for item in result.stage_timings],
        "semantic_attempts": [asdict(item.review.telemetry) for item in result.semantic_observations if item.review.telemetry],
        "issue_decomposition_attempt": (
            asdict(issue_plan.telemetry) if issue_plan.telemetry else None
        ),
    }
    common = {
        "request_id": request_id,
        "mode": body.response_mode,
        "query": retrieval.query,
        "rewritten_query": rewrite.standalone_question if rewrite is not None else None,
        "as_of": retrieval.as_of,
        "status": result.status.value,
        "semantic_mode": result.semantic_mode,
        "semantic_status": result.semantic_status,
        "coverage_status": result.coverage_status,
        "attempts": result.attempts,
        "answer": result.answer.answer,
        "claims": [claim.to_dict() for claim in result.answer.claims],
        "limitations": list(result.answer.limitations),
        "confidence": result.answer.confidence,
        "legal_issues": [asdict(item) for item in issue_plan.issues],
        "issue_resolutions": [item.to_dict() for item in result.answer.issue_resolutions],
        "citations": citations,
        "timing": timing,
    }
    for observation in result.semantic_observations:
        log_event("rag_semantic_review", request_id=request_id,
            mode=observation.mode, attempt=observation.attempt,
            passed=observation.review.passed, error=observation.review.error,
            verdicts=[item.verdict for item in observation.review.checks],
            quote_issue_codes=[item.code for item in observation.review.quote_issues],
            plan_complete=observation.review.plan_complete,
            coverage_verdicts=[
                {"issue_id": item.issue_id, "verdict": item.verdict}
                for item in observation.review.coverage_checks
            ],
            total_seconds=observation.review.total_seconds,
            cache_hit=observation.review.cache_hit,
            telemetry=asdict(observation.review.telemetry) if observation.review.telemetry else None)
    if body.response_mode == "compact":
        log_event(
            "rag_request_completed",
            request_id=request_id,
            rewritten_query=rewrite.standalone_question if rewrite is not None else None,
            status=result.status.value,
            attempts=result.attempts,
            total_seconds=timing["total"],
            queue_wait_seconds=timing["queue_wait"],
            processing_seconds=timing["processing"],
            retrieval_seconds=timing["retrieval"],
            context_seconds=timing["context"],
            generation_seconds=timing["generation_and_verification"],
            retrieval_stage_seconds=retrieval.timings,
            generation_stage_seconds=timing["stages"],
            llm_ttft_seconds=timing["llm_ttft"],
        )
        return CompactAnswerResponse.model_validate(common)

    response = {
        **common,
        "semantic_observations": [asdict(item) for item in result.semantic_observations],
        "verification": {
            "status": result.verification.status.value,
            "issues": [
                {
                    "code": issue.code.value,
                    "message": issue.message,
                    "claim_index": issue.claim_index,
                }
                for issue in result.verification.issues
            ],
        },
        "retrieval": {
            "sources": retrieval.retrieval_sources,
            "searched_candidates": retrieval.searched_candidates,
            "rejected_candidates": retrieval.rejected_candidates,
            "rejection_reasons": retrieval.rejection_reasons,
            "warnings": retrieval.warnings,
            "timings": retrieval.timings,
        },
        "context": {
            "evidence_count": len(context.evidence),
            "used_tokens": context.used_tokens,
            "token_budget": context.token_budget,
            "dropped_chunk_ids": context.dropped_chunk_ids,
            "issue_evidence_counts": {
                issue.issue_id: sum(
                    issue.issue_id in evidence.issue_ids
                    and evidence.role != "graph_relationship"
                    for evidence in context.evidence
                )
                for issue in retrieval.legal_issues
            },
        },
        "evidence": [
            {
                **_citation_payload(item),
                "text": item.text,
                "role": item.role,
            }
            for item in context.evidence
        ],
    }
    log_event(
        "rag_request_completed",
        request_id=request_id,
        rewritten_query=rewrite.standalone_question if rewrite is not None else None,
        status=result.status.value,
        attempts=result.attempts,
        total_seconds=timing["total"],
        queue_wait_seconds=timing["queue_wait"],
        processing_seconds=timing["processing"],
        retrieval_seconds=timing["retrieval"],
        context_seconds=timing["context"],
        generation_seconds=timing["generation_and_verification"],
        retrieval_stage_seconds=retrieval.timings,
        generation_stage_seconds=timing["stages"],
        llm_ttft_seconds=timing["llm_ttft"],
    )
    return VerboseAnswerResponse.model_validate(response)


@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(
    current_user: CurrentUserDependency,
    body: ChatBody,
    repository: ChatRepositoryDependency,
    workspace_id: WorkspaceDependency,
    retrieval_service: RetrievalServiceDependency,
    rag_runtime: RAGRuntimeDependency,
):
    answer_body = _answer_body_from_chat(body)
    _validate_body(answer_body)
    # Check token quota before creating the user message
    # or running the RAG/LLM pipeline.
    session_factory = app.state.session_factory

    with session_factory() as db:
        _check_user_quota(
            db,
            current_user.id,
        )
    try:
        user_message, created = await run_in_threadpool(
            repository.create_user_message,
            current_user.id,
            workspace_id,
            body.conversation_id,
            content=body.message.strip(),
            client_message_id=body.client_message_id,
            as_of=body.as_of,
        )
    except ChatNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    if not created:
        existing = await run_in_threadpool(
            repository.find_assistant_reply,
            body.conversation_id,
            body.client_message_id,
        )

        if existing is None:
            raise HTTPException(
                status_code=409,
                detail="Message is already processing",
            )

        return {
            "conversation_id": body.conversation_id,
            "user_message_id": user_message.id,
            "assistant_message_id": existing.id,
            "result": _message_result_payload(existing),
        }

    request_id = uuid4().hex
    started = perf_counter()

    try:
        history = await run_in_threadpool(
            repository.list_messages,
            current_user.id,
            workspace_id,
            body.conversation_id,
        )
        pipeline = await _execute_answer_pipeline(
            answer_body,
            retrieval_service,
            rag_runtime,
            conversation_id=body.conversation_id,
            history=history,
        )

        payload = _compact_payload_from_pipeline(
            answer_body,
            pipeline,
            request_id=request_id,
            started=started,
        )

        # Add actual token usage of every LLM call to the user's quota.
        await run_in_threadpool(_record_token_usage, current_user.id, pipeline)

    except Exception as exc:
        log_event(
            "chat_stream_failed",
            request_id=request_id,
            error_type=type(exc).__name__,
            error=str(exc),
        )

        status_code, detail, error_code = _pipeline_error(exc)

        failure = _failure_payload(
            request_id,
            body.message,
            body.as_of,
            detail,
            error_code,
        )

        await run_in_threadpool(
            repository.save_assistant_reply,
            current_user.id,
            workspace_id,
            body.conversation_id,
            client_message_id=body.client_message_id,
            payload=failure,
        )

        raise HTTPException(
            status_code=status_code,
            detail=detail,
        ) from exc

    encoded_payload = jsonable_encoder(payload)

    assistant = await run_in_threadpool(
        repository.save_assistant_reply,
        current_user.id,
        workspace_id,
        body.conversation_id,
        client_message_id=body.client_message_id,
        payload=encoded_payload,
    )

    return {
        "conversation_id": body.conversation_id,
        "user_message_id": user_message.id,
        "assistant_message_id": assistant.id,
        "result": encoded_payload,
    }


@app.post("/api/v1/chat/stream")
async def chat_stream(
    body: ChatBody,
    current_user: CurrentUserDependency,
    repository: ChatRepositoryDependency,
    workspace_id: WorkspaceDependency,
    retrieval_service: RetrievalServiceDependency,
    rag_runtime: RAGRuntimeDependency,
):
    answer_body = _answer_body_from_chat(body)
    _validate_body(answer_body)

    # Check token quota before creating the user message
    # or running the RAG/LLM pipeline.
    session_factory = app.state.session_factory

    with session_factory() as db:
        _check_user_quota(
            db,
            current_user.id,
        )
    async def events():
        request_id = uuid4().hex
        started = perf_counter()
        pipeline_task: asyncio.Task | None = None
        # Only the request that created the user message may write its reply;
        # a retry must never overwrite an answer that is already stored.
        created = False
        try:
            try:
                user_message, created = await run_in_threadpool(
                    repository.create_user_message,
                    current_user.id,
                    workspace_id,
                    body.conversation_id,
                    content=body.message.strip(),
                    client_message_id=body.client_message_id,
                    as_of=body.as_of,
                )
            except ChatNotFoundError:
                yield _sse("chat.failed", {
                    "request_id": request_id,
                    "code": "CONVERSATION_NOT_FOUND",
                    "message": "Conversation not found",
                })
                return
            yield _sse("chat.started", {
                "request_id": request_id,
                "conversation_id": str(body.conversation_id),
                "user_message_id": str(user_message.id),
            })
            if not created:
                existing = await run_in_threadpool(
                    repository.find_assistant_reply,
                    body.conversation_id,
                    body.client_message_id,
                )
                if existing is None:
                    yield _sse("chat.failed", {
                        "request_id": request_id,
                        "code": "MESSAGE_IN_PROGRESS",
                        "message": "Message is already processing",
                    })
                    return
                yield _sse("chat.completed", {
                    "request_id": existing.request_id or request_id,
                    "assistant_message_id": str(existing.id),
                    "result": _message_result_payload(existing),
                    "replayed": True,
                })
                return

            event_queue: asyncio.Queue[tuple[str, dict[str, Any]]] = asyncio.Queue()
            history = await run_in_threadpool(
                repository.list_messages,
                current_user.id,
                workspace_id,
                body.conversation_id,
            )
            pipeline_task = asyncio.create_task(_execute_answer_pipeline(
                answer_body, retrieval_service, rag_runtime,
                event_queue=event_queue,
                conversation_id=body.conversation_id,
                history=history,
            ))
            while not pipeline_task.done() or not event_queue.empty():
                try:
                    event_name, event_data = await asyncio.wait_for(
                        event_queue.get(), timeout=10
                    )
                    yield _sse(event_name, {"request_id": request_id, **event_data})
                except TimeoutError:
                    yield ": keep-alive\n\n"
            pipeline = await pipeline_task
            payload = _compact_payload_from_pipeline(
                answer_body, pipeline, request_id=request_id, started=started
            )
            encoded_payload = jsonable_encoder(payload)
            assistant = await run_in_threadpool(
                repository.save_assistant_reply,
                current_user.id,
                workspace_id,
                body.conversation_id,
                client_message_id=body.client_message_id,
                payload=encoded_payload,
            )
            await run_in_threadpool(_record_token_usage, current_user.id, pipeline)

            # Only verified/refused final text is exposed; provider drafts stay private.
            for delta in _answer_deltas(payload["answer"]):
                yield _sse("answer.delta", {
                    "request_id": request_id, "text": delta
                })
            for citation in encoded_payload["citations"]:
                yield _sse("citation", {
                    "request_id": request_id, "citation": citation
                })
            yield _sse("chat.completed", {
                "request_id": request_id,
                "assistant_message_id": str(assistant.id),
                "result": encoded_payload,
            })
        except asyncio.CancelledError:
            if pipeline_task is not None:
                pipeline_task.cancel()
            raise
        except Exception as exc:
            print(
                f"[CHAT_STREAM_FAILED] request_id={request_id} "
                f"type={type(exc).__name__} error={exc}",
                flush=True,
            )
            import traceback
            traceback.print_exc()
            status_code, detail, error_code = _pipeline_error(exc)
            failure = _failure_payload(
                request_id, body.message, body.as_of, detail, error_code
            )
            if created:
                try:
                    await run_in_threadpool(
                        repository.save_assistant_reply,
                        current_user.id,
                        workspace_id,
                        body.conversation_id,
                        client_message_id=body.client_message_id,
                        payload=failure,
                    )
                except Exception:
                    pass
            yield _sse("chat.failed", {
                "request_id": request_id,
                "code": error_code,
                "message": detail,
                "http_status": status_code,
            })
        finally:
            if pipeline_task is not None and not pipeline_task.done():
                pipeline_task.cancel()

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )


def _project_payload(item) -> dict[str, Any]:
    return {
        "id": item.id,
        "name": item.name,
        "is_shared": item.is_shared,
        "created_at": item.created_at,
        "updated_at": item.updated_at,
    }


def _conversation_payload(item) -> dict[str, Any]:
    return {
        "id": item.id,
        "title": item.title,
        "project_id": item.project_id,
        "pinned": item.pinned,
        "is_shared": item.is_shared,
        "created_at": item.created_at,
        "updated_at": item.updated_at,
    }


def _message_payload(item) -> dict[str, Any]:
    return {
        "id": item.id,
        "conversation_id": item.conversation_id,
        "client_message_id": item.client_message_id,
        "role": item.role,
        "status": item.status,
        "content": item.content,
        "citations": item.citations,
        "claims": item.claims,
        "limitations": item.limitations,
        "as_of": item.as_of,
        "request_id": item.request_id,
        "semantic_status": item.semantic_status,
        "coverage_status": item.coverage_status,
        "confidence": item.confidence,
        "error_code": item.error_code,
        "created_at": item.created_at,
        "completed_at": item.completed_at,
    }


def _message_result_payload(item) -> dict[str, Any]:
    return jsonable_encoder({
        "request_id": item.request_id,
        "status": "VERIFIED" if item.status == "COMPLETED" else item.status,
        "answer": item.content,
        "claims": item.claims,
        "citations": item.citations,
        "limitations": item.limitations,
        "as_of": item.as_of,
        "semantic_status": item.semantic_status,
        "coverage_status": item.coverage_status,
        "confidence": item.confidence,
        "error_code": item.error_code,
    })


def _answer_body_from_chat(body: ChatBody) -> AnswerBody:
    return AnswerBody(
        query=body.message.strip(),
        as_of=body.as_of,
        limit=body.limit,
        candidate_limit=body.candidate_limit,
        document_numbers=body.document_numbers,
        document_types=body.document_types,
        authorities=body.authorities,
        legal_fields=body.legal_fields,
        statuses=body.statuses,
        response_mode=body.response_mode,
    )


def _compact_payload_from_pipeline(
    body: AnswerBody,
    pipeline,
    *,
    request_id: str,
    started: float,
) -> dict[str, Any]:
    (
        issue_plan, retrieval, context, result, retrieval_seconds,
        context_seconds, generation_seconds, decomposition_seconds,
        queue_wait_seconds, processing_seconds, rewrite,
    ) = pipeline
    evidence_by_id = {item.evidence_id: item for item in context.evidence}
    cited_ids = tuple(dict.fromkeys(
        evidence_id
        for claim in result.answer.claims
        for evidence_id in claim.evidence_ids
    ))
    citations = [
        _citation_payload(evidence_by_id[evidence_id])
        for evidence_id in cited_ids
        if evidence_id in evidence_by_id
    ]
    return {
        "request_id": request_id,
        "mode": body.response_mode,
        "query": retrieval.query,
        "rewritten_query": rewrite.standalone_question if rewrite is not None else None,
        "as_of": retrieval.as_of,
        "status": result.status.value,
        "semantic_mode": result.semantic_mode,
        "semantic_status": result.semantic_status,
        "coverage_status": result.coverage_status,
        "attempts": result.attempts,
        "answer": result.answer.answer,
        "claims": [claim.to_dict() for claim in result.answer.claims],
        "limitations": list(result.answer.limitations),
        "confidence": result.answer.confidence,
        "legal_issues": [asdict(item) for item in issue_plan.issues],
        "issue_resolutions": [
            item.to_dict() for item in result.answer.issue_resolutions
        ],
        "citations": citations,
        "timing": {
            "queue_wait": round(queue_wait_seconds, 3),
            "issue_decomposition": round(decomposition_seconds, 3),
            "retrieval": round(retrieval_seconds, 3),
            "context": round(context_seconds, 3),
            "generation_and_verification": round(generation_seconds, 3),
            "processing": round(processing_seconds, 3),
            "total": round(perf_counter() - started, 3),
            "llm_ttft": (
                round(result.telemetry[0].ttft_seconds, 3)
                if result.telemetry and result.telemetry[0].ttft_seconds is not None
                else None
            ),
            "generation_attempts": [asdict(item) for item in result.telemetry],
            "semantic_attempts": [
                asdict(item.review.telemetry)
                for item in result.semantic_observations
                if item.review.telemetry
            ],
            "issue_decomposition_attempt": (
                asdict(issue_plan.telemetry) if issue_plan.telemetry else None
            ),
            "stages": [asdict(item) for item in result.stage_timings],
        },
    }


def _pipeline_error(exc: Exception) -> tuple[int, str, str]:
    if isinstance(exc, UnsupportedAsOfDate):
        return 422, str(exc), "UNSUPPORTED_AS_OF"
    if isinstance(exc, IssueDecompositionError):
        return 503, "Không thể phân tích đầy đủ các vấn đề trong câu hỏi.", "DECOMPOSITION_FAILED"
    if isinstance(exc, RAGQueueFullError):
        return 429, "RAG request queue is full", "QUEUE_FULL"
    if isinstance(exc, TimeoutError):
        return 504, "RAG request deadline exceeded", "REQUEST_TIMEOUT"
    return 503, "LawChat could not complete the request", "CHAT_FAILED"


def _failure_payload(
    request_id: str,
    query: str,
    as_of: date | None,
    detail: str,
    error_code: str,
) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "query": query,
        "as_of": as_of,
        "status": "FAILED",
        "answer": detail,
        "claims": [],
        "citations": [],
        "limitations": [detail],
        "semantic_status": "NOT_CHECKED",
        "coverage_status": "NOT_CHECKED",
        "confidence": "low",
        "error_code": error_code,
    }


def _sse(event: str, data: dict[str, Any]) -> str:
    encoded = json.dumps(jsonable_encoder(data), ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {encoded}\n\n"


def _answer_deltas(answer_text: str, max_chars: int = 180):
    paragraphs = answer_text.splitlines(keepends=True)
    for paragraph in paragraphs:
        if not paragraph:
            continue
        for start in range(0, len(paragraph), max_chars):
            yield paragraph[start:start + max_chars]


def _validate_body(body: SearchBody) -> None:
    if body.candidate_limit < body.limit:
        raise HTTPException(
            status_code=422,
            detail="candidate_limit must be greater than or equal to limit",
        )
    try:
        ensure_supported_as_of(body.as_of)
    except UnsupportedAsOfDate as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _to_retrieval_request(body: SearchBody, *, resolved_as_of: date | None = None) -> RetrievalRequest:
    request_options: dict[str, Any] = {
        "query": body.query,
        "as_of": resolved_as_of or body.as_of,
        "limit": body.limit,
        "candidate_limit": body.candidate_limit,
        "max_candidate_limit": 500,
        "document_numbers": tuple(body.document_numbers),
        "document_types": tuple(body.document_types),
        "authorities": tuple(body.authorities),
        "legal_fields": tuple(body.legal_fields),
    }
    if body.statuses:
        request_options["statuses"] = tuple(item.value for item in body.statuses)
    return RetrievalRequest(**request_options)


def _citation_payload(evidence) -> dict[str, Any]:
    citation = evidence.citation
    return {
        "evidence_id": evidence.evidence_id,
        "document_id": evidence.document_id,
        "chunk_id": evidence.chunk_id,
        "title": citation.title,
        "document_number": citation.document_number,
        "article": citation.article,
        "clause": citation.clause,
        "point": citation.point,
        "source_url": citation.source_url,
        "as_of": citation.as_of,
        "status": citation.status,
        "status_scope": citation.status_scope,
        "version_id": citation.version_id,
        "version_source_url": citation.version_source_url,
        "version_source_revision": citation.version_source_revision,
        "content_valid_from": citation.content_valid_from,
        "content_valid_to": citation.content_valid_to,
    }


async def _execute_answer_pipeline(
    body,
    retrieval_service,
    rag_runtime,
    *,
    event_queue: asyncio.Queue[tuple[str, dict[str, Any]]] | None = None,
    conversation_id: UUID | None = None,
    history: list[Any] | None = None,
):
    async with rag_runtime.execution_gate.slot() as queue_wait_seconds:
        _emit_pipeline_event(event_queue, "queue.completed", {
            "queue_wait_seconds": round(queue_wait_seconds, 3)
        })
        processing_started = perf_counter()
        async with asyncio.timeout(
            rag_runtime.execution_gate.request_timeout_seconds
        ):
            # Retrieval, decomposition and generation see the question
            # rewritten into a standalone one using the conversation; the
            # reference heuristic is the fallback when the rewriter fails.
            rewrite = None
            if not requires_case_specific_prediction_refusal(body.query):
                rewrite = await _rewrite_query(
                    body.query,
                    _prior_turns(body.query, history or []),
                    rag_runtime,
                    event_queue,
                )
            retrieval_query = (
                rewrite.standalone_question
                if rewrite is not None
                else _enrich_with_history(body.query, history or []) or body.query
            )
            parsed_original = LegalQueryParser().parse(retrieval_query, as_of=body.as_of)
            base_request = replace(
                _to_retrieval_request(body, resolved_as_of=parsed_original.as_of),
                query=retrieval_query,
            )
            if requires_case_specific_prediction_refusal(body.query):
                _emit_pipeline_event(event_queue, "generation.started", {})
                _emit_pipeline_event(event_queue, "verification.started", {})
                context_started = perf_counter()
                retrieval = RetrievalResponse(
                    query=body.query,
                    semantic_query=parsed_original.semantic_query,
                    as_of=parsed_original.as_of,
                    results=(),
                    searched_candidates=0,
                    rejected_candidates=0,
                )
                context = await run_in_threadpool(
                    rag_runtime.context_builder.build, retrieval
                )
                context_seconds = perf_counter() - context_started
                generation_started = perf_counter()
                result = await rag_runtime.generation_service.answer_async(
                    GenerationRequest.from_retrieval(retrieval, context)
                )
                generation_seconds = perf_counter() - generation_started
                _emit_pipeline_event(event_queue, "verification.completed", {
                    "status": result.status.value,
                    "semantic_status": result.semantic_status,
                    "coverage_status": result.coverage_status,
                })
                return (
                    IssuePlan(()), retrieval, context, result,
                    0.0, context_seconds, generation_seconds, 0.0,
                    queue_wait_seconds, perf_counter() - processing_started,
                    rewrite,
                )
            _emit_pipeline_event(event_queue, "decomposition.started", {})
            decomposition_started = perf_counter()
            if rag_runtime.issue_decomposer is None:
                issue_plan = IssuePlan(())
            else:
                try:
                    issue_plan = await rag_runtime.issue_decomposer.decompose_async(
                        IssueDecompositionRequest(retrieval_query)
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    raise IssueDecompositionError(
                        "Issue decomposition provider failed"
                    ) from exc
            decomposition_seconds = perf_counter() - decomposition_started
            _emit_pipeline_event(event_queue, "decomposition.completed", {
                "issue_count": len(issue_plan.issues),
                "seconds": round(decomposition_seconds, 3),
            })

            _emit_pipeline_event(event_queue, "retrieval.started", {
                "issue_count": len(issue_plan.issues)
            })
            retrieval_started = perf_counter()
            if issue_plan.issues:
                retrieval = await retrieve_issue_plan_async(
                    retrieval_service,
                    base_request,
                    issue_plan,
                )
            else:
                retrieval = await run_in_threadpool(
                    retrieval_service.retrieve,
                    base_request,
                )
            retrieval_seconds = perf_counter() - retrieval_started
            _emit_pipeline_event(event_queue, "retrieval.completed", {
                "result_count": len(retrieval.results),
                "sources": list(retrieval.retrieval_sources),
                "seconds": round(retrieval_seconds, 3),
            })

            _emit_pipeline_event(event_queue, "context.started", {})
            context_started = perf_counter()
            context = await run_in_threadpool(
                rag_runtime.context_builder.build,
                retrieval,
            )
            context_seconds = perf_counter() - context_started
            _emit_pipeline_event(event_queue, "context.completed", {
                "evidence_count": len(context.evidence),
                "used_tokens": context.used_tokens,
                "seconds": round(context_seconds, 3),
            })
            _emit_pipeline_event(event_queue, "generation.started", {})
            _emit_pipeline_event(event_queue, "verification.started", {})
            generation_started = perf_counter()
            result = await rag_runtime.generation_service.answer_async(
                replace(
                    GenerationRequest.from_retrieval(retrieval, context),
                    question=(
                        rewrite.standalone_question if rewrite is not None else body.query
                    ),
                    original_question=body.query if rewrite is not None else None,
                ),
            )
            generation_seconds = perf_counter() - generation_started
            _emit_pipeline_event(event_queue, "verification.completed", {
                "status": result.status.value,
                "semantic_status": result.semantic_status,
                "coverage_status": result.coverage_status,
                "attempts": result.attempts,
                "seconds": round(generation_seconds, 3),
            })
            return (
                issue_plan,
                retrieval,
                context,
                result,
                retrieval_seconds,
                context_seconds,
                generation_seconds,
                decomposition_seconds,
                queue_wait_seconds,
                perf_counter() - processing_started,
                rewrite,
            )


async def _rewrite_query(
    query: str,
    prior_turns: list[Any],
    rag_runtime,
    event_queue: asyncio.Queue[tuple[str, dict[str, Any]]] | None,
) -> QueryRewrite | None:
    """Standalone retrieval question from the LLM, or None to fall back."""
    rewriter = getattr(rag_runtime, "query_rewriter", None)
    if rewriter is None:
        return None
    _emit_pipeline_event(event_queue, "rewrite.started", {})
    started = perf_counter()
    rewrite = await rewriter.rewrite_async(
        QueryRewriteRequest(query, history_for_rewrite(prior_turns))
    )
    _emit_pipeline_event(event_queue, "rewrite.completed", {
        "rewritten": rewrite is not None,
        "standalone_question": rewrite.standalone_question if rewrite else None,
        "seconds": round(perf_counter() - started, 3),
    })
    return rewrite


def _emit_pipeline_event(
    queue: asyncio.Queue[tuple[str, dict[str, Any]]] | None,
    event: str,
    data: dict[str, Any],
) -> None:
    if queue is not None:
        queue.put_nowait((event, data))


_FOLLOW_UP_MARKERS = (
    "điều đó", "khoản đó", "điểm đó", "điều trên", "khoản trên", "điều này",
    "khoản này", "nội dung trên", "văn bản đó", "văn bản này", "văn bản trên",
    "luật đó", "luật này", "nghị định đó", "nghị định này", "quy định đó",
    "quy định này", "quy định trên", "trường hợp đó", "trường hợp này",
)
_FOLLOW_UP_START_RE = re.compile(r"^\s*(?:vậy|thế|còn|nếu vậy|thế còn|vậy còn)\b", re.IGNORECASE)


def _prior_turns(query: str, history: list[Any]) -> list[Any]:
    """Return earlier turns only; list_messages() already holds the current one."""
    turns = list(history)
    if turns:
        last = turns[-1]
        if (
            (getattr(last, "role", "") or "").lower() in {"user", "human"}
            and (getattr(last, "content", "") or "").strip() == query.strip()
        ):
            turns.pop()
    return turns


def _extract_history_entities(history: list[Any]) -> tuple[list[str], list[str]]:
    """Article and document references from the latest prior turn.

    The latest user message wins; the assistant reply is only consulted when
    that message names nothing, because answers usually cite many documents.
    """
    parser = LegalQueryParser()
    for role_group in ({"user", "human"}, {"assistant"}):
        for message in reversed(history[-2:]):
            if (getattr(message, "role", "") or "").lower() not in role_group:
                continue
            content = (getattr(message, "content", "") or "").strip()
            if not content:
                continue
            parsed = parser.parse(content[:2000])
            if parsed.document_numbers or parsed.referenced_articles:
                return (
                    [f"Điều {item}" for item in parsed.referenced_articles[:2]],
                    list(parsed.document_numbers[:2]),
                )
    return [], []


def _enrich_with_history(query: str, history: list[Any]) -> str:
    """Resolve a follow-up question against the previous turn for retrieval.

    Returns "" when the question stands on its own. Only legal references
    (Điều N, document numbers) are appended, never whole earlier messages,
    so the dense query is not diluted. With
    :data:`FeatureFlags.conversation_context_enabled` every question without
    its own reference is enriched, not only explicit follow-ups.
    """
    prior = _prior_turns(query, history)
    if not prior:
        return ""
    folded = query.casefold()
    is_follow_up = (
        any(marker in folded for marker in _FOLLOW_UP_MARKERS)
        or bool(_FOLLOW_UP_START_RE.search(query))
    )
    if not is_follow_up and not _feature_flags_or_default().conversation_context_enabled:
        return ""
    own = LegalQueryParser().parse(query)
    articles, documents = _extract_history_entities(prior)
    references = [
        *([] if own.referenced_articles else articles),
        *([] if own.document_numbers else documents),
    ]
    if not references:
        return ""
    return f"{query} (tham chiếu: {', '.join(references)})"


def _feature_flags_or_default():
    """Lazy feature-flag accessor to keep the module importable in tests."""
    try:
        try:
            from config import get_feature_flags
        except (ImportError, ValueError):
            from ..config import get_feature_flags  # type: ignore[import-not-found]

        return get_feature_flags()
    except Exception:
        # Fallback: minimal flags with conversation_context_enabled=False.
        from dataclasses import replace

        try:
            from config import FeatureFlags, load_feature_flags
        except (ImportError, ValueError):
            from ..config import FeatureFlags, load_feature_flags

        return replace(load_feature_flags(), conversation_context_enabled=False)



async def _wait_for_disconnect(request: Request) -> bool:
    while True:
        if await request.is_disconnected():
            return True
        await asyncio.sleep(0.1)
def _pipeline_token_usage(pipeline) -> int:
    """Prompt + completion tokens of every LLM call made for one answer.

    Covers query rewrite, issue decomposition, each generation attempt and each semantic
    review (cache hits carry no telemetry).
    """
    issue_plan, _retrieval, _context, result = pipeline[:4]
    rewrite = pipeline[10] if len(pipeline) > 10 else None
    telemetry_items = [
        getattr(rewrite, "telemetry", None),
        getattr(issue_plan, "telemetry", None),
        *getattr(result, "telemetry", ()),
        *(
            observation.review.telemetry
            for observation in getattr(result, "semantic_observations", ())
        ),
    ]
    return sum(
        (item.prompt_tokens or 0) + (item.completion_tokens or 0)
        for item in telemetry_items
        if item is not None
    )


def _record_token_usage(user_id: UUID, pipeline) -> None:
    with app.state.session_factory() as db:
        _add_user_token_usage(db, user_id, _pipeline_token_usage(pipeline))
        db.commit()


def _check_user_quota(
    db,
    user_id: UUID,
) -> None:
    result = db.execute(
        text(
            """
            SELECT
                token_limit,
                tokens_used,
                period_end
            FROM user_usage
            WHERE user_id = :user_id
            """
        ),
        {
            "user_id": user_id,
        },
    ).mappings().first()

    if result is None:
        raise HTTPException(
            status_code=404,
            detail="Usage quota not found",
        )

    token_limit = int(result["token_limit"])
    tokens_used = int(result["tokens_used"])

    if tokens_used >= token_limit:
        raise HTTPException(
            status_code=429,
            detail="Token quota exceeded",
        )
def _add_user_token_usage(
    db,
    user_id: UUID,
    tokens: int,
) -> None:
    if tokens <= 0:
        return

    db.execute(
        text(
            """
            UPDATE user_usage
            SET
                tokens_used = LEAST(
                    tokens_used + :tokens,
                    token_limit
                ),
                updated_at = CURRENT_TIMESTAMP
            WHERE user_id = :user_id
            """
        ),
        {
            "user_id": user_id,
            "tokens": tokens,
        },
    )
