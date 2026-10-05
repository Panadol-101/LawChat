from __future__ import annotations

import secrets
import uuid
from datetime import date, datetime, timezone
from typing import Any

from sqlalchemy import or_, select
from sqlalchemy.orm import Session, sessionmaker

from database.models import ChatConversation, ChatMessage, ChatProject


class ChatNotFoundError(LookupError):
    pass


class ChatRepository:
    """Workspace-scoped persistence; callers never issue chat SQL directly."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self.session_factory = session_factory

    def create_project(
        self,
        user_id: uuid.UUID,
        workspace_id: str,
        name: str,
    ) -> ChatProject:
        with self.session_factory() as session, session.begin():
            project = ChatProject(
                user_id=user_id,
                workspace_id=workspace_id,
                name=name.strip(),
            )
            session.add(project)
            session.flush()
            return project

    def list_projects(
        self,
        user_id: uuid.UUID,
        workspace_id: str,
    ) -> list[ChatProject]:
        with self.session_factory() as session:
            return list(session.scalars(
                select(ChatProject)
                .where(
                    ChatProject.user_id == user_id,
                    ChatProject.workspace_id == workspace_id,
                )
                .order_by(ChatProject.updated_at.desc())
            ))

    def get_project(
        self,
        user_id: uuid.UUID,
        workspace_id: str,
        project_id: uuid.UUID,
    ) -> ChatProject:
        with self.session_factory() as session:
            project = session.scalar(
                select(ChatProject).where(
                    ChatProject.id == project_id,
                    ChatProject.user_id == user_id,
                    ChatProject.workspace_id == workspace_id,
                )
            )
            if project is None:
                raise ChatNotFoundError("project not found")
            return project

    def update_project(
        self, user_id: uuid.UUID, workspace_id: str, project_id: uuid.UUID, *, name: str
    ) -> ChatProject:
        with self.session_factory() as session, session.begin():
            project = self._project(session, user_id, workspace_id, project_id)
            project.name = name.strip()
            project.updated_at = self._now()
            session.flush()
            return project

    def delete_project(self, user_id: uuid.UUID, workspace_id: str, project_id: uuid.UUID) -> None:
        with self.session_factory() as session, session.begin():
            session.delete(self._project(session, user_id, workspace_id, project_id))

    def create_conversation(
        self,
        user_id: uuid.UUID,
        workspace_id: str,
        title: str,
        *,
        project_id: uuid.UUID | None = None,
    ) -> ChatConversation:
        with self.session_factory() as session, session.begin():
            if project_id is not None:
                self._project(session, user_id, workspace_id, project_id)
            conversation = ChatConversation(
                workspace_id=workspace_id,
                project_id=project_id,
                title=title.strip(),
            )
            session.add(conversation)
            session.flush()
            return conversation

    def list_conversations(
        self,
        user_id: uuid.UUID,
        workspace_id: str,
        *,
        project_id: uuid.UUID | None = None,
        limit: int = 100,
    ) -> list[ChatConversation]:
        statement = select(ChatConversation).where(
            ChatConversation.workspace_id == workspace_id,
            ChatConversation.project_id.in_(
                select(ChatProject.id).where(
                    ChatProject.user_id == user_id
                )
            ),
        )

        if project_id is not None:
            statement = statement.where(
                ChatConversation.project_id == project_id
            )
        statement = statement.order_by(
            ChatConversation.pinned.desc(), ChatConversation.updated_at.desc()
        ).limit(limit)
        with self.session_factory() as session:
            return list(session.scalars(statement))

    def get_conversation(
        self, user_id: uuid.UUID, workspace_id: str, conversation_id: uuid.UUID
    ) -> ChatConversation:
        with self.session_factory() as session:
            return self._conversation(session, user_id, workspace_id, conversation_id)

    def update_conversation(
        self,
        user_id: uuid.UUID,
        workspace_id: str,
        conversation_id: uuid.UUID,
        *,
        title: str | None = None,
        pinned: bool | None = None,
        project_id: uuid.UUID | None | object = ...,
    ) -> ChatConversation:
        with self.session_factory() as session, session.begin():
            conversation = self._conversation(session, user_id, workspace_id, conversation_id)
            if title is not None:
                conversation.title = title.strip()
            if pinned is not None:
                conversation.pinned = pinned
            if project_id is not ...:
                if project_id is not None:
                    self._project(session, user_id, workspace_id, project_id)
                conversation.project_id = project_id
            conversation.updated_at = self._now()
            session.flush()
            return conversation

    def delete_conversation(
        self, user_id: uuid.UUID, workspace_id: str, conversation_id: uuid.UUID
    ) -> None:
        with self.session_factory() as session, session.begin():
            session.delete(self._conversation(session, user_id, workspace_id, conversation_id))

    def list_messages(
        self, user_id: uuid.UUID, workspace_id: str, conversation_id: uuid.UUID
    ) -> list[ChatMessage]:
        with self.session_factory() as session:
            self._conversation(session, user_id, workspace_id, conversation_id)
            return list(session.scalars(
                select(ChatMessage)
                .where(ChatMessage.conversation_id == conversation_id)
                .order_by(ChatMessage.created_at, ChatMessage.id)
            ))

    def create_user_message(
        self,
        user_id: uuid.UUID,
        workspace_id: str,
        conversation_id: uuid.UUID,
        *,
        content: str,
        client_message_id: str,
        as_of: date | None,
    ) -> tuple[ChatMessage, bool]:
        """Return the durable user message and whether it was newly created."""
        with self.session_factory() as session, session.begin():
            conversation = self._conversation(session, user_id, workspace_id, conversation_id)
            existing = session.scalar(
                select(ChatMessage).where(
                    ChatMessage.conversation_id == conversation_id,
                    ChatMessage.client_message_id == client_message_id,
                )
            )
            if existing is not None:
                return existing, False
            message = ChatMessage(
                conversation_id=conversation_id,
                client_message_id=client_message_id,
                role="user",
                status="COMPLETED",
                content=content,
                as_of=as_of,
                completed_at=self._now(),
            )
            conversation.updated_at = self._now()
            session.add(message)
            session.flush()
            return message, True

    def find_assistant_reply(
        self, conversation_id: uuid.UUID, client_message_id: str
    ) -> ChatMessage | None:
        reply_key = self._assistant_client_id(client_message_id)
        with self.session_factory() as session:
            return session.scalar(
                select(ChatMessage).where(
                    ChatMessage.conversation_id == conversation_id,
                    ChatMessage.client_message_id == reply_key,
                )
            )

    def save_assistant_reply(
        self,
        user_id: uuid.UUID,
        workspace_id: str,
        conversation_id: uuid.UUID,
        *,
        client_message_id: str,
        payload: dict[str, Any],
    ) -> ChatMessage:
        with self.session_factory() as session, session.begin():
            conversation = self._conversation(session, user_id, workspace_id, conversation_id)
            reply_key = self._assistant_client_id(client_message_id)
            message = session.scalar(
                select(ChatMessage).where(
                    ChatMessage.conversation_id == conversation_id,
                    ChatMessage.client_message_id == reply_key,
                )
            )
            if message is None:
                message = ChatMessage(
                    conversation_id=conversation_id,
                    client_message_id=reply_key,
                    role="assistant",
                )
                session.add(message)
            status = str(payload.get("status", "FAILED"))
            message.status = "COMPLETED" if status == "VERIFIED" else (
                "REFUSED" if status == "REFUSED" else "FAILED"
            )
            message.content = str(payload.get("answer", ""))
            message.citations = list(payload.get("citations", ()))
            message.claims = list(payload.get("claims", ()))
            message.limitations = list(payload.get("limitations", ()))
            message.as_of = payload.get("as_of")
            message.request_id = payload.get("request_id")
            message.semantic_status = payload.get("semantic_status")
            message.coverage_status = payload.get("coverage_status")
            message.confidence = payload.get("confidence")
            message.error_code = payload.get("error_code")
            message.completed_at = self._now()
            conversation.updated_at = self._now()
            session.flush()
            return message

    def enable_conversation_share(
        self, user_id: uuid.UUID, workspace_id: str, conversation_id: uuid.UUID
    ) -> ChatConversation:
        with self.session_factory() as session, session.begin():
            conversation = self._conversation(session, user_id, workspace_id, conversation_id)
            conversation.share_token = conversation.share_token or secrets.token_urlsafe(32)
            conversation.is_shared = True
            conversation.updated_at = self._now()
            session.flush()
            return conversation

    def disable_conversation_share(
        self, user_id: uuid.UUID, workspace_id: str, conversation_id: uuid.UUID
    ) -> None:
        with self.session_factory() as session, session.begin():
            conversation = self._conversation(session, user_id, workspace_id, conversation_id)
            conversation.is_shared = False
            conversation.share_token = None

    def get_shared_conversation(self, token: str) -> tuple[ChatConversation, list[ChatMessage]]:
        with self.session_factory() as session:
            conversation = session.scalar(
                select(ChatConversation).where(
                    ChatConversation.share_token == token,
                    ChatConversation.is_shared.is_(True),
                )
            )
            if conversation is None:
                raise ChatNotFoundError("shared conversation not found")
            messages = list(session.scalars(
                select(ChatMessage)
                .where(ChatMessage.conversation_id == conversation.id)
                .order_by(ChatMessage.created_at, ChatMessage.id)
            ))
            return conversation, messages

    @staticmethod
    def _project(
        session: Session, user_id: uuid.UUID, workspace_id: str, project_id: uuid.UUID
    ) -> ChatProject:
        project = session.scalar(select(ChatProject).where(
            ChatProject.id == project_id,
            ChatProject.user_id == user_id,
            ChatProject.workspace_id == workspace_id,
        ))
        if project is None:
            raise ChatNotFoundError("project not found")
        return project

    
    @staticmethod
    def _conversation(
        session: Session,
        user_id: uuid.UUID,
        workspace_id: str,
        conversation_id: uuid.UUID,
   ) -> ChatConversation:
       conversation = session.scalar(
           select(ChatConversation)
           .outerjoin(
               ChatProject,
               ChatProject.id == ChatConversation.project_id,
           )
           .where(
               ChatConversation.id == conversation_id,
               ChatConversation.workspace_id == workspace_id,
               or_(
                   ChatConversation.project_id.is_(None),
                   ChatProject.user_id == user_id,
               ),
           )
       )

       if conversation is None:
           raise ChatNotFoundError("conversation not found")

       return conversation


    def _assistant_client_id(self, client_message_id: str) -> str:
        return f"assistant:{client_message_id}"


    def _now(self) -> datetime:
        return datetime.now(timezone.utc)
