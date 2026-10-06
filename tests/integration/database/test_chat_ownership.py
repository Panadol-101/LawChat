from __future__ import annotations

import os
import uuid

import pytest
from sqlalchemy import create_engine, delete
from sqlalchemy.orm import sessionmaker

from chat.repository import ChatNotFoundError, ChatRepository
from database.models import User


TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="TEST_DATABASE_URL is required for PostgreSQL integration tests",
)

WORKSPACE = "local"


@pytest.fixture
def setup():
    engine = create_engine(TEST_DATABASE_URL, pool_pre_ping=True)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    user_ids = []
    with factory() as session, session.begin():
        for name in ("owner", "intruder"):
            user = User(
                id=uuid.uuid4(),
                username=f"{name}-{uuid.uuid4().hex[:8]}",
                password_hash="x",
            )
            session.add(user)
            user_ids.append(user.id)
    try:
        yield ChatRepository(factory), *user_ids
    finally:
        with factory() as session, session.begin():
            session.execute(delete(User).where(User.id.in_(user_ids)))
        engine.dispose()


def _intruder_is_blocked(repository, intruder, conversation_id):
    attempts = [
        lambda: repository.get_conversation(intruder, WORKSPACE, conversation_id),
        lambda: repository.list_messages(intruder, WORKSPACE, conversation_id),
        lambda: repository.create_user_message(
            intruder, WORKSPACE, conversation_id,
            content="x", client_message_id=uuid.uuid4().hex, as_of=None,
        ),
        lambda: repository.save_assistant_reply(
            intruder, WORKSPACE, conversation_id,
            client_message_id=uuid.uuid4().hex, payload={"status": "VERIFIED"},
        ),
        lambda: repository.update_conversation(
            intruder, WORKSPACE, conversation_id, title="hijacked"
        ),
        lambda: repository.enable_conversation_share(intruder, WORKSPACE, conversation_id),
        lambda: repository.disable_conversation_share(intruder, WORKSPACE, conversation_id),
        lambda: repository.delete_conversation(intruder, WORKSPACE, conversation_id),
    ]
    for attempt in attempts:
        with pytest.raises(ChatNotFoundError):
            attempt()


def test_conversation_without_project_is_private(setup):
    repository, owner, intruder = setup
    conversation = repository.create_conversation(owner, WORKSPACE, "riêng tư")

    _intruder_is_blocked(repository, intruder, conversation.id)
    assert repository.get_conversation(owner, WORKSPACE, conversation.id).title == "riêng tư"
    assert [c.id for c in repository.list_conversations(owner, WORKSPACE)] == [conversation.id]
    assert repository.list_conversations(intruder, WORKSPACE) == []


def test_conversation_stays_private_after_project_is_deleted(setup):
    repository, owner, intruder = setup
    project = repository.create_project(owner, WORKSPACE, "dự án")
    conversation = repository.create_conversation(
        owner, WORKSPACE, "trong dự án", project_id=project.id
    )
    repository.delete_project(owner, WORKSPACE, project.id)

    _intruder_is_blocked(repository, intruder, conversation.id)
    assert repository.get_conversation(owner, WORKSPACE, conversation.id).project_id is None


def test_conversation_stays_private_after_removal_from_project(setup):
    repository, owner, intruder = setup
    project = repository.create_project(owner, WORKSPACE, "dự án")
    conversation = repository.create_conversation(
        owner, WORKSPACE, "trong dự án", project_id=project.id
    )
    repository.update_conversation(owner, WORKSPACE, conversation.id, project_id=None)

    _intruder_is_blocked(repository, intruder, conversation.id)


def test_intruder_cannot_move_conversation_into_own_project(setup):
    repository, owner, intruder = setup
    conversation = repository.create_conversation(owner, WORKSPACE, "riêng tư")
    intruder_project = repository.create_project(intruder, WORKSPACE, "của kẻ xâm nhập")

    with pytest.raises(ChatNotFoundError):
        repository.update_conversation(
            intruder, WORKSPACE, conversation.id, project_id=intruder_project.id
        )
