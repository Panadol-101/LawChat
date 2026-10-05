"""Persistent chat application layer."""

from .repository import ChatNotFoundError, ChatRepository

__all__ = ["ChatNotFoundError", "ChatRepository"]
