from __future__ import annotations

import os
from dataclasses import dataclass

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker


DEFAULT_DATABASE_URL = (
    "postgresql+psycopg://lawchat:lawchat@localhost:5432/lawchat"
)


@dataclass(frozen=True, slots=True)
class DatabaseSettings:
    """Database configuration sourced from environment variables."""

    url: str = DEFAULT_DATABASE_URL
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout_seconds: int = 30
    pool_recycle_seconds: int = 1800
    echo_sql: bool = False

    @classmethod
    def from_env(cls) -> "DatabaseSettings":
        return cls(
            url=os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL),
            pool_size=_positive_int("DB_POOL_SIZE", 10),
            max_overflow=_non_negative_int("DB_MAX_OVERFLOW", 20),
            pool_timeout_seconds=_positive_int("DB_POOL_TIMEOUT", 30),
            pool_recycle_seconds=_positive_int("DB_POOL_RECYCLE", 1800),
            echo_sql=os.getenv("DB_ECHO_SQL", "false").casefold()
            in {"1", "true", "yes", "on"},
        )


def _positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be > 0")
    return value


def _non_negative_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value < 0:
        raise ValueError(f"{name} must be >= 0")
    return value


def create_db_engine(
    settings: DatabaseSettings | None = None,
) -> Engine:
    settings = settings or DatabaseSettings.from_env()
    return create_engine(
        settings.url,
        pool_pre_ping=True,
        pool_size=settings.pool_size,
        max_overflow=settings.max_overflow,
        pool_timeout=settings.pool_timeout_seconds,
        pool_recycle=settings.pool_recycle_seconds,
        echo=settings.echo_sql,
    )


def create_session_factory(
    engine: Engine,
) -> sessionmaker[Session]:
    return sessionmaker(
        bind=engine,
        class_=Session,
        autoflush=False,
        expire_on_commit=False,
    )
