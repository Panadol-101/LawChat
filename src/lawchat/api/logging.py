from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        fields = getattr(record, "structured_fields", None)
        if isinstance(fields, dict):
            payload.update(fields)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging() -> None:
    if os.getenv("LAWCHAT_JSON_LOGS", "true").casefold() not in {
        "1", "true", "yes", "on"
    }:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("lawchat.api")
    logger.handlers[:] = [handler]
    logger.setLevel(os.getenv("LAWCHAT_LOG_LEVEL", "INFO"))
    logger.propagate = False


def log_event(event: str, **fields) -> None:
    logging.getLogger("lawchat.api").info(
        event,
        extra={"structured_fields": {"event": event, **fields}},
    )
