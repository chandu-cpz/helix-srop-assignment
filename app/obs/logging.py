"""
Structured logging setup.

All log lines must include session_id, trace_id, user_id when available.
Use structlog's context vars for request-scoped fields.
"""
import logging
import re
import sys
from typing import Any

import structlog

EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_PATTERN = re.compile(r"(?<!\w)\+?\d[\d .()-]{7,}\d\b")
SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")


def redact_pii(value: Any) -> Any:
    if isinstance(value, str):
        redacted = EMAIL_PATTERN.sub("[REDACTED_EMAIL]", value)
        redacted = SSN_PATTERN.sub("[REDACTED_SSN]", redacted)
        redacted = PHONE_PATTERN.sub("[REDACTED_PHONE]", redacted)
        return redacted
    if isinstance(value, dict):
        return {key: redact_pii(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_pii(item) for item in value]
    return value


def redact_event_dict(_: Any, __: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    return {key: redact_pii(value) for key, value in event_dict.items()}


def configure_logging() -> None:
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            redact_event_dict,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
    )


# Usage in request handlers:
#   import structlog
#   log = structlog.get_logger()
#   structlog.contextvars.bind_contextvars(session_id=session_id, trace_id=trace_id)
#   log.info("pipeline_started", user_message_len=len(message))
