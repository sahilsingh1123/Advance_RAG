"""Logging configuration for the application."""

import structlog
from typing import Any, Dict

from advance_rag.core.config import get_settings


def configure_logging() -> None:
    """Configure structured logging."""
    settings = get_settings()

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            (
                structlog.processors.JSONRenderer()
                if not settings.DEBUG
                else structlog.dev.ConsoleRenderer()
            ),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


def log_request(request_id: str, method: str, path: str, **kwargs: Any) -> None:
    """Log HTTP request."""
    logger = get_logger("api")
    logger.info(
        "request_received", request_id=request_id, method=method, path=path, **kwargs
    )


def log_response(
    request_id: str, status_code: int, duration_ms: float, **kwargs: Any
) -> None:
    """Log HTTP response."""
    logger = get_logger("api")
    logger.info(
        "response_sent",
        request_id=request_id,
        status_code=status_code,
        duration_ms=duration_ms,
        **kwargs,
    )


def log_error(error: Exception, request_id: str | None = None, **kwargs: Any) -> None:
    """Log error with context."""
    logger = get_logger("error")
    logger.error(
        "error_occurred",
        error=str(error),
        error_type=type(error).__name__,
        request_id=request_id,
        **kwargs,
    )


def log_query(
    query_type: str,
    query: str,
    duration_ms: float,
    num_results: int | None = None,
    **kwargs: Any,
) -> None:
    """Log database/query operation."""
    logger = get_logger("query")
    logger.info(
        "query_executed",
        query_type=query_type,
        duration_ms=duration_ms,
        num_results=num_results,
        **kwargs,
    )


def log_ingestion(
    file_path: str, file_size: int, num_chunks: int, duration_ms: float, **kwargs: Any
) -> None:
    """Log data ingestion operation."""
    logger = get_logger("ingestion")
    logger.info(
        "file_ingested",
        file_path=file_path,
        file_size=file_size,
        num_chunks=num_chunks,
        duration_ms=duration_ms,
        **kwargs,
    )


def log_llm_call(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    duration_ms: float,
    **kwargs: Any,
) -> None:
    """Log LLM API call."""
    logger = get_logger("llm")
    logger.info(
        "llm_call",
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        duration_ms=duration_ms,
        **kwargs,
    )
