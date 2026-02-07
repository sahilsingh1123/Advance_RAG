"""Custom exceptions for the RAG system."""

from typing import Any, Dict, Optional


class AdvanceRAGError(Exception):
    """Base exception for Advance RAG system."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}


class ValidationError(AdvanceRAGError):
    """Raised when data validation fails."""

    pass


class IngestionError(AdvanceRAGError):
    """Raised when data ingestion fails."""

    pass


class RetrievalError(AdvanceRAGError):
    """Raised when retrieval operation fails."""

    pass


class EmbeddingError(AdvanceRAGError):
    """Raised when embedding generation fails."""

    pass


class GraphError(AdvanceRAGError):
    """Raised when graph operation fails."""

    pass


class DatabaseError(AdvanceRAGError):
    """Raised when database operation fails."""

    pass


class ConfigurationError(AdvanceRAGError):
    """Raised when configuration is invalid."""

    pass


class AuthenticationError(AdvanceRAGError):
    """Raised when authentication fails."""

    pass


class AuthorizationError(AdvanceRAGError):
    """Raised when authorization fails."""

    pass


class RateLimitError(AdvanceRAGError):
    """Raised when rate limit is exceeded."""

    pass


class ServiceUnavailableError(AdvanceRAGError):
    """Raised when a service is unavailable."""

    pass


class FileProcessingError(AdvanceRAGError):
    """Raised when file processing fails."""

    pass


class ChunkingError(AdvanceRAGError):
    """Raised when document chunking fails."""

    pass


class LLMError(AdvanceRAGError):
    """Raised when LLM operation fails."""

    pass


class CacheError(AdvanceRAGError):
    """Raised when cache operation fails."""

    pass


class TaskError(AdvanceRAGError):
    """Raised when background task fails."""

    pass
