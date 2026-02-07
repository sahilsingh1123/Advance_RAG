"""Data models for the RAG system."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


class QueryMode(str, Enum):
    """Query modes."""

    QA = "qa"
    CODE_GENERATION = "code_generation"
    SPECIFICATION = "specification"
    GLOBAL_SEARCH = "global_search"
    LOCAL_SEARCH = "local_search"


class DocumentType(str, Enum):
    """Document types."""

    SDTM = "sdtm"
    ADAM = "adam"
    SAP = "sap"
    PROTOCOL = "protocol"
    OTHER = "other"


class Chunk(BaseModel):
    """Document chunk model."""

    id: str = Field(..., description="Chunk unique identifier")
    content: str = Field(..., description="Chunk text content")
    document_id: str = Field(..., description="Parent document ID")
    document_type: DocumentType = Field(..., description="Document type")
    chunk_index: int = Field(..., description="Chunk index in document")
    start_char: int = Field(..., description="Start character position")
    end_char: int = Field(..., description="End character position")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")
    embedding: Optional[List[float]] = Field(None, description="Embedding vector")
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @validator("content")
    def validate_content(cls, v: str) -> str:
        """Validate content is not empty."""
        if not v.strip():
            raise ValueError("Content cannot be empty")
        return v


class Document(BaseModel):
    """Document model."""

    id: str = Field(..., description="Document unique identifier")
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Full document content")
    document_type: DocumentType = Field(..., description="Document type")
    file_path: str = Field(..., description="Original file path")
    file_size: int = Field(..., description="File size in bytes")
    study_id: Optional[str] = Field(None, description="Study identifier")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Document metadata"
    )
    chunks: List[Chunk] = Field(default_factory=list, description="Document chunks")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @validator("title")
    def validate_title(cls, v: str) -> str:
        """Validate title is not empty."""
        if not v.strip():
            raise ValueError("Title cannot be empty")
        return v


class Query(BaseModel):
    """Query model."""

    id: str = Field(..., description="Query unique identifier")
    text: str = Field(..., description="Query text")
    mode: QueryMode = Field(..., description="Query mode")
    study_id: Optional[str] = Field(None, description="Study identifier filter")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Query filters")
    top_k: int = Field(default=10, description="Number of results to retrieve")
    user_id: Optional[str] = Field(None, description="User identifier")
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @validator("text")
    def validate_text(cls, v: str) -> str:
        """Validate query text is not empty."""
        if not v.strip():
            raise ValueError("Query text cannot be empty")
        return v


class RetrievalResult(BaseModel):
    """Retrieval result model."""

    chunk: Chunk = Field(..., description="Retrieved chunk")
    score: float = Field(..., description="Retrieval score")
    source: str = Field(..., description="Retrieval source (dense/sparse/graph)")


class QueryResponse(BaseModel):
    """Query response model."""

    query_id: str = Field(..., description="Query identifier")
    answer: str = Field(..., description="Generated answer")
    sources: List[RetrievalResult] = Field(..., description="Source chunks")
    mode: QueryMode = Field(..., description="Query mode")
    llm_model: str = Field(..., description="LLM model used")
    prompt_tokens: int = Field(..., description="Number of prompt tokens")
    completion_tokens: int = Field(..., description="Number of completion tokens")
    duration_ms: float = Field(..., description="Total duration in milliseconds")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class GraphEntity(BaseModel):
    """Graph entity model."""

    id: str = Field(..., description="Entity unique identifier")
    type: str = Field(..., description="Entity type")
    name: str = Field(..., description="Entity name")
    properties: Dict[str, Any] = Field(
        default_factory=dict, description="Entity properties"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)


class GraphRelation(BaseModel):
    """Graph relation model."""

    id: str = Field(..., description="Relation unique identifier")
    source_id: str = Field(..., description="Source entity ID")
    target_id: str = Field(..., description="Target entity ID")
    type: str = Field(..., description="Relation type")
    properties: Dict[str, Any] = Field(
        default_factory=dict, description="Relation properties"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)


class GraphCommunity(BaseModel):
    """Graph community model."""

    id: str = Field(..., description="Community unique identifier")
    entities: List[str] = Field(..., description="Entity IDs in community")
    summary: str = Field(..., description="Community summary")
    level: int = Field(default=0, description="Community level")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class IngestionTask(BaseModel):
    """Ingestion task model."""

    id: str = Field(..., description="Task unique identifier")
    file_paths: List[str] = Field(..., description="File paths to ingest")
    status: str = Field(..., description="Task status")
    progress: float = Field(default=0.0, description="Progress percentage")
    error_message: Optional[str] = Field(None, description="Error message")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class User(BaseModel):
    """User model."""

    id: str = Field(..., description="User unique identifier")
    email: str = Field(..., description="User email")
    name: str = Field(..., description="User name")
    role: str = Field(..., description="User role")
    studies: List[str] = Field(default_factory=list, description="Accessible study IDs")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = Field(None, description="Last login timestamp")


class Study(BaseModel):
    """Study model."""

    id: str = Field(..., description="Study unique identifier")
    title: str = Field(..., description="Study title")
    description: str = Field(..., description="Study description")
    protocol_id: Optional[str] = Field(None, description="Protocol document ID")
    sap_id: Optional[str] = Field(None, description="SAP document ID")
    sdtm_domains: List[str] = Field(
        default_factory=list, description="SDTM domain names"
    )
    adam_datasets: List[str] = Field(
        default_factory=list, description="ADaM dataset names"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Study metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
