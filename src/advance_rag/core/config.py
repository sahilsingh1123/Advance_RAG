"""Application configuration management."""

from functools import lru_cache
from typing import Any, Dict, List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )

    # Environment
    ENVIRONMENT: str = "development"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # Database
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/advance_rag",
        description="PostgreSQL database URL",
    )
    REDIS_URL: str = Field(
        default="redis://localhost:6379", description="Redis URL for caching and Celery"
    )
    NEO4J_URI: str = Field(
        default="bolt://localhost:7687", description="Neo4j connection URI"
    )
    NEO4J_USER: str = Field(default="neo4j", description="Neo4j username")
    NEO4J_PASSWORD: str = Field(default="password123", description="Neo4j password")
    ELASTICSEARCH_URL: str = Field(
        default="http://localhost:9200", description="Elasticsearch URL"
    )

    # API Keys
    OPENAI_API_KEY: Optional[str] = Field(default=None, description="OpenAI API key")
    ANTHROPIC_API_KEY: Optional[str] = Field(
        default=None, description="Anthropic API key"
    )

    # Embedding Configuration
    EMBEDDING_MODEL: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Embedding model name",
    )
    EMBEDDING_DIMENSION: int = Field(
        default=384, description="Embedding vector dimension"
    )
    EMBEDDING_BATCH_SIZE: int = Field(
        default=32, description="Batch size for embedding generation"
    )
    EMBEDDING_CACHE_TTL: int = Field(
        default=3600, description="Cache TTL for embeddings (seconds)"
    )

    # LLM Configuration
    LLM_PROVIDER: str = Field(default="anthropic", description="LLM provider")
    LLM_MODEL: str = Field(
        default="claude-3-opus-20240229", description="LLM model name"
    )
    LLM_MAX_TOKENS: int = Field(default=4096, description="Max tokens for LLM")
    LLM_TEMPERATURE: float = Field(default=0.1, description="LLM temperature")
    LLM_TIMEOUT: int = Field(default=60, description="LLM timeout (seconds)")

    # Celery Configuration
    CELERY_BROKER_URL: str = Field(
        default="redis://localhost:6379", description="Celery broker URL"
    )
    CELERY_RESULT_BACKEND: str = Field(
        default="redis://localhost:6379", description="Celery result backend URL"
    )
    CELERY_TASK_SERIALIZER: str = Field(default="json", description="Task serializer")
    CELERY_RESULT_SERIALIZER: str = Field(
        default="json", description="Result serializer"
    )
    CELERY_ACCEPT_CONTENT: List[str] = Field(
        default=["json"], description="Accepted content types"
    )
    CELERY_TIMEZONE: str = Field(default="UTC", description="Celery timezone")

    # Application Configuration
    MAX_CONCURRENT_REQUESTS: int = Field(
        default=1000, description="Maximum concurrent requests"
    )
    RATE_LIMIT_PER_MINUTE: int = Field(default=60, description="Rate limit per minute")
    REQUEST_TIMEOUT: int = Field(default=30, description="Request timeout (seconds)")

    # Security
    SECRET_KEY: str = Field(
        default="your-secret-key-here", description="Secret key for JWT"
    )
    JWT_ALGORITHM: str = Field(default="HS256", description="JWT algorithm")
    JWT_EXPIRE_MINUTES: int = Field(
        default=1440, description="JWT expiration (minutes)"
    )
    CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        description="CORS allowed origins",
    )

    # Vector Store Configuration
    VECTOR_INDEX_TYPE: str = Field(default="hnsw", description="Vector index type")
    VECTOR_HNSW_M: int = Field(default=16, description="HNSW M parameter")
    VECTOR_HNSW_EF_CONSTRUCTION: int = Field(
        default=64, description="HNSW ef_construction parameter"
    )
    VECTOR_HNSW_EF_SEARCH: int = Field(
        default=40, description="HNSW ef_search parameter"
    )
    VECTOR_SIMILARITY_THRESHOLD: float = Field(
        default=0.7, description="Similarity threshold"
    )

    # Retrieval Configuration
    RETRIEVAL_TOP_K: int = Field(default=10, description="Top-K retrieval")
    RETRIEVAL_DENSE_WEIGHT: float = Field(
        default=0.7, description="Weight for dense retrieval"
    )
    RETRIEVAL_SPARSE_WEIGHT: float = Field(
        default=0.3, description="Weight for sparse retrieval"
    )
    RETRIEVAL_RERANK_TOP_K: int = Field(default=5, description="Top-K after reranking")
    RETRIEVAL_MAX_CONTEXT_LENGTH: int = Field(
        default=4000, description="Max context length in tokens"
    )

    # GraphRAG Configuration
    GRAPH_COMMUNITY_ALGORITHM: str = Field(
        default="leiden", description="Community detection algorithm"
    )
    GRAPH_COMMUNITY_RESOLUTION: float = Field(
        default=1.0, description="Community resolution parameter"
    )
    GRAPH_MAX_HOPS: int = Field(default=3, description="Max graph hops")
    GRAPH_ENTITY_TYPES: List[str] = Field(
        default=["subject", "treatment", "lab_test", "adverse_event", "medication"],
        description="Graph entity types",
    )

    # Data Processing
    CHUNK_SIZE: int = Field(default=1000, description="Chunk size in tokens")
    CHUNK_OVERLAP: int = Field(default=200, description="Chunk overlap in tokens")
    MAX_FILE_SIZE_MB: int = Field(default=100, description="Max file size in MB")
    SUPPORTED_FILE_TYPES: List[str] = Field(
        default=["json", "md", "txt", "csv"], description="Supported file types"
    )

    # Monitoring
    METRICS_ENABLED: bool = Field(default=True, description="Enable metrics")
    METRICS_PORT: int = Field(default=9090, description="Metrics port")
    TRACING_ENABLED: bool = Field(default=True, description="Enable tracing")
    TRACING_SAMPLE_RATE: float = Field(default=0.1, description="Tracing sample rate")

    # Performance
    CACHE_TTL: int = Field(default=300, description="Cache TTL (seconds)")
    CACHE_MAX_SIZE: int = Field(default=1000, description="Max cache size")
    CONNECTION_POOL_SIZE: int = Field(default=20, description="Connection pool size")
    QUERY_TIMEOUT: int = Field(default=10, description="Query timeout (seconds)")

    # PHI/PHI Protection
    PHI_REDACTION_ENABLED: bool = Field(
        default=True, description="Enable PHI redaction"
    )
    PHI_REDACTION_ENTITIES: List[str] = Field(
        default=["PERSON", "LOCATION", "ORGANIZATION", "DATE"],
        description="PHI entities to redact",
    )
    COMPLIANCE_AUDIT_ENABLED: bool = Field(
        default=True, description="Enable compliance audit"
    )

    # Development Settings
    RELOAD_ON_CHANGE: bool = Field(default=True, description="Reload on code change")
    SHOW_SQL_QUERIES: bool = Field(default=False, description="Show SQL queries")
    ENABLE_PROFILER: bool = Field(default=False, description="Enable profiler")

    # Authentication Settings
    SMTP_SERVER: str = Field(default="smtp.gmail.com", description="SMTP server")
    SMTP_PORT: int = Field(default=587, description="SMTP port")
    SMTP_USERNAME: str = Field(default="", description="SMTP username")
    SMTP_PASSWORD: str = Field(default="", description="SMTP password")
    FROM_EMAIL: str = Field(
        default="noreply@advancerag.com", description="From email address"
    )
    WEBHOOK_URLS: Dict[str, str] = Field(
        default_factory=dict, description="Webhook URLs"
    )

    # Export Settings
    EXPORT_DIR: str = Field(default="exports", description="Export directory")

    # Backup Settings
    BACKUP_DIR: str = Field(default="backups", description="Backup directory")
    BACKUP_RETENTION_DAYS: int = Field(
        default=30, description="Backup retention in days"
    )


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
