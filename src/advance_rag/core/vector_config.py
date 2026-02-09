"""Vector store configuration extensions."""

from pydantic import Field
from typing import Optional, List


class VectorStoreConfig:
    """Extended configuration for vector stores."""

    # Vector Store Type Selection
    VECTOR_STORE_TYPE: str = Field(
        default="pgvector",
        description="Vector store type: pgvector, pinecone, weaviate",
    )

    # Pinecone Configuration
    PINECONE_API_KEY: Optional[str] = Field(
        default=None, description="Pinecone API key"
    )
    PINECONE_INDEX_NAME: str = Field(
        default="advance-rag", description="Pinecone index name"
    )
    PINECONE_NAMESPACE: str = Field(default="chunks", description="Pinecone namespace")
    PINECONE_BATCH_SIZE: int = Field(default=100, description="Pinecone batch size")
    PINECONE_SERVERLESS: bool = Field(
        default=True, description="Use Pinecone serverless"
    )
    PINECONE_CLOUD: str = Field(default="aws", description="Pinecone cloud provider")
    PINECONE_REGION: str = Field(default="us-west-2", description="Pinecone region")

    # Weaviate Configuration
    WEAVIATE_URL: str = Field(
        default="http://localhost:8080", description="Weaviate URL"
    )
    WEAVIATE_API_KEY: Optional[str] = Field(
        default=None, description="Weaviate API key"
    )
    WEAVIATE_CLASS_NAME: str = Field(default="Chunk", description="Weaviate class name")
    WEAVIATE_BATCH_SIZE: int = Field(default=100, description="Weaviate batch size")

    # Common Vector Configuration
    EMBEDDING_DIMENSION: int = Field(
        default=384, description="Embedding vector dimension"
    )
    VECTOR_SIMILARITY_THRESHOLD: float = Field(
        default=0.7, description="Similarity threshold"
    )
    VECTOR_BATCH_SIZE: int = Field(default=100, description="Vector batch size")


# Environment variable mappings for easy configuration
VECTOR_STORE_ENV_VARS = {
    "VECTOR_STORE_TYPE": "VECTOR_STORE_TYPE",
    "PINECONE_API_KEY": "PINECONE_API_KEY",
    "PINECONE_INDEX_NAME": "PINECONE_INDEX_NAME",
    "PINECONE_NAMESPACE": "PINECONE_NAMESPACE",
    "PINECONE_BATCH_SIZE": "PINECONE_BATCH_SIZE",
    "PINECONE_SERVERLESS": "PINECONE_SERVERLESS",
    "PINECONE_CLOUD": "PINECONE_CLOUD",
    "PINECONE_REGION": "PINECONE_REGION",
    "WEAVIATE_URL": "WEAVIATE_URL",
    "WEAVIATE_API_KEY": "WEAVIATE_API_KEY",
    "WEAVIATE_CLASS_NAME": "WEAVIATE_CLASS_NAME",
    "WEAVIATE_BATCH_SIZE": "WEAVIATE_BATCH_SIZE",
}
