"""Factory for creating vector store instances."""

from typing import Dict, Any
from enum import Enum

from advance_rag.core.config import get_settings
from advance_rag.core.logging import get_logger
from .vector_stores import BaseVectorStore, PineconeVectorStore, WeaviateVectorStore

logger = get_logger(__name__)
settings = get_settings()


class VectorStoreType(Enum):
    """Supported vector store types."""

    PINECONE = "pinecone"
    WEAVIATE = "weaviate"
    PGVECTOR = "pgvector"


class VectorStoreFactory:
    """Factory for creating vector store instances."""

    @staticmethod
    def create_vector_store(
        store_type: str = None, config: Dict[str, Any] = None
    ) -> BaseVectorStore:
        """Create vector store instance based on type."""

        # Determine store type
        if not store_type:
            store_type = getattr(settings, "VECTOR_STORE_TYPE", "pgvector")

        store_type = store_type.lower()

        # Get configuration
        if not config:
            config = VectorStoreFactory._get_store_config(store_type)

        logger.info(f"Creating vector store: {store_type}")

        # Create appropriate store
        if store_type == VectorStoreType.PINECONE.value:
            return PineconeVectorStore(config)
        elif store_type == VectorStoreType.WEAVIATE.value:
            return WeaviateVectorStore(config)
        elif store_type == VectorStoreType.PGVECTOR.value:
            # Import here to avoid circular imports
            from .advanced_vector_store import VectorStore

            return VectorStore(config.get("database_url", settings.DATABASE_URL))
        else:
            raise ValueError(f"Unsupported vector store type: {store_type}")

    @staticmethod
    def _get_store_config(store_type: str) -> Dict[str, Any]:
        """Get configuration for specific store type."""

        if store_type == VectorStoreType.PINECONE.value:
            return {
                "index_name": getattr(settings, "PINECONE_INDEX_NAME", "advance-rag"),
                "namespace": getattr(settings, "PINECONE_NAMESPACE", "chunks"),
                "dimension": settings.EMBEDDING_DIMENSION,
                "metric": "cosine",
                "batch_size": getattr(settings, "PINECONE_BATCH_SIZE", 100),
                "serverless": getattr(settings, "PINECONE_SERVERLESS", True),
                "cloud": getattr(settings, "PINECONE_CLOUD", "aws"),
                "region": getattr(settings, "PINECONE_REGION", "us-west-2"),
                "api_key": getattr(settings, "PINECONE_API_KEY", None),
            }

        elif store_type == VectorStoreType.WEAVIATE.value:
            return {
                "url": getattr(settings, "WEAVIATE_URL", "http://localhost:8080"),
                "api_key": getattr(settings, "WEAVIATE_API_KEY", None),
                "class_name": getattr(settings, "WEAVIATE_CLASS_NAME", "Chunk"),
                "batch_size": getattr(settings, "WEAVIATE_BATCH_SIZE", 100),
                "dimension": settings.EMBEDDING_DIMENSION,
            }

        elif store_type == VectorStoreType.PGVECTOR.value:
            return {
                "database_url": settings.DATABASE_URL,
                "dimension": settings.EMBEDDING_DIMENSION,
            }

        else:
            raise ValueError(f"No configuration available for store type: {store_type}")

    @staticmethod
    def get_supported_stores() -> list:
        """Get list of supported vector stores."""
        return [store_type.value for store_type in VectorStoreType]

    @staticmethod
    def validate_config(store_type: str) -> bool:
        """Validate configuration for a store type."""
        try:
            config = VectorStoreFactory._get_store_config(store_type)

            if store_type == VectorStoreType.PINECONE.value:
                return bool(config.get("api_key"))
            elif store_type == VectorStoreType.WEAVIATE.value:
                return True  # Weaviate can work without API key locally
            elif store_type == VectorStoreType.PGVECTOR.value:
                return bool(config.get("database_url"))

            return False

        except Exception as e:
            logger.error(f"Config validation failed for {store_type}: {e}")
            return False
