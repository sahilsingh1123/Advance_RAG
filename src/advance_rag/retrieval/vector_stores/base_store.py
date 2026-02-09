"""Base interface for vector store implementations."""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Dict, Any
import asyncio


class BaseVectorStore(ABC):
    """Abstract base class for vector stores."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize vector store with configuration."""
        self.config = config

    @abstractmethod
    async def initialize(self):
        """Initialize the vector store connection."""
        pass

    @abstractmethod
    async def store_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Store chunks with embeddings."""
        pass

    @abstractmethod
    async def vector_search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        study_id: Optional[str] = None,
        document_types: Optional[List[str]] = None,
        threshold: float = 0.7,
    ) -> List[Tuple[str, float]]:
        """Perform vector similarity search."""
        pass

    @abstractmethod
    async def full_text_search(
        self,
        query_text: str,
        top_k: int = 10,
        study_id: Optional[str] = None,
        document_types: Optional[List[str]] = None,
    ) -> List[Tuple[str, float]]:
        """Perform full-text search."""
        pass

    @abstractmethod
    async def hybrid_search(
        self,
        query_embedding: List[float],
        query_text: str,
        top_k: int = 10,
        study_id: Optional[str] = None,
        document_types: Optional[List[str]] = None,
        dense_weight: float = 0.7,
    ) -> List[Tuple[str, float]]:
        """Perform hybrid search combining vector and text."""
        pass

    @abstractmethod
    async def get_chunks_by_ids(self, chunk_ids: List[str]) -> List[Dict[str, Any]]:
        """Get chunks by their IDs."""
        pass

    @abstractmethod
    async def get_statistics(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        pass

    @abstractmethod
    async def close(self):
        """Close connections and cleanup."""
        pass
