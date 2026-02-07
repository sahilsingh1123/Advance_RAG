"""Service registry for global service instances."""

from typing import Optional

from advance_rag.retrieval.vector_store import VectorStore
from advance_rag.embedding.service import EmbeddingService
from advance_rag.graph.neo4j_service import Neo4jService
from advance_rag.graph.graphrag import GraphRAGService
from advance_rag.retrieval.hybrid import AdvancedHybridRetriever
from advance_rag.services.query_service import QueryService
from advance_rag.ingestion.service import IngestionService

# Global service instances
vector_store: Optional[VectorStore] = None
embedding_service: Optional[EmbeddingService] = None
neo4j_service: Optional[Neo4jService] = None
graphrag_service: Optional[GraphRAGService] = None
retriever: Optional[AdvancedHybridRetriever] = None
query_service: Optional[QueryService] = None
ingestion_service: Optional[IngestionService] = None
