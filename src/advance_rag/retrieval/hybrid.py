"""Hybrid retrieval combining vector and lexical search."""

from typing import List, Tuple

from advance_rag.core.config import get_settings
from advance_rag.core.logging import get_logger
from advance_rag.models import Query, RetrievalResult
from advance_rag.retrieval.advanced_hybrid import AdvancedHybridRetriever

logger = get_logger(__name__)
settings = get_settings()


class HybridRetriever(AdvancedHybridRetriever):
    """Hybrid retrieval combining dense (vector) and sparse (lexical) search."""

    def __init__(self, vector_store, embedding_service):
        """Initialize hybrid retriever."""
        super().__init__(vector_store, embedding_service)
        self.dense_weight = settings.RETRIEVAL_DENSE_WEIGHT
        self.sparse_weight = settings.RETRIEVAL_SPARSE_WEIGHT
        logger.info("Initialized HybridRetriever (using AdvancedHybridRetriever)")
