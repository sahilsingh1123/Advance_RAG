"""Hybrid retrieval combining vector and lexical search."""

from typing import List, Tuple

from advance_rag.core.config import get_settings
from advance_rag.core.logging import get_logger
from advance_rag.models import Query, RetrievalResult
from advance_rag.retrieval.vector_store import VectorStore

logger = get_logger(__name__)
settings = get_settings()


class HybridRetriever:
    """Hybrid retrieval combining dense (vector) and sparse (lexical) search."""

    def __init__(self, vector_store: VectorStore, embedding_service):
        """Initialize hybrid retriever."""
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.dense_weight = settings.RETRIEVAL_DENSE_WEIGHT
        self.sparse_weight = settings.RETRIEVAL_SPARSE_WEIGHT
        logger.info("Initialized HybridRetriever")

    async def retrieve(self, query: Query) -> List[RetrievalResult]:
        """Retrieve relevant chunks using hybrid search."""
        # Generate query embedding
        query_embedding = await self.embedding_service.generate_embedding(query.text)

        # Perform parallel searches
        dense_results = await self._dense_search(query, query_embedding)
        sparse_results = await self._sparse_search(query)

        # Combine and rerank results
        combined_results = self._combine_results(
            dense_results, sparse_results, query.top_k
        )

        # Get full chunk data
        chunk_ids = [result[0] for result in combined_results]
        chunks = await self.vector_store.get_chunks_by_ids(chunk_ids)

        # Create RetrievalResult objects
        retrieval_results = []
        chunk_map = {chunk["id"]: chunk for chunk in chunks}

        for chunk_id, combined_score in combined_results:
            if chunk_id in chunk_map:
                chunk_data = chunk_map[chunk_id]
                retrieval_results.append(
                    RetrievalResult(
                        chunk=chunk_data,  # Will be converted to Chunk object
                        score=combined_score,
                        source="hybrid",
                    )
                )

        return retrieval_results

    async def _dense_search(
        self, query: Query, query_embedding: List[float]
    ) -> List[Tuple[str, float]]:
        """Perform dense vector search."""
        results = await self.vector_store.vector_search(
            query_embedding=query_embedding,
            top_k=query.top_k,
            study_id=query.study_id,
            document_types=query.filters.get("document_types"),
            threshold=settings.VECTOR_SIMILARITY_THRESHOLD,
        )

        logger.info(f"Dense search returned {len(results)} results")
        return results

    async def _sparse_search(self, query: Query) -> List[Tuple[str, float]]:
        """Perform sparse lexical search."""
        results = await self.vector_store.full_text_search(
            query_text=query.text,
            top_k=query.top_k,
            study_id=query.study_id,
            document_types=query.filters.get("document_types"),
        )

        logger.info(f"Sparse search returned {len(results)} results")
        return results

    def _combine_results(
        self,
        dense_results: List[Tuple[str, float]],
        sparse_results: List[Tuple[str, float]],
        top_k: int,
    ) -> List[Tuple[str, float]]:
        """Combine dense and sparse results using reciprocal rank fusion."""
        # Create score maps
        dense_scores = {doc_id: score for doc_id, score in dense_results}
        sparse_scores = {doc_id: score for doc_id, score in sparse_results}

        # Get all unique document IDs
        all_doc_ids = set(dense_scores.keys()) | set(sparse_scores.keys())

        # Calculate combined scores
        combined_scores = []
        for doc_id in all_doc_ids:
            dense_score = dense_scores.get(doc_id, 0)
            sparse_score = sparse_scores.get(doc_id, 0)

            # Weighted combination
            combined_score = (
                self.dense_weight * dense_score + self.sparse_weight * sparse_score
            )

            combined_scores.append((doc_id, combined_score))

        # Sort by combined score and return top-k
        combined_scores.sort(key=lambda x: x[1], reverse=True)
        return combined_scores[:top_k]


class ReciprocalRankFusion:
    """Reciprocal Rank Fusion for combining multiple result lists."""

    def __init__(self, k: int = 60):
        """Initialize RRF with constant k."""
        self.k = k

    def fuse(
        self, result_lists: List[List[Tuple[str, float]]]
    ) -> List[Tuple[str, float]]:
        """Fuse multiple result lists using RRF."""
        # Collect all document IDs
        all_doc_ids = set()
        for results in result_lists:
            all_doc_ids.update(doc_id for doc_id, _ in results)

        # Calculate RRF scores
        rrf_scores = {}
        for doc_id in all_doc_ids:
            score = 0
            for results in result_lists:
                # Find rank of document in this list
                for rank, (result_doc_id, _) in enumerate(results, 1):
                    if result_doc_id == doc_id:
                        score += 1.0 / (self.k + rank)
                        break
            rrf_scores[doc_id] = score

        # Sort by RRF score
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        return sorted_results


class CrossEncoderReranker:
    """Reranker using cross-encoder models."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """Initialize cross-encoder reranker."""
        try:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(model_name)
            logger.info(f"Loaded CrossEncoder model: {model_name}")
        except ImportError:
            logger.warning("CrossEncoder not available, reranking disabled")
            self.model = None

    async def rerank(
        self,
        query: str,
        documents: List[dict],
        top_k: int = settings.RETRIEVAL_RERANK_TOP_K,
    ) -> List[Tuple[int, float]]:
        """Rerank documents using cross-encoder."""
        if not self.model:
            # Return original order if reranker not available
            return [(i, 1.0) for i in range(len(documents))]

        # Prepare query-document pairs
        pairs = [(query, doc["content"]) for doc in documents]

        # Predict scores
        scores = self.model.predict(pairs)

        # Create list of (index, score) tuples
        indexed_scores = list(enumerate(scores))

        # Sort by score and return top-k
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        return indexed_scores[:top_k]


class AdvancedHybridRetriever(HybridRetriever):
    """Advanced hybrid retriever with reranking."""

    def __init__(self, vector_store: VectorStore, embedding_service):
        """Initialize advanced hybrid retriever."""
        super().__init__(vector_store, embedding_service)
        self.rrf = ReciprocalRankFusion()
        self.reranker = CrossEncoderReranker()

    async def retrieve(self, query: Query) -> List[RetrievalResult]:
        """Retrieve with advanced fusion and reranking."""
        # Generate query embedding
        query_embedding = await self.embedding_service.generate_embedding(query.text)

        # Perform parallel searches
        dense_results = await self._dense_search(query, query_embedding)
        sparse_results = await self._sparse_search(query)

        # Fuse results using RRF
        fused_results = self.rrf.fuse([dense_results, sparse_results])

        # Get candidate documents
        candidate_ids = [doc_id for doc_id, _ in fused_results[: query.top_k * 2]]
        candidates = await self.vector_store.get_chunks_by_ids(candidate_ids)

        # Rerank if enabled
        if len(candidates) > settings.RETRIEVAL_RERANK_TOP_K:
            reranked_indices = await self.reranker.rerank(
                query.text, candidates, settings.RETRIEVAL_RERANK_TOP_K
            )

            # Create final results
            final_results = []
            for idx, score in reranked_indices:
                chunk_data = candidates[idx]
                final_results.append(
                    RetrievalResult(
                        chunk=chunk_data, score=score, source="hybrid_reranked"
                    )
                )
        else:
            # No reranking needed
            final_results = []
            for chunk_data in candidates:
                final_results.append(
                    RetrievalResult(chunk=chunk_data, score=1.0, source="hybrid")
                )

        return final_results
