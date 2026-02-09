"""Advanced hybrid retrieval with optimized search strategies."""

from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import asyncio

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

from advance_rag.core.config import get_settings
from advance_rag.core.logging import get_logger
from advance_rag.models import Query, QueryMode, RetrievalResult
from advance_rag.retrieval.vector_stores import BaseVectorStore
from advance_rag.services.query_service import LLMService

logger = get_logger(__name__)
settings = get_settings()


class SearchStrategy(Enum):
    """Search strategies for different query types."""

    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"


@dataclass
class SearchResult:
    """Enhanced search result with multiple scores."""

    chunk_id: str
    semantic_score: float
    keyword_score: float
    hybrid_score: float
    source: str
    metadata: Dict[str, Any]


class QueryAnalyzer:
    """Analyzes queries to determine optimal search strategy."""

    def __init__(self):
        """Initialize query analyzer."""
        self.question_words = {
            "what",
            "where",
            "when",
            "who",
            "why",
            "how",
            "which",
            "whose",
        }
        self.medical_terms = {
            "adverse",
            "efficacy",
            "safety",
            "dosage",
            "administration",
            "hemoglobin",
            "creatinine",
            "laboratory",
            "clinical",
            "trial",
        }

    def analyze_query(self, query_text: str) -> Dict[str, Any]:
        """Analyze query to determine optimal strategy."""
        query_lower = query_text.lower()

        # Check for question patterns
        is_question = any(word in query_lower for word in self.question_words)

        # Check for medical terminology
        medical_term_count = sum(
            1 for term in self.medical_terms if term in query_lower
        )

        # Check for specific identifiers (subjects, studies, etc.)
        has_identifiers = bool(
            re.search(r"\b[A-Z]{2}\d{4}\b", query_text)  # Subject/study IDs
            or re.search(r"\b(?:trt|ae|lb)\d*\b", query_lower)  # Domain codes
        )

        # Determine query complexity
        word_count = len(query_lower.split())
        is_complex = word_count > 10 or medical_term_count > 2

        # Select optimal strategy
        if has_identifiers:
            strategy = SearchStrategy.KEYWORD
        elif is_complex and medical_term_count > 1:
            strategy = SearchStrategy.SEMANTIC
        elif is_question:
            strategy = SearchStrategy.HYBRID
        else:
            strategy = SearchStrategy.ADAPTIVE

        return {
            "strategy": strategy,
            "is_question": is_question,
            "medical_term_count": medical_term_count,
            "has_identifiers": has_identifiers,
            "is_complex": is_complex,
            "word_count": word_count,
        }


class AdvancedHybridRetriever:
    """Advanced hybrid retriever with adaptive strategies."""

    def __init__(
        self,
        vector_store: BaseVectorStore,
        embedding_service,
        llm_service: Optional[LLMService] = None,
    ):
        """Initialize advanced hybrid retriever."""
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.llm_service = llm_service or LLMService()
        self.query_analyzer = QueryAnalyzer()
        self.tfidf_vectorizer = None
        self.query_cache = {}
        logger.info("Initialized AdvancedHybridRetriever")

    async def retrieve(self, query: Query) -> List[RetrievalResult]:
        """Retrieve relevant chunks using adaptive strategy."""
        # Check cache first
        cache_key = self._get_cache_key(query)
        if cache_key in self.query_cache:
            logger.info(f"Using cached results for query: {query.text[:50]}...")
            return self.query_cache[cache_key]

        # Generate query embedding once to avoid redundant calls
        query_embedding = await self.embedding_service.generate_embedding(query.text)

        # Analyze query to determine strategy
        query_analysis = self.query_analyzer.analyze_query(query.text)
        strategy = query_analysis["strategy"]

        logger.info(f"Using {strategy.value} strategy for query: {query.text[:50]}...")

        # Execute search based on strategy
        if strategy == SearchStrategy.SEMANTIC:
            results = await self._semantic_search(query, query_embedding)
        elif strategy == SearchStrategy.KEYWORD:
            results = await self._keyword_search(query)
        elif strategy == SearchStrategy.HYBRID:
            results = await self._hybrid_search(query, query_embedding)
        else:  # ADAPTIVE
            results = await self._adaptive_search(
                query, query_analysis, query_embedding
            )

        # Apply advanced reranking
        results = await self._advanced_rerank(results, query)

        # Cache results
        self.query_cache[cache_key] = results

        return results

    async def _llm_query_expansion(self, query_text: str) -> List[str]:
        """Use LLM for intelligent query expansion."""
        try:
            expansion_prompt = f"""
            You are a clinical data expert. Expand the following query with relevant synonyms and related terms for clinical trial data analysis.
            
            Original query: "{query_text}"
            
            Provide 3-5 expanded queries that would help find relevant clinical documents.
            Focus on:
            - Medical terminology and synonyms
            - CDISC standard terms (ADaM, SDTM)
            - Clinical trial concepts
            - Related medical concepts
            
            Return only the expanded queries, one per line, without numbering.
            """

            # Use the LLM service for expansion (cached to avoid repeated calls)
            expansion_result = await self.llm_service.generate_response(
                expansion_prompt, max_tokens=200, temperature=0.3
            )

            # Parse the response
            expanded_queries = []
            for line in expansion_result.strip().split("\n"):
                line = line.strip()
                if line and line != query_text:
                    expanded_queries.append(line)

            return expanded_queries[:3]  # Limit to 3 expansions

        except Exception as e:
            logger.warning(
                f"LLM query expansion failed: {e}, falling back to manual expansion"
            )
            # Fallback to manual expansion
            return await self._expand_query(query_text)

    async def _semantic_search(
        self, query: Query, query_embedding: np.ndarray
    ) -> List[RetrievalResult]:
        """Perform semantic search with intelligent query expansion."""
        # Use LLM-based query expansion for better semantic understanding
        expanded_queries = await self._llm_query_expansion(query.text)

        # Perform searches with original and expanded queries in parallel
        search_tasks = []

        # Original query with pre-computed embedding
        search_tasks.append(
            self.vector_store.vector_search(
                query_embedding=query_embedding.tolist(),
                top_k=query.top_k,
                study_id=query.study_id,
                document_types=query.filters.get("document_types"),
                threshold=settings.VECTOR_SIMILARITY_THRESHOLD,
            )
        )

        # Expanded queries (limit to avoid too many API calls)
        for expanded_query in expanded_queries[:2]:  # Limit expansions
            expanded_embedding = await self.embedding_service.generate_embedding(
                expanded_query
            )
            search_tasks.append(
                self.vector_store.vector_search(
                    query_embedding=expanded_embedding.tolist(),
                    top_k=query.top_k // 2,  # Fewer results for expansions
                    study_id=query.study_id,
                    document_types=query.filters.get("document_types"),
                    threshold=settings.VECTOR_SIMILARITY_THRESHOLD
                    * 0.8,  # Lower threshold for expansions
                )
            )

        # Execute all searches in parallel
        all_results = await asyncio.gather(*search_tasks)

        # Merge results from all searches
        merged_results = []
        for results in all_results:
            merged_results.extend(results)

        # Remove duplicates and rerank
        unique_results = self._deduplicate_results(merged_results)
        return await self._create_retrieval_results(unique_results, "semantic")

    async def _keyword_search(self, query: Query) -> List[RetrievalResult]:
        """Perform keyword search with advanced matching."""
        # Extract key terms
        key_terms = self._extract_key_terms(query.text)

        # Build search query with Boolean operators
        search_query = self._build_boolean_query(key_terms)

        # Perform full-text search
        results = await self.vector_store.full_text_search(
            query_text=search_query,
            top_k=query.top_k,
            study_id=query.study_id,
            document_types=query.filters.get("document_types"),
        )

        return await self._create_retrieval_results(results, "keyword")

    async def _hybrid_search(
        self, query: Query, query_embedding: np.ndarray
    ) -> List[RetrievalResult]:
        """Perform optimized hybrid search using vector store capabilities."""
        # Use vector store's built-in hybrid search for better performance
        combined_results = await self.vector_store.hybrid_search(
            query_embedding=query_embedding.tolist(),
            query_text=query.text,
            top_k=query.top_k,
            study_id=query.study_id,
            document_types=query.filters.get("document_types"),
            dense_weight=self._calculate_adaptive_weights(query),
        )

        return await self._create_retrieval_results(combined_results, "hybrid")

    async def _adaptive_search(
        self, query: Query, analysis: Dict[str, Any], query_embedding: np.ndarray
    ) -> List[RetrievalResult]:
        """Perform adaptive search based on query analysis."""
        # Start with hybrid search
        hybrid_results = await self._hybrid_search(query, query_embedding)

        # If query has identifiers, boost exact matches
        if analysis["has_identifiers"]:
            hybrid_results = self._boost_identifier_matches(hybrid_results, query.text)

        # If complex query, add semantic expansion
        if analysis["is_complex"]:
            # Use semantic search with expanded queries for complex queries
            expanded_results = await self._semantic_search(query, query_embedding)
            hybrid_results = self._merge_and_rerank(hybrid_results, expanded_results)

        return hybrid_results

    async def _expand_query(self, query_text: str) -> List[str]:
        """Expand query with medical synonyms and related terms."""
        # Simple medical term expansion
        expansions = {
            "adverse event": ["ae", "side effect", "adverse reaction"],
            "treatment": ["therapy", "medication", "drug", "intervention"],
            "laboratory": ["lab", "test", "measurement", "assay"],
            "efficacy": ["effectiveness", "outcome", "response"],
            "safety": ["tolerability", "toxicity", "adverse events"],
        }

        expanded_queries = []
        query_lower = query_text.lower()

        for term, synonyms in expansions.items():
            if term in query_lower:
                for synonym in synonyms[:2]:  # Limit expansions
                    expanded_query = query_lower.replace(term, synonym)
                    expanded_queries.append(expanded_query)

        return expanded_queries

    def _extract_key_terms(self, query_text: str) -> List[str]:
        """Extract key terms from query."""
        # Simple keyword extraction
        words = query_text.lower().split()
        key_terms = []

        # Filter stopwords and keep medical terms
        stopwords = {"the", "and", "or", "but", "in", "on", "at", "to", "for"}
        medical_terms = self.query_analyzer.medical_terms

        for word in words:
            if (len(word) > 3 and word not in stopwords) or word in medical_terms:
                key_terms.append(word)

        return key_terms

    def _build_boolean_query(self, key_terms: List[str]) -> str:
        """Build Boolean query for full-text search."""
        if not key_terms:
            return ""

        # Use OR operator for broader matching
        return " | ".join(key_terms)

    def _calculate_adaptive_weights(self, query: Query) -> float:
        """Calculate adaptive weights for hybrid search."""
        query_lower = query.text.lower()

        # Base weights
        dense_weight = settings.RETRIEVAL_DENSE_WEIGHT

        # Adjust based on query characteristics
        if any(word in query_lower for word in self.query_analyzer.medical_terms):
            dense_weight += 0.1  # Boost semantic for medical queries

        if len(query.text.split()) > 8:  # Long queries
            dense_weight += 0.1

        if query.mode == QueryMode.SEMANTIC:
            dense_weight += 0.2

        return min(dense_weight, 1.0)

    def _advanced_fusion(
        self,
        vector_results: List[Tuple[str, float]],
        text_results: List[Tuple[str, float]],
        dense_weight: float,
    ) -> List[Tuple[str, float]]:
        """Advanced result fusion using multiple strategies."""
        # Reciprocal Rank Fusion (RRF)
        k = 60
        combined_scores = {}

        # Add vector scores with RRF
        for rank, (doc_id, score) in enumerate(vector_results):
            rrf_score = 1.0 / (k + rank + 1)
            combined_scores[doc_id] = dense_weight * rrf_score

        # Add text scores with RRF
        for rank, (doc_id, score) in enumerate(text_results):
            rrf_score = 1.0 / (k + rank + 1)
            if doc_id in combined_scores:
                combined_scores[doc_id] += (1 - dense_weight) * rrf_score
            else:
                combined_scores[doc_id] = (1 - dense_weight) * rrf_score

        # Sort by combined score
        sorted_results = sorted(
            combined_scores.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_results

    def _deduplicate_results(
        self, results: List[Tuple[str, float]]
    ) -> List[Tuple[str, float]]:
        """Remove duplicate results keeping highest score."""
        seen = {}
        for doc_id, score in results:
            if doc_id not in seen or score > seen[doc_id]:
                seen[doc_id] = score
        return list(seen.items())

    def _boost_identifier_matches(
        self, results: List[RetrievalResult], query_text: str
    ) -> List[RetrievalResult]:
        """Boost results that contain exact identifier matches."""
        # Find identifiers in query
        identifiers = re.findall(r"\b[A-Z]{2}\d{4}\b", query_text)
        if not identifiers:
            return results

        for result in results:
            content_lower = result.chunk.get("content", "").lower()
            boost = 0.0

            for identifier in identifiers:
                if identifier.lower() in content_lower:
                    boost += 0.2

            result.score += boost

        # Re-sort by boosted scores
        return sorted(results, key=lambda x: x.score, reverse=True)

    def _merge_and_rerank(
        self, results1: List[RetrievalResult], results2: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Merge and rerank two result sets."""
        # Combine and deduplicate
        all_results = {}

        for result in results1 + results2:
            chunk_id = result.chunk.get("id", "")
            if chunk_id not in all_results:
                all_results[chunk_id] = result
            else:
                # Keep the one with higher score
                if result.score > all_results[chunk_id].score:
                    all_results[chunk_id] = result

        # Sort by score
        return sorted(all_results.values(), key=lambda x: x.score, reverse=True)

    async def _advanced_rerank(
        self, results: List[RetrievalResult], query: Query
    ) -> List[RetrievalResult]:
        """Apply advanced reranking strategies."""
        if not results:
            return results

        # Diversity-based reranking
        diverse_results = self._diversify_results(results)

        # Freshness boost (recently accessed chunks)
        for result in diverse_results:
            last_accessed = result.chunk.get("last_accessed")
            if last_accessed:
                # Simple freshness boost
                result.score *= 1.1

        # Re-sort final results
        return sorted(diverse_results, key=lambda x: x.score, reverse=True)

    def _diversify_results(
        self, results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Diversify results to avoid redundancy."""
        if len(results) <= 3:
            return results

        diverse_results = [results[0]]  # Always keep top result

        for result in results[1:]:
            # Check similarity with already selected results
            is_similar = False
            for selected in diverse_results:
                if self._are_chunks_similar(result.chunk, selected.chunk):
                    is_similar = True
                    break

            if not is_similar:
                diverse_results.append(result)

            if len(diverse_results) >= results[0].top_k:
                break

        return diverse_results

    def _are_chunks_similar(
        self, chunk1: Dict[str, Any], chunk2: Dict[str, Any]
    ) -> bool:
        """Check if two chunks are too similar."""
        content1 = chunk1.get("content", "")[:200]  # Compare first 200 chars
        content2 = chunk2.get("content", "")[:200]

        # Simple similarity check
        common_words = set(content1.lower().split()) & set(content2.lower().split())
        total_words = set(content1.lower().split()) | set(content2.lower().split())

        if total_words:
            similarity = len(common_words) / len(total_words)
            return similarity > 0.7

        return False

    async def _create_retrieval_results(
        self, results: List[Tuple[str, float]], source: str
    ) -> List[RetrievalResult]:
        """Create RetrievalResult objects from search results."""
        if not results:
            return []

        # Get full chunk data
        chunk_ids = [result[0] for result in results]
        chunks = await self.vector_store.get_chunks_by_ids(chunk_ids)

        chunk_map = {chunk["id"]: chunk for chunk in chunks}
        retrieval_results = []

        for chunk_id, score in results:
            if chunk_id in chunk_map:
                chunk_data = chunk_map[chunk_id]
                retrieval_results.append(
                    RetrievalResult(
                        chunk=chunk_data,
                        score=score,
                        source=source,
                    )
                )

        return retrieval_results

    def _get_cache_key(self, query: Query) -> str:
        """Generate cache key for query."""
        key_data = f"{query.text}_{query.mode}_{query.top_k}"
        if query.filters:
            key_data += f"_{sorted(query.filters.items())}"
        return hash(key_data)
