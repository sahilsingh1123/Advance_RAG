"""Pinecone vector store implementation."""

import os
import asyncio
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime
import json

import pinecone
from pinecone import Pinecone, ServerlessSpec, PodSpec

from advance_rag.core.config import get_settings
from advance_rag.core.logging import get_logger
from .base_store import BaseVectorStore

logger = get_logger(__name__)
settings = get_settings()


class PineconeVectorStore(BaseVectorStore):
    """Pinecone implementation of vector store."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize Pinecone vector store."""
        super().__init__(config)
        self.index_name = config.get("index_name", "advance-rag")
        self.namespace = config.get("namespace", "chunks")
        self.dimension = config.get("dimension", settings.EMBEDDING_DIMENSION)
        self.metric = config.get("metric", "cosine")
        self.batch_size = config.get("batch_size", 100)

        # Pinecone client
        self.pc = None
        self.index = None

        logger.info(f"Initialized PineconeVectorStore with index: {self.index_name}")

    async def initialize(self):
        """Initialize Pinecone connection and index."""
        try:
            # Initialize Pinecone client
            api_key = self.config.get("api_key") or os.getenv("PINECONE_API_KEY")
            if not api_key:
                raise ValueError("Pinecone API key is required")

            self.pc = Pinecone(api_key=api_key)

            # Check if index exists
            if self.index_name not in self.pc.list_indexes().names():
                logger.info(f"Creating new Pinecone index: {self.index_name}")
                self._create_index()
            else:
                logger.info(f"Using existing Pinecone index: {self.index_name}")

            # Get index connection
            self.index = self.pc.Index(self.index_name)

            # Wait for index to be ready
            self._wait_for_index_ready()

            logger.info("Pinecone vector store initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            raise

    def _create_index(self):
        """Create a new Pinecone index."""
        # Choose spec based on configuration
        if self.config.get("serverless", True):
            spec = ServerlessSpec(
                cloud=self.config.get("cloud", "aws"),
                region=self.config.get("region", "us-west-2"),
            )
        else:
            spec = PodSpec(
                environment=self.config.get("environment", "gcp-starter"),
                pod_type=self.config.get("pod_type", "p1.x1"),
            )

        self.pc.create_index(
            name=self.index_name,
            dimension=self.dimension,
            metric=self.metric,
            spec=spec,
        )

    def _wait_for_index_ready(self):
        """Wait for index to be ready."""
        import time

        max_wait = 300  # 5 minutes
        start_time = time.time()

        while time.time() - start_time < max_wait:
            try:
                stats = self.pc.describe_index(self.index_name)
                if stats.status and stats.status.ready:
                    logger.info("Pinecone index is ready")
                    return
                logger.info(f"Index status: {stats.status.state}")
            except Exception as e:
                logger.warning(f"Error checking index status: {e}")

            time.sleep(10)

        raise TimeoutError("Index not ready within timeout period")

    async def store_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Store chunks in Pinecone with metadata."""
        if not chunks:
            return

        logger.info(f"Storing {len(chunks)} chunks in Pinecone")

        # Process in batches
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i : i + self.batch_size]
            await self._store_batch(batch)

        logger.info(f"Successfully stored {len(chunks)} chunks")

    async def _store_batch(self, batch: List[Dict[str, Any]]):
        """Store a batch of chunks."""
        vectors = []

        for chunk in batch:
            # Prepare metadata
            metadata = {
                "content": chunk.get("content", "")[:1000],  # Limit metadata size
                "document_id": chunk.get("document_id"),
                "document_type": chunk.get("document_type"),
                "chunk_index": chunk.get("chunk_index"),
                "study_id": chunk.get("study_id"),
                "created_at": chunk.get("created_at", datetime.utcnow().isoformat()),
                **chunk.get("metadata", {}),
            }

            # Create vector
            vector = {
                "id": chunk["id"],
                "values": chunk["embedding"],
                "metadata": metadata,
            }
            vectors.append(vector)

        # Upsert to Pinecone
        try:
            self.index.upsert(vectors=vectors, namespace=self.namespace)
        except Exception as e:
            logger.error(f"Failed to upsert batch: {e}")
            raise

    async def vector_search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        study_id: Optional[str] = None,
        document_types: Optional[List[str]] = None,
        threshold: float = 0.7,
    ) -> List[Tuple[str, float]]:
        """Perform vector similarity search in Pinecone."""
        try:
            # Build filter
            filter_dict = self._build_filter(study_id, document_types)

            # Query Pinecone
            response = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                namespace=self.namespace,
                filter=filter_dict if filter_dict else None,
                include_values=False,
            )

            # Process results
            results = []
            for match in response.matches:
                if match.score >= threshold:
                    results.append((match.id, match.score))

            return results

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    async def full_text_search(
        self,
        query_text: str,
        top_k: int = 10,
        study_id: Optional[str] = None,
        document_types: Optional[List[str]] = None,
    ) -> List[Tuple[str, float]]:
        """Perform full-text search using metadata filtering."""
        try:
            # For Pinecone, we'll search content metadata
            # Note: This is a simplified implementation
            # For production, consider using a dedicated text search engine

            filter_dict = self._build_filter(study_id, document_types)

            # Add text filter (basic implementation)
            if query_text:
                filter_dict = filter_dict or {}
                filter_dict["content"] = {"$contains": query_text}

            # Use a dummy embedding for text-only search
            # In production, you'd use a proper text search engine
            dummy_embedding = [0.0] * self.dimension

            response = self.index.query(
                vector=dummy_embedding,
                top_k=top_k,
                include_metadata=True,
                namespace=self.namespace,
                filter=filter_dict,
                include_values=False,
            )

            results = []
            for match in response.matches:
                # Simple relevance scoring based on content match
                relevance = self._calculate_text_relevance(
                    query_text, match.metadata.get("content", "")
                )
                results.append((match.id, relevance))

            return sorted(results, key=lambda x: x[1], reverse=True)

        except Exception as e:
            logger.error(f"Full-text search failed: {e}")
            return []

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
        try:
            # Run both searches in parallel
            vector_task = self.vector_search(
                query_embedding, top_k * 2, study_id, document_types, threshold=0.5
            )
            text_task = self.full_text_search(
                query_text, top_k * 2, study_id, document_types
            )

            vector_results, text_results = await asyncio.gather(vector_task, text_task)

            # Combine using reciprocal rank fusion
            combined_scores = self._reciprocal_rank_fusion(
                vector_results, text_results, dense_weight
            )

            # Return top-k results
            sorted_results = sorted(
                combined_scores.items(), key=lambda x: x[1], reverse=True
            )
            return sorted_results[:top_k]

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []

    def _build_filter(
        self, study_id: Optional[str], document_types: Optional[List[str]]
    ) -> Optional[Dict[str, Any]]:
        """Build Pinecone filter from parameters."""
        filters = {}

        if study_id:
            filters["study_id"] = study_id

        if document_types:
            filters["document_type"] = {"$in": document_types}

        return filters if filters else None

    def _calculate_text_relevance(self, query: str, content: str) -> float:
        """Calculate simple text relevance score."""
        if not content:
            return 0.0

        query_lower = query.lower()
        content_lower = content.lower()

        # Simple relevance based on term frequency
        score = 0.0
        query_terms = query_lower.split()

        for term in query_terms:
            if term in content_lower:
                score += content_lower.count(term) / len(content_lower.split())

        return min(score, 1.0)

    def _reciprocal_rank_fusion(
        self,
        vector_results: List[Tuple[str, float]],
        text_results: List[Tuple[str, float]],
        dense_weight: float,
    ) -> Dict[str, float]:
        """Combine results using reciprocal rank fusion."""
        k = 60  # RRF constant
        combined_scores = {}

        # Add vector scores
        for rank, (doc_id, score) in enumerate(vector_results):
            combined_scores[doc_id] = dense_weight * (1.0 / (k + rank + 1))

        # Add text scores
        for rank, (doc_id, score) in enumerate(text_results):
            if doc_id in combined_scores:
                combined_scores[doc_id] += (1 - dense_weight) * (1.0 / (k + rank + 1))
            else:
                combined_scores[doc_id] = (1 - dense_weight) * (1.0 / (k + rank + 1))

        return combined_scores

    async def get_chunks_by_ids(self, chunk_ids: List[str]) -> List[Dict[str, Any]]:
        """Get chunks by IDs from Pinecone."""
        try:
            # Fetch vectors by ID
            response = self.index.fetch(ids=chunk_ids, namespace=self.namespace)

            chunks = []
            for vector_id, vector_data in response.vectors.items():
                chunk = {
                    "id": vector_id,
                    "content": vector_data.metadata.get("content", ""),
                    "document_id": vector_data.metadata.get("document_id"),
                    "document_type": vector_data.metadata.get("document_type"),
                    "chunk_index": vector_data.metadata.get("chunk_index"),
                    "study_id": vector_data.metadata.get("study_id"),
                    "metadata": vector_data.metadata,
                    "embedding": (
                        vector_data.values if hasattr(vector_data, "values") else []
                    ),
                }
                chunks.append(chunk)

            # Sort by input order
            chunk_dict = {chunk["id"]: chunk for chunk in chunks}
            return [
                chunk_dict[chunk_id] for chunk_id in chunk_ids if chunk_id in chunk_dict
            ]

        except Exception as e:
            logger.error(f"Failed to fetch chunks: {e}")
            return []

    async def get_statistics(self) -> Dict[str, Any]:
        """Get Pinecone index statistics."""
        try:
            stats = self.pc.describe_index(self.index_name)

            return {
                "total_chunks": stats.dimension,  # Approximate
                "index_name": self.index_name,
                "dimension": stats.dimension,
                "metric": stats.metric,
                "status": stats.status.state if stats.status else "unknown",
                "namespace": self.namespace,
            }

        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}

    async def close(self):
        """Close Pinecone connection."""
        try:
            if self.pc:
                self.pc.close()
            logger.info("Pinecone connection closed")
        except Exception as e:
            logger.error(f"Error closing Pinecone: {e}")
