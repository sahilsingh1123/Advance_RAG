"""Advanced vector store with automated indexing and optimized retrieval."""

from typing import List, Optional, Tuple, Dict, Any
import asyncio
from dataclasses import dataclass
from enum import Enum

import asyncpg
import numpy as np
from pgvector.asyncpg import register_vector
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from advance_rag.core.config import get_settings
from advance_rag.core.logging import get_logger, log_query
from advance_rag.models import Chunk
from .vector_stores.base_store import BaseVectorStore

logger = get_logger(__name__)
settings = get_settings()


class IndexType(Enum):
    """Types of indexes available."""

    HNSW = "hnsw"
    IVFFLAT = "ivfflat"
    EXACT = "exact"


@dataclass
class IndexConfig:
    """Configuration for vector index."""

    index_type: IndexType
    m: int = 16  # HNSW connectivity
    ef_construction: int = 64  # HNSW construction ef
    ef_search: int = 40  # HNSW search ef
    nlist: int = 100  # IVFFLAT number of lists


class VectorStore(BaseVectorStore):
    """Advanced vector store with automated index management."""

    def __init__(self, db_url: str = settings.DATABASE_URL):
        """Initialize advanced vector store."""
        self.db_url = db_url
        self.pool = None
        self.tfidf_vectorizer = None
        self.index_config = self._determine_optimal_index()
        logger.info(
            f"Initialized AdvancedVectorStore with {self.index_config.index_type.value} index"
        )

    def _determine_optimal_index(self) -> IndexConfig:
        """Determine optimal index configuration based on data size and requirements."""
        # For production use, HNSW is generally best
        return IndexConfig(
            index_type=IndexType.HNSW,
            m=settings.VECTOR_HNSW_M,
            ef_construction=settings.VECTOR_HNSW_EF_CONSTRUCTION,
            ef_search=40,  # Runtime ef for better recall
        )

    async def initialize(self):
        """Initialize database connection and create optimized tables."""
        self.pool = await asyncpg.create_pool(self.db_url)

        # Register pgvector extension
        async with self.pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await register_vector(conn)

            # Create enhanced chunks table
            await self._create_enhanced_chunks_table(conn)

            # Create optimal indexes
            await self._create_optimal_indexes(conn)

            # Initialize full-text search
            await self._initialize_full_text_search(conn)

            logger.info("Advanced vector store initialized successfully")

    async def _create_enhanced_chunks_table(self, conn):
        """Create enhanced chunks table with additional optimizations."""
        await conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS chunks (
                id VARCHAR(255) PRIMARY KEY,
                content TEXT NOT NULL,
                document_id VARCHAR(255) NOT NULL,
                document_type VARCHAR(50) NOT NULL,
                chunk_index INTEGER NOT NULL,
                start_char INTEGER NOT NULL,
                end_char INTEGER NOT NULL,
                metadata JSONB,
                embedding vector({settings.EMBEDDING_DIMENSION}),
                content_hash VARCHAR(64) UNIQUE,  -- For deduplication
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                study_id VARCHAR(255),
                access_count INTEGER DEFAULT 0,  -- For popularity-based ranking
                last_accessed TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """
        )

        # Create trigger for updated_at
        await conn.execute(
            """
            CREATE OR REPLACE FUNCTION update_updated_at_column()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.updated_at = NOW();
                RETURN NEW;
            END;
            $$ language 'plpgsql';
        """
        )

        await conn.execute(
            """
            CREATE TRIGGER update_chunks_updated_at 
                BEFORE UPDATE ON chunks 
                FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
        """
        )

    async def _create_optimal_indexes(self, conn):
        """Create optimized indexes based on configuration."""
        if self.index_config.index_type == IndexType.HNSW:
            await conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
                ON chunks
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = {self.index_config.m}, ef_construction = {self.index_config.ef_construction})
            """
            )

        # Composite indexes for common query patterns
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS chunks_study_type_idx 
            ON chunks(study_id, document_type)
        """
        )

        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS chunks_content_hash_idx 
            ON chunks(content_hash)
        """
        )

        # Partial indexes for common filters
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS chunks_active_idx 
            ON chunks(document_id, study_id) 
            WHERE last_accessed > NOW() - INTERVAL '30 days'
        """
        )

    async def _initialize_full_text_search(self, conn):
        """Initialize advanced full-text search."""
        # Create GIN index with configuration
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS chunks_content_fts_idx
            ON chunks USING gin(
                to_tsvector('english', coalesce(content, '') || ' ' || coalesce(document_title, ''))
            )
        """
        )

        # Create materialized view for search optimization
        await conn.execute(
            """
            CREATE MATERIALIZED VIEW IF NOT EXISTS chunks_search_mv AS
            SELECT 
                id,
                content,
                document_id,
                document_type,
                study_id,
                to_tsvector('english', content) as search_vector,
                embedding,
                access_count,
                last_accessed
            FROM chunks
            WITH DATA;
        """
        )

        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS chunks_search_mv_idx 
            ON chunks_search_mv USING gin(search_vector);
        """
        )

    async def store_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Store chunks with deduplication and batch processing."""
        if not chunks:
            return

        # Process in batches for better performance
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            await self._store_chunk_batch(batch)

        # Update materialized view
        async with self.pool.acquire() as conn:
            await conn.execute("REFRESH MATERIALIZED VIEW chunks_search_mv")

        logger.info(f"Stored {len(chunks)} chunks in batches")

    async def _store_chunk_batch(self, chunks: List[Dict[str, Any]]):
        """Store a batch of chunks with UPSERT logic."""
        async with self.pool.acquire() as conn:
            for chunk in chunks:
                # Generate content hash for deduplication
                content_hash = hash(chunk["content"])

                await conn.execute(
                    """
                    INSERT INTO chunks (
                        id, content, document_id, document_type, chunk_index,
                        start_char, end_char, metadata, embedding, content_hash,
                        study_id
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    ON CONFLICT (content_hash) DO UPDATE SET
                        access_count = chunks.access_count + 1,
                        last_accessed = NOW()
                """,
                    chunk["id"],
                    chunk["content"],
                    chunk["document_id"],
                    chunk["document_type"],
                    chunk["chunk_index"],
                    chunk.get("start_char", 0),
                    chunk.get("end_char", 0),
                    json.dumps(chunk.get("metadata", {})),
                    chunk["embedding"],
                    content_hash,
                    chunk.get("study_id"),
                )

    async def vector_search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        study_id: Optional[str] = None,
        document_types: Optional[List[str]] = None,
        threshold: float = 0.7,
    ) -> List[Tuple[str, float]]:
        """Perform optimized vector search."""

        # Set HNSW ef parameter for better recall
        async with self.pool.acquire() as conn:
            await conn.execute(f"SET hnsw.ef_search = {self.index_config.ef_search}")

            # Build dynamic query
            query_conditions = []
            query_params = [query_embedding, threshold]

            if study_id:
                query_conditions.append("study_id = ${len(query_params) + 1}")
                query_params.append(study_id)

            if document_types:
                placeholders = ",".join(
                    [
                        f"${len(query_params) + i + 1}"
                        for i in range(len(document_types))
                    ]
                )
                query_conditions.append(f"document_type = ANY(ARRAY[{placeholders}])")
                query_params.extend(document_types)

            where_clause = " AND ".join(query_conditions) if query_conditions else "1=1"

            query = f"""
                SELECT id, 1 - (embedding <=> $1::vector) as similarity
                FROM chunks 
                WHERE 1 - (embedding <=> $1::vector) > $2
                AND {where_clause}
                ORDER BY similarity DESC, access_count DESC
                LIMIT $3
            """

            query_params.append(top_k)

            results = await conn.fetch(query, *query_params)
            return [(row["id"], row["similarity"]) for row in results]

    async def full_text_search(
        self,
        query_text: str,
        top_k: int = 10,
        study_id: Optional[str] = None,
        document_types: Optional[List[str]] = None,
    ) -> List[Tuple[str, float]]:
        """Perform optimized full-text search using materialized view."""
        async with self.pool.acquire() as conn:
            # Use ts_rank_cd for better ranking
            query_conditions = []
            query_params = [query_text]

            if study_id:
                query_conditions.append(f"study_id = ${len(query_params) + 1}")
                query_params.append(study_id)

            if document_types:
                placeholders = ",".join(
                    [
                        f"${len(query_params) + i + 1}"
                        for i in range(len(document_types))
                    ]
                )
                query_conditions.append(f"document_type = ANY(ARRAY[{placeholders}])")
                query_params.extend(document_types)

            where_clause = " AND ".join(query_conditions) if query_conditions else "1=1"

            query = f"""
                SELECT id, ts_rank_cd(search_vector, plainto_tsquery('english', $1), 32) as rank
                FROM chunks_search_mv
                WHERE search_vector @@ plainto_tsquery('english', $1)
                AND {where_clause}
                ORDER BY rank DESC, access_count DESC
                LIMIT $2
            """

            query_params.append(top_k)

            results = await conn.fetch(query, *query_params)
            return [(row["id"], row["rank"]) for row in results]

    async def hybrid_search(
        self,
        query_embedding: List[float],
        query_text: str,
        top_k: int = 10,
        study_id: Optional[str] = None,
        document_types: Optional[List[str]] = None,
        dense_weight: float = 0.7,
    ) -> List[Tuple[str, float]]:
        """Perform hybrid search combining vector and text search."""

        # Run searches in parallel
        vector_task = self.vector_search(
            query_embedding, top_k * 2, study_id, document_types
        )
        text_task = self.full_text_search(
            query_text, top_k * 2, study_id, document_types
        )

        vector_results, text_results = await asyncio.gather(vector_task, text_task)

        # Combine using reciprocal rank fusion
        combined_scores = self._reciprocal_rank_fusion(
            vector_results, text_results, dense_weight
        )

        # Get top-k results
        sorted_results = sorted(
            combined_scores.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_results[:top_k]

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
        """Get chunks by IDs with access tracking."""
        if not chunk_ids:
            return []

        async with self.pool.acquire() as conn:
            # Update access count
            await conn.execute(
                """
                UPDATE chunks 
                SET access_count = access_count + 1,
                    last_accessed = NOW()
                WHERE id = ANY($1)
            """,
                chunk_ids,
            )

            # Get chunks
            results = await conn.fetch(
                """
                SELECT * FROM chunks WHERE id = ANY($1)
                ORDER BY array_position($1, id)
            """,
                chunk_ids,
            )

            return [dict(row) for row in results]

    async def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        async with self.pool.acquire() as conn:
            stats = await conn.fetchrow(
                """
                SELECT 
                    COUNT(*) as total_chunks,
                    COUNT(DISTINCT document_id) as total_documents,
                    COUNT(DISTINCT study_id) as total_studies,
                    AVG(access_count) as avg_access_count,
                    MAX(access_count) as max_access_count,
                    AVG(token_count) as avg_chunk_size
                FROM chunks
            """
            )

            return dict(stats)

    async def close(self):
        """Close database connection."""
        if self.pool:
            await self.pool.close()
