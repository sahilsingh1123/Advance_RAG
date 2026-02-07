"""Vector store implementation using PostgreSQL with pgvector."""

from typing import List, Optional, Tuple

import asyncpg
import numpy as np
from pgvector.asyncpg import register_vector

from advance_rag.core.config import get_settings
from advance_rag.core.logging import get_logger, log_query

logger = get_logger(__name__)
settings = get_settings()


class VectorStore:
    """PostgreSQL pgvector implementation for vector storage."""

    def __init__(self, db_url: str = settings.DATABASE_URL):
        """Initialize vector store."""
        self.db_url = db_url
        self.pool = None
        logger.info("Initialized VectorStore with PostgreSQL pgvector")

    async def initialize(self):
        """Initialize database connection and create tables."""
        self.pool = await asyncpg.create_pool(self.db_url)

        # Register pgvector extension
        async with self.pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await register_vector(conn)

            # Create chunks table with vector column
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id VARCHAR(255) PRIMARY KEY,
                    content TEXT NOT NULL,
                    document_id VARCHAR(255) NOT NULL,
                    document_type VARCHAR(50) NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    start_char INTEGER NOT NULL,
                    end_char INTEGER NOT NULL,
                    metadata JSONB,
                    embedding vector({dimension}),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    study_id VARCHAR(255)
                )
            """.format(
                    dimension=settings.EMBEDDING_DIMENSION
                )
            )

            # Create HNSW index for fast vector search
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS chunks_embedding_idx
                ON chunks
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = {m}, ef_construction = {ef_construction})
            """.format(
                    m=settings.VECTOR_HNSW_M,
                    ef_construction=settings.VECTOR_HNSW_EF_CONSTRUCTION,
                )
            )

            # Create additional indexes
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS chunks_document_id_idx
                ON chunks(document_id)
            """
            )

            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS chunks_study_id_idx
                ON chunks(study_id)
            """
            )

            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS chunks_document_type_idx
                ON chunks(document_type)
            """
            )

            # Create full-text search index
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS chunks_content_fts_idx
                ON chunks USING gin(to_tsvector('english', content))
            """
            )

            logger.info("Database tables and indexes created")

    async def store_chunks(self, chunks: List[dict]) -> None:
        """Store chunks with embeddings."""
        if not self.pool:
            await self.initialize()

        async with self.pool.acquire() as conn:
            await register_vector(conn)

            # Prepare batch insert
            values = []
            for chunk in chunks:
                values.append(
                    (
                        chunk["id"],
                        chunk["content"],
                        chunk["document_id"],
                        chunk["document_type"],
                        chunk["chunk_index"],
                        chunk["start_char"],
                        chunk["end_char"],
                        chunk.get("metadata", {}),
                        (
                            np.array(chunk["embedding"])
                            if chunk.get("embedding")
                            else None
                        ),
                        chunk.get("study_id"),
                    )
                )

            # Execute batch insert
            await conn.executemany(
                """
                INSERT INTO chunks (
                    id, content, document_id, document_type,
                    chunk_index, start_char, end_char, metadata,
                    embedding, study_id
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                ON CONFLICT (id) DO UPDATE SET
                    content = EXCLUDED.content,
                    metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding,
                    study_id = EXCLUDED.study_id
            """,
                values,
            )

            logger.info(f"Stored {len(chunks)} chunks in vector store")

    async def vector_search(
        self,
        query_embedding: List[float],
        top_k: int = settings.RETRIEVAL_TOP_K,
        study_id: Optional[str] = None,
        document_types: Optional[List[str]] = None,
        threshold: float = settings.VECTOR_SIMILARITY_THRESHOLD,
    ) -> List[Tuple[str, float]]:
        """Perform vector similarity search."""
        if not self.pool:
            await self.initialize()

        import time

        start_time = time.time()

        async with self.pool.acquire() as conn:
            await register_vector(conn)

            # Build query
            query = """
                SELECT id, 1 - (embedding <=> $1) as similarity
                FROM chunks
                WHERE 1 - (embedding <=> $1) >= $2
            """
            params = [query_embedding, threshold]

            # Add filters
            if study_id:
                query += " AND study_id = ${}".format(len(params) + 1)
                params.append(study_id)

            if document_types:
                placeholders = ", ".join(
                    "${}".format(i + len(params) + 1)
                    for i in range(len(document_types))
                )
                query += f" AND document_type IN ({placeholders})"
                params.extend(document_types)

            # Add ordering and limit
            query += " ORDER BY similarity DESC LIMIT ${}".format(len(params) + 1)
            params.append(top_k)

            # Execute query
            results = await conn.fetch(query, *params)

            # Log query
            duration_ms = (time.time() - start_time) * 1000
            log_query(
                query_type="vector_search",
                query=f"vector_search(top_k={top_k}, study_id={study_id})",
                duration_ms=duration_ms,
                num_results=len(results),
            )

            return [(row["id"], row["similarity"]) for row in results]

    async def full_text_search(
        self,
        query_text: str,
        top_k: int = settings.RETRIEVAL_TOP_K,
        study_id: Optional[str] = None,
        document_types: Optional[List[str]] = None,
    ) -> List[Tuple[str, float]]:
        """Perform full-text search using PostgreSQL."""
        if not self.pool:
            await self.initialize()

        import time

        start_time = time.time()

        async with self.pool.acquire() as conn:
            # Build query using ts_rank
            query = """
                SELECT id, ts_rank(to_tsvector('english', content), plainto_tsquery('english', $1)) as rank
                FROM chunks
                WHERE to_tsvector('english', content) @@ plainto_tsquery('english', $1)
            """
            params = [query_text]

            # Add filters
            if study_id:
                query += " AND study_id = ${}".format(len(params) + 1)
                params.append(study_id)

            if document_types:
                placeholders = ", ".join(
                    "${}".format(i + len(params) + 1)
                    for i in range(len(document_types))
                )
                query += f" AND document_type IN ({placeholders})"
                params.extend(document_types)

            # Add ordering and limit
            query += " ORDER BY rank DESC LIMIT ${}".format(len(params) + 1)
            params.append(top_k)

            # Execute query
            results = await conn.fetch(query, *params)

            # Log query
            duration_ms = (time.time() - start_time) * 1000
            log_query(
                query_type="full_text_search",
                query=f"fts_search(query='{query_text[:50]}...', top_k={top_k})",
                duration_ms=duration_ms,
                num_results=len(results),
            )

            return [(row["id"], row["rank"]) for row in results]

    async def get_chunks_by_ids(self, chunk_ids: List[str]) -> List[dict]:
        """Retrieve chunks by their IDs."""
        if not self.pool:
            await self.initialize()

        if not chunk_ids:
            return []

        async with self.pool.acquire() as conn:
            # Build query
            placeholders = ", ".join("${}".format(i + 1) for i in range(len(chunk_ids)))
            query = f"""
                SELECT id, content, document_id, document_type,
                       chunk_index, start_char, end_char, metadata,
                       study_id, created_at
                FROM chunks
                WHERE id IN ({placeholders})
                ORDER BY chunk_index
            """

            # Execute query
            results = await conn.fetch(query, *chunk_ids)

            # Convert to dicts
            chunks = []
            for row in results:
                chunk = dict(row)
                # Convert metadata from JSON if needed
                if chunk["metadata"]:
                    chunk["metadata"] = dict(chunk["metadata"])
                chunks.append(chunk)

            return chunks

    async def delete_chunks(self, document_id: str) -> int:
        """Delete all chunks for a document."""
        if not self.pool:
            await self.initialize()

        async with self.pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM chunks WHERE document_id = $1", document_id
            )

            # Extract count from result string
            count = int(result.split()[-1]) if result else 0
            logger.info(f"Deleted {count} chunks for document {document_id}")

            return count

    async def get_statistics(self) -> dict:
        """Get vector store statistics."""
        if not self.pool:
            await self.initialize()

        async with self.pool.acquire() as conn:
            # Get total chunks
            total_chunks = await conn.fetchval("SELECT COUNT(*) FROM chunks")

            # Get chunks by document type
            by_type = await conn.fetch(
                """
                SELECT document_type, COUNT(*) as count
                FROM chunks
                GROUP BY document_type
                ORDER BY count DESC
            """
            )

            # Get chunks by study
            by_study = await conn.fetch(
                """
                SELECT study_id, COUNT(*) as count
                FROM chunks
                WHERE study_id IS NOT NULL
                GROUP BY study_id
                ORDER BY count DESC
                LIMIT 10
            """
            )

            return {
                "total_chunks": total_chunks,
                "by_document_type": [dict(row) for row in by_type],
                "by_study": [dict(row) for row in by_study],
            }

    async def close(self):
        """Close database connection."""
        if self.pool:
            await self.pool.close()
            logger.info("Closed vector store connection")
