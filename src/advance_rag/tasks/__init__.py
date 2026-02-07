"""Celery tasks for background processing."""

import os
import time
from typing import List

from celery import current_task
from advance_rag.core.celery import celery_app
from advance_rag.core.logging import get_logger

logger = get_logger(__name__)


@celery_app.task(bind=True)
def ingest_files_task(self, file_paths: List[str]) -> str:
    """Ingest files in background."""
    task_id = self.request.id

    try:
        # Update task state
        self.update_state(
            state="PROGRESS", meta={"status": "Starting ingestion", "progress": 0}
        )

        # Import here to avoid circular imports
        from advance_rag.ingestion.service import IngestionService
        from advance_rag.db.vector_store import VectorStore
        from advance_rag.embedding.service import EmbeddingService
        from advance_rag.graph.neo4j_service import Neo4jService
        from advance_rag.graph.graphrag import GraphRAGService

        # Initialize services
        vector_store = VectorStore()
        embedding_service = EmbeddingService()
        neo4j_service = Neo4jService()
        graphrag_service = GraphRAGService(neo4j_service)

        ingestion_service = IngestionService(
            storage_service=vector_store,
            embedding_service=embedding_service,
            graph_service=graphrag_service,
        )

        # Process files
        total_files = len(file_paths)
        for i, file_path in enumerate(file_paths):
            try:
                await ingestion_service.ingest_file(file_path)

                # Update progress
                progress = (i + 1) / total_files * 100
                self.update_state(
                    state="PROGRESS",
                    meta={
                        "status": f"Processed {i + 1}/{total_files} files",
                        "progress": progress,
                    },
                )

            except Exception as e:
                logger.error(f"Failed to ingest {file_path}: {e}")
                continue

        # Cleanup
        await neo4j_service.close()
        await vector_store.close()

        return f"Successfully ingested {total_files} files"

    except Exception as e:
        logger.error(f"Ingestion task {task_id} failed: {e}")
        self.update_state(state="FAILURE", meta={"error": str(e)})
        raise


@celery_app.task
def generate_embeddings_task(chunk_ids: List[str]) -> dict:
    """Generate embeddings for chunks."""
    try:
        from advance_rag.db.vector_store import VectorStore
        from advance_rag.embedding.service import EmbeddingService

        # Initialize services
        vector_store = VectorStore()
        embedding_service = EmbeddingService()

        # Get chunks
        chunks = await vector_store.get_chunks_by_ids(chunk_ids)

        # Generate embeddings
        texts = [chunk["content"] for chunk in chunks]
        embeddings = await embedding_service.generate_embeddings(texts)

        # Update chunks with embeddings
        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding

        # Store embeddings
        await vector_store.store_embeddings(chunks)

        await vector_store.close()

        return {
            "status": "success",
            "num_chunks": len(chunks),
            "dimension": embedding_service.get_dimension(),
        }

    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise


@celery_app.task
def update_communities_task() -> dict:
    """Update graph communities."""
    try:
        from advance_rag.graph.neo4j_service import Neo4jService
        from advance_rag.graph.graphrag import GraphRAGService

        # Initialize services
        neo4j_service = Neo4jService()
        graphrag_service = GraphRAGService(neo4j_service)

        # Detect and store communities
        await graphrag_service.detect_and_store_communities()

        # Get statistics
        stats = await neo4j_service.get_graph_statistics()

        await neo4j_service.close()

        return {
            "status": "success",
            "communities": stats["total_communities"],
            "entities": stats["total_entities"],
        }

    except Exception as e:
        logger.error(f"Community update failed: {e}")
        raise


@celery_app.task
def cleanup_old_task_results() -> dict:
    """Clean up old task results."""
    try:
        from advance_rag.core.celery import celery_app

        # Get all tasks
        inspect = celery_app.control.inspect()
        active_tasks = inspect.active()
        scheduled_tasks = inspect.scheduled()

        # Count tasks
        active_count = (
            sum(len(tasks) for tasks in active_tasks.values()) if active_tasks else 0
        )
        scheduled_count = (
            sum(len(tasks) for tasks in scheduled_tasks.values())
            if scheduled_tasks
            else 0
        )

        return {
            "status": "success",
            "active_tasks": active_count,
            "scheduled_tasks": scheduled_count,
        }

    except Exception as e:
        logger.error(f"Cleanup task failed: {e}")
        raise
