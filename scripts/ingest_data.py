"""Ingest data into the RAG system."""

import asyncio
import argparse
from pathlib import Path

from advance_rag.ingestion.service import IngestionService
from advance_rag.db.vector_store import VectorStore
from advance_rag.embedding.service import EmbeddingService
from advance_rag.graph.neo4j_service import Neo4jService
from advance_rag.graph.graphrag import GraphRAGService
from advance_rag.core.logging import configure_logging, get_logger

# Configure logging
configure_logging()
logger = get_logger(__name__)


async def main():
    """Main ingestion script."""
    parser = argparse.ArgumentParser(description="Ingest data into Advance RAG")
    parser.add_argument(
        "--data-dir", type=str, required=True, help="Data directory path"
    )
    parser.add_argument("--recursive", action="store_true", help="Search recursively")
    parser.add_argument(
        "--file-pattern", type=str, default="*", help="File pattern (glob)"
    )
    args = parser.parse_args()

    # Validate data directory
    data_path = Path(args.data_dir)
    if not data_path.exists():
        logger.error(f"Data directory not found: {data_path}")
        return

    # Initialize services
    logger.info("Initializing services...")
    vector_store = VectorStore()
    await vector_store.initialize()

    embedding_service = EmbeddingService()

    neo4j_service = Neo4jService()
    await neo4j_service.create_constraints()

    graphrag_service = GraphRAGService(neo4j_service)

    ingestion_service = IngestionService(
        storage_service=vector_store,
        embedding_service=embedding_service,
        graph_service=graphrag_service,
    )

    try:
        # Ingest data
        logger.info(f"Ingesting data from {data_path}")

        if data_path.is_file():
            # Single file
            document_ids = await ingestion_service.ingest_file(str(data_path))
            logger.info(f"Ingested {len(document_ids)} documents")

        elif data_path.is_dir():
            # Directory
            document_ids = await ingestion_service.ingest_directory(
                str(data_path), recursive=args.recursive
            )
            logger.info(f"Ingested {len(document_ids)} documents")

        # Get statistics
        stats = await vector_store.get_statistics()
        logger.info(f"Vector store: {stats['total_chunks']} chunks")

        graph_stats = await neo4j_service.get_graph_statistics()
        logger.info(
            f"Graph: {graph_stats['total_entities']} entities, {graph_stats['total_relations']} relations"
        )

        # Detect communities
        logger.info("Detecting graph communities...")
        await graphrag_service.detect_and_store_communities()

        logger.info("Ingestion complete!")

    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)

    finally:
        # Cleanup
        await neo4j_service.close()
        await vector_store.close()


if __name__ == "__main__":
    asyncio.run(main())
