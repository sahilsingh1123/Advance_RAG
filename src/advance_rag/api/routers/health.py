"""Health check endpoints."""

from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, Depends

from advance_rag.api.service_registry import (
    vector_store,
    embedding_service,
    neo4j_service,
    graphrag_service,
    retriever,
)
from advance_rag.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get("/")
async def health_check() -> Dict[str, Any]:
    """Basic health check."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "advance-rag-api",
        "version": "0.1.0",
    }


@router.get("/detailed")
async def detailed_health_check() -> Dict[str, Any]:
    """Detailed health check with service status."""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "advance-rag-api",
        "version": "0.1.0",
        "services": {},
    }

    # Check vector store
    try:
        if vector_store:
            stats = await vector_store.get_statistics()
            health_status["services"]["vector_store"] = {
                "status": "healthy",
                "total_chunks": stats["total_chunks"],
            }
        else:
            health_status["services"]["vector_store"] = {"status": "uninitialized"}
    except Exception as e:
        logger.error(f"Vector store health check failed: {e}")
        health_status["services"]["vector_store"] = {
            "status": "unhealthy",
            "error": str(e),
        }
        health_status["status"] = "degraded"

    # Check embedding service
    try:
        if embedding_service:
            dimension = embedding_service.get_dimension()
            health_status["services"]["embedding_service"] = {
                "status": "healthy",
                "dimension": dimension,
                "provider": type(embedding_service.provider).__name__,
            }
        else:
            health_status["services"]["embedding_service"] = {"status": "uninitialized"}
    except Exception as e:
        logger.error(f"Embedding service health check failed: {e}")
        health_status["services"]["embedding_service"] = {
            "status": "unhealthy",
            "error": str(e),
        }
        health_status["status"] = "degraded"

    # Check Neo4j service
    try:
        if neo4j_service:
            stats = await neo4j_service.get_graph_statistics()
            health_status["services"]["neo4j"] = {
                "status": "healthy",
                "total_entities": stats["total_entities"],
                "total_relations": stats["total_relations"],
                "total_communities": stats["total_communities"],
            }
        else:
            health_status["services"]["neo4j"] = {"status": "uninitialized"}
    except Exception as e:
        logger.error(f"Neo4j service health check failed: {e}")
        health_status["services"]["neo4j"] = {"status": "unhealthy", "error": str(e)}
        health_status["status"] = "degraded"

    # Check GraphRAG service
    try:
        if graphrag_service:
            health_status["services"]["graphrag"] = {"status": "healthy"}
        else:
            health_status["services"]["graphrag"] = {"status": "uninitialized"}
    except Exception as e:
        logger.error(f"GraphRAG service health check failed: {e}")
        health_status["services"]["graphrag"] = {"status": "unhealthy", "error": str(e)}
        health_status["status"] = "degraded"

    # Check retriever
    try:
        if retriever:
            health_status["services"]["retriever"] = {
                "status": "healthy",
                "type": type(retriever).__name__,
            }
        else:
            health_status["services"]["retriever"] = {"status": "uninitialized"}
    except Exception as e:
        logger.error(f"Retriever health check failed: {e}")
        health_status["services"]["retriever"] = {
            "status": "unhealthy",
            "error": str(e),
        }
        health_status["status"] = "degraded"

    return health_status


@router.get("/ready")
async def readiness_check() -> Dict[str, Any]:
    """Readiness check for Kubernetes."""
    # Check if all critical services are ready
    critical_services = ["vector_store", "embedding_service"]
    ready = True

    for service in critical_services:
        if not locals().get(service):
            ready = False
            break

    return {"ready": ready, "timestamp": datetime.utcnow().isoformat()}


@router.get("/live")
async def liveness_check() -> Dict[str, Any]:
    """Liveness check for Kubernetes."""
    return {"alive": True, "timestamp": datetime.utcnow().isoformat()}
