"""Main FastAPI application."""

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from prometheus_client import Counter, Histogram, generate_latest

from advance_rag.api.dependencies import get_query_service, get_ingestion_service
from advance_rag.api.routers import ingestion, query, health, auth
from advance_rag.core.config import get_settings
from advance_rag.core.logging import (
    configure_logging,
    get_logger,
    log_request,
    log_response,
)
from advance_rag.core.tracing import tracing_config
from advance_rag.core.auth import create_default_admin
from advance_rag.retrieval.vector_store_factory import VectorStoreFactory
from advance_rag.embedding.service import EmbeddingService
from advance_rag.graph.neo4j_service import Neo4jService
from advance_rag.graph.graphrag import GraphRAGService
from advance_rag.ingestion.service import IngestionService
from advance_rag.retrieval.hybrid import AdvancedHybridRetriever
from advance_rag.services.query_service import QueryService, LLMService
from advance_rag.services.notification_service import notification_service
from advance_rag.services.backup_service import backup_service

# Configure logging
configure_logging()
logger = get_logger(__name__)
settings = get_settings()

# Prometheus metrics
REQUEST_COUNT = Counter(
    "http_requests_total", "Total HTTP requests", ["method", "endpoint", "status"]
)

REQUEST_DURATION = Histogram(
    "http_request_duration_seconds", "HTTP request duration", ["method", "endpoint"]
)

QUERY_COUNT = Counter("rag_queries_total", "Total RAG queries", ["mode", "status"])

QUERY_DURATION = Histogram("rag_query_duration_seconds", "RAG query duration", ["mode"])

INGESTION_COUNT = Counter(
    "rag_ingestion_total", "Total documents ingested", ["document_type", "status"]
)

from advance_rag.api.service_registry import (
    vector_store,
    embedding_service,
    neo4j_service,
    graphrag_service,
    retriever,
    query_service,
    ingestion_service,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Advance RAG API...")

    # Setup tracing
    tracing_config.setup_tracing(app)

    # Create default admin user
    create_default_admin()

    try:
        # Initialize services
        global vector_store, embedding_service, neo4j_service
        global graphrag_service, retriever, query_service, ingestion_service

        # Initialize vector store using factory
        vector_store = VectorStoreFactory.create_vector_store()
        await vector_store.initialize()

        # Initialize embedding service
        embedding_service = EmbeddingService()

        # Initialize Neo4j service
        neo4j_service = Neo4jService()
        await neo4j_service.create_constraints()

        # Initialize GraphRAG service
        graphrag_service = GraphRAGService(neo4j_service)

        # Initialize LLM service
        llm_service = LLMService()

        # Initialize retriever with LLM service
        retriever = AdvancedHybridRetriever(
            vector_store, embedding_service, llm_service
        )

        # Initialize services
        query_service = QueryService(
            retriever=retriever,
            embedding_service=embedding_service,
            graphrag_service=graphrag_service,
        )

        ingestion_service = IngestionService(
            storage_service=vector_store,
            embedding_service=embedding_service,
            graph_service=graphrag_service,
        )

        # Set global instances in service registry
        import advance_rag.api.service_registry as registry

        registry.vector_store = vector_store
        registry.embedding_service = embedding_service
        registry.neo4j_service = neo4j_service
        registry.graphrag_service = graphrag_service
        registry.retriever = retriever
        registry.query_service = query_service
        registry.ingestion_service = ingestion_service

        # Set global instances in dependencies module
        from advance_rag.api.dependencies import query_service as dep_query_service
        from advance_rag.api.dependencies import (
            ingestion_service as dep_ingestion_service,
        )

        dep_query_service = query_service
        dep_ingestion_service = ingestion_service

        # Start notification service
        await notification_service.start()

        # Start backup scheduler
        backup_task = asyncio.create_task(backup_service.schedule_backups())

        logger.info("All services initialized successfully")

        yield

    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    finally:
        # Shutdown
        logger.info("Shutting down Advance RAG API...")

        # Stop notification service
        await notification_service.stop()

        if neo4j_service:
            await neo4j_service.close()

        if vector_store:
            await vector_store.close()

        logger.info("Shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Advance RAG API",
    description="Production-Grade RAG System for ADaM/SDTM Analysis",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


# Request logging middleware
@app.middleware("http")
async def log_requests(request, call_next):
    """Log HTTP requests and responses."""
    request_id = str(uuid.uuid4())
    start_time = time.time()

    # Log request
    log_request(
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        query_params=str(request.query_params),
    )

    # Process request
    response = await call_next(request)

    # Calculate duration
    duration_ms = (time.time() - start_time) * 1000

    # Log response
    log_response(
        request_id=request_id,
        status_code=response.status_code,
        duration_ms=duration_ms,
    )

    # Update metrics
    REQUEST_COUNT.labels(
        method=request.method, endpoint=request.url.path, status=response.status_code
    ).inc()

    REQUEST_DURATION.labels(method=request.method, endpoint=request.url.path).observe(
        duration_ms / 1000
    )

    # Add request ID to response headers
    response.headers["X-Request-ID"] = request_id

    return response


# Include routers
app.include_router(health.router, prefix="/health", tags=["Health"])

app.include_router(auth.router, prefix="/v1/auth", tags=["Authentication"])

app.include_router(
    ingestion.router,
    prefix="/v1/ingestion",
    tags=["Ingestion"],
    dependencies=[Depends(get_ingestion_service)],
)

app.include_router(
    query.router,
    prefix="/v1/query",
    tags=["Query"],
    dependencies=[Depends(get_query_service)],
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Advance RAG API",
        "version": "0.1.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    if not settings.METRICS_ENABLED:
        raise HTTPException(status_code=404, detail="Metrics not enabled")

    return generate_latest()


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    return {
        "error": "Internal server error",
        "message": "An unexpected error occurred",
        "request_id": request.headers.get("X-Request-ID"),
    }


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """HTTP exception handler."""
    return {
        "error": exc.detail,
        "status_code": exc.status_code,
        "request_id": request.headers.get("X-Request-ID"),
    }


# Dependency injection
def get_vector_store() -> VectorStore:
    """Get vector store instance."""
    if vector_store is None:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    return vector_store


def get_embedding_service() -> EmbeddingService:
    """Get embedding service instance."""
    if embedding_service is None:
        raise HTTPException(status_code=503, detail="Embedding service not initialized")
    return embedding_service


def get_neo4j_service() -> Neo4jService:
    """Get Neo4j service instance."""
    if neo4j_service is None:
        raise HTTPException(status_code=503, detail="Neo4j service not initialized")
    return neo4j_service


def get_graphrag_service() -> GraphRAGService:
    """Get GraphRAG service instance."""
    if graphrag_service is None:
        raise HTTPException(status_code=503, detail="GraphRAG service not initialized")
    return graphrag_service


def get_retriever() -> AdvancedHybridRetriever:
    """Get retriever instance."""
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")
    return retriever


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "advance_rag.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.RELOAD_ON_CHANGE,
        log_level=settings.LOG_LEVEL.lower(),
    )
