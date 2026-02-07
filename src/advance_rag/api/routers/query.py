"""Query endpoints for the RAG system."""

import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from advance_rag.api.dependencies import QueryServiceDep
from advance_rag.core.logging import get_logger, log_query
from advance_rag.models import Query, QueryMode, QueryResponse

logger = get_logger(__name__)
router = APIRouter()


class QueryRequest(BaseModel):
    """Request model for queries."""

    query: str = Field(..., description="Query text")
    mode: QueryMode = Field(QueryMode.QA, description="Query mode")
    study_id: Optional[str] = Field(None, description="Study ID filter")
    filters: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Query filters"
    )
    top_k: int = Field(default=10, description="Number of results to retrieve")
    user_id: Optional[str] = Field(None, description="User ID for tracking")


class QueryResponse(BaseModel):
    """Response model for queries."""

    query_id: str = Field(..., description="Query ID")
    answer: str = Field(..., description="Generated answer")
    sources: List[Dict[str, Any]] = Field(..., description="Source information")
    mode: QueryMode = Field(..., description="Query mode")
    llm_model: str = Field(..., description="LLM model used")
    prompt_tokens: int = Field(..., description="Prompt tokens used")
    completion_tokens: int = Field(..., description="Completion tokens used")
    duration_ms: float = Field(..., description="Query duration in milliseconds")
    created_at: str = Field(..., description="Response timestamp")


class GraphQueryRequest(BaseModel):
    """Request model for graph-specific queries."""

    query: str = Field(..., description="Query text")
    search_type: str = Field(
        "local", description="Graph search type: local, global, drift"
    )
    entity_name: Optional[str] = Field(None, description="Entity name for local search")
    max_hops: int = Field(default=3, description="Maximum graph hops")
    top_k: int = Field(default=5, description="Number of results")


@router.post("/", response_model=QueryResponse)
async def query_rag(
    request: QueryRequest, query_service: QueryServiceDep
) -> QueryResponse:
    """Execute a RAG query."""
    start_time = time.time()

    # Create query object
    query_obj = Query(
        id=str(uuid.uuid4()),
        text=request.query,
        mode=request.mode,
        study_id=request.study_id,
        filters=request.filters or {},
        top_k=request.top_k,
        user_id=request.user_id,
    )

    try:
        # Execute query
        response = await query_service.execute_query(query_obj)

        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000

        # Log query
        log_query(
            query_type="rag_query",
            query=request.query,
            duration_ms=duration_ms,
            num_results=len(response.sources),
            mode=request.mode.value,
        )

        # Convert sources to dict format
        sources = []
        for source in response.sources:
            sources.append(
                {
                    "chunk_id": source.chunk.get("id"),
                    "content": source.chunk.get("content", "")[:500] + "...",
                    "score": source.score,
                    "source": source.source,
                    "document_id": source.chunk.get("document_id"),
                    "document_type": source.chunk.get("document_type"),
                }
            )

        return QueryResponse(
            query_id=response.query_id,
            answer=response.answer,
            sources=sources,
            mode=response.mode,
            llm_model=response.llm_model,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            duration_ms=duration_ms,
            created_at=response.created_at.isoformat(),
        )

    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query execution failed: {str(e)}")


@router.post("/graph", response_model=Dict[str, Any])
async def query_graph(
    request: GraphQueryRequest, query_service: QueryServiceDep
) -> Dict[str, Any]:
    """Execute a graph-based query."""
    start_time = time.time()

    try:
        # Execute graph query
        if request.search_type == "local":
            results = await query_service.graphrag_service.local_search(
                query=request.query,
                entity_name=request.entity_name or "",
                max_hops=request.max_hops,
            )
        elif request.search_type == "global":
            results = await query_service.graphrag_service.global_search(
                query=request.query, top_k=request.top_k
            )
        elif request.search_type == "drift":
            results = await query_service.graphrag_service.drift_search(
                query=request.query,
                entity_name=request.entity_name or "",
                top_k=request.top_k,
            )
        else:
            raise HTTPException(
                status_code=400, detail=f"Invalid search type: {request.search_type}"
            )

        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000

        # Log query
        log_query(
            query_type="graph_query",
            query=request.query,
            duration_ms=duration_ms,
            num_results=len(results),
            search_type=request.search_type,
        )

        return {
            "query": request.query,
            "search_type": request.search_type,
            "results": results,
            "duration_ms": duration_ms,
            "num_results": len(results),
        }

    except Exception as e:
        logger.error(f"Graph query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Graph query failed: {str(e)}")


@router.post("/code-generation", response_model=Dict[str, Any])
async def generate_code(
    request: QueryRequest, query_service: QueryServiceDep
) -> Dict[str, Any]:
    """Generate ADaM/SDTM code."""
    start_time = time.time()

    try:
        # Execute code generation
        response = await query_service.generate_code(
            query_text=request.query, study_id=request.study_id, top_k=request.top_k
        )

        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000

        # Log query
        log_query(
            query_type="code_generation",
            query=request.query,
            duration_ms=duration_ms,
            num_results=len(response.get("sources", [])),
        )

        return {
            "query_id": response.get("query_id"),
            "code": response.get("code"),
            "explanation": response.get("explanation"),
            "sources": response.get("sources", []),
            "duration_ms": duration_ms,
        }

    except Exception as e:
        logger.error(f"Code generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Code generation failed: {str(e)}")


@router.get("/similar/{chunk_id}", response_model=List[Dict[str, Any]])
async def find_similar_chunks(
    chunk_id: str, query_service: QueryServiceDep, top_k: int = 10
) -> List[Dict[str, Any]]:
    """Find chunks similar to a given chunk."""
    try:
        # Get chunk embedding
        chunk = await query_service.vector_store.get_chunks_by_ids([chunk_id])

        if not chunk:
            raise HTTPException(status_code=404, detail="Chunk not found")

        chunk_embedding = chunk[0].get("embedding")
        if not chunk_embedding:
            raise HTTPException(status_code=400, detail="Chunk has no embedding")

        # Find similar chunks
        similar_ids = await query_service.embedding_service.find_most_similar(
            query_embedding=chunk_embedding,
            candidate_embeddings=[],  # Will be fetched from vector store
            top_k=top_k,
        )

        # Get full chunk data
        similar_chunk_ids = [chunk_id for chunk_id, _ in similar_ids]
        similar_chunks = await query_service.vector_store.get_chunks_by_ids(
            similar_chunk_ids
        )

        # Format results
        results = []
        for chunk_data, (chunk_id, similarity) in zip(similar_chunks, similar_ids):
            results.append(
                {
                    "chunk_id": chunk_data.get("id"),
                    "content": chunk_data.get("content", "")[:500] + "...",
                    "similarity": similarity,
                    "document_id": chunk_data.get("document_id"),
                    "document_type": chunk_data.get("document_type"),
                }
            )

        return results

    except Exception as e:
        logger.error(f"Similarity search failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Similarity search failed: {str(e)}"
        )


@router.get("/entities", response_model=List[Dict[str, Any]])
async def search_entities(
    query_service: QueryServiceDep,
    entity_type: Optional[str] = None,
    name_pattern: Optional[str] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """Search for entities in the knowledge graph."""
    try:
        entities = await query_service.graphrag_service.neo4j.find_entities(
            entity_type=entity_type, name_pattern=name_pattern, limit=limit
        )

        return [
            {
                "id": entity.id,
                "type": entity.type,
                "name": entity.name,
                "properties": entity.properties,
            }
            for entity in entities
        ]

    except Exception as e:
        logger.error(f"Entity search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Entity search failed: {str(e)}")


@router.get("/stats", response_model=Dict[str, Any])
async def get_statistics(query_service: QueryServiceDep) -> Dict[str, Any]:
    """Get system statistics."""
    try:
        # Get vector store stats
        vector_stats = await query_service.vector_store.get_statistics()

        # Get graph stats
        graph_stats = await query_service.graphrag_service.neo4j.get_graph_statistics()

        return {
            "vector_store": vector_stats,
            "knowledge_graph": graph_stats,
            "timestamp": time.time(),
        }

    except Exception as e:
        logger.error(f"Statistics retrieval failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Statistics retrieval failed: {str(e)}"
        )
