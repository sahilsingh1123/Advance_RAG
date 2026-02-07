"""API dependencies for dependency injection."""

from typing import Annotated

from fastapi import Depends

from advance_rag.services.query_service import QueryService
from advance_rag.ingestion.service import IngestionService


# Global instances (will be initialized in main.py)
query_service: QueryService = None
ingestion_service: IngestionService = None


async def get_ingestion_service():
    """Get ingestion service instance."""
    return ingestion_service


async def get_query_service():
    """Get query service instance."""
    return query_service


# Type aliases for cleaner signatures
IngestionServiceDep = Annotated[object, Depends(get_ingestion_service)]
QueryServiceDep = Annotated[object, Depends(get_query_service)]
