"""Weaviate vector store implementation."""

import os
import asyncio
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime
import json

import weaviate
from weaviate.client import Client
from weaviate.config import AdditionalConfig
from weaviate.classes.config import Configure, Property, DataType

from advance_rag.core.config import get_settings
from advance_rag.core.logging import get_logger
from .base_store import BaseVectorStore

logger = get_logger(__name__)
settings = get_settings()


class WeaviateVectorStore(BaseVectorStore):
    """Weaviate implementation of vector store."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize Weaviate vector store."""
        super().__init__(config)
        self.url = config.get("url") or os.getenv(
            "WEAVIATE_URL", "http://localhost:8080"
        )
        self.api_key = config.get("api_key") or os.getenv("WEAVIATE_API_KEY")
        self.class_name = config.get("class_name", "Chunk")
        self.batch_size = config.get("batch_size", 100)
        self.dimension = config.get("dimension", settings.EMBEDDING_DIMENSION)

        # Weaviate client
        self.client = None

        logger.info(f"Initialized WeaviateVectorStore with class: {self.class_name}")

    async def initialize(self):
        """Initialize Weaviate connection and schema."""
        try:
            # Initialize Weaviate client
            if self.api_key:
                self.client = Client(
                    self.url,
                    auth_client_secret=self.api_key,
                    additional_config=AdditionalConfig(
                        timeout=(10, 60)  # (connect, read) timeout in seconds
                    ),
                )
            else:
                self.client = Client(
                    self.url,
                    additional_config=AdditionalConfig(
                        timeout=(10, 60)  # (connect, read) timeout in seconds
                    ),
                )

            # Check if class exists, create if not
            if not self.client.schema.exists(self.class_name):
                logger.info(f"Creating Weaviate class: {self.class_name}")
                self._create_schema()
            else:
                logger.info(f"Using existing Weaviate class: {self.class_name}")

            # Wait for schema to be ready
            await self._wait_for_schema_ready()

            logger.info("Weaviate vector store initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Weaviate: {e}")
            raise

    def _create_schema(self):
        """Create Weaviate class schema."""
        schema = {
            "class": self.class_name,
            "description": "Document chunks for RAG system",
            "vectorizer": "none",  # We provide our own vectors
            "properties": [
                {"name": "content", "dataType": "text", "description": "Chunk content"},
                {
                    "name": "documentId",
                    "dataType": "string",
                    "description": "Source document ID",
                },
                {
                    "name": "documentType",
                    "dataType": "string",
                    "description": "Document type",
                },
                {
                    "name": "chunkIndex",
                    "dataType": "int",
                    "description": "Chunk index within document",
                },
                {
                    "name": "studyId",
                    "dataType": "string",
                    "description": "Study identifier",
                },
                {
                    "name": "metadata",
                    "dataType": "object",
                    "description": "Additional metadata",
                },
                {
                    "name": "createdAt",
                    "dataType": "date",
                    "description": "Creation timestamp",
                },
            ],
        }

        self.client.schema.create_class(schema)

    async def _wait_for_schema_ready(self):
        """Wait for schema to be ready."""
        max_wait = 60  # 1 minute
        start_time = asyncio.get_event_loop().time()

        while asyncio.get_event_loop().time() - start_time < max_wait:
            try:
                if self.client.schema.exists(self.class_name):
                    logger.info("Weaviate schema is ready")
                    return
            except Exception as e:
                logger.warning(f"Error checking schema: {e}")

            await asyncio.sleep(2)

        raise TimeoutError("Schema not ready within timeout period")

    async def store_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Store chunks in Weaviate."""
        if not chunks:
            return

        logger.info(f"Storing {len(chunks)} chunks in Weaviate")

        # Process in batches
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i : i + self.batch_size]
            await self._store_batch(batch)

        logger.info(f"Successfully stored {len(chunks)} chunks")

    async def _store_batch(self, batch: List[Dict[str, Any]]):
        """Store a batch of chunks."""
        data_objects = []

        for chunk in batch:
            # Prepare data object
            data_object = {
                "class": self.class_name,
                "id": chunk["id"],
                "vector": chunk["embedding"],
                "properties": {
                    "content": chunk.get("content", ""),
                    "documentId": chunk.get("document_id"),
                    "documentType": chunk.get("document_type"),
                    "chunkIndex": chunk.get("chunk_index"),
                    "studyId": chunk.get("study_id"),
                    "metadata": chunk.get("metadata", {}),
                    "createdAt": datetime.utcnow().isoformat(),
                },
            }
            data_objects.append(data_object)

        # Batch import to Weaviate
        try:
            self.client.batch.configure(batch_size=self.batch_size)
            with self.client.batch as batch:
                for data_object in data_objects:
                    batch.add_data_object(**data_object)

        except Exception as e:
            logger.error(f"Failed to store batch: {e}")
            raise

    async def vector_search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        study_id: Optional[str] = None,
        document_types: Optional[List[str]] = None,
        threshold: float = 0.7,
    ) -> List[Tuple[str, float]]:
        """Perform vector similarity search in Weaviate."""
        try:
            # Build near vector query
            near_vector = {
                "vector": query_embedding,
                "distance": 1 - threshold,  # Convert similarity to distance
                "certainty": threshold,
            }

            # Build where clause
            where_clause = self._build_where_clause(study_id, document_types)

            # Query Weaviate
            response = self.client.query.get(
                self.class_name,
                near_vector=near_vector,
                limit=top_k,
                where=where_clause,
                additional_properties=["_id", "_additional {id certainty distance}"],
            )

            # Process results
            results = []
            for obj in response.objects:
                certainty = obj._additional.get("certainty", 0)
                if certainty >= threshold:
                    results.append((obj.properties["_id"], certainty))

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
        """Perform full-text search in Weaviate."""
        try:
            # Build BM25 query
            where_clause = self._build_where_clause(study_id, document_types)

            # Use Weaviate's BM25 search
            response = self.client.query.get(
                self.class_name,
                query=query_text,
                limit=top_k,
                where=where_clause,
                additional_properties=["_id", "_additional {id bm25Score}"],
            )

            # Process results
            results = []
            for obj in response.objects:
                bm25_score = obj._additional.get("bm25Score", 0)
                # Normalize BM25 score to 0-1 range
                normalized_score = min(bm25_score / 10.0, 1.0)
                results.append((obj.properties["_id"], normalized_score))

            return results

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
        """Perform hybrid search in Weaviate."""
        try:
            # Weaviate supports hybrid search natively
            near_vector = {
                "vector": query_embedding,
                "distance": 0.3,  # Relaxed for hybrid
            }

            where_clause = self._build_where_clause(study_id, document_types)

            # Hybrid query combining vector and text
            response = self.client.query.get(
                self.class_name,
                query=query_text,
                near_vector=near_vector,
                limit=top_k,
                where=where_clause,
                additional_properties=[
                    "_id",
                    "_additional {id certainty distance bm25Score}",
                ],
            )

            # Combine scores
            results = []
            for obj in response.objects:
                additional = obj._additional
                vector_score = additional.get("certainty", 0)
                text_score = min(additional.get("bm25Score", 0) / 10.0, 1.0)

                # Weighted combination
                combined_score = (
                    dense_weight * vector_score + (1 - dense_weight) * text_score
                )
                results.append((obj.properties["_id"], combined_score))

            return sorted(results, key=lambda x: x[1], reverse=True)

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []

    def _build_where_clause(
        self, study_id: Optional[str], document_types: Optional[List[str]]
    ) -> Optional[Dict[str, Any]]:
        """Build Weaviate where clause."""
        conditions = []

        if study_id:
            conditions.append(
                {"path": ["studyId"], "operator": "Equal", "valueText": study_id}
            )

        if document_types:
            conditions.append(
                {
                    "path": ["documentType"],
                    "operator": "ContainsAny",
                    "valueStringList": document_types,
                }
            )

        if len(conditions) == 1:
            return conditions[0]
        elif len(conditions) > 1:
            return {"operator": "And", "operands": conditions}

        return None

    async def get_chunks_by_ids(self, chunk_ids: List[str]) -> List[Dict[str, Any]]:
        """Get chunks by IDs from Weaviate."""
        try:
            # Build where clause for IDs
            where_clause = {
                "path": ["_id"],
                "operator": "ContainsAny",
                "valueStringList": chunk_ids,
            }

            response = self.client.query.get(
                self.class_name,
                where=where_clause,
                additional_properties=[
                    "_id",
                    "content",
                    "documentId",
                    "documentType",
                    "chunkIndex",
                    "studyId",
                    "metadata",
                    "createdAt",
                ],
            )

            # Convert to expected format
            chunks = []
            for obj in response.objects:
                chunk = {
                    "id": obj.properties["_id"],
                    "content": obj.properties.get("content", ""),
                    "document_id": obj.properties.get("documentId"),
                    "document_type": obj.properties.get("documentType"),
                    "chunk_index": obj.properties.get("chunkIndex"),
                    "study_id": obj.properties.get("studyId"),
                    "metadata": obj.properties.get("metadata", {}),
                    "created_at": obj.properties.get("createdAt"),
                    "embedding": [],  # Not returned by default query
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
        """Get Weaviate statistics."""
        try:
            # Get schema info
            schema = self.client.schema.get(self.class_name)

            # Get object count
            response = self.client.query.aggregate(self.class_name)
            total_count = (
                response.total_count if hasattr(response, "total_count") else 0
            )

            return {
                "total_chunks": total_count,
                "class_name": self.class_name,
                "dimension": self.dimension,
                "vectorizer": "none",  # We provide our own vectors
                "properties": len(schema.get("properties", [])),
                "status": "ready",
            }

        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}

    async def close(self):
        """Close Weaviate connection."""
        try:
            if self.client:
                self.client.close()
            logger.info("Weaviate connection closed")
        except Exception as e:
            logger.error(f"Error closing Weaviate: {e}")
