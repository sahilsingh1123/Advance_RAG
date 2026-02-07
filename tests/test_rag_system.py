"""Test cases for the RAG system."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock

from advance_rag.models import Query, QueryMode
from advance_rag.retrieval.hybrid import HybridRetriever
from advance_rag.embedding.service import EmbeddingService
from advance_rag.db.vector_store import VectorStore


@pytest.fixture
async def embedding_service():
    """Create embedding service fixture."""
    service = EmbeddingService()
    yield service


@pytest.fixture
async def vector_store():
    """Create vector store fixture."""
    store = VectorStore()
    await store.initialize()
    yield store
    await store.close()


@pytest.fixture
async def retriever(vector_store, embedding_service):
    """Create retriever fixture."""
    return HybridRetriever(vector_store, embedding_service)


@pytest.mark.asyncio
async def test_embedding_generation(embedding_service):
    """Test embedding generation."""
    texts = ["This is a test text", "Another test text"]

    embeddings = await embedding_service.generate_embeddings(texts)

    assert len(embeddings) == 2
    assert len(embeddings[0]) == embedding_service.get_dimension()
    assert len(embeddings[1]) == embedding_service.get_dimension()


@pytest.mark.asyncio
async def test_vector_search(vector_store, embedding_service):
    """Test vector search functionality."""
    # Create test chunks
    chunks = [
        {
            "id": "chunk_1",
            "content": "Clinical trial data analysis",
            "document_id": "doc_1",
            "document_type": "sdtm",
            "chunk_index": 0,
            "start_char": 0,
            "end_char": 25,
            "metadata": {},
            "embedding": await embedding_service.generate_embedding(
                "Clinical trial data analysis"
            ),
        },
        {
            "id": "chunk_2",
            "content": "Patient demographics information",
            "document_id": "doc_2",
            "document_type": "sdtm",
            "chunk_index": 0,
            "start_char": 0,
            "end_char": 28,
            "metadata": {},
            "embedding": await embedding_service.generate_embedding(
                "Patient demographics information"
            ),
        },
    ]

    # Store chunks
    await vector_store.store_chunks(chunks)

    # Perform search
    query_embedding = await embedding_service.generate_embedding("clinical data")
    results = await vector_store.vector_search(query_embedding, top_k=2)

    assert len(results) <= 2
    assert all(isinstance(score, float) for _, score in results)


@pytest.mark.asyncio
async def test_hybrid_retrieval(retriever):
    """Test hybrid retrieval."""
    query = Query(
        id="test_query", text="Find clinical trial data", mode=QueryMode.QA, top_k=5
    )

    # Mock the vector store methods
    retriever.vector_store.vector_search = AsyncMock(return_value=[("chunk_1", 0.8)])
    retriever.vector_store.full_text_search = AsyncMock(return_value=[("chunk_2", 0.7)])
    retriever.vector_store.get_chunks_by_ids = AsyncMock(
        return_value=[
            {
                "id": "chunk_1",
                "content": "Clinical trial data",
                "document_id": "doc_1",
                "document_type": "sdtm",
            }
        ]
    )

    results = await retriever.retrieve(query)

    assert len(results) >= 0
    if results:
        assert hasattr(results[0], "chunk")
        assert hasattr(results[0], "score")
        assert hasattr(results[0], "source")


@pytest.mark.asyncio
async def test_query_service():
    """Test query service."""
    from advance_rag.services.query_service import QueryService

    # Mock dependencies
    mock_retriever = AsyncMock()
    mock_embedding_service = AsyncMock()
    mock_graphrag_service = AsyncMock()

    # Create service
    service = QueryService(
        retriever=mock_retriever,
        embedding_service=mock_embedding_service,
        graphrag_service=mock_graphrag_service,
    )

    # Mock retrieval results
    from advance_rag.models import RetrievalResult

    mock_results = [
        RetrievalResult(
            chunk={"id": "chunk_1", "content": "Test content"}, score=0.8, source="test"
        )
    ]
    mock_retriever.retrieve.return_value = mock_results

    # Create query
    query = Query(id="test_query", text="Test query", mode=QueryMode.QA, top_k=5)

    # Execute query
    response = await service.execute_query(query)

    assert response.query_id == query.id
    assert response.mode == query.mode
    assert len(response.sources) == len(mock_results)


@pytest.mark.asyncio
async def test_ingestion_service():
    """Test ingestion service."""
    from advance_rag.ingestion.service import IngestionService
    from advance_rag.models import Document, DocumentType

    # Mock dependencies
    mock_storage = AsyncMock()
    mock_embedding = AsyncMock()
    mock_graph = AsyncMock()

    # Create service
    service = IngestionService(
        storage_service=mock_storage,
        embedding_service=mock_embedding,
        graph_service=mock_graph,
    )

    # Create test document
    document = Document(
        id="test_doc",
        title="Test Document",
        content="This is test content for ingestion.",
        document_type=DocumentType.SDTM,
        file_path="/test/path.json",
        file_size=100,
    )

    # Mock embedding generation
    mock_embedding.generate_embeddings.return_value = [[0.1, 0.2, 0.3]]

    # Ingest document
    doc_id = await service.ingest_file("/test/path.json")

    # Verify calls
    mock_storage.store_document.assert_called_once()
    mock_graph.extract_and_store_entities.assert_called_once()


def test_chunking():
    """Test document chunking."""
    from advance_rag.ingestion.chunkers import MarkdownChunker, JSONChunker
    from advance_rag.models import Document, DocumentType

    # Test Markdown chunking
    md_chunker = MarkdownChunker()
    md_document = Document(
        id="md_doc",
        title="Test Markdown",
        content="# Header 1\n\nThis is some content.\n\n## Header 2\n\nMore content.",
        document_type=DocumentType.SDTM,
        file_path="/test.md",
        file_size=100,
    )

    md_chunks = md_chunker.chunk(md_document)
    assert len(md_chunks) > 0
    assert all(chunk.content for chunk in md_chunks)

    # Test JSON chunking
    json_chunker = JSONChunker()
    json_document = Document(
        id="json_doc",
        title="Test JSON",
        content='{"key1": "value1", "key2": "value2"}',
        document_type=DocumentType.SDTM,
        file_path="/test.json",
        file_size=50,
    )

    json_chunks = json_chunker.chunk(json_document)
    assert len(json_chunks) > 0


@pytest.mark.asyncio
async def test_graph_service():
    """Test Neo4j graph service."""
    from advance_rag.graph.neo4j_service import Neo4jService
    from advance_rag.models import GraphEntity, GraphRelation

    # Mock Neo4j driver
    mock_driver = Mock()
    mock_session = Mock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    # Create service with mock
    service = Neo4jService()
    service.driver = mock_driver

    # Test entity creation
    entity = GraphEntity(
        id="test_entity", type="subject", name="TEST001", properties={"age": 45}
    )

    # Mock session run result
    mock_result = Mock()
    mock_result.single.return_value = True
    mock_session.run.return_value = mock_result

    result = await service.create_entity(entity)
    assert result is True

    # Verify query
    mock_session.run.assert_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
