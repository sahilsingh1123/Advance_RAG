"""Data ingestion service for processing documents."""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from advance_rag.core.config import get_settings
from advance_rag.core.logging import get_logger, log_ingestion
from advance_rag.ingestion.chunkers import get_chunker
from advance_rag.models import Document, DocumentType

logger = get_logger(__name__)
settings = get_settings()


class DocumentProcessor:
    """Process documents for ingestion."""

    def __init__(self):
        """Initialize document processor."""
        self.supported_extensions = {
            ".json": DocumentType.OTHER,
            ".md": DocumentType.OTHER,
            ".txt": DocumentType.OTHER,
            ".csv": DocumentType.OTHER,
        }

    def detect_document_type(self, file_path: Path) -> DocumentType:
        """Detect document type based on file content and name."""
        file_path_lower = file_path.name.lower()

        # Check filename patterns
        if "sdtm" in file_path_lower:
            return DocumentType.SDTM
        elif "adam" in file_path_lower:
            return DocumentType.ADAM
        elif "sap" in file_path_lower:
            return DocumentType.STUDY_ANALYSIS_PLAN
        elif "protocol" in file_path_lower:
            return DocumentType.PROTOCOL

        # Check file extension
        ext = file_path.suffix.lower()
        if ext in self.supported_extensions:
            return self.supported_extensions[ext]

        return DocumentType.OTHER

    def extract_metadata(self, file_path: Path, content: str) -> Dict[str, Any]:
        """Extract metadata from document."""
        metadata = {
            "file_name": file_path.name,
            "file_extension": file_path.suffix,
            "file_size": len(content.encode("utf-8")),
        }

        # Try to extract study ID from filename or content
        study_id = self._extract_study_id(file_path.name, content)
        if study_id:
            metadata["study_id"] = study_id

        # Extract domain/dataset names for SDTM/ADaM
        if metadata.get("document_type") in [DocumentType.SDTM, DocumentType.ADAM]:
            domains = self._extract_domains(content)
            if domains:
                metadata["domains"] = domains

        return metadata

    def _extract_study_id(self, filename: str, content: str) -> Optional[str]:
        """Extract study ID from filename or content."""
        # Common study ID patterns
        patterns = [
            r"STUDY(\d{4,6})",
            r"STUDYID[_-]?(\w+)",
            r"STUDY[_-]?(\w+)",
            r"PROTOCOL[_-]?(\w+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                return match.group(1)

            match = re.search(pattern, content[:1000], re.IGNORECASE)
            if match:
                return match.group(1)

        return None

    def _extract_domains(self, content: str) -> List[str]:
        """Extract SDTM/ADaM domain names from content."""
        domains = []

        # Try to parse as JSON first
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                domains.extend(data.keys())
            elif isinstance(data, list) and data:
                domains.extend(data[0].keys() if isinstance(data[0], dict) else [])
        except json.JSONDecodeError:
            # For non-JSON, use regex patterns
            sdtm_domains = [
                "DM",
                "AE",
                "CM",
                "DA",
                "DS",
                "DV",
                "EX",
                "FA",
                "IE",
                "IN",
                "MH",
                "MO",
                "PE",
                "PR",
                "QS",
                "SC",
                "SU",
                "TA",
                "TE",
                "TI",
                "TR",
                "TV",
                "VS",
            ]

            adam_datasets = [
                "ADSL",
                "ADAE",
                "ADCM",
                "ADDA",
                "ADDS",
                "ADLB",
                "ADMB",
                "ADMH",
                "ADPC",
                "ADPE",
                "ADRG",
                "ADRS",
                "ADTE",
                "ADTI",
                "ADTR",
                "ADVS",
                "ADYD",
            ]

            all_domains = sdtm_domains + adam_datasets

            for domain in all_domains:
                if domain in content:
                    domains.append(domain)

        return list(set(domains))

    def process_file(self, file_path: Path) -> Document:
        """Process a single file into a document."""
        logger.info(f"Processing file: {file_path}")

        # Validate file
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        file_size = file_path.stat().st_size
        if file_size > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
            raise ValueError(f"File too large: {file_size} bytes")

        # Read content
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Detect document type
        document_type = self.detect_document_type(file_path)

        # Extract metadata
        metadata = self.extract_metadata(file_path, content)
        metadata["document_type"] = document_type

        # Generate document ID
        doc_id = self._generate_document_id(file_path, content)

        # Create document
        document = Document(
            id=doc_id,
            title=file_path.stem,
            content=content,
            document_type=document_type,
            file_path=str(file_path),
            file_size=file_size,
            study_id=metadata.get("study_id"),
            metadata=metadata,
        )

        # Chunk document
        chunker = get_chunker(document_type)
        chunks = chunker.chunk(document)
        document.chunks = chunks

        logger.info(f"Processed document {doc_id}: {len(chunks)} chunks")
        return document

    def _generate_document_id(self, file_path: Path, content: str) -> str:
        """Generate unique document ID."""
        # Use file path and content hash for uniqueness
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
        path_hash = hashlib.sha256(str(file_path).encode("utf-8")).hexdigest()[:16]
        return f"doc_{path_hash}_{content_hash}"


class IngestionService:
    """Service for ingesting documents into the RAG system."""

    def __init__(self, storage_service, embedding_service, graph_service):
        """Initialize ingestion service."""
        self.processor = DocumentProcessor()
        self.storage = storage_service
        self.embedding = embedding_service
        self.graph = graph_service

    async def ingest_files(self, file_paths: List[str]) -> List[str]:
        """Ingest multiple files."""
        document_ids = []

        for file_path in file_paths:
            try:
                doc_id = await self.ingest_file(file_path)
                document_ids.append(doc_id)
            except Exception as e:
                logger.error(f"Failed to ingest file {file_path}: {e}")
                raise

        return document_ids

    async def ingest_file(self, file_path: str) -> str:
        """Ingest a single file."""
        import time

        start_time = time.time()

        # Process file
        path = Path(file_path)
        document = self.processor.process_file(path)

        # Store document
        await self.storage.store_document(document)

        # Generate and store embeddings
        if document.chunks:
            embeddings = await self.embedding.generate_embeddings(
                [chunk.content for chunk in document.chunks]
            )

            # Update chunks with embeddings
            for chunk, embedding in zip(document.chunks, embeddings):
                chunk.embedding = embedding

            # Store embeddings
            await self.storage.store_embeddings(document.chunks)

        # Extract and store graph entities
        await self.graph.extract_and_store_entities(document)

        # Log ingestion
        duration_ms = (time.time() - start_time) * 1000
        log_ingestion(
            file_path=file_path,
            file_size=document.file_size,
            num_chunks=len(document.chunks),
            duration_ms=duration_ms,
        )

        return document.id

    async def ingest_directory(
        self, directory_path: str, recursive: bool = True
    ) -> List[str]:
        """Ingest all files in a directory."""
        path = Path(directory_path)
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        # Find all supported files
        pattern = "**/*" if recursive else "*"
        files = []

        for ext in settings.SUPPORTED_FILE_TYPES:
            files.extend(path.glob(f"{pattern}.{ext}"))

        # Convert to strings and sort
        file_paths = sorted([str(f) for f in files])

        # Ingest files
        return await self.ingest_files(file_paths)
