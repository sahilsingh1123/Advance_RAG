"""Document chunking strategies for the RAG system."""

import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import markdown
import tiktoken
from bs4 import BeautifulSoup

from advance_rag.core.config import get_settings
from advance_rag.core.logging import get_logger
from advance_rag.models import Chunk, Document, DocumentType

logger = get_logger(__name__)
settings = get_settings()


class BaseChunker(ABC):
    """Base class for document chunkers."""

    def __init__(
        self,
        chunk_size: int = settings.CHUNK_SIZE,
        chunk_overlap: int = settings.CHUNK_OVERLAP,
        encoding_name: str = "cl100k_base",
    ):
        """Initialize chunker."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tiktoken.get_encoding(encoding_name)

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer.encode(text))

    @abstractmethod
    def chunk(self, document: Document) -> List[Chunk]:
        """Chunk document into smaller pieces."""
        pass

    def _create_chunk(
        self,
        content: str,
        document: Document,
        chunk_index: int,
        start_char: int,
        end_char: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Chunk:
        """Create a chunk object."""
        chunk_id = f"{document.id}_chunk_{chunk_index}"
        chunk_metadata = {
            "document_title": document.title,
            "document_type": document.document_type,
            "study_id": document.study_id,
            "token_count": self.count_tokens(content),
            **(metadata or {}),
        }

        return Chunk(
            id=chunk_id,
            content=content,
            document_id=document.id,
            document_type=document.document_type,
            chunk_index=chunk_index,
            start_char=start_char,
            end_char=end_char,
            metadata=chunk_metadata,
        )


class MarkdownChunker(BaseChunker):
    """Chunker for Markdown documents."""

    def chunk(self, document: Document) -> List[Chunk]:
        """Chunk Markdown document by headers."""
        chunks = []

        # Parse Markdown to HTML
        html = markdown.markdown(document.content, extensions=["toc", "tables"])
        soup = BeautifulSoup(html, "html.parser")

        # Find all headers
        headers = soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])

        if not headers:
            # No headers found, use fixed-size chunking
            return self._fixed_size_chunk(document)

        # Chunk by sections
        current_pos = 0
        chunk_index = 0

        for i, header in enumerate(headers):
            # Find the next header or end of document
            next_elements = headers[i + 1 :] if i + 1 < len(headers) else []
            end_pos = len(document.content)

            if next_elements:
                # Find position of next header in original text
                header_text = header.get_text().strip()
                next_header_text = next_elements[0].get_text().strip()

                # Simple heuristic to find header positions
                header_pattern = re.escape(header_text)
                next_header_pattern = re.escape(next_header_text)

                header_match = re.search(header_pattern, document.content[current_pos:])
                if header_match:
                    start_pos = current_pos + header_match.start()

                    next_match = re.search(
                        next_header_pattern, document.content[start_pos:]
                    )
                    if next_match:
                        end_pos = start_pos + next_match.start()
                else:
                    start_pos = current_pos
            else:
                start_pos = current_pos

            # Extract section content
            section_content = document.content[start_pos:end_pos].strip()

            if section_content:
                # Split section if too long
                section_chunks = self._split_long_text(
                    section_content,
                    document,
                    chunk_index,
                    start_pos,
                    {"section_title": header.get_text().strip()},
                )
                chunks.extend(section_chunks)
                chunk_index += len(section_chunks)

            current_pos = end_pos

        return chunks

    def _fixed_size_chunk(self, document: Document) -> List[Chunk]:
        """Fallback to fixed-size chunking."""
        return self._split_long_text(document.content, document, 0, 0)

    def _split_long_text(
        self,
        text: str,
        document: Document,
        start_chunk_index: int,
        start_char: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """Split long text into chunks."""
        chunks = []
        tokens = self.tokenizer.encode(text)

        i = 0
        chunk_index = start_chunk_index

        while i < len(tokens):
            # Calculate chunk boundaries
            end_idx = min(i + self.chunk_size, len(tokens))
            chunk_tokens = tokens[i:end_idx]
            chunk_text = self.tokenizer.decode(chunk_tokens)

            # Find actual character positions
            chunk_start_char = start_char + len(self.tokenizer.decode(tokens[:i]))
            chunk_end_char = start_char + len(self.tokenizer.decode(tokens[:end_idx]))

            # Create chunk
            chunk = self._create_chunk(
                chunk_text,
                document,
                chunk_index,
                chunk_start_char,
                chunk_end_char,
                metadata,
            )
            chunks.append(chunk)

            # Move to next chunk with overlap
            i = end_idx - self.chunk_overlap if end_idx < len(tokens) else end_idx
            chunk_index += 1

        return chunks


class JSONChunker(BaseChunker):
    """Chunker for JSON documents."""

    def chunk(self, document: Document) -> List[Chunk]:
        """Chunk JSON document by records or objects."""
        try:
            data = json.loads(document.content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON document {document.id}: {e}")
            return self._fixed_size_chunk(document)

        chunks = []

        if isinstance(data, list):
            # Array of objects - chunk each item
            for i, item in enumerate(data):
                chunk_content = json.dumps(item, indent=2)
                chunk = self._create_chunk(
                    chunk_content,
                    document,
                    i,
                    0,  # Will be calculated based on actual position
                    0,
                    {"array_index": i, "item_type": type(item).__name__},
                )
                chunks.append(chunk)

        elif isinstance(data, dict):
            # Object - chunk by significant keys
            chunks = self._chunk_dict(data, document, "")

        else:
            # Primitive value - treat as single chunk
            chunk_content = json.dumps(data, indent=2)
            chunk = self._create_chunk(
                chunk_content,
                document,
                0,
                0,
                len(document.content),
                {"value_type": type(data).__name__},
            )
            chunks.append(chunk)

        return chunks

    def _chunk_dict(
        self,
        data: Dict[str, Any],
        document: Document,
        prefix: str,
        start_index: int = 0,
    ) -> List[Chunk]:
        """Recursively chunk dictionary by keys."""
        chunks = []
        chunk_index = start_index

        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, (dict, list)):
                # Nested structure - recurse
                if isinstance(value, dict):
                    sub_chunks = self._chunk_dict(
                        value, document, full_key, chunk_index
                    )
                    chunks.extend(sub_chunks)
                    chunk_index += len(sub_chunks)
                else:
                    # Array - chunk each item
                    for i, item in enumerate(value):
                        chunk_content = json.dumps(item, indent=2)
                        chunk = self._create_chunk(
                            chunk_content,
                            document,
                            chunk_index,
                            0,  # Will be calculated
                            0,
                            {
                                "key": full_key,
                                "array_index": i,
                                "item_type": type(item).__name__,
                            },
                        )
                        chunks.append(chunk)
                        chunk_index += 1
            else:
                # Primitive value - create chunk
                chunk_content = json.dumps({key: value}, indent=2)
                chunk = self._create_chunk(
                    chunk_content,
                    document,
                    chunk_index,
                    0,  # Will be calculated
                    0,
                    {"key": full_key, "value_type": type(value).__name__},
                )
                chunks.append(chunk)
                chunk_index += 1

        return chunks

    def _fixed_size_chunk(self, document: Document) -> List[Chunk]:
        """Fallback to fixed-size chunking."""
        return self._split_long_text(document.content, document, 0, 0)


class TextChunker(BaseChunker):
    """Chunker for plain text documents."""

    def chunk(self, document: Document) -> List[Chunk]:
        """Chunk text document by paragraphs or fixed size."""
        # Try to split by paragraphs first
        paragraphs = re.split(r"\n\s*\n", document.content)

        # Filter out empty paragraphs
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        if not paragraphs:
            return self._fixed_size_chunk(document)

        chunks = []
        current_chunk = ""
        current_tokens = 0
        chunk_index = 0
        start_char = 0

        for paragraph in paragraphs:
            paragraph_tokens = self.count_tokens(paragraph)

            # If paragraph is too long, split it
            if paragraph_tokens > self.chunk_size:
                # Save current chunk if not empty
                if current_chunk:
                    chunks.append(
                        self._create_chunk(
                            current_chunk,
                            document,
                            chunk_index,
                            start_char,
                            start_char + len(current_chunk),
                        )
                    )
                    chunk_index += 1
                    start_char += len(current_chunk)
                    current_chunk = ""
                    current_tokens = 0

                # Split long paragraph
                para_chunks = self._split_long_text(
                    paragraph,
                    document,
                    chunk_index,
                    start_char,
                )
                chunks.extend(para_chunks)
                chunk_index += len(para_chunks)
                start_char += len(paragraph)

            # If adding paragraph exceeds chunk size
            elif current_tokens + paragraph_tokens > self.chunk_size:
                # Save current chunk
                if current_chunk:
                    chunks.append(
                        self._create_chunk(
                            current_chunk,
                            document,
                            chunk_index,
                            start_char,
                            start_char + len(current_chunk),
                        )
                    )
                    chunk_index += 1
                    start_char += len(current_chunk)

                # Start new chunk
                current_chunk = paragraph
                current_tokens = paragraph_tokens

            # Add to current chunk
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
                current_tokens += paragraph_tokens

        # Save final chunk
        if current_chunk:
            chunks.append(
                self._create_chunk(
                    current_chunk,
                    document,
                    chunk_index,
                    start_char,
                    start_char + len(current_chunk),
                )
            )

        return chunks

    def _fixed_size_chunk(self, document: Document) -> List[Chunk]:
        """Fallback to fixed-size chunking."""
        return self._split_long_text(document.content, document, 0, 0)


class CSVChunker(BaseChunker):
    """Chunker for CSV documents."""

    def chunk(self, document: Document) -> List[Chunk]:
        """Chunk CSV document by rows."""
        import pandas as pd

        try:
            df = pd.read_csv(Path(document.file_path))
        except Exception as e:
            logger.error(f"Failed to parse CSV document {document.id}: {e}")
            return self._fixed_size_chunk(document)

        chunks = []
        rows_per_chunk = max(1, self.chunk_size // 100)  # Estimate 100 tokens per row

        for i in range(0, len(df), rows_per_chunk):
            chunk_df = df.iloc[i : i + rows_per_chunk]
            chunk_content = chunk_df.to_csv(index=False)

            chunk = self._create_chunk(
                chunk_content,
                document,
                i // rows_per_chunk,
                0,  # Will be calculated
                0,
                {
                    "row_range": f"{i}-{min(i + rows_per_chunk, len(df))}",
                    "num_rows": len(chunk_df),
                },
            )
            chunks.append(chunk)

        return chunks

    def _fixed_size_chunk(self, document: Document) -> List[Chunk]:
        """Fallback to fixed-size chunking."""
        return self._split_long_text(document.content, document, 0, 0)


def get_chunker(document_type: DocumentType) -> BaseChunker:
    """Get appropriate chunker for document type."""
    chunkers = {
        DocumentType.MD: MarkdownChunker,
        DocumentType.SDTM: JSONChunker,
        DocumentType.ADAM: JSONChunker,
        DocumentType.SAP: MarkdownChunker,
        DocumentType.PROTOCOL: MarkdownChunker,
        DocumentType.OTHER: TextChunker,
    }

    chunker_class = chunkers.get(document_type, TextChunker)
    return chunker_class()
