"""Advanced document chunking using modern RAG techniques."""

import re
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

import tiktoken
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from advance_rag.core.config import get_settings
from advance_rag.core.logging import get_logger
from advance_rag.models import Chunk, Document, DocumentType

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class ChunkBoundary:
    """Represents a potential chunk boundary."""

    start: int
    end: int
    score: float
    text: str

    def __post_init__(self):
        self.semantic_score = self.score


class SemanticChunker:
    """Semantic chunking using sentence embeddings and similarity scoring."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize semantic chunker."""
        self.model = SentenceTransformer(model_name)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using multiple strategies."""
        # Try multiple sentence splitting patterns
        patterns = [
            r"(?<=[.!?])\s+(?=[A-Z])",  # Basic sentence split
            r"(?<=[.!?])\s+(?=[A-Z][a-z])",  # More conservative
            r"(?<=[.!?])\s+",  # Simple split on punctuation
        ]

        sentences = []
        for pattern in patterns:
            sentences = re.split(pattern, text.strip())
            if len(sentences) > 1:  # If we got multiple sentences, use this split
                break

        # Filter out empty sentences and strip whitespace
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def _compute_sentence_embeddings(self, sentences: List[str]) -> np.ndarray:
        """Compute embeddings for all sentences."""
        return self.model.encode(sentences, convert_to_numpy=True)

    def _find_chunk_boundaries(
        self, sentences: List[str], embeddings: np.ndarray
    ) -> List[ChunkBoundary]:
        """Find optimal chunk boundaries based on semantic similarity."""
        if len(sentences) <= 1:
            return [
                ChunkBoundary(
                    0,
                    len(sentences[0]) if sentences else 0,
                    1.0,
                    sentences[0] if sentences else "",
                )
            ]

        # Compute similarity between consecutive sentences
        similarities = cosine_similarity(embeddings[:-1], embeddings[1:])
        similarity_scores = np.diag(similarities)

        # Find semantic breaks (low similarity points)
        threshold = np.percentile(
            similarity_scores, 25
        )  # Bottom 25% as potential breaks
        breaks = []

        current_start = 0
        for i, score in enumerate(similarity_scores):
            if score < threshold:
                # Create boundary up to this point
                chunk_text = " ".join(sentences[current_start : i + 1])
                breaks.append(
                    ChunkBoundary(
                        start=current_start, end=i + 1, score=score, text=chunk_text
                    )
                )
                current_start = i + 1

        # Add final chunk
        if current_start < len(sentences):
            chunk_text = " ".join(sentences[current_start:])
            breaks.append(
                ChunkBoundary(
                    start=current_start, end=len(sentences), score=1.0, text=chunk_text
                )
            )

        return breaks

    def chunk(self, document: Document, max_tokens: int = 512) -> List[Chunk]:
        """Chunk document using semantic boundaries."""
        sentences = self._split_sentences(document.content)
        if not sentences:
            return []

        # Compute embeddings
        embeddings = self._compute_sentence_embeddings(sentences)

        # Find semantic boundaries
        boundaries = self._find_chunk_boundaries(sentences, embeddings)

        # Create chunks respecting token limits
        chunks = []
        chunk_index = 0

        for boundary in boundaries:
            # Check token count and split if needed
            text = boundary.text.strip()
            tokens = len(self.tokenizer.encode(text))

            if tokens <= max_tokens:
                # Create chunk
                chunk = self._create_chunk(text, document, chunk_index, boundary)
                chunks.append(chunk)
                chunk_index += 1
            else:
                # Split large chunk recursively
                sub_chunks = self._split_large_chunk(
                    text, document, chunk_index, max_tokens
                )
                chunks.extend(sub_chunks)
                chunk_index += len(sub_chunks)

        return chunks

    def _create_chunk(
        self, text: str, document: Document, chunk_index: int, boundary: ChunkBoundary
    ) -> Chunk:
        """Create a chunk object."""
        chunk_id = f"{document.id}_semantic_{chunk_index}"

        # Find character positions in original text
        sentences = self._split_sentences(document.content)
        char_start = 0
        char_end = 0

        for i, sentence in enumerate(sentences):
            if i == boundary.start:
                char_start = document.content.find(sentence, char_start)
            if i == boundary.end - 1:
                char_end = document.content.find(sentence, char_start) + len(sentence)
                break

        chunk_metadata = {
            "document_title": document.title,
            "document_type": document.document_type,
            "study_id": document.study_id,
            "token_count": len(self.tokenizer.encode(text)),
            "chunking_method": "semantic",
            "semantic_score": boundary.semantic_score,
            "sentence_count": boundary.end - boundary.start,
        }

        return Chunk(
            id=chunk_id,
            content=text,
            document_id=document.id,
            document_type=document.document_type,
            chunk_index=chunk_index,
            start_char=char_start,
            end_char=char_end,
            metadata=chunk_metadata,
        )

    def _split_large_chunk(
        self, text: str, document: Document, start_index: int, max_tokens: int
    ) -> List[Chunk]:
        """Split a large chunk into smaller ones."""
        tokens = self.tokenizer.encode(text)
        chunks = []

        for i in range(0, len(tokens), max_tokens):
            chunk_tokens = tokens[i : i + max_tokens]
            chunk_text = self.tokenizer.decode(chunk_tokens)

            chunk_id = f"{document.id}_semantic_split_{start_index}_{i // max_tokens}"
            chunk_metadata = {
                "document_title": document.title,
                "document_type": document.document_type,
                "study_id": document.study_id,
                "token_count": len(chunk_tokens),
                "chunking_method": "semantic_split",
                "split_index": i // max_tokens,
            }

            chunks.append(
                Chunk(
                    id=chunk_id,
                    content=chunk_text,
                    document_id=document.id,
                    document_type=document.document_type,
                    chunk_index=start_index + (i // max_tokens),
                    start_char=0,  # Would need proper calculation
                    end_char=len(chunk_text),
                    metadata=chunk_metadata,
                )
            )

        return chunks


class HierarchicalChunker:
    """Hierarchical chunking combining multiple strategies."""

    def __init__(self):
        """Initialize hierarchical chunker."""
        self.semantic_chunker = SemanticChunker()
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def chunk(self, document: Document) -> List[Chunk]:
        """Chunk using hierarchical approach."""
        # First try semantic chunking
        semantic_chunks = self.semantic_chunker.chunk(document)

        # If semantic chunking produces too few/too many chunks, fall back to fixed-size
        if len(semantic_chunks) < 2 or len(semantic_chunks) > 20:
            return self._fixed_size_chunk(document)

        return semantic_chunks

    def _fixed_size_chunk(self, document: Document) -> List[Chunk]:
        """Fixed-size chunking as fallback."""
        chunks = []
        content = document.content
        tokens = self.tokenizer.encode(content)

        chunk_size = settings.CHUNK_SIZE
        overlap = settings.CHUNK_OVERLAP

        for i in range(0, len(tokens), chunk_size - overlap):
            chunk_tokens = tokens[i : i + chunk_size]
            chunk_text = self.tokenizer.decode(chunk_tokens)

            chunk_id = f"{document.id}_fixed_{i // (chunk_size - overlap)}"

            # Calculate character positions
            start_char = len(self.tokenizer.decode(tokens[:i]))
            end_char = len(self.tokenizer.decode(tokens[: i + chunk_size]))

            chunk_metadata = {
                "document_title": document.title,
                "document_type": document.document_type,
                "study_id": document.study_id,
                "token_count": len(chunk_tokens),
                "chunking_method": "fixed_size",
            }

            chunks.append(
                Chunk(
                    id=chunk_id,
                    content=chunk_text,
                    document_id=document.id,
                    document_type=document.document_type,
                    chunk_index=i // (chunk_size - overlap),
                    start_char=start_char,
                    end_char=end_char,
                    metadata=chunk_metadata,
                )
            )

        return chunks


class AdaptiveChunker:
    """Adaptive chunking that selects best strategy based on document type."""

    def __init__(self):
        """Initialize adaptive chunker."""
        self.semantic_chunker = SemanticChunker()
        self.hierarchical_chunker = HierarchicalChunker()

    def chunk(self, document: Document) -> List[Chunk]:
        """Chunk using adaptive strategy."""
        # Select strategy based on document type and content
        if document.document_type in [
            DocumentType.CLINICAL_TRIAL,
            DocumentType.AE_REPORT,
        ]:
            # Use semantic chunking for clinical documents
            return self.semantic_chunker.chunk(document)
        elif document.document_type == DocumentType.STUDY_PROTOCOL:
            # Use hierarchical for protocols (has structure)
            return self.hierarchical_chunker.chunk(document)
        else:
            # Default to hierarchical
            return self.hierarchical_chunker.chunk(document)
