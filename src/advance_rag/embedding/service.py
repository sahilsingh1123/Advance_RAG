"""Embedding service with caching for the RAG system."""

import hashlib
import json
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
import redis
from sentence_transformers import SentenceTransformer

from advance_rag.core.config import get_settings
from advance_rag.core.logging import get_logger, log_llm_call

logger = get_logger(__name__)
settings = get_settings()


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    async def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode texts into embeddings."""
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        pass


class SentenceTransformerProvider(EmbeddingProvider):
    """Sentence Transformers embedding provider."""

    def __init__(self, model_name: str = settings.EMBEDDING_MODEL):
        """Initialize Sentence Transformer provider."""
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        logger.info(f"Loaded Sentence Transformer model: {model_name}")

    async def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode texts using Sentence Transformer."""
        embeddings = self.model.encode(
            texts,
            batch_size=settings.EMBEDDING_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.model.get_sentence_embedding_dimension()


class OpenAIProvider(EmbeddingProvider):
    """OpenAI embedding provider."""

    def __init__(self, api_key: str, model: str = "text-embedding-3-large"):
        """Initialize OpenAI provider."""
        import openai

        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model = model
        logger.info(f"Initialized OpenAI embedding model: {model}")

    async def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode texts using OpenAI API."""
        import time

        start_time = time.time()

        response = await self.client.embeddings.create(
            model=self.model,
            input=texts,
        )

        embeddings = [data.embedding for data in response.data]

        # Log API call
        duration_ms = (time.time() - start_time) * 1000
        log_llm_call(
            model=self.model,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=0,
            duration_ms=duration_ms,
        )

        return embeddings

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        dimensions = {
            "text-embedding-3-large": 3072,
            "text-embedding-3-small": 1536,
            "text-embedding-ada-002": 1536,
        }
        return dimensions.get(self.model, 1536)


class VoyageAIProvider(EmbeddingProvider):
    """Voyage AI embedding provider."""

    def __init__(self, api_key: str, model: str = "voyage-large-2"):
        """Initialize Voyage AI provider."""
        import voyageai

        self.client = voyageai.AsyncClient(api_key=api_key)
        self.model = model
        logger.info(f"Initialized Voyage AI model: {model}")

    async def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode texts using Voyage AI API."""
        import time

        start_time = time.time()

        result = await self.client.embed(
            texts,
            model=self.model,
            input_type="document",
        )

        embeddings = result.embeddings

        # Log API call
        duration_ms = (time.time() - start_time) * 1000
        log_llm_call(
            model=self.model,
            prompt_tokens=result.total_tokens,
            completion_tokens=0,
            duration_ms=duration_ms,
        )

        return embeddings

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        dimensions = {
            "voyage-large-2": 1536,
            "voyage-code-2": 1536,
            "voyage-2": 1024,
            "voyage-lite-02": 512,
        }
        return dimensions.get(self.model, 1024)


class EmbeddingCache:
    """Redis-based embedding cache."""

    def __init__(self, redis_url: str = settings.REDIS_URL):
        """Initialize embedding cache."""
        self.redis_client = redis.from_url(redis_url, decode_responses=False)
        self.ttl = settings.EMBEDDING_CACHE_TTL
        logger.info("Initialized embedding cache with Redis")

    def _get_cache_key(self, text: str, model: str) -> str:
        """Generate cache key for text."""
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"embedding:{model}:{text_hash}"

    async def get(self, text: str, model: str) -> Optional[List[float]]:
        """Get cached embedding."""
        cache_key = self._get_cache_key(text, model)

        try:
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            logger.error(f"Failed to get cached embedding: {e}")

        return None

    async def set(self, text: str, model: str, embedding: List[float]) -> None:
        """Cache embedding."""
        cache_key = self._get_cache_key(text, model)

        try:
            self.redis_client.setex(cache_key, self.ttl, json.dumps(embedding))
        except Exception as e:
            logger.error(f"Failed to cache embedding: {e}")

    async def get_batch(
        self, texts: List[str], model: str
    ) -> List[Optional[List[float]]]:
        """Get multiple cached embeddings."""
        cache_keys = [self._get_cache_key(text, model) for text in texts]

        try:
            cached_data = self.redis_client.mget(cache_keys)
            return [json.loads(data) if data else None for data in cached_data]
        except Exception as e:
            logger.error(f"Failed to get batch cached embeddings: {e}")
            return [None] * len(texts)

    async def set_batch(
        self, texts: List[str], model: str, embeddings: List[List[float]]
    ) -> None:
        """Cache multiple embeddings."""
        pipe = self.redis_client.pipeline()

        for text, embedding in zip(texts, embeddings):
            cache_key = self._get_cache_key(text, model)
            pipe.setex(cache_key, self.ttl, json.dumps(embedding))

        try:
            pipe.execute()
        except Exception as e:
            logger.error(f"Failed to cache batch embeddings: {e}")


class EmbeddingService:
    """Service for generating and caching embeddings."""

    def __init__(self):
        """Initialize embedding service."""
        self.cache = EmbeddingCache()
        self.provider = self._get_provider()
        logger.info(
            f"Initialized embedding service with {type(self.provider).__name__}"
        )

    def _get_provider(self) -> EmbeddingProvider:
        """Get embedding provider based on settings."""
        if settings.OPENAI_API_KEY and settings.LLM_PROVIDER == "openai":
            return OpenAIProvider(
                api_key=settings.OPENAI_API_KEY, model="text-embedding-3-large"
            )
        elif settings.EMBEDDING_MODEL.startswith("voyage-"):
            # Voyage AI would need API key - fallback to sentence transformer
            logger.warning("Voyage AI API key not provided, using Sentence Transformer")
            return SentenceTransformerProvider()
        else:
            return SentenceTransformerProvider(settings.EMBEDDING_MODEL)

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts with caching."""
        if not texts:
            return []

        # Check cache first
        cached_embeddings = await self.cache.get_batch(texts, self.provider.model_name)

        # Find texts that need to be encoded
        texts_to_encode = []
        indices_to_encode = []

        for i, (text, cached) in enumerate(zip(texts, cached_embeddings)):
            if cached is None:
                texts_to_encode.append(text)
                indices_to_encode.append(i)
            else:
                cached_embeddings[i] = cached

        # Generate embeddings for uncached texts
        if texts_to_encode:
            new_embeddings = await self.provider.encode(texts_to_encode)

            # Update cache
            await self.cache.set_batch(
                texts_to_encode, self.provider.model_name, new_embeddings
            )

            # Update results
            for idx, embedding in zip(indices_to_encode, new_embeddings):
                cached_embeddings[idx] = embedding

        return cached_embeddings

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        embeddings = await self.generate_embeddings([text])
        return embeddings[0] if embeddings else []

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.provider.get_dimension()

    async def compute_similarity(
        self, embedding1: List[float], embedding2: List[float]
    ) -> float:
        """Compute cosine similarity between two embeddings."""
        # Convert to numpy arrays
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        # Compute cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    async def find_most_similar(
        self,
        query_embedding: List[float],
        candidate_embeddings: List[List[float]],
        top_k: int = 10,
        threshold: float = 0.0,
    ) -> List[tuple[int, float]]:
        """Find most similar embeddings to query."""
        if not candidate_embeddings:
            return []

        similarities = []
        query_vec = np.array(query_embedding)

        for i, candidate in enumerate(candidate_embeddings):
            candidate_vec = np.array(candidate)

            # Compute cosine similarity
            dot_product = np.dot(query_vec, candidate_vec)
            norm_query = np.linalg.norm(query_vec)
            norm_candidate = np.linalg.norm(candidate_vec)

            if norm_query > 0 and norm_candidate > 0:
                similarity = dot_product / (norm_query * norm_candidate)
            else:
                similarity = 0.0

            if similarity >= threshold:
                similarities.append((i, similarity))

        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
