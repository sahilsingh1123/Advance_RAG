"""Query service for the RAG system."""

import time
from typing import Any, Dict, List, Optional

from advance_rag.core.config import get_settings
from advance_rag.core.logging import get_logger, log_llm_call
from advance_rag.models import Query, QueryMode, QueryResponse
from advance_rag.retrieval.hybrid import AdvancedHybridRetriever
from advance_rag.embedding.service import EmbeddingService
from advance_rag.graph.graphrag import GraphRAGService

logger = get_logger(__name__)
settings = get_settings()


class LLMService:
    """Service for interacting with LLMs."""

    def __init__(self):
        """Initialize LLM service."""
        self.provider = settings.LLM_PROVIDER
        self.model = settings.LLM_MODEL
        self.max_tokens = settings.LLM_MAX_TOKENS
        self.temperature = settings.LLM_TEMPERATURE

        if self.provider == "anthropic" and settings.ANTHROPIC_API_KEY:
            import anthropic

            self.client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        elif self.provider == "openai" and settings.OPENAI_API_KEY:
            import openai

            self.client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        else:
            logger.warning("No LLM API key provided, using mock responses")
            self.client = None

    async def generate_response(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate response from LLM."""
        if not self.client:
            return "Mock response: LLM service not configured"

        start_time = time.time()

        try:
            if self.provider == "anthropic":
                response = await self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens or self.max_tokens,
                    temperature=temperature or self.temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                result = response.content[0].text
                prompt_tokens = response.usage.input_tokens
                completion_tokens = response.usage.output_tokens

            elif self.provider == "openai":
                response = await self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=max_tokens or self.max_tokens,
                    temperature=temperature or self.temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                result = response.choices[0].message.content
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens

            else:
                raise ValueError(f"Unsupported provider: {self.provider}")

            # Log API call
            duration_ms = (time.time() - start_time) * 1000
            log_llm_call(
                model=self.model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                duration_ms=duration_ms,
            )

            return result

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise


class QueryService:
    """Service for executing RAG queries."""

    def __init__(
        self,
        retriever: AdvancedHybridRetriever,
        embedding_service: EmbeddingService,
        graphrag_service: GraphRAGService,
    ):
        """Initialize query service."""
        self.retriever = retriever
        self.embedding_service = embedding_service
        self.graphrag_service = graphrag_service
        self.llm_service = LLMService()
        self.vector_store = retriever.vector_store

        # Load prompt templates
        self.prompts = self._load_prompt_templates()

        logger.info("Initialized QueryService")

    def _load_prompt_templates(self) -> Dict[str, str]:
        """Load prompt templates."""
        return {
            "qa": """You are a clinical data expert specializing in ADaM and SDTM standards.
Answer the following question based on the provided context.

Context:
{context}

Question: {query}

Provide a comprehensive answer using only the information from the context.
If the context doesn't contain enough information, say so clearly.
Focus on clinical trial data standards and best practices.""",
            "code_generation": """You are a clinical data programmer. Generate R/SAS code based on the ADaM specification and SDTM context.

Context:
{context}

Task: {query}

Requirements:
1. Follow CDISC ADaM standards
2. Use clear variable names
3. Include comments explaining key steps
4. Ensure data integrity checks
5. Follow best practices for clinical programming

Generate the complete code with explanation:""",
            "specification": """You are a clinical data scientist. Create or analyze ADaM specifications based on the provided context.

Context:
{context}

Task: {query}

Provide a detailed specification including:
1. Dataset structure
2. Key variables
3. Derivation logic
4. SDTM source data
5. Quality checks

Be thorough and follow CDISC standards.""",
            "global_search": """You are analyzing clinical trial data across multiple studies and domains.

Community Summaries:
{context}

Query: {query}

Synthesize information from the community summaries to answer the query comprehensively.
Focus on patterns, relationships, and insights across the data.""",
            "local_search": """You are examining specific entities and their relationships in clinical trial data.

Local Context:
{context}

Query: {query}

Provide detailed analysis focusing on the specific entities and their relationships.
Include relevant clinical interpretations and implications.""",
        }

    async def execute_query(self, query: Query) -> QueryResponse:
        """Execute a RAG query."""
        start_time = time.time()

        try:
            # Retrieve relevant chunks
            retrieval_results = await self.retriever.retrieve(query)

            # Get graph context if needed
            graph_context = ""
            if query.mode in [QueryMode.GLOBAL_SEARCH, QueryMode.LOCAL_SEARCH]:
                graph_context = await self.graphrag_service.get_graph_context(
                    query.text
                )

            # Build prompt
            prompt = self._build_prompt(query, retrieval_results, graph_context)

            # Generate response
            answer = await self.llm_service.generate_response(prompt)

            # Create response
            response = QueryResponse(
                query_id=query.id,
                answer=answer,
                sources=retrieval_results,
                mode=query.mode,
                llm_model=self.llm_service.model,
                prompt_tokens=len(prompt.split()),  # Approximate
                completion_tokens=len(answer.split()),  # Approximate
                duration_ms=(time.time() - start_time) * 1000,
            )

            return response

        except Exception as e:
            logger.error(f"Query execution failed: {e}", exc_info=True)
            raise

    def _build_prompt(
        self, query: Query, retrieval_results: List, graph_context: Dict[str, Any]
    ) -> str:
        """Build prompt for LLM."""
        # Get template
        template_key = query.mode.value
        if template_key not in self.prompts:
            template_key = "qa"

        template = self.prompts[template_key]

        # Build context
        context_parts = []

        # Add retrieval context
        if retrieval_results:
            context_parts.append("Retrieved Information:")
            for i, result in enumerate(retrieval_results[:5], 1):
                chunk_content = result.chunk.get("content", "")
                context_parts.append(f"{i}. {chunk_content[:500]}...")

        # Add graph context
        if graph_context:
            context_parts.append("\nGraph Context:")
            if graph_context.get("entities"):
                context_parts.append(
                    f"Related Entities: {', '.join(graph_context['entities'])}"
                )
            if graph_context.get("communities"):
                context_parts.append(
                    f"Relevant Communities: {graph_context['communities']}"
                )
            if graph_context.get("relations"):
                context_parts.append(
                    f"Key Relationships: {graph_context['relations'][:3]}"
                )

        context = "\n\n".join(context_parts)

        # Format prompt
        prompt = template.format(context=context, query=query.text)

        return prompt

    async def generate_code(
        self, query_text: str, study_id: Optional[str] = None, top_k: int = 10
    ) -> Dict[str, Any]:
        """Generate ADaM/SDTM code."""
        # Create query
        query = Query(
            id=str(time.time()),
            text=query_text,
            mode=QueryMode.CODE_GENERATION,
            study_id=study_id,
            top_k=top_k,
        )

        # Execute query
        response = await self.execute_query(query)

        # Parse code and explanation
        answer = response.answer
        code = ""
        explanation = ""

        # Simple parsing - look for code blocks
        if "```" in answer:
            parts = answer.split("```")
            for i in range(1, len(parts), 2):
                if parts[i]:
                    code += parts[i] + "\n"
            explanation = parts[0] + parts[-1] if len(parts) > 1 else answer
        else:
            explanation = answer

        return {
            "query_id": response.query_id,
            "code": code.strip(),
            "explanation": explanation.strip(),
            "sources": [
                {
                    "chunk_id": r.chunk.get("id"),
                    "content": r.chunk.get("content", "")[:200] + "...",
                    "score": r.score,
                }
                for r in response.sources
            ],
        }
