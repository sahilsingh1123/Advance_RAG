# Low-Level Design (LLD) - Advance RAG System

## 1. Component Detailed Design

### 1.1 API Layer

#### 1.1.1 FastAPI Application Structure
```python
# src/advance_rag/api/main.py
app = FastAPI(
    title="Advance RAG API",
    description="Production-Grade RAG System for ADaM/SDTM Analysis",
    version="0.1.0",
    lifespan=lifespan,
)

# Middleware Stack
app.add_middleware(CORSMiddleware, ...)
app.add_middleware(GZipMiddleware, ...)
app.add_middleware("http", log_requests)
```

#### 1.1.2 Router Architecture
```python
# API Endpoints Structure
/api/v1/
├── auth/
│   ├── POST /login
│   ├── POST /register
│   └── POST /refresh
├── query/
│   ├── POST / (execute query)
│   ├── GET /history
│   └── POST /generate-code
├── ingestion/
│   ├── POST /files
│   ├── GET /status/{job_id}
│   └── POST /bulk
└── health/
    ├── GET /
    └── GET /detailed
```

#### 1.1.3 Request/Response Models
```python
# src/advance_rag/models/schemas.py
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    mode: QueryMode = QueryMode.QA
    study_id: Optional[str] = None
    top_k: int = Field(default=10, ge=1, le=50)
    filters: Dict[str, Any] = Field(default_factory=dict)

class QueryResponse(BaseModel):
    query_id: str
    answer: str
    sources: List[RetrievalResult]
    mode: QueryMode
    llm_model: str
    prompt_tokens: int
    completion_tokens: int
    duration_ms: float
    timestamp: datetime
```

### 1.2 Retrieval Engine

#### 1.2.1 Hybrid Retrieval Architecture
```python
# src/advance_rag/retrieval/advanced_hybrid.py
class AdvancedHybridRetriever:
    def __init__(self, vector_store, embedding_service):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.query_analyzer = QueryAnalyzer()
        self.query_cache = {}
    
    async def retrieve(self, query: Query) -> List[RetrievalResult]:
        # 1. Query Analysis
        analysis = self.query_analyzer.analyze_query(query.text)
        strategy = analysis["strategy"]
        
        # 2. Strategy Selection & Execution
        if strategy == SearchStrategy.SEMANTIC:
            results = await self._semantic_search(query)
        elif strategy == SearchStrategy.KEYWORD:
            results = await self._keyword_search(query)
        elif strategy == SearchStrategy.HYBRID:
            results = await self._hybrid_search(query)
        else:  # ADAPTIVE
            results = await self._adaptive_search(query, analysis)
        
        # 3. Advanced Reranking
        results = await self._advanced_rerank(results, query)
        
        return results
```

#### 1.2.2 Query Analysis Algorithm
```python
class QueryAnalyzer:
    def analyze_query(self, query_text: str) -> Dict[str, Any]:
        query_lower = query_text.lower()
        
        # Feature Extraction
        features = {
            "is_question": self._has_question_words(query_lower),
            "medical_term_count": self._count_medical_terms(query_lower),
            "has_identifiers": self._has_identifiers(query_text),
            "word_count": len(query_lower.split()),
            "is_complex": self._is_complex(query_text)
        }
        
        # Strategy Selection Logic
        if features["has_identifiers"]:
            strategy = SearchStrategy.KEYWORD
        elif features["is_complex"] and features["medical_term_count"] > 1:
            strategy = SearchStrategy.SEMANTIC
        elif features["is_question"]:
            strategy = SearchStrategy.HYBRID
        else:
            strategy = SearchStrategy.ADAPTIVE
        
        return {"strategy": strategy, **features}
```

#### 1.2.3 Vector Search Implementation
```python
async def _semantic_search(self, query: Query) -> List[RetrievalResult]:
    # 1. Query Embedding
    query_embedding = await self.embedding_service.generate_embedding(query.text)
    
    # 2. Query Expansion
    expanded_queries = await self._expand_query(query.text)
    
    # 3. Parallel Search Execution
    search_tasks = []
    for expanded_query in [query.text] + expanded_queries[:2]:
        expanded_embedding = await self.embedding_service.generate_embedding(expanded_query)
        task = self.vector_store.vector_search(
            query_embedding=expanded_embedding,
            top_k=query.top_k,
            study_id=query.study_id,
            threshold=settings.VECTOR_SIMILARITY_THRESHOLD
        )
        search_tasks.append(task)
    
    # 4. Result Aggregation
    all_results = await asyncio.gather(*search_tasks)
    unique_results = self._deduplicate_results(all_results)
    
    return await self._create_retrieval_results(unique_results, "semantic")
```

#### 1.2.4 Result Fusion Algorithm
```python
def _advanced_fusion(self, vector_results, text_results, dense_weight):
    # Reciprocal Rank Fusion (RRF)
    k = 60
    combined_scores = {}
    
    # Vector Results
    for rank, (doc_id, score) in enumerate(vector_results):
        rrf_score = 1.0 / (k + rank + 1)
        combined_scores[doc_id] = dense_weight * rrf_score
    
    # Text Results
    for rank, (doc_id, score) in enumerate(text_results):
        rrf_score = 1.0 / (k + rank + 1)
        if doc_id in combined_scores:
            combined_scores[doc_id] += (1 - dense_weight) * rrf_score
        else:
            combined_scores[doc_id] = (1 - dense_weight) * rrf_score
    
    return sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
```

### 1.3 GraphRAG System

#### 1.3.1 Entity Extraction Pipeline
```python
# src/advance_rag/graph/advanced_graphrag.py
class AdvancedEntityExtractor:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self._load_entity_patterns()
        self._load_relation_patterns()
    
    def extract_entities(self, text: str) -> List[Entity]:
        entities = []
        entity_counter = 0
        
        # Pattern-based Extraction
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    entity_name = self._extract_entity_name(match)
                    confidence = self._calculate_entity_confidence(entity_name, entity_type, match)
                    
                    if confidence > 0.5:
                        entity = Entity(
                            id=f"entity_{entity_counter}",
                            type=entity_type,
                            name=entity_name,
                            confidence=confidence,
                            context=self._extract_context(text, match.start(), match.end()),
                            properties={"source": "pattern_extraction"},
                            source_spans=[(match.start(), match.end())]
                        )
                        entities.append(entity)
                        entity_counter += 1
        
        return self._deduplicate_entities(entities)
```

#### 1.3.2 Graph Construction Algorithm
```python
class GraphBuilder:
    async def build_graph_from_documents(self, documents: List[Document], 
                                        neo4j_service: Neo4jService):
        # 1. Entity & Relation Extraction
        all_entities = []
        all_relations = []
        
        for doc in documents:
            entities = self.entity_extractor.extract_entities(doc.content)
            relations = self.entity_extractor.extract_relations(doc.content, entities)
            
            # Add document context
            for entity in entities:
                entity.properties['document_id'] = doc.id
                entity.properties['study_id'] = doc.study_id
            
            all_entities.extend(entities)
            all_relations.extend(relations)
        
        # 2. Deduplication
        deduplicated_entities = self._deduplicate_entities_across_docs(all_entities)
        deduplicated_relations = self._deduplicate_relations_across_docs(all_relations)
        
        # 3. Graph Analysis
        nx_graph = self._build_networkx_graph(deduplicated_entities, deduplicated_relations)
        communities = self._detect_communities(nx_graph)
        centrality = self._calculate_centrality(nx_graph)
        
        # 4. Storage
        await self._store_in_neo4j(deduplicated_entities, deduplicated_relations, 
                                   communities, neo4j_service)
```

#### 1.3.3 Community Detection
```python
def _detect_communities(self, graph: nx.Graph) -> List[Dict[str, Any]]:
    communities = []
    
    # DBSCAN Clustering
    if len(graph) > 10:
        adj_matrix = nx.adjacency_matrix(graph).todense()
        clustering = DBSCAN(eps=0.5, min_samples=2, metric='cosine')
        labels = clustering.fit_predict(adj_matrix)
        
        for label in set(labels):
            if label != -1:  # Not noise
                nodes = [node for i, node in enumerate(graph.nodes()) if labels[i] == label]
                if len(nodes) >= 2:
                    communities.append({
                        "id": f"community_{label}",
                        "nodes": nodes,
                        "size": len(nodes),
                        "algorithm": "dbscan"
                    })
    
    # Fallback to K-means
    if not communities and len(graph) > 5:
        n_clusters = min(5, len(graph) // 2)
        clustering = KMeans(n_clusters=n_clusters, random_state=42)
        node_embeddings = self._get_node_embeddings(graph)
        labels = clustering.fit_predict(node_embeddings)
        
        for label in range(n_clusters):
            nodes = [node for i, node in enumerate(graph.nodes()) if labels[i] == label]
            if len(nodes) >= 2:
                communities.append({
                    "id": f"community_{label}",
                    "nodes": nodes,
                    "size": len(nodes),
                    "algorithm": "kmeans"
                })
    
    return communities
```

### 1.4 Data Storage Layer

#### 1.4.1 Vector Store Schema
```sql
-- PostgreSQL with pgvector
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    study_id VARCHAR(50) NOT NULL,
    document_type VARCHAR(50) NOT NULL,
    title TEXT,
    content TEXT NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding vector(384),  -- For all-MiniLM-L6-v2
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- HNSW Index for fast vector search
CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX ON chunks (study_id, document_type);
CREATE INDEX ON chunks (document_id);
```

#### 1.4.2 Neo4j Graph Schema
```cypher
-- Node Constraints
CREATE CONSTRAINT entity_id_unique FOR (e:Entity) REQUIRE e.id IS UNIQUE;
CREATE CONSTRAINT document_id_unique FOR (d:Document) REQUIRE d.id IS UNIQUE;

-- Entity Nodes
(:Entity {
    id: String,
    name: String,
    type: String,  // SUBJECT, TREATMENT, LAB_TEST, etc.
    confidence: Float,
    context: String,
    properties: Map
})

(:Document {
    id: String,
    study_id: String,
    type: String,
    title: String
})

(:Community {
    id: String,
    size: Integer,
    algorithm: String,
    properties: Map
})

-- Relationship Types
-[:RECEIVED {confidence: Float, context: String}]->
-[:EXPERIENCED {confidence: Float, context: String}]->
-[:TREATED_WITH {confidence: Float, context: String}]->
-[:PART_OF {document_id: String}]->
-[:MEMBER_OF {community_id: String}]->
```

#### 1.4.3 Redis Cache Structure
```python
# Cache Keys Structure
CACHE_KEYS = {
    "query": "query:{query_hash}",
    "embedding": "embed:{text_hash}",
    "document": "doc:{document_id}",
    "user_session": "session:{user_id}",
    "rate_limit": "rate:{user_id}:{endpoint}",
    "graph_cache": "graph:{entity_name}:{entity_type}"
}

# Cache TTL Configuration
CACHE_TTL = {
    "query": 300,  # 5 minutes
    "embedding": 3600,  # 1 hour
    "document": 1800,  # 30 minutes
    "user_session": 86400,  # 24 hours
    "rate_limit": 60,  # 1 minute
    "graph_cache": 1800  # 30 minutes
}
```

### 1.5 LLM Integration

#### 1.5.1 LLM Service Architecture
```python
class LLMService:
    def __init__(self):
        self.provider = settings.LLM_PROVIDER
        self.model = settings.LLM_MODEL
        self.max_tokens = settings.LLM_MAX_TOKENS
        self.temperature = settings.LLM_TEMPERATURE
        
        # Initialize client based on provider
        if self.provider == "anthropic":
            self.client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        elif self.provider == "openai":
            self.client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    
    async def generate_response(self, prompt: str, **kwargs) -> str:
        # Provider-specific implementation
        if self.provider == "anthropic":
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=kwargs.get('max_tokens', self.max_tokens),
                temperature=kwargs.get('temperature', self.temperature),
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        
        elif self.provider == "openai":
            response = await self.client.chat.completions.create(
                model=self.model,
                max_tokens=kwargs.get('max_tokens', self.max_tokens),
                temperature=kwargs.get('temperature', self.temperature),
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
```

#### 1.5.2 Prompt Template System
```python
class PromptTemplates:
    TEMPLATES = {
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

Be thorough and follow CDISC standards."""
    }
    
    @classmethod
    def get_template(cls, template_name: str) -> str:
        return cls.TEMPLATES.get(template_name, cls.TEMPLATES["qa"])
```

### 1.6 Background Processing

#### 1.6.1 Celery Task Architecture
```python
# src/advance_rag/tasks/celery_app.py
celery_app = Celery(
    "advance_rag",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=[
        "advance_rag.tasks.ingestion_tasks",
        "advance_rag.tasks.embedding_tasks",
        "advance_rag.tasks.graph_tasks",
        "advance_rag.tasks.maintenance_tasks"
    ]
)

# Task Configuration
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes
    task_soft_time_limit=25 * 60,  # 25 minutes
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
)
```

#### 1.6.2 Ingestion Task Pipeline
```python
# src/advance_rag/tasks/ingestion_tasks.py
@celery_app.task(bind=True, max_retries=3)
def process_document(self, document_id: str, file_path: str):
    try:
        # 1. Document Validation
        document = validate_document(file_path)
        
        # 2. Content Extraction
        content = extract_content(file_path, document.type)
        
        # 3. Chunking
        chunks = chunk_content(content, document.metadata)
        
        # 4. Parallel Processing
        tasks = []
        for chunk in chunks:
            # Embedding Generation
            embed_task = generate_embedding_task.delay(chunk.id, chunk.content)
            tasks.append(embed_task)
            
            # Entity Extraction
            entity_task = extract_entities_task.delay(chunk.id, chunk.content)
            tasks.append(entity_task)
        
        # 5. Wait for completion
        results = asyncio.gather(*tasks)
        
        # 6. Storage
        store_document.delay(document_id, chunks, results)
        
        return {"status": "completed", "document_id": document_id}
        
    except Exception as exc:
        logger.error(f"Document processing failed: {exc}")
        raise self.retry(exc=exc, countdown=60)
```

### 1.7 Configuration Management

#### 1.7.1 Settings Architecture
```python
# src/advance_rag/core/config.py
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )
    
    # Database Configuration
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/advance_rag"
    )
    REDIS_URL: str = Field(default="redis://localhost:6379")
    NEO4J_URI: str = Field(default="bolt://localhost:7687")
    
    # Vector Store Configuration
    VECTOR_INDEX_TYPE: str = Field(default="hnsw")
    VECTOR_HNSW_M: int = Field(default=16)
    VECTOR_HNSW_EF_CONSTRUCTION: int = Field(default=64)
    VECTOR_SIMILARITY_THRESHOLD: float = Field(default=0.7)
    
    # LLM Configuration
    LLM_PROVIDER: str = Field(default="anthropic")
    LLM_MODEL: str = Field(default="claude-3-opus-20240229")
    LLM_MAX_TOKENS: int = Field(default=4096)
    LLM_TEMPERATURE: float = Field(default=0.1)
    
    # Performance Configuration
    MAX_CONCURRENT_REQUESTS: int = Field(default=1000)
    RATE_LIMIT_PER_MINUTE: int = Field(default=60)
    CACHE_TTL: int = Field(default=300)
    
    # Security Configuration
    SECRET_KEY: str = Field(default="your-secret-key-here")
    JWT_ALGORITHM: str = Field(default="HS256")
    JWT_EXPIRE_MINUTES: int = Field(default=1440)
```

### 1.8 Error Handling & Resilience

#### 1.8.1 Exception Hierarchy
```python
# src/advance_rag/core/exceptions.py
class AdvanceRAGException(Exception):
    """Base exception for Advance RAG system."""
    pass

class ValidationException(AdvanceRAGException):
    """Raised when input validation fails."""
    pass

class RetrievalException(AdvanceRAGException):
    """Raised when retrieval fails."""
    pass

class LLMException(AdvanceRAGException):
    """Raised when LLM generation fails."""
    pass

class DatabaseException(AdvanceRAGException):
    """Raised when database operation fails."""
    pass

class AuthenticationException(AdvanceRAGException):
    """Raised when authentication fails."""
    pass

class AuthorizationException(AdvanceRAGException):
    """Raised when authorization fails."""
    pass
```

#### 1.8.2 Retry Mechanism
```python
# src/advance_rag/core/resilience.py
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((DatabaseException, LLMException))
)
async def resilient_llm_call(prompt: str) -> str:
    """Make resilient LLM call with retry logic."""
    try:
        return await llm_service.generate_response(prompt)
    except Exception as e:
        logger.warning(f"LLM call failed, retrying: {e}")
        raise

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=8),
    retry=retry_if_exception_type((DatabaseException,))
)
async def resilient_vector_search(embedding: np.ndarray) -> List[Tuple[str, float]]:
    """Make resilient vector search with retry logic."""
    try:
        return await vector_store.vector_search(embedding)
    except Exception as e:
        logger.warning(f"Vector search failed, retrying: {e}")
        raise
```

### 1.9 Monitoring & Observability

#### 1.9.1 Metrics Collection
```python
# src/advance_rag/core/metrics.py
from prometheus_client import Counter, Histogram, Gauge

# Request Metrics
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"]
)

REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration",
    ["method", "endpoint"]
)

# RAG Metrics
QUERY_COUNT = Counter(
    "rag_queries_total",
    "Total RAG queries",
    ["mode", "status"]
)

QUERY_DURATION = Histogram(
    "rag_query_duration_seconds",
    "RAG query duration",
    ["mode"]
)

RETRIEVAL_RESULTS = Histogram(
    "retrieval_results_count",
    "Number of retrieval results",
    ["strategy"]
)

# Database Metrics
DB_CONNECTION_POOL = Gauge(
    "db_connection_pool_size",
    "Database connection pool size"
)

VECTOR_SEARCH_DURATION = Histogram(
    "vector_search_duration_seconds",
    "Vector search duration"
)
```

#### 1.9.2 Distributed Tracing
```python
# src/advance_rag/core/tracing.py
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

class TracingConfig:
    def __init__(self):
        self.tracer_provider = TracerProvider()
        self.otlp_exporter = OTLPSpanExporter(
            endpoint=settings.OTEL_EXPORTER_OTLP_ENDPOINT
        )
        self.span_processor = BatchSpanProcessor(self.otlp_exporter)
        
    def setup_tracing(self, app):
        self.tracer_provider.add_span_processor(self.span_processor)
        trace.set_tracer_provider(self.tracer_provider)
        
        # FastAPI instrumentation
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        FastAPIInstrumentor.instrument_app(app)
        
        # Database instrumentation
        from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
        SQLAlchemyInstrumentor.instrument()
        
        # Redis instrumentation
        from opentelemetry.instrumentation.redis import RedisInstrumentor
        RedisInstrumentor.instrument()
```

### 1.10 Security Implementation

#### 1.10.1 Authentication System
```python
# src/advance_rag/core/auth.py
class AuthenticationService:
    def __init__(self):
        self.secret_key = settings.SECRET_KEY
        self.algorithm = settings.JWT_ALGORITHM
        self.expire_minutes = settings.JWT_EXPIRE_MINUTES
    
    def create_access_token(self, data: dict) -> str:
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=self.expire_minutes)
        to_encode.update({"exp": expire})
        
        encoded_jwt = jwt.encode(
            to_encode, 
            self.secret_key, 
            algorithm=self.algorithm
        )
        return encoded_jwt
    
    def verify_token(self, token: str) -> Optional[dict]:
        try:
            payload = jwt.decode(
                token, 
                self.secret_key, 
                algorithms=[self.algorithm]
            )
            return payload
        except jwt.PyJWTError:
            return None
    
    def hash_password(self, password: str) -> str:
        return pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        return pwd_context.verify(plain_password, hashed_password)
```

#### 1.10.2 Authorization Middleware
```python
# src/advance_rag/core/middleware.py
class AuthorizationMiddleware:
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            request = Request(scope, receive)
            
            # Extract token
            authorization = request.headers.get("Authorization")
            if not authorization or not authorization.startswith("Bearer "):
                await self._send_error(send, 401, "Missing or invalid token")
                return
            
            token = authorization.split(" ")[1]
            payload = auth_service.verify_token(token)
            
            if not payload:
                await self._send_error(send, 401, "Invalid token")
                return
            
            # Add user info to request state
            scope["user"] = payload
        
        await self.app(scope, receive, send)
```

This low-level design provides detailed implementation specifications for each component of the Advance RAG system, including code structures, algorithms, data schemas, and integration patterns.
