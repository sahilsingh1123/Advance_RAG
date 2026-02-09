# Technology Stack & Models Used

## 1. Core Technology Stack

### 1.1 Backend Technologies
| Technology | Version | Purpose | Key Features |
|------------|---------|---------|--------------|
| **Python** | 3.12+ | Core programming language | Async/await, type hints, performance |
| **FastAPI** | 0.128.3+ | Web framework | High performance, automatic docs, async |
| **Uvicorn** | 0.40.0+ | ASGI server | Lightning-fast, HTTP/2 support |
| **Pydantic** | 2.12.5+ | Data validation | Type safety, automatic serialization |
| **Rye** | Latest | Package management | Modern dependency management, lock files |

### 1.2 Database Technologies
| Technology | Version | Purpose | Configuration |
|------------|---------|---------|---------------|
| **PostgreSQL** | 15+ | Primary database + vector storage | pgvector extension, HNSW indexing |
| **Redis** | 7.1+ | Caching & session storage | In-memory, pub/sub, persistence |
| **Neo4j** | 6.1+ | Knowledge graph storage | Cypher queries, GDS library |
| **pgvector** | 0.4.2+ | Vector similarity search | HNSW indexing, cosine similarity |

### 1.3 AI/ML Technologies
| Technology | Version | Purpose | Model/Algorithm |
|------------|---------|---------|-----------------|
| **Sentence Transformers** | 5.2.2+ | Text embeddings | all-MiniLM-L6-v2 (384 dimensions) |
| **scikit-learn** | 1.8.0+ | Machine learning | DBSCAN, K-means clustering |
| **NetworkX** | 3.6.1+ | Graph analysis | Community detection, centrality |
| **spaCy** | 3.7+ | NLP processing | Entity extraction, tokenization |
| **NumPy** | 1.24+ | Numerical computing | Vector operations, embeddings |

### 1.4 LLM Integration
| Provider | Model | Context Window | Use Case |
|----------|-------|----------------|----------|
| **Anthropic** | Claude 3 Opus | 200K tokens | Primary response generation |
| **OpenAI** | GPT-4 Turbo | 128K tokens | Fallback, specialized tasks |
| **Local Models** | Llama 2 7B/13B | 4K tokens | Development, testing |

## 2. Vector Database Configuration

### 2.1 pgvector Setup
```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create vector table
CREATE TABLE chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id),
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding vector(384),  -- all-MiniLM-L6-v2 dimensions
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- HNSW index for fast approximate search
CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Additional indexes for filtering
CREATE INDEX ON chunks (document_id);
CREATE INDEX ON chunks (study_id);
CREATE INDEX ON chunks (document_type);
```

### 2.2 Vector Search Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| **Embedding Dimension** | 384 | all-MiniLM-L6-v2 output |
| **Index Type** | HNSW | Hierarchical Navigable Small World |
| **M Parameter** | 16 | Number of bi-directional links |
| **ef_construction** | 64 | Index construction accuracy |
| **ef_search** | 40 | Query time accuracy |
| **Similarity Metric** | Cosine | Angular distance for embeddings |

## 3. Knowledge Graph Configuration

### 3.1 Neo4j Schema Design
```cypher
-- Entity types for clinical data
CREATE CONSTRAINT entity_id_unique FOR (e:Entity) REQUIRE e.id IS UNIQUE;
CREATE CONSTRAINT document_id_unique FOR (d:Document) REQUIRE d.id IS UNIQUE;

-- Entity nodes
(:Entity {
    id: String,
    name: String,
    type: String,  // SUBJECT, TREATMENT, LAB_TEST, ADVERSE_EVENT, etc.
    confidence: Float,
    context: String,
    properties: Map,
    document_ids: [String],
    study_id: String
})

(:Document {
    id: String,
    study_id: String,
    type: String,
    title: String,
    content: String,
    metadata: Map
})

(:Community {
    id: String,
    size: Integer,
    algorithm: String,  // dbscan, kmeans
    properties: Map,
    nodes: [String]
})
```

### 3.2 Graph Algorithms Used
| Algorithm | Purpose | Implementation |
|-----------|---------|----------------|
| **DBSCAN** | Community detection | scikit-learn, density-based |
| **K-means** | Fallback clustering | scikit-learn, centroid-based |
| **PageRank** | Node importance | NetworkX, centrality measure |
| **Betweenness** | Bridge nodes | NetworkX, centrality measure |
| **Closeness** | Information flow | NetworkX, centrality measure |

## 4. Embedding Models

### 4.1 Primary Embedding Model
```python
# all-MiniLM-L6-v2 Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
MAX_SEQUENCE_LENGTH = 512
BATCH_SIZE = 32

# Model Characteristics
- **Architecture**: Transformer-based
- **Training Data**: 1B+ sentence pairs
- **Performance**: Fast inference, good quality
- **Use Case**: General purpose semantic search
```

### 4.2 Embedding Pipeline
```python
class EmbeddingService:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.cache = RedisCache()
    
    async def generate_embedding(self, text: str) -> np.ndarray:
        # Check cache first
        cache_key = hashlib.md5(text.encode()).hexdigest()
        cached = await self.cache.get(f"embed:{cache_key}")
        if cached:
            return np.array(cached)
        
        # Generate embedding
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2 normalization
            batch_size=1,
            show_progress_bar=False
        )
        
        # Cache result
        await self.cache.set(
            f"embed:{cache_key}",
            embedding.tolist(),
            ttl=3600  # 1 hour
        )
        
        return embedding
```

## 5. LLM Integration Details

### 5.1 Anthropic Claude 3 Opus Configuration
```python
ANTHROPIC_CONFIG = {
    "model": "claude-3-opus-20240229",
    "max_tokens": 4096,
    "temperature": 0.1,
    "top_p": 0.9,
    "top_k": 40,
    "context_window": 200000,
    "use_case": "Primary response generation"
}
```

### 5.2 Prompt Engineering Templates
```python
PROMPT_TEMPLATES = {
    "qa": {
        "system": "You are a clinical data expert specializing in ADaM and SDTM standards.",
        "context": "Answer based on provided clinical trial documentation.",
        "requirements": [
            "Use only information from context",
            "Focus on CDISC standards",
            "Provide comprehensive answers"
        ]
    },
    
    "code_generation": {
        "system": "You are a clinical data programmer.",
        "context": "Generate R/SAS code for ADaM datasets.",
        "requirements": [
            "Follow CDISC ADaM standards",
            "Include clear comments",
            "Add data integrity checks",
            "Use best practices"
        ]
    },
    
    "specification": {
        "system": "You are a clinical data scientist.",
        "context": "Create ADaM specifications from SDTM data.",
        "requirements": [
            "Define dataset structure",
            "Specify key variables",
            "Include derivation logic",
            "Add quality checks"
        ]
    }
}
```

## 6. Task Queue Configuration

### 6.1 Celery Setup
```python
# Celery Configuration
CELERY_CONFIG = {
    "broker_url": "redis://localhost:6379/0",
    "result_backend": "redis://localhost:6379/0",
    "task_serializer": "json",
    "result_serializer": "json",
    "accept_content": ["json"],
    "timezone": "UTC",
    "enable_utc": True,
    "task_track_started": True,
    "task_time_limit": 30 * 60,  # 30 minutes
    "task_soft_time_limit": 25 * 60,  # 25 minutes
    "worker_prefetch_multiplier": 1,
    "worker_max_tasks_per_child": 1000,
    "task_routes": {
        "advance_rag.tasks.ingestion_tasks.*": {"queue": "ingestion"},
        "advance_rag.tasks.embedding_tasks.*": {"queue": "embedding"},
        "advance_rag.tasks.graph_tasks.*": {"queue": "graph"},
        "advance_rag.tasks.maintenance_tasks.*": {"queue": "maintenance"}
    }
}
```

### 6.2 Task Types & Queues
| Queue | Tasks | Priority | Concurrency |
|-------|-------|----------|-------------|
| **ingestion** | Document processing, chunking | High | 4 workers |
| **embedding** | Embedding generation | High | 8 workers |
| **graph** | Graph construction, analysis | Medium | 2 workers |
| **maintenance** | Backup, cleanup, indexing | Low | 1 worker |

## 7. Monitoring & Observability Stack

### 7.1 Metrics Collection
```python
# Prometheus Metrics
METRICS_CONFIG = {
    "prometheus_client": "0.24.1",
    "metrics_port": 9090,
    "scrape_interval": "15s",
    "retention": "15d"
}

# Key Metrics
- **HTTP Metrics**: Request count, duration, error rate
- **RAG Metrics**: Query volume, retrieval performance
- **Database Metrics**: Connection pool, query performance
- **LLM Metrics**: Token usage, API latency, cost
- **Business Metrics**: User engagement, feature usage
```

### 7.2 Distributed Tracing
```python
# OpenTelemetry Configuration
TRACING_CONFIG = {
    "opentelemetry_api": "1.39.1",
    "opentelemetry_sdk": "1.39.1",
    "otlp_endpoint": "http://jaeger:4317",
    "sample_rate": 0.1,  # 10% sampling
    "service_name": "advance-rag",
    "service_version": "0.1.0"
}

# Instrumentations
- FastAPI instrumentation
- SQLAlchemy instrumentation
- Redis instrumentation
- HTTP client instrumentation
```

## 8. Security Technologies

### 8.1 Authentication & Authorization
```python
# JWT Configuration
JWT_CONFIG = {
    "algorithm": "HS256",
    "secret_key": "your-secret-key-here",
    "expire_minutes": 1440,  # 24 hours
    "refresh_expire_days": 30
}

# Password Hashing
PASSWORD_CONFIG = {
    "scheme": "bcrypt",
    "deprecated": "auto",
    "bcrypt__rounds": 12
}
```

### 8.2 Data Protection
```python
# Encryption Configuration
ENCRYPTION_CONFIG = {
    "at_rest": "AES-256-GCM",
    "in_transit": "TLS-1.3",
    "key_management": "AWS KMS",
    "rotation_period": "90d"
}

# PHI Redaction
PHI_CONFIG = {
    "entities": ["PERSON", "LOCATION", "DATE", "PHONE", "EMAIL"],
    "model": "en_core_web_sm",
    "confidence_threshold": 0.8
}
```

## 9. Infrastructure Technologies

### 9.1 Container Technologies
```dockerfile
# Multi-stage Docker Build
FROM python:3.12-slim as base
# System dependencies

FROM base as builder
# Python dependencies

FROM base as runtime
# Application code
```

### 9.2 Kubernetes Configuration
```yaml
# Deployment Configuration
apiVersion: apps/v1
kind: Deployment
metadata:
  name: advance-rag-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: advance-rag-api
  template:
    spec:
      containers:
      - name: api
        image: advance-rag:latest
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
```

## 10. Development & Testing Tools

### 10.1 Development Tools
| Tool | Version | Purpose |
|------|---------|---------|
| **Black** | 26.1.0+ | Code formatting |
| **Ruff** | 0.15.0+ | Linting & formatting |
| **MyPy** | 1.19.1+ | Type checking |
| **pytest** | 9.0.2+ | Testing framework |
| **pytest-asyncio** | 1.3.0+ | Async testing |

### 10.2 Testing Configuration
```python
# pytest.ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --strict-markers
    --strict-config
    --cov=src
    --cov-report=html
    --cov-report=term-missing
    --cov-fail-under=80
asyncio_mode = auto
```

## 11. Performance Benchmarks

### 11.1 Expected Performance
| Metric | Target | Measurement |
|--------|--------|-------------|
| **API Response Time** | <500ms p95 | Request duration |
| **Vector Search** | <100ms | Embedding similarity |
| **Graph Query** | <200ms | Neo4j traversal |
| **LLM Generation** | <2s | Response generation |
| **Throughput** | 1-2K QPS | Queries per second |

### 11.2 Scaling Characteristics
| Scale Level | API Pods | Workers | Database | Expected QPS |
|-------------|----------|---------|----------|-------------|
| **Small** | 2 | 4 | Single instance | 100-200 |
| **Medium** | 6 | 12 | Primary + 1 replica | 500-1000 |
| **Large** | 12 | 24 | Primary + 2 replicas | 1000-2000 |
| **XLarge** | 24+ | 48+ | Sharded cluster | 2000+ |

## 12. Cost Optimization

### 12.1 Resource Optimization
| Component | Optimization Strategy |
|-----------|---------------------|
| **Compute** | Auto-scaling, spot instances |
| **Storage** | Data lifecycle policies, compression |
| **Network** | CDN, compression, caching |
| **LLM API** | Prompt optimization, caching |
| **Database** | Read replicas, connection pooling |

### 12.2 Monitoring Costs
```python
# Cost Tracking Metrics
COST_METRICS = {
    "llm_tokens": "Input + Output tokens",
    "api_calls": "Number of LLM API calls",
    "storage_gb": "Storage usage in GB",
    "compute_hours": "Compute instance hours",
    "data_transfer": "Network data transfer"
}
```

This comprehensive technology stack provides the foundation for a production-grade RAG system optimized for clinical data analysis with high performance, scalability, and reliability.
