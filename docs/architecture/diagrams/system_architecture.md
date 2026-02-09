# System Architecture Diagrams

## 1. Overall System Architecture

```mermaid
graph TB
    subgraph "Client Layer"
        WEB[Web UI]
        MOBILE[Mobile App]
        API_CLIENT[API Clients]
        CLI[CLI Tools]
    end
    
    subgraph "API Gateway"
        LB[Load Balancer]
        GATEWAY[FastAPI Gateway]
        AUTH[Authentication Service]
        RATE[Rate Limiting]
        CORS[CORS Handler]
    end
    
    subgraph "Application Layer"
        QUERY_SVC[Query Service]
        INGEST_SVC[Ingestion Service]
        EXPORT_SVC[Export Service]
        NOTIFY_SVC[Notification Service]
        BACKUP_SVC[Backup Service]
    end
    
    subgraph "Processing Layer"
        RETRIEVAL[Hybrid Retrieval Engine]
        GRAPHRAG[GraphRAG Engine]
        LLM_SVC[LLM Integration]
        CELERY[Celery Workers]
    end
    
    subgraph "Data Layer"
        PG[(PostgreSQL+pgvector)]
        REDIS[(Redis Cache)]
        NEO4J[(Neo4j Graph)]
        S3[(File Storage)]
    end
    
    subgraph "Infrastructure"
        K8S[Kubernetes]
        MONITOR[Monitoring]
        LOGS[Logging]
        CI_CD[CI/CD Pipeline]
    end
    
    WEB --> LB
    MOBILE --> LB
    API_CLIENT --> LB
    CLI --> LB
    
    LB --> GATEWAY
    GATEWAY --> AUTH
    GATEWAY --> RATE
    GATEWAY --> CORS
    
    AUTH --> QUERY_SVC
    AUTH --> INGEST_SVC
    AUTH --> EXPORT_SVC
    
    QUERY_SVC --> RETRIEVAL
    INGEST_SVC --> CELERY
    RETRIEVAL --> GRAPHRAG
    RETRIEVAL --> LLM_SVC
    
    RETRIEVAL --> PG
    RETRIEVAL --> REDIS
    GRAPHRAG --> NEO4J
    CELERY --> PG
    CELERY --> NEO4J
    LLM_SVC --> S3
    
    GATEWAY --> MONITOR
    GATEWAY --> LOGS
    
    K8S --> GATEWAY
    K8S --> CELERY
    K8S --> PG
    K8S --> REDIS
    K8S --> NEO4J
```

## 2. Query Processing Flow

```mermaid
sequenceDiagram
    participant Client
    participant Gateway
    participant Auth
    participant QuerySvc
    participant Analyzer
    participant Retrieval
    participant GraphRAG
    participant LLM
    participant VectorDB
    participant GraphDB
    participant Cache
    
    Client->>Gateway: POST /v1/query
    Gateway->>Auth: Validate JWT
    Auth-->>Gateway: User Info
    Gateway->>QuerySvc: Forward Request
    
    QuerySvc->>Analyzer: Analyze Query
    Analyzer-->>QuerySvc: Strategy + Features
    
    QuerySvc->>Cache: Check Query Cache
    alt Cache Hit
        Cache-->>QuerySvc: Cached Results
    else Cache Miss
        QuerySvc->>Retrieval: Execute Retrieval
        
        par Vector Search
            Retrieval->>VectorDB: Semantic Search
            VectorDB-->>Retrieval: Vector Results
        and Keyword Search
            Retrieval->>VectorDB: Full-text Search
            VectorDB-->>Retrieval: Text Results
        and Graph Search
            Retrieval->>GraphRAG: Graph Query
            GraphRAG->>GraphDB: Entity/Relation Search
            GraphDB-->>GraphRAG: Graph Results
            GraphRAG-->>Retrieval: Contextual Results
        end
        
        Retrieval->>Retrieval: Result Fusion
        Retrieval->>Retrieval: Re-ranking
        Retrieval-->>QuerySvc: Final Results
        
        QuerySvc->>Cache: Store Results
    end
    
    QuerySvc->>LLM: Generate Response
    LLM-->>QuerySvc: Generated Answer
    
    QuerySvc-->>Gateway: Query Response
    Gateway-->>Client: JSON Response
```

## 3. Data Ingestion Pipeline

```mermaid
flowchart TD
    START([Document Upload]) --> VALIDATE[Validate Document]
    VALIDATE --> |Invalid| ERROR[Return Error]
    VALIDATE --> |Valid| EXTRACT[Extract Content]
    
    EXTRACT --> CHUNK[Intelligent Chunking]
    CHUNK --> EMBED[Generate Embeddings]
    CHUNK --> ENTITY[Extract Entities]
    CHUNK --> RELATION[Extract Relations]
    
    EMBED --> VEC_STORE[Store in Vector DB]
    ENTITY --> GRAPH_STORE[Store in Graph DB]
    RELATION --> GRAPH_STORE
    
    VEC_STORE --> INDEX[Update Indexes]
    GRAPH_STORE --> ANALYZE[Graph Analysis]
    
    ANALYZE --> COMMUNITY[Community Detection]
    ANALYZE --> CENTRALITY[Centrality Calculation]
    
    COMMUNITY --> UPDATE_GRAPH[Update Graph Properties]
    CENTRALITY --> UPDATE_GRAPH
    
    UPDATE_GRAPH --> NOTIFY[Send Notification]
    NOTIFY --> COMPLETE([Ingestion Complete])
    
    style START fill:#e1f5fe
    style COMPLETE fill:#e8f5e8
    style ERROR fill:#ffebee
```

## 4. Hybrid Retrieval Architecture

```mermaid
graph LR
    subgraph "Query Input"
        QUERY[User Query]
        FILTERS[Filters & Context]
    end
    
    subgraph "Query Analysis"
        ANALYZER[Query Analyzer]
        FEATURES[Feature Extraction]
        STRATEGY[Strategy Selection]
    end
    
    subgraph "Retrieval Strategies"
        subgraph "Semantic Search"
            EMBED[Query Embedding]
            EXPAND[Query Expansion]
            VECTOR[Vector Search]
        end
        
        subgraph "Keyword Search"
            TOKENIZE[Tokenization]
            BOOLEAN[Boolean Query]
            TEXT[Full-text Search]
        end
        
        subgraph "Graph Search"
            ENTITY[Entity Extraction]
            TRAVERSE[Graph Traversal]
            CONTEXT[Context Retrieval]
        end
    end
    
    subgraph "Result Processing"
        FUSION[Result Fusion]
        RERANK[Re-ranking]
        DIVERSIFY[Diversification]
        CACHE[Cache Results]
    end
    
    QUERY --> ANALYZER
    FILTERS --> ANALYZER
    ANALYZER --> FEATURES
    FEATURES --> STRATEGY
    
    STRATEGY --> EMBED
    STRATEGY --> TOKENIZE
    STRATEGY --> ENTITY
    
    EMBED --> EXPAND
    EXPAND --> VECTOR
    TOKENIZE --> BOOLEAN
    BOOLEAN --> TEXT
    ENTITY --> TRAVERSE
    TRAVERSE --> CONTEXT
    
    VECTOR --> FUSION
    TEXT --> FUSION
    CONTEXT --> FUSION
    
    FUSION --> RERANK
    RERANK --> DIVERSIFY
    DIVERSIFY --> CACHE
```

## 5. GraphRAG Architecture

```mermaid
graph TB
    subgraph "Document Processing"
        DOC[Clinical Documents]
        NLP[NLP Processing]
        ENTITY_EXTRACT[Entity Extraction]
        RELATION_EXTRACT[Relation Extraction]
    end
    
    subgraph "Graph Construction"
        ENTITIES[Entities]
        RELATIONS[Relations]
        GRAPH_BUILDER[Graph Builder]
        NETWORKX[NetworkX Analysis]
    end
    
    subgraph "Graph Analysis"
        COMMUNITY[Community Detection]
        CENTRALITY[Centrality Analysis]
        CLUSTERING[Clustering]
        EMBEDDINGS[Node Embeddings]
    end
    
    subgraph "Neo4j Storage"
        NODES[Entity Nodes]
        EDGES[Relation Edges]
        COMMUNITIES[Community Nodes]
        PROPERTIES[Graph Properties]
    end
    
    subgraph "Query Processing"
        QUERY_GRAPH[Graph Query]
        PATH_FINDING[Path Finding]
        REASONING[Multi-hop Reasoning]
        CONTEXT_GRAPH[Graph Context]
    end
    
    DOC --> NLP
    NLP --> ENTITY_EXTRACT
    NLP --> RELATION_EXTRACT
    
    ENTITY_EXTRACT --> ENTITIES
    RELATION_EXTRACT --> RELATIONS
    
    ENTITIES --> GRAPH_BUILDER
    RELATIONS --> GRAPH_BUILDER
    
    GRAPH_BUILDER --> NETWORKX
    NETWORKX --> COMMUNITY
    NETWORKX --> CENTRALITY
    NETWORKX --> CLUSTERING
    NETWORKX --> EMBEDDINGS
    
    COMMUNITY --> COMMUNITIES
    CENTRALITY --> PROPERTIES
    ENTITIES --> NODES
    RELATIONS --> EDGES
    
    QUERY_GRAPH --> PATH_FINDING
    PATH_FINDING --> REASONING
    REASONING --> CONTEXT_GRAPH
```

## 6. Database Schema Architecture

```mermaid
erDiagram
    POSTGRES {
        uuid id PK
        varchar study_id
        varchar document_type
        text title
        text content
        jsonb metadata
        timestamp created_at
    }
    
    CHUNKS {
        uuid id PK
        uuid document_id FK
        integer chunk_index
        text content
        vector embedding
        jsonb metadata
        timestamp created_at
    }
    
    ENTITIES {
        string id PK
        string name
        string type
        float confidence
        text context
        jsonb properties
        string document_id FK
    }
    
    RELATIONS {
        string id PK
        string subject FK
        string object FK
        string relation_type
        float confidence
        text context
        jsonb properties
    }
    
    COMMUNITIES {
        string id PK
        string name
        integer size
        string algorithm
        jsonb properties
        jsonb nodes
    }
    
    USERS {
        uuid id PK
        varchar email UK
        varchar password_hash
        varchar role
        jsonb permissions
        timestamp created_at
        timestamp last_login
    }
    
    QUERIES {
        uuid id PK
        uuid user_id FK
        text query_text
        varchar mode
        jsonb filters
        jsonb response
        integer prompt_tokens
        integer completion_tokens
        float duration_ms
        timestamp created_at
    }
    
    POSTGRES ||--o{ CHUNKS : "has"
    CHUNKS ||--o{ ENTITIES : "contains"
    ENTITIES ||--o{ RELATIONS : "subject"
    ENTITIES ||--o{ RELATIONS : "object"
    COMMUNITIES ||--o{ ENTITIES : "contains"
    USERS ||--o{ QUERIES : "executes"
```

## 7. Microservices Communication

```mermaid
graph TB
    subgraph "API Gateway"
        GATEWAY[FastAPI Gateway]
        AUTH_MW[Auth Middleware]
        RATE_MW[Rate Limiting]
        LOG_MW[Logging Middleware]
    end
    
    subgraph "Core Services"
        QUERY_SVC[Query Service]
        INGEST_SVC[Ingestion Service]
        USER_SVC[User Service]
        EXPORT_SVC[Export Service]
    end
    
    subgraph "Support Services"
        EMBED_SVC[Embedding Service]
        GRAPH_SVC[Graph Service]
        NOTIFY_SVC[Notification Service]
        BACKUP_SVC[Backup Service]
    end
    
    subgraph "Background Workers"
        CELERY_WORKER[Celery Worker]
        SCHEDULER[Celery Beat]
    end
    
    subgraph "Data Stores"
        POSTGRES[(PostgreSQL)]
        REDIS[(Redis)]
        NEO4J[(Neo4j)]
        S3[(S3 Storage)]
    end
    
    subgraph "External APIs"
        OPENAI[OpenAI API]
        ANTHROPIC[Anthropic API]
        SMTP[SMTP Service]
    end
    
    GATEWAY --> AUTH_MW
    AUTH_MW --> RATE_MW
    RATE_MW --> LOG_MW
    
    LOG_MW --> QUERY_SVC
    LOG_MW --> INGEST_SVC
    LOG_MW --> USER_SVC
    LOG_MW --> EXPORT_SVC
    
    QUERY_SVC --> EMBED_SVC
    QUERY_SVC --> GRAPH_SVC
    INGEST_SVC --> EMBED_SVC
    INGEST_SVC --> GRAPH_SVC
    
    EMBED_SVC --> OPENAI
    EMBED_SVC --> ANTHROPIC
    GRAPH_SVC --> NEO4J
    
    NOTIFY_SVC --> SMTP
    BACKUP_SVC --> S3
    
    CELERY_WORKER --> POSTGRES
    CELERY_WORKER --> REDIS
    CELERY_WORKER --> NEO4J
    
    SCHEDULER --> CELERY_WORKER
    
    QUERY_SVC --> POSTGRES
    QUERY_SVC --> REDIS
    USER_SVC --> POSTGRES
```

## 8. Security Architecture

```mermaid
graph TB
    subgraph "External Access"
        CLIENT[Client Applications]
        FIREWALL[Web Application Firewall]
        DDOS[DDoS Protection]
    end
    
    subgraph "Authentication & Authorization"
        OAUTH[OAuth 2.0 Provider]
        JWT[JWT Service]
        RBAC[Role-Based Access Control]
        MFA[Multi-Factor Authentication]
    end
    
    subgraph "API Security"
        RATE_LIMIT[Rate Limiting]
        INPUT_VALID[Input Validation]
        OUTPUT_SAN[Output Sanitization]
        CORS[CORS Policy]
    end
    
    subgraph "Data Protection"
        ENCRYPTION[Encryption Service]
        PHI_REDACTION[PHI Redaction]
        ANONYMIZATION[Data Anonymization]
        AUDIT[Audit Logging]
    end
    
    subgraph "Infrastructure Security"
        VPC[Virtual Private Cloud]
        SECURITY_GROUPS[Security Groups]
        IAM[Identity & Access Management]
        SECRETS[Secrets Manager]
    end
    
    subgraph "Monitoring & Compliance"
        SIEM[Security Information & Event Management]
        COMPLIANCE[Compliance Monitoring]
        INCIDENT[Incident Response]
        BACKUP[Secure Backup]
    end
    
    CLIENT --> FIREWALL
    FIREWALL --> DDOS
    DDOS --> OAUTH
    
    OAUTH --> JWT
    JWT --> RBAC
    RBAC --> MFA
    
    MFA --> RATE_LIMIT
    RATE_LIMIT --> INPUT_VALID
    INPUT_VALID --> OUTPUT_SAN
    OUTPUT_SAN --> CORS
    
    CORS --> ENCRYPTION
    ENCRYPTION --> PHI_REDACTION
    PHI_REDACTION --> ANONYMIZATION
    ANONYMIZATION --> AUDIT
    
    AUDIT --> VPC
    VPC --> SECURITY_GROUPS
    SECURITY_GROUPS --> IAM
    IAM --> SECRETS
    
    SECRETS --> SIEM
    SIEM --> COMPLIANCE
    COMPLIANCE --> INCIDENT
    INCIDENT --> BACKUP
```

## 9. Deployment Architecture

```mermaid
graph TB
    subgraph "CDN & Edge"
        CDN[CloudFront CDN]
        EDGE[Edge Locations]
    end
    
    subgraph "Load Balancing"
        ALB[Application Load Balancer]
        NLB[Network Load Balancer]
    end
    
    subgraph "Kubernetes Cluster"
        subgraph "API Pods"
            API_POD1[API Pod 1]
            API_POD2[API Pod 2]
            API_POD3[API Pod N]
        end
        
        subgraph "Worker Pods"
            WORKER_POD1[Worker Pod 1]
            WORKER_POD2[Worker Pod 2]
            WORKER_POD3[Worker Pod N]
        end
        
        subgraph "System Pods"
            INGRESS[Ingress Controller]
            MONITOR[Monitoring Stack]
            LOGGING[Logging Stack]
        end
    end
    
    subgraph "Database Cluster"
        subgraph "PostgreSQL"
            PG_MASTER[(Primary)]
            PG_REPLICA1[(Replica 1)]
            PG_REPLICA2[(Replica 2)]
        end
        
        subgraph "Redis"
            REDIS_MASTER[(Master)]
            REDIS_SLAVE[(Slave)]
        end
        
        subgraph "Neo4j"
            NEO4J_CORE[(Core Server)]
            NEO4J_REPLICA[(Read Replica)]
        end
    end
    
    subgraph "Storage"
        S3[(S3 Buckets)]
        EFS[(EFS Storage)]
    end
    
    subgraph "Monitoring & Observability"
        PROMETHEUS[Prometheus]
        GRAFANA[Grafana]
        JAEGER[Jaeger Tracing]
        ELASTICSEARCH[Elasticsearch]
    end
    
    CDN --> ALB
    EDGE --> ALB
    ALB --> INGRESS
    NLB --> INGRESS
    
    INGRESS --> API_POD1
    INGRESS --> API_POD2
    INGRESS --> API_POD3
    
    API_POD1 --> PG_MASTER
    API_POD2 --> PG_REPLICA1
    API_POD3 --> PG_REPLICA2
    
    API_POD1 --> REDIS_MASTER
    API_POD2 --> REDIS_MASTER
    API_POD3 --> REDIS_MASTER
    
    WORKER_POD1 --> PG_MASTER
    WORKER_POD2 --> PG_REPLICA1
    WORKER_POD3 --> PG_REPLICA2
    
    API_POD1 --> NEO4J_CORE
    WORKER_POD1 --> NEO4J_CORE
    
    API_POD1 --> S3
    WORKER_POD1 --> S3
    
    MONITOR --> PROMETHEUS
    MONITOR --> GRAFANA
    MONITOR --> JAEGER
    LOGGING --> ELASTICSEARCH
```

## 10. Performance & Scalability

```mermaid
graph LR
    subgraph "Performance Optimization"
        subgraph "Caching Layers"
            L1[L1: Application Cache]
            L2[L2: Redis Cache]
            L3[L3: CDN Cache]
        end
        
        subgraph "Database Optimization"
            INDEXING[Smart Indexing]
            PARTITIONING[Data Partitioning]
            POOLING[Connection Pooling]
        end
        
        subgraph "Query Optimization"
            PARALLEL[Parallel Execution]
            FUSION[Result Fusion]
            RERANK[Intelligent Reranking]
        end
    end
    
    subgraph "Scalability Patterns"
        subgraph "Horizontal Scaling"
            AUTO_SCALE[Auto Scaling]
            LOAD_BALANCE[Load Balancing]
            SHARDING[Data Sharding]
        end
        
        subgraph "Resource Management"
            RATE_LIMIT[Rate Limiting]
            QUEUE_MGMT[Queue Management]
            RESOURCE_ALLOC[Resource Allocation]
        end
    end
    
    subgraph "Monitoring & Optimization"
        PERFORMANCE_METRICS[Performance Metrics]
        BOTTLENECK_ANALYSIS[Bottleneck Analysis]
        AUTO_TUNING[Auto Tuning]
    end
    
    L1 --> L2
    L2 --> L3
    
    INDEXING --> PARTITIONING
    PARTITIONING --> POOLING
    
    PARALLEL --> FUSION
    FUSION --> RERANK
    
    AUTO_SCALE --> LOAD_BALANCE
    LOAD_BALANCE --> SHARDING
    
    RATE_LIMIT --> QUEUE_MGMT
    QUEUE_MGMT --> RESOURCE_ALLOC
    
    PERFORMANCE_METRICS --> BOTTLENECK_ANALYSIS
    BOTTLENECK_ANALYSIS --> AUTO_TUNING
```
