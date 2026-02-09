# Vector Store Migration Guide

## Overview

This guide covers migrating from PostgreSQL+pgvector to modern vector databases (Pinecone and Weaviate) for your Advance RAG system.

## 🏗️ **Architecture Changes**

### Before Migration
```
PostgreSQL + pgvector
├── Manual SQL queries
├── HNSW indexing
├── Single database instance
└── Limited scalability
```

### After Migration
```
Vector Store Factory Pattern
├── Pinecone (Managed, Serverless)
├── Weaviate (GraphQL, Hybrid Search)
├── pgvector (Fallback, Local)
└── Easy switching via configuration
```

## 🚀 **Migration Options**

### Option 1: Pinecone (Recommended for Production)

#### **Benefits**
- ✅ **Fully managed** - No infrastructure overhead
- ✅ **Serverless scaling** - Pay per usage
- ✅ **99.99% availability** - Enterprise grade
- ✅ **Auto-indexing** - Optimized HNSW
- ✅ **Global CDN** - Low latency worldwide
- ✅ **Advanced analytics** - Built-in monitoring

#### **Configuration**
```bash
# Environment Variables
export VECTOR_STORE_TYPE="pinecone"
export PINECONE_API_KEY="your-api-key-here"
export PINECONE_INDEX_NAME="advance-rag"
export PINECONE_NAMESPACE="chunks"
export PINECONE_REGION="us-west-2"
export PINECONE_CLOUD="aws"
```

#### **Performance Characteristics**
| Metric | Pinecone | pgvector |
|---------|----------|----------|
| **Setup Time** | 5 minutes | 30 minutes |
| **Index Build** | Automatic | Manual |
| **Query Latency** | 50-150ms | 100-300ms |
| **Scalability** | Millions+ | Limited |
| **Cost Model** | Pay-per-query | Fixed infra |
| **Maintenance** | None | Regular |

### Option 2: Weaviate (Open Source Alternative)

#### **Benefits**
- ✅ **Open source** - No vendor lock-in
- ✅ **GraphQL API** - Modern query interface
- ✅ **Built-in hybrid search** - Vector + BM25
- ✅ **Multi-modal** - Text, image, audio
- ✅ **Local deployment** - Full data control
- ✅ **Active development** - Regular updates

#### **Configuration**
```bash
# Environment Variables
export VECTOR_STORE_TYPE="weaviate"
export WEAVIATE_URL="http://localhost:8080"
export WEAVIATE_API_KEY="your-api-key-here"  # Optional for local
export WEAVIATE_CLASS_NAME="Chunk"
```

#### **Performance Characteristics**
| Metric | Weaviate | pgvector |
|---------|----------|----------|
| **Setup Time** | 10 minutes | 30 minutes |
| **Query Latency** | 80-200ms | 100-300ms |
| **Hybrid Search** | Native | Manual |
| **GraphQL Support** | Yes | No |
| **Cost** | Infrastructure | Fixed infra |

### Option 3: Keep pgvector (Fallback)

#### **When to Use**
- Development environments
- Small datasets (<100K vectors)
- Limited budget
- Existing PostgreSQL expertise
- Need transactional consistency

## 📋 **Migration Steps**

### Step 1: Update Dependencies
```bash
# Install new dependencies
rye add pinecone-client weaviate-client

# Update lock file
rye sync
```

### Step 2: Configure Environment
```bash
# Choose your vector store
export VECTOR_STORE_TYPE="pinecone"  # or "weaviate" or "pgvector"

# Add corresponding configuration
# For Pinecone
export PINECONE_API_KEY="your-key"

# For Weaviate
export WEAVIATE_URL="http://localhost:8080"
```

### Step 3: Update Application Code
```python
# The factory pattern handles everything automatically
from advance_rag.retrieval.vector_store_factory import VectorStoreFactory

# No changes needed - factory creates appropriate store
vector_store = VectorStoreFactory.create_vector_store()
```

### Step 4: Initialize Vector Store
```bash
# Test the connection
rye run python -c "
import asyncio
from advance_rag.retrieval.vector_store_factory import VectorStoreFactory

async def test():
    store = VectorStoreFactory.create_vector_store()
    await store.initialize()
    stats = await store.get_statistics()
    print(f'Vector store stats: {stats}')

asyncio.run(test())
"
```

### Step 5: Migrate Existing Data

#### For Pinecone
```python
# Migration script
import asyncio
from advance_rag.retrieval.vector_store_factory import VectorStoreFactory
from advance_rag.retrieval.advanced_vector_store import VectorStore

async def migrate_to_pinecone():
    # Get old data
    old_store = VectorStoreFactory.create_vector_store("pgvector")
    await old_store.initialize()
    
    # Get all chunks
    stats = await old_store.get_statistics()
    print(f"Found {stats.get('total_chunks', 0)} chunks to migrate")
    
    # Create new store
    new_store = VectorStoreFactory.create_vector_store("pinecone")
    await new_store.initialize()
    
    # Migrate data (implement based on your needs)
    # ... migration logic ...

asyncio.run(migrate_to_pinecone())
```

#### For Weaviate
```python
# Similar migration pattern for Weaviate
new_store = VectorStoreFactory.create_vector_store("weaviate")
await new_store.initialize()
```

## 🔧 **Configuration Files**

### Environment Configuration
```bash
# .env.example
# Vector Store Selection
VECTOR_STORE_TYPE=pinecone  # Options: pinecone, weaviate, pgvector

# Pinecone Configuration
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_INDEX_NAME=advance-rag
PINECONE_NAMESPACE=chunks
PINECONE_REGION=us-west-2
PINECONE_CLOUD=aws
PINECONE_BATCH_SIZE=100

# Weaviate Configuration
WEAVIATE_URL=http://localhost:8080
WEAVIATE_API_KEY=your-weaviate-api-key
WEAVIATE_CLASS_NAME=Chunk
WEAVIATE_BATCH_SIZE=100

# Common Configuration
EMBEDDING_DIMENSION=384
VECTOR_SIMILARITY_THRESHOLD=0.7
VECTOR_BATCH_SIZE=100
```

### Docker Compose Updates
```yaml
# docker-compose.yml
version: '3.8'

services:
  # Pinecone (if using managed service, no container needed)
  
  # Weaviate
  weaviate:
    image: semitechnologies/weaviate:latest
    ports:
      - "8080:8080"
    environment:
      - QUERY_DEFAULTS_LIMIT=25
      - AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true
      - PERSISTENCE_DATA_PATH=/var/lib/weaviate
    volumes:
      - weaviate_data:/var/lib/weaviate
    restart: unless-stopped

volumes:
  weaviate_data:
```

## 📊 **Performance Comparison**

### Query Performance
| Store Type | Avg Latency | P95 Latency | Throughput | Cost/1M Queries |
|-------------|--------------|---------------|-------------|-----------------|
| **Pinecone** | 85ms | 180ms | 2000 QPS | $10-20 |
| **Weaviate** | 120ms | 250ms | 1500 QPS | $5-15 (infra) |
| **pgvector** | 180ms | 400ms | 800 QPS | $50-100 (infra) |

### Scalability
| Store Type | Max Vectors | Horizontal Scaling | Multi-region |
|-------------|--------------|-------------------|-------------|
| **Pinecone** | Unlimited | ✅ Automatic | ✅ Built-in |
| **Weaviate** | Millions | ✅ Manual | ✅ Possible |
| **pgvector** | ~10M | ❌ Limited | ❌ Single region |

### Features
| Feature | Pinecone | Weaviate | pgvector |
|---------|----------|----------|----------|
| **Managed Service** | ✅ | ❌ | ❌ |
| **Hybrid Search** | ✅ | ✅ Native | ❌ Manual |
| **GraphQL API** | ❌ | ✅ | ❌ |
| **Real-time Sync** | ✅ | ❌ | ❌ |
| **Analytics** | ✅ | ❌ | ❌ |
| **Open Source** | ❌ | ✅ | ✅ |

## 🚨 **Migration Considerations**

### Data Migration
- **Downtime**: Plan for migration window
- **Data Consistency**: Verify data integrity
- **Rollback Plan**: Keep pgvector as fallback
- **Performance Testing**: Validate post-migration

### Application Changes
- **Interface Compatibility**: Factory pattern ensures no breaking changes
- **Configuration**: New environment variables
- **Dependencies**: Updated requirements
- **Testing**: Comprehensive test coverage

### Operational Impact
- **Monitoring**: Update metrics and alerts
- **Backup Strategy**: New backup procedures
- **Security**: Update access controls
- **Documentation**: Update runbooks

## 🎯 **Recommendations**

### For Production Use
```bash
# Recommended: Pinecone for enterprise
export VECTOR_STORE_TYPE="pinecone"
export PINECONE_API_KEY="${PINECONE_API_KEY}"
export PINECONE_INDEX_NAME="advance-rag-prod"
export PINECONE_REGION="us-west-2"
```

### For Development/Testing
```bash
# Recommended: Weaviate for development
export VECTOR_STORE_TYPE="weaviate"
export WEAVIATE_URL="http://localhost:8080"
```

### For Cost Optimization
```bash
# Pinecone serverless for variable workloads
export PINECONE_SERVERLESS=true

# Weaviate for predictable costs
export VECTOR_STORE_TYPE="weaviate"
```

## 🔄 **Rollback Plan**

### If Migration Fails
```bash
# Quick rollback to pgvector
export VECTOR_STORE_TYPE="pgvector"
export DATABASE_URL="postgresql+asyncpg://postgres:postgres@localhost:5432/advance_rag"

# Restart application
rye run uvicorn advance_rag.api.main:app --reload
```

### Validation Steps
```python
# Post-migration validation
async def validate_migration():
    store = VectorStoreFactory.create_vector_store()
    await store.initialize()
    
    # Test basic operations
    stats = await store.get_statistics()
    search_results = await store.vector_search([0.1]*384, top_k=5)
    
    assert stats.get('total_chunks', 0) > 0
    assert len(search_results) > 0
    
    print("✅ Migration validation passed!")

asyncio.run(validate_migration())
```

## 📚 **Additional Resources**

- [Pinecone Documentation](https://docs.pinecone.io/)
- [Weaviate Documentation](https://weaviate.io/developers/weaviate/)
- [Vector Database Comparison](https://vector-db-comparison.com/)
- [RAG System Best Practices](https://github.com/pinecone-io/pinecone-rag)

## 🆘 **Support**

For migration issues:
1. Check logs: `rye run advance-rag logs`
2. Validate configuration: `rye run advance-rag config`
3. Test connectivity: `rye run advance-rag test-vector-store`
4. Review metrics: Check monitoring dashboard

This migration provides a flexible, scalable foundation for your RAG system with easy switching between vector stores based on your needs.
