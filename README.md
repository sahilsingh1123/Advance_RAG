# Advance RAG: Production-Grade RAG for ADaM/SDTM Analysis

A robust retrieval-augmented generation (RAG) system specifically designed for clinical data analysis, focusing on ADaM (Analysis Data Model) and SDTM (Study Data Tabulation Model) standards.

## Features

- **Hybrid Retrieval**: Combines semantic (vector) and lexical (keyword) search
- **GraphRAG Integration**: Knowledge graph-based retrieval for complex clinical reasoning
- **Production Ready**: Scalable architecture supporting 50K-100K users/day
- **Domain Specific**: Optimized for clinical trial data and CDISC standards
- **Secure**: HIPAA-compliant with PHI redaction and access controls
- **Rye Managed**: Modern Python package management with Rye

## Architecture

### Core Components

1. **Data Ingestion**: JSON/Markdown processing with intelligent chunking
2. **Embedding Service**: Domain-specific embeddings with Redis caching
3. **Hybrid Retrieval**: Vector search (pgvector) + Lexical search (PostgreSQL/Elasticsearch)
4. **GraphRAG**: Neo4j-based knowledge graph for multi-hop reasoning
5. **API Layer**: FastAPI with async endpoints
6. **Background Tasks**: Celery workers for heavy operations
7. **Authentication**: JWT-based auth with role-based access control
8. **Monitoring**: OpenTelemetry tracing with Prometheus metrics
9. **Notifications**: Multi-channel alert system
10. **Export Service**: Data export in multiple formats
11. **Backup Service**: Automated backup and restore

### Tech Stack

- **Package Manager**: Rye (modern Python package management)
- **Application**: Python 3.12, FastAPI, Uvicorn
- **Databases**: PostgreSQL (pgvector), Redis, Neo4j
- **Vector Search**: pgvector with HNSW indexing
- **Graph**: Neo4j with GDS library
- **Task Queue**: Celery with Redis
- **LLM**: Anthropic Opus 4.5 / OpenAI GPT-4
- **Deployment**: Docker, Kubernetes, AWS

## Quick Start

### Prerequisites

- Python 3.12+ (managed by Rye)
- Rye package manager
- Docker & Docker Compose
- PostgreSQL with pgvector extension
- Redis
- Neo4j

### Installation

1. **Install Rye** (if not already installed):
   ```bash
   curl -sSf https://rye.astral.sh/get | bash
   # or: brew install rye
   ```

2. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd advance-rag
   ```

3. **Install dependencies with Rye**:
   ```bash
   rye sync
   ```

4. **Setup environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration (API keys, database URLs, etc.)
   ```

5. **Start all services**:
   ```bash
   docker-compose up -d
   ```

6. **Initialize databases**:
   ```bash
   # Initialize PostgreSQL
   rye run python scripts/init_postgres.py

   # Initialize Neo4j
   rye run python scripts/init_neo4j.py
   ```

7. **Generate dummy data** (optional):
   ```bash
   rye run advance-rag generate-data --study-id STUDY001 --n-subjects 100
   ```

8. **Ingest data**:
   ```bash
   rye run advance-rag ingest --data-dir data/dummy
   ```

9. **Start the API server**:
   ```bash
   rye run uvicorn advance_rag.api.main:app --reload
   ```

### Docker Deployment

```bash
# Build and start all services
docker-compose up --build

# Initialize databases
docker-compose exec api rye run python scripts/init_postgres.py
docker-compose exec api rye run python scripts/init_neo4j.py

# Start API server
docker-compose exec api rye run uvicorn advance_rag.api.main:app
```

## API Usage

### Authentication

First, create an admin user:
```bash
curl -X POST "http://localhost:8000/v1/auth/init-admin"
```

Then login to get a token:
```bash
curl -X POST "http://localhost:8000/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"email": "admin@example.com", "password": "admin123"}'
```

### Query the RAG System

```python
import httpx

# Get token from login response
token = "your_jwt_token_here"

async def query_rag():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/v1/query",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "query": "Generate ADaM code for ADLB dataset",
                "mode": "code_generation",
                "study_id": "STUDY001"
            }
        )
        return response.json()
```

### Data Ingestion

```python
# Ingest new SDTM/ADaM specifications
curl -X POST "http://localhost:8000/v1/ingestion/files" \
  -H "Authorization: Bearer your_token" \
  -H "Content-Type: application/json" \
  -d '{"file_paths": ["path/to/sdtm.json", "path/to/adam.md"]}'
```

## Development

### Running Tests

```bash
# Run all tests
rye run pytest

# Run with coverage
rye run pytest --cov=src --cov-report=html

# Run specific test suites
rye run pytest tests/unit/
rye run pytest tests/integration/
```

### Code Quality

```bash
# Format code
rye run black src/ tests/

# Lint
rye run ruff check src/ tests/

# Type checking
rye run mypy src/
```

### CLI Commands

```bash
# Generate dummy data
rye run advance-rag generate-data --help

# Ingest data
rye run advance-rag ingest --help

# Check system status
rye run advance-rag status

# Start services
rye run advance-rag serve --help
rye run advance-rag worker --help
rye run advance-rag beat --help
```

## Configuration

Key environment variables:

```bash
# Database
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/advance_rag
REDIS_URL=redis://localhost:6379
NEO4J_URI=bolt://localhost:7687

# API Keys
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Embedding Model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384

# Application
MAX_CONCURRENT_REQUESTS=1000
RATE_LIMIT_PER_MINUTE=60

# Authentication
SECRET_KEY=your-secret-key-here-change-in-production
JWT_EXPIRE_MINUTES=1440

# Export/Backup
EXPORT_DIR=exports
BACKUP_DIR=backups
BACKUP_RETENTION_DAYS=30
```

## Monitoring

- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Prometheus Metrics**: http://localhost:8000/metrics
- **Tracing**: OpenTelemetry with OTLP exporter
- **Logging**: Structured JSON logs with correlation IDs

## Security

- **Authentication**: JWT-based auth with OAuth 2.0 support
- **Authorization**: Role-based access control (RBAC)
- **Data Protection**: PHI redaction, encryption at rest and in transit
- **Audit**: Complete query logging for compliance
- **Rate Limiting**: Configurable rate limits per endpoint

## Performance

- **Throughput**: Supports 1-2K QPS at peak
- **Latency**: <500ms p95 for simple queries
- **Scalability**: Horizontal scaling with Kubernetes
- **Caching**: Multi-layer caching (Redis, application, CDN)

## Production Deployment

### Using Rye

```bash
# Install production dependencies
rye sync --no-dev

# Run with production settings
export ENVIRONMENT=production
rye run uvicorn advance_rag.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Using Docker

```bash
# Production build
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Scale workers
docker-compose up -d --scale worker=3
```

## Backup and Restore

```bash
# Create backup
rye run python -c "
import asyncio
from advance_rag.services.backup_service import backup_service
asyncio.run(backup_service.create_full_backup())
"

# List backups
rye run python -c "
import asyncio
from advance_rag.services.backup_service import backup_service
print(asyncio.run(backup_service.list_backups()))
"
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're using `rye run` instead of plain `python`
2. **Database Connection**: Check Docker services are running
3. **API Keys**: Verify API keys are set in `.env`
4. **Port Conflicts**: Change ports in `docker-compose.yml` if needed

### Getting Help

```bash
# Check system status
rye run advance-rag status

# Check logs
docker-compose logs -f api
docker-compose logs -f postgres
docker-compose logs -f neo4j
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Run tests: `rye run pytest`
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

For support and questions:
- Create an issue in the repository
- Check the API documentation at `/docs`
- Review the logs for detailed error messages
