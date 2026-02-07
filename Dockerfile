# Multi-stage Dockerfile for production-grade RAG system

# Base stage with Python dependencies
FROM python:3.12-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY pyproject.toml ./
RUN pip install --upgrade pip && \
    pip install -e . --no-deps && \
    pip install -e .[dev]

# Development stage
FROM base as development
ENV ENVIRONMENT=development
CMD ["uvicorn", "advance_rag.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# API stage for production
FROM base as api
ENV ENVIRONMENT=production

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Start API server
CMD ["uvicorn", "advance_rag.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

# Worker stage
FROM base as worker
ENV ENVIRONMENT=production

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Start Celery worker
CMD ["celery", "-A", "advance_rag.core.celery", "worker", "--loglevel=info", "--concurrency=4"]

# Scheduler stage
FROM base as scheduler
ENV ENVIRONMENT=production

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Start Celery beat scheduler
CMD ["celery", "-A", "advance_rag.core.celery", "beat", "--loglevel=info"]

# Testing stage
FROM base as test
ENV ENVIRONMENT=test

# Copy test files
COPY tests/ ./tests/
COPY pytest.ini ./

# Run tests
CMD ["pytest", "tests/", "-v", "--cov=src", "--cov-report=term-missing"]
