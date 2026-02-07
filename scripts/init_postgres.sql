"""Initialize PostgreSQL database with pgvector extension."""

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create chunks table
CREATE TABLE IF NOT EXISTS chunks (
    id VARCHAR(255) PRIMARY KEY,
    content TEXT NOT NULL,
    document_id VARCHAR(255) NOT NULL,
    document_type VARCHAR(50) NOT NULL,
    chunk_index INTEGER NOT NULL,
    start_char INTEGER NOT NULL,
    end_char INTEGER NOT NULL,
    metadata JSONB,
    embedding vector(384),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    study_id VARCHAR(255)
);

-- Create indexes
CREATE INDEX IF NOT EXISTS chunks_document_id_idx ON chunks(document_id);
CREATE INDEX IF NOT EXISTS chunks_study_id_idx ON chunks(study_id);
CREATE INDEX IF NOT EXISTS chunks_document_type_idx ON chunks(document_type);

-- Create HNSW index for vector search
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
ON chunks
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Create full-text search index
CREATE INDEX IF NOT EXISTS chunks_content_fts_idx
ON chunks USING gin(to_tsvector('english', content));

-- Create documents table
CREATE TABLE IF NOT EXISTS documents (
    id VARCHAR(255) PRIMARY KEY,
    title VARCHAR(500) NOT NULL,
    content TEXT,
    document_type VARCHAR(50) NOT NULL,
    file_path VARCHAR(1000),
    file_size BIGINT,
    study_id VARCHAR(255),
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for documents
CREATE INDEX IF NOT EXISTS documents_study_id_idx ON documents(study_id);
CREATE INDEX IF NOT EXISTS documents_type_idx ON documents(document_type);

-- Create studies table
CREATE TABLE IF NOT EXISTS studies (
    id VARCHAR(255) PRIMARY KEY,
    title VARCHAR(500) NOT NULL,
    description TEXT,
    protocol_id VARCHAR(255),
    sap_id VARCHAR(255),
    sdtm_domains TEXT[],
    adam_datasets TEXT[],
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create users table
CREATE TABLE IF NOT EXISTS users (
    id VARCHAR(255) PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    role VARCHAR(50) NOT NULL,
    studies TEXT[],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_login TIMESTAMP WITH TIME ZONE
);

-- Create queries table for logging
CREATE TABLE IF NOT EXISTS queries (
    id VARCHAR(255) PRIMARY KEY,
    text TEXT NOT NULL,
    mode VARCHAR(50) NOT NULL,
    study_id VARCHAR(255),
    user_id VARCHAR(255),
    filters JSONB,
    top_k INTEGER DEFAULT 10,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create query_results table for tracking responses
CREATE TABLE IF NOT EXISTS query_results (
    id VARCHAR(255) PRIMARY KEY,
    query_id VARCHAR(255) REFERENCES queries(id),
    answer TEXT,
    sources JSONB,
    llm_model VARCHAR(100),
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    duration_ms FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for queries
CREATE INDEX IF NOT EXISTS queries_study_id_idx ON queries(study_id);
CREATE INDEX IF NOT EXISTS queries_user_id_idx ON queries(user_id);
CREATE INDEX IF NOT EXISTS queries_created_at_idx ON queries(created_at);

-- Create ingestion_tasks table
CREATE TABLE IF NOT EXISTS ingestion_tasks (
    id VARCHAR(255) PRIMARY KEY,
    file_paths TEXT[],
    status VARCHAR(50) NOT NULL,
    progress FLOAT DEFAULT 0.0,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Grant permissions (adjust as needed)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO advance_rag_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO advance_rag_user;
