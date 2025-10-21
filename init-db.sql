-- Database initialization script for RAG-Anything
-- This script runs when PostgreSQL container starts for the first time

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE rag_db TO rag_user;

-- Log initialization
DO $$
BEGIN
    RAISE NOTICE 'RAG-Anything database initialized successfully';
    RAISE NOTICE 'pgvector extension enabled';
END $$;
