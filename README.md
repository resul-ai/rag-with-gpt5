# RAG-with-GPT5-API

A production-ready, flexible **Retrieval-Augmented Generation (RAG)** API service powered by OpenAI's GPT-5-nano and PostgreSQL with pgvector.

## 🎯 Overview

RAG-with-gpt5 is a RESTful API service that enables intelligent question-answering over your custom knowledge base using state-of-the-art language models and vector similarity search.

### Key Features

- 🤖 **OpenAI GPT-5-nano** - Cost-effective and fast language model
- 🔍 **Vector Similarity Search** - PostgreSQL with pgvector extension
- 📚 **Document Management** - Upload, store, and chunk documents automatically
- 💬 **Conversational RAG** - Context-aware question answering with source citations
- 🐳 **Docker-Ready** - Fully containerized with Docker Compose
- 📊 **Swagger Documentation** - Interactive API docs at `/docs`
- ✅ **Health Monitoring** - Built-in health check endpoints

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      RAG-Anything API                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │   FastAPI    │───▶│  RAG Engine  │───▶│   OpenAI     │ │
│  │  REST API    │    │   Service    │    │ GPT-5-nano   │ │
│  └──────────────┘    └──────────────┘    └──────────────┘ │
│         │                    │                              │
│         │                    │                              │
│  ┌──────▼────────────────────▼──────────┐                  │
│  │     PostgreSQL + pgvector             │                  │
│  │  ┌─────────────┬──────────────────┐  │                  │
│  │  │  Documents  │  Vector Embeddings│  │                  │
│  │  │  Metadata   │  Similarity Search│  │                  │
│  │  └─────────────┴──────────────────┘  │                  │
│  └───────────────────────────────────────┘                  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Docker Desktop installed
- OpenAI API key
- 8GB RAM minimum
- 10GB free disk space

### Installation

1. **Clone the repository**
   ```bash
   cd /Users/res/Projects/rag-anything
   ```

2. **Configure environment variables**

   The `.env` file is already configured with:
   ```bash
   # OpenAI Configuration
   OPENAI_API_KEY=your_api_key_here
   OPENAI_MODEL=gpt-5-nano
   OPENAI_EMBEDDING_MODEL=text-embedding-3-small

   # Database Configuration
   DATABASE_ASYNC_URL=postgresql+asyncpg://rag_user:rag_password@postgres:5432/rag_db

   # Service Configuration
   SERVICE_PORT=8000
   LOG_LEVEL=INFO

   # RAG Configuration
   RAG_TOP_K=5
   RAG_SIMILARITY_THRESHOLD=0.4
   CHUNK_SIZE=1000
   CHUNK_OVERLAP=200
   ```

3. **Start the services**
   ```bash
   docker-compose up -d
   ```

4. **Verify the deployment**
   ```bash
   curl http://localhost:8000/health
   ```

   Expected response:
   ```json
   {
     "status": "healthy",
     "database": "healthy",
     "openai": "healthy",
     "timestamp": "2025-10-21T12:00:00.000000"
   }
   ```

## 📖 API Documentation

### Interactive Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### API Endpoints

#### Health Check

```bash
GET /health
```

Returns service health status including database and OpenAI connectivity.

#### Root Endpoint

```bash
GET /
```

Returns API information and documentation links.

#### Upload Document

```bash
POST /api/documents/
Content-Type: application/json

{
  "title": "Document Title",
  "content": "Your document content here...",
  "metadata": {
    "category": "documentation",
    "author": "Your Name"
  }
}
```

**Response:**
```json
{
  "id": 1,
  "title": "Document Title",
  "content": "Your document content here...",
  "metadata": {
    "category": "documentation",
    "author": "Your Name"
  },
  "created_at": "2025-10-21T12:00:00.000000",
  "updated_at": "2025-10-21T12:00:00.000000"
}
```

#### RAG Query (Ask Question)

```bash
POST /api/conversation/messages
Content-Type: application/json

{
  "query": "What is RAG-Anything?",
  "top_k": 3,
  "include_sources": true
}
```

**Response:**
```json
{
  "conversation_id": 1,
  "message_id": 2,
  "query": "What is RAG-Anything?",
  "response": "RAG-Anything is a flexible Retrieval-Augmented Generation API service...",
  "sources": [
    {
      "document_id": 1,
      "chunk_id": 1,
      "content": "RAG-Anything is a flexible...",
      "similarity_score": 0.85,
      "metadata": {}
    }
  ],
  "metadata": {
    "model": "gpt-5-nano-2025-08-07",
    "usage": {
      "prompt_tokens": 150,
      "completion_tokens": 50,
      "total_tokens": 200
    },
    "retrieved_chunks": 1
  }
}
```

## 💡 Usage Examples

### Example 1: Upload and Query

```bash
# 1. Upload a document
curl -X POST http://localhost:8000/api/documents/ \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Python Best Practices",
    "content": "Always use virtual environments. Follow PEP 8 style guide. Use type hints for better code quality.",
    "metadata": {"category": "programming"}
  }'

# 2. Ask a question
curl -X POST http://localhost:8000/api/conversation/messages \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are Python best practices?",
    "top_k": 3,
    "include_sources": true
  }'
```

### Example 2: Check Document

```bash
# Get document by ID
curl http://localhost:8000/api/documents/1
```

### Example 3: Health Check

```bash
# Check service health
curl http://localhost:8000/health | python3 -m json.tool
```

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | Required | Your OpenAI API key |
| `OPENAI_MODEL` | `gpt-5-nano` | LLM model to use |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `SERVICE_PORT` | `8000` | API service port |
| `LOG_LEVEL` | `INFO` | Logging level |
| `RAG_TOP_K` | `5` | Number of chunks to retrieve |
| `RAG_SIMILARITY_THRESHOLD` | `0.4` | Minimum similarity score |
| `CHUNK_SIZE` | `1000` | Document chunk size |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |

### Docker Compose Services

- **postgres**: PostgreSQL 15 with pgvector extension (port 5432)
- **api**: FastAPI application (port 8000)

## 🧠 RAG Pipeline

The system processes queries through the following pipeline:

```
1. User Query
   ↓
2. Generate Query Embedding (text-embedding-3-small)
   ↓
3. Vector Similarity Search (pgvector)
   ↓
4. Retrieve Top-K Relevant Chunks
   ↓
5. Build Context Prompt
   ↓
6. Generate Response (gpt-5-nano)
   ↓
7. Return Answer with Sources
```

## 📊 Database Schema

### Documents Table
```sql
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    title VARCHAR(500) NOT NULL,
    content TEXT NOT NULL,
    meta_data JSON DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

### Document Chunks Table (with Vector Embeddings)
```sql
CREATE TABLE document_chunks (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    chunk_text TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    embedding vector(1536),  -- pgvector type
    meta_data JSON DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW()
);
```

### Conversations Table
```sql
CREATE TABLE conversations (
    id SERIAL PRIMARY KEY,
    title VARCHAR(500),
    user_id VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

### Messages Table
```sql
CREATE TABLE messages (
    id SERIAL PRIMARY KEY,
    conversation_id INTEGER REFERENCES conversations(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL,  -- 'user', 'assistant', 'system'
    content TEXT NOT NULL,
    meta_data JSON DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW()
);
```

## 🎯 GPT-5-nano Specifics

**Important**: GPT-5-nano has unique requirements compared to GPT-4:

1. **Parameter Differences**:
   - ✅ Use `max_completion_tokens` (NOT `max_tokens`)
   - ❌ Does NOT support custom `temperature` (default 1.0 only)

2. **Characteristics**:
   - **Cost-effective**: Lower cost per token
   - **Fast**: Faster response times
   - **Model ID**: `gpt-5-nano-2025-08-07`

3. **Code Implementation**:
   ```python
   # Correct usage for gpt-5-nano
   if model.startswith("gpt-5"):
       params["max_completion_tokens"] = 2048
       # Don't set temperature - not supported
   else:
       params["max_tokens"] = 2048
       params["temperature"] = 0.0
   ```

## 🐳 Docker Commands

### Start Services
```bash
docker-compose up -d
```

### Stop Services
```bash
docker-compose down
```

### View Logs
```bash
# All logs
docker-compose logs -f

# API logs only
docker-compose logs -f api

# Last 50 lines
docker-compose logs api --tail=50
```

### Rebuild After Changes
```bash
docker-compose down
docker-compose build --no-cache api
docker-compose up -d
```

### Clean Everything (including volumes)
```bash
docker-compose down -v
```

## 📁 Project Structure

```
rag-anything/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application
│   ├── config.py               # Configuration settings
│   ├── db.py                   # Database setup
│   ├── models.py               # SQLAlchemy models
│   ├── schemas.py              # Pydantic schemas
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── health.py           # Health check endpoints
│   │   ├── conversation.py     # RAG query endpoints
│   │   └── documents.py        # Document management
│   └── services/
│       ├── __init__.py
│       ├── llm_service.py      # OpenAI integration
│       ├── vector_service.py   # Vector search
│       ├── document_service.py # Document processing
│       └── rag_service.py      # RAG orchestration
├── docker-compose.yml          # Docker orchestration
├── Dockerfile                  # API container image
├── requirements.txt            # Python dependencies
├── supervisord.conf            # Process management
├── init-db.sql                 # Database initialization
├── .env                        # Environment variables
└── README.md                   # This file
```

## 🔍 Troubleshooting

### Service Won't Start

```bash
# Check logs
docker-compose logs api

# Check if ports are available
lsof -i :8000
lsof -i :5432

# Restart services
docker-compose restart
```

### Database Connection Issues

```bash
# Check PostgreSQL health
docker-compose exec postgres psql -U rag_user -d rag_db -c "SELECT 1;"

# Check pgvector extension
docker-compose exec postgres psql -U rag_user -d rag_db -c "SELECT * FROM pg_extension WHERE extname='vector';"
```

### OpenAI API Errors

```bash
# Verify API key
docker-compose exec api python3 -c "from app.config import settings; print('Model:', settings.openai_model)"

# Test OpenAI connection
curl http://localhost:8000/health
```

### Model Not Working

If you see errors about `max_tokens` or `temperature`:
- Ensure you're using the latest image: `docker-compose build --no-cache api`
- Verify model is set to `gpt-5-nano` in `.env` and `docker-compose.yml`

## 🔒 Security Considerations

1. **API Key Protection**:
   - Never commit `.env` file to version control
   - Use environment variables in production
   - Rotate API keys regularly

2. **Database Security**:
   - Change default passwords in production
   - Use SSL/TLS for database connections
   - Implement proper access controls

3. **API Security**:
   - Add authentication middleware (not included in this version)
   - Implement rate limiting
   - Use HTTPS in production

## 📈 Performance Optimization

### Recommended Settings for Production

```bash
# .env optimizations
RAG_TOP_K=3                    # Reduce for faster queries
CHUNK_SIZE=800                 # Smaller chunks = better precision
RAG_SIMILARITY_THRESHOLD=0.5   # Higher = more relevant results
```

### Database Optimization

```sql
-- Create index on embeddings for faster search
CREATE INDEX ON document_chunks USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 100);
```

## 🧪 Testing

### Manual Testing

```bash
# 1. Health check
curl http://localhost:8000/health

# 2. Upload test document
curl -X POST http://localhost:8000/api/documents/ \
  -H "Content-Type: application/json" \
  -d '{"title":"Test","content":"Test content","metadata":{}}'

# 3. Query
curl -X POST http://localhost:8000/api/conversation/messages \
  -H "Content-Type: application/json" \
  -d '{"query":"What is this about?"}'
```

### Expected Response Times

- Document upload (1KB): ~500ms
- RAG query: ~2-3 seconds
- Health check: ~100ms

## 📚 Additional Resources

- [OpenAI API Documentation](https://platform.openai.com/docs)
- [pgvector Documentation](https://github.com/pgvector/pgvector)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)

## 🤝 Contributing

This is a production-ready template. Customize it for your needs:

1. Add authentication/authorization
2. Implement file upload (PDF, DOCX, etc.)
3. Add streaming responses
4. Implement conversation history
5. Add multi-user support
6. Integrate with Streamlit UI

## 📝 License

This project is provided as-is for educational and production use.

## 🔄 Version History

- **v1.0.0** (2025-10-21)
  - Initial release
  - GPT-5-nano integration
  - PostgreSQL + pgvector
  - Docker deployment
  - Basic RAG pipeline

## 🎓 How It Works

### Document Ingestion Flow

1. User uploads document via API
2. Document is stored in PostgreSQL
3. Content is split into chunks (1000 chars with 200 char overlap)
4. Each chunk is embedded using `text-embedding-3-small`
5. Embeddings stored as vectors in pgvector

### Query Processing Flow

1. User submits question
2. Question is embedded using same embedding model
3. Vector similarity search finds top-K relevant chunks
4. Context is built from retrieved chunks
5. GPT-5-nano generates answer based on context
6. Response returned with source citations

## 💰 Cost Estimation

Approximate costs using OpenAI GPT-5-nano (prices may vary):

- **Document embedding**: ~$0.0001 per 1000 words
- **Query embedding**: ~$0.00001 per query
- **GPT-5-nano response**: ~$0.0005 per query (avg)

Example: 1000 queries/day ≈ $0.50/day ≈ $15/month

## 🚀 Deployment to Production

### Recommended Steps

1. **Use environment-specific configs**
   ```bash
   # production.env
   DEBUG=false
   LOG_LEVEL=WARNING
   ```

2. **Add reverse proxy (nginx)**
   ```nginx
   server {
       listen 80;
       server_name api.yourdomain.com;

       location / {
           proxy_pass http://localhost:8000;
       }
   }
   ```

3. **Enable SSL/TLS**
   ```bash
   certbot --nginx -d api.yourdomain.com
   ```

4. **Set up monitoring**
   - Prometheus + Grafana
   - CloudWatch / DataDog
   - Sentry for error tracking

5. **Database backups**
   ```bash
   # Automated backup script
   pg_dump -U rag_user rag_db > backup_$(date +%Y%m%d).sql
   ```

---

**Built with ❤️ using FastAPI, OpenAI GPT-5-nano, and PostgreSQL pgvector**

For questions or issues, please check the troubleshooting section or review the API documentation at `/docs`.
