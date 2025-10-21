"""Main FastAPI application for RAG-Anything API."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.db import close_db, init_db
from app.routers import conversation, documents, health

# Configure logging
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown."""
    # Startup
    logger.info("Starting RAG-Anything API service...")
    try:
        await init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down RAG-Anything API service...")
    try:
        await close_db()
        logger.info("Database connections closed")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


# Create FastAPI application
app = FastAPI(
    title="RAG-Anything API",
    description="A flexible RAG (Retrieval-Augmented Generation) API service",
    version="1.0.0",
    lifespan=lifespan,
    debug=settings.debug,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(conversation.router, prefix="/api/conversation", tags=["Conversation"])
app.include_router(documents.router, prefix="/api/documents", tags=["Documents"])


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "RAG-Anything API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "description": "A flexible RAG API service powered by OpenAI and pgvector"
    }
