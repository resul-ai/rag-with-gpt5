"""Conversation and RAG query endpoints."""

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_async_db
from app.schemas import RAGQueryRequest, RAGQueryResponse
from app.services.rag_service import RAGService

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/messages", response_model=RAGQueryResponse)
async def create_message(
    request: RAGQueryRequest,
    db: AsyncSession = Depends(get_async_db)
):
    """
    Submit a query to the RAG system.

    This endpoint:
    1. Retrieves relevant documents from the vector store
    2. Generates a response using OpenAI GPT-4o
    3. Stores the conversation in the database
    4. Returns the response with source documents
    """
    try:
        # Initialize RAG service
        rag_service = RAGService(db)

        # Process query
        response = await rag_service.process_query(
            query=request.query,
            conversation_id=request.conversation_id,
            top_k=request.top_k or 5,
            include_sources=request.include_sources
        )

        return response

    except Exception as e:
        logger.error(f"Error processing RAG query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process query: {str(e)}")
