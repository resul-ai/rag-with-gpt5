"""Document management endpoints."""

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_async_db
from app.schemas import DocumentCreate, DocumentResponse
from app.services.document_service import DocumentService

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/", response_model=DocumentResponse)
async def create_document(
    document: DocumentCreate,
    db: AsyncSession = Depends(get_async_db)
):
    """
    Upload a document to the RAG system.

    This endpoint:
    1. Stores the document in the database
    2. Chunks the document text
    3. Generates embeddings for each chunk
    4. Stores embeddings in the vector store
    """
    try:
        doc_service = DocumentService(db)
        result = await doc_service.create_document(
            title=document.title,
            content=document.content,
            metadata=document.metadata
        )
        return result

    except Exception as e:
        logger.error(f"Error creating document: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to create document: {str(e)}")


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: int,
    db: AsyncSession = Depends(get_async_db)
):
    """Get a document by ID."""
    try:
        doc_service = DocumentService(db)
        result = await doc_service.get_document(document_id)

        if not result:
            raise HTTPException(status_code=404, detail="Document not found")

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving document: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve document: {str(e)}")
