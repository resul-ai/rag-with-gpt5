"""Document management service."""

import logging
from typing import Dict, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models import Document
from app.schemas import DocumentResponse
from app.services.vector_service import VectorService

logger = logging.getLogger(__name__)


class DocumentService:
    """Service for managing documents and their embeddings."""

    def __init__(self, db: AsyncSession):
        """Initialize document service."""
        self.db = db
        self.vector_service = VectorService(db)

    def _chunk_text(self, text: str) -> list[str]:
        """
        Split text into chunks for embedding.

        Simple chunking strategy: split by size with overlap.

        Args:
            text: Text to chunk

        Returns:
            List of text chunks
        """
        chunk_size = settings.chunk_size
        overlap = settings.chunk_overlap

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]

            # Avoid very small final chunks
            if len(chunk) > overlap or start == 0:
                chunks.append(chunk)

            start = end - overlap

        logger.info(f"Split text into {len(chunks)} chunks")
        return chunks

    async def create_document(
        self,
        title: str,
        content: str,
        metadata: Dict = None
    ) -> DocumentResponse:
        """
        Create a document and generate embeddings.

        Args:
            title: Document title
            content: Document content
            metadata: Optional metadata

        Returns:
            Created document
        """
        try:
            # Create document
            document = Document(
                title=title,
                content=content,
                metadata=metadata or {}
            )
            self.db.add(document)
            await self.db.flush()  # Get ID without committing

            # Chunk the content
            chunks = self._chunk_text(content)

            # Add chunks with embeddings
            await self.vector_service.add_document_chunks(
                document_id=document.id,
                chunks=chunks
            )

            await self.db.commit()
            await self.db.refresh(document)

            logger.info(f"Created document {document.id} with {len(chunks)} chunks")

            return DocumentResponse.model_validate(document)

        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error creating document: {e}")
            raise

    async def get_document(self, document_id: int) -> Optional[DocumentResponse]:
        """
        Get document by ID.

        Args:
            document_id: Document ID

        Returns:
            Document if found, None otherwise
        """
        try:
            result = await self.db.execute(
                select(Document).where(Document.id == document_id)
            )
            document = result.scalar_one_or_none()

            if document:
                return DocumentResponse.model_validate(document)
            return None

        except Exception as e:
            logger.error(f"Error getting document: {e}")
            raise

    async def delete_document(self, document_id: int) -> bool:
        """
        Delete document and its chunks.

        Args:
            document_id: Document ID

        Returns:
            True if deleted, False if not found
        """
        try:
            result = await self.db.execute(
                select(Document).where(Document.id == document_id)
            )
            document = result.scalar_one_or_none()

            if not document:
                return False

            # Delete chunks (via cascade)
            await self.db.delete(document)
            await self.db.commit()

            logger.info(f"Deleted document {document_id}")
            return True

        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error deleting document: {e}")
            raise
