"""Vector store service for similarity search."""

import logging
from typing import List, Tuple

import numpy as np
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models import DocumentChunk
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)


class VectorService:
    """Service for vector similarity search using pgvector."""

    def __init__(self, db: AsyncSession):
        """Initialize vector service."""
        self.db = db
        self.llm_service = LLMService()

    async def add_document_chunks(
        self,
        document_id: int,
        chunks: List[str],
        metadata: List[dict] = None
    ) -> List[DocumentChunk]:
        """
        Add document chunks with embeddings to vector store.

        Args:
            document_id: ID of parent document
            chunks: List of text chunks
            metadata: Optional metadata for each chunk

        Returns:
            List of created DocumentChunk objects
        """
        try:
            # Generate embeddings for all chunks
            embeddings = await self.llm_service.generate_embeddings(chunks)

            # Create DocumentChunk objects
            chunk_objects = []
            for idx, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_meta = metadata[idx] if metadata and idx < len(metadata) else {}

                chunk = DocumentChunk(
                    document_id=document_id,
                    chunk_text=chunk_text,
                    chunk_index=idx,
                    embedding=embedding,
                    metadata=chunk_meta
                )
                self.db.add(chunk)
                chunk_objects.append(chunk)

            await self.db.commit()
            logger.info(f"Added {len(chunk_objects)} chunks for document {document_id}")

            return chunk_objects

        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error adding document chunks: {e}")
            raise

    async def similarity_search(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = None
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Search for similar document chunks using cosine similarity.

        Args:
            query: Search query
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score (0-1)

        Returns:
            List of (DocumentChunk, similarity_score) tuples
        """
        try:
            # Generate query embedding
            query_embeddings = await self.llm_service.generate_embeddings([query])
            query_embedding = query_embeddings[0]

            # Use similarity threshold from settings if not provided
            threshold = similarity_threshold or settings.rag_similarity_threshold

            # Perform similarity search using pgvector
            # Using cosine similarity: 1 - cosine_distance
            # Format embedding as PostgreSQL array
            embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

            query_sql = f"""
                SELECT
                    id,
                    document_id,
                    chunk_text,
                    chunk_index,
                    meta_data,
                    created_at,
                    1 - (embedding <=> '{embedding_str}'::vector) as similarity
                FROM document_chunks
                WHERE 1 - (embedding <=> '{embedding_str}'::vector) >= {threshold}
                ORDER BY embedding <=> '{embedding_str}'::vector
                LIMIT {top_k}
            """

            result = await self.db.execute(text(query_sql))

            rows = result.fetchall()

            # Construct results - use dict instead of ORM model to avoid greenlet issues
            results = []
            for row in rows:
                # Create a simple object-like dict
                class ChunkData:
                    def __init__(self, row_data):
                        self.id = row_data[0]
                        self.document_id = row_data[1]
                        self.chunk_text = row_data[2]
                        self.chunk_index = row_data[3]
                        self.meta_data = row_data[4] or {}
                        self.created_at = row_data[5]

                chunk = ChunkData(row)
                similarity = float(row[6])  # similarity is the 7th column
                results.append((chunk, similarity))

            logger.info(f"Found {len(results)} similar chunks for query")
            return results

        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            raise

    async def delete_document_chunks(self, document_id: int):
        """
        Delete all chunks for a document.

        Args:
            document_id: ID of document
        """
        try:
            result = await self.db.execute(
                select(DocumentChunk).where(DocumentChunk.document_id == document_id)
            )
            chunks = result.scalars().all()

            for chunk in chunks:
                await self.db.delete(chunk)

            await self.db.commit()
            logger.info(f"Deleted {len(chunks)} chunks for document {document_id}")

        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error deleting document chunks: {e}")
            raise
