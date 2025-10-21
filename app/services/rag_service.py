"""RAG (Retrieval-Augmented Generation) service."""

import logging
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models import Conversation, Message
from app.schemas import RAGQueryResponse, SourceDocument
from app.services.llm_service import LLMService
from app.services.vector_service import VectorService

logger = logging.getLogger(__name__)


class RAGService:
    """Service for RAG query processing."""

    def __init__(self, db: AsyncSession):
        """Initialize RAG service."""
        self.db = db
        self.llm_service = LLMService()
        self.vector_service = VectorService(db)

    async def _get_or_create_conversation(
        self,
        conversation_id: Optional[int] = None
    ) -> Conversation:
        """
        Get existing conversation or create new one.

        Args:
            conversation_id: Optional conversation ID

        Returns:
            Conversation object
        """
        if conversation_id:
            result = await self.db.execute(
                select(Conversation)
                .where(Conversation.id == conversation_id)
                .options(selectinload(Conversation.messages))
            )
            conversation = result.scalar_one_or_none()

            if not conversation:
                raise ValueError(f"Conversation {conversation_id} not found")

            return conversation

        # Create new conversation
        conversation = Conversation(title="New Conversation")
        self.db.add(conversation)
        await self.db.flush()

        # Initialize messages list to prevent lazy loading issues
        await self.db.refresh(conversation, ["messages"])

        return conversation

    def _build_prompt(
        self,
        query: str,
        context_chunks: list,
        conversation_history: list = None
    ) -> list[dict]:
        """
        Build prompt messages for LLM.

        Args:
            query: User query
            context_chunks: Retrieved context chunks
            conversation_history: Previous messages

        Returns:
            List of message dicts for LLM
        """
        # System prompt
        system_prompt = """You are a helpful AI assistant with access to a knowledge base.
Use the provided context to answer questions accurately and concisely.
If the context doesn't contain relevant information, say so clearly.
Always cite which parts of the context you used in your answer."""

        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history if available
        if conversation_history:
            for msg in conversation_history[-5:]:  # Last 5 messages for context
                messages.append({
                    "role": msg.role,
                    "content": msg.content
                })

        # Build context from retrieved chunks
        if context_chunks:
            context_text = "\n\n---\n\n".join([
                f"[Source {idx + 1}]\n{chunk[0].chunk_text}"
                for idx, chunk in enumerate(context_chunks)
            ])

            user_message = f"""Context from knowledge base:

{context_text}

---

Question: {query}

Please answer based on the context provided above."""
        else:
            user_message = f"""Question: {query}

Note: No relevant context was found in the knowledge base. Please answer based on your general knowledge and mention that this is not from the knowledge base."""

        messages.append({"role": "user", "content": user_message})

        return messages

    async def process_query(
        self,
        query: str,
        conversation_id: Optional[int] = None,
        top_k: int = 5,
        include_sources: bool = True
    ) -> RAGQueryResponse:
        """
        Process RAG query end-to-end.

        Steps:
        1. Retrieve relevant documents
        2. Build prompt with context
        3. Generate response using LLM
        4. Store conversation
        5. Return response with sources

        Args:
            query: User query
            conversation_id: Optional conversation ID for context
            top_k: Number of documents to retrieve
            include_sources: Include source documents in response

        Returns:
            RAG query response
        """
        try:
            # Get or create conversation
            conversation = await self._get_or_create_conversation(conversation_id)

            # Step 1: Retrieve relevant documents
            logger.info(f"Retrieving top {top_k} documents for query")
            similar_chunks = await self.vector_service.similarity_search(
                query=query,
                top_k=top_k
            )

            # Step 2: Build prompt
            # Skip conversation history for now to avoid lazy loading issues
            messages = self._build_prompt(
                query=query,
                context_chunks=similar_chunks,
                conversation_history=None
            )

            # Step 3: Generate response
            logger.info("Generating response with OpenAI")
            llm_response = await self.llm_service.generate_chat_completion(messages)

            # Step 4: Store conversation
            # Add user message
            user_message = Message(
                conversation_id=conversation.id,
                role="user",
                content=query,
                meta_data={}
            )
            self.db.add(user_message)

            # Add assistant message
            assistant_message = Message(
                conversation_id=conversation.id,
                role="assistant",
                content=llm_response["content"],
                meta_data={
                    "model": llm_response["model"],
                    "usage": llm_response["usage"],
                    "sources_count": len(similar_chunks)
                }
            )
            self.db.add(assistant_message)

            await self.db.commit()
            await self.db.refresh(assistant_message)

            # Step 5: Build response
            sources = []
            if include_sources:
                sources = [
                    SourceDocument(
                        document_id=chunk.document_id,
                        chunk_id=chunk.id,
                        content=chunk.chunk_text,
                        similarity_score=score,
                        metadata=chunk.meta_data
                    )
                    for chunk, score in similar_chunks
                ]

            response = RAGQueryResponse(
                conversation_id=conversation.id,
                message_id=assistant_message.id,
                query=query,
                response=llm_response["content"],
                sources=sources,
                metadata={
                    "model": llm_response["model"],
                    "usage": llm_response["usage"],
                    "retrieved_chunks": len(similar_chunks)
                }
            )

            logger.info(f"Successfully processed query in conversation {conversation.id}")
            return response

        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error processing RAG query: {e}", exc_info=True)
            raise
