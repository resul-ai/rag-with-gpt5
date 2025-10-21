"""Pydantic schemas for API requests and responses."""

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# Document schemas
class DocumentBase(BaseModel):
    """Base document schema."""
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Document content")
    metadata: Dict = Field(default_factory=dict, description="Additional metadata")


class DocumentCreate(DocumentBase):
    """Schema for creating a document."""
    pass


class DocumentResponse(DocumentBase):
    """Schema for document response."""
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# Conversation schemas
class MessageBase(BaseModel):
    """Base message schema."""
    content: str = Field(..., description="Message content")


class MessageCreate(MessageBase):
    """Schema for creating a message."""
    pass


class MessageResponse(BaseModel):
    """Schema for message response."""
    id: int
    role: str
    content: str
    metadata: Dict = Field(default_factory=dict)
    created_at: datetime

    class Config:
        from_attributes = True


class ConversationResponse(BaseModel):
    """Schema for conversation response."""
    id: int
    title: Optional[str] = None
    user_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    messages: List[MessageResponse] = []

    class Config:
        from_attributes = True


# RAG query schemas
class RAGQueryRequest(BaseModel):
    """Schema for RAG query request."""
    query: str = Field(..., description="User query")
    conversation_id: Optional[int] = Field(None, description="Conversation ID for context")
    top_k: Optional[int] = Field(5, description="Number of documents to retrieve")
    include_sources: bool = Field(True, description="Include source documents in response")


class SourceDocument(BaseModel):
    """Schema for source document in RAG response."""
    document_id: int
    chunk_id: int
    content: str
    similarity_score: float
    metadata: Dict = Field(default_factory=dict)


class RAGQueryResponse(BaseModel):
    """Schema for RAG query response."""
    conversation_id: int
    message_id: int
    query: str
    response: str
    sources: List[SourceDocument] = []
    metadata: Dict = Field(default_factory=dict, description="Additional info (model, tokens, etc.)")


# Health check schema
class HealthResponse(BaseModel):
    """Schema for health check response."""
    status: str = Field(..., description="Service status")
    database: str = Field(..., description="Database connection status")
    openai: str = Field(..., description="OpenAI API status")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
