"""Health check endpoints."""

import logging
from datetime import datetime

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db import get_async_db
from app.schemas import HealthResponse
from app.services.llm_service import LLMService

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/health", response_model=HealthResponse)
async def health_check(db: AsyncSession = Depends(get_async_db)):
    """
    Health check endpoint.

    Verifies:
    - API service is running
    - Database connection is active
    - OpenAI API is accessible
    """
    # Check database connection
    db_status = "healthy"
    try:
        await db.execute(text("SELECT 1"))
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        db_status = f"unhealthy: {str(e)}"

    # Check OpenAI API
    openai_status = "healthy"
    try:
        llm_service = LLMService()
        await llm_service.test_connection()
    except Exception as e:
        logger.error(f"OpenAI health check failed: {e}")
        openai_status = f"unhealthy: {str(e)}"

    # Overall status
    overall_status = "healthy" if db_status == "healthy" and openai_status == "healthy" else "degraded"

    return HealthResponse(
        status=overall_status,
        database=db_status,
        openai=openai_status,
        timestamp=datetime.utcnow()
    )
