"""LLM service for OpenAI integration."""

import logging
from typing import List, Optional

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings

logger = logging.getLogger(__name__)


class LLMService:
    """Service for interacting with OpenAI API."""

    def __init__(self):
        """Initialize OpenAI client."""
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model
        self.temperature = settings.llm_temperature
        self.max_tokens = settings.llm_max_tokens

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    async def generate_chat_completion(
        self,
        messages: List[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> dict:
        """
        Generate chat completion using OpenAI API.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Override default temperature
            max_tokens: Override default max tokens

        Returns:
            Response dict with 'content' and metadata
        """
        try:
            logger.info(f"Using OpenAI model: {self.model}")
            # gpt-5 models have different parameter requirements
            completion_params = {
                "model": self.model,
                "messages": messages,
            }

            # gpt-5 models: use max_completion_tokens, don't set temperature (only default 1.0 supported)
            if self.model.startswith("gpt-5"):
                completion_params["max_completion_tokens"] = max_tokens or self.max_tokens
                # gpt-5 models don't support custom temperature, skip it
            else:
                completion_params["max_tokens"] = max_tokens or self.max_tokens
                completion_params["temperature"] = temperature or self.temperature

            response = await self.client.chat.completions.create(**completion_params)

            return {
                "content": response.choices[0].message.content,
                "model": response.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                "finish_reason": response.choices[0].finish_reason
            }

        except Exception as e:
            logger.error(f"Error generating chat completion: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for texts using OpenAI API.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        try:
            response = await self.client.embeddings.create(
                model=settings.openai_embedding_model,
                input=texts
            )

            embeddings = [item.embedding for item in response.data]
            return embeddings

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    async def test_connection(self) -> bool:
        """
        Test OpenAI API connection.

        Returns:
            True if connection is successful
        """
        try:
            # Simple test with minimal tokens
            messages = [{"role": "user", "content": "test"}]
            await self.generate_chat_completion(messages, max_tokens=5)
            return True

        except Exception as e:
            logger.error(f"OpenAI connection test failed: {e}")
            raise
