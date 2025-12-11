"""
API adapter for OpenAI-compatible endpoints.

This adapter connects to OpenAI-compatible API servers (vLLM, FastChat, etc.)
for inference without requiring local GPU resources.
"""

import time
import logging
from typing import Optional

from ..base import OnlineAdapter, GenerationConfig, ModelResponse, ExecutionMode

logger = logging.getLogger(__name__)


class APIAdapter(OnlineAdapter):
    """
    Adapter for OpenAI-compatible API endpoints.
    
    This adapter works with any OpenAI-compatible API server including:
    - vLLM served models
    - FastChat API servers
    - LocalAI
    - OpenAI API (with valid key)
    """
    
    def __init__(
        self,
        name: str,
        api_base: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        model_name: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the API adapter.
        
        Args:
            name: Unique name for this adapter.
            api_base: Base URL for the API endpoint.
            api_key: API key (use "EMPTY" for local servers).
            model_name: Model name to use in API calls (defaults to name).
        """
        self._name = name
        self.api_base = api_base.rstrip('/')
        self.api_key = api_key
        self.model_name = model_name or name
        self._client = None
    
    @property
    def name(self) -> str:
        return self._name
    
    def health_check(self) -> bool:
        """Check if the API endpoint is available."""
        try:
            import requests
            resp = requests.get(f"{self.api_base}/models", timeout=5)
            return resp.status_code == 200
        except Exception as e:
            logger.debug(f"Health check failed for {self._name}: {e}")
            return False
    
    def generate(
        self,
        prompt: str,
        config: GenerationConfig = None
    ) -> ModelResponse:
        """
        Generate a response using the API endpoint.
        
        Args:
            prompt: The input prompt.
            config: Generation configuration.
            
        Returns:
            ModelResponse with generated text and metadata.
        """
        config = config or GenerationConfig()
        start_time = time.time()
        
        try:
            import openai
            openai.api_key = self.api_key
            openai.api_base = self.api_base
            
            completion = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                timeout=config.timeout,
            )
            
            latency_ms = (time.time() - start_time) * 1000
            text = completion.choices[0].message.content
            tokens_used = completion.get('usage', {}).get('total_tokens')
            
            return ModelResponse(
                text=text,
                model_name=self._name,
                latency_ms=latency_ms,
                tokens_used=tokens_used,
                execution_mode=ExecutionMode.ONLINE
            )
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(f"API generation failed for {self._name}: {e}")
            
            # Return error message as response (for graceful degradation)
            return ModelResponse(
                text=f"[ERROR: {str(e)}]",
                model_name=self._name,
                latency_ms=latency_ms,
                execution_mode=ExecutionMode.ONLINE
            )
    
    def __repr__(self) -> str:
        return f"APIAdapter(name={self._name}, endpoint={self.api_base})"
