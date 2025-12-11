"""
Cached response adapter for testing and reproducibility.

This adapter uses precomputed/cached responses for fast evaluation
without requiring actual model inference.
"""

import json
import hashlib
import logging
from pathlib import Path
from typing import Optional, Dict

from ..base import OfflineAdapter, GenerationConfig, ModelResponse, MemoryInfo, ExecutionMode

logger = logging.getLogger(__name__)


class CachedAdapter(OfflineAdapter):
    """
    Adapter using precomputed/cached responses.
    
    This is useful for:
    - Fast evaluation without GPU
    - Reproducible experiments
    - Testing the DER pipeline
    - Offline evaluation with collected responses
    """
    
    def __init__(
        self,
        name: str,
        cache_file: str,
        fallback_adapter: Optional[OfflineAdapter] = None,
        save_new_responses: bool = True,
        **kwargs
    ):
        """
        Initialize the cached response adapter.
        
        Args:
            name: Unique name for this adapter.
            cache_file: Path to the JSON cache file.
            fallback_adapter: Optional adapter to use on cache miss.
            save_new_responses: Whether to save new responses to cache.
        """
        self._name = name
        self.cache_file = Path(cache_file)
        self.fallback = fallback_adapter
        self.save_new_responses = save_new_responses
        
        self._cache: Dict[str, str] = {}
        self._loaded = False
        self._cache_hits = 0
        self._cache_misses = 0
    
    @property
    def name(self) -> str:
        return self._name
    
    def load(self) -> None:
        """Load the cache from disk."""
        self.load_to_device("cpu")
    
    def load_to_device(self, device: str) -> None:
        """
        Load the cache file.
        
        Args:
            device: Ignored for cached adapter (always uses CPU/memory).
        """
        if self.cache_file.exists():
            logger.info(f"Loading cache from {self.cache_file}")
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self._cache = json.load(f)
                logger.info(f"Loaded {len(self._cache)} cached responses")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to load cache: {e}")
                self._cache = {}
        else:
            logger.info(f"No cache file found at {self.cache_file}, starting fresh")
            self._cache = {}
        
        self._loaded = True
    
    def offload_to_cpu(self) -> None:
        """No-op for cached adapter (already in memory)."""
        pass
    
    def unload(self) -> None:
        """Save and clear the cache."""
        self._save_cache()
        self._cache = {}
        self._loaded = False
    
    def is_loaded(self) -> bool:
        """Check if the cache is loaded."""
        return self._loaded
    
    def get_memory_usage(self) -> MemoryInfo:
        """Return memory usage (negligible for cache)."""
        return MemoryInfo(
            gpu_memory_mb=0,
            cpu_memory_mb=len(self._cache) * 0.01,  # Rough estimate
            is_loaded=self._loaded
        )
    
    def _hash_prompt(self, prompt: str) -> str:
        """Create a hash key for the prompt."""
        return hashlib.md5(prompt.encode()).hexdigest()
    
    def generate(
        self,
        prompt: str,
        config: GenerationConfig = None
    ) -> ModelResponse:
        """
        Generate a response from cache or fallback.
        
        Args:
            prompt: The input prompt.
            config: Generation configuration (used for fallback).
            
        Returns:
            ModelResponse with cached or generated text.
        """
        if not self._loaded:
            self.load()
        
        key = self._hash_prompt(prompt)
        
        # Check cache
        if key in self._cache:
            self._cache_hits += 1
            return ModelResponse(
                text=self._cache[key],
                model_name=self._name,
                latency_ms=0.1,
                execution_mode=ExecutionMode.OFFLINE
            )
        
        self._cache_misses += 1
        
        # Try fallback adapter
        if self.fallback is not None:
            logger.debug(f"Cache miss, using fallback adapter")
            response = self.fallback.generate(prompt, config)
            
            if self.save_new_responses:
                self._cache[key] = response.text
                self._save_cache()
            
            return response
        
        # No cache hit and no fallback
        logger.warning(f"Cache miss for {self._name} with no fallback")
        return ModelResponse(
            text="[CACHE_MISS: No cached response available]",
            model_name=self._name,
            latency_ms=0.0,
            execution_mode=ExecutionMode.OFFLINE
        )
    
    def _save_cache(self) -> None:
        """Save the cache to disk."""
        if not self.save_new_responses:
            return
        
        # Ensure directory exists
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self._cache, f, indent=2)
            logger.debug(f"Saved cache with {len(self._cache)} entries")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        total = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total * 100) if total > 0 else 0
        return {
            "cache_size": len(self._cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate_percent": hit_rate,
        }
    
    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "unloaded"
        return f"CachedAdapter(name={self._name}, entries={len(self._cache)}, status={status})"
