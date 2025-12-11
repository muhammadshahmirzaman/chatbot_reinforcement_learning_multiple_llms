"""
Resource manager for offline model adapters.

This module handles GPU memory management for multiple offline models,
implementing LRU eviction to prevent out-of-memory errors.
"""

import logging
from typing import Dict, List, Optional
from .base import OfflineAdapter, MemoryInfo

logger = logging.getLogger(__name__)


class ResourceManager:
    """
    Manages GPU resources for offline models with lazy loading and LRU eviction.
    
    This class tracks loaded offline models and automatically evicts the
    least recently used models when GPU memory is constrained.
    """
    
    def __init__(
        self,
        max_gpu_memory_gb: float = 24.0,
        max_loaded_models: int = 2
    ):
        """
        Initialize the resource manager.
        
        Args:
            max_gpu_memory_gb: Maximum GPU memory to use (in GB).
            max_loaded_models: Maximum number of models to keep loaded.
        """
        self.max_gpu_memory = max_gpu_memory_gb * 1024  # Convert to MB
        self.max_loaded_models = max_loaded_models
        self._loaded_adapters: Dict[str, OfflineAdapter] = {}
        self._load_order: List[str] = []  # LRU tracking (oldest first)
    
    def ensure_loaded(
        self,
        adapter: OfflineAdapter,
        device: str = "cuda"
    ) -> None:
        """
        Ensure the adapter is loaded, evicting others if necessary.
        
        Args:
            adapter: The offline adapter to load.
            device: Target device for loading.
        """
        if adapter.name in self._loaded_adapters:
            # Move to end of LRU list (most recently used)
            self._load_order.remove(adapter.name)
            self._load_order.append(adapter.name)
            logger.debug(f"Model '{adapter.name}' already loaded, updated LRU")
            return
        
        # Evict models if we've reached the limit
        while len(self._loaded_adapters) >= self.max_loaded_models:
            self._evict_lru()
        
        # Check GPU memory availability
        estimated_memory = adapter.get_memory_usage().gpu_memory_mb
        self._ensure_memory_available(estimated_memory)
        
        # Load the model
        logger.info(f"Loading model '{adapter.name}' to {device}")
        adapter.load_to_device(device)
        self._loaded_adapters[adapter.name] = adapter
        self._load_order.append(adapter.name)
    
    def _evict_lru(self) -> None:
        """Evict the least recently used model."""
        if not self._load_order:
            return
        
        lru_name = self._load_order.pop(0)
        adapter = self._loaded_adapters.pop(lru_name)
        
        logger.info(f"Evicting model '{lru_name}' (LRU)")
        adapter.offload_to_cpu()
        
        # Clear GPU cache after eviction
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    
    def _ensure_memory_available(self, required_mb: float) -> None:
        """
        Ensure enough GPU memory is available by evicting models if needed.
        
        Args:
            required_mb: Required memory in MB.
        """
        while self._get_current_gpu_usage() + required_mb > self.max_gpu_memory:
            if not self._load_order:
                raise RuntimeError(
                    f"Cannot free enough GPU memory. "
                    f"Required: {required_mb}MB, "
                    f"Available: {self.max_gpu_memory - self._get_current_gpu_usage()}MB"
                )
            self._evict_lru()
    
    def _get_current_gpu_usage(self) -> float:
        """Get current GPU memory usage in MB."""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / (1024 * 1024)
        except ImportError:
            pass
        return 0.0
    
    def get_loaded_models(self) -> List[str]:
        """Return list of currently loaded model names."""
        return list(self._loaded_adapters.keys())
    
    def unload_all(self) -> None:
        """Unload all loaded models."""
        while self._load_order:
            self._evict_lru()
        logger.info("All models unloaded")
    
    def get_status(self) -> Dict:
        """Get current resource manager status."""
        return {
            "loaded_models": self.get_loaded_models(),
            "num_loaded": len(self._loaded_adapters),
            "max_loaded": self.max_loaded_models,
            "current_gpu_usage_mb": self._get_current_gpu_usage(),
            "max_gpu_memory_mb": self.max_gpu_memory,
        }
