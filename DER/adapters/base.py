"""
Base adapter classes for DER model integration.

This module defines the abstract base classes and data structures for
integrating both online (API-based) and offline (local) LLM models with DER.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any
import time


class ExecutionMode(Enum):
    """Execution mode for model adapters."""
    ONLINE = "online"    # Requires external server/API
    OFFLINE = "offline"  # Local inference, no network needed


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    timeout: int = 1800
    do_sample: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "timeout": self.timeout,
            "do_sample": self.do_sample,
        }


@dataclass
class ModelResponse:
    """Standardized response from model generation."""
    text: str
    model_name: str
    latency_ms: float
    tokens_used: Optional[int] = None
    execution_mode: ExecutionMode = ExecutionMode.OFFLINE
    
    def __repr__(self) -> str:
        return (f"ModelResponse(model={self.model_name}, "
                f"latency={self.latency_ms:.1f}ms, "
                f"mode={self.execution_mode.value})")


@dataclass
class MemoryInfo:
    """Memory usage information for offline models."""
    gpu_memory_mb: float = 0.0
    cpu_memory_mb: float = 0.0
    is_loaded: bool = False


class BaseModelAdapter(ABC):
    """
    Abstract base class for all model adapters.
    
    This class defines the interface that all model adapters must implement,
    whether they are online (API-based) or offline (local inference).
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique name of this model adapter."""
        ...
    
    @property
    @abstractmethod
    def execution_mode(self) -> ExecutionMode:
        """Return the execution mode (ONLINE or OFFLINE)."""
        ...
    
    @abstractmethod
    def generate(self, prompt: str, config: GenerationConfig = None) -> ModelResponse:
        """
        Generate a response for the given prompt.
        
        Args:
            prompt: The input text prompt.
            config: Optional generation configuration.
            
        Returns:
            ModelResponse containing the generated text and metadata.
        """
        ...
    
    @abstractmethod
    def load(self) -> None:
        """Load the model (for offline) or initialize connection (for online)."""
        ...
    
    @abstractmethod
    def unload(self) -> None:
        """Unload the model to free resources."""
        ...
    
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if the model is currently loaded."""
        ...
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the model is available for inference.
        
        For online models: server is reachable.
        For offline models: model is loaded.
        """
        ...
    
    def get_memory_usage(self) -> MemoryInfo:
        """Return memory usage information (mainly for offline models)."""
        return MemoryInfo()


class OnlineAdapter(BaseModelAdapter):
    """
    Base class for online/API-based adapters.
    
    Online adapters connect to external services (vLLM, FastChat, OpenAI, etc.)
    and don't require local GPU resources for inference.
    """
    
    @property
    def execution_mode(self) -> ExecutionMode:
        return ExecutionMode.ONLINE
    
    def is_loaded(self) -> bool:
        """Online models are always 'loaded' - they don't need local loading."""
        return True
    
    def load(self) -> None:
        """No-op for online models."""
        pass
    
    def unload(self) -> None:
        """No-op for online models."""
        pass
    
    @abstractmethod
    def health_check(self) -> bool:
        """
        Check if the remote endpoint is available.
        
        Returns:
            True if the endpoint is reachable and responding.
        """
        ...
    
    def is_available(self) -> bool:
        """Check availability by performing a health check."""
        return self.health_check()


class OfflineAdapter(BaseModelAdapter):
    """
    Base class for offline/local model adapters.
    
    Offline adapters load models into local GPU/CPU memory for inference.
    They require careful resource management to avoid OOM errors.
    """
    
    @property
    def execution_mode(self) -> ExecutionMode:
        return ExecutionMode.OFFLINE
    
    @abstractmethod
    def load_to_device(self, device: str) -> None:
        """
        Load model to the specified device.
        
        Args:
            device: Target device (e.g., "cuda:0", "cpu").
        """
        ...
    
    @abstractmethod
    def offload_to_cpu(self) -> None:
        """Offload model from GPU to CPU to free VRAM."""
        ...
    
    def is_available(self) -> bool:
        """Offline models are available when loaded."""
        return self.is_loaded()
