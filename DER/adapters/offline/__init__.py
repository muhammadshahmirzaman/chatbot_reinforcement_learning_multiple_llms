# Offline model adapters
# These adapters perform local inference without requiring external servers.

from .transformers_adapter import TransformersAdapter
from .openassistant_adapter import OpenAssistantAdapter
from .cached_adapter import CachedAdapter

__all__ = [
    "TransformersAdapter",
    "OpenAssistantAdapter", 
    "CachedAdapter",
]
