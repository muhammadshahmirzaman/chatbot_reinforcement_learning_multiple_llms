"""
Model registry for DER.

This module provides a central registry for managing model adapters,
supporting both online and offline models with configuration-driven loading.
"""

import logging
import importlib
from typing import Dict, List, Type, Optional
from pathlib import Path

import yaml

from .base import BaseModelAdapter, ExecutionMode, OfflineAdapter
from .resource_manager import ResourceManager

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Central registry for managing both online and offline model adapters.
    
    The registry handles:
    - Registration and lookup of model adapters
    - Configuration-driven instantiation
    - Resource management for offline models
    - Model cost tracking for reward shaping
    """
    
    # Mapping of adapter type names to their module paths
    ADAPTER_CLASSES = {
        "api": "adapters.online.api_adapter.APIAdapter",
        "inference_server": "adapters.online.inference_server.InferenceServerAdapter",
        "transformers": "adapters.offline.transformers_adapter.TransformersAdapter",
        "openassistant": "adapters.offline.openassistant_adapter.OpenAssistantAdapter",
        "quantized": "adapters.offline.quantized_adapter.QuantizedAdapter",
        "cached": "adapters.offline.cached_adapter.CachedAdapter",
    }
    
    def __init__(self, resource_manager: Optional[ResourceManager] = None):
        """
        Initialize the model registry.
        
        Args:
            resource_manager: Optional ResourceManager for offline models.
                            If not provided, a default one is created.
        """
        self._adapters: Dict[str, BaseModelAdapter] = {}
        self._model_costs: Dict[str, float] = {}
        self.resource_manager = resource_manager or ResourceManager()
    
    def register(
        self,
        name: str,
        adapter: BaseModelAdapter,
        cost: float = 0.01
    ) -> None:
        """
        Register a model adapter.
        
        Args:
            name: Unique name for the model.
            adapter: The model adapter instance.
            cost: Cost penalty for reward shaping.
        """
        self._adapters[name] = adapter
        self._model_costs[name] = cost
        logger.info(f"Registered model '{name}' ({adapter.execution_mode.value})")
    
    def get(self, name: str) -> Optional[BaseModelAdapter]:
        """
        Get a registered adapter by name.
        
        For offline adapters, this also ensures the model is loaded
        via the resource manager.
        
        Args:
            name: The model name to look up.
            
        Returns:
            The model adapter, or None if not found.
        """
        adapter = self._adapters.get(name)
        if adapter is None:
            logger.warning(f"Model '{name}' not found in registry")
            return None
        
        # For offline models, ensure they're loaded
        if isinstance(adapter, OfflineAdapter):
            target_device = getattr(adapter, "device", "cpu")
            # Default to CPU if CUDA is unavailable or explicitly requested
            if target_device.startswith("cuda"):
                try:
                    import torch
                    if not torch.cuda.is_available():
                        target_device = "cpu"
                except ImportError:
                    target_device = "cpu"
            try:
                self.resource_manager.ensure_loaded(adapter, device=target_device)
            except Exception as e:
                logger.error(f"Failed to load adapter '{name}' to {target_device}: {e}")
                return None
        
        return adapter
    
    def list_models(self) -> List[str]:
        """Return list of all registered model names."""
        return list(self._adapters.keys())
    
    def list_online_models(self) -> List[str]:
        """Return list of online model names."""
        return [
            name for name, adapter in self._adapters.items()
            if adapter.execution_mode == ExecutionMode.ONLINE
        ]
    
    def list_offline_models(self) -> List[str]:
        """Return list of offline model names."""
        return [
            name for name, adapter in self._adapters.items()
            if adapter.execution_mode == ExecutionMode.OFFLINE
        ]
    
    def get_model_cost(self, name: str) -> float:
        """Get the cost for a model (for reward shaping)."""
        return self._model_costs.get(name, 0.01)
    
    def get_available_models(self) -> List[str]:
        """Return list of models currently available for inference."""
        return [
            name for name, adapter in self._adapters.items()
            if adapter.is_available()
        ]
    
    def get_num_models(self) -> int:
        """Return the total number of registered models."""
        return len(self._adapters)
    
    @classmethod
    def from_config(cls, config_path: str) -> "ModelRegistry":
        """
        Create a registry from a YAML configuration file.
        
        Args:
            config_path: Path to the models.yaml configuration file.
            
        Returns:
            A configured ModelRegistry instance.
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Setup resource manager from config
        rm_config = config.get('resource_management', {})
        resource_manager = ResourceManager(
            max_gpu_memory_gb=rm_config.get('max_gpu_memory_gb', 24.0),
            max_loaded_models=rm_config.get('max_loaded_offline_models', 2)
        )
        
        registry = cls(resource_manager)
        
        # Load model costs
        model_costs = config.get('model_costs', {})
        
        # Register each enabled model
        for model_def in config.get('models', []):
            if not model_def.get('enabled', True):
                logger.debug(f"Skipping disabled model: {model_def.get('name')}")
                continue
            
            try:
                adapter = cls._create_adapter(model_def)
                cost = model_costs.get(model_def['name'], 0.01)
                registry.register(model_def['name'], adapter, cost)
            except Exception as e:
                logger.error(f"Failed to create adapter for {model_def.get('name')}: {e}")
        
        return registry
    
    @classmethod
    def _create_adapter(cls, model_def: dict) -> BaseModelAdapter:
        """
        Factory method to create an adapter from a model definition.
        
        Args:
            model_def: Dictionary with model configuration.
            
        Returns:
            Instantiated model adapter.
        """
        adapter_type = model_def['type']
        if adapter_type not in cls.ADAPTER_CLASSES:
            raise ValueError(f"Unknown adapter type: {adapter_type}")
        
        adapter_class_path = cls.ADAPTER_CLASSES[adapter_type]
        
        # Dynamic import
        module_path, class_name = adapter_class_path.rsplit('.', 1)
        try:
            module = importlib.import_module(module_path)
            adapter_class = getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Failed to import {adapter_class_path}: {e}")
        
        # Instantiate with config
        return adapter_class(
            name=model_def['name'],
            **model_def.get('config', {})
        )
    
    def get_status(self) -> Dict:
        """Get registry status summary."""
        return {
            "total_models": len(self._adapters),
            "online_models": len(self.list_online_models()),
            "offline_models": len(self.list_offline_models()),
            "available_models": len(self.get_available_models()),
            "resource_manager": self.resource_manager.get_status(),
        }
