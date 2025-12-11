"""
Model configuration dataclasses for DER.

This module provides configuration loading utilities for the model registry.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from pathlib import Path
import yaml


@dataclass
class GenerationDefaults:
    """Default generation settings."""
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    timeout: int = 1800
    do_sample: bool = True


@dataclass
class ResourceManagementConfig:
    """Resource management settings."""
    max_gpu_memory_gb: float = 24.0
    max_loaded_offline_models: int = 2
    prefer_offline: bool = True


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    name: str
    type: str
    execution_mode: str
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DERConfig:
    """Complete DER configuration."""
    version: str
    resource_management: ResourceManagementConfig
    defaults: GenerationDefaults
    models: List[ModelConfig]
    model_costs: Dict[str, float]
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "DERConfig":
        """Load configuration from YAML file."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Parse resource management
        rm_data = data.get('resource_management', {})
        resource_management = ResourceManagementConfig(
            max_gpu_memory_gb=rm_data.get('max_gpu_memory_gb', 24.0),
            max_loaded_offline_models=rm_data.get('max_loaded_offline_models', 2),
            prefer_offline=rm_data.get('prefer_offline', True)
        )
        
        # Parse defaults
        defaults_data = data.get('defaults', {}).get('generation', {})
        defaults = GenerationDefaults(
            max_tokens=defaults_data.get('max_tokens', 512),
            temperature=defaults_data.get('temperature', 0.7),
            top_p=defaults_data.get('top_p', 0.9),
            timeout=defaults_data.get('timeout', 1800),
            do_sample=defaults_data.get('do_sample', True)
        )
        
        # Parse models
        models = []
        for model_data in data.get('models', []):
            model = ModelConfig(
                name=model_data['name'],
                type=model_data['type'],
                execution_mode=model_data.get('execution_mode', 'online'),
                enabled=model_data.get('enabled', True),
                config=model_data.get('config', {})
            )
            models.append(model)
        
        return cls(
            version=data.get('version', '1.0'),
            resource_management=resource_management,
            defaults=defaults,
            models=models,
            model_costs=data.get('model_costs', {})
        )
    
    def get_enabled_models(self) -> List[ModelConfig]:
        """Return only enabled models."""
        return [m for m in self.models if m.enabled]
    
    def get_model_cost(self, name: str) -> float:
        """Get cost for a model."""
        return self.model_costs.get(name, 0.01)
