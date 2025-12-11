"""
Unit tests for DER model adapters.

Run with: pytest tests/test_adapters.py -v
"""

import pytest
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from adapters.base import (
    ExecutionMode,
    GenerationConfig,
    ModelResponse,
    MemoryInfo,
    BaseModelAdapter,
    OnlineAdapter,
    OfflineAdapter,
)
from adapters.resource_manager import ResourceManager


class TestGenerationConfig:
    """Tests for GenerationConfig dataclass."""
    
    def test_default_values(self):
        config = GenerationConfig()
        assert config.max_tokens == 512
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.timeout == 1800
        assert config.do_sample == True
    
    def test_custom_values(self):
        config = GenerationConfig(max_tokens=256, temperature=0.5)
        assert config.max_tokens == 256
        assert config.temperature == 0.5
    
    def test_to_dict(self):
        config = GenerationConfig()
        d = config.to_dict()
        assert d["max_tokens"] == 512
        assert d["temperature"] == 0.7


class TestModelResponse:
    """Tests for ModelResponse dataclass."""
    
    def test_creation(self):
        response = ModelResponse(
            text="Hello world",
            model_name="test-model",
            latency_ms=100.5
        )
        assert response.text == "Hello world"
        assert response.model_name == "test-model"
        assert response.latency_ms == 100.5
        assert response.execution_mode == ExecutionMode.OFFLINE
    
    def test_repr(self):
        response = ModelResponse(
            text="test",
            model_name="test-model",
            latency_ms=50.0
        )
        repr_str = repr(response)
        assert "test-model" in repr_str
        assert "50.0ms" in repr_str


class TestResourceManager:
    """Tests for ResourceManager."""
    
    def test_init(self):
        rm = ResourceManager(max_gpu_memory_gb=16.0, max_loaded_models=3)
        assert rm.max_gpu_memory == 16.0 * 1024
        assert rm.max_loaded_models == 3
    
    def test_get_status(self):
        rm = ResourceManager()
        status = rm.get_status()
        assert "loaded_models" in status
        assert "num_loaded" in status
        assert "max_loaded" in status
    
    def test_ensure_loaded_new_adapter(self):
        """Test loading a new adapter."""
        rm = ResourceManager(max_loaded_models=2)
        
        # Create mock adapter
        mock_adapter = Mock(spec=OfflineAdapter)
        mock_adapter.name = "test-adapter"
        mock_adapter.get_memory_usage.return_value = MemoryInfo(gpu_memory_mb=1000)
        
        rm.ensure_loaded(mock_adapter, device="cuda:0")
        
        mock_adapter.load_to_device.assert_called_once_with("cuda:0")
        assert "test-adapter" in rm.get_loaded_models()
    
    def test_lru_eviction(self):
        """Test that LRU eviction works."""
        rm = ResourceManager(max_loaded_models=2)
        
        # Create mock adapters
        adapters = []
        for i in range(3):
            adapter = Mock(spec=OfflineAdapter)
            adapter.name = f"adapter-{i}"
            adapter.get_memory_usage.return_value = MemoryInfo(gpu_memory_mb=100)
            adapters.append(adapter)
        
        # Load first two
        rm.ensure_loaded(adapters[0])
        rm.ensure_loaded(adapters[1])
        
        assert len(rm.get_loaded_models()) == 2
        
        # Load third - should evict first
        with patch('torch.cuda.empty_cache'):
            rm.ensure_loaded(adapters[2])
        
        assert len(rm.get_loaded_models()) == 2
        assert adapters[0].offload_to_cpu.called


class TestExecutionMode:
    """Tests for ExecutionMode enum."""
    
    def test_values(self):
        assert ExecutionMode.ONLINE.value == "online"
        assert ExecutionMode.OFFLINE.value == "offline"


# Mock adapter implementations for testing
class MockOnlineAdapter(OnlineAdapter):
    """Mock online adapter for testing."""
    
    def __init__(self, name="mock-online", healthy=True):
        self._name = name
        self._healthy = healthy
    
    @property
    def name(self):
        return self._name
    
    def health_check(self):
        return self._healthy
    
    def generate(self, prompt, config=None):
        return ModelResponse(
            text=f"Response to: {prompt}",
            model_name=self._name,
            latency_ms=10.0,
            execution_mode=ExecutionMode.ONLINE
        )


class MockOfflineAdapter(OfflineAdapter):
    """Mock offline adapter for testing."""
    
    def __init__(self, name="mock-offline"):
        self._name = name
        self._loaded = False
        self._device = "cpu"
    
    @property
    def name(self):
        return self._name
    
    def load(self):
        self._loaded = True
    
    def load_to_device(self, device):
        self._loaded = True
        self._device = device
    
    def offload_to_cpu(self):
        self._device = "cpu"
    
    def unload(self):
        self._loaded = False
    
    def is_loaded(self):
        return self._loaded
    
    def get_memory_usage(self):
        return MemoryInfo(
            gpu_memory_mb=1000 if "cuda" in self._device else 0,
            cpu_memory_mb=1000 if self._device == "cpu" else 0,
            is_loaded=self._loaded
        )
    
    def generate(self, prompt, config=None):
        if not self._loaded:
            raise RuntimeError("Model not loaded")
        return ModelResponse(
            text=f"Offline response to: {prompt}",
            model_name=self._name,
            latency_ms=50.0,
            execution_mode=ExecutionMode.OFFLINE
        )


class TestMockAdapters:
    """Tests using mock adapters."""
    
    def test_online_adapter(self):
        adapter = MockOnlineAdapter()
        assert adapter.execution_mode == ExecutionMode.ONLINE
        assert adapter.is_loaded() == True
        assert adapter.health_check() == True
        
        response = adapter.generate("Hello")
        assert "Hello" in response.text
        assert response.execution_mode == ExecutionMode.ONLINE
    
    def test_offline_adapter(self):
        adapter = MockOfflineAdapter()
        assert adapter.execution_mode == ExecutionMode.OFFLINE
        assert adapter.is_loaded() == False
        
        adapter.load()
        assert adapter.is_loaded() == True
        
        response = adapter.generate("Hello")
        assert "Hello" in response.text
        assert response.execution_mode == ExecutionMode.OFFLINE
    
    def test_offline_adapter_not_loaded_error(self):
        adapter = MockOfflineAdapter()
        with pytest.raises(RuntimeError):
            adapter.generate("Hello")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
